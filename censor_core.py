"""
censor_core.py — Shared logic for the audiobook censoring pipeline.

Architecture:
  1. Whisper transcribes the source audio in chunks (for memory efficiency),
     collecting all mute intervals without touching the audio data.
  2. A single ffmpeg pass applies ALL mute intervals via the `volume` filter.
     No intermediate files, no re-encodes, no pydub.

Works with .m4b, .mp3, .mp4, .aac, .ogg, .flac — anything ffmpeg can read.
"""

import re
import subprocess
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Generator, Iterable

from faster_whisper import WhisperModel

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Regex — catches the f-word in its common written and spoken forms.
# Using fullmatch (after stripping punctuation) to avoid false positives
# on words like "efflorescence", "focus", "fucks" misheard as "flux", etc.
# ---------------------------------------------------------------------------
_FWORD_RE = re.compile(
    r"^(mother)?f[u#@u0]?[ck]k?(ing?|ed?|ers?|s|'s|in'|ery)?$",
    re.IGNORECASE,
)

# Spoken word splits Whisper sometimes produces, e.g. "f" + "uck"
_FWORD_PREFIX = re.compile(r"^(mother)?f[u*#@uo0]?$", re.IGNORECASE)
_FWORD_SUFFIX = re.compile(r"^[ck](?:k?(?:ing?|ed?|ers?|'s|a|in\'|az?|ery))?$", re.IGNORECASE)


@dataclass
class MuteInterval:
    start: float  # seconds
    end: float    # seconds
    word: str     # what was censored (for logging)


@dataclass
class CensorResult:
    mutes: list[MuteInterval] = field(default_factory=list)
    total_censored: int = 0
    duration: float = 0.0


# ---------------------------------------------------------------------------
# Step 1 — Transcribe and collect mute intervals
# ---------------------------------------------------------------------------

def _strip(w: str) -> str:
    """Remove punctuation/whitespace, keep apostrophes inside contractions."""
    return re.sub(r"[^\w']", "", w).strip("'")


def _is_fword(word: str) -> bool:
    return bool(_FWORD_RE.match(_strip(word)))


def _is_split_fword(w1: str, w2: str) -> bool:
    """Catch 'f' + 'uck', 'moth' + 'erfuck', etc."""
    return bool(_FWORD_PREFIX.fullmatch(_strip(w1)) and _FWORD_SUFFIX.fullmatch(_strip(w2)))


# How far into an adjacent gap we're willing to extend the mute window.
# Only applies when there's actually a gap — if the previous word ends right
# before ours starts, we won't eat into it.
MUTE_MAX_PRE  = 0.08  # max seconds to steal before the word
MUTE_MAX_POST = 0.1  # max seconds to steal after the word

def _padded(start: float, end: float, prev_end: float | None, next_start: float | None) -> tuple[float, float]:
    """
    Extend the mute window into the silence gap on each side, but never
    more than MUTE_MAX_PRE/POST and never past the midpoint of the gap.
    This catches phoneme bleed without clipping adjacent words.
    """
    if prev_end is not None:
        gap = start - prev_end           # silence between previous word and this one
        steal = min(MUTE_MAX_PRE, gap / 2)  # take at most half the gap, capped at max
        start = start - steal
    else:
        start = max(0.0, start - MUTE_MAX_PRE)

    if next_start is not None:
        gap = next_start - end
        steal = min(MUTE_MAX_POST, gap / 2)
        end = end + steal
    else:
        end = end + MUTE_MAX_POST

    return max(0.0, start), end


def _get_duration(audio_path: Path) -> float:
    """Use ffprobe to get audio duration in seconds without loading the file."""
    import json
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", str(audio_path)],
        capture_output=True, text=True,
    )
    if result.returncode == 0:
        fmt = json.loads(result.stdout).get("format", {})
        return float(fmt.get("duration", 0))
    return 0.0


CHUNK_OVERLAP_SECONDS = 5  # overlap between chunks so boundary words aren't missed

def _split_to_chunks(audio_path: Path, chunk_dir: Path, chunk_seconds: int = 1800) -> list[tuple[Path, float]]:
    """
    Split audio into WAV chunks for Whisper with overlap.
    Each chunk overlaps the next by CHUNK_OVERLAP_SECONDS so words at
    boundaries are never missed. Duplicate detections are deduplicated
    by timestamp when results are collected.
    WAV at 16kHz mono matches Whisper's native format exactly.
    """
    chunk_dir.mkdir(parents=True, exist_ok=True)
    total = _get_duration(audio_path)
    offsets = []
    idx = 0
    start = 0.0
    while start < total:
        out_path = chunk_dir / f"chunk_{idx:04d}.wav"
        duration = chunk_seconds + CHUNK_OVERLAP_SECONDS
        subprocess.run([
            "ffmpeg", "-y",
            "-ss", str(start),
            "-i", str(audio_path),
            "-t", str(duration),
            "-ar", "16000",
            "-ac", "1",
            "-c:a", "pcm_s16le",
            str(out_path),
        ], check=True, capture_output=True)
        offsets.append((out_path, start))
        start += chunk_seconds  # advance by chunk_seconds, not chunk_seconds+overlap
        idx += 1

    return offsets


def _collect_mutes_from_chunk(
    model: "WhisperModel",
    chunk_path: Path,
    time_offset: float,
    beam_size: int,
) -> list[MuteInterval]:
    """Transcribe one chunk and return mute intervals with corrected timestamps."""
    segments_gen, _ = model.transcribe(
        str(chunk_path),
        beam_size=beam_size,
        word_timestamps=True,
        vad_filter=False,  # off — VAD was skipping words near silence boundaries
    )

    mutes = []
    for segment in segments_gen:
        words = segment.words
        i = 0
        while i < len(words):
            w = words[i]
            prev_end = words[i - 1].end if i > 0 else None
            next_start = words[i + 1].start if i + 1 < len(words) else None

            if _is_fword(w.word):
                s, e = _padded(w.start, w.end, prev_end, next_start)
                mutes.append(MuteInterval(s + time_offset, e + time_offset, _strip(w.word)))
                i += 1

            elif i + 1 < len(words) and _is_split_fword(w.word, words[i + 1].word):
                next_next_start = words[i + 2].start if i + 2 < len(words) else None
                s, e = _padded(w.start, words[i + 1].end, prev_end, next_next_start)
                combined = _strip(w.word) + _strip(words[i + 1].word)
                mutes.append(MuteInterval(s + time_offset, e + time_offset, combined))
                i += 2

            else:
                i += 1

    return mutes


def transcribe_and_collect(
    audio_path: Path,
    model_size: str = "small",
    beam_size: int = 5,
    chunk_minutes: int = 30,
    progress_cb: Callable[[str], None] | None = None,
) -> CensorResult:
    """
    Split audio into chunks, transcribe each one, collect all mute intervals.
    Chunking keeps RAM usage low regardless of book length (~200 MB per chunk).
    """
    import tempfile, shutil

    def _log(msg: str) -> None:
        if progress_cb:
            progress_cb(msg)

    total_duration = _get_duration(audio_path)
    _log(f"Audio duration: {total_duration / 3600:.1f} hours")

    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"
    _log(f"Loading Whisper model '{model_size}' ({device.upper()}, {compute_type})…")
    model = WhisperModel(model_size, compute_type=compute_type, device=device)

    tmp_dir = Path(tempfile.mkdtemp())
    try:
        _log(f"Splitting into {chunk_minutes}-minute chunks for processing…")
        chunks = _split_to_chunks(audio_path, tmp_dir, chunk_seconds=chunk_minutes * 60)
        _log(f"  Created {len(chunks)} chunk(s).")

        all_mutes: list[MuteInterval] = []

        for idx, (chunk_path, offset) in enumerate(chunks):
            pct = offset / total_duration * 100 if total_duration else 0
            _log(f"  [{idx + 1}/{len(chunks)}] Transcribing chunk at {offset / 3600:.1f}h  ({pct:.0f}%)…")
            chunk_mutes = _collect_mutes_from_chunk(model, chunk_path, offset, beam_size)
            for m in chunk_mutes:
                _log(f"    🚫 '{m.word}' at {m.start:.2f}s")
            all_mutes.extend(chunk_mutes)

        # Deduplicate: remove any mute whose start time is within 1s of a previous one
        # (caused by the chunk overlap producing the same word twice)
        all_mutes.sort(key=lambda m: m.start)
        deduped: list[MuteInterval] = []
        for m in all_mutes:
            if not deduped or m.start - deduped[-1].start > 1.0:
                deduped.append(m)
        if len(deduped) < len(all_mutes):
            _log(f"  Removed {len(all_mutes) - len(deduped)} duplicate(s) from chunk overlap.")
        all_mutes = deduped

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    result = CensorResult(mutes=all_mutes, total_censored=len(all_mutes), duration=total_duration)
    _log(f"Transcription complete. Found {len(all_mutes)} word(s) to censor.")
    return result


# ---------------------------------------------------------------------------
# Step 2 — Apply all mutes in ONE ffmpeg pass
# ---------------------------------------------------------------------------

def _build_volume_filter(mutes: Iterable[MuteInterval]) -> str:
    """
    Build an ffmpeg `volume` filter expression that silences each interval.
    Example:  volume=enable='between(t,1.2,1.8)+between(t,45.0,45.6)':volume=0
    """
    if not mutes:
        return ""
    parts = [f"between(t,{m.start:.4f},{m.end:.4f})" for m in mutes]
    expr = "+".join(parts)
    return f"volume=enable='{expr}':volume=0"


def apply_mutes_ffmpeg(
    input_path: Path,
    output_path: Path,
    mutes: list[MuteInterval],
    audio_bitrate: str = "192k",
    title_suffix: str = " (Censored)",
    progress_cb: Callable[[str], None] | None = None,
) -> None:
    """
    Apply all mute intervals to `input_path` in a single ffmpeg pass.
    Output is AAC in an M4B/M4A container (or MP3 if output_path ends .mp3).
    Metadata, chapter markers, and cover art are carried through automatically.
    `title_suffix` is appended to the title tag so audiobook apps treat it as
    a distinct entry (e.g. "My Book" → "My Book (Censored)").
    """
    def _log(msg: str) -> None:
        if progress_cb:
            progress_cb(msg)

    if not mutes:
        _log("No words to censor — copying file as-is.")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(input_path), "-c", "copy", str(output_path)],
            check=True, capture_output=True,
        )
        return

    vf = _build_volume_filter(mutes)
    suffix = output_path.suffix.lower()

    # Choose output codec based on container
    if suffix in (".m4b", ".m4a", ".mp4"):
        audio_codec = ["aac", "-b:a", audio_bitrate]
    elif suffix == ".mp3":
        audio_codec = ["libmp3lame", "-q:a", "4"]
    else:
        audio_codec = ["aac", "-b:a", audio_bitrate]

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ffmpeg can't read the original title and append to it in one step,
    # so we read it first then pass the modified value explicitly.
    probe = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", str(input_path)],
        capture_output=True, text=True,
    )
    original_title = ""
    if probe.returncode == 0:
        import json
        fmt = json.loads(probe.stdout).get("format", {})
        original_title = fmt.get("tags", {}).get("title", "") or fmt.get("tags", {}).get("TITLE", "")

    new_title = (original_title + title_suffix) if original_title else title_suffix.strip()
    _log(f"  Title tag: '{original_title}' → '{new_title}'")

    cmd = [
        "ffmpeg", "-y",
        "-i", str(input_path),
        "-af", vf,
        "-c:a", *audio_codec,
        # Preserve all non-audio streams (chapters, cover art, metadata)
        "-c:v", "copy",
        "-map_metadata", "0",
        "-map_chapters", "0",
        # Override just the title — everything else (author, album, year…) is kept
        "-metadata", f"title={new_title}",
        str(output_path),
    ]

    _log(f"Applying {len(mutes)} mute(s) via ffmpeg…")
    _log(f"  Input:  {input_path}")
    _log(f"  Output: {output_path}")

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:\n{result.stderr}")

    _log(f"✅ Done! Saved to {output_path}")


# ---------------------------------------------------------------------------
# Convenience wrapper — does everything end to end
# ---------------------------------------------------------------------------

def censor_audiobook(
    input_path: Path,
    output_path: Path,
    model_size: str = "small",
    beam_size: int = 5,
    chunk_minutes: int = 30,
    audio_bitrate: str = "192k",
    title_suffix: str = " (Censored)",
    progress_cb: Callable[[str], None] | None = None,
) -> CensorResult:
    """
    Full pipeline: transcribe → collect intervals → apply mutes → save.
    Single ffmpeg pass, no intermediate files, no pydub.
    """
    result = transcribe_and_collect(
        input_path,
        model_size=model_size,
        beam_size=beam_size,
        chunk_minutes=chunk_minutes,
        progress_cb=progress_cb,
    )
    apply_mutes_ffmpeg(
        input_path,
        output_path,
        result.mutes,
        audio_bitrate=audio_bitrate,
        title_suffix=title_suffix,
        progress_cb=progress_cb,
    )
    return result