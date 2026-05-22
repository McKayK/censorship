"""
censor.py — CLI for the audiobook censoring tool.

Usage:
    python censor.py "My Audiobook.m4b"
    python censor.py "My Audiobook.m4b" --output "censored.m4b"
    python censor.py "My Audiobook.mp3" --model medium --bitrate 256k
"""

import argparse
import logging
import sys
from pathlib import Path

from censor_core import censor_audiobook

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Censor f-words from an audiobook using Whisper + ffmpeg.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python censor.py "book.m4b"
  python censor.py "book.m4b" --output "book_clean.m4b"
  python censor.py "book.mp3" --model medium --bitrate 256k

Supported formats: .m4b  .mp3  .mp4  .aac  .ogg  .flac  (anything ffmpeg reads)

Whisper model sizes (bigger = more accurate, slower):
  tiny    ~39 MB   fastest, works for clean narration
  small   ~244 MB  good balance  ← default
  medium  ~769 MB  best accuracy for tricky accents
  large   ~1550 MB most accurate, very slow on CPU
        """,
    )
    parser.add_argument("input", help="Path to the audiobook file")
    parser.add_argument(
        "--output", "-o",
        help="Output file path (default: <input>_censored<ext>)",
    )
    parser.add_argument(
        "--model", "-m",
        default="small",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model size (default: small)",
    )
    parser.add_argument(
        "--beam-size", "-b",
        type=int,
        default=5,
        help="Whisper beam size — higher = more accurate but slower (default: 5)",
    )
    parser.add_argument(
        "--bitrate",
        default="192k",
        help="Output audio bitrate for AAC/MP3 (default: 192k)",
    )
    parser.add_argument(
        "--title-suffix",
        default=" (Censored)",
        help="Text appended to the title metadata tag (default: ' (Censored)')",
    )
    parser.add_argument(
        "--start", "-s",
        default=None,
        help="Start time to process from, e.g. 1:32:10 or 5420 (seconds). Useful for testing a chapter.",
    )
    parser.add_argument(
        "--end", "-e",
        default=None,
        help="End time to process to, e.g. 1:45:00 or 6300 (seconds).",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"❌ File not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.with_stem(input_path.stem + "_censored")

    if output_path == input_path:
        print("❌ Output path is the same as input. Use --output to specify a different path.", file=sys.stderr)
        sys.exit(1)

    print(f"📚 Input:   {input_path}")
    print(f"💾 Output:  {output_path}")
    print(f"🤖 Model:   {args.model}  (beam={args.beam_size})")
    print(f"🔊 Bitrate: {args.bitrate}")
    print()

    from censor_core import _parse_time
    start_time = _parse_time(args.start) if args.start else None
    end_time   = _parse_time(args.end)   if args.end   else None

    result = censor_audiobook(
        input_path=input_path,
        output_path=output_path,
        model_size=args.model,
        beam_size=args.beam_size,
        audio_bitrate=args.bitrate,
        title_suffix=args.title_suffix,
        start_time=start_time,
        end_time=end_time,
        progress_cb=print,
    )

    print()
    print("=" * 50)
    print(f"✅  Complete!")
    print(f"🚫  Words censored: {result.total_censored}")
    print(f"⏱️   Audio duration: {result.duration / 3600:.1f} hours")
    print(f"💾  Saved to: {output_path}")


if __name__ == "__main__":
    main()
