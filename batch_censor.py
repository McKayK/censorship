"""
batch_censor.py — Process multiple audiobooks one after another.

Usage:
    python batch_censor.py "Book1.m4b" "Book2.m4b" "Book3.m4b"
    python batch_censor.py *.m4b
    python batch_censor.py --model medium "Book1.m4b" "Book2.m4b"
"""

import argparse
import sys
import time
from pathlib import Path

from censor_core import censor_audiobook


def fmt_time(seconds: float) -> str:
    h, m = divmod(int(seconds), 3600)
    m, s = divmod(m, 60)
    return f"{h}h {m:02d}m {s:02d}s"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Censor f-words from multiple audiobooks in sequence.",
    )
    parser.add_argument("inputs", nargs="+", help="Audiobook files to process")
    parser.add_argument("--model", "-m", default="small",
                        choices=["tiny", "small", "medium", "large"],
                        help="Whisper model size (default: small)")
    parser.add_argument("--beam-size", "-b", type=int, default=5,
                        help="Whisper beam size (default: 5)")
    parser.add_argument("--bitrate", default="192k",
                        help="Output audio bitrate (default: 192k)")
    parser.add_argument("--title-suffix", default=" (Censored)",
                        help="Suffix appended to title metadata (default: ' (Censored)')")
    args = parser.parse_args()

    books = [Path(p) for p in args.inputs]

    # Skip already-censored files
    skipped = [b for b in books if "_censored" in b.stem]
    if skipped:
        print("⚠️  Skipping already-censored files:")
        for s in skipped:
            print(f"   {s.name}")
    books = [b for b in books if "_censored" not in b.stem]

    if not books:
        print("No books to process.")
        sys.exit(0)

    # Validate all files exist before starting
    missing = [b for b in books if not b.exists()]
    if missing:
        for m in missing:
            print(f"❌ File not found: {m}")
        sys.exit(1)

    print(f"📚 Batch censoring {len(books)} book(s)")
    print(f"🤖 Model: {args.model}  beam={args.beam_size}")
    print("=" * 60)

    summary: list[tuple[str, int, float, str]] = []  # (title, count, duration, status)
    batch_start = time.time()

    for idx, book in enumerate(books):
        output_path = book.with_stem(book.stem + "_censored")
        print(f"\n[{idx + 1}/{len(books)}] {book.name}")
        print(f"  → {output_path.name}")

        book_start = time.time()
        try:
            result = censor_audiobook(
                input_path=book,
                output_path=output_path,
                model_size=args.model,
                beam_size=args.beam_size,
                audio_bitrate=args.bitrate,
                title_suffix=args.title_suffix,
                progress_cb=lambda msg: print(f"  {msg}"),
            )
            elapsed = time.time() - book_start
            summary.append((book.stem, result.total_censored, result.duration, "✅"))
            print(f"  ✅ Done in {fmt_time(elapsed)} — {result.total_censored} word(s) censored")

        except Exception as e:
            elapsed = time.time() - book_start
            summary.append((book.stem, 0, 0, "❌"))
            print(f"  ❌ Failed after {fmt_time(elapsed)}: {e}")

    # Final summary
    total_elapsed = time.time() - batch_start
    print("\n" + "=" * 60)
    print("BATCH COMPLETE")
    print("=" * 60)
    for title, count, duration, status in summary:
        dur_str = f"{duration / 3600:.1f}h" if duration else "  —  "
        print(f"  {status}  {title[:45]:<45}  {count:>4} F-words  ({dur_str})")
    print("-" * 60)
    total_words = sum(c for _, c, _, s in summary if s == "✅")
    print(f"  Total censored: {total_words} F-words across {len(books)} book(s)")
    print(f"  Total time:     {fmt_time(total_elapsed)}")


if __name__ == "__main__":
    main()