"""Fetch a video's own subtitles, to check the machine transcript against.

The captions burned into these shorts came from running speech recognition over
a re-upload's audio, and on colloquial Vietnamese they produce syllables that
are plausible and meaningless -- "dính mắt" for "dính mắc", which is the word
the passage is about. Decoding settings help (see transcribe.py) but do not
solve it: the recogniser has no way to know the sentence has to mean something.

A second opinion does. YouTube's own captions come from a different recogniser
trained on different data, so it makes **different** mistakes. Where the two
agree the text is almost certainly right; where they disagree is exactly where
a human -- or a language model reading for sense -- has to decide. That turns
reading forty lines into adjudicating five.

A human-written track, when the uploader provides one, is better still: then
the words are already right and only the timing has to be re-derived.

This only fetches and reports. Alignment against the project's own transcript
is a separate step, because the two live on different clocks: subtitles are
timed to the original upload and a project's cuts have their own.

Example:
  .\\.venv\\Scripts\\python.exe scripts\\shorts\\fetch_subs.py ^
    --url https://www.youtube.com/watch?v=... ^
    --work-dir projects\\myjob
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def describe(info: dict, lang: str) -> None:
    """Say what the video actually offers, before anything is downloaded."""
    manual = info.get("subtitles") or {}
    auto = info.get("automatic_captions") or {}
    print(f"  title    {info.get('title', '?')}")
    print(f"  duration {info.get('duration', 0) / 60:.1f} min")
    have_manual = lang in manual
    print(f"  written by a person, '{lang}': {'YES' if have_manual else 'no'}")
    print(f"  auto-generated, '{lang}':      {'yes' if lang in auto else 'NO'}")
    others = sorted(set(manual) - {lang})
    if others:
        print(f"  other written tracks: {', '.join(others[:12])}")
    if have_manual:
        print("  -> taking the written track; its words need no correction, only timing")
    elif lang in auto:
        print("  -> taking the auto track; it is a second opinion, not an answer key")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Fetch a video's subtitles for cross-checking.")
    ap.add_argument("--url", required=True)
    ap.add_argument("--work-dir", type=Path, required=True)
    ap.add_argument("--lang", default="vi")
    ap.add_argument("--list", action="store_true", help="Report what exists and stop.")
    ap.add_argument(
        "--cookies-from-browser",
        default=None,
        help="e.g. chrome, edge. Only needed if YouTube challenges the request.",
    )
    args = ap.parse_args(argv)

    import yt_dlp

    out_dir = args.work_dir / ".aive" / "subs"
    out_dir.mkdir(parents=True, exist_ok=True)

    probe: dict = {"quiet": True, "no_warnings": True, "skip_download": True}
    if args.cookies_from_browser:
        probe["cookiesfrombrowser"] = (args.cookies_from_browser,)
    with yt_dlp.YoutubeDL(probe) as ydl:
        info = ydl.extract_info(args.url, download=False)

    describe(info, args.lang)
    manual = info.get("subtitles") or {}
    auto = info.get("automatic_captions") or {}
    if args.lang not in manual and args.lang not in auto:
        print(f"\nno '{args.lang}' subtitles of either kind; nothing to fetch")
        return 1
    if args.list:
        return 0

    opts = dict(probe)
    opts.update(
        {
            "writesubtitles": args.lang in manual,
            "writeautomaticsub": args.lang not in manual,
            "subtitleslangs": [args.lang],
            # vtt first: YouTube serves it natively, so nothing is re-timed on
            # the way through a converter.
            "subtitlesformat": "vtt/srt/best",
            "outtmpl": str(out_dir / "%(id)s.%(ext)s"),
        }
    )
    with yt_dlp.YoutubeDL(opts) as ydl:
        ydl.download([args.url])

    got = sorted(p for p in out_dir.iterdir() if p.suffix in {".vtt", ".srt"})
    if not got:
        print("\nyt-dlp reported success but wrote no subtitle file")
        return 1
    for p in got:
        cues = p.read_text(encoding="utf-8", errors="replace").count("-->")
        print(f"\nwrote {p}  {p.stat().st_size / 1024:.0f} KB, {cues} cues")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
