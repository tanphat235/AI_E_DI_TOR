"""Download a talk from YouTube for the raw-clips / shorts pipeline.

Wraps yt-dlp rather than shelling out to a standalone binary, so the only new
dependency is the pip package. Always remuxes to a single mp4 named
``source.mp4`` under ``<work-dir>/source/`` -- the rest of the pipeline
(transcribe.py, segment_qa.py, cut_raw_clips.py) takes ``--source`` as a
plain path and does not care where the file came from.

Example:
  .\\.venv\\Scripts\\python.exe scripts\\shorts\\fetch_youtube.py ^
    --url https://youtu.be/XXXXXXXXXXX --work-dir projects\\myjob
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def download(url: str, out_dir: Path, *, format_sel: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_tmpl = str(out_dir / "source.%(ext)s")
    cmd = [
        sys.executable,
        "-m",
        "yt_dlp",
        "-f",
        format_sel,
        "--merge-output-format",
        "mp4",
        "--no-playlist",
        "-o",
        out_tmpl,
        # Printed once the final container exists -- after ffmpeg has merged
        # separate video/audio streams, not the (possibly two) temp files
        # yt-dlp downloads before that.
        "--print",
        "after_move:filepath",
        url,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        raise SystemExit(proc.stderr[-2000:] or "yt-dlp failed")
    lines = [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()]
    if not lines:
        raise SystemExit("yt-dlp reported no output file; stdout was:\n" + proc.stdout[-2000:])
    return Path(lines[-1])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download a talk from YouTube.")
    parser.add_argument("--url", required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument(
        "--format",
        default="bv*[height<=1080]+ba/b[height<=1080]",
        help="yt-dlp format selector. Capped at 1080p: the shorts pipeline "
        "crops and downscales anyway, so a 4K source only costs time and disk.",
    )
    args = parser.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    source_dir = args.work_dir / "source"
    path = download(args.url, source_dir, format_sel=args.format)
    if not path.is_file():
        raise SystemExit(f"yt-dlp reported {path} but it does not exist")
    print(json.dumps({"source": str(path)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
