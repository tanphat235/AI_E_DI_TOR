"""A YouTube link (or a file already on disk) to whole raw content clips.

Chains the four steps the README has you run by hand: fetch_youtube.py (only
if ``--url`` is given), transcribe.py, segment_qa.py, cut_raw_clips.py. One
command for the common case; run the four scripts separately when a step
needs non-default flags (a crop, a different Whisper model, VAD instead of
silencedetect, ...).

``--target-sec``/``--max-sec`` default far larger than the shorts pipeline's
(120/175s): the point here is one clip per whole teaching or whole
question-and-answer, not a short. A block is only forced to split before
``--max-sec`` at the best available pause -- never mid-sentence -- so a short
answer stays one clip and a long one is only cut where the audio actually
allows it.

Examples:
  .\\.venv\\Scripts\\python.exe scripts\\shorts\\ingest_talk.py ^
    --url https://youtu.be/XXXXXXXXXXX --job phap-thoai-01

  .\\.venv\\Scripts\\python.exe scripts\\shorts\\ingest_talk.py ^
    --source C:\\Users\\User\\Downloads\\talk.mp4 --job phap-thoai-01
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def run(cmd: list[str]) -> str:
    print("$ " + " ".join(f'"{c}"' if " " in c else c for c in cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    sys.stdout.write(proc.stdout)
    sys.stderr.write(proc.stderr)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)
    return proc.stdout


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--url", help="YouTube link to download first.")
    src.add_argument("--source", type=Path, help="Video already on disk.")
    parser.add_argument("--job", required=True, help="Project name; work-dir is projects/<job>.")
    parser.add_argument(
        "--projects-root",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "projects",
    )
    parser.add_argument("--model", default="small", help="faster-whisper model name.")
    parser.add_argument("--language", default="vi")
    parser.add_argument("--min-sec", type=float, default=45.0)
    parser.add_argument("--target-sec", type=float, default=300.0)
    parser.add_argument("--max-sec", type=float, default=900.0)
    parser.add_argument("--min-boundary", type=float, default=1.5)
    args = parser.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    work_dir = args.projects_root / args.job
    work_dir.mkdir(parents=True, exist_ok=True)

    if args.url:
        out = run(
            [
                sys.executable,
                str(HERE / "fetch_youtube.py"),
                "--url",
                args.url,
                "--work-dir",
                str(work_dir),
            ]
        )
        source = Path(json.loads(out.strip().splitlines()[-1])["source"])
    else:
        source = args.source
        if not source.is_file():
            raise SystemExit(f"source not found: {source}")

    run(
        [
            sys.executable,
            str(HERE / "transcribe.py"),
            "--source",
            str(source),
            "--work-dir",
            str(work_dir),
            "--model",
            args.model,
            "--language",
            args.language,
        ]
    )
    run(
        [
            sys.executable,
            str(HERE / "segment_qa.py"),
            "--work-dir",
            str(work_dir),
            "--min-sec",
            str(args.min_sec),
            "--target-sec",
            str(args.target_sec),
            "--max-sec",
            str(args.max_sec),
            "--min-boundary",
            str(args.min_boundary),
        ]
    )
    run(
        [
            sys.executable,
            str(HERE / "cut_raw_clips.py"),
            "--source",
            str(source),
            "--work-dir",
            str(work_dir),
        ]
    )
    print(f"-> raw clips in {work_dir / 'raw_clips'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
