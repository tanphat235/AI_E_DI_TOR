"""Cut a long talk into whole raw clips at real content boundaries.

Reads ``.aive/answer_segments.json`` (segment_qa.py's output -- run that
first) and slices the source video at each span. Every clip is a re-encode,
not a stream copy: ``-c copy`` can only start on a keyframe, which drifts the
cut a keyframe-interval early and can carry in a few seconds of the previous
answer. These clips are meant to hold one whole teaching or one whole
question-and-answer each, so landing on the exact second the boundary was
chosen at matters more than the encode being free.

No crop, caption or B-roll -- that is build_shorts.py's job once a clip is
picked to become a short. This just splits the recording.

Example:
  .\\.venv\\Scripts\\python.exe scripts\\shorts\\cut_raw_clips.py ^
    --source talk.mp4 --work-dir projects\\myjob
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import unicodedata
from pathlib import Path


def _ffmpeg() -> Path:
    try:
        import imageio_ffmpeg

        return Path(imageio_ffmpeg.get_ffmpeg_exe())
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"ffmpeg not found via imageio-ffmpeg: {exc}") from exc


def _slug(text: str, *, max_len: int = 60) -> str:
    decomposed = unicodedata.normalize("NFD", text.lower()).replace("đ", "d").replace("Đ", "d")
    ascii_text = "".join(c for c in decomposed if unicodedata.category(c) != "Mn")
    slug = re.sub(r"[^a-z0-9]+", "-", ascii_text).strip("-")
    return slug[:max_len].rstrip("-") or "clip"


def cut_clip(
    source: Path,
    out: Path,
    ffmpeg: Path,
    *,
    start: float,
    end: float,
    crf: int,
    preset: str,
) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(ffmpeg),
        "-y",
        "-ss",
        f"{start:.3f}",
        "-i",
        str(source),
        "-t",
        f"{end - start:.3f}",
        "-c:v",
        "libx264",
        "-preset",
        preset,
        "-crf",
        str(crf),
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        str(out),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        raise SystemExit(proc.stderr[-800:] if proc.stderr else f"ffmpeg failed on {out.name}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Cut a talk into whole raw content clips.")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument(
        "--out-dir", type=Path, default=None, help="Defaults to <work-dir>/raw_clips."
    )
    parser.add_argument("--crf", type=int, default=18)
    parser.add_argument(
        "--preset", default="veryfast", help="libx264 preset; slower = smaller file, same crf."
    )
    args = parser.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    if not args.source.is_file():
        raise SystemExit(f"source not found: {args.source}")

    segments_path = args.work_dir / ".aive" / "answer_segments.json"
    if not segments_path.is_file():
        raise SystemExit(f"missing {segments_path}; run segment_qa.py first")
    segments = json.loads(segments_path.read_text(encoding="utf-8"))

    out_dir = args.out_dir or (args.work_dir / "raw_clips")
    ffmpeg = _ffmpeg()

    manifest: list[dict] = []
    for seg in segments:
        slug = _slug(seg["title"])
        out = out_dir / f"{seg['id']}_{slug}.mp4"
        print(f"cutting {out.name}  {seg['start']:.1f}-{seg['end']:.1f}  ({seg['duration']:.1f}s)")
        cut_clip(
            args.source,
            out,
            ffmpeg,
            start=seg["start"],
            end=seg["end"],
            crf=args.crf,
            preset=args.preset,
        )
        manifest.append({**seg, "path": str(out)})

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"-> {len(manifest)} clips in {out_dir}")
    print(f"-> {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
