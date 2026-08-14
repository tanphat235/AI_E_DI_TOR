"""Transcribe a talk video/audio to JSON for short-clip building.

Example:
  .\\.venv\\Scripts\\python.exe scripts\\shorts\\transcribe.py ^
    --source video.mp4 ^
    --work-dir projects\\myjob
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def _ffmpeg() -> Path:
    try:
        import imageio_ffmpeg

        return Path(imageio_ffmpeg.get_ffmpeg_exe())
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"ffmpeg not found via imageio-ffmpeg: {exc}") from exc


def log(msg: str, log_file: Path) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("a", encoding="utf-8") as fh:
        fh.write(msg + "\n")
    sys.stdout.buffer.write((msg + "\n").encode("utf-8", errors="replace"))
    sys.stdout.flush()


def extract_wav(source: Path, wav: Path, ffmpeg: Path) -> None:
    wav.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(ffmpeg),
        "-y",
        "-i",
        str(source),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(wav),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        raise SystemExit(proc.stderr[-500:] if proc.stderr else "ffmpeg extract failed")


def transcribe(
    wav: Path,
    out: Path,
    log_file: Path,
    *,
    model_name: str,
    language: str,
    force: bool,
) -> int:
    from faster_whisper import WhisperModel

    if out.exists() and out.stat().st_size > 1000 and not force:
        log(f"CACHE_EXISTS {out}", log_file)
        return 0
    if not wav.is_file():
        log(f"missing wav {wav}", log_file)
        return 1

    log_file.write_text("", encoding="utf-8")
    log(f"Loading model {model_name} (cpu int8)...", log_file)
    model = WhisperModel(model_name, device="cpu", compute_type="int8")
    log("Transcribing...", log_file)
    t0 = time.time()
    segments, info = model.transcribe(
        str(wav),
        language=language,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 800},
        word_timestamps=False,
        beam_size=1,
        best_of=1,
    )
    rows: list[dict] = []
    for i, seg in enumerate(segments):
        rows.append(
            {
                "i": i,
                "start": round(seg.start, 3),
                "end": round(seg.end, 3),
                "text": seg.text.strip(),
            }
        )
        if i % 20 == 0:
            log(f"seg {i} t={seg.start:.1f}s text={seg.text.strip()[:60]}", log_file)
            out.write_text(
                json.dumps(
                    {
                        "language": language,
                        "duration": None,
                        "segments": rows,
                        "partial": True,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

    doc = {
        "language": info.language,
        "duration": info.duration,
        "segments": rows,
        "partial": False,
    }
    out.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"DONE segs={len(rows)} elapsed={round(time.time() - t0, 1)}s -> {out}", log_file)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Transcribe talk media for shorts pipeline.")
    parser.add_argument("--source", type=Path, required=True, help="Input video or audio file.")
    parser.add_argument(
        "--work-dir",
        type=Path,
        required=True,
        help="Project work dir (writes .aive/transcript.json here).",
    )
    parser.add_argument("--model", default="base", help="faster-whisper model name.")
    parser.add_argument("--language", default="vi")
    parser.add_argument("--force", action="store_true", help="Ignore cached transcript.")
    parser.add_argument(
        "--skip-extract",
        action="store_true",
        help="Reuse existing .aive/audio.wav if present.",
    )
    args = parser.parse_args(argv)

    work = args.work_dir
    cache = work / ".aive"
    wav = cache / "audio.wav"
    out = cache / "transcript.json"
    log_file = cache / "progress.txt"
    cache.mkdir(parents=True, exist_ok=True)

    if not args.source.is_file():
        print(f"source not found: {args.source}", file=sys.stderr)
        return 2

    if not (args.skip_extract and wav.is_file()):
        log(f"Extracting wav from {args.source}...", log_file)
        extract_wav(args.source, wav, _ffmpeg())

    return transcribe(
        wav,
        out,
        log_file,
        model_name=args.model,
        language=args.language,
        force=args.force,
    )


if __name__ == "__main__":
    raise SystemExit(main())
