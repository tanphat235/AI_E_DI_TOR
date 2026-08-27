"""Build vertical split-screen shorts from a talk video + B-roll folder.

Layout: top = talk video, bottom = B-roll. Audio from talk only.
Writes paired files into <work-dir>/output/shorts/:
  clip_001.mp4 + clip_001.jpg

Example:
  .\\.venv\\Scripts\\python.exe scripts\\shorts\\build_shorts.py ^
    --source talk.mp4 ^
    --work-dir projects\\myjob ^
    --broll-dir projects\\stock-pexels\\raw ^
    --badge "THAY PHAP HOA" ^
    --topic "NHAN QUA"
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path


def _ffmpeg() -> Path:
    try:
        import imageio_ffmpeg

        return Path(imageio_ffmpeg.get_ffmpeg_exe())
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"ffmpeg not found via imageio-ffmpeg: {exc}") from exc


def _media_duration(path: Path) -> float:
    """Duration in seconds. PyAV, because the vendored wheel has no ffprobe."""
    import av

    with av.open(str(path)) as container:
        if container.duration:
            return float(container.duration) / 1_000_000.0
        stream = container.streams.video[0]
        return float(stream.duration * stream.time_base) if stream.duration else 0.0


@dataclass(frozen=True, slots=True)
class Settings:
    source: Path
    work_dir: Path
    broll_dir: Path
    max_sec: float = 175.0
    min_sec: float = 35.0
    target_sec: float = 110.0
    pause_cut: float = 1.2
    out_w: int = 1080
    out_h: int = 1920
    badge: str = "SPEAKER"
    topic: str = "SHORT"
    title_fallback: str = "Short clip"
    limit: int = 0
    # "w:h:x:y" applied to the talk before framing, to drop baked-on
    # decoration such as a pillarbox border or a scrolling promo banner.
    src_crop: str = ""
    # Read .aive/answer_segments.json instead of re-deriving chapters, so a
    # purpose-built segmenter (see segment_qa.py) can own the cut points.
    reuse_segments: bool = False
    # Alternate segments JSON, for rendering a hand-picked span.
    segments_file: Path | None = None
    # Sequence many B-roll scenes under one clip rather than looping one.
    broll_bed: bool = False
    broll_chunk: float = 12.0
    broll_stride: float = 47.0


def build_segments(
    segments: list[dict],
    *,
    max_sec: float,
    min_sec: float,
    target_sec: float,
    pause_cut: float,
    title_fallback: str,
) -> list[dict]:
    """Pack transcript lines into short chapters at silence boundaries."""
    if not segments:
        return []

    chapters: list[list[dict]] = []
    cur: list[dict] = [segments[0]]
    for s in segments[1:]:
        gap = s["start"] - cur[-1]["end"]
        dur = s["end"] - cur[0]["start"]
        if dur >= max_sec or (dur >= target_sec and gap >= pause_cut) or (
            dur >= min_sec and gap >= 2.5
        ):
            chapters.append(cur)
            cur = [s]
        else:
            cur.append(s)
    if cur:
        chapters.append(cur)

    out: list[dict] = []
    for i, body in enumerate(chapters):
        start = body[0]["start"]
        end = body[-1]["end"]
        dur = end - start
        if dur < min_sec and out:
            prev = out[-1]
            prev["end"] = end
            prev["duration"] = round(end - prev["start"], 3)
            prev["text"] = (prev["text"] + " " + " ".join(x["text"] for x in body)).strip()
            continue
        text = " ".join(x["text"] for x in body).strip()
        title = _title_from(text, index=i + 1, fallback=title_fallback)
        out.append(
            {
                "id": f"clip_{i + 1:03d}",
                "start": round(start, 3),
                "end": round(end, 3),
                "duration": round(dur, 3),
                "text": text,
                "title": title,
            }
        )

    final: list[dict] = []
    for seg in out:
        if seg["duration"] <= max_sec:
            final.append(seg)
            continue
        t0 = seg["start"]
        n = 0
        while t0 < seg["end"] - min_sec:
            t1 = min(seg["end"], t0 + max_sec)
            final.append(
                {
                    "id": f"{seg['id']}_{n:02d}",
                    "start": round(t0, 3),
                    "end": round(t1, 3),
                    "duration": round(t1 - t0, 3),
                    "text": seg["text"],
                    "title": f"{seg['title']} ({n + 1})",
                }
            )
            t0 = t1
            n += 1
    return final


def _title_from(text: str, *, index: int, fallback: str) -> str:
    clean = re.sub(r"\s+", " ", text).strip(" .,;:-")
    for sep in (". ", "? ", "! ", ", "):
        if sep in clean[:80]:
            clean = clean.split(sep, 1)[0]
            break
    if len(clean) < 12:
        clean = f"{fallback} #{index:02d}"
    if len(clean) > 48:
        clean = clean[:48].rsplit(" ", 1)[0] + "..."
    return clean


def broll_files(broll_dir: Path) -> list[Path]:
    files = sorted(broll_dir.glob("*.mp4"))
    if not files:
        raise FileNotFoundError(f"no b-roll mp4 in {broll_dir}")
    return files


def build_bed(
    files: list[Path],
    bed: Path,
    *,
    ffmpeg: Path,
    chunk: float,
    out_w: int,
    half_h: int,
) -> Path:
    """Concatenate every B-roll scene into one strip sized for the bottom half.

    Looping a single 10s clip under a two-minute answer reads as obviously
    automated. Encoding the strip once and reading a different offset per clip
    gives each clip a changing bottom without re-encoding B-roll per clip.
    """
    if bed.is_file() and bed.stat().st_size > 100_000:
        return bed
    bed.parent.mkdir(parents=True, exist_ok=True)
    cmd = [str(ffmpeg), "-y"]
    for f in files:
        cmd += ["-t", f"{chunk:.3f}", "-i", str(f)]
    parts = "".join(
        f"[{i}:v]scale={out_w}:{half_h}:force_original_aspect_ratio=increase,"
        f"crop={out_w}:{half_h},fps=30,setsar=1[b{i}];"
        for i in range(len(files))
    )
    joins = "".join(f"[b{i}]" for i in range(len(files)))
    filt = f"{parts}{joins}concat=n={len(files)}:v=1:a=0[v]"
    cmd += [
        "-filter_complex", filt, "-map", "[v]",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "22",
        "-pix_fmt", "yuv420p", "-an", "-movflags", "+faststart", str(bed),
    ]
    proc = subprocess.run(
        cmd, capture_output=True, text=True, encoding="utf-8", errors="replace"
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr[-1200:] if proc.stderr else "bed build failed")
    return bed


def render_clip(
    seg: dict,
    *,
    source: Path,
    broll: Path,
    shorts_dir: Path,
    ffmpeg: Path,
    out_w: int,
    out_h: int,
    src_crop: str = "",
    bed_offset: float | None = None,
) -> Path:
    shorts_dir.mkdir(parents=True, exist_ok=True)
    out = shorts_dir / f"{seg['id']}.mp4"
    half_h = out_h // 2
    start, dur = seg["start"], seg["duration"]
    pre = f"crop={src_crop}," if src_crop else ""
    filt = (
        f"[0:v]{pre}scale={out_w}:{half_h}:force_original_aspect_ratio=increase,"
        f"crop={out_w}:{half_h},fps=30,setsar=1[top];"
        f"[1:v]scale={out_w}:{half_h}:force_original_aspect_ratio=increase,"
        f"crop={out_w}:{half_h},fps=30,setsar=1[bot];"
        f"[top][bot]vstack=inputs=2[v]"
    )
    bottom_in = ["-stream_loop", "-1"]
    if bed_offset is not None:
        bottom_in += ["-ss", f"{bed_offset:.3f}"]
    bottom_in += ["-t", f"{dur:.3f}", "-i", str(broll)]
    cmd = [
        str(ffmpeg),
        "-y",
        "-ss",
        f"{start:.3f}",
        "-t",
        f"{dur:.3f}",
        "-i",
        str(source),
        *bottom_in,
        "-filter_complex",
        filt,
        "-map",
        "[v]",
        "-map",
        "0:a",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "20",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        "-shortest",
        "-movflags",
        "+faststart",
        str(out),
    ]
    proc = subprocess.run(
        cmd, capture_output=True, text=True, encoding="utf-8", errors="replace"
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr[-800:] if proc.stderr else "ffmpeg failed")
    return out


def make_thumbnail(
    seg: dict,
    clip: Path,
    *,
    shorts_dir: Path,
    ffmpeg: Path,
    badge: str,
    topic: str,
) -> Path:
    shorts_dir.mkdir(parents=True, exist_ok=True)
    frame = shorts_dir / f"{seg['id']}_frame.jpg"
    thumb = shorts_dir / f"{seg['id']}.jpg"
    mid = max(0.8, min(seg["duration"] * 0.35, seg["duration"] - 0.5))
    subprocess.run(
        [
            str(ffmpeg),
            "-y",
            "-ss",
            f"{mid:.2f}",
            "-i",
            str(clip),
            "-frames:v",
            "1",
            "-q:v",
            "2",
            str(frame),
        ],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    title = seg["title"].replace("'", "").replace('"', "").replace(":", " -").replace("%", "")
    lines = textwrap.wrap(title, width=16)[:3] or [seg["id"]]
    font = "C\\:/Windows/Fonts/arial.ttf"
    if not Path(r"C:\Windows\Fonts\arial.ttf").is_file():
        font = "C\\:/Windows/Fonts/tahoma.ttf"

    badge_safe = badge.replace("'", "").replace(":", " -")
    topic_safe = topic.replace("'", "").replace(":", " -")
    draws = [
        f"drawbox=x=0:y=80:w=iw:h={80 + 78 * len(lines)}:color=black@0.55:t=fill",
        (
            f"drawtext=fontfile={font}:text='{badge_safe}':fontsize=36:fontcolor=gold:"
            f"borderw=3:bordercolor=black:x=(w-text_w)/2:y=40"
        ),
    ]
    y = 100
    for line in lines:
        safe = line.replace("\\", "\\\\").replace(":", "\\:")
        draws.append(
            f"drawtext=fontfile={font}:text='{safe}':fontsize=58:fontcolor=white:"
            f"borderw=4:bordercolor=black:x=(w-text_w)/2:y={y}"
        )
        y += 78
    draws.append(
        f"drawtext=fontfile={font}:text='{topic_safe}':fontsize=42:fontcolor=white:"
        f"borderw=3:bordercolor=black:x=(w-text_w)/2:y=h-120"
    )
    vf = ",".join(draws)
    proc = subprocess.run(
        [str(ffmpeg), "-y", "-i", str(frame), "-vf", vf, "-q:v", "3", str(thumb)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if proc.returncode != 0 or not thumb.is_file():
        if frame.is_file():
            thumb.write_bytes(frame.read_bytes())
    if frame.is_file():
        frame.unlink(missing_ok=True)
    return thumb


def run(settings: Settings) -> int:
    cache = settings.work_dir / ".aive"
    shorts_dir = settings.work_dir / "output" / "shorts"
    transcript_path = cache / "transcript.json"
    segments_path = settings.segments_file or cache / "answer_segments.json"
    # The manifest follows the segments list, so a one-off span rendered with
    # --segments-file does not clobber the main run's manifest.
    manifest_path = (
        cache / f"render_manifest_{settings.segments_file.stem}.json"
        if settings.segments_file is not None
        else cache / "render_manifest.json"
    )

    if not settings.source.is_file():
        raise SystemExit(f"source not found: {settings.source}")

    # --segments-file names a list to render, so it always reads and is never
    # written back to. Deriving chapters would otherwise overwrite the caller's
    # hand-picked spans.
    if settings.reuse_segments or settings.segments_file is not None:
        if not segments_path.is_file():
            raise SystemExit(
                f"missing segments: {segments_path}\n"
                "Run scripts/shorts/segment_qa.py first, or drop --reuse-segments."
            )
        segs = json.loads(segments_path.read_text(encoding="utf-8"))
    else:
        if not transcript_path.is_file():
            raise SystemExit(
                f"missing transcript: {transcript_path}\n"
                "Run scripts/shorts/transcribe.py first."
            )
        doc = json.loads(transcript_path.read_text(encoding="utf-8"))
        segs = build_segments(
            doc["segments"],
            max_sec=settings.max_sec,
            min_sec=settings.min_sec,
            target_sec=settings.target_sec,
            pause_cut=settings.pause_cut,
            title_fallback=settings.title_fallback,
        )
        cache.mkdir(parents=True, exist_ok=True)
        segments_path.write_text(
            json.dumps(segs, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    if settings.limit:
        segs = segs[: settings.limit]
    print(f"segments={len(segs)}")
    for s in segs[:8]:
        print(f"  {s['id']} {s['start']:.1f}-{s['end']:.1f} ({s['duration']:.1f}s) {s['title']}")

    brolls = broll_files(settings.broll_dir)
    ffmpeg = _ffmpeg()
    bed: Path | None = None
    bed_dur = 0.0
    if settings.broll_bed:
        bed = build_bed(
            brolls,
            cache / "broll_bed.mp4",
            ffmpeg=ffmpeg,
            chunk=settings.broll_chunk,
            out_w=settings.out_w,
            half_h=settings.out_h // 2,
        )
        bed_dur = _media_duration(bed)
        print(f"broll bed: {bed.name} {bed_dur:.1f}s from {len(brolls)} scenes")
    rendered: list[dict] = []
    for i, seg in enumerate(segs):
        broll = brolls[i % len(brolls)]
        offset: float | None = None
        if bed is not None and bed_dur > 0:
            broll = bed
            offset = (i * settings.broll_stride) % bed_dur
        out_clip = shorts_dir / f"{seg['id']}.mp4"
        out_thumb = shorts_dir / f"{seg['id']}.jpg"
        print(f"[{i + 1}/{len(segs)}] render {seg['id']} + {broll.name}")
        try:
            if out_clip.is_file() and out_clip.stat().st_size > 10000:
                clip = out_clip
                print("  skip existing clip")
            else:
                clip = render_clip(
                    seg,
                    source=settings.source,
                    broll=broll,
                    shorts_dir=shorts_dir,
                    ffmpeg=ffmpeg,
                    out_w=settings.out_w,
                    out_h=settings.out_h,
                    src_crop=settings.src_crop,
                    bed_offset=offset,
                )
            if out_thumb.is_file() and out_thumb.stat().st_size > 1000:
                thumb = out_thumb
                print("  skip existing thumb")
            else:
                thumb = make_thumbnail(
                    seg,
                    clip,
                    shorts_dir=shorts_dir,
                    ffmpeg=ffmpeg,
                    badge=settings.badge,
                    topic=settings.topic,
                )
            rendered.append(
                {
                    "id": seg["id"],
                    "start": seg["start"],
                    "end": seg["end"],
                    "duration": seg["duration"],
                    "title": seg["title"],
                    "clip": str(clip),
                    "thumb": str(thumb),
                    "broll": broll.name,
                    "broll_offset": None if offset is None else round(offset, 2),
                }
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  FAIL: {exc}")

    manifest_path.write_text(json.dumps(rendered, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"DONE rendered={len(rendered)} -> {shorts_dir}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build split-screen talk shorts.")
    parser.add_argument("--source", type=Path, required=True, help="Talk video file.")
    parser.add_argument("--work-dir", type=Path, required=True, help="Job folder.")
    parser.add_argument("--broll-dir", type=Path, required=True, help="Folder of B-roll mp4.")
    parser.add_argument("--max-sec", type=float, default=175.0)
    parser.add_argument("--min-sec", type=float, default=35.0)
    parser.add_argument("--target-sec", type=float, default=110.0)
    parser.add_argument("--badge", default="SPEAKER", help="Top badge on thumbnail.")
    parser.add_argument("--topic", default="SHORT", help="Bottom topic label on thumbnail.")
    parser.add_argument("--title-fallback", default="Short clip")
    parser.add_argument(
        "--src-crop",
        default="",
        help='"w:h:x:y" crop on the talk, to drop a pillarbox or promo banner.',
    )
    parser.add_argument(
        "--reuse-segments",
        action="store_true",
        help="Use .aive/answer_segments.json as-is (see segment_qa.py).",
    )
    parser.add_argument(
        "--segments-file",
        type=Path,
        default=None,
        help="Segments JSON to render instead of .aive/answer_segments.json.",
    )
    parser.add_argument(
        "--broll-bed",
        action="store_true",
        help="Sequence every B-roll scene into one strip instead of looping one clip.",
    )
    parser.add_argument("--broll-chunk", type=float, default=12.0,
                        help="Seconds taken from each B-roll scene for the strip.")
    parser.add_argument("--broll-stride", type=float, default=47.0,
                        help="Seconds of strip offset between consecutive clips.")
    parser.add_argument("--limit", type=int, default=0,
                        help="Render only the first N clips, to preview framing.")
    args = parser.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    settings = Settings(
        source=args.source,
        work_dir=args.work_dir,
        broll_dir=args.broll_dir,
        max_sec=args.max_sec,
        min_sec=args.min_sec,
        target_sec=args.target_sec,
        badge=args.badge,
        topic=args.topic,
        title_fallback=args.title_fallback,
        limit=args.limit,
        src_crop=args.src_crop,
        reuse_segments=args.reuse_segments,
        segments_file=args.segments_file,
        broll_bed=args.broll_bed,
        broll_chunk=args.broll_chunk,
        broll_stride=args.broll_stride,
    )
    return run(settings)


if __name__ == "__main__":
    raise SystemExit(main())
