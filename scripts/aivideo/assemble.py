"""Assemble the stills, the narration and the captions into a finished video.

One still per line, held for exactly the span the narration measured, with a
slow push so a static drawing does not read as a slideshow. The caption is the
written line, not a transcription, so it is correct by construction.

The caption style is the one measured off the approved reference and shared
with the shorts pipeline -- Segoe UI Black, the same yellow, the same scrim --
so a generated video and a cut video look like the same channel.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import imageio_ffmpeg

# layout_center owns every measured style value the channel uses; it is a
# sibling of this folder, so a generated video and a cut video match.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "shorts"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import layout_center as lc  # noqa: E402  (needs the path above)

FPS = 30
# A 16:9 video, the shape the doodle style was written for.
FRAME_W, FRAME_H = 1920, 1080
# How far the slow push travels over a shot. 6% is enough to feel alive and
# small enough that the drawing's lines do not visibly crawl.
ZOOM = 1.06
MUSIC_UNDER_DB = 16.0


def work_dir(script_path: str) -> Path:
    """Where this video's artefacts live: the project holding the script.

    The tool sits in `scripts/`, which is source and is tracked; everything it
    produces belongs beside its own script under `projects/<name>/`, which is
    project data and is not. So `projects/x/scripts/y.json` writes to
    `projects/x/`.
    """
    return Path(script_path).resolve().parents[1]


def images_for(doc: dict, work: Path) -> list[Path]:
    d = work / ".aive" / "images" / doc["id"]
    out = []
    for i in range(1, len(doc["lines"]) + 1):
        hits = sorted(d.glob(f"{i:03d}_*.png"))
        if not hits:
            raise SystemExit(f"missing image {i:03d} in {d}; run generate_images.py")
        out.append(hits[0])
    return out


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: assemble.py <script.json>")
        return 2
    doc = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    work = work_dir(sys.argv[1])
    timing = json.loads(
        (work / ".aive" / "audio" / f"{doc['id']}.json").read_text(encoding="utf-8")
    )
    spans = timing["lines"]
    images = images_for(doc, work)
    mp3 = work / ".aive" / "audio" / f"{doc['id']}.mp3"
    out = work / "output" / f"{doc['id']}.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    scratch = work / ".aive" / "text"
    scratch.mkdir(parents=True, exist_ok=True)

    style = lc.CentreStyle(frame_w=FRAME_W, frame_h=FRAME_H, caption_size=46)
    total = spans[-1]["end"]

    # A shot runs from its own line's start to the *next* line's start, not to
    # its own end: the small silences between lines have to be covered by
    # something, and holding the picture that is already up is what an editor
    # would do. The first shot starts at 0 and the last runs to the end, so the
    # concatenated picture is exactly as long as the narration and the captions
    # cannot drift against it.
    bounds = [0.0] + [s["start"] for s in spans[1:]] + [total]

    ff = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [ff, "-y"]
    chains = []
    for i, img in enumerate(images):
        dur = max(0.4, bounds[i + 1] - bounds[i])
        frames = max(1, int(round(dur * FPS)))
        # -framerate must match FPS and zoompan must take d=1. Given a
        # multi-frame input, zoompan emits `d` frames for *each* one, so the
        # first shot alone became six minutes long and -t cropped the video to
        # that single picture -- 62 s of narration over one still.
        cmd += ["-loop", "1", "-framerate", str(FPS), "-t", f"{dur:.3f}", "-i", str(img)]
        # The still is scaled up first and cropped back; zooming a 1920-wide
        # source directly makes the ink shimmer.
        chains.append(
            f"[{i}:v]scale={FRAME_W * 2}:{FRAME_H * 2},"
            f"zoompan=z='1+{ZOOM - 1:.4f}*on/{frames}':d=1"
            f":x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'"
            f":s={FRAME_W}x{FRAME_H}:fps={FPS},setsar=1[s{i}]"
        )
    cmd += ["-i", str(mp3)]
    audio_idx = len(images)

    chains.append("".join(f"[s{i}]" for i in range(len(images)))
                  + f"concat=n={len(images)}:v=1:a=0[bg]")

    # Captions, drawn with the shorts pipeline's own style helpers.
    line_h = int(style.caption_size * style.line_ratio)
    top = FRAME_H - 190
    draws = []
    for i, span in enumerate(spans):
        lines = lc.wrap(span["text"], size=style.caption_size, style=style)[:2]
        enable = f"between(t,{span['start']:.3f},{span['end']:.3f})"
        draws.append(
            lc._plate(style=style, size=style.caption_size, top=top,
                      line_h=line_h, slots=2, enable=enable)
        )
        draws += lc._text_block(
            lines=lines, style=style, size=style.caption_size,
            colors=(style.caption_color,), top=top, line_h=line_h, slots=2,
            text_dir=scratch, stem=f"{doc['id']}_{i:03d}", enable=enable,
        )
    chains.append("[bg]" + ",".join(draws) + "[v]")

    graph = ";".join(chains)
    script_file = work / ".aive" / f"{doc['id']}_graph.txt"
    script_file.write_text(graph, encoding="utf-8")

    cmd += [
        "-filter_complex_script", str(script_file),
        "-map", "[v]", "-map", f"{audio_idx}:a",
        "-t", f"{total:.3f}",
        "-c:v", "libx264", "-preset", "medium", "-crf", "19", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", str(out),
    ]
    print(f"{len(images)} shots, narration {total:.2f}s -> {out.name}")
    proc = subprocess.run(cmd, capture_output=True, text=True,
                          encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        print((proc.stderr or "")[-1500:])
        return 1
    print(f"wrote {out}  {out.stat().st_size / 1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
