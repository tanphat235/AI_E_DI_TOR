"""Assemble the stills, the narration and the captions into a finished video.

Each shot is held for exactly the span the narration measured. A shot is either
a generated picture (``image``) with a slow push, or the reference map
(``map``) with one region lit -- held still, because the realms are named one
after another over the same map and a push restarting every two seconds reads
as a stutter. A generated picture may name a ``where`` region; a small copy of
the map then sits in the corner with that region lit, so every scene says which
part of the cosmology it belongs to.

The caption is the written line, not a transcription, so it is correct by
construction. Its style is the one measured off the approved reference and
shared with the shorts pipeline -- Segoe UI Black, the same yellow, the same
scrim -- so a generated video and a cut video look like the same channel.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import imageio_ffmpeg

HERE = Path(__file__).resolve().parent
# layout_center owns every measured style value the channel uses; it is a
# sibling of this folder, so a generated video and a cut video match.
sys.path.insert(0, str(HERE.parent / "shorts"))
sys.path.insert(0, str(HERE))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import layout_center as lc  # noqa: E402  (needs the path above)
import locator  # noqa: E402
from generate_images import image_path  # noqa: E402

FPS = 30
FRAME_W, FRAME_H = 1920, 1080
# How far the slow push travels over a generated shot. 6% is enough to feel
# alive and small enough that fine detail does not visibly crawl.
ZOOM = 1.06
# The corner map: width in pixels and its margin from the top-right corner.
MINI_W, MINI_MARGIN = 460, 28


def work_dir(script_path: str) -> Path:
    """Where this video's artefacts live: the project holding the script.

    `projects/x/scripts/y.json` writes to `projects/x/`.
    """
    return Path(script_path).resolve().parents[1]


def shot_starts(doc: dict) -> list[int]:
    """Index of every line that starts a new picture.

    A line with an ``image`` or a ``map`` opens a shot; the lines after it with
    neither are spoken over the same picture. A script with an image on every
    line (the older format) is simply one shot per line.
    """
    starts = [i for i, line in enumerate(doc["lines"]) if line.get("image") or line.get("map")]
    if not starts or starts[0] != 0:
        raise SystemExit("the first line must carry an image or a map")
    return starts


def region(doc: dict, name: str) -> dict:
    try:
        return doc["map"]["regions"][name]
    except KeyError:
        raise SystemExit(f"no map region {name!r} in the script's map.regions") from None


def shot_picture(doc: dict, work: Path, line: dict) -> Path:
    if line.get("map"):
        out = work / ".aive" / "map" / f"full_{line['map']}.png"
        return locator.render_full(work / doc["map"]["image"], region(doc, line["map"]), out,
                                   (FRAME_W, FRAME_H))
    p = image_path(doc, work, line["image"])
    if not p.is_file():
        raise SystemExit(f"missing {p.name} for {line['image'][:50]!r}; run generate_images.py")
    return p


def mean_db(ff: str, path: Path, seconds: float | None = None) -> float:
    """Mean level from ffmpeg's volumedetect."""
    cmd = [ff, "-hide_banner", "-nostats"]
    if seconds:
        cmd += ["-t", f"{seconds:.1f}"]
    cmd += ["-i", str(path), "-af", "volumedetect", "-f", "null", "-"]
    log = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                         errors="replace").stderr
    m = re.search(r"mean_volume: (-?[\d.]+) dB", log)
    if not m:
        raise SystemExit(f"could not measure the level of {path.name}")
    return float(m.group(1))


def windows(spans: list[tuple[float, float]]) -> str:
    return "+".join(f"between(t,{a:.3f},{b:.3f})" for a, b in spans)


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
    mp3 = work / ".aive" / "audio" / f"{doc['id']}.mp3"
    out = work / "output" / f"{doc['id']}.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    scratch = work / ".aive" / "text"
    scratch.mkdir(parents=True, exist_ok=True)

    style = lc.CentreStyle(frame_w=FRAME_W, frame_h=FRAME_H, caption_size=46)
    narration_end = spans[-1]["end"]
    card = doc.get("end_card") or {}
    card_sec = float(card.get("seconds", 0))
    total = narration_end + card_sec

    # A shot runs from its first line's start to the *next* shot's first line,
    # not to its own last line's end: the small silences between lines have to
    # be covered by something, and holding the picture already up is what an
    # editor would do. The first shot starts at 0 and the last runs to the end
    # (end card included), so the picture is exactly as long as the soundtrack
    # and the captions cannot drift against it.
    starts = shot_starts(doc)
    bounds = [0.0] + [spans[i]["start"] for i in starts[1:]] + [total]
    shots = [doc["lines"][i] for i in starts]
    pictures = [shot_picture(doc, work, s) for s in shots]

    ff = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [ff, "-y"]
    chains = []
    for i, (img, shot) in enumerate(zip(pictures, shots, strict=True)):
        dur = max(0.4, bounds[i + 1] - bounds[i])
        frames = max(1, int(round(dur * FPS)))
        # -framerate must match FPS and zoompan must take d=1. Given a
        # multi-frame input, zoompan emits `d` frames for *each* one, so the
        # first shot alone became six minutes long and -t cropped the video to
        # that single picture -- 62 s of narration over one still.
        cmd += ["-loop", "1", "-framerate", str(FPS), "-t", f"{dur:.3f}", "-i", str(img)]
        push = 0.0 if shot.get("map") else ZOOM - 1
        # The still is scaled up first and cropped back; zooming a 1920-wide
        # source directly makes fine lines shimmer.
        chains.append(
            f"[{i}:v]scale={FRAME_W * 2}:{FRAME_H * 2},"
            f"zoompan=z='1+{push:.4f}*on/{frames}':d=1"
            f":x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'"
            f":s={FRAME_W}x{FRAME_H}:fps={FPS},setsar=1[s{i}]"
        )
    n_in = len(pictures)
    chains.append("".join(f"[s{i}]" for i in range(n_in)) + f"concat=n={n_in}:v=1:a=0[bg]")

    # The corner map: one input per region used, shown over every shot that
    # names it. A single-frame input repeats its last frame under overlay.
    where: dict[str, list[tuple[float, float]]] = {}
    for i, shot in enumerate(shots):
        if shot.get("where") and not shot.get("map"):
            where.setdefault(shot["where"], []).append((bounds[i], bounds[i + 1]))
    label = "bg"
    for n, (name, times) in enumerate(where.items()):
        mini = locator.render_mini(work / doc["map"]["image"], region(doc, name),
                                   work / ".aive" / "map" / f"mini_{name}.png", MINI_W)
        cmd += ["-i", str(mini)]
        nxt = f"m{n}"
        chains.append(
            f"[{label}][{n_in + n}:v]overlay=x=W-w-{MINI_MARGIN}:y={MINI_MARGIN}"
            f":enable='{windows(times)}'[{nxt}]"
        )
        label = nxt
    n_in += len(where)

    # Captions, drawn with the shorts pipeline's own style helpers.
    line_h = int(style.caption_size * style.line_ratio)
    top = FRAME_H - 190
    draws = []
    for i, span in enumerate(spans):
        lines = lc.wrap(span["text"], size=style.caption_size, style=style)
        # Never truncate: a dropped caption line is text the viewer hears but
        # never sees. A third line is allowed and reported so it can be split.
        if len(lines) > 2:
            print(f"  NOTE caption {i + 1} wraps to {len(lines)} lines: {span['text'][:50]}")
        slots = max(2, len(lines))
        enable = f"between(t,{span['start']:.3f},{span['end']:.3f})"
        draws.append(
            lc._plate(style=style, size=style.caption_size, top=top - (slots - 2) * line_h,
                      line_h=line_h, slots=slots, enable=enable)
        )
        draws += lc._text_block(
            lines=lines, style=style, size=style.caption_size,
            colors=(style.caption_color,), top=top - (slots - 2) * line_h,
            line_h=line_h, slots=slots,
            text_dir=scratch, stem=f"{doc['id']}_{i:03d}", enable=enable,
        )

    if card_sec and card.get("lines"):
        # An end card with words: the last picture dims and they sit over it.
        # Without words the last shot is simply held -- the map is its own card.
        after = f"gte(t,{narration_end:.3f})"
        draws.append(f"drawbox=x=0:y=0:w=iw:h=ih:color=black@0.55:t=fill:enable='{after}'")
        for n, (text, size, color) in enumerate(
            zip(card["lines"], (88, 46), (style.title_colors[0], "0xFFFFFF"), strict=False)
        ):
            draws += lc._text_block(
                lines=[text], style=style, size=size, colors=(color,),
                top=FRAME_H // 2 - 120 + n * 150, line_h=int(size * style.line_ratio),
                slots=1, text_dir=scratch, stem=f"{doc['id']}_card{n}", enable=after,
            )
    chains.append(f"[{label}]" + ",".join(draws) + "[v]")

    # The narration, padded with silence so it runs as long as the picture, and
    # an optional music bed set a fixed number of dB under it -- measured, not
    # a raw gain, since a gain is not portable between tracks.
    cmd += ["-i", str(mp3)]
    voice_idx = n_in
    n_in += 1
    fmt = "aformat=sample_fmts=fltp:sample_rates=48000:channel_layouts=stereo"
    voice = f"[{voice_idx}:a]{fmt},apad=whole_dur={total:.3f}[voice]"
    if doc.get("music"):
        music = (work / doc["music"]).resolve()
        under = float(doc.get("music_under_db", 18))
        gain = mean_db(ff, mp3) - under - mean_db(ff, music, 600)
        print(f"music {music.name}: {gain:+.1f} dB, {under:.0f} dB under the voice")
        cmd += ["-stream_loop", "-1", "-i", str(music)]
        chains.append(voice)
        chains.append(
            f"[{n_in}:a]{fmt},atrim=0:{total:.3f},volume={gain:.2f}dB,"
            f"afade=t=in:d=2,afade=t=out:st={max(0.0, total - 4):.3f}:d=4[bed]"
        )
        chains.append("[voice][bed]amix=inputs=2:duration=first:normalize=0[a]")
    else:
        chains.append(voice.replace("[voice]", "[a]"))

    graph = ";".join(chains)
    script_file = work / ".aive" / f"{doc['id']}_graph.txt"
    script_file.write_text(graph, encoding="utf-8")

    cmd += [
        "-filter_complex_script", str(script_file),
        "-map", "[v]", "-map", "[a]",
        "-t", f"{total:.3f}",
        "-c:v", "libx264", "-preset", "medium", "-crf", "19", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", str(out),
    ]
    maps = sum(1 for s in shots if s.get("map"))
    print(f"{len(shots)} shots ({maps} map, {len(shots) - maps} generated), "
          f"{len(where)} corner maps, {total:.2f}s -> {out.name}")
    proc = subprocess.run(cmd, capture_output=True, text=True,
                          encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        print((proc.stderr or "")[-1500:])
        return 1
    print(f"wrote {out}  {out.stat().st_size / 1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
