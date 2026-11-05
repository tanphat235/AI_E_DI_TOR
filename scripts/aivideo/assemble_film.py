"""Assemble a paced video with an old-film look from a paced script.

The earlier ``assemble.py`` joins every still with a hard cut and prints each
whole sentence as one caption. This one does what the reference channel does:

* **Rests.** The silences come from ``narrate_vieneu.py`` (paced mode); a picture
  change happens *inside* a rest, finishing a moment before the next word.
* **Chapter cards.** A line carrying ``card`` becomes a title card, announced by
  the narrator, so the viewer knows what part comes next.
* **Phrase captions.** One short phrase at a time, appearing as the voice
  reaches it and leaving as it moves on.
* **Picture changes.** Between chapters a projector-style slide change (the
  picture scrolls up through a cream slide mount, blurring as it moves); inside
  a chapter a rotation of plain dissolves, wipes and slides. Each has its own
  small sound, mixed far under the voice.
* **Old film.** A warm grade, grain, lamp flicker, vignette, and a looping layer
  of dust, scratches, hairs and drifting embers.

Structure: every shot is rendered to its own small clip (picture change + slow
push, pull or pan), cached by content. A single final pass joins them and adds
the film look, the captions and the sound -- so changing a caption or a level
does not redraw sixty shots.

Run with the ``.venv-sdxl`` interpreter (it has Pillow, NumPy and imageio-ffmpeg):

  .\\.venv-sdxl\\Scripts\\python.exe scripts\\aivideo\\assemble_film.py ^
      projects\\<name>\\scripts\\<id>.json [--seconds 120] [--allow-missing]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import wave
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import imageio_ffmpeg
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "shorts"))
sys.path.insert(0, str(HERE))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import film_assets as fa  # noqa: E402
import layout_center as lc  # noqa: E402
from generate_images import image_path  # noqa: E402

FPS = 30
W, H = 1920, 1080
# Bump when the look of a clip changes, so cached clips are redrawn.
CLIP_VERSION = "4"

# Applied on painted shots only. Web screenshots stay in their own colours.
PAINT_GRADE = (
    "eq=contrast=1.07:saturation=0.90:gamma=0.96,"
    "colorbalance=rs=0.05:gs=0.01:bs=-0.06:rm=0.05:bm=-0.05:rh=0.04:bh=-0.05,"
    "vignette=angle=PI/5:eval=init,"
    "eq=brightness='0.010*sin(2*PI*6.5*t)+0.018*(random(1)-0.5)':eval=frame,"
    "noise=c0s=8:c0f=t+u:c1s=3:c1f=t+u:c2s=3:c2f=t+u"
)

# Seconds a picture change takes. The slide change is a bigger gesture.
T_PROJ = 0.80
T_PLAIN = 0.55
# Between parts of the video: slower, so it reads as a change of chapter.
T_SECTION = 1.30
# How long the part's name stays up, counted from the first spoken word.
HINT_AFTER_SPEECH = 2.2
HINT_FADE = 0.9
# A change ends this long before the next word, and starts no earlier than this
# long after the last one.
LEAD_BEFORE_SPEECH = 0.30
TAIL_AFTER_SPEECH = 0.08
MIN_TRANSITION = 0.25
# The mount strip that scrolls between two pictures.
BAND_H = 340
# Plain changes, in rotation. Never the same one twice running.
PLAIN = ["fade", "wipeleft", "dissolve", "slideright", "fadeblack", "wiperight",
         "smoothleft", "slideleft", "zoomin", "smoothright"]
# How much of the way through a picture change the blur peaks (sin shaped).
BLUR_SIGMA_V = 26

# Slow camera moves: z (zoom) and the crop's centre fraction fx, fy, start -> end.
MOTIONS = [
    dict(z0=1.00, z1=1.09, fx0=0.5, fx1=0.5, fy0=0.5, fy1=0.5),    # push in
    dict(z0=1.08, z1=1.08, fx0=0.0, fx1=1.0, fy0=0.5, fy1=0.5),    # pan right
    dict(z0=1.09, z1=1.00, fx0=0.5, fx1=0.5, fy0=0.5, fy1=0.5),    # pull out
    dict(z0=1.08, z1=1.08, fx0=1.0, fx1=0.0, fy0=0.5, fy1=0.5),    # pan left
    dict(z0=1.00, z1=1.10, fx0=0.5, fx1=0.30, fy0=0.5, fy1=0.40),  # push to upper left
    dict(z0=1.00, z1=1.10, fx0=0.5, fx1=0.70, fy0=0.5, fy1=0.45),  # push to upper right
]
CARD_MOTION = dict(z0=1.00, z1=1.04, fx0=0.5, fx1=0.5, fy0=0.5, fy1=0.5)


# ---------------------------------------------------------------------------
# Timeline
# ---------------------------------------------------------------------------


def work_dir(script_path: str) -> Path:
    return Path(script_path).resolve().parents[1]


def shot_starts(doc: dict) -> list[int]:
    """Index of every line that opens a shot: a picture or a chapter card."""
    starts = [i for i, ln in enumerate(doc["lines"])
              if ln.get("image") or ln.get("card") or ln.get("file")]
    if not starts or starts[0] != 0:
        raise SystemExit("the first line must carry an image or a file")
    return starts


def build_shots(doc: dict, timing: dict, total: float) -> list[dict]:
    """One dict per shot: kind, picture source, boundaries and the change into it."""
    spans = timing["lines"]
    starts = shot_starts(doc)
    shots: list[dict] = []
    plain_i = 0
    last_plain = ""
    for k, li in enumerate(starts):
        line = doc["lines"][li]
        shot: dict = {
            "line": li,
            "card": line.get("card"),
            "image": line.get("image"),
            "file": line.get("file"),
            "section": line.get("section"),
            "speech_start": spans[li]["start"],
        }
        if k == 0:
            shot.update(b=0.0, trans=None, T=0.0)
        else:
            prev = doc["lines"][starts[k - 1]]
            after_card = bool(prev.get("card"))
            if line.get("card") or after_card:
                trans, T = "proj", T_PROJ
            elif line.get("section"):
                # A new part: one slow dissolve, in the rest the narration leaves.
                trans, T = "fade", T_SECTION
            else:
                trans = PLAIN[plain_i % len(PLAIN)]
                if trans == last_plain:
                    plain_i += 1
                    trans = PLAIN[plain_i % len(PLAIN)]
                plain_i += 1
                last_plain, T = trans, T_PLAIN
            prev_end = spans[li - 1]["end"]
            b = max(prev_end + TAIL_AFTER_SPEECH, shot["speech_start"] - T - LEAD_BEFORE_SPEECH)
            T = min(T, shot["speech_start"] - TAIL_AFTER_SPEECH - b)
            if T < MIN_TRANSITION:
                T = MIN_TRANSITION
                b = max(prev_end, shot["speech_start"] - T - 0.02)
            shot.update(b=b, trans=trans, T=T)
        shots.append(shot)
    for shot in shots:
        shot["fb"] = int(round(shot["b"] * FPS))
    for k, shot in enumerate(shots):
        shot["fb_end"] = shots[k + 1]["fb"] if k + 1 < len(shots) else int(round(total * FPS))
    return shots


def assign_motion(shots: list[dict]) -> None:
    """A camera move per shot, never the same one twice in a row."""
    prev = -1
    for k, shot in enumerate(shots):
        if shot["card"] or shot["file"]:
            shot["motion"] = CARD_MOTION   # a card or a photograph: barely moves
            continue
        m = (k * 5 + 1) % len(MOTIONS)
        if m == prev:
            m = (m + 1) % len(MOTIONS)
        prev = m
        shot["motion"] = MOTIONS[m]


def state(motion: dict, p: float) -> tuple[float, float, float]:
    """Zoom and crop centre at progress ``p`` (0..1) through the move."""
    z = motion["z0"] + (motion["z1"] - motion["z0"]) * p
    fx = motion["fx0"] + (motion["fx1"] - motion["fx0"]) * p
    fy = motion["fy0"] + (motion["fy1"] - motion["fy0"]) * p
    return z, fx, fy


# ---------------------------------------------------------------------------
# One clip per shot
# ---------------------------------------------------------------------------


def crop_state(z: float, fx: float, fy: float) -> str:
    """A still frame of the camera move at one instant."""
    return (
        f"crop=w=iw/{z:.4f}:h=ih/{z:.4f}:x=(iw-iw/{z:.4f})*{fx:.4f}"
        f":y=(ih-ih/{z:.4f})*{fy:.4f},scale={W}:{H}:flags=lanczos,setsar=1,"
        f"fps={FPS},format=yuv420p"
    )


def kenburns(motion: dict, frames: int) -> str:
    """The slow move itself. The picture is enlarged first: zooming a source at its
    own size rounds the crop to whole pixels and the fine lines crawl."""
    m = motion
    n = max(1, frames)
    return (
        f"scale=2880:1620:flags=lanczos,"
        f"zoompan=z='{m['z0']}+({m['z1']}-{m['z0']})*on/{n}'"
        f":x='(iw-iw/zoom)*({m['fx0']}+({m['fx1']}-{m['fx0']})*on/{n})'"
        f":y='(ih-ih/zoom)*({m['fy0']}+({m['fy1']}-{m['fy0']})*on/{n})'"
        f":d=1:s={W}x{H}:fps={FPS},setsar=1,format=yuv420p"
    )


def clip_graph(shot: dict, prev: dict | None, tf: int, kf: int) -> tuple[str, int]:
    """Filter graph for one shot's clip, and how many inputs it reads.

    Input 0 is the previous picture, 1 the new one (both for the change), 2 the
    new one again for the move, and, for the slide change, 3 the mount strip.
    """
    if prev is None or tf <= 0:
        graph = f"[0:v]{kenburns(shot['motion'], kf)}[v]"
        if not shot.get("file"):
            graph += f";[v]{PAINT_GRADE}[v]"
        return graph, 1
    p_state = crop_state(*state(prev["motion"], 1.0))
    n_state = crop_state(*state(shot["motion"], 0.0))
    chains = [f"[0:v]{p_state}[p]", f"[1:v]{n_state}[n]"]
    t = tf / FPS
    if shot["trans"] == "proj":
        scroll = H + BAND_H
        e = f"min(t/{t:.4f},1)"
        wgt = f"sin(PI*min(T/{t:.4f},1))"
        chains += [
            "[p][3:v][n]vstack=inputs=3[strip]",
            "[strip]split[s1][s2]",
            f"[s2]gblur=sigma=0.5:sigmaV={BLUR_SIGMA_V}[sb]",
            f"[s1][sb]blend=all_expr='A*(1-{wgt})+B*{wgt}'[m]",
            f"[m]crop=w={W}:h={H}:x=0:y='{scroll}*(3*pow({e},2)-2*pow({e},3))',"
            f"setsar=1,format=yuv420p[tr]",
        ]
        n_inputs = 4
    else:
        chains.append(
            f"[p][n]xfade=transition={shot['trans']}:duration={t:.4f}:offset=0,"
            f"format=yuv420p[tr]"
        )
        n_inputs = 3
    chains.append(f"[tr]trim=end_frame={tf},setpts=PTS-STARTPTS[trc]")
    chains.append(f"[2:v]{kenburns(shot['motion'], kf)},setpts=PTS-STARTPTS[kb]")
    chains.append("[trc][kb]concat=n=2:v=1:a=0[v]")
    graph = ";".join(chains)
    if not shot.get("file"):
        graph += f";[v]{PAINT_GRADE}[v]"
    return graph, n_inputs


def clip_key(shot: dict, prev: dict | None, pic: Path, prev_pic: Path | None,
             band: Path, frames: int, tf: int) -> str:
    def sig(p: Path | None) -> str:
        return f"{p.name}:{p.stat().st_size}" if p and p.is_file() else "-"

    blob = json.dumps(
        [CLIP_VERSION, shot["trans"], shot["motion"], prev["motion"] if prev else None,
         frames, tf, sig(pic), sig(prev_pic), sig(band) if shot["trans"] == "proj" else "-"],
        sort_keys=True,
    )
    return hashlib.sha1(blob.encode()).hexdigest()[:14]


def render_clip(ff: str, shot: dict, prev: dict | None, pic: Path, prev_pic: Path | None,
                band: Path, clips: Path, workdir: Path, force: bool) -> Path:
    frames = shot["fb_end"] - shot["fb"]
    tf = 0
    if prev is not None and shot["trans"]:
        tf = min(int(round(shot["T"] * FPS)), max(1, frames - 1))
    kf = max(1, frames - tf)
    out = clips / f"{clip_key(shot, prev, pic, prev_pic, band, frames, tf)}.mp4"
    if out.is_file() and out.stat().st_size > 1000 and not force:
        return out

    graph, n_inputs = clip_graph(shot, prev, tf, kf)
    graph += f";[v]tpad=stop_mode=clone:stop=30[vout]"
    script = workdir / ".aive" / "film" / f"{out.stem}.graph.txt"
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text(graph, encoding="utf-8")

    cmd = [ff, "-y", "-loglevel", "error"]

    def still(path: Path, seconds: float) -> list[str]:
        return ["-loop", "1", "-framerate", str(FPS), "-t", f"{seconds:.3f}", "-i", str(path)]

    if n_inputs == 1:
        cmd += still(pic, frames / FPS + 0.5)
    else:
        cmd += still(prev_pic, tf / FPS + 0.2)
        cmd += still(pic, tf / FPS + 0.2)
        cmd += still(pic, kf / FPS + 0.5)
        if n_inputs == 4:
            cmd += still(band, tf / FPS + 0.2)
    cmd += [
        "-filter_complex_script", str(script), "-map", "[vout]", "-frames:v", str(frames),
        "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "14",
        "-pix_fmt", "yuv420p", "-r", str(FPS), str(out),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        out.unlink(missing_ok=True)
        raise SystemExit(f"clip for line {shot['line']} failed:\n{(proc.stderr or '')[-1200:]}")
    return out


# ---------------------------------------------------------------------------
# Audio
# ---------------------------------------------------------------------------


def mean_db(ff: str, path: Path, seconds: float | None = None) -> float:
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


def build_sfx_track(shots: list[dict], total: float, voice_mean: float,
                    under_db: float, film_dir: Path) -> Path:
    """One WAV holding every picture-change sound at its place on the timeline.

    Levels are set against the *voice*, not as raw gain: the peak of the main
    click sits ``under_db`` below the voice's mean level, and the quieter kinds
    sit a few dB below that. Small by construction.
    """
    # The extra attenuation (dB) of each kind relative to the clack.
    extra = {"clack": 0.0, "swish": 4.0, "soft": 7.0}
    cache: dict[str, np.ndarray] = {}
    track = np.zeros(int(total * fa.SR) + fa.SR, dtype=np.float64)
    for shot in shots:
        if not shot["trans"]:
            continue
        kind = fa.sfx_for(shot["trans"])
        if kind not in cache:
            cache[kind] = fa.SFX_MAKERS[kind]()
        peak_db = voice_mean - under_db - extra[kind]
        fa._place(track, cache[kind], shot["b"], 10 ** (peak_db / 20))
    return fa.write_wav(film_dir / "sfx_track.wav", track[: int(total * fa.SR)])


# ---------------------------------------------------------------------------
# Captions
# ---------------------------------------------------------------------------


def caption_filters(phrases: list[dict], style: lc.CentreStyle, scratch: Path, stem: str,
                    total: float, narration_end: float) -> list[str]:
    """One drawtext per caption line. Each fades in and out; no plate behind it.

    Captions are phrases a few words long, so a heavy bar across the picture is
    no longer needed -- a thick outline and a soft shadow carry the reading.
    """
    size = style.caption_size
    line_h = int(size * style.line_ratio)
    base = H - 150
    draws: list[str] = []
    shown = [p for p in phrases if not p.get("card")]
    for n, ph in enumerate(shown):
        start = max(0.0, ph["start"] - 0.03)
        end = ph["end"] + 0.15
        if n + 1 < len(shown):
            end = min(end, shown[n + 1]["start"] - 0.02)
        end = min(end, narration_end + 0.4)
        if start >= total or end <= start:
            continue
        lines = lc.wrap(ph["text"], size=size, style=style)[:2]
        fade = min(0.14, (end - start) / 3)
        alpha = f"min(1,(t-{start:.3f})/{fade:.3f})*min(1,({end:.3f}-t)/{fade:.3f})"
        top = base - (len(lines) - 1) * line_h
        for i, line in enumerate(lines):
            f = scratch / f"{stem}_{n:04d}_{i}.txt"
            f.write_text(line, encoding="utf-8")
            opts = [
                f"fontfile='{lc.drawtext_path(style.font)}'",
                f"textfile='{lc.drawtext_path(f)}'",
                f"fontsize={size}",
                f"fontcolor={style.caption_color}",
                f"borderw={max(4, size // 9)}",
                "bordercolor=black",
                "shadowx=3", "shadowy=3", "shadowcolor=black@0.75",
                "x=(w-text_w)/2",
                f"y={top + i * line_h}",
                f"alpha='{alpha}'",
                f"enable='between(t,{start:.3f},{end:.3f})'",
            ]
            draws.append("drawtext=" + ":".join(opts))
    return draws


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def placeholder(path: Path) -> Path:
    """A dark ochre frame standing in for a picture that is not drawn yet."""
    if not path.is_file():
        from PIL import Image

        path.parent.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(3)
        base = np.zeros((768, 1360, 3), dtype=np.float32)
        base[:] = (96, 64, 30)
        base += rng.normal(0, 10, base.shape)
        Image.fromarray(np.clip(base, 0, 255).astype("uint8")).save(path)
    return path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("script")
    ap.add_argument("--seconds", type=float, default=0.0,
                    help="render only the first N seconds, for a quick look")
    ap.add_argument("--allow-missing", action="store_true",
                    help="stand in a plain frame for pictures not generated yet")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--force", action="store_true", help="redraw cached clips")
    args = ap.parse_args()

    doc = json.loads(Path(args.script).read_text(encoding="utf-8-sig"))
    work = work_dir(args.script)
    film_dir = work / ".aive" / "film"
    film_dir.mkdir(parents=True, exist_ok=True)
    timing = json.loads(
        (work / ".aive" / "audio" / f"{doc['id']}.json").read_text(encoding="utf-8")
    )
    if "phrases" not in timing:
        raise SystemExit("the narration has no phrase timings; run narrate_vieneu.py on a "
                         "script with a 'pauses' key")
    mp3 = work / ".aive" / "audio" / f"{doc['id']}.mp3"
    ff = imageio_ffmpeg.get_ffmpeg_exe()

    narration_end = timing["lines"][-1]["end"]
    card = doc.get("end_card") or {}
    card_sec = float(card.get("seconds", 0))
    card_at = narration_end + 0.6
    full_total = card_at + card_sec
    total = min(full_total, args.seconds) if args.seconds else full_total

    shots = build_shots(doc, timing, full_total)
    assign_motion(shots)
    if args.seconds:
        shots = [s for s in shots if s["b"] < total]
        shots[-1]["fb_end"] = int(round(total * FPS))
    print(f"{len(shots)} shots, {total:.1f}s  "
          f"({sum(1 for s in shots if s['card'])} chapter cards)")

    # Pictures --------------------------------------------------------------
    missing = []

    def picture(scene: str) -> Path:
        p = image_path(doc, work, scene)
        if p.is_file():
            return p
        missing.append(scene[:50])
        return placeholder(film_dir / "placeholder.png")

    bg = None
    if doc.get("card_bg"):
        bg = picture(doc["card_bg"])
    pics: list[Path] = []
    card_n = 0
    for shot in shots:
        if shot["card"]:
            card_n += 1
            png = fa.render_card(bg, shot["card"], film_dir / "cards" / f"{doc['id']}_{card_n}.png")
            pics.append(png)
        elif shot["file"]:
            src = (work / shot["file"]).resolve()
            if not src.is_file():
                raise SystemExit(f"missing supplied photo {src}")
            pics.append(fa.render_photo(src, film_dir / "photos" / f"{src.stem}.png"))
        else:
            pics.append(picture(shot["image"]))
    if missing and not args.allow_missing:
        print(f"{len(set(missing))} pictures are not generated yet, e.g. {missing[0]!r}; "
              f"run generate_images.py (or pass --allow-missing)")
        return 1

    band = fa.make_band(film_dir / "band.png", BAND_H)
    overlay = fa.make_overlay(ff, film_dir / "dust_overlay.mov")

    # Clips -----------------------------------------------------------------
    clips_dir = film_dir / "clips"
    clips_dir.mkdir(exist_ok=True)

    def job(k: int) -> Path:
        prev = shots[k - 1] if k else None
        prev_pic = pics[k - 1] if k else None
        return render_clip(ff, shots[k], prev, pics[k], prev_pic, band, clips_dir, work,
                           args.force)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        clip_paths = []
        for k, p in enumerate(pool.map(job, range(len(shots)))):
            clip_paths.append(p)
            print(f"  clip {k + 1:3d}/{len(shots)}  {shots[k]['trans'] or 'start':9s} "
                  f"{(shots[k]['fb_end'] - shots[k]['fb']) / FPS:6.1f}s", flush=True)
    listing = film_dir / "clips.txt"
    listing.write_text("".join(f"file '{p.as_posix()}'\n" for p in clip_paths), encoding="utf-8")

    # Sound -----------------------------------------------------------------
    voice_mean = mean_db(ff, mp3)
    sfx_wav = build_sfx_track(shots, total, voice_mean, float(doc.get("sfx_under_db", 6)),
                              film_dir)
    fmt = "aformat=sample_fmts=fltp:sample_rates=48000:channel_layouts=stereo"

    # Picture and captions --------------------------------------------------
    style = lc.CentreStyle(frame_w=W, frame_h=H, caption_size=56,
                         caption_color="0xF6EBCB")
    scratch = work / ".aive" / "text"
    scratch.mkdir(parents=True, exist_ok=True)
    for old in scratch.glob(f"{doc['id']}_f*"):
        old.unlink()
    # Colour grade lives on painted clips. This pass only lays dust and type.
    chains = [
        "[0:v]format=yuv420p[g]",
        "[g][1:v]overlay=0:0:format=auto[o]",
    ]
    draws = caption_filters(timing["phrases"], style, scratch, f"{doc['id']}_f", total,
                            narration_end)
    if card_sec and card.get("lines"):
        after = f"gte(t,{card_at:.3f})"
        draws.append(f"drawbox=x=0:y=0:w=iw:h=ih:color=black@0.55:t=fill:enable='{after}'")
        for n, (text, size, color) in enumerate(
            zip(card["lines"], (88, 46), (style.title_colors[0], "0xFFFFFF"), strict=False)
        ):
            draws += lc._text_block(
                lines=[text], style=style, size=size, colors=(color,),
                top=H // 2 - 120 + n * 150, line_h=int(size * style.line_ratio),
                slots=1, text_dir=scratch, stem=f"{doc['id']}_fcard{n}", enable=after,
            )
    draws.append(f"fade=t=in:st=0:d=1.2")
    draws.append(f"fade=t=out:st={max(0.0, total - 1.6):.3f}:d=1.6")
    chains.append("[o]" + ",".join(draws) + "[v]")

    # Audio graph: voice, music bed, and the small picture-change sounds.
    chains.append(f"[2:a]{fmt},apad=whole_dur={total:.3f},atrim=0:{total:.3f}[voice]")
    n_audio = 1
    mix_inputs = "[voice]"
    inputs = ["-f", "concat", "-safe", "0", "-i", str(listing),
              "-stream_loop", "-1", "-i", str(overlay),
              "-i", str(mp3)]
    idx = 3
    if doc.get("music"):
        music = (work / doc["music"]).resolve()
        under = float(doc.get("music_under_db", 18))
        gain = voice_mean - under - mean_db(ff, music, 600)
        print(f"music {music.name}: {gain:+.1f} dB, {under:.0f} dB under the voice")
        inputs += ["-stream_loop", "-1", "-i", str(music)]
        chains.append(
            f"[{idx}:a]{fmt},atrim=0:{total:.3f},volume={gain:.2f}dB,"
            f"afade=t=in:d=2,afade=t=out:st={max(0.0, total - 4):.3f}:d=4[bed]"
        )
        mix_inputs += "[bed]"
        n_audio += 1
        idx += 1
    inputs += ["-i", str(sfx_wav)]
    chains.append(f"[{idx}:a]{fmt},apad=whole_dur={total:.3f},atrim=0:{total:.3f}[sfx]")
    mix_inputs += "[sfx]"
    n_audio += 1
    chains.append(f"{mix_inputs}amix=inputs={n_audio}:duration=first:normalize=0,"
                  f"alimiter=limit=0.95[a]")

    # The name of each new part, laid over the picture that opens it. These come
    # last among the inputs so the indexes above stay put; the grade is applied
    # after them so the grain and flicker cover the lettering too.
    src_label = "0:v"
    hint_chains: list[str] = []
    hint_input = idx + 1
    for j, shot in enumerate(s for s in shots if s["section"]):
        start = shot["b"] + 0.35
        end = shot["speech_start"] + HINT_AFTER_SPEECH
        if start >= total:
            continue
        png, hx, hy = fa.render_section_hint(
            shot["section"], film_dir / "hints" / f"{doc['id']}_{j}.png")
        # A short finite stream, started at ``start`` by shifting its timestamps.
        span = end + HINT_FADE - start
        inputs += ["-loop", "1", "-framerate", str(FPS), "-t", f"{span:.3f}", "-i", str(png)]
        hint_chains.append(
            f"[{hint_input}:v]format=rgba,fade=t=in:st=0:d={HINT_FADE}:alpha=1,"
            f"fade=t=out:st={span - HINT_FADE:.3f}:d={HINT_FADE}:alpha=1,"
            f"setpts=PTS+{start:.3f}/TB[hint{j}]"
        )
        hint_chains.append(
            f"[{src_label}][hint{j}]overlay={hx}:{hy}:format=auto:eof_action=pass[vh{j}]"
        )
        src_label = f"vh{j}"
        hint_input += 1
    chains[0] = f"[{src_label}]format=yuv420p[g]"
    chains = hint_chains + chains

    graph = film_dir / f"{doc['id']}_final_graph.txt"
    graph.write_text(";".join(chains), encoding="utf-8")
    out_dir = work / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_first{int(args.seconds)}s" if args.seconds else ""
    out = out_dir / f"{doc['id']}{suffix}.mp4"
    cmd = [
        ff, "-y", "-loglevel", "error", "-stats", *inputs,
        "-filter_complex_script", str(graph),
        "-map", "[v]", "-map", "[a]", "-t", f"{total:.3f}",
        "-c:v", "libx264", "-preset", "faster", "-crf", "21",
        "-maxrate", "6M", "-bufsize", "12M", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", str(out),
    ]
    print(f"final pass -> {out.name}", flush=True)
    proc = subprocess.run(cmd, text=True, encoding="utf-8", errors="replace",
                          stderr=subprocess.PIPE)
    if proc.returncode != 0:
        print((proc.stderr or "")[-1800:])
        return 1
    print(f"wrote {out}  {out.stat().st_size / 1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
