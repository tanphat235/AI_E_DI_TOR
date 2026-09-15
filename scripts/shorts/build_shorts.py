"""Build vertical split-screen shorts from a talk video + B-roll folder.

Two layouts. "half" is the original: top = talk, bottom = B-roll. "center"
puts the talk in the middle of a full-frame scene background, with a title
above it and a caption of the speech below it, styled to match
projects/tui-tu-tui-nhan/shorts/clip_001.mp4 -- see layout_center.py, which
owns that geometry and the measurements behind it.
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
import random
import re
import subprocess
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import layout_center as lc  # noqa: E402  (needs the path above)


def _ffmpeg() -> Path:
    try:
        import imageio_ffmpeg

        return Path(imageio_ffmpeg.get_ffmpeg_exe())
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"ffmpeg not found via imageio-ffmpeg: {exc}") from exc


def _music_offset_for(index: int, settings: "Settings", music_dur: float) -> float:
    """Where in the track this clip's bed starts."""
    if music_dur <= 0:
        return 0.0
    if settings.music_offset >= 0:
        return settings.music_offset % music_dur
    idx = index + settings.broll_start_index
    return ((idx * 0.6180339887498949) % 1.0) * music_dur


def _video_size(path: Path) -> tuple[int, int]:
    """Coded width and height. PyAV, because the vendored wheel has no ffprobe."""
    import av

    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        return int(stream.codec_context.width), int(stream.codec_context.height)


def _talk_height(source: Path, src_crop: str, out_w: int) -> int:
    """Height the talk occupies at full width, from its own aspect ratio.

    Taken from --src-crop when given, because that crop is what will actually
    be shown; falling back to the coded size would letterbox or stretch a clip
    whose furniture has been cropped away. Rounded to an even number, which
    yuv420p requires.
    """
    if src_crop:
        parts = src_crop.split(":")
        w, h = int(parts[0]), int(parts[1])
    else:
        w, h = _video_size(source)
    return max(2, int(round(out_w * h / w)) // 2 * 2)


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
    # Shuffle the B-roll order with this seed; 0 keeps them sorted by name.
    broll_seed: int = 0
    # How per-clip strip offsets are chosen. "stride" steps by broll_stride and
    # repeats whenever the strip length is near a multiple of it -- a 329.5s strip
    # with stride 47 gave only 8 distinct offsets. "golden" walks by the golden
    # ratio, which spreads any number of clips without tuning the stride.
    broll_spread: str = "stride"
    # Continue the offset sequence from here, so a second render pass over the
    # same project does not restart at 0 and duplicate the first pass.
    broll_start_index: int = 0
    # "half" stacks talk over B-roll. "center" frames the talk inside a
    # full-frame scene background with title above and caption below.
    layout: str = "half"
    # Height of the talk inside the centre layout; 0 derives it from the
    # source's own aspect after --src-crop, so nothing is stretched.
    talk_h: int = 0
    # Draw a caption of the speech under the talk (centre layout only). Needs
    # .aive/transcript.json, which transcribe.py writes.
    captions: bool = True
    centre_font: Path = Path(r"C:/Windows/Fonts/seguibl.ttf")
    title_size: int = 78
    title_max_lines: int = 2
    caption_size: int = 54
    caption_max_lines: int = 2
    # Clean up the talk's own audio: rumble out, low-mid mud down, consonant
    # band up, gentle levelling. Off by default so finished projects are
    # unaffected.
    voice_clarity: bool = False
    # Resample ratio for the talk's pitch, tempo restored afterwards so the
    # duration does not change. 0.95 is five percent deeper.
    voice_pitch: float = 1.0
    # Hold the bed at one level and move it out of the voice's band. Both are
    # what the approved reference does; both default off.
    music_compress: bool = False
    music_dip_hz: float = 450.0
    music_dip_db: float = 0.0
    # "flattest" pins each clip's bed to the steadiest stretch of the track of
    # that clip's own length, instead of spreading offsets across it. Costs one
    # decode of the whole track, cached.
    music_window: str = "fixed"
    # Cross-fade between the spans of a multi-part clip. "cut" is a hard join,
    # which is right inside one continuous answer and wrong between passages
    # taken from different places in the talk.
    part_transition: str = "cut"
    part_transition_sec: float = 0.5
    # Ceiling for the finished mix, in dBFS. Zero disables the limiter, which
    # is what the finished projects were rendered without. A hot source plus
    # the clarity chain's makeup and the bed reached -0.00 dBFS on one clip.
    peak_dbfs: float = 0.0
    # Framing of the talk band. "auto" measures the speaker's face and zooms
    # until it fills talk_face_frac of the band, recentring on it; a number is
    # a fixed zoom; 1.0 leaves the framing alone, which is what the finished
    # projects were rendered with.
    talk_zoom: str = "1.0"
    talk_zoom_max: float = 1.5
    talk_face_frac: float = 0.32
    talk_face_y: float = 0.36
    # Playback speed. 0.75 slows everything to three quarters; the picture is
    # stretched with setpts and the speech with a second atempo, while the bed
    # keeps its own tempo and is simply given the longer output to cover.
    speed: float = 1.0
    # Aim the bed this many dB under the measured speech, instead of asking
    # for a raw --music-db. Zero keeps the raw-gain behaviour the finished
    # projects were tuned with.
    music_under_db: float = 0.0
    # "top" mirrors the talk only, "all" the finished frame, "none" neither.
    flip: str = "none"
    # Optional instrumental bed, ducked under the speech.
    music: Path | None = None
    music_db: float = -26.0
    music_fade: float = 1.5
    # Pin the bed to one chosen stretch of the track. Negative = spread each
    # clip's offset across the track automatically.
    music_offset: float = -1.0
    # Thumbnail headline. The text itself is written per clip into the segment
    # as "thumb_text"; these control how it is drawn.
    thumb_font: Path = Path(r"C:/Windows/Fonts/arialbd.ttf")
    thumb_color: str = "0xFFD24A"
    thumb_size: int = 84
    thumb_lines: int = 3
    thumb_margin: int = 150
    # Sidechain ducking makes the bed rise and fall with the speech, which is
    # audible as pumping. Off holds one constant level for the whole clip.
    # Threshold is linear amplitude: speech sits near 0.08 and room tone near
    # 0.003, so a value between the two engages under speech only.
    music_duck: bool = True
    music_duck_threshold: float = 0.01
    music_duck_ratio: float = 12.0
    music_duck_release: float = 300.0


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
        if (
            dur >= max_sec
            or (dur >= target_sec and gap >= pause_cut)
            or (dur >= min_sec and gap >= 2.5)
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
    """Concatenate every B-roll scene into one strip of the given height.

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
        "-filter_complex",
        filt,
        "-map",
        "[v]",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "22",
        "-pix_fmt",
        "yuv420p",
        "-an",
        "-movflags",
        "+faststart",
        str(bed),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
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
    flip: str = "none",
    layout: str = "half",
    geom: lc.Geometry | None = None,
    style: lc.CentreStyle | None = None,
    title_lines: list[str] | None = None,
    cues: list[lc.Cue] | None = None,
    scratch_dir: Path | None = None,
    zoom_filter: str = "",
    speed: float = 1.0,
    transition: str = "cut",
    transition_sec: float = 0.5,
    peak_dbfs: float = 0.0,
    voice_clarity: bool = False,
    voice_pitch: float = 1.0,
    music_compress: bool = False,
    music_dip_hz: float = 450.0,
    music_dip_db: float = 0.0,
    music: Path | None = None,
    music_db: float = -26.0,
    music_offset: float = 0.0,
    music_fade: float = 1.5,
    duck: bool = True,
    duck_threshold: float = 0.01,
    duck_ratio: float = 12.0,
    duck_release: float = 300.0,
) -> Path:
    shorts_dir.mkdir(parents=True, exist_ok=True)
    out = shorts_dir / f"{seg['id']}.mp4"
    half_h = out_h // 2
    # A segment is either one span, or a "parts" list spliced in the order given
    # -- which need not be chronological.
    parts = seg.get("parts") or [{"start": seg["start"], "end": seg["end"]}]
    # Every junction removes the cross-fade from the timeline, so the finished
    # clip is shorter than the sum of its spans. The B-roll length, the music
    # fade and the caption times all read this, not the raw sum.
    overlap = lc.overlap_for(transition, transition_sec) if len(parts) > 1 else 0.0
    dur = sum(p["end"] - p["start"] for p in parts) - overlap * (len(parts) - 1)
    # The talk is stretched, so the finished clip runs longer than its spans.
    out_dur = dur / speed if speed else dur
    pre = f"crop={src_crop}," if src_crop else ""

    # "top" mirrors only the talk, so a speaker facing left now faces right;
    # "all" mirrors the finished frame, B-roll included.
    top_flip = "hflip," if flip == "top" else ""
    out_flip = ",hflip" if flip == "all" else ""

    if layout == "center":
        if geom is None or style is None:
            raise ValueError("centre layout needs geom and style")
        video_graph, video_label = lc.video_graph(
            parts=parts,
            broll_index=len(parts),
            src_crop=src_crop,
            flip=flip,
            geom=geom,
            style=style,
            title_lines=title_lines or [],
            cues=cues or [],
            text_dir=(scratch_dir or shorts_dir / ".scratch") / "text",
            stem=seg["id"],
            zoom=zoom_filter,
            transition=transition,
            transition_sec=transition_sec,
        )
    else:
        video_graph = video_label = ""

    top_chain = "".join(
        f"[{i}:v]{pre}scale={out_w}:{half_h}:force_original_aspect_ratio=increase,"
        f"crop={out_w}:{half_h},{top_flip}fps=30,setsar=1[p{i}];"
        for i in range(len(parts))
    )
    if len(parts) == 1:
        joined = "[p0]null[top];[0:a]anull[speech];"
    else:
        pairs = "".join(f"[p{i}][{i}:a]" for i in range(len(parts)))
        joined = f"{pairs}concat=n={len(parts)}:v=1:a=1[top][speech];"
    bed_idx = len(parts)

    # The talk's own audio is treated before anything is mixed onto it, so
    # the bed's measured level is a level against the treated voice.
    voice = lc.voice_chain(clarity=voice_clarity, pitch=voice_pitch, speed=speed)
    if voice:
        pre_audio = f"[speech]{voice}[speechx];"
        speech_label = "[speechx]"
    else:
        pre_audio = ""
        speech_label = "[speech]"

    if music is None:
        audio_chain = pre_audio
        audio_map = speech_label
    else:
        # The music is mastered far hotter than the talk (-9.6 LUFS against
        # -21.7 on this source), so it is cut right down and then ducked
        # against the speech itself -- otherwise it buries the teaching.
        fade_out = max(0.0, out_dur - music_fade)
        fmt = "aformat=sample_fmts=fltp:sample_rates=48000:channel_layouts=stereo"
        if music_compress or music_dip_db:
            shaped = lc.music_chain(
                gain_db=music_db,
                fade=music_fade,
                duration=out_dur,
                compress=music_compress,
                dip_hz=music_dip_hz,
                dip_db=music_dip_db,
            )
            music_chain = f"[{bed_idx + 1}:a]{shaped}[mus];"
        else:
            music_chain = (
                f"[{bed_idx + 1}:a]{fmt},volume={music_db}dB,"
                f"afade=t=in:st=0:d={music_fade},"
                f"afade=t=out:st={fade_out:.3f}:d={music_fade}[mus];"
            )
        if duck:
            audio_chain = (
                f"{pre_audio}{speech_label}{fmt},asplit=2[sp][sc];"
                f"{music_chain}"
                f"[mus][sc]sidechaincompress="
                f"threshold={duck_threshold}:ratio={duck_ratio}:"
                f"attack=5:release={duck_release}[musd];"
                f"[sp][musd]amix=inputs=2:duration=first:normalize=0[aout];"
            )
        else:
            # One constant level. Only the entrance and exit fades change it.
            audio_chain = (
                f"{pre_audio}{speech_label}{fmt}[sp];"
                f"{music_chain}"
                f"[sp][mus]amix=inputs=2:duration=first:normalize=0[aout];"
            )
        audio_map = "[aout]"

    if peak_dbfs:
        # Last in the chain, after the bed is mixed in: the ceiling has to
        # govern what actually reaches the encoder, not one contributor to it.
        # level=disabled stops alimiter normalising the input up to the ceiling
        # as well as down, which would undo the measured bed-to-voice ratio.
        limit = 10.0 ** (peak_dbfs / 20.0)
        src = audio_map
        audio_chain += f"{src}alimiter=limit={limit:.4f}:level=disabled[alim];"
        audio_map = "[alim]"

    if abs(speed - 1.0) > 1e-6 and layout == "center":
        # After the captions are drawn, not before: drawtext's enable times are
        # on the unstretched clock, and stretching afterwards carries the words
        # and the speech along together.
        video_graph += f";{video_label}setpts=PTS/{speed:.6f}[vspd]"
        video_label = "[vspd]"

    if layout == "center":
        # The talk chains, the concat and the whole picture side come from
        # layout_center; only the audio is shared with the half layout.
        filt = f"{video_graph};{audio_chain.rstrip(';')}"
        map_video = video_label
    else:
        filt = (
            f"{top_chain}{joined}"
            f"[{bed_idx}:v]scale={out_w}:{half_h}:force_original_aspect_ratio=increase,"
            f"crop={out_w}:{half_h},fps=30,setsar=1[bot];"
            f"{audio_chain}"
            f"[top][bot]vstack=inputs=2{out_flip}[v]"
        )
        map_video = "[v]"

    cmd = [str(ffmpeg), "-y"]
    for p in parts:
        cmd += [
            "-ss",
            f"{p['start']:.3f}",
            "-t",
            f"{p['end'] - p['start']:.3f}",
            "-i",
            str(source),
        ]
    cmd += ["-stream_loop", "-1"]
    if bed_offset is not None:
        cmd += ["-ss", f"{bed_offset:.3f}"]
    cmd += ["-t", f"{dur:.3f}", "-i", str(broll)]
    if music is not None:
        cmd += [
            "-stream_loop",
            "-1",
            "-ss",
            f"{music_offset:.3f}",
            "-t",
            f"{out_dur:.3f}",
            "-i",
            str(music),
        ]
    # A caption per transcript line puts dozens of drawtext filters in the
    # graph, and Windows caps a command line at 32767 characters. Passing the
    # graph as a file sidesteps the limit rather than hoping it fits.
    if len(filt) > 8000:
        graphs = scratch_dir or shorts_dir / ".scratch"
        graphs.mkdir(parents=True, exist_ok=True)
        script = graphs / f"{seg['id']}_graph.txt"
        script.write_text(filt, encoding="utf-8")
        cmd += ["-filter_complex_script", str(script)]
    else:
        cmd += ["-filter_complex", filt]
    cmd += [
        "-map",
        map_video,
        "-map",
        audio_map,
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
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr[-800:] if proc.stderr else "ffmpeg failed")
    return out


def _drawtext_path(path: Path) -> str:
    """Escape a Windows path for a drawtext option value.

    drawtext splits options on ':' and treats a backslash as an escape, so a
    raw Windows path silently truncates the filter. chr(92) rather than a
    literal so the escaping is unambiguous to read.
    """
    sep = chr(92)
    return str(path).replace(sep, "/").replace(":", sep + ":")


def _wrap_headline(text: str, *, font_size: int, width_px: int, max_lines: int) -> list[str]:
    """Wrap to the drawn width, estimating Arial Bold at ~0.52em per glyph."""
    per_line = max(8, int(width_px / (font_size * 0.52)))
    lines = textwrap.wrap(" ".join(text.split()), width=per_line)
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = lines[-1].rstrip(" ,.;:") + "..."
    return lines or [""]


def make_thumbnail(
    seg: dict,
    clip: Path,
    *,
    shorts_dir: Path,
    ffmpeg: Path,
    badge: str,
    topic: str,
    font: Path,
    color: str,
    size: int,
    max_lines: int,
    margin: int,
    headline: bool = True,
) -> Path:
    """A frame from the clip with a gold headline over the lower, B-roll half.

    ``headline=False`` returns the frame alone. The centre layout already burns
    the same line into the picture, so drawing it again put the title on screen
    twice and landed the second copy across the caption.

    The headline is "thumb_text" on the segment when present -- a line written
    for the thumbnail rather than the first sentence of the transcript, which
    is what "title" holds and rarely reads as a hook.

    The text goes through a file, not an inline text= value: Vietnamese
    headlines carry commas, colons and question marks, and each of those needs
    different escaping inside a filter string.

    Impact and Arial Narrow are not options here -- both lack the Vietnamese
    u-horn glyphs and render tofu boxes for u/uu/ur.
    """
    shorts_dir.mkdir(parents=True, exist_ok=True)
    frame = shorts_dir / f"{seg['id']}_frame.jpg"
    thumb = shorts_dir / f"{seg['id']}.jpg"
    mid = max(0.8, min(seg["duration"] * 0.35, seg["duration"] - 0.5))
    subprocess.run(
        [
            str(ffmpeg), "-y", "-ss", f"{mid:.2f}", "-i", str(clip),
            "-frames:v", "1", "-q:v", "2", str(frame),
        ],
        check=True, capture_output=True, text=True, encoding="utf-8", errors="replace",
    )

    if not headline:
        thumb.write_bytes(frame.read_bytes())
        frame.unlink(missing_ok=True)
        return thumb

    text = str(seg.get("thumb_text") or seg.get("title") or seg["id"]).strip()
    if not font.is_file():
        font = Path(r"C:/Windows/Fonts/arial.ttf")
    lines = _wrap_headline(
        text, font_size=size, width_px=1080 - 2 * 60, max_lines=max_lines
    )

    # Vietnamese stacks diacritics, so it needs more leading than Latin text.
    line_h = int(size * 1.30)
    block_h = line_h * len(lines)
    top = max(0, 1920 - margin - block_h)
    pad = int(size * 0.35)

    # One drawtext per line, each centred on its own width. Passing all the
    # lines as a single multi-line textfile centres the block but left-aligns
    # the lines inside it, so a short last line hangs off to one side.
    draws = [
        f"drawbox=x=0:y={top - pad}:w=iw:h={block_h + 2 * pad}:color=black@0.55:t=fill"
    ]
    line_files: list[Path] = []
    for i, line in enumerate(lines):
        line_file = shorts_dir / f"{seg['id']}_thumb{i}.txt"
        line_file.write_text(line, encoding="utf-8")
        line_files.append(line_file)
        draws.append(
            f"drawtext=fontfile='{_drawtext_path(font)}'"
            f":textfile='{_drawtext_path(line_file)}'"
            f":fontsize={size}:fontcolor={color}"
            f":borderw={max(4, size // 14)}:bordercolor=black"
            f":x=(w-text_w)/2:y={top + i * line_h}"
        )
    proc = subprocess.run(
        [str(ffmpeg), "-y", "-i", str(frame), "-vf", ",".join(draws), "-q:v", "3", str(thumb)],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    if proc.returncode != 0 or not thumb.is_file():
        if frame.is_file():
            thumb.write_bytes(frame.read_bytes())
    for tmp in (frame, *line_files):
        tmp.unlink(missing_ok=True)
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
                f"missing transcript: {transcript_path}\nRun scripts/shorts/transcribe.py first."
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
        segments_path.write_text(json.dumps(segs, ensure_ascii=False, indent=2), encoding="utf-8")
    if settings.limit:
        segs = segs[: settings.limit]
    print(f"segments={len(segs)}")
    for s in segs[:8]:
        print(f"  {s['id']} {s['start']:.1f}-{s['end']:.1f} ({s['duration']:.1f}s) {s['title']}")

    brolls = broll_files(settings.broll_dir)
    ffmpeg = _ffmpeg()

    # The centre layout needs a full-frame scene background, a talk height, and
    # -- for captions -- the transcript, whichever list of segments was used.
    style: lc.CentreStyle | None = None
    talk_h = 0
    caption_source: list[dict] = []
    if settings.layout == "center":
        style = lc.CentreStyle(
            frame_w=settings.out_w,
            frame_h=settings.out_h,
            font=settings.centre_font,
            title_size=settings.title_size,
            title_lines=settings.title_max_lines,
            caption_size=settings.caption_size,
            caption_lines=settings.caption_max_lines,
        )
        talk_h = settings.talk_h or _talk_height(
            settings.source, settings.src_crop, settings.out_w
        )
        print(
            f"layout=center frame={settings.out_w}x{settings.out_h} "
            f"talk={settings.out_w}x{talk_h} font={style.font.name}"
        )
        if settings.captions:
            if not transcript_path.is_file():
                raise SystemExit(
                    f"captions need a transcript: {transcript_path}\n"
                    "Run scripts/shorts/transcribe.py first, or pass --no-captions."
                )
            caption_source = json.loads(transcript_path.read_text(encoding="utf-8"))["segments"]
            print(f"captions from {len(caption_source)} transcript lines")

    bed_h = settings.out_h if settings.layout == "center" else settings.out_h // 2
    bed: Path | None = None
    bed_dur = 0.0
    if settings.broll_bed:
        if settings.broll_seed:
            random.Random(settings.broll_seed).shuffle(brolls)
        # The seed is part of the name so a different order rebuilds the strip
        # instead of silently reusing the cached one.
        bed_name = (
            f"broll_bed_s{settings.broll_seed}.mp4" if settings.broll_seed else "broll_bed.mp4"
        )
        # The height is part of the name too. A full-frame strip and a
        # half-frame one are not interchangeable, and the finished projects
        # must keep reading the file they already built.
        if bed_h != settings.out_h // 2 or settings.out_w != 1080:
            bed_name = bed_name.replace(".mp4", f"_{settings.out_w}x{bed_h}.mp4")
        bed = build_bed(
            brolls,
            cache / bed_name,
            ffmpeg=ffmpeg,
            chunk=settings.broll_chunk,
            out_w=settings.out_w,
            half_h=bed_h,
        )
        bed_dur = _media_duration(bed)
        print(f"broll bed: {bed.name} {bed_dur:.1f}s from {len(brolls)} scenes")
    music_dur = 0.0
    music_db_eff = settings.music_db
    music_levels: list[float] = []
    if settings.music is not None:
        if not settings.music.is_file():
            raise SystemExit(f"music not found: {settings.music}")
        music_dur = _media_duration(settings.music)
        how = "ducked" if settings.music_duck else "constant level"
        print(f"music: {settings.music.name} {music_dur:.0f}s at {settings.music_db}dB, {how}")
        raw_db, shaped_db = lc.bed_levels_db(
            ffmpeg,
            settings.music,
            compress=settings.music_compress,
            dip_hz=settings.music_dip_hz,
            dip_db=settings.music_dip_db,
        )
        if settings.music_under_db > 0:
            # Aim the bed at the voice instead of asking for a raw gain. A gain
            # is not portable between recordings: -20 dB sat 16 dB under the
            # voice on song-tot and 20 dB under it here, because the voices are
            # at different levels. "N dB under the speech" is the thing that
            # was actually meant, so measure both and solve for the gain.
            voice = lc.voice_chain(
                clarity=settings.voice_clarity, pitch=settings.voice_pitch
            )
            spans = [(float(s["start"]), float(s["end"])) for s in segs]
            speech_db = lc.speech_level_db(ffmpeg, settings.source, spans, chain=voice)
            music_db_eff = speech_db - settings.music_under_db - shaped_db
            print(
                f"  speech {speech_db:.1f}dB, shaped bed {shaped_db:.1f}dB -> gain "
                f"{music_db_eff:.1f}dB for {settings.music_under_db:.0f}dB under the voice"
            )
        elif shaped_db != raw_db:
            # Shaping costs level as a side effect. Give it back so --music-db
            # stays a gain relative to the track as delivered.
            music_db_eff = settings.music_db + (raw_db - shaped_db)
            print(
                f"  shaping costs {raw_db - shaped_db:.1f}dB; "
                f"bed gain set to {music_db_eff:.1f}dB"
            )
        if settings.music_window == "flattest":
            music_levels = lc.track_levels(
                ffmpeg, settings.music, cache / f"music_levels_{settings.music.stem}.json"
            )
            print(f"  scanning {len(music_levels)}s of track for the steadiest window per clip")
    rendered: list[dict] = []
    for i, seg in enumerate(segs):
        broll = brolls[i % len(brolls)]
        offset: float | None = None
        if bed is not None and bed_dur > 0:
            broll = bed
            idx = i + settings.broll_start_index
            if settings.broll_spread == "golden":
                offset = ((idx * 0.6180339887498949) % 1.0) * bed_dur
            else:
                offset = (idx * settings.broll_stride) % bed_dur
        geom: lc.Geometry | None = None
        title_lines: list[str] = []
        cues: list[lc.Cue] = []
        # The title's own size can come back reduced, so this clip's style is
        # not necessarily the project's; geometry and drawing both use it.
        clip_style = style
        if style is not None:
            headline = str(seg.get("thumb_text") or seg.get("title") or seg["id"]).strip()
            title_lines, clip_style = lc.fit_title(
                headline, style=style, ffmpeg=ffmpeg, scratch=cache / "fit"
            )
            if clip_style.title_size != style.title_size:
                print(f"  title set at {clip_style.title_size}px so the words break cleanly")
            geom = lc.geometry(
                talk_h=talk_h, title_line_count=len(title_lines), style=clip_style
            )
            if caption_source:
                clip_parts = seg.get("parts") or [
                    {"start": seg["start"], "end": seg["end"]}
                ]
                cues = lc.caption_cues(
                    caption_source,
                    clip_parts,
                    style=style,
                    overlap=(
                        lc.overlap_for(
                            settings.part_transition, settings.part_transition_sec
                        )
                        if len(clip_parts) > 1
                        else 0.0
                    ),
                )
            for note in geom.notes:
                print(f"  NOTE {note}")

        # Framing is measured on this clip's own spans, not on the head of the
        # file: a talk can change camera or seating part-way through, and the
        # frames that matter are the ones being rendered.
        zoom_filter = ""
        if style is not None and settings.talk_zoom != "1.0":
            spans = seg.get("parts") or [{"start": seg["start"], "end": seg["end"]}]
            times = []
            for part in spans:
                lo, hi = float(part["start"]), float(part["end"])
                times += [lo + (hi - lo) * f for f in (0.15, 0.4, 0.65, 0.9)]
            if settings.talk_zoom == "auto":
                fr = lc.measure_face(
                    ffmpeg,
                    settings.source,
                    src_crop=settings.src_crop,
                    talk_h=talk_h,
                    times=times[:8],
                    frame_w=settings.out_w,
                    target_frac=settings.talk_face_frac,
                    zoom_max=settings.talk_zoom_max,
                )
                if fr is None:
                    print("  no face found; framing left alone")
                else:
                    zoom_filter = lc.zoom_filter(
                        zoom=fr.zoom, face_x=fr.face_x, face_y=fr.face_y,
                        talk_h=talk_h, place_y=settings.talk_face_y,
                        frame_w=settings.out_w,
                    )
                    print(
                        f"  face {fr.frames_found}/{fr.frames_tried} frames: "
                        f"x{fr.face_x:.2f} y{fr.face_y:.2f} h{fr.face_frac:.2f}"
                        f" -> zoom {fr.zoom:.2f}"
                    )
            else:
                # A fixed zoom still centres, because a number alone would
                # crop the middle and can cut the speaker in half.
                zoom_filter = lc.zoom_filter(
                    zoom=float(settings.talk_zoom), face_x=0.5, face_y=0.5,
                    talk_h=talk_h, place_y=0.5, frame_w=settings.out_w,
                )

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
                    flip=settings.flip,
                    layout=settings.layout,
                    scratch_dir=cache,
                    zoom_filter=zoom_filter,
                    speed=settings.speed,
                    transition=settings.part_transition,
                    transition_sec=settings.part_transition_sec,
                    peak_dbfs=settings.peak_dbfs,
                    geom=geom,
                    style=clip_style,
                    title_lines=title_lines,
                    cues=cues,
                    voice_clarity=settings.voice_clarity,
                    voice_pitch=settings.voice_pitch,
                    music_compress=settings.music_compress,
                    music_dip_hz=settings.music_dip_hz,
                    music_dip_db=settings.music_dip_db,
                    music=settings.music,
                    music_db=music_db_eff,
                    music_offset=(
                        lc.flattest_offset(music_levels, seg["duration"])
                        if music_levels
                        else _music_offset_for(i, settings, music_dur)
                    ),
                    music_fade=settings.music_fade,
                    duck=settings.music_duck,
                    duck_threshold=settings.music_duck_threshold,
                    duck_ratio=settings.music_duck_ratio,
                    duck_release=settings.music_duck_release,
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
                    font=settings.thumb_font,
                    color=settings.thumb_color,
                    size=settings.thumb_size,
                    max_lines=settings.thumb_lines,
                    margin=settings.thumb_margin,
                    headline=settings.layout != "center",
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
                    "captions": len(cues),
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
        "--thumb-font",
        type=Path,
        default=Path(r"C:/Windows/Fonts/arialbd.ttf"),
        help="Headline font. Impact and Arial Narrow lack Vietnamese u-horn glyphs.",
    )
    parser.add_argument("--thumb-color", default="0xFFD24A", help="Headline colour.")
    parser.add_argument("--thumb-size", type=int, default=84)
    parser.add_argument("--thumb-lines", type=int, default=3)
    parser.add_argument(
        "--thumb-margin", type=int, default=150,
        help="Gap from the bottom of the frame to the headline block.",
    )
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
    parser.add_argument(
        "--broll-chunk",
        type=float,
        default=12.0,
        help="Seconds taken from each B-roll scene for the strip.",
    )
    parser.add_argument(
        "--broll-stride",
        type=float,
        default=47.0,
        help="Seconds of strip offset between consecutive clips.",
    )
    parser.add_argument(
        "--broll-seed",
        type=int,
        default=0,
        help="Shuffle B-roll order with this seed (0 = sorted).",
    )
    parser.add_argument(
        "--layout",
        choices=("half", "center"),
        default="half",
        help="half: talk over B-roll. center: talk framed by scenes, title above, caption below.",
    )
    parser.add_argument(
        "--talk-h",
        type=int,
        default=0,
        help="Talk height in the centre layout; 0 derives it from the source aspect.",
    )
    parser.add_argument(
        "--no-captions",
        dest="captions",
        action="store_false",
        help="Centre layout without the spoken caption under the talk.",
    )
    parser.add_argument(
        "--centre-font",
        type=Path,
        default=Path(r"C:/Windows/Fonts/seguibl.ttf"),
        help="Title and caption font. Arial Black and Oswald lack Vietnamese glyphs.",
    )
    parser.add_argument("--title-size", type=int, default=78)
    parser.add_argument("--title-max-lines", type=int, default=2)
    parser.add_argument("--caption-size", type=int, default=54)
    parser.add_argument("--caption-max-lines", type=int, default=2)
    parser.add_argument(
        "--voice-clarity",
        action="store_true",
        help="Clean up the talk audio: rumble out, low-mid down, consonants up, levelled.",
    )
    parser.add_argument(
        "--voice-pitch",
        type=float,
        default=1.0,
        help="Pitch ratio for the talk; 0.95 is five percent deeper. Duration is preserved.",
    )
    parser.add_argument(
        "--music-compress",
        action="store_true",
        help="Hold the bed at one level; a track with struck bells swings 6dB on its own.",
    )
    parser.add_argument(
        "--music-dip-hz",
        type=float,
        default=450.0,
        help="Centre of the bed's dip, where the voice and the room tone sit.",
    )
    parser.add_argument(
        "--music-dip-db",
        type=float,
        default=0.0,
        help="Depth of that dip in dB, e.g. -5. Zero disables it.",
    )
    parser.add_argument(
        "--music-under-db",
        type=float,
        default=0.0,
        help="Put the bed this many dB under the measured speech, e.g. 14. Overrides --music-db.",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Playback speed, e.g. 0.75 for three-quarter speed. Pitch is preserved.",
    )
    parser.add_argument("--out-w", type=int, default=1080)
    parser.add_argument(
        "--out-h", type=int, default=1920,
        help="Frame size. 1920x1080 gives the same bands in a landscape frame.",
    )
    parser.add_argument(
        "--talk-zoom",
        default="1.0",
        help='Zoom on the talk: a number, or "auto" to measure the face and frame on it.',
    )
    parser.add_argument("--talk-zoom-max", type=float, default=1.5)
    parser.add_argument(
        "--talk-face-frac",
        type=float,
        default=0.32,
        help="Fraction of the talk band's height the face should fill under --talk-zoom auto.",
    )
    parser.add_argument(
        "--talk-face-y",
        type=float,
        default=0.36,
        help="Where the face centre sits vertically in the band, 0=top 1=bottom.",
    )
    parser.add_argument(
        "--peak-dbfs",
        type=float,
        default=0.0,
        help="Ceiling for the finished mix, e.g. -1. Zero leaves the mix unlimited.",
    )
    parser.add_argument(
        "--part-transition",
        choices=("cut", "fade", "dissolve", "fadeblack", "wipeleft", "smoothleft"),
        default="cut",
        help="Join between the spans of a multi-part clip. 'cut' is a hard join.",
    )
    parser.add_argument(
        "--part-transition-sec",
        type=float,
        default=0.5,
        help="Cross-fade length in seconds; each junction shortens the clip by this.",
    )
    parser.add_argument(
        "--music-window",
        choices=("fixed", "flattest"),
        default="fixed",
        help="flattest: pin each clip's bed to the steadiest stretch of the track.",
    )
    parser.add_argument(
        "--flip",
        choices=("none", "top", "all"),
        default="none",
        help="Mirror horizontally: the talk only, the whole frame, or neither.",
    )
    parser.add_argument(
        "--music", type=Path, default=None, help="Instrumental track to lay under the talk."
    )
    parser.add_argument(
        "--music-db", type=float, default=-26.0, help="Resting music gain in dB before ducking."
    )
    parser.add_argument("--music-fade", type=float, default=1.5, help="Music fade in/out seconds.")
    parser.add_argument(
        "--music-offset",
        type=float,
        default=-1.0,
        help="Start the bed at this second of the track; negative spreads it per clip.",
    )
    parser.add_argument(
        "--music-duck",
        choices=("on", "off"),
        default="on",
        help="Sidechain ducking. 'off' holds one constant music level.",
    )
    parser.add_argument(
        "--music-duck-threshold",
        type=float,
        default=0.01,
        help="Sidechain threshold (linear amplitude of the speech).",
    )
    parser.add_argument("--music-duck-ratio", type=float, default=12.0)
    parser.add_argument(
        "--music-duck-release",
        type=float,
        default=300.0,
        help="Ducking release in ms; shorter lets music breathe in pauses.",
    )
    parser.add_argument(
        "--broll-spread",
        choices=("stride", "golden"),
        default="stride",
        help="How to space strip offsets across clips.",
    )
    parser.add_argument(
        "--broll-start-index",
        type=int,
        default=0,
        help="Continue the offset sequence from this clip index.",
    )
    parser.add_argument(
        "--limit", type=int, default=0, help="Render only the first N clips, to preview framing."
    )
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
        thumb_font=args.thumb_font,
        thumb_color=args.thumb_color,
        thumb_size=args.thumb_size,
        thumb_lines=args.thumb_lines,
        thumb_margin=args.thumb_margin,
        limit=args.limit,
        src_crop=args.src_crop,
        reuse_segments=args.reuse_segments,
        segments_file=args.segments_file,
        broll_bed=args.broll_bed,
        broll_chunk=args.broll_chunk,
        broll_stride=args.broll_stride,
        broll_seed=args.broll_seed,
        broll_spread=args.broll_spread,
        broll_start_index=args.broll_start_index,
        flip=args.flip,
        layout=args.layout,
        talk_h=args.talk_h,
        captions=args.captions,
        centre_font=args.centre_font,
        title_size=args.title_size,
        title_max_lines=args.title_max_lines,
        caption_size=args.caption_size,
        caption_max_lines=args.caption_max_lines,
        voice_clarity=args.voice_clarity,
        voice_pitch=args.voice_pitch,
        music_compress=args.music_compress,
        music_dip_hz=args.music_dip_hz,
        music_dip_db=args.music_dip_db,
        music_window=args.music_window,
        peak_dbfs=args.peak_dbfs,
        talk_zoom=args.talk_zoom,
        talk_zoom_max=args.talk_zoom_max,
        talk_face_frac=args.talk_face_frac,
        talk_face_y=args.talk_face_y,
        speed=args.speed,
        out_w=args.out_w,
        out_h=args.out_h,
        part_transition=args.part_transition,
        part_transition_sec=args.part_transition_sec,
        music_under_db=args.music_under_db,
        music=args.music,
        music_db=args.music_db,
        music_fade=args.music_fade,
        music_offset=args.music_offset,
        music_duck=args.music_duck == "on",
        music_duck_threshold=args.music_duck_threshold,
        music_duck_ratio=args.music_duck_ratio,
        music_duck_release=args.music_duck_release,
    )
    return run(settings)


if __name__ == "__main__":
    raise SystemExit(main())
