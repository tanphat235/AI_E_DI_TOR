"""The centre layout: scenes top and bottom, the talk in the middle, text between.

    +----------------------------+
    |        scene video         |
    |   TITLE, TWO-COLOUR CAPS   |
    +----------------------------+
    |                            |
    |      the original talk     |
    |                            |
    +----------------------------+
    |   caption of the speech    |
    |        scene video         |
    +----------------------------+

The style is not invented. It is measured off
``projects/tui-tu-tui-nhan/shorts/clip_001.mp4`` -- the reference the user
approved -- by sampling the real pixels of a frame. Every default in
``CentreStyle`` carries the measurement it came from, because these numbers
look arbitrary and are not:

* title fill ``#F6FE02`` on line 1 and ``#FD7201`` on line 2 -- median RGB of
  the 18733 yellow and 21736 orange glyph pixels in the title band;
* caption fill ``#F9FE0B`` -- median of 24369 glyph pixels;
* scrim ``black@0.55`` -- inside/outside luminance ratio across the box edge
  read 0.55 at the title and 0.58 at the caption;
* stroke ``size/9`` -- the black run against a glyph edge measured 8 px at the
  title's 78 px and 6 px at the caption's 54 px (25th percentile; the median is
  inflated by the gaps between letters);
* sizes 78 and 54 -- calibrated by rendering the reference's own strings and
  matching their drawn width (896 px title, 766 px caption) to within 1%.

The font is Segoe UI Black. It is the only heavy face on this machine that
draws Vietnamese: Arial Black, Montserrat and Oswald all render tofu boxes for
the u-horn and circumflex-tilde stack, which was caught by rendering a sheet
and looking at it, not by reading a font's coverage table.
"""

from __future__ import annotations

import textwrap
from dataclasses import dataclass, field
from pathlib import Path

FRAME_W = 1080
FRAME_H = 1920


@dataclass(frozen=True, slots=True)
class CentreStyle:
    """Type and geometry for the centre layout. See the module docstring."""

    font: Path = Path(r"C:/Windows/Fonts/seguibl.ttf")
    # Line 1 yellow, line 2 orange, then repeating -- the reference's own
    # two-colour title.
    title_colors: tuple[str, ...] = ("0xF6FE02", "0xFD7201")
    title_size: int = 78
    title_lines: int = 2
    caption_color: str = "0xF9FE0B"
    caption_size: int = 54
    caption_lines: int = 2
    # Vietnamese stacks diacritics above the cap line, so it needs more leading
    # than Latin text. The reference's baseline pitch was 95 px at size 78 and
    # 62 px at 54, i.e. 1.22 and 1.15; 1.25 clears the tallest stack.
    line_ratio: float = 1.25
    scrim: str = "black@0.55"
    # Padding inside the scrim box, as a fraction of the font size.
    pad_ratio: float = 0.30
    # Clear space between the talk and each text block.
    gap: int = 18
    # Fraction of the frame width text may occupy before it wraps.
    wrap_frac: float = 0.90
    # Advance width per character, in ems, used to wrap without a font engine.
    # Segoe UI Black measured 0.578 em for caps and 0.530 for mixed case.
    em_caps: float = 0.578
    em_mixed: float = 0.530
    # Height of one drawn line including diacritics and descenders: measured
    # 1.128 em for caps and 1.148 for mixed case, so 1.15 covers both. The
    # scrim is sized from this rather than from the font's declared line
    # height, which for this face is a useless 2.67 em.
    ink_ratio: float = 1.15
    # Caption gaps shorter than this keep one scrim plate alight across them,
    # so the box does not blink between consecutive transcript lines.
    plate_join: float = 0.6


@dataclass(frozen=True, slots=True)
class Cue:
    start: float
    end: float
    lines: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Geometry:
    """Where each band lands in the 1080x1920 frame."""

    talk_h: int
    talk_y: int
    title_top: int
    title_line_h: int
    caption_top: int
    caption_line_h: int
    scene_top_h: int
    scene_bottom_h: int
    notes: tuple[str, ...] = field(default=())


_BS = chr(92)


def drawtext_path(path: Path) -> str:
    """Escape a Windows path for a drawtext option value.

    drawtext splits options on ':' and treats a backslash as an escape, so a
    raw "C:\\Windows\\..." silently truncates the filter.
    """
    return str(path).replace(_BS, "/").replace(":", _BS + ":")


def _em(text: str, style: CentreStyle) -> float:
    """Advance width per character. Caps are wider than mixed case."""
    letters = [c for c in text if c.isalpha()]
    if letters and sum(c.isupper() for c in letters) / len(letters) > 0.7:
        return style.em_caps
    return style.em_mixed


def wrap(
    text: str,
    *,
    size: int,
    style: CentreStyle,
    width_px: int | None = None,
    balance: bool = False,
) -> list[str]:
    """Wrap to the drawn width, estimating the advance width per character.

    There is no font engine here (no PIL, no freetype binding), so the width is
    estimated from the measured em values and then given a 10% margin by
    ``wrap_frac``. Being one character conservative is invisible; overflowing
    the frame is not.

    ``balance`` spreads the words evenly over the lines it needs instead of
    filling each line greedily. For a title that is the difference between
    "THỜ SAO CHO TRANG / NGHIÊM?" and "THỜ SAO CHO / TRANG NGHIÊM?" -- greedy
    wrapping leaves a one-word orphan, which at 78 px is the first thing the
    eye lands on. Captions stay greedy: they change every couple of seconds
    and a stable left edge reads better than a balanced one.
    """
    text = " ".join(text.split())
    if not text:
        return []
    avail = width_px if width_px is not None else int(FRAME_W * style.wrap_frac)
    per_line = max(6, int(avail / (size * _em(text, style))))
    lines = textwrap.wrap(text, width=per_line) or [text]
    if not balance or len(lines) < 2:
        return lines
    # Narrow the width until the wrap needs one more line; the last width that
    # still fits the same number of lines is the most even one.
    target = len(lines)
    best = lines
    for width in range(per_line, 5, -1):
        trial = textwrap.wrap(text, width=width)
        if len(trial) != target:
            break
        best = trial
    return best


def geometry(*, talk_h: int, title_line_count: int, style: CentreStyle) -> Geometry:
    """Place the talk centred, then hang the title above it and the caption below.

    The talk is centred rather than pinned so both scene bands survive: with a
    16:9 source (608 px tall) they come out 384 and 459 px, and with a squarer
    9:8 crop (960 px) still 208 and 283 px. A talk taller than about 1250 px
    leaves no room for a scene band, which is why ``notes`` reports the
    remaining heights for the caller to print.
    """
    title_line_h = int(style.title_size * style.line_ratio)
    caption_line_h = int(style.caption_size * style.line_ratio)
    title_pad = int(style.title_size * style.pad_ratio)
    caption_pad = int(style.caption_size * style.pad_ratio)

    talk_y = (FRAME_H - talk_h) // 2
    title_block = title_line_h * max(1, title_line_count) + 2 * title_pad
    title_top = talk_y - style.gap - title_block + title_pad
    caption_top = talk_y + talk_h + style.gap + caption_pad

    scene_top_h = max(0, title_top - title_pad)
    caption_block = caption_line_h * style.caption_lines + 2 * caption_pad
    scene_bottom_h = max(0, FRAME_H - (caption_top - caption_pad + caption_block))

    notes: list[str] = []
    if scene_top_h < 80:
        notes.append(f"top scene band is only {scene_top_h}px; lower --talk-h to widen it")
    if scene_bottom_h < 80:
        notes.append(f"bottom scene band is only {scene_bottom_h}px; lower --talk-h")
    if title_top - title_pad < 0:
        notes.append("title block runs off the top of the frame; lower --talk-h")
    return Geometry(
        talk_h=talk_h,
        talk_y=talk_y,
        title_top=title_top,
        title_line_h=title_line_h,
        caption_top=caption_top,
        caption_line_h=caption_line_h,
        scene_top_h=scene_top_h,
        scene_bottom_h=scene_bottom_h,
        notes=tuple(notes),
    )


def _split_span(total: float, weights: list[int], floor: float) -> list[float]:
    """Divide a span by weight, but never give a page less than ``floor``.

    Weighting by character count alone produced 0.18 s pages -- about five
    frames, which reads as a flicker rather than a line of text. The short
    pages are raised to the floor and the difference is taken back from the
    ones with slack, in proportion to how much slack they have. When the whole
    span is too short to give every page the floor, they share it equally,
    which is the best available answer rather than a good one.
    """
    n = len(weights)
    if n == 1:
        return [total]
    if total <= floor * n:
        return [total / n] * n
    w_total = sum(weights)
    spans = [total * w / w_total for w in weights]
    short = [i for i, s in enumerate(spans) if s < floor]
    if not short:
        return spans
    debt = sum(floor - spans[i] for i in short)
    slack = {i: spans[i] - floor for i in range(n) if spans[i] > floor}
    slack_total = sum(slack.values())
    for i in short:
        spans[i] = floor
    for i, have in slack.items():
        spans[i] -= debt * have / slack_total
    return spans


def caption_cues(
    transcript: list[dict],
    parts: list[dict],
    *,
    style: CentreStyle,
    min_dur: float = 0.35,
    overlap: float = 0.0,
) -> list[Cue]:
    """Transcript lines that fall inside the clip, in clip time.

    A "parts" segment splices spans that need not be chronological, so each
    part is remapped separately by its own offset -- mapping on the segment's
    overall start would put every cue after the first splice in the wrong place.

    A transcript line too long for ``caption_lines`` is paged rather than
    truncated, and its duration is split between pages by their share of the
    characters, so a page of two words does not hold the screen as long as a
    page of nine.
    """
    cues: list[Cue] = []
    offset = 0.0
    for part in parts:
        p0, p1 = float(part["start"]), float(part["end"])
        for seg in transcript:
            s, e = float(seg["start"]), float(seg["end"])
            if e <= p0 or s >= p1:
                continue
            text = str(seg.get("text") or "").strip()
            if not text:
                continue
            s, e = max(s, p0), min(e, p1)
            if e - s < min_dur:
                continue
            lines = wrap(text, size=style.caption_size, style=style)
            pages = [
                lines[i : i + style.caption_lines]
                for i in range(0, len(lines), style.caption_lines)
            ]
            weights = [max(1, sum(len(x) for x in page)) for page in pages]
            spans = _split_span(e - s, weights, min_dur)
            t = s - p0 + offset
            for page, span in zip(pages, spans, strict=True):
                cues.append(Cue(start=t, end=t + span, lines=tuple(page)))
                t += span
        # A cross-fade eats ``overlap`` seconds at each junction, so every part
        # after the first begins that much earlier than its spans imply.
        offset += (p1 - p0) - overlap

    cues.sort(key=lambda c: c.start)
    # Transcript lines abut, and a cue that ends exactly where the next begins
    # can flicker on the shared frame. Ending each one just before the next
    # starts removes the overlap without leaving a visible hole.
    out: list[Cue] = []
    for i, c in enumerate(cues):
        end = c.end
        if i + 1 < len(cues):
            end = min(end, cues[i + 1].start - 0.01)
        if end - c.start >= min_dur * 0.5:
            out.append(Cue(start=c.start, end=end, lines=c.lines))
    return out


def _text_block(
    *,
    lines: list[str],
    style: CentreStyle,
    size: int,
    colors: tuple[str, ...],
    top: int,
    line_h: int,
    slots: int,
    text_dir: Path,
    stem: str,
    enable: str = "",
) -> list[str]:
    """One drawtext per line, each centred on its own width.

    Per line rather than one multi-line drawtext, for two reasons. The title
    needs a different colour on each line, which a single drawtext cannot do.
    And drawtext's own line_spacing is unusable with this face: Segoe UI Black
    declares a 2.67 em default line height, so two 54 px lines came out 144 px
    apart, and line_spacing is applied at double weight on top of that -- which
    makes the spacing a font-specific correction rather than a measurement.
    Positioning each line is exact instead, because drawtext puts the ink top
    at y with no offset (measured: +0 px at both sizes).

    ``slots`` is how many lines the block is sized for, so a one-line caption
    sits centred in the same plate a two-line one fills and the box never
    resizes from cue to cue.
    """
    text_dir.mkdir(parents=True, exist_ok=True)
    if not lines:
        return []
    offset = int((slots - len(lines)) * line_h / 2)
    out: list[str] = []
    for i, line in enumerate(lines):
        f = text_dir / f"{stem}_{i}.txt"
        f.write_text(line, encoding="utf-8")
        opts = [
            f"fontfile='{drawtext_path(style.font)}'",
            f"textfile='{drawtext_path(f)}'",
            f"fontsize={size}",
            f"fontcolor={colors[i % len(colors)]}",
            f"borderw={max(3, size // 9)}",
            "bordercolor=black",
            "x=(w-text_w)/2",
            f"y={top + offset + i * line_h}",
        ]
        if enable:
            opts.append(f"enable='{enable}'")
        out.append("drawtext=" + ":".join(opts))
    return out


def _plate(
    *, style: CentreStyle, size: int, top: int, line_h: int, slots: int, enable: str = ""
) -> str:
    """The scrim behind a text block.

    A separate drawbox rather than drawtext's own box=1: that box is sized from
    the font's declared line height, which for this face is 2.67 em, so a
    two-line caption came out in a plate half again too tall. This one is sized
    from the measured ink height instead.
    """
    pad = int(size * style.pad_ratio)
    h = (slots - 1) * line_h + int(size * style.ink_ratio) + 2 * pad
    opts = ["x=0", f"y={top - pad}", "w=iw", f"h={h}", f"color={style.scrim}", "t=fill"]
    if enable:
        opts.append(f"enable='{enable}'")
    return "drawbox=" + ":".join(opts)


def _cue_runs(cues: list[Cue], join: float) -> list[tuple[float, float]]:
    """Contiguous stretches of caption, so one plate covers a run of cues."""
    runs: list[tuple[float, float]] = []
    for cue in cues:
        if runs and cue.start - runs[-1][1] <= join:
            runs[-1] = (runs[-1][0], cue.end)
        else:
            runs.append((cue.start, cue.end))
    return runs


def overlap_for(transition: str, transition_sec: float) -> float:
    """Seconds each junction removes from the timeline. Zero for a hard cut.

    xfade and acrossfade both consume the transition from *both* sides, so a
    clip of n parts finishes (n-1)*transition_sec shorter than the sum of its
    spans. Every other clock -- the caption times, the music fade-out, the -t
    on the B-roll -- has to be told, or the captions drift half a second per
    junction and the bed fades early.
    """
    return 0.0 if transition == "cut" else max(0.0, transition_sec)


def _transition_chain(parts: list[dict], transition: str, secs: float) -> list[str]:
    """Cross-fade the parts together, picture and sound, one junction at a time.

    A dissolve here is not decoration: these spans are cut from different
    places in one talk, and a hard cut between them reads as a glitch where a
    dissolve reads as "later, on the same subject".
    """
    kind = "fade" if transition == "dissolve" else transition
    chains: list[str] = []
    for i in range(len(parts)):
        chains.append(f"[{i}:a]asetpts=PTS-STARTPTS[a{i}]")
    v_prev, a_prev = "p0", "a0"
    # Length of the chain built so far, which is where the next junction goes.
    acc = float(parts[0]["end"] - parts[0]["start"])
    for i in range(1, len(parts)):
        d = float(parts[i]["end"] - parts[i]["start"])
        offset = max(0.0, acc - secs)
        v_out, a_out = f"vx{i}", f"ax{i}"
        chains.append(
            f"[{v_prev}][p{i}]xfade=transition={kind}:duration={secs:.3f}"
            f":offset={offset:.3f}[{v_out}]"
        )
        chains.append(f"[{a_prev}][a{i}]acrossfade=d={secs:.3f}:c1=tri:c2=tri[{a_out}]")
        v_prev, a_prev = v_out, a_out
        acc = acc + d - secs
    chains.append(f"[{v_prev}]null[talk]")
    chains.append(f"[{a_prev}]anull[speech]")
    return chains


def video_graph(
    *,
    parts: list[dict],
    broll_index: int,
    src_crop: str,
    flip: str,
    geom: Geometry,
    style: CentreStyle,
    title_lines: list[str],
    cues: list[Cue],
    text_dir: Path,
    stem: str,
    transition: str = "cut",
    transition_sec: float = 0.5,
) -> tuple[str, str]:
    """Filter graph for the centre layout, plus the label carrying the picture.

    The text is always drawn after any mirroring, so ``--flip all`` mirrors the
    picture and leaves the words readable. Mirroring the finished frame is what
    made the reference's own burned-in text come out backwards, and that is the
    mistake this ordering exists to avoid.
    """
    text_dir.mkdir(parents=True, exist_ok=True)
    talk_flip = "hflip," if flip == "top" else ""
    chains: list[str] = []

    for i in range(len(parts)):
        pre = f"crop={src_crop}," if src_crop else ""
        # setpts resets each part's clock to zero: xfade reads its offset on
        # the first input's own timeline, and a part cut with -ss carries the
        # source's timestamps unless they are reset.
        #
        # The order matters and is not cosmetic. With setpts AFTER fps, xfade
        # refuses the input with "the inputs needs to be a constant frame rate;
        # current rate of 1/0 is invalid" -- setpts clears the frame-rate
        # metadata that fps had just established. fps has to come last.
        chains.append(
            f"[{i}:v]{pre}scale={FRAME_W}:{geom.talk_h}:force_original_aspect_ratio=increase,"
            f"crop={FRAME_W}:{geom.talk_h},{talk_flip}setpts=PTS-STARTPTS,"
            f"fps=30,setsar=1[p{i}]"
        )
    if len(parts) == 1:
        chains.append("[p0]null[talk]")
        chains.append("[0:a]anull[speech]")
    elif transition == "cut":
        pairs = "".join(f"[p{i}][{i}:a]" for i in range(len(parts)))
        chains.append(f"{pairs}concat=n={len(parts)}:v=1:a=1[talk][speech]")
    else:
        chains.extend(_transition_chain(parts, transition, transition_sec))

    chains.append(
        f"[{broll_index}:v]scale={FRAME_W}:{FRAME_H}:force_original_aspect_ratio=increase,"
        f"crop={FRAME_W}:{FRAME_H},fps=30,setsar=1[bg]"
    )
    frame_flip = ",hflip" if flip == "all" else ""
    chains.append(f"[bg][talk]overlay=0:{geom.talk_y}{frame_flip}[framed]")

    draws: list[str] = []
    title = title_lines[: style.title_lines]
    if title:
        draws.append(
            _plate(
                style=style,
                size=style.title_size,
                top=geom.title_top,
                line_h=geom.title_line_h,
                slots=len(title),
            )
        )
        draws += _text_block(
            lines=title,
            style=style,
            size=style.title_size,
            colors=style.title_colors,
            top=geom.title_top,
            line_h=geom.title_line_h,
            slots=len(title),
            text_dir=text_dir,
            stem=f"{stem}_title",
        )

    # One plate per contiguous run of captions, not one per cue: the box would
    # otherwise blink off and on between every transcript line.
    for a, b in _cue_runs(cues, style.plate_join):
        draws.append(
            _plate(
                style=style,
                size=style.caption_size,
                top=geom.caption_top,
                line_h=geom.caption_line_h,
                slots=style.caption_lines,
                enable=f"between(t,{a:.3f},{b:.3f})",
            )
        )
    for i, cue in enumerate(cues):
        draws += _text_block(
            lines=list(cue.lines),
            style=style,
            size=style.caption_size,
            colors=(style.caption_color,),
            top=geom.caption_top,
            line_h=geom.caption_line_h,
            slots=style.caption_lines,
            text_dir=text_dir,
            stem=f"{stem}_cap{i:04d}",
            enable=f"between(t,{cue.start:.3f},{cue.end:.3f})",
        )

    if draws:
        chains.append("[framed]" + ",".join(draws) + "[v]")
    else:
        chains.append("[framed]null[v]")
    return ";".join(chains), "[v]"


def voice_chain(*, clarity: bool, pitch: float, sample_rate: int = 48000) -> str:
    """Filters applied to the talk's own audio, or "" when both are off.

    Clarity is deliberately mild. The measured problem on these recordings is
    not hiss but a loud, broadband room tone -- on the reference it sat only
    11 dB under the speech -- so the chain lifts the consonant band and thins
    the low-mid mud rather than reaching for a denoiser, which smears the
    voice at the level of reduction this would need.

    Pitch is shifted by resampling and then restoring the tempo, so the clip's
    duration is unchanged and stays in sync with the picture. atempo accepts
    0.5-100, so a 5% shift (1/0.95 = 1.0526) is well inside its range.
    """
    steps: list[str] = []
    if clarity:
        # Every gain below is a net figure, measured on a 60 s sample of the
        # song-tot talk after subtracting the chain's own overall gain, so each
        # is a tone change and not a level change:
        #   0-80 Hz  -10.1 dB   handling rumble and room boom, no voice there
        #   250-500   -1.4 dB   the small room's boxiness
        #   2-4 kHz   +3.3 dB   consonants, which is what "clearer" means here
        #   5-8 kHz   +0.5 dB   held down deliberately; see the shelf below
        #   8-16 kHz  -0.1 dB
        # A -2 dB dip at 350 Hz was tried first and came out at +0.1 dB net:
        # the compressor put the low-mid straight back. -4 dB is what survives.
        steps.append("highpass=f=80")
        steps.append("equalizer=f=350:width_type=o:width=1.0:g=-4")
        # width 1.0 octave, not 1.4: the wider filter spilled another 0.5 dB
        # into 5-8 kHz, which on a compressed talk reads as sibilance.
        steps.append("equalizer=f=2800:width_type=o:width=1.0:g=3.5")
        # acompressor's makeup is a LINEAR factor, not dB -- makeup=2 is +6 dB
        # and drove the finished mix to -1.1 dBFS. 1.6 leaves the speech within
        # 0.7 dB of where it started, takes 2 dB off the crest factor, and
        # keeps 3 dB of headroom for the bed.
        steps.append("acompressor=threshold=-20dB:ratio=2.5:attack=15:release=250:makeup=1.6")
        # Last, and after the compressor on purpose. The presence lift and the
        # compressor together were pushing 5-8 kHz up 2.75 dB and 8-16 kHz up
        # 2.67 dB -- neither was asked for, and on a re-upload that band is
        # mostly sibilance and codec chatter. This shelf takes them to +0.5 and
        # -0.1 while leaving the consonant lift at +3.3.
        steps.append("equalizer=f=9000:width_type=o:width=2.0:g=-3")
    if abs(pitch - 1.0) > 1e-6:
        # aresample FIRST. asetrate reinterprets the stream at a new rate, so
        # it only shifts by the intended ratio if the incoming rate is the one
        # in the expression. Without this the 44.1 kHz sources came out
        # 23.893 s instead of 26.000 s -- asetrate=48000*0.95 is 45600, which
        # is faster than 44100, not slower, and atempo then removed more.
        steps.append(f"aresample={sample_rate}")
        steps.append(f"asetrate={sample_rate}*{pitch:.6f}")
        steps.append(f"aresample={sample_rate}")
        steps.append(f"atempo={1.0 / pitch:.6f}")
    return ",".join(steps)


def music_chain(
    *,
    gain_db: float,
    fade: float,
    duration: float,
    compress: bool,
    dip_hz: float,
    dip_db: float,
    sample_rate: int = 48000,
) -> str:
    """Filters for the bed: one steady level, sitting out of the voice's way.

    ``compress`` is what holds the level. On the reference track the flattest
    39 s window still swung 6.6 dB on its own -- it has struck bells in it --
    and compression took that to 2.6 dB. The dip is the other half: it moves
    the bed out of the band the voice and the room tone share, which on the
    reference cut the bed's 200-800 Hz share from 47.8% to 30.8%.

    A mono source is upmixed with pan, not aformat. swresample rematrixes mono
    to stereo to preserve total energy and so costs exactly 3.01 dB, which is
    enough to put a carefully measured bed under the noise floor.
    """
    steps = [
        "pan=stereo|c0=c0|c1=c0",
        f"aformat=sample_fmts=fltp:sample_rates={sample_rate}",
    ]
    if compress:
        steps.append("acompressor=threshold=-26dB:ratio=4:attack=20:release=400")
    if dip_db:
        steps.append(f"equalizer=f={dip_hz:g}:width_type=o:width=1.6:g={dip_db:g}")
    steps.append(f"volume={gain_db}dB")
    if fade > 0:
        steps.append(f"afade=t=in:st=0:d={fade}")
        steps.append(f"afade=t=out:st={max(0.0, duration - fade):.3f}:d={fade}")
    return ",".join(steps)


# --------------------------------------------------------------------------
# Measuring the bed. Both of these exist because a shaped bed cannot be set by
# eye: the compressor and the dip change its level as a side effect, and the
# track's own level wanders.
# --------------------------------------------------------------------------


def _decode_mono(
    ffmpeg: Path,
    path: Path,
    *,
    start: float | None = None,
    dur: float | None = None,
    rate: int = 8000,
    chain: str = "",
):
    """Decode to mono float samples through an optional filter chain."""
    import subprocess

    import numpy as np

    cmd = [str(ffmpeg), "-v", "error", "-y"]
    if start is not None:
        cmd += ["-ss", f"{start:.3f}"]
    cmd += ["-i", str(path)]
    if dur is not None:
        cmd += ["-t", f"{dur:.3f}"]
    if chain:
        cmd += ["-af", chain]
    cmd += ["-vn", "-ac", "1", "-ar", str(rate), "-f", "s16le", "-"]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError((proc.stderr or b"").decode("utf-8", "replace")[-600:])
    return np.frombuffer(proc.stdout, dtype="<i2").astype(np.float64) / 32768.0


def _rms_db(x) -> float:
    import numpy as np

    return float(20 * np.log10(max(float(np.sqrt((x**2).mean())), 1e-9)))


def bed_levels_db(
    ffmpeg: Path,
    music: Path,
    *,
    compress: bool,
    dip_hz: float,
    dip_db: float,
    sample: float = 120.0,
    start: float = 60.0,
) -> tuple[float, float]:
    """The track's level before and after shaping, in dB.

    Both are needed. The difference is what --music-db has to be given back,
    because the same -20 landed the bed 27.8 dB under the voice with shaping
    and about 15 dB under it without: the compressor alone eats around 9 dB.
    The shaped figure is what --music-under-db aims with.
    """
    shape: list[str] = []
    if compress:
        shape.append("acompressor=threshold=-26dB:ratio=4:attack=20:release=400")
    if dip_db:
        shape.append(f"equalizer=f={dip_hz:g}:width_type=o:width=1.6:g={dip_db:g}")
    raw = _decode_mono(ffmpeg, music, start=start, dur=sample)
    shaped = (
        _decode_mono(ffmpeg, music, start=start, dur=sample, chain=",".join(shape))
        if shape
        else raw
    )
    n = min(len(raw), len(shaped))
    if n == 0:
        return 0.0, 0.0
    return _rms_db(raw[:n]), _rms_db(shaped[:n])


def speech_level_db(
    ffmpeg: Path,
    source: Path,
    spans: list[tuple[float, float]],
    *,
    chain: str,
    budget: float = 90.0,
) -> float:
    """Level of the treated speech, so the bed can be set against it.

    Measured on the talk itself, through the same voice chain the render will
    use, because clarity and pitch both move the level -- the compressor's
    makeup alone is worth several dB. Frames are gated at the 55th percentile
    so pauses do not drag the figure down; the result is the level of the
    speech, not of the recording.

    Sampled from the spans that will actually be cut rather than from the head
    of the file, which on these re-uploads is often an intro of a different
    loudness.
    """
    import numpy as np

    if not spans:
        return 0.0
    per = max(4.0, budget / len(spans))
    chunks = []
    for start, end in spans:
        take = min(per, max(0.0, end - start))
        if take < 1.0:
            continue
        chunks.append(_decode_mono(ffmpeg, source, start=start, dur=take, rate=48000, chain=chain))
    if not chunks:
        return 0.0
    x = np.concatenate(chunks)
    hop = 4800  # 100 ms
    frames = x[: (len(x) // hop) * hop].reshape(-1, hop)
    level = np.sqrt((frames**2).mean(axis=1) + 1e-12)
    voiced = level[level > np.percentile(level, 55)]
    return _rms_db(voiced) if len(voiced) else _rms_db(x)


def track_levels(ffmpeg: Path, music: Path, cache: Path) -> list[float]:
    """Per-second level of the whole track in dB, cached.

    Decoded once at 8 kHz mono -- the level of a music bed does not need
    fidelity, and a 68-minute track comes back in a couple of seconds.
    """
    import json

    import numpy as np

    if cache.is_file():
        return list(json.loads(cache.read_text(encoding="utf-8")))
    rate = 8000
    x = _decode_mono(ffmpeg, music, rate=rate)
    whole = x[: (len(x) // rate) * rate].reshape(-1, rate)
    levels = 20 * np.log10(np.maximum(np.sqrt((whole**2).mean(axis=1)), 1e-9))
    out = [round(float(v), 3) for v in levels]
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(out), encoding="utf-8")
    return out


def flattest_offset(levels: list[float], duration: float) -> float:
    """Start of the steadiest window of that length, in seconds.

    Steadiest means smallest peak-to-trough of the per-second level, not
    closest to the mean: what the ear objects to is the bed rising and falling,
    which is a range, not a variance. On the reference track this picked 21:31
    with 6.6 dB of swing where the stretch at 0:40 had 11.3.
    """
    w = max(1, int(round(duration)))
    if len(levels) <= w:
        return 0.0
    best_i, best_swing = 0, float("inf")
    for i in range(len(levels) - w):
        window = levels[i : i + w]
        swing = max(window) - min(window)
        if swing < best_swing:
            best_i, best_swing = i, swing
    return float(best_i)


# --------------------------------------------------------------------------
# Fitting the title. The em estimate in wrap() is good enough for captions,
# which change every couple of seconds, but not for a title drawn at 78 px and
# read first: measured against real renders it is 0.58 em/char on one string
# and 0.67 on another, a 10% spread that decides whether a line overflows.
# These functions measure instead.
# --------------------------------------------------------------------------


def _token_widths(
    ffmpeg: Path, font: Path, size: int, tokens: list[str], scratch: Path
) -> tuple[dict[str, int], int]:
    """Drawn width of each word, plus the width of a space, in pixels.

    Measured once per title: n+2 short ffmpeg runs. drawtext applies no
    inter-word kerning, so a line's width is the sum of its words plus the
    spaces between them, which matched the directly measured line width to
    within a pixel on every case tried.
    """
    import subprocess

    import numpy as np

    scratch.mkdir(parents=True, exist_ok=True)
    canvas_w = 2400  # wider than any line, so nothing is clipped while measuring

    def draw(text: str) -> int:
        f = scratch / "measure.txt"
        f.write_text(text, encoding="utf-8")
        raw = scratch / "measure.raw"
        vf = (
            f"drawtext=fontfile='{drawtext_path(font)}':textfile='{drawtext_path(f)}'"
            f":fontsize={size}:fontcolor=white:borderw={max(3, size // 9)}"
            ":bordercolor=black:x=(w-text_w)/2:y=30"
        )
        subprocess.run(
            [str(ffmpeg), "-v", "error", "-y", "-f", "lavfi",
             "-i", f"color=c=black:s={canvas_w}x{int(size * 2.6)}",
             "-frames:v", "1", "-vf", vf, "-f", "rawvideo", "-pix_fmt", "gray", str(raw)],
            capture_output=True,
        )
        img = np.frombuffer(raw.read_bytes(), dtype=np.uint8).reshape(-1, canvas_w)
        cols = np.where((img > 120).any(axis=0))[0]
        return int(cols.max() - cols.min() + 1) if len(cols) else 0

    widths = {t: draw(t) for t in dict.fromkeys(tokens)}
    # A space has no ink, so it is measured as the difference between a pair
    # drawn together and the two words drawn apart.
    if len(tokens) >= 2:
        pair = draw(f"{tokens[0]} {tokens[1]}")
        space = max(0, pair - widths[tokens[0]] - widths[tokens[1]])
    else:
        space = int(size * 0.28)
    return widths, space


def _splits(n_tokens: int, n_lines: int) -> list[list[int]]:
    """Every way to cut a token list into n_lines contiguous, non-empty runs."""
    import itertools

    if n_lines == 1:
        return [[n_tokens]]
    out: list[list[int]] = []
    for cuts in itertools.combinations(range(1, n_tokens), n_lines - 1):
        bounds = (0, *cuts, n_tokens)
        out.append([bounds[i + 1] - bounds[i] for i in range(n_lines)])
    return out


def fit_title(
    text: str,
    *,
    style: CentreStyle,
    ffmpeg: Path,
    scratch: Path,
    min_size_frac: float = 0.75,
) -> tuple[list[str], CentreStyle]:
    """Break the title and pick its size so every line fits and reads right.

    Two rules beyond fitting, in order:

    * **No line may begin with a one- or two-character word.** Vietnamese is
      full of two-syllable compounds, and a greedy or evenness-only wrap
      happily splits them: "CÁCH GIÚP NGƯỜI ĐỔ / VỠ TRONG HÔN NHÂN" fits the
      frame and is still wrong, because ĐỔ VỠ is one word.
    * Then the most even split, which is what stops a one-word orphan line.

    If no acceptable break fits at the requested size, the size steps down in
    fours rather than letting a line overflow or a word break. The returned
    style carries whatever size was used, so the geometry and the drawing agree.
    """
    from dataclasses import replace

    tokens = " ".join(text.split()).split()
    if not tokens:
        return [], style
    budget = int(FRAME_W * style.wrap_frac)
    size = style.title_size
    floor = max(24, int(style.title_size * min_size_frac))

    while size >= floor:
        widths, space = _token_widths(ffmpeg, style.font, size, tokens, scratch)

        # Bound as defaults: the closure would otherwise read whatever the
        # next loop iteration re-measured.
        def line_px(
            words: list[str], _w: dict[str, int] = widths, _s: int = space
        ) -> int:
            return sum(_w[t] for t in words) + _s * (len(words) - 1)

        best: tuple[int, int, list[str]] | None = None
        for n_lines in range(1, style.title_lines + 1):
            for counts in _splits(len(tokens), n_lines):
                lines, i = [], 0
                for c in counts:
                    lines.append(tokens[i : i + c])
                    i += c
                px = [line_px(x) for x in lines]
                if max(px) > budget:
                    continue
                orphans = sum(1 for x in lines[1:] if len(x[0]) <= 2)
                spread = max(px) - min(px)
                key = (orphans, spread, [" ".join(x) for x in lines])
                if best is None or key[:2] < best[:2]:
                    best = key  # type: ignore[assignment]
        if best is not None and best[0] == 0:
            return best[2], replace(style, title_size=size)
        if best is not None and size - 4 < floor:
            # Nothing clean fits anywhere in the size range; take the fitting
            # break rather than overflowing, and let the caller see the size.
            return best[2], replace(style, title_size=size)
        size -= 4
    return wrap(text, size=style.title_size, style=style, balance=True)[: style.title_lines], style
