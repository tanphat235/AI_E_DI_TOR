"""Shared defaults for the --layout center short format.

Single source of truth for the flags every center-format project's batch.py
should pass to build_shorts.py, so a change here (a new zoom cap, a pitch
correction) takes effect everywhere instead of being copy-pasted and drifting
project to project. Values are measured/decided house style, not guesses --
see scripts/shorts/README.md's "centre layout" section for the reasoning
behind each one.

Per-project batch.py still owns what genuinely varies per source: the crop,
the delogo boxes (every re-upload's watermark layout differs), the source
path, and --speed when a job needs pacing adjusted to hit a length target.

The talk is 62.5% of the 1080x1920 frame and sits in the middle
(``--talk-h 1200``; was 1344/70% until 2026-10-12 -- see that date's note on
``TALK_H`` below). Above the title and below the tag are background-scene
bands, the same arrangement as the original centre layout. Do not pass
``--text-edges``: that pins the title to the top edge and the tag to the
bottom edge and removes both scene bands. It was tried on 2026-10-08 and
rejected the next day.

The title plate sits on the talk and the tag plate sits under it. Both plates
are the ink only, so there is no light strip of scene between the words and
the picture. That strip was the gap marked in red on 2026-10-08. Do not put
the 18px gap or the plate padding back unless asked. `layout_center.py`'s
``CentreStyle.min_top_margin`` additionally pushes the talk down from centre
so the title clears a crop YouTube applies in its own UI -- see that field's
docstring, not duplicated here.

``--src-crop`` should match this band's aspect (1080x1200, about 1080x1200
scaled, i.e. ~0.9:1 -- roughly 972x1080 on a 1920x1080 source). A crop
measured for the old 960 or 1344 target (most existing per-project crops,
measured at the 1080:960 = 1.125 aspect) is not a perfect match and gets
cropped further to fill this band, which crops in on the face slightly more
than intended -- re-measure a project's crop against the current TALK_H when
its framing looks tighter than expected, rather than assuming the old
numbers still apply exactly.
"""

from __future__ import annotations

TAG = "@loiphatdayquathayphaphoa"
TRANSITION_SEC = 0.5
# 62.5% of the 1920-tall frame, down from 1344/70% on 2026-10-12. Raising
# min_top_margin (layout_center.CentreStyle) to clear YouTube's own crop of
# the title pushed the talk down far enough that 1344 left almost no room
# for the bottom scene band / tag -- this gives that margin back. Even,
# which yuv420 wants.
TALK_H = 1200
TALK_ZOOM_MAX = 1.7  # as of 2026-10-07; was 1.4 and was clamping short of
# --talk-face-frac's own target on a normal (non-upscaled) 1280x720 source.
TALK_FACE_FRAC = 0.45
TALK_FACE_Y = 0.5
VOICE_PITCH = 0.94
PEAK_DBFS = -1.0
SPEED = 1.1  # as of 2026-10-08; the default pace for every center-format short
MUSIC_DIP_DB = -5.0
MUSIC_FADE_SEC = 1.0
MUSIC_UNDER_DB = 14.0
BROLL_CHUNK = 12
BROLL_SPREAD = "golden"


def common_args(
    *,
    crop: str,
    music: str,
    delogo: list[str] | tuple[str, ...] = (),
    tag: str = TAG,
    speed: float = SPEED,
    transition_sec: float = TRANSITION_SEC,
    talk_h: int = TALK_H,
    talk_zoom_max: float = TALK_ZOOM_MAX,
    talk_face_frac: float = TALK_FACE_FRAC,
    talk_face_y: float = TALK_FACE_Y,
    voice_pitch: float = VOICE_PITCH,
    peak_dbfs: float = PEAK_DBFS,
    broll_seed: int = 11,
    broll_chunk: int = BROLL_CHUNK,
    broll_spread: str = BROLL_SPREAD,
    captions: bool = False,
) -> list[str]:
    """The center-layout flag block shared by every project's batch.py.

    `delogo` is zero or more "W:H:X:Y" boxes in source-frame coordinates.
    `speed` defaults to the house pace (1.1); pass 1.0 to disable it for a
    job that doesn't need the extra pacing.
    """
    args: list[str] = [
        "--layout", "center",
        "--src-crop", crop,
        *[a for box in delogo for a in ("--src-delogo", box)],
        "--talk-h", str(talk_h),
        "--flip", "top",
        "--talk-zoom", "auto",
        "--talk-face-frac", str(talk_face_frac),
        "--talk-zoom-max", str(talk_zoom_max),
        "--talk-face-y", str(talk_face_y),
        "--voice-clarity",
        "--voice-pitch", str(voice_pitch),
        "--peak-dbfs", str(peak_dbfs),
        "--speed", str(speed),
        "--part-transition", "dissolve",
        "--part-transition-sec", str(transition_sec),
        "--music", music,
        "--music-duck", "off",
        "--music-compress",
        "--music-dip-db", str(MUSIC_DIP_DB),
        "--music-fade", str(MUSIC_FADE_SEC),
        "--music-window", "flattest",
        "--music-under-db", str(MUSIC_UNDER_DB),
        "--broll-bed",
        "--broll-seed", str(broll_seed),
        "--broll-chunk", str(broll_chunk),
        "--broll-spread", broll_spread,
    ]
    if not captions:
        args += ["--no-captions"]
    args += ["--channel-tag", tag]
    return args
