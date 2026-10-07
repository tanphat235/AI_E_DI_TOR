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
"""

from __future__ import annotations

TAG = "@loiphatdayquathayphaphoa"
TRANSITION_SEC = 0.5
TALK_H = 960
TALK_ZOOM_MAX = 1.7  # as of 2026-10-07; was 1.4 and was clamping short of
# --talk-face-frac's own target on a normal (non-upscaled) 1280x720 source.
TALK_FACE_FRAC = 0.45
TALK_FACE_Y = 0.5
VOICE_PITCH = 0.94
PEAK_DBFS = -1.0
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
    speed: float | None = None,
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
    `speed` is omitted (not passed) unless a job overrides it from 1.0.
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
    ]
    if speed is not None:
        args += ["--speed", str(speed)]
    args += [
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
