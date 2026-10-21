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

As of 2026-10-13, the talk's height is **derived**, not chosen: 15% of the
1920-tall frame is reserved as a plain scene band at the very top and
another 15% at the very bottom (``CentreStyle.edge_margin_frac = 0.15``,
the default), the title/talk/tag group is ink-tight internally (no gap, no
plate padding -- that strip of scene showing through was the complaint on
2026-10-08) and sits centred as one block in the remaining 70%. For the
standard 2-line title / 1-line tag this works out to a ~1108px talk --
``TALK_H`` below is that number for logging/reference, but the actual value
build_shorts.py uses comes back from ``layout_center.geometry()`` per clip,
because a title that needs a smaller font to fit (``fit_title``) has a
shorter plate and so a taller derived talk. Passing ``--talk-h`` has no
effect while ``edge_margin_frac`` is set. Do not pass ``--text-edges``:
that's the other, unrelated way of deriving the talk height (pins title/tag
to the literal frame edges, no scene band at all) -- tried 2026-10-08,
rejected the next day.

This replaced two earlier, narrower attempts the same week: talk_h fixed at
1344 (70%) with scene bands as whatever was left over (title's leftover
shrank to near-zero and got clipped by YouTube's own UI in two different
views), then a one-sided push (``min_top_margin``) that fixed the top but
left the composition reading as bottom-heavy. Fixed, equal percentages on
both sides with the group centred as a whole is what was actually asked
for; don't reintroduce either earlier shape without being asked.

``--src-crop`` should match the derived talk's aspect (1080:1108, about
0.975:1 -- roughly 1053x1080 on a 1920x1080 source) for the standard title/
tag sizing above. A crop measured for an older target (most existing
per-project crops were measured at 1080:960 = 1.125) is not a perfect match
and gets cropped further to fill the band, which crops in on the face
slightly more than intended -- re-measure a project's crop if its framing
looks tighter than expected, rather than assuming the old numbers still
apply exactly.
"""

from __future__ import annotations

import loudness

TAG = "@loiphatdayquathayphaphoa"
TRANSITION_SEC = 0.5
# Reference only -- see the module docstring. The real value is derived per
# clip by layout_center.geometry() whenever CentreStyle.edge_margin_frac is
# set, which it is by default; --talk-h is passed but has no effect there.
TALK_H = 1108
TALK_ZOOM_MAX = 1.7  # as of 2026-10-07; was 1.4 and was clamping short of
# --talk-face-frac's own target on a normal (non-upscaled) 1280x720 source.
TALK_FACE_FRAC = 0.45
TALK_FACE_Y = 0.5
VOICE_PITCH = 0.94
PEAK_DBFS = -1.0
# Every short is brought to this integrated loudness, up or down (loudness.py).
# Fixed on the established close-up shorts, -15.3 LUFS; a clip from a distant
# camera used to land 1.5-2 dB lower because the old check only lowered.
LOUDNESS_TARGET = loudness.TARGET_LUFS
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
    loudness_target: float = LOUDNESS_TARGET,
    broll_seed: int = 11,
    broll_chunk: int = BROLL_CHUNK,
    broll_spread: str = BROLL_SPREAD,
    captions: bool = False,
    edge_margin_frac: float | None = None,
) -> list[str]:
    """The center-layout flag block shared by every project's batch.py.

    `delogo` is zero or more "W:H:X:Y" boxes in source-frame coordinates.
    `speed` defaults to the house pace (1.1); pass 1.0 to disable it for a
    job that doesn't need the extra pacing. `talk_h` is ignored by
    build_shorts.py unless `edge_margin_frac=0` is also passed (the derived-
    height behaviour is the default -- see this module's docstring); pass
    both together for a job that must keep the pre-2026-10-13 fixed-talk_h
    framing.
    """
    args: list[str] = [
        "--layout", "center",
        "--src-crop", crop,
        *[a for box in delogo for a in ("--src-delogo", box)],
        "--talk-h", str(talk_h),
        *(["--edge-margin-frac", str(edge_margin_frac)] if edge_margin_frac is not None else []),
        "--flip", "top",
        "--talk-zoom", "auto",
        "--talk-face-frac", str(talk_face_frac),
        "--talk-zoom-max", str(talk_zoom_max),
        "--talk-face-y", str(talk_face_y),
        "--voice-clarity",
        "--voice-pitch", str(voice_pitch),
        "--peak-dbfs", str(peak_dbfs),
        "--loudness-target", str(loudness_target),
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
