"""Generating real test video with the vendored FFmpeg.

Integration tests need genuine H.264 files, and committing binaries to the repo is worse
than making them: a generated file is reproducible, reviewable as a filter graph, and
carries *known* properties, so a test can assert that a dark clip reads as dark rather
than merely that the code ran.

Every clip is built to exercise one property in isolation. Imported by ``conftest.py``.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

VIDEO_SIZE = "640x360"
FRAME_RATE = 25


@dataclass(frozen=True, slots=True)
class GeneratedClip:
    """A test clip and what it is supposed to demonstrate."""

    name: str
    filters: list[str]
    duration: float
    purpose: str


CLIPS: tuple[GeneratedClip, ...] = (
    GeneratedClip(
        name="three_scenes.mp4",
        filters=[
            "-f",
            "lavfi",
            "-i",
            f"color=c=red:s={VIDEO_SIZE}:d=2:r={FRAME_RATE}",
            "-f",
            "lavfi",
            "-i",
            f"color=c=green:s={VIDEO_SIZE}:d=2:r={FRAME_RATE}",
            "-f",
            "lavfi",
            "-i",
            f"testsrc=s={VIDEO_SIZE}:d=2:r={FRAME_RATE}",
            "-filter_complex",
            "[0][1][2]concat=n=3:v=1:a=0",
        ],
        duration=6.0,
        purpose="three visually distinct scenes, so cut detection must find two cuts",
    ),
    GeneratedClip(
        name="sharp.mp4",
        filters=["-f", "lavfi", "-i", f"testsrc2=s={VIDEO_SIZE}:d=2:r={FRAME_RATE}"],
        duration=2.0,
        purpose="high detail, so blur must score high",
    ),
    GeneratedClip(
        name="blurred.mp4",
        filters=[
            "-f",
            "lavfi",
            "-i",
            f"testsrc2=s={VIDEO_SIZE}:d=2:r={FRAME_RATE}",
            "-vf",
            "boxblur=12:2",
        ],
        duration=2.0,
        purpose="the same source heavily blurred, so blur must score far lower",
    ),
    GeneratedClip(
        name="dark.mp4",
        filters=["-f", "lavfi", "-i", f"color=c=0x0a0a0a:s={VIDEO_SIZE}:d=2:r={FRAME_RATE}"],
        duration=2.0,
        purpose="near-black, so brightness must score low",
    ),
    GeneratedClip(
        name="panning.mp4",
        filters=[
            "-f",
            "lavfi",
            "-i",
            f"testsrc2=s=1280x360:d=3:r={FRAME_RATE}",
            # Crop a moving window out of a wider source: a genuine horizontal pan.
            "-vf",
            "crop=640:360:x='min(iw-ow,t*180)':y=0",
        ],
        duration=3.0,
        purpose="a real horizontal pan, so camera_move must read as pan",
    ),
    GeneratedClip(
        name="red_again.mp4",
        filters=["-f", "lavfi", "-i", f"color=c=red:s={VIDEO_SIZE}:d=2:r={FRAME_RATE}"],
        duration=2.0,
        purpose="identical to the first scene of three_scenes, so it must be a duplicate",
    ),
)


def ffmpeg_binary() -> Path:
    """The FFmpeg AIVE itself would use."""
    from app.config.settings import MediaSettings
    from app.services.ffmpeg_locator import FFmpegLocator

    return FFmpegLocator(MediaSettings()).locate().ffmpeg.path


def generate_clips(destination: Path) -> dict[str, Path]:
    """Build every test clip into ``destination``, skipping any that already exist.

    Skipping matters: these are session-scoped, and re-encoding six clips for every test
    module would dominate the suite's runtime.
    """
    destination.mkdir(parents=True, exist_ok=True)
    ffmpeg = ffmpeg_binary()
    produced: dict[str, Path] = {}

    for clip in CLIPS:
        target = destination / clip.name
        if not target.is_file():
            subprocess.run(
                [
                    str(ffmpeg),
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-y",
                    *clip.filters,
                    "-pix_fmt",
                    "yuv420p",
                    str(target),
                ],
                check=True,
                capture_output=True,
                timeout=180,
            )
        produced[clip.name] = target

    # A clip carrying a real display-rotation flag, which needs a second pass: setting
    # the flag on the input re-encodes and bakes the rotation into the pixels instead,
    # so the flag has to be applied while copying the stream.
    rotated = destination / "rotated.mp4"
    if not rotated.is_file():
        subprocess.run(
            [
                str(ffmpeg),
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-display_rotation",
                "90",
                "-i",
                str(produced["sharp.mp4"]),
                "-c",
                "copy",
                str(rotated),
            ],
            check=True,
            capture_output=True,
            timeout=180,
        )
    produced["rotated.mp4"] = rotated
    return produced


__all__ = ["CLIPS", "GeneratedClip", "ffmpeg_binary", "generate_clips"]
