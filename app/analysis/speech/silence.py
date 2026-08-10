"""Silence detection using FFmpeg's ``silencedetect`` filter.

FFmpeg rather than pydub or librosa, for three reasons. It needs no dependency we do
not already guarantee — a static ffmpeg ships with the package. It never holds the
audio in Python memory, which matters because pydub would materialise a twenty-minute
narration as a Python array. And it is the same binary that will later perform the
cuts, so the levels it measures are the levels that get cut on.

The cost is that ``silencedetect`` reports only *that* a span fell below the
threshold, never how far below, so :attr:`~app.models.speech.SilenceSpan.mean_db` is
left as ``None``. Measuring it would need one extra decode pass per span, which is not
worth a number nothing consumes.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

from app.models.common import TimeRange
from app.models.speech import SilenceSpan
from app.services.ffmpeg_locator import FFmpegLocator
from app.utils.logging import get_logger

logger = get_logger(__name__)

DETECTOR_NAME = "ffmpeg-silencedetect"

_SILENCE_START = re.compile(r"silence_start:\s*(-?[\d.]+)")
_SILENCE_END = re.compile(r"silence_end:\s*(-?[\d.]+)")
"""``silencedetect`` writes its findings to stderr as human-readable lines:

    [silencedetect @ 0x...] silence_start: 4.20015
    [silencedetect @ 0x...] silence_end: 5.10842 | silence_duration: 0.908265

There is no machine-readable alternative, so these patterns are the interface. They
are deliberately loose about surrounding text so a change to the log prefix or the
addition of a field does not break parsing.
"""

_ANALYSIS_TIMEOUT = 900.0
"""Seconds before giving up. Generous: this decodes the whole file, and an hour of
narration on a slow disk is legitimately slow. Present only so a wedged process
cannot hang the pipeline forever."""


class SilenceDetectionError(RuntimeError):
    """Raised when FFmpeg could not analyse the audio at all."""


class FFmpegSilenceDetector:
    """A :class:`~app.analysis.speech.base.SilenceDetector` driven by FFmpeg."""

    def __init__(self, locator: FFmpegLocator) -> None:
        self._locator = locator

    @property
    def name(self) -> str:
        return DETECTOR_NAME

    def detect(
        self,
        audio: Path,
        *,
        threshold_db: float,
        min_duration: float,
    ) -> tuple[SilenceSpan, ...]:
        """Return quiet spans in ``audio``, ascending and non-overlapping."""
        if not audio.is_file():
            msg = f"audio file not found: {audio}"
            raise FileNotFoundError(msg)

        ffmpeg = self._locator.locate().ffmpeg
        command = [
            str(ffmpeg.path),
            "-hide_banner",
            "-nostats",
            "-i",
            str(audio),
            "-af",
            f"silencedetect=noise={threshold_db}dB:d={min_duration}",
            # Decode and measure, but write no output file.
            "-f",
            "null",
            "-",
        ]

        logger.debug("Running silencedetect: threshold=%sdB min=%ss", threshold_db, min_duration)
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=_ANALYSIS_TIMEOUT,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            msg = f"failed to run ffmpeg for silence detection: {exc}"
            raise SilenceDetectionError(msg) from exc

        if completed.returncode != 0:
            tail = (completed.stderr or "").strip().splitlines()[-3:]
            msg = "ffmpeg failed during silence detection: " + " | ".join(tail)
            raise SilenceDetectionError(msg)

        return parse_silencedetect(completed.stderr or "")


def parse_silencedetect(stderr: str) -> tuple[SilenceSpan, ...]:
    """Parse ``silencedetect`` output into spans.

    A module-level function rather than a method so it can be tested against captured
    FFmpeg output with no binary present.

    Handles the two ragged edges of the real format: a ``silence_start`` with no
    matching ``silence_end`` when the file ends mid-silence, and a negative start,
    which FFmpeg can emit by a rounding hair at the very beginning of a file.
    """
    spans: list[SilenceSpan] = []
    pending_start: float | None = None

    for line in stderr.splitlines():
        start_match = _SILENCE_START.search(line)
        if start_match is not None:
            pending_start = max(0.0, float(start_match.group(1)))
            # Fall through: FFmpeg can put a start and an end on one line.
        end_match = _SILENCE_END.search(line)
        if end_match is not None and pending_start is not None:
            end = float(end_match.group(1))
            if end > pending_start:
                spans.append(SilenceSpan(range=TimeRange(start=pending_start, end=end)))
            pending_start = None

    if pending_start is not None:
        # The file ended inside a silence, so FFmpeg never printed an end. The caller
        # knows the true duration and can extend the final span; reporting it here
        # with a nominal end would be inventing data.
        logger.debug("Audio ends inside a silence beginning at %.3fs", pending_start)

    return tuple(spans)


def trailing_silence_start(stderr: str) -> float | None:
    """The start of an unterminated final silence, if the file ended inside one.

    Exposed separately so a caller holding the real duration can close the span,
    rather than having the parser guess at an end it cannot know.
    """
    pending: float | None = None
    for line in stderr.splitlines():
        if _SILENCE_START.search(line) is not None:
            pending = max(0.0, float(_SILENCE_START.search(line).group(1)))  # type: ignore[union-attr]
        if _SILENCE_END.search(line) is not None:
            pending = None
    return pending


__all__ = [
    "DETECTOR_NAME",
    "FFmpegSilenceDetector",
    "SilenceDetectionError",
    "parse_silencedetect",
    "trailing_silence_start",
]
