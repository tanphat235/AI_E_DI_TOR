"""Reading FFmpeg's progress stream.

``-progress`` emits a machine-readable block of ``key=value`` lines terminated by
``progress=continue`` (or ``progress=end``), repeated every few hundred milliseconds. It
exists precisely so a caller does not have to scrape the human status line, which changes
between builds and interleaves carriage returns.

Sent to the subprocess's **stdout** so its stderr stays a clean log. That is a separate
channel from AIVE's own stdout contract — this is FFmpeg's stdout, which we consume — but
the reasoning is the same one, applied one process down.

Progress is reported as a fraction of the *expected* duration, and the fraction is clamped
below 1.0 until the process actually exits. FFmpeg's ``out_time`` can overshoot slightly on
the final flush, and a bar that reads 100% while the encoder is still working is worse than
one that sits at 99%.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass

PROGRESS_VERSION = "ffmpeg-progress/1"

_CEILING = 0.999
"""Cap on reported progress before completion; see the module docstring."""


@dataclass(frozen=True, slots=True)
class ProgressUpdate:
    """One progress block from FFmpeg."""

    out_time: float
    """Position in the output timeline, in seconds."""
    frame: int
    fps: float
    speed: float
    """Encoding rate relative to real time. ``2.0`` means twice as fast as playback."""
    finished: bool

    def fraction(self, total: float) -> float:
        """Completion as a fraction of ``total`` seconds."""
        if self.finished:
            return 1.0
        if total <= 0.0:
            return 0.0
        return min(_CEILING, max(0.0, self.out_time / total))

    def eta(self, total: float) -> float | None:
        """Seconds remaining, or ``None`` when the speed is not yet known.

        Derived from FFmpeg's own ``speed`` rather than from elapsed wall time, so a render
        that starts slowly while the OS caches the first file does not report a wildly
        pessimistic estimate for the rest of its run.
        """
        if self.finished or self.speed <= 0.0 or total <= 0.0:
            return None
        remaining = max(0.0, total - self.out_time)
        return remaining / self.speed


def parse_progress(lines: Iterable[str]) -> Iterator[ProgressUpdate]:
    """Yield one :class:`ProgressUpdate` per completed block.

    A generator over lines rather than a function over a whole string, because the caller is
    reading a live pipe. Malformed and unrecognised keys are skipped rather than raised on:
    FFmpeg adds fields between versions, and a render must not fail because its progress
    report grew a column.
    """
    block: dict[str, str] = {}
    for raw in lines:
        line = raw.strip()
        if not line or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip()

        if key != "progress":
            block[key] = value
            continue

        yield ProgressUpdate(
            out_time=_seconds(block),
            frame=_as_int(block.get("frame")),
            fps=_as_float(block.get("fps")),
            speed=_speed(block.get("speed")),
            finished=value == "end",
        )
        block = {}


def _seconds(block: dict[str, str]) -> float:
    """Output position in seconds, from whichever field this build provides.

    ``out_time_us`` is preferred: it is an integer and needs no timecode parsing.
    ``out_time_ms`` is, despite its name, also microseconds in every FFmpeg release that
    emits it - a long-standing quirk, and reading it as milliseconds makes a render appear
    to finish a thousand times over.
    """
    for key in ("out_time_us", "out_time_ms"):
        raw = block.get(key)
        if raw and raw.lstrip("-").isdigit():
            return max(0.0, int(raw) / 1_000_000.0)

    timecode = block.get("out_time")
    if not timecode:
        return 0.0
    try:
        hours, minutes, seconds = timecode.split(":")
        return max(0.0, int(hours) * 3600 + int(minutes) * 60 + float(seconds))
    except (ValueError, AttributeError):
        return 0.0


def _as_int(raw: str | None) -> int:
    if raw is None:
        return 0
    try:
        return int(raw)
    except ValueError:
        return 0


def _as_float(raw: str | None) -> float:
    if raw is None:
        return 0.0
    try:
        return float(raw)
    except ValueError:
        return 0.0


def _speed(raw: str | None) -> float:
    """Parse the ``speed`` field, which carries a trailing ``x`` and may be ``N/A``."""
    if raw is None:
        return 0.0
    return _as_float(raw.rstrip("xX").strip())


__all__ = ["PROGRESS_VERSION", "ProgressUpdate", "parse_progress"]
