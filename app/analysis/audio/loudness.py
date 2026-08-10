"""Integrated loudness measurement, via FFmpeg.

Loudness is the one audio number that must be right rather than approximately right:
ducking is a comparison between two levels, and if music and narration are measured by
different rulers the bed sits wrong under every line.

So this does not measure anything itself. It runs FFmpeg's ``loudnorm`` filter in analysis
mode, which is a conforming EBU R128 / ITU-R BS.1770 meter — K-weighted, gated, the actual
standard. A numpy reimplementation would be a worse copy of a tool already on disk, and
its errors would be invisible because there would be nothing to compare against.

``print_format=json`` rather than parsing the human summary: the JSON block is a stable
machine interface, whereas the ``ebur128`` text summary has changed layout between FFmpeg
releases.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

from app.services.ffmpeg_locator import FFmpegLocator, FFmpegNotFoundError
from app.utils.logging import get_logger

logger = get_logger(__name__)

LOUDNESS_VERSION = "loudnorm/1"

_JSON_BLOCK = re.compile(r"\{[^{}]*\"input_i\"[^{}]*\}", re.DOTALL)
"""The loudnorm report. Matched by content rather than position because FFmpeg writes it
among however many other log lines the build feels like emitting."""

_MEASUREMENT_TIMEOUT = 300.0
"""Five minutes. Analysis-mode loudnorm decodes the whole file but encodes nothing, so
even a long track is fast; a run that exceeds this has hung rather than got busy."""

MINIMUM_MEASURABLE_DURATION = 3.0
"""Integrated loudness is gated over 400 ms blocks with a 3 s relative window. Below this
the gate never settles and the reported value is not meaningful."""


class FFmpegLoudnessMeter:
    """Measures integrated loudness with FFmpeg's ``loudnorm`` filter."""

    def __init__(self, locator: FFmpegLocator) -> None:
        self._locator = locator

    @property
    def version(self) -> str:
        return LOUDNESS_VERSION

    def measure(self, path: Path) -> float | None:
        """Integrated loudness in LUFS, or ``None`` when it could not be measured.

        Never raises. A track whose loudness is unknown is still perfectly usable — it
        just cannot be level-matched automatically — so a measurement failure degrades the
        description rather than failing the analysis of an otherwise fine file.
        """
        try:
            tools = self._locator.locate()
        except FFmpegNotFoundError:
            logger.warning("No ffmpeg available, so loudness cannot be measured")
            return None

        command = [
            str(tools.ffmpeg.path),
            "-nostdin",
            "-hide_banner",
            "-i",
            str(path),
            "-map",
            "a:0",
            "-af",
            "loudnorm=print_format=json",
            "-f",
            "null",
            "-",
        ]

        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=_MEASUREMENT_TIMEOUT,
                check=False,
                encoding="utf-8",
                errors="replace",
            )
        except (OSError, subprocess.SubprocessError) as exc:
            logger.warning("Loudness measurement failed for %s: %s", path.name, exc)
            return None

        if completed.returncode != 0:
            logger.warning("ffmpeg exited %d measuring %s", completed.returncode, path.name)
            return None

        return parse_loudnorm(completed.stderr)


def parse_loudnorm(output: str) -> float | None:
    """Extract ``input_i`` from a loudnorm JSON report.

    Split out as a plain function so it can be tested against captured FFmpeg output
    without running FFmpeg — which is the only way to have a regression test for a log
    format we do not control.
    """
    match = _JSON_BLOCK.search(output)
    if match is None:
        logger.debug("No loudnorm JSON block in ffmpeg output")
        return None

    try:
        report = json.loads(match.group(0))
    except json.JSONDecodeError:
        logger.debug("loudnorm block was not valid JSON")
        return None

    raw = report.get("input_i")
    if raw is None:
        return None

    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None

    # Digital silence measures as -inf, and -70 is the R128 absolute gate. Either way
    # there is no programme material to level-match against.
    if value != value or value in {float("-inf"), float("inf")} or value <= -70.0:
        logger.debug("loudnorm reported %s, treating as unmeasurable", raw)
        return None
    return round(value, 2)


__all__ = [
    "LOUDNESS_VERSION",
    "MINIMUM_MEASURABLE_DURATION",
    "FFmpegLoudnessMeter",
    "parse_loudnorm",
]
