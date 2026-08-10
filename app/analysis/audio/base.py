"""Audio analysis boundaries.

Two protocols, split because they are answered by different machinery and fail
independently.

:class:`LoudnessMeter` measures integrated loudness. It delegates to FFmpeg's
``ebur128`` filter, which *is* the EBU R128 reference implementation — reimplementing a
K-weighted gated loudness meter in numpy would be inventing a worse version of something
already on disk. Loudness matters because ducking only works if music and narration are
measured in the same units.

:class:`MusicAnalyzer` describes a track well enough for the director to choose one
deliberately rather than alphabetically: tempo, energy, brightness, mood, and where the
intro stops.

Both are protocols so a future implementation — a librosa beat tracker, or a model that
actually listens — drops in without touching a caller.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from app.models.audio import MusicTrack
from app.models.common import MediaRef


class AudioDependencyMissingError(RuntimeError):
    """Raised when an optional audio dependency is not installed.

    Distinct from a decode failure: the fix is `pip install`, not a different file, and
    the CLI maps it to exit 5 (environment) rather than 4 (bad input).
    """


class AudioDecodeError(RuntimeError):
    """Raised when a file cannot be decoded. The file is the problem, not the install."""


@runtime_checkable
class LoudnessMeter(Protocol):
    """Measures integrated loudness."""

    def measure(self, path: Path) -> float | None:
        """Integrated loudness in LUFS, or ``None`` when it could not be measured.

        ``None`` rather than an exception: a track whose loudness is unknown is still
        usable, it just cannot be level-matched automatically.
        """
        ...


@runtime_checkable
class MusicAnalyzer(Protocol):
    """Describes one music file."""

    @property
    def version(self) -> str:
        """Analyser version, stamped into results so a stale cache is invalidated."""
        ...

    def analyze(self, path: Path, *, ref: MediaRef) -> MusicTrack:
        """Analyse one track.

        Raises:
            AudioDecodeError: when the file cannot be read.
            AudioDependencyMissingError: when a required package is absent.
        """
        ...


__all__ = [
    "AudioDecodeError",
    "AudioDependencyMissingError",
    "LoudnessMeter",
    "MusicAnalyzer",
]
