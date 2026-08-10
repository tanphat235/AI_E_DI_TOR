"""Finding the FFmpeg binaries.

AIVE must work on a machine with no FFmpeg installed, because that is the common
case. A static ``ffmpeg`` build ships in the ``imageio-ffmpeg`` wheel and is used
as the last resort.

There is one wrinkle worth knowing about: **the vendored wheel contains ffmpeg but
not ffprobe.** So ``ffprobe`` is resolved separately and is allowed to be absent.
Nothing in Phase 1 needs it; Phase 3's media probing will prefer PyAV, which
arrives as a dependency of ``faster-whisper`` and reads container metadata in
process, and fall back to a system ``ffprobe`` when one is present. Pretending a
vendored ffprobe exists would just move the failure to render time.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from app.config.settings import MediaSettings


class FFmpegNotFoundError(RuntimeError):
    """Raised when no usable ``ffmpeg`` binary could be located."""


class BinarySource(StrEnum):
    """Where a resolved binary came from. Reported by ``aive doctor``."""

    CONFIGURED = "configured"
    """An explicit path from settings."""

    SYSTEM_PATH = "system_path"
    """Found on ``PATH``. Preferred over the vendored build: a full system build
    usually carries more codecs and filters than the essentials build we ship."""

    VENDORED = "vendored"
    """The static binary inside the ``imageio-ffmpeg`` wheel."""

    SIBLING = "sibling"
    """Found next to an already-resolved binary, which is how a manually
    installed FFmpeg keeps ffmpeg and ffprobe together."""


@dataclass(frozen=True, slots=True)
class ResolvedBinary:
    """A located executable and how it was found."""

    path: Path
    source: BinarySource

    def version(self, *, timeout: float = 15.0) -> str:
        """First line of ``<binary> -version``, or a diagnostic on failure.

        Never raises. This is called by ``doctor``, whose whole purpose is to
        report a broken environment rather than crash inside it.
        """
        try:
            completed = subprocess.run(
                [str(self.path), "-version"],
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            return f"unavailable ({type(exc).__name__}: {exc})"
        if completed.returncode != 0:
            return f"exit {completed.returncode}"
        first_line = completed.stdout.strip().splitlines()
        return first_line[0] if first_line else "unknown"


@dataclass(frozen=True, slots=True)
class FFmpegTools:
    """The FFmpeg binaries available to this process.

    ``ffprobe`` is optional by design; see the module docstring.
    """

    ffmpeg: ResolvedBinary
    ffprobe: ResolvedBinary | None

    @property
    def has_ffprobe(self) -> bool:
        return self.ffprobe is not None


class FFmpegLocator:
    """Resolves FFmpeg binaries according to configuration.

    A class rather than a function so it can be injected and faked. Tests should
    never depend on what happens to be installed on the developer's machine.
    """

    def __init__(self, settings: MediaSettings) -> None:
        self._settings = settings

    def locate(self) -> FFmpegTools:
        """Resolve both binaries.

        Raises:
            FFmpegNotFoundError: if no ``ffmpeg`` can be found at all.
        """
        return FFmpegTools(ffmpeg=self._locate_ffmpeg(), ffprobe=self._locate_ffprobe())

    # -- ffmpeg ------------------------------------------------------------- #

    def _locate_ffmpeg(self) -> ResolvedBinary:
        configured = self._settings.ffmpeg_path
        if configured is not None:
            if not configured.is_file():
                msg = (
                    f"configured ffmpeg_path does not exist: {configured}\n"
                    "Fix media.ffmpeg_path in your config, or clear it to auto-detect."
                )
                raise FFmpegNotFoundError(msg)
            return ResolvedBinary(path=configured, source=BinarySource.CONFIGURED)

        on_path = shutil.which("ffmpeg")
        if on_path is not None:
            return ResolvedBinary(path=Path(on_path), source=BinarySource.SYSTEM_PATH)

        vendored = self._vendored_ffmpeg()
        if vendored is not None:
            return ResolvedBinary(path=vendored, source=BinarySource.VENDORED)

        msg = (
            "No ffmpeg binary found.\n"
            "AIVE looks in three places, in order:\n"
            "  1. media.ffmpeg_path in your config\n"
            "  2. 'ffmpeg' on PATH\n"
            "  3. the static build vendored by imageio-ffmpeg\n"
            "All three failed. Install the vendored build with:\n"
            "  pip install imageio-ffmpeg\n"
            "or install FFmpeg system-wide with:\n"
            "  winget install Gyan.FFmpeg"
        )
        raise FFmpegNotFoundError(msg)

    @staticmethod
    def _vendored_ffmpeg() -> Path | None:
        """The ``imageio-ffmpeg`` binary, if that package is installed and usable."""
        try:
            import imageio_ffmpeg
        except ImportError:
            return None
        try:
            candidate = Path(imageio_ffmpeg.get_ffmpeg_exe())
        except Exception:
            return None
        return candidate if candidate.is_file() else None

    # -- ffprobe ------------------------------------------------------------ #

    def _locate_ffprobe(self) -> ResolvedBinary | None:
        configured = self._settings.ffprobe_path
        if configured is not None:
            if not configured.is_file():
                msg = (
                    f"configured ffprobe_path does not exist: {configured}\n"
                    "Fix media.ffprobe_path in your config, or clear it to auto-detect."
                )
                raise FFmpegNotFoundError(msg)
            return ResolvedBinary(path=configured, source=BinarySource.CONFIGURED)

        on_path = shutil.which("ffprobe")
        if on_path is not None:
            return ResolvedBinary(path=Path(on_path), source=BinarySource.SYSTEM_PATH)

        # A hand-installed FFmpeg keeps both binaries in one folder, so if ffmpeg
        # came from an explicit path or from PATH, look beside it.
        try:
            ffmpeg = self._locate_ffmpeg()
        except FFmpegNotFoundError:
            return None
        if ffmpeg.source is BinarySource.VENDORED:
            # The wheel ships ffmpeg alone; there is nothing to find next to it.
            return None
        for name in ("ffprobe.exe", "ffprobe"):
            sibling = ffmpeg.path.parent / name
            if sibling.is_file():
                return ResolvedBinary(path=sibling, source=BinarySource.SIBLING)
        return None


__all__ = [
    "BinarySource",
    "FFmpegLocator",
    "FFmpegNotFoundError",
    "FFmpegTools",
    "ResolvedBinary",
]
