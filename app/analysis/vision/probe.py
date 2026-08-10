"""Reading container metadata.

**PyAV, not ffprobe.** The vendored FFmpeg wheel ships ``ffmpeg`` alone, so on a clean
install there is no ``ffprobe`` to shell out to. PyAV arrives as a ``faster-whisper``
dependency, reads the same libraries in process, and needs no subprocess at all.

Rotation is the awkward part, and it matters: phone footage is routinely stored
landscape with a 90-degree display flag, and analysis that measures the encoded
orientation measures the wrong thing. Three approaches were tried:

* ``stream.metadata["rotate"]`` - the legacy tag. Modern FFmpeg no longer writes it.
* Comparing ``display_aspect_ratio`` to the encoded aspect - **unreliable.** A file
  with a genuine 270-degree flag still reports a 16:9 DAR, so this silently misses it.
* ``cv2.CAP_PROP_ORIENTATION_META`` - reads the display matrix correctly, and comes
  from the same library that will decode the frames.

The third wins. It also means OpenCV hands back display-oriented frames, so every
downstream metric measures the video as a viewer sees it.
"""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path
from typing import Any

from app.models.common import MediaRef
from app.models.media import AudioStreamInfo, MediaProbe, VideoStreamInfo
from app.utils.logging import get_logger

logger = get_logger(__name__)

PROBER_NAME = "pyav"


class ProbeError(RuntimeError):
    """Raised when a media file cannot be read at all."""


class VideoDependencyMissingError(RuntimeError):
    """Raised when the video analysis dependencies are not installed."""

    def __init__(self, package: str) -> None:
        super().__init__(
            f"{package} is not installed, so video analysis is unavailable.\n"
            'Install it with: pip install -e ".[video]"'
        )


class PyAvProber:
    """A :class:`~app.analysis.vision.base.MediaProber` backed by PyAV."""

    @property
    def name(self) -> str:
        return PROBER_NAME

    def probe(self, media: Path, *, ref: MediaRef) -> MediaProbe:
        """Read container and stream metadata for ``media``."""
        if not media.is_file():
            msg = f"media file not found: {media}"
            raise FileNotFoundError(msg)

        try:
            import av
        except ImportError as exc:  # pragma: no cover - av ships with faster-whisper
            raise VideoDependencyMissingError("PyAV") from exc

        try:
            with av.open(str(media)) as container:
                video_stream = container.streams.video[0] if container.streams.video else None
                audio_stream = container.streams.audio[0] if container.streams.audio else None
                duration = _container_duration(container, video_stream)
                video = self._video_info(video_stream, media) if video_stream is not None else None
                audio = self._audio_info(audio_stream) if audio_stream is not None else None
                format_name = getattr(container.format, "name", "") or ""
        except FileNotFoundError:
            raise
        except Exception as exc:
            msg = f"could not read {media.name}: {type(exc).__name__}: {exc}"
            raise ProbeError(msg) from exc

        if duration <= 0.0:
            msg = f"{media.name} reports no duration; it may be truncated or empty"
            raise ProbeError(msg)

        return MediaProbe(
            source=ref,
            duration=duration,
            size_bytes=media.stat().st_size,
            format_name=format_name,
            video=video,
            audio=audio,
        )

    # -- Streams ------------------------------------------------------------- #

    def _video_info(self, stream: Any, media: Path) -> VideoStreamInfo:
        context = stream.codec_context
        width = int(context.width)
        height = int(context.height)
        if width <= 0 or height <= 0:
            msg = f"{media.name} reports a {width}x{height} video stream"
            raise ProbeError(msg)

        return VideoStreamInfo(
            width=width,
            height=height,
            fps=_frame_rate(stream),
            codec=str(context.name or "unknown"),
            pixel_format=str(context.pix_fmt) if context.pix_fmt else None,
            bit_rate=_positive_int(getattr(context, "bit_rate", None)),
            rotation=read_rotation(media),
        )

    @staticmethod
    def _audio_info(stream: Any) -> AudioStreamInfo:
        context = stream.codec_context
        return AudioStreamInfo(
            codec=str(context.name or "unknown"),
            sample_rate=int(context.sample_rate or 48000),
            channels=int(getattr(context, "channels", None) or _layout_channels(context) or 1),
            bit_rate=_positive_int(getattr(context, "bit_rate", None)),
        )


def read_rotation(media: Path) -> int:
    """Display rotation in degrees, from OpenCV's view of the display matrix.

    Returns 0 when there is no rotation or OpenCV cannot say. Normalised into
    ``[0, 360)`` because the matrix can legitimately yield a negative angle.

    Asked of OpenCV rather than PyAV because PyAV 18 exposes only *setters* for the
    display matrix, and the obvious alternative - inferring from the aspect ratio - is
    demonstrably wrong on real files.
    """
    try:
        import cv2
    except ImportError:
        return 0

    capture = cv2.VideoCapture(str(media))
    try:
        if not capture.isOpened():
            return 0
        raw = capture.get(cv2.CAP_PROP_ORIENTATION_META)
    finally:
        capture.release()

    if not raw or raw != raw:
        return 0
    return round(float(raw)) % 360


def _container_duration(container: Any, video_stream: Any) -> float:
    """Duration in seconds, preferring the container's own figure.

    The container is authoritative when present. Falling back to the stream matters for
    formats that carry a duration only per stream, and computing from frame count is a
    last resort because it is wrong for variable-frame-rate footage - which is what
    phones produce.
    """
    if container.duration:
        return float(container.duration) / 1_000_000.0  # PyAV reports microseconds

    if video_stream is not None:
        if video_stream.duration and video_stream.time_base:
            return float(video_stream.duration * video_stream.time_base)
        frames = getattr(video_stream, "frames", 0) or 0
        if frames > 0:
            rate = _frame_rate(video_stream)
            if rate > 0:
                return float(frames) / rate
    return 0.0


def _frame_rate(stream: Any) -> float:
    """Average frame rate, falling back through PyAV's several notions of it.

    ``average_rate`` is the honest figure for a whole file. ``guessed_rate`` is
    FFmpeg's estimate when the container does not say, and ``base_rate`` is the
    container's nominal rate, which for VFR footage overstates reality.
    """
    for attribute in ("average_rate", "guessed_rate", "base_rate"):
        value = getattr(stream, attribute, None)
        if isinstance(value, Fraction) and value.denominator and value > 0:
            return float(value)
        if isinstance(value, int | float) and value and value > 0:
            return float(value)
    return 0.0


def _layout_channels(context: Any) -> int | None:
    """Channel count from the audio layout, for PyAV versions without ``channels``."""
    layout = getattr(context, "layout", None)
    channels = getattr(layout, "channels", None)
    if channels is None:
        return None
    try:
        return len(channels)
    except TypeError:
        return int(channels) or None


def _positive_int(value: object) -> int | None:
    """Coerce to a positive int, or ``None``. Bit rate is often reported as 0."""
    if value is None:
        return None
    try:
        number = int(value)  # type: ignore[call-overload]
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


__all__ = [
    "PROBER_NAME",
    "ProbeError",
    "PyAvProber",
    "VideoDependencyMissingError",
    "read_rotation",
]
