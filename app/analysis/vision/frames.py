"""Reading frames out of a video.

Every metric in this package needs pixels, and decoding video is by far the most
expensive thing AIVE does. So there is exactly one way to get frames, and it is built
around three decisions:

**Sample, never stream.** A three-minute clip at 30 fps is 5,400 frames. Blur and
exposure do not change meaningfully between neighbours, so the analyser seeks to a
handful of positions per scene instead of decoding everything.

**Read pairs, not singles.** Motion needs two consecutive frames. Reading a pair at
each probe point is what lets one pass produce stills *and* movement, instead of
decoding the file twice.

**Downscale immediately.** Classical CV on a 4K frame costs 30x what it costs on a
640-wide one, and the metrics are scale-invariant enough not to care. The one exception
is keyframes written to disk, which stay larger because a future vision model will want
the detail.

Frames come back display-oriented: ``CAP_PROP_ORIENTATION_AUTO`` is set explicitly so a
rotated phone clip is measured the way a viewer sees it, rather than sideways.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from app.analysis.vision.probe import VideoDependencyMissingError
from app.models.common import TimeRange
from app.utils.logging import get_logger

if TYPE_CHECKING:  # pragma: no cover
    import numpy as np

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class FramePair:
    """Two consecutive frames sampled at one probe point.

    ``first`` and ``second`` are at *analysis* scale, downscaled for cheap CV work.
    ``full`` is the same frame as ``first`` at its original resolution, kept so keyframes
    can be written with their detail intact - a future vision model wants pixels, and
    re-decoding the clip to get them would double the cost of the whole pipeline.

    ``second`` is ``None`` when the probe landed on the final frame of the file. Callers
    must treat that as "no motion information here" rather than as zero motion - a
    still frame at the end of a clip is not evidence the camera was locked off.
    """

    timestamp: float
    first: np.ndarray
    second: np.ndarray | None
    full: np.ndarray | None = None

    @property
    def has_pair(self) -> bool:
        return self.second is not None

    @property
    def still(self) -> np.ndarray:
        """The best available frame for writing a keyframe.

        Falls back to the analysis-scale frame when no full-resolution copy was kept, so
        a caller that constructed a pair by hand still gets an image rather than ``None``.
        """
        return self.first if self.full is None else self.full


class FrameReadError(RuntimeError):
    """Raised when a video cannot be opened for reading."""


@contextmanager
def open_capture(video: Path) -> Iterator[Any]:
    """Open ``video`` for reading, display-oriented, and always release it.

    A context manager because an unreleased ``VideoCapture`` holds a file handle open,
    which on Windows means the file cannot be moved or deleted - and a long analysis run
    would accumulate one per clip.
    """
    try:
        import cv2
    except ImportError as exc:
        raise VideoDependencyMissingError("opencv-python") from exc

    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        capture.release()
        msg = f"could not open {video.name} for reading; the codec may be unsupported"
        raise FrameReadError(msg)

    # Explicit rather than relying on the default: OpenCV's auto-rotation default has
    # changed between versions, and measuring a phone clip sideways is a silent, total
    # failure of every orientation-dependent metric.
    capture.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)
    try:
        yield capture
    finally:
        capture.release()


def downscale(frame: np.ndarray, target_width: int) -> np.ndarray:
    """Shrink a frame to ``target_width``, preserving aspect. Never upscales."""
    import cv2

    height, width = frame.shape[:2]
    if width <= target_width or width == 0:
        return frame
    scale = target_width / float(width)
    # INTER_AREA is the correct filter for shrinking; INTER_LINEAR aliases, which would
    # show up as spurious high-frequency detail and inflate the blur score.
    return cv2.resize(
        frame,
        (target_width, max(1, round(height * scale))),
        interpolation=cv2.INTER_AREA,
    )


def to_grayscale(frame: np.ndarray) -> np.ndarray:
    """Single-channel view of a frame, tolerating already-grey input."""
    import cv2

    if frame.ndim == 2:
        return frame
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


class FrameSampler:
    """Seeks to probe points in a video and returns downscaled frame pairs."""

    def __init__(self, *, analysis_width: int) -> None:
        self._analysis_width = analysis_width

    def sample_scene(
        self,
        video: Path,
        *,
        scene: TimeRange,
        count: int,
    ) -> list[FramePair]:
        """Read ``count`` frame pairs spread across ``scene``.

        Probe points are placed *inside* the scene rather than at its edges. A frame
        taken exactly on a cut belongs to neither shot and would contaminate both.
        """
        with open_capture(video) as capture:
            return [
                pair
                for timestamp in probe_timestamps(scene, count)
                if (pair := self._read_pair(capture, timestamp)) is not None
            ]

    def sample_frame(self, video: Path, *, timestamp: float) -> np.ndarray | None:
        """One frame at ``timestamp``, at full resolution.

        Full resolution because this is what writes keyframes, and a still is the one
        artefact where detail is the point.
        """
        with open_capture(video) as capture:
            return self._seek_and_read(capture, timestamp)

    # -- Internals ----------------------------------------------------------- #

    def _read_pair(self, capture: Any, timestamp: float) -> FramePair | None:
        first = self._seek_and_read(capture, timestamp)
        if first is None:
            return None
        # No seek before the second read: the capture is already positioned on the next
        # frame, and seeking again would land on a keyframe and lose the adjacency that
        # makes motion measurable.
        second = self._read(capture)
        return FramePair(
            timestamp=timestamp,
            first=downscale(first, self._analysis_width),
            second=None if second is None else downscale(second, self._analysis_width),
            # The undownscaled frame, for writing a keyframe. Only the first is kept: the
            # second exists solely to measure motion, so holding a 4K copy of it would be
            # megabytes per probe point for nothing.
            full=first,
        )

    @staticmethod
    def _seek_and_read(capture: Any, timestamp: float) -> np.ndarray | None:
        import cv2

        capture.set(cv2.CAP_PROP_POS_MSEC, max(0.0, timestamp) * 1000.0)
        return FrameSampler._read(capture)

    @staticmethod
    def _read(capture: Any) -> np.ndarray | None:
        ok, frame = capture.read()
        if not ok or frame is None or frame.size == 0:
            return None
        result: np.ndarray = frame
        return result


def probe_timestamps(scene: TimeRange, count: int) -> list[float]:
    """Evenly spaced sample points strictly inside ``scene``.

    For ``count=3`` the points sit at 25%, 50% and 75% of the scene rather than at 0%,
    50% and 100%. The endpoints are where a cut lives, and a frame from a cut is a blend
    of two shots - which reads as motion that never happened and blur that is not there.
    """
    if count <= 0 or scene.duration <= 0.0:
        return []
    step = scene.duration / (count + 1)
    return [scene.start + step * (index + 1) for index in range(count)]


__all__ = [
    "FramePair",
    "FrameReadError",
    "FrameSampler",
    "downscale",
    "open_capture",
    "probe_timestamps",
    "to_grayscale",
]
