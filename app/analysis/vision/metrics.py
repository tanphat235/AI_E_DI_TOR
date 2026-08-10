"""Technical quality and motion measurement.

Every score leaving this module is normalised so that **1.0 is better than 0.0**,
whatever the natural direction of the underlying measurement. Blur is the trap: the raw
Laplacian variance rises with sharpness, so it is inverted and normalised here rather
than leaving each consumer to remember which way round it goes.

What is honest about these numbers, and what is not:

* **Brightness and exposure are absolute.** A histogram is a histogram; a clipped
  highlight is clipped on any machine.
* **Blur is comparative.** Laplacian variance depends on scene content as much as on
  focus - a perfectly sharp photograph of a white wall has almost no variance. So the
  score is meaningful for ranking shots *within a project* and close to meaningless as
  an absolute. The raw variance is kept on the model so the normalisation can be
  retuned without re-decoding anything.
* **Motion is measured at analysis scale**, in pixels per frame of the downscaled
  frame. Changing ``vision.downscale_width`` changes the numbers, which is why the
  thresholds live beside it in config.

Camera movement is classified from where the tracked points go, not from how far. A pan
and a handheld wobble can have identical magnitude; what separates them is whether the
points agree on a direction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple

from app.analysis.vision.frames import FramePair, to_grayscale
from app.config.settings import VisionSettings
from app.models.common import CameraMove, MotionLevel
from app.models.video import MotionStats, QualityScores
from app.utils.logging import get_logger

if TYPE_CHECKING:  # pragma: no cover
    import numpy as np

logger = get_logger(__name__)

METRICS_VERSION = "metrics/1"

_QUALITY_WEIGHTS: dict[str, float] = {
    "blur": 0.40,
    "brightness": 0.20,
    "exposure": 0.15,
    "stability": 0.25,
}
"""How the aggregate score is composed.

Blur dominates because a soft shot is unusable while a slightly dark one is gradeable.
Stability is next: shake cannot be fixed in post. Brightness and exposure matter least
precisely because they *can* be corrected later.
"""

_CLIP_LIMIT = 0.02
"""Fraction of pixels at the extremes before exposure is considered clipped.

Two percent, because a specular highlight or a genuinely black shadow is normal and
should not be penalised; a fifth of the frame blown out is not.
"""


@dataclass(frozen=True, slots=True)
class FrameMetrics:
    """Per-frame measurements, before aggregation across a scene."""

    blur_variance: float
    brightness: float
    exposure: float


class PointFlow(NamedTuple):
    """Where a tracked point was, and how far it moved.

    The *position* is not incidental. Displacement alone cannot separate a zoom from
    handheld shake: both scatter symmetrically about zero. What distinguishes a zoom is
    that each point moves along the line from the frame centre through it - a
    correlation that is invisible without knowing where the point was.
    """

    x: float
    y: float
    dx: float
    dy: float

    @property
    def magnitude(self) -> float:
        return float((self.dx * self.dx + self.dy * self.dy) ** 0.5)


def measure_frame(frame: np.ndarray) -> FrameMetrics:
    """Measure one frame's sharpness and exposure.

    Pure and stateless, so it can be tested on a synthetic array with no video involved.
    """
    import cv2
    import numpy as np

    grey = to_grayscale(frame)
    # CV_64F, not the input dtype: the Laplacian is signed, and computing it in uint8
    # clips every negative edge response to zero and roughly halves the variance.
    variance = float(cv2.Laplacian(grey, cv2.CV_64F).var())

    mean_luma = float(np.mean(grey)) / 255.0
    return FrameMetrics(
        blur_variance=variance,
        brightness=mean_luma,
        exposure=_exposure_score(grey),
    )


def _exposure_score(grey: np.ndarray) -> float:
    """How much of the frame is crushed to black or blown to white.

    Measured as a clipped-pixel fraction rather than from the mean, because the mean
    cannot distinguish a well-exposed mid-grey frame from one that is half black and
    half white - and those look nothing alike.
    """
    import numpy as np

    total = grey.size
    if total == 0:
        return 0.0
    crushed = float(np.count_nonzero(grey <= 2)) / total
    blown = float(np.count_nonzero(grey >= 253)) / total
    clipped = crushed + blown
    if clipped <= _CLIP_LIMIT:
        return 1.0
    # Linear falloff from the allowance to fully clipped.
    return max(0.0, 1.0 - (clipped - _CLIP_LIMIT) / (1.0 - _CLIP_LIMIT))


def brightness_score(mean_luma: float) -> float:
    """Score a mean luma, penalising both ends.

    A tent function peaking at 0.5. Both a black frame and a white one are unusable, so
    a metric that only punished darkness would happily accept a blown-out sky.
    """
    return max(0.0, 1.0 - abs(mean_luma - 0.5) * 2.0)


class OpenCvQualityAnalyzer:
    """Quality and motion measurement over sampled frame pairs.

    Takes pre-sampled frames rather than a file path, unlike the protocol's
    file-oriented signature, because the orchestrator reads each pair once and feeds it
    to several consumers. Decoding a clip separately for quality and for motion would
    double the most expensive part of the pipeline for no benefit.
    """

    def __init__(self, settings: VisionSettings) -> None:
        self._settings = settings

    @property
    def version(self) -> str:
        return METRICS_VERSION

    # -- Quality ------------------------------------------------------------- #

    def score_frames(self, pairs: list[FramePair], *, stability: float) -> QualityScores:
        """Aggregate per-frame metrics across a scene.

        The **median** is used, not the mean. One frame of motion blur in an otherwise
        sharp shot is normal; the mean would drag the whole scene down for it, while the
        median reports what most of the scene actually looks like.
        """
        if not pairs:
            # No frames could be read. Zero would claim the footage is bad, which is a
            # different statement from "not measured"; the neutral midpoint at least
            # does not rule the scene out or falsely promote it.
            return QualityScores(blur=0.5, brightness=0.5, exposure=0.5, stability=0.5, overall=0.5)

        measurements = [measure_frame(pair.first) for pair in pairs]
        variance = _median([item.blur_variance for item in measurements])
        blur = min(1.0, variance / self._settings.blur_reference_variance)
        brightness = brightness_score(_median([item.brightness for item in measurements]))
        exposure = _median([item.exposure for item in measurements])

        overall = (
            blur * _QUALITY_WEIGHTS["blur"]
            + brightness * _QUALITY_WEIGHTS["brightness"]
            + exposure * _QUALITY_WEIGHTS["exposure"]
            + stability * _QUALITY_WEIGHTS["stability"]
        )
        return QualityScores(
            blur=_clamp(blur),
            brightness=_clamp(brightness),
            exposure=_clamp(exposure),
            stability=_clamp(stability),
            overall=_clamp(overall),
            blur_variance=variance,
        )

    # -- Motion -------------------------------------------------------------- #

    def measure_motion(self, pairs: list[FramePair]) -> MotionStats:
        """Estimate movement and classify the camera move across a scene."""
        flows: list[PointFlow] = []
        for pair in pairs:
            if pair.second is None:
                continue
            flows.extend(track_points(pair.first, pair.second))

        if not flows:
            # Nothing trackable. A featureless frame - fog, a white wall, black - gives
            # optical flow nothing to hold on to, and that is not the same as a locked
            # off camera, so the move is reported as unknown rather than static.
            return MotionStats(
                level=MotionLevel.STATIC,
                mean_magnitude=0.0,
                camera_move=CameraMove.UNKNOWN,
                shake=0.0,
            )

        mean_magnitude = sum(flow.magnitude for flow in flows) / len(flows)
        consistency, move = classify_camera_move(
            flows,
            mean_magnitude=mean_magnitude,
            settings=self._settings,
        )
        return MotionStats(
            level=self._motion_level(mean_magnitude),
            mean_magnitude=mean_magnitude,
            camera_move=move,
            shake=_clamp(self._shake_score(mean_magnitude, consistency)),
        )

    def stability_from_motion(self, motion: MotionStats) -> float:
        """Turn measured shake into a stability score.

        Stability is the complement of *incoherent* movement, not of movement itself. A
        smooth drone push is rock-steady footage with a large mean magnitude; scoring it
        as unstable would rule out the best shot in the project.
        """
        return _clamp(1.0 - motion.shake)

    def _motion_level(self, magnitude: float) -> MotionLevel:
        settings = self._settings
        if magnitude < settings.motion_low_threshold:
            return MotionLevel.STATIC
        if magnitude < settings.motion_medium_threshold:
            return MotionLevel.LOW
        if magnitude < settings.motion_high_threshold:
            return MotionLevel.MEDIUM
        return MotionLevel.HIGH

    def _shake_score(self, magnitude: float, consistency: float) -> float:
        """Jitter: movement that goes nowhere.

        Scaled by how *inconsistent* the tracked points are, so deliberate camera moves
        - where the points agree - contribute almost nothing however fast they are.
        """
        incoherence = max(0.0, 1.0 - consistency)
        return min(1.0, magnitude / self._settings.shake_reference) * incoherence


def track_points(first: np.ndarray, second: np.ndarray) -> list[PointFlow]:
    """Sparse optical flow between two frames, as positioned displacements.

    Sparse Lucas-Kanade rather than dense Farneback: it is roughly an order of magnitude
    faster, and the individual point vectors are exactly what camera-move classification
    needs. Dense flow would give a magnitude field and throw away the agreement between
    points that distinguishes a pan from a wobble.
    """
    import cv2
    import numpy as np

    grey_first = to_grayscale(first)
    grey_second = to_grayscale(second)
    if grey_first.shape != grey_second.shape:
        # Frame size changed mid-clip. Rare, but comparing them would be meaningless.
        return []

    corners = cv2.goodFeaturesToTrack(
        grey_first,
        maxCorners=200,
        qualityLevel=0.01,
        minDistance=8,
        blockSize=7,
    )
    if corners is None or len(corners) == 0:
        return []

    moved, status, _error = cv2.calcOpticalFlowPyrLK(  # type: ignore[call-overload]
        grey_first,
        grey_second,
        corners.astype(np.float32),
        None,
        winSize=(21, 21),
        maxLevel=3,
    )
    if moved is None or status is None:
        return []

    flows: list[PointFlow] = []
    for index, tracked in enumerate(status.reshape(-1)):
        if not tracked:
            continue
        start = corners[index].reshape(2)
        end = moved[index].reshape(2)
        flows.append(
            PointFlow(
                x=float(start[0]),
                y=float(start[1]),
                dx=float(end[0] - start[0]),
                dy=float(end[1] - start[1]),
            )
        )
    return flows


def classify_camera_move(
    flows: list[PointFlow],
    *,
    mean_magnitude: float,
    settings: VisionSettings,
) -> tuple[float, CameraMove]:
    """Classify a camera move, returning ``(consistency, move)``.

    ``consistency`` is the length of the mean displacement vector divided by the mean
    displacement length: 1.0 when every point moves the same way, near 0 when they
    disagree. That single number separates a deliberate move from shake, because a pan
    and a handheld wobble are both "lots of movement" and only their coherence differs.

    Low consistency alone does *not* mean handheld, though - a zoom is also incoherent by
    that measure, since opposite sides of the frame move in opposite directions and
    cancel out. Telling those two apart is what :func:`radial_score` is for.
    """
    if not flows:
        return 0.0, CameraMove.UNKNOWN

    count = len(flows)
    mean_dx = sum(flow.dx for flow in flows) / count
    mean_dy = sum(flow.dy for flow in flows) / count
    mean_vector_length = (mean_dx * mean_dx + mean_dy * mean_dy) ** 0.5
    consistency = mean_vector_length / mean_magnitude if mean_magnitude > 0 else 0.0

    if mean_magnitude < settings.motion_low_threshold:
        return consistency, CameraMove.STATIC

    if consistency < settings.camera_move_consistency:
        if abs(radial_score(flows)) >= settings.zoom_radial_threshold:
            return consistency, CameraMove.ZOOM
        return consistency, CameraMove.HANDHELD

    return consistency, CameraMove.PAN if abs(mean_dx) >= abs(mean_dy) else CameraMove.TILT


def radial_score(flows: list[PointFlow]) -> float:
    """How radial the movement is: +1 fully outward, -1 fully inward, ~0 not radial.

    For each point, the displacement is projected onto the outward unit vector from the
    centre of the tracked points, then normalised by the displacement's own length. The
    mean of those projections is the score.

    This is the measurement that needs positions, and the reason
    :class:`PointFlow` carries them. An earlier version tested only whether the
    displacement components were symmetric about zero - which random handheld jitter
    satisfies perfectly, so every shaky shot was reported as a zoom. Projection onto the
    radial direction has no such confusion: jitter averages to nothing because its
    direction is unrelated to where the point sits.
    """
    if len(flows) < 4:
        # Too few points to distinguish a pattern from coincidence.
        return 0.0

    centre_x = sum(flow.x for flow in flows) / len(flows)
    centre_y = sum(flow.y for flow in flows) / len(flows)

    projections: list[float] = []
    for flow in flows:
        offset_x = flow.x - centre_x
        offset_y = flow.y - centre_y
        radius = (offset_x * offset_x + offset_y * offset_y) ** 0.5
        magnitude = flow.magnitude
        if radius < 1e-6 or magnitude < 1e-6:
            # A point at the centre has no radial direction, and a point that did not
            # move carries no evidence either way.
            continue
        projections.append((flow.dx * offset_x + flow.dy * offset_y) / (radius * magnitude))

    if not projections:
        return 0.0
    return sum(projections) / len(projections)


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def _clamp(value: float) -> float:
    """Into 0..1. Weighted sums and 1-x can drift a hair outside by rounding."""
    return min(1.0, max(0.0, value))


__all__ = [
    "METRICS_VERSION",
    "FrameMetrics",
    "OpenCvQualityAnalyzer",
    "PointFlow",
    "brightness_score",
    "classify_camera_move",
    "measure_frame",
    "radial_score",
    "track_points",
]
