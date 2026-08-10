"""Tests for quality and motion measurement.

Every test here builds its input as a numpy array, so no video is decoded and the whole
module runs in milliseconds. That is possible because the metric functions were kept
pure: they take frames, not paths.

The assertions are mostly *relative* - sharper scores higher than blurrier, a pan is
distinguished from a wobble - because that is what the metrics honestly provide. Pinning
an absolute blur score would be testing the configured reference value, not the code.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.analysis.vision.frames import FramePair, downscale, probe_timestamps, to_grayscale
from app.analysis.vision.metrics import (
    OpenCvQualityAnalyzer,
    PointFlow,
    brightness_score,
    classify_camera_move,
    measure_frame,
    radial_score,
    track_points,
)
from app.config.settings import VisionSettings
from app.models.common import CameraMove, MotionLevel, TimeRange


def _noise(width: int = 320, height: int = 180, *, seed: int = 0) -> np.ndarray:
    """A high-detail frame. Sharp, mid-exposure, and rich in trackable corners."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (height, width, 3), dtype=np.uint8)


def _flat(value: int = 128, width: int = 320, height: int = 180) -> np.ndarray:
    return np.full((height, width, 3), value, dtype=np.uint8)


def _checkerboard(width: int = 320, height: int = 180, square: int = 20) -> np.ndarray:
    """Strong, regular edges - the sharpest realistic input."""
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    for row in range(height):
        for column in range(width):
            if ((row // square) + (column // square)) % 2 == 0:
                frame[row, column] = 255
    return frame


def _blurred(frame: np.ndarray, strength: int = 21) -> np.ndarray:
    import cv2

    return cv2.GaussianBlur(frame, (strength, strength), 0)


def _shift(frame: np.ndarray, dx: int, dy: int) -> np.ndarray:
    """Translate a frame, which is what a pan looks like between two frames."""
    return np.roll(np.roll(frame, dy, axis=0), dx, axis=1)


@pytest.fixture
def vision() -> VisionSettings:
    return VisionSettings()


@pytest.fixture
def analyzer(vision: VisionSettings) -> OpenCvQualityAnalyzer:
    return OpenCvQualityAnalyzer(vision)


class TestFrameHelpers:
    def test_downscale_shrinks_and_preserves_aspect(self) -> None:
        result = downscale(_noise(1920, 1080), 640)
        assert result.shape[1] == 640
        assert result.shape[0] == pytest.approx(360, abs=1)

    def test_downscale_never_upscales(self) -> None:
        """Upscaling would invent detail and inflate the sharpness score."""
        frame = _noise(320, 180)
        assert downscale(frame, 640).shape == frame.shape

    def test_to_grayscale_is_idempotent(self) -> None:
        grey = to_grayscale(_noise())
        assert grey.ndim == 2
        assert to_grayscale(grey).shape == grey.shape

    def test_probe_points_are_strictly_inside_the_scene(self) -> None:
        """A frame taken on a cut belongs to neither shot and contaminates both."""
        scene = TimeRange(start=10.0, end=14.0)
        points = probe_timestamps(scene, 3)
        assert points == [11.0, 12.0, 13.0]
        assert all(scene.start < point < scene.end for point in points)

    def test_probe_points_scale_with_the_count(self) -> None:
        assert len(probe_timestamps(TimeRange(start=0.0, end=10.0), 5)) == 5

    def test_no_probe_points_when_none_requested(self) -> None:
        assert probe_timestamps(TimeRange(start=0.0, end=10.0), 0) == []

    def test_frame_pair_reports_a_missing_second_frame(self) -> None:
        """The last frame of a file has no successor, which is not zero motion."""
        pair = FramePair(timestamp=1.0, first=_noise(), second=None)
        assert pair.has_pair is False


class TestBlur:
    def test_a_sharp_frame_scores_far_above_a_blurred_one(self) -> None:
        sharp = measure_frame(_checkerboard())
        blurred = measure_frame(_blurred(_checkerboard()))
        assert sharp.blur_variance > blurred.blur_variance * 5

    def test_a_flat_frame_has_almost_no_variance(self) -> None:
        """The honest limitation: a sharp photo of a blank wall looks blurred."""
        assert measure_frame(_flat()).blur_variance < 1.0

    def test_the_raw_variance_is_preserved_on_the_scores(
        self, analyzer: OpenCvQualityAnalyzer
    ) -> None:
        """Kept so the reference value can be retuned without re-decoding footage."""
        pairs = [FramePair(timestamp=1.0, first=_checkerboard(), second=None)]
        scores = analyzer.score_frames(pairs, stability=1.0)
        assert scores.blur_variance is not None
        assert scores.blur_variance > 0.0

    def test_the_score_saturates_rather_than_exceeding_one(self, vision: VisionSettings) -> None:
        tuned = OpenCvQualityAnalyzer(vision.model_copy(update={"blur_reference_variance": 1.0}))
        pairs = [FramePair(timestamp=1.0, first=_checkerboard(), second=None)]
        assert tuned.score_frames(pairs, stability=1.0).blur == 1.0


class TestBrightnessAndExposure:
    @pytest.mark.parametrize(
        ("luma", "expected"),
        [(0.5, 1.0), (0.0, 0.0), (1.0, 0.0), (0.25, 0.5), (0.75, 0.5)],
    )
    def test_brightness_peaks_at_mid_grey(self, luma: float, expected: float) -> None:
        """Both ends are unusable, so a metric that only punished darkness would
        happily accept a blown-out sky."""
        assert brightness_score(luma) == pytest.approx(expected, abs=0.01)

    def test_a_dark_frame_scores_low(self) -> None:
        assert brightness_score(measure_frame(_flat(10)).brightness) < 0.2

    def test_a_blown_out_frame_also_scores_low(self) -> None:
        assert brightness_score(measure_frame(_flat(250)).brightness) < 0.2

    def test_a_mid_grey_frame_scores_high(self) -> None:
        assert brightness_score(measure_frame(_flat(128)).brightness) > 0.9

    def test_exposure_is_perfect_when_nothing_clips(self) -> None:
        assert measure_frame(_flat(128)).exposure == 1.0

    def test_exposure_penalises_a_crushed_frame(self) -> None:
        assert measure_frame(_flat(0)).exposure < 0.1

    def test_exposure_penalises_a_blown_frame(self) -> None:
        assert measure_frame(_flat(255)).exposure < 0.1

    def test_a_small_specular_highlight_is_tolerated(self) -> None:
        """A glint or a genuinely black shadow is normal and must not be penalised."""
        frame = _flat(128)
        frame[:2, :2] = 255
        assert measure_frame(frame).exposure == 1.0

    def test_exposure_distinguishes_mid_grey_from_half_black_half_white(self) -> None:
        """Both have the same mean, and they look nothing alike."""
        split = np.zeros((180, 320, 3), dtype=np.uint8)
        split[:, 160:] = 255
        assert measure_frame(split).exposure < measure_frame(_flat(128)).exposure


class TestMotionTracking:
    def test_a_translation_is_measured_as_movement(self) -> None:
        frame = _noise(seed=1)
        flows = track_points(frame, _shift(frame, 6, 0))
        assert flows
        mean_dx = sum(flow.dx for flow in flows) / len(flows)
        assert mean_dx == pytest.approx(6.0, abs=2.0)

    def test_tracked_points_carry_their_position(self) -> None:
        """Positions are what make a zoom distinguishable from shake."""
        frame = _noise(seed=1)
        flows = track_points(frame, _shift(frame, 6, 0))
        assert flows
        assert all(flow.x >= 0.0 and flow.y >= 0.0 for flow in flows)
        # Points must be spread across the frame, not clustered at the origin.
        assert max(flow.x for flow in flows) > 50.0

    def test_identical_frames_show_no_movement(self) -> None:
        frame = _noise(seed=2)
        flows = track_points(frame, frame.copy())
        assert max((flow.magnitude for flow in flows), default=0.0) < 0.5

    def test_a_featureless_frame_yields_nothing_to_track(self) -> None:
        assert track_points(_flat(), _flat()) == []

    def test_mismatched_frame_sizes_are_refused(self) -> None:
        """Comparing different geometries would be meaningless, not merely wrong."""
        assert track_points(_noise(320, 180), _noise(160, 90)) == []


def _grid_positions(count: int = 36) -> list[tuple[float, float]]:
    """Point positions spread over a 320x180 frame."""
    side = int(count**0.5)
    return [
        (20.0 + column * (280.0 / max(1, side - 1)), 20.0 + row * (140.0 / max(1, side - 1)))
        for row in range(side)
        for column in range(side)
    ]


class TestCameraMove:
    def _uniform(self, dx: float, dy: float) -> list[PointFlow]:
        """Every point moving identically: a pan or a tilt."""
        return [PointFlow(x=x, y=y, dx=dx, dy=dy) for x, y in _grid_positions()]

    def _handheld(self) -> list[PointFlow]:
        """Random displacement, unrelated to position: shake."""
        rng = np.random.default_rng(7)
        return [
            PointFlow(x=x, y=y, dx=float(rng.uniform(-4, 4)), dy=float(rng.uniform(-4, 4)))
            for x, y in _grid_positions()
        ]

    def _zoom(self, *, outward: bool = True) -> list[PointFlow]:
        """Displacement along the line from the centre through each point."""
        positions = _grid_positions()
        centre_x = sum(x for x, _ in positions) / len(positions)
        centre_y = sum(y for _, y in positions) / len(positions)
        sign = 1.0 if outward else -1.0
        flows: list[PointFlow] = []
        for x, y in positions:
            offset_x, offset_y = x - centre_x, y - centre_y
            radius = (offset_x**2 + offset_y**2) ** 0.5 or 1.0
            flows.append(
                PointFlow(
                    x=x,
                    y=y,
                    dx=sign * 4.0 * offset_x / radius,
                    dy=sign * 4.0 * offset_y / radius,
                )
            )
        return flows

    def test_consistent_horizontal_movement_is_a_pan(self, vision: VisionSettings) -> None:
        consistency, move = classify_camera_move(
            self._uniform(5.0, 0.2), mean_magnitude=5.0, settings=vision
        )
        assert move is CameraMove.PAN
        assert consistency > 0.9

    def test_consistent_vertical_movement_is_a_tilt(self, vision: VisionSettings) -> None:
        _consistency, move = classify_camera_move(
            self._uniform(0.2, 5.0), mean_magnitude=5.0, settings=vision
        )
        assert move is CameraMove.TILT

    def test_barely_any_movement_is_static(self, vision: VisionSettings) -> None:
        _consistency, move = classify_camera_move(
            self._uniform(0.05, 0.05), mean_magnitude=0.07, settings=vision
        )
        assert move is CameraMove.STATIC

    def test_incoherent_movement_is_handheld(self, vision: VisionSettings) -> None:
        """A pan and a wobble can have identical magnitude; coherence separates them.

        Regression cover for a real bug: the first zoom test only asked whether the
        displacement components straddled zero, which random jitter satisfies perfectly -
        so every handheld shot was reported as a zoom.
        """
        consistency, move = classify_camera_move(
            self._handheld(), mean_magnitude=3.0, settings=vision
        )
        assert move is CameraMove.HANDHELD
        assert consistency < vision.camera_move_consistency

    def test_outward_radial_movement_is_a_zoom(self, vision: VisionSettings) -> None:
        """A zoom sums to nearly zero, so it must not be mistaken for handheld."""
        _consistency, move = classify_camera_move(
            self._zoom(outward=True), mean_magnitude=4.0, settings=vision
        )
        assert move is CameraMove.ZOOM

    def test_inward_radial_movement_is_also_a_zoom(self, vision: VisionSettings) -> None:
        _consistency, move = classify_camera_move(
            self._zoom(outward=False), mean_magnitude=4.0, settings=vision
        )
        assert move is CameraMove.ZOOM

    def test_no_displacements_is_unknown_not_static(self, vision: VisionSettings) -> None:
        _consistency, move = classify_camera_move([], mean_magnitude=0.0, settings=vision)
        assert move is CameraMove.UNKNOWN


class TestRadialScore:
    """The measurement that separates a zoom from shake, and needs positions to do it."""

    def test_a_zoom_out_scores_near_plus_one(self) -> None:
        flows = TestCameraMove()._zoom(outward=True)
        assert radial_score(flows) > 0.9

    def test_a_zoom_in_scores_near_minus_one(self) -> None:
        flows = TestCameraMove()._zoom(outward=False)
        assert radial_score(flows) < -0.9

    def test_handheld_jitter_scores_near_zero(self) -> None:
        assert abs(radial_score(TestCameraMove()._handheld())) < 0.4

    def test_a_pan_is_not_radial(self) -> None:
        """Every point moves the same way, so the radial components cancel."""
        assert abs(radial_score(TestCameraMove()._uniform(5.0, 0.0))) < 0.3

    def test_too_few_points_is_not_a_pattern(self) -> None:
        assert radial_score([PointFlow(x=1.0, y=1.0, dx=1.0, dy=1.0)]) == 0.0

    def test_stationary_points_contribute_nothing(self) -> None:
        flows = [PointFlow(x=float(i), y=float(i), dx=0.0, dy=0.0) for i in range(10)]
        assert radial_score(flows) == 0.0


class TestMotionLevels:
    def test_a_still_scene_is_static(self, analyzer: OpenCvQualityAnalyzer) -> None:
        frame = _noise(seed=3)
        pairs = [FramePair(timestamp=1.0, first=frame, second=frame.copy())]
        motion = analyzer.measure_motion(pairs)
        assert motion.level is MotionLevel.STATIC
        assert motion.mean_magnitude < 0.5

    def test_a_fast_pan_is_high_motion(self, analyzer: OpenCvQualityAnalyzer) -> None:
        frame = _noise(seed=4)
        pairs = [FramePair(timestamp=1.0, first=frame, second=_shift(frame, 12, 0))]
        motion = analyzer.measure_motion(pairs)
        assert motion.level is MotionLevel.HIGH
        assert motion.camera_move is CameraMove.PAN

    def test_pairs_without_a_second_frame_contribute_nothing(
        self, analyzer: OpenCvQualityAnalyzer
    ) -> None:
        pairs = [FramePair(timestamp=1.0, first=_noise(), second=None)]
        motion = analyzer.measure_motion(pairs)
        assert motion.camera_move is CameraMove.UNKNOWN

    def test_an_untrackable_scene_reports_unknown_not_static(
        self, analyzer: OpenCvQualityAnalyzer
    ) -> None:
        """Fog gives flow nothing to hold; that differs from a locked-off camera."""
        pairs = [FramePair(timestamp=1.0, first=_flat(), second=_flat())]
        assert analyzer.measure_motion(pairs).camera_move is CameraMove.UNKNOWN


class TestStability:
    def test_a_smooth_pan_stays_stable(self, analyzer: OpenCvQualityAnalyzer) -> None:
        """A drone push is rock-steady footage with a large magnitude.

        Scoring it unstable would rule out the best shot in a project.
        """
        frame = _noise(seed=5)
        pairs = [FramePair(timestamp=1.0, first=frame, second=_shift(frame, 10, 0))]
        motion = analyzer.measure_motion(pairs)
        assert analyzer.stability_from_motion(motion) > 0.8

    def test_incoherent_movement_costs_stability(
        self, analyzer: OpenCvQualityAnalyzer, vision: VisionSettings
    ) -> None:
        from app.models.video import MotionStats

        shaky = MotionStats(
            level=MotionLevel.HIGH,
            mean_magnitude=5.0,
            camera_move=CameraMove.HANDHELD,
            shake=0.8,
        )
        assert analyzer.stability_from_motion(shaky) == pytest.approx(0.2)


class TestAggregation:
    def test_the_median_ignores_one_bad_frame(self, analyzer: OpenCvQualityAnalyzer) -> None:
        """One frame of motion blur in a sharp shot is normal; the mean would
        drag the whole scene down for it."""
        sharp = _checkerboard()
        pairs = [
            FramePair(timestamp=1.0, first=sharp, second=None),
            FramePair(timestamp=2.0, first=_blurred(sharp), second=None),
            FramePair(timestamp=3.0, first=sharp, second=None),
        ]
        scores = analyzer.score_frames(pairs, stability=1.0)
        blurred_only = analyzer.score_frames(
            [FramePair(timestamp=1.0, first=_blurred(sharp), second=None)], stability=1.0
        )
        assert scores.blur > blurred_only.blur

    def test_no_frames_yields_a_neutral_score_not_zero(
        self, analyzer: OpenCvQualityAnalyzer
    ) -> None:
        """Zero would claim the footage is bad, which is not the same as unmeasured."""
        scores = analyzer.score_frames([], stability=0.5)
        assert scores.overall == pytest.approx(0.5)
        assert scores.blur_variance is None

    def test_overall_is_a_weighted_blend_within_range(
        self, analyzer: OpenCvQualityAnalyzer
    ) -> None:
        pairs = [FramePair(timestamp=1.0, first=_checkerboard(), second=None)]
        scores = analyzer.score_frames(pairs, stability=1.0)
        assert 0.0 <= scores.overall <= 1.0
        # Blur carries the most weight, so a sharp mid-exposure frame must score well.
        assert scores.overall > 0.7

    def test_stability_is_carried_into_the_scores(self, analyzer: OpenCvQualityAnalyzer) -> None:
        pairs = [FramePair(timestamp=1.0, first=_checkerboard(), second=None)]
        steady = analyzer.score_frames(pairs, stability=1.0)
        shaky = analyzer.score_frames(pairs, stability=0.0)
        assert steady.stability == 1.0
        assert shaky.stability == 0.0
        assert steady.overall > shaky.overall
