"""Tests for scene-boundary logic and container probing.

The scene functions are pure - cut points in, ranges out - so they are tested exhaustively
here with no video. What PySceneDetect itself returns is covered by the integration tests;
what matters at this level is that its output is turned into ranges that **tile the whole
clip**, because a gap would make footage invisible to the director.
"""

from __future__ import annotations

import itertools
from pathlib import Path

import pytest

from app.analysis.vision.probe import ProbeError, PyAvProber, _frame_rate, read_rotation
from app.analysis.vision.scenes import (
    SUPPORTED_DETECTORS,
    PySceneDetectDetector,
    SceneDetectionError,
    merge_short_scenes,
    ranges_from_cuts,
)
from app.config.settings import VisionSettings
from app.models.common import MediaRef, TimeRange
from app.models.media import MediaProbe, VideoStreamInfo


def _ranges(*pairs: tuple[float, float]) -> tuple[TimeRange, ...]:
    return tuple(TimeRange(start=start, end=end) for start, end in pairs)


class TestRangesFromCuts:
    def test_no_cuts_is_one_scene_covering_the_clip(self) -> None:
        """Not zero scenes: a single continuous take is one scene."""
        assert ranges_from_cuts([], duration=10.0) == _ranges((0.0, 10.0))

    def test_cuts_become_contiguous_ranges(self) -> None:
        assert ranges_from_cuts([3.0, 7.0], duration=10.0) == _ranges(
            (0.0, 3.0), (3.0, 7.0), (7.0, 10.0)
        )

    def test_the_ranges_tile_the_clip_with_no_gaps(self) -> None:
        scenes = ranges_from_cuts([2.5, 6.1, 8.0], duration=12.0)
        assert scenes[0].start == 0.0
        assert scenes[-1].end == 12.0
        for earlier, later in itertools.pairwise(scenes):
            assert earlier.end == later.start

    def test_unsorted_cuts_are_ordered(self) -> None:
        assert ranges_from_cuts([7.0, 3.0], duration=10.0)[0].end == 3.0

    def test_duplicate_cuts_collapse(self) -> None:
        """A duplicate would otherwise produce a zero-length range and fail validation."""
        assert len(ranges_from_cuts([5.0, 5.0], duration=10.0)) == 2

    @pytest.mark.parametrize("cut", [0.0, 10.0, 12.0, -1.0])
    def test_cuts_outside_the_clip_are_discarded(self, cut: float) -> None:
        assert ranges_from_cuts([cut], duration=10.0) == _ranges((0.0, 10.0))

    def test_a_zero_duration_clip_has_no_scenes(self) -> None:
        assert ranges_from_cuts([1.0], duration=0.0) == ()


class TestMergeShortScenes:
    def test_scenes_above_the_minimum_are_untouched(self) -> None:
        scenes = _ranges((0.0, 2.0), (2.0, 4.0))
        assert merge_short_scenes(scenes, minimum=1.0) == scenes

    def test_a_short_scene_merges_forward(self) -> None:
        """Forward, so a kept scene's start stays exactly on a real cut.

        Starting a shot mid-action is far more visible than ending one slightly late.
        """
        scenes = _ranges((0.0, 0.2), (0.2, 3.0))
        assert merge_short_scenes(scenes, minimum=1.0) == _ranges((0.0, 3.0))

    def test_several_consecutive_short_scenes_collapse_into_one(self) -> None:
        scenes = _ranges((0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 3.0))
        assert merge_short_scenes(scenes, minimum=1.0) == _ranges((0.0, 3.0))

    def test_a_short_tail_merges_backwards(self) -> None:
        """With nothing after it, the last scene has to join the one before."""
        scenes = _ranges((0.0, 3.0), (3.0, 3.1))
        assert merge_short_scenes(scenes, minimum=1.0) == _ranges((0.0, 3.1))

    def test_a_clip_of_only_short_scenes_becomes_one_scene(self) -> None:
        """The honest answer for footage that is one continuous take."""
        scenes = _ranges((0.0, 0.2), (0.2, 0.4), (0.4, 0.6))
        assert merge_short_scenes(scenes, minimum=1.0) == _ranges((0.0, 0.6))

    def test_merging_still_tiles_the_clip(self) -> None:
        scenes = _ranges((0.0, 0.1), (0.1, 2.0), (2.0, 2.1), (2.1, 5.0))
        merged = merge_short_scenes(scenes, minimum=0.8)
        assert merged[0].start == 0.0
        assert merged[-1].end == 5.0
        for earlier, later in itertools.pairwise(merged):
            assert earlier.end == later.start

    def test_empty_input(self) -> None:
        assert merge_short_scenes((), minimum=1.0) == ()


class TestSceneDetectorConfiguration:
    def test_the_name_records_the_algorithm(self) -> None:
        detector = PySceneDetectDetector(VisionSettings(scene_detector="adaptive"))
        assert detector.name == "pyscenedetect/adaptive"

    def test_an_unknown_algorithm_is_rejected_with_the_alternatives(self, tmp_path: Path) -> None:
        video = tmp_path / "clip.mp4"
        video.write_bytes(b"\0" * 32)
        detector = PySceneDetectDetector(VisionSettings(scene_detector="magic"))
        probe = MediaProbe(
            source=MediaRef(path="raw/clip.mp4"),
            duration=5.0,
            size_bytes=32,
            video=VideoStreamInfo(width=640, height=360, fps=25.0, codec="h264"),
        )
        with pytest.raises(SceneDetectionError, match="unknown scene_detector"):
            detector.detect(video, probe=probe)

    def test_every_supported_algorithm_is_named(self) -> None:
        assert set(SUPPORTED_DETECTORS) == {"content", "adaptive", "threshold"}

    def test_a_missing_file_is_reported_before_decoding(self, tmp_path: Path) -> None:
        detector = PySceneDetectDetector(VisionSettings())
        probe = MediaProbe(
            source=MediaRef(path="raw/gone.mp4"),
            duration=5.0,
            size_bytes=1,
            video=VideoStreamInfo(width=640, height=360, fps=25.0, codec="h264"),
        )
        with pytest.raises(FileNotFoundError):
            detector.detect(tmp_path / "gone.mp4", probe=probe)


class TestProber:
    def test_a_missing_file_raises_before_opening_a_decoder(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="media file not found"):
            PyAvProber().probe(tmp_path / "absent.mp4", ref=MediaRef(path="raw/absent.mp4"))

    def test_a_file_that_is_not_media_is_reported_clearly(self, tmp_path: Path) -> None:
        junk = tmp_path / "notavideo.mp4"
        junk.write_text("this is plain text", encoding="utf-8")
        with pytest.raises(ProbeError, match="could not read"):
            PyAvProber().probe(junk, ref=MediaRef(path="raw/notavideo.mp4"))

    def test_the_name_is_stable(self) -> None:
        assert PyAvProber().name == "pyav"

    def test_rotation_of_a_non_media_file_is_zero_not_an_error(self, tmp_path: Path) -> None:
        """Rotation is a supplementary fact; failing to read it must not abort a probe."""
        junk = tmp_path / "junk.mp4"
        junk.write_text("nope", encoding="utf-8")
        assert read_rotation(junk) == 0

    def test_rotation_of_a_missing_file_is_zero(self, tmp_path: Path) -> None:
        assert read_rotation(tmp_path / "nothing.mp4") == 0


class TestFrameRateFallbacks:
    """PyAV has several notions of frame rate, and VFR footage makes them disagree."""

    class _Stream:
        def __init__(self, **rates: object) -> None:
            for name, value in rates.items():
                setattr(self, name, value)

    def test_average_rate_is_preferred(self) -> None:
        from fractions import Fraction

        stream = self._Stream(average_rate=Fraction(30000, 1001), guessed_rate=Fraction(30, 1))
        assert _frame_rate(stream) == pytest.approx(29.97, abs=0.01)

    def test_guessed_rate_is_the_fallback(self) -> None:
        from fractions import Fraction

        stream = self._Stream(average_rate=None, guessed_rate=Fraction(25, 1))
        assert _frame_rate(stream) == 25.0

    def test_no_usable_rate_yields_zero(self) -> None:
        stream = self._Stream(average_rate=None, guessed_rate=None, base_rate=None)
        assert _frame_rate(stream) == 0.0

    def test_a_zero_rate_is_not_accepted(self) -> None:
        from fractions import Fraction

        stream = self._Stream(average_rate=Fraction(0, 1), guessed_rate=Fraction(24, 1))
        assert _frame_rate(stream) == 24.0
