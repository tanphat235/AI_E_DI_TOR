"""Tests for perceptual hashing and duplicate detection.

The most important test in this file is
:meth:`TestFlatFrames.test_differently_coloured_flat_frames_are_not_duplicates`. It covers
a real bug: dHash alone returns all-zeros for any frame without gradients, so a red shot
and a green shot hashed identically and were grouped as the same take. Suppressing
unrelated footage is a far worse failure than missing a duplicate, because the director
never sees the missing option.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.analysis.vision.duplicates import (
    PerceptualDuplicateDetector,
    colour_hash,
    dhash,
    group_by_clip,
    hash_similarity,
    perceptual_hash,
)
from app.models.common import CameraMove, MediaRef, MotionLevel, ShotType, TimeRange
from app.models.video import MotionStats, QualityScores, Scene, SceneTags


def _flat(blue: int, green: int, red: int, size: int = 64) -> np.ndarray:
    frame = np.zeros((size, size, 3), dtype=np.uint8)
    frame[:, :] = (blue, green, red)
    return frame


def _split(shift: int = 0, size: int = 64) -> np.ndarray:
    """A frame with real structure: a vertical light/dark edge."""
    frame = np.zeros((size, size, 3), dtype=np.uint8)
    frame[:, size // 2 + shift :] = 255
    return frame


def _noise(seed: int = 0, size: int = 64) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (size, size, 3), dtype=np.uint8)


def _scene(
    key_index: int,
    *,
    clip: str = "raw/001.mp4",
    phash: str | None = None,
    quality: float = 0.8,
    duration: float = 3.0,
) -> Scene:
    return Scene(
        clip=MediaRef(path=clip),
        index=key_index,
        range=TimeRange(start=0.0, end=duration),
        quality=QualityScores(
            blur=quality, brightness=0.7, exposure=0.8, stability=0.9, overall=quality
        ),
        motion=MotionStats(
            level=MotionLevel.LOW, mean_magnitude=1.0, camera_move=CameraMove.STATIC
        ),
        tags=SceneTags(provider="test"),
        shot_type=ShotType.MEDIUM,
        phash=phash,
    )


class TestHashShape:
    def test_the_structure_half_is_64_bits(self) -> None:
        assert len(dhash(_noise())) == 16

    def test_the_colour_half_is_48_bits(self) -> None:
        assert len(colour_hash(_noise())) == 12

    def test_the_combined_hash_is_112_bits(self) -> None:
        assert len(perceptual_hash(_noise())) == 28

    def test_hashes_are_valid_hex(self) -> None:
        int(perceptual_hash(_noise()), 16)

    def test_hashing_is_deterministic(self) -> None:
        frame = _noise(3)
        assert perceptual_hash(frame) == perceptual_hash(frame.copy())

    def test_a_greyscale_frame_still_hashes_to_the_full_width(self) -> None:
        grey = np.full((64, 64), 128, dtype=np.uint8)
        assert len(perceptual_hash(grey)) == 28


class TestFlatFrames:
    """The degenerate case that motivated adding colour to the hash."""

    def test_dhash_alone_cannot_tell_flat_frames_apart(self) -> None:
        """Documents the underlying limitation the combined hash works around."""
        assert dhash(_flat(0, 0, 255)) == dhash(_flat(0, 255, 0))
        assert hash_similarity(dhash(_flat(0, 0, 255)), dhash(_flat(0, 255, 0))) == 1.0

    def test_differently_coloured_flat_frames_are_not_duplicates(self) -> None:
        """The regression test. A red shot and a green shot are not the same take."""
        similarity = hash_similarity(
            perceptual_hash(_flat(0, 0, 255)), perceptual_hash(_flat(0, 255, 0))
        )
        assert similarity < 0.9

    def test_a_red_frame_and_a_dark_frame_are_not_duplicates(self) -> None:
        similarity = hash_similarity(
            perceptual_hash(_flat(0, 0, 255)), perceptual_hash(_flat(10, 10, 10))
        )
        assert similarity < 0.9

    def test_identical_flat_frames_are_duplicates(self) -> None:
        assert (
            hash_similarity(perceptual_hash(_flat(0, 0, 255)), perceptual_hash(_flat(0, 0, 255)))
            == 1.0
        )

    def test_a_small_exposure_shift_still_matches(self) -> None:
        """Two takes of one shot under drifting light must stay in the same buckets."""
        assert (
            hash_similarity(perceptual_hash(_flat(0, 0, 255)), perceptual_hash(_flat(0, 0, 250)))
            >= 0.9
        )

    def test_the_similarity_floor_for_flat_frames_is_documented(self) -> None:
        """Flat frames agree on all 64 structure bits, so similarity cannot go below 0.57.

        ``duplicate_similarity`` must stay above this or every flat shot merges.
        """
        worst = hash_similarity(
            perceptual_hash(_flat(0, 0, 0)), perceptual_hash(_flat(255, 255, 255))
        )
        assert worst == pytest.approx(64 / 112, abs=0.02)


class TestStructuredFrames:
    def test_the_same_composition_matches(self) -> None:
        assert hash_similarity(perceptual_hash(_split()), perceptual_hash(_split())) == 1.0

    def test_a_shifted_composition_does_not_match(self) -> None:
        similarity = hash_similarity(perceptual_hash(_split()), perceptual_hash(_split(20)))
        assert similarity < 0.9

    def test_unrelated_noise_frames_do_not_match(self) -> None:
        assert hash_similarity(perceptual_hash(_noise(1)), perceptual_hash(_noise(2))) < 0.9

    def test_a_resized_frame_still_matches(self) -> None:
        """Two takes of a shot at different resolutions are still the same shot."""
        import cv2

        original = _noise(5, size=256)
        smaller = cv2.resize(original, (128, 128), interpolation=cv2.INTER_AREA)
        assert hash_similarity(perceptual_hash(original), perceptual_hash(smaller)) >= 0.9


class TestHashSimilarity:
    def test_identical_hashes(self) -> None:
        assert hash_similarity("abcd", "abcd") == 1.0

    def test_a_fully_inverted_hash(self) -> None:
        assert hash_similarity("0000", "ffff") == 0.0

    def test_one_differing_bit(self) -> None:
        assert hash_similarity("0000", "0001") == pytest.approx(1.0 - 1 / 16)

    @pytest.mark.parametrize(
        ("first", "second"),
        [("", "abcd"), ("abcd", ""), ("abcd", "abcdef"), ("zzzz", "abcd")],
    )
    def test_malformed_or_mismatched_hashes_score_zero_rather_than_raising(
        self, first: str, second: str
    ) -> None:
        """A hash from an older analyser must fail to match, not abort the run."""
        assert hash_similarity(first, second) == 0.0


class TestDuplicateDetector:
    @pytest.fixture
    def detector(self) -> PerceptualDuplicateDetector:
        return PerceptualDuplicateDetector(similarity_threshold=0.9)

    def test_identical_hashes_are_grouped(self, detector: PerceptualDuplicateDetector) -> None:
        shared = perceptual_hash(_noise(1))
        scenes = (
            _scene(0, clip="raw/001.mp4", phash=shared, quality=0.9),
            _scene(0, clip="raw/002.mp4", phash=shared, quality=0.5),
        )
        groups = detector.find_duplicates(scenes)
        assert len(groups) == 1
        assert groups[0].representative == "001#0"
        assert groups[0].duplicates == ("002#0",)

    def test_the_highest_quality_take_survives(self, detector: PerceptualDuplicateDetector) -> None:
        shared = perceptual_hash(_noise(2))
        scenes = (
            _scene(0, clip="raw/001.mp4", phash=shared, quality=0.4),
            _scene(0, clip="raw/002.mp4", phash=shared, quality=0.95),
        )
        assert detector.find_duplicates(scenes)[0].representative == "002#0"

    def test_duration_breaks_a_quality_tie(self, detector: PerceptualDuplicateDetector) -> None:
        """A longer take gives the director room to choose an in and an out point."""
        shared = perceptual_hash(_noise(3))
        scenes = (
            _scene(0, clip="raw/001.mp4", phash=shared, quality=0.8, duration=2.0),
            _scene(0, clip="raw/002.mp4", phash=shared, quality=0.8, duration=6.0),
        )
        assert detector.find_duplicates(scenes)[0].representative == "002#0"

    def test_distinct_scenes_are_not_grouped(self, detector: PerceptualDuplicateDetector) -> None:
        scenes = (
            _scene(0, clip="raw/001.mp4", phash=perceptual_hash(_noise(1))),
            _scene(0, clip="raw/002.mp4", phash=perceptual_hash(_noise(2))),
        )
        assert detector.find_duplicates(scenes) == ()

    def test_three_takes_collapse_to_one_group(self, detector: PerceptualDuplicateDetector) -> None:
        """The case that matters: cutting between three takes is the obvious tell."""
        shared = perceptual_hash(_noise(4))
        scenes = tuple(
            _scene(0, clip=f"raw/00{index}.mp4", phash=shared, quality=0.5 + index * 0.1)
            for index in (1, 2, 3)
        )
        groups = detector.find_duplicates(scenes)
        assert len(groups) == 1
        assert len(groups[0].duplicates) == 2

    def test_scenes_without_a_hash_are_skipped(self, detector: PerceptualDuplicateDetector) -> None:
        scenes = (
            _scene(0, clip="raw/001.mp4", phash=None),
            _scene(0, clip="raw/002.mp4", phash=None),
        )
        assert detector.find_duplicates(scenes) == ()

    def test_a_single_scene_cannot_be_a_duplicate(
        self, detector: PerceptualDuplicateDetector
    ) -> None:
        assert detector.find_duplicates((_scene(0, phash=perceptual_hash(_noise())),)) == ()

    def test_no_scenes(self, detector: PerceptualDuplicateDetector) -> None:
        assert detector.find_duplicates(()) == ()

    def test_the_representative_never_appears_among_its_duplicates(
        self, detector: PerceptualDuplicateDetector
    ) -> None:
        """The model forbids it, so a bug here would surface as a ValidationError."""
        shared = perceptual_hash(_noise(6))
        scenes = tuple(_scene(0, clip=f"raw/00{index}.mp4", phash=shared) for index in (1, 2, 3))
        for group in detector.find_duplicates(scenes):
            assert group.representative not in group.duplicates

    def test_a_strict_threshold_groups_nothing(self) -> None:
        strict = PerceptualDuplicateDetector(similarity_threshold=1.0)
        scenes = (
            _scene(0, clip="raw/001.mp4", phash=perceptual_hash(_split())),
            _scene(0, clip="raw/002.mp4", phash=perceptual_hash(_split(2))),
        )
        assert strict.find_duplicates(scenes) == ()

    def test_reported_similarity_reflects_the_group(
        self, detector: PerceptualDuplicateDetector
    ) -> None:
        shared = perceptual_hash(_noise(7))
        scenes = (
            _scene(0, clip="raw/001.mp4", phash=shared),
            _scene(0, clip="raw/002.mp4", phash=shared),
        )
        assert detector.find_duplicates(scenes)[0].similarity == pytest.approx(1.0)


class TestGroupByClip:
    def test_scenes_are_bucketed_by_source(self) -> None:
        scenes = (
            _scene(0, clip="raw/001.mp4"),
            _scene(1, clip="raw/001.mp4"),
            _scene(0, clip="raw/002.mp4"),
        )
        buckets = group_by_clip(scenes)
        assert len(buckets["raw/001.mp4"]) == 2
        assert len(buckets["raw/002.mp4"]) == 1

    def test_no_scenes_yields_no_buckets(self) -> None:
        assert group_by_clip(()) == {}
