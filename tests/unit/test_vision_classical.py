"""Tests for the classical-CV vision provider.

The point of this module is the *contract*, not the accuracy of Haar cascades. Two
properties matter, and both are about honesty:

1. **It never invents tags.** ``objects``, ``actions`` and ``setting`` stay empty, because
   a fabricated tag is acted upon by the director while a missing one is worked around.
2. **It degrades rather than failing.** A missing cascade, an unreadable still, or an
   OpenCV build without ``CascadeClassifier`` all produce "unknown", not an exception
   that aborts a forty-clip analysis.

Face detection itself is stubbed, because Haar cascades do not fire on synthetic shapes
and committing a photograph of a face to the repo to test a size threshold would be a
poor trade.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from app.analysis.vision.classical import (
    PROVIDER_NAME,
    ClassicalCvVisionProvider,
    FaceDetection,
    read_image,
)
from app.config.settings import VisionSettings
from app.models.common import MediaRef, ShotType, TimeRange
from app.models.video import Keyframe
from app.services.paths import ProjectPaths

SCENE = TimeRange(start=0.0, end=3.0)


def _write_still(paths: ProjectPaths, name: str, value: int = 128) -> Keyframe:
    """Write a real JPEG into the project cache and return a reference to it."""
    import cv2

    destination = paths.keyframes / name
    destination.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(destination), np.full((180, 320, 3), value, dtype=np.uint8))
    return Keyframe(timestamp=1.0, image=paths.to_ref(destination))


@pytest.fixture
def provider(paths: ProjectPaths) -> ClassicalCvVisionProvider:
    return ClassicalCvVisionProvider(VisionSettings(), project_root=paths.root)


class TestHonestDegradation:
    """A guessed tag is worse than no tag, because the director acts on it."""

    def test_objects_actions_and_setting_are_always_empty(
        self, provider: ClassicalCvVisionProvider, paths: ProjectPaths
    ) -> None:
        keyframe = _write_still(paths, "001/0000_0.jpg")
        tags = provider.describe((keyframe,), scene=SCENE)
        assert tags.objects == ()
        assert tags.actions == ()
        assert tags.setting is None
        assert tags.mood is None
        assert tags.caption is None

    def test_the_provider_names_itself_so_tags_are_traceable(
        self, provider: ClassicalCvVisionProvider, paths: ProjectPaths
    ) -> None:
        """A project can mix providers, so every tag records which one produced it."""
        keyframe = _write_still(paths, "001/0000_0.jpg")
        assert provider.describe((keyframe,), scene=SCENE).provider == PROVIDER_NAME
        assert provider.name == "classical_cv"

    def test_no_keyframes_reports_unknown_not_nobody(
        self, provider: ClassicalCvVisionProvider
    ) -> None:
        """None means "not measured"; 0 would claim the shot is empty of people."""
        tags = provider.describe((), scene=SCENE)
        assert tags.people_count is None
        assert tags.confidence == 0.0

    def test_unreadable_keyframes_report_unknown(
        self, provider: ClassicalCvVisionProvider, paths: ProjectPaths
    ) -> None:
        missing = Keyframe(timestamp=1.0, image=MediaRef(path=".aive/keyframes/001/absent.jpg"))
        tags = provider.describe((missing,), scene=SCENE)
        assert tags.people_count is None
        assert tags.confidence == 0.0

    def test_a_still_outside_the_project_is_skipped_not_fatal(
        self, provider: ClassicalCvVisionProvider
    ) -> None:
        """MediaRef blocks traversal at validation; this covers the second barrier."""
        escaping = Keyframe(
            timestamp=1.0, image=MediaRef.model_construct(path=Path("../outside.jpg"))
        )
        assert provider.describe((escaping,), scene=SCENE).confidence == 0.0

    def test_a_measured_empty_shot_reports_zero_not_none(
        self, provider: ClassicalCvVisionProvider, paths: ProjectPaths
    ) -> None:
        """A flat grey frame is genuinely measured and contains nobody."""
        keyframe = _write_still(paths, "001/0000_0.jpg")
        assert provider.describe((keyframe,), scene=SCENE).people_count == 0


class TestPeopleCounting:
    def test_the_busiest_frame_wins(
        self,
        provider: ClassicalCvVisionProvider,
        paths: ProjectPaths,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Haar misses faces far more often than it invents them, so max beats mean."""
        frames = tuple(_write_still(paths, f"001/0000_{index}.jpg") for index in range(3))
        counts = iter([1, 3, 0])
        monkeypatch.setattr(
            provider,
            "detect_faces",
            lambda _image: [FaceDetection(x=0, y=0, width=40, height=40)] * next(counts),
        )
        assert provider.describe(frames, scene=SCENE).people_count == 3

    def test_confidence_rises_when_a_face_is_found(
        self,
        provider: ClassicalCvVisionProvider,
        paths: ProjectPaths,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        keyframe = _write_still(paths, "001/0000_0.jpg")
        monkeypatch.setattr(
            provider, "detect_faces", lambda _image: [FaceDetection(x=0, y=0, width=40, height=40)]
        )
        with_face = provider.describe((keyframe,), scene=SCENE)

        monkeypatch.setattr(provider, "detect_faces", lambda _image: [])
        without_face = provider.describe((keyframe,), scene=SCENE)

        assert with_face.confidence > without_face.confidence


class TestShotType:
    def _image(self) -> np.ndarray:
        return np.full((300, 500, 3), 128, dtype=np.uint8)

    @pytest.mark.parametrize(
        ("face_height", "expected"),
        [
            (150, ShotType.CLOSE_UP),  # 0.50 of frame height, above close_up 0.35
            (60, ShotType.MEDIUM),  # 0.20, above medium 0.15
            (20, ShotType.WIDE),  # 0.067, below medium
        ],
    )
    def test_framing_is_inferred_from_face_size(
        self,
        provider: ClassicalCvVisionProvider,
        monkeypatch: pytest.MonkeyPatch,
        face_height: int,
        expected: ShotType,
    ) -> None:
        monkeypatch.setattr(
            provider,
            "detect_faces",
            lambda _image: [FaceDetection(x=0, y=0, width=face_height, height=face_height)],
        )
        assert provider.infer_shot_type(self._image()) is expected

    def test_no_face_means_unknown_not_a_guess(
        self, provider: ClassicalCvVisionProvider, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The common case for B-roll. Without a subject of known size there is no scale
        reference in a single 2D frame, so inventing one would mislabel most footage."""
        monkeypatch.setattr(provider, "detect_faces", lambda _image: [])
        assert provider.infer_shot_type(self._image()) is ShotType.UNKNOWN

    def test_the_largest_face_sets_the_framing(
        self, provider: ClassicalCvVisionProvider, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A close-up with a bystander behind is still a close-up."""
        monkeypatch.setattr(
            provider,
            "detect_faces",
            lambda _image: [
                FaceDetection(x=0, y=0, width=20, height=20),
                FaceDetection(x=100, y=0, width=150, height=150),
            ],
        )
        assert provider.infer_shot_type(self._image()) is ShotType.CLOSE_UP

    def test_the_tightest_framing_across_a_scene_wins(
        self,
        provider: ClassicalCvVisionProvider,
        paths: ProjectPaths,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A scene containing one close-up is editorially a close-up scene."""
        frames = tuple(_write_still(paths, f"001/0000_{index}.jpg") for index in range(2))
        sizes = iter([20, 150])
        monkeypatch.setattr(
            provider,
            "detect_faces",
            lambda _image: [FaceDetection(x=0, y=0, width=(size := next(sizes)), height=size)],
        )
        assert provider.shot_type_for(frames) is ShotType.CLOSE_UP

    def test_no_keyframes_yields_unknown(self, provider: ClassicalCvVisionProvider) -> None:
        assert provider.shot_type_for(()) is ShotType.UNKNOWN

    def test_a_zero_height_frame_does_not_divide_by_zero(
        self, provider: ClassicalCvVisionProvider, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            provider, "detect_faces", lambda _image: [FaceDetection(x=0, y=0, width=1, height=1)]
        )
        empty = np.zeros((0, 0, 3), dtype=np.uint8)
        assert provider.infer_shot_type(empty) is ShotType.UNKNOWN


class TestCascadeLoading:
    def test_a_missing_cascade_degrades_to_no_detection(
        self, paths: ProjectPaths, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Absence must not abort a forty-clip analysis."""
        provider = ClassicalCvVisionProvider(
            VisionSettings(face_cascade="haarcascade_does_not_exist.xml"),
            project_root=paths.root,
        )
        assert provider.detect_faces(np.full((180, 320, 3), 128, dtype=np.uint8)) == []

    def test_the_failure_is_cached_so_it_costs_one_attempt(self, paths: ProjectPaths) -> None:
        """Otherwise a missing cascade would be retried once per scene."""
        provider = ClassicalCvVisionProvider(
            VisionSettings(face_cascade="nope.xml"), project_root=paths.root
        )
        image = np.full((180, 320, 3), 128, dtype=np.uint8)
        provider.detect_faces(image)
        assert provider._cascade_failed is True
        assert provider.detect_faces(image) == []

    def test_the_real_bundled_cascade_loads(self, paths: ProjectPaths) -> None:
        """Guards the opencv<5 pin: OpenCV 5 ships no cascades at all.

        If this fails, the classical provider has silently stopped counting people.
        """
        provider = ClassicalCvVisionProvider(VisionSettings(), project_root=paths.root)
        assert provider._load_cascade() is not None

    def test_detection_returns_boxes_on_a_real_image(self, paths: ProjectPaths) -> None:
        """Not asserting on faces found - a grey frame has none - only that it runs."""
        provider = ClassicalCvVisionProvider(VisionSettings(), project_root=paths.root)
        found = provider.detect_faces(np.full((240, 320, 3), 128, dtype=np.uint8))
        assert isinstance(found, list)


class TestReadImage:
    def test_a_real_image_is_loaded(self, tmp_path: Path) -> None:
        import cv2

        destination = tmp_path / "still.jpg"
        cv2.imwrite(str(destination), np.full((60, 80, 3), 200, dtype=np.uint8))
        image = read_image(destination)
        assert image is not None
        assert image.shape[:2] == (60, 80)

    def test_a_missing_file_is_none(self, tmp_path: Path) -> None:
        assert read_image(tmp_path / "absent.jpg") is None

    def test_a_file_that_is_not_an_image_is_none(self, tmp_path: Path) -> None:
        junk = tmp_path / "notanimage.jpg"
        junk.write_text("plain text", encoding="utf-8")
        assert read_image(junk) is None


class TestFaceDetectionModel:
    def test_area(self) -> None:
        assert FaceDetection(x=0, y=0, width=10, height=20).area == 200
