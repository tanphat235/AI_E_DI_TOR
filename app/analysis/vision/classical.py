"""Classical-CV vision provider: the honest baseline.

This is the default :class:`~app.analysis.vision.base.VisionProvider`, and its defining
constraint is that it **downloads nothing**. Everything here runs on cascades bundled
inside the OpenCV wheel, so a fresh install can analyse footage offline with no model
fetch and no GPU.

What that buys, and what it costs:

* It can **count faces**, so ``people_count`` is real.
* It can **infer shot type from face size** - a face filling a third of the frame is a
  close-up, a small one is a wide shot. This is the one genuinely useful semantic signal
  classical CV provides.
* It **cannot name objects, actions or settings.** Those fields stay empty.

That last point is the important one, and it is deliberate. An empty collection means
"unknown", and a consumer treats it as a gap. A *guessed* tag would be indistinguishable
from a real one, and the director would place footage based on it. A provider that
invents "garden" from a green histogram is worse than one that says nothing, because
wrong metadata is acted upon while missing metadata is worked around.

The upgrade path is a different provider, not more heuristics here: CLIP for
scene-text similarity, YOLO for objects, or a multimodal model for captions. Each would
populate the same :class:`~app.models.video.SceneTags` and declare a different
``provider`` name, so a project can mix them.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from app.analysis.vision.frames import to_grayscale
from app.config.settings import VisionSettings
from app.models.common import ShotType, TimeRange
from app.models.video import Keyframe, SceneTags
from app.utils.logging import get_logger

if TYPE_CHECKING:  # pragma: no cover
    import numpy as np

logger = get_logger(__name__)

PROVIDER_NAME = "classical_cv"


@dataclass(frozen=True, slots=True)
class FaceDetection:
    """A detected face box, in pixels of the frame it was found in."""

    x: int
    y: int
    width: int
    height: int

    @property
    def area(self) -> int:
        return self.width * self.height


class ClassicalCvVisionProvider:
    """Face-based scene tagging using only OpenCV's bundled cascades."""

    def __init__(self, settings: VisionSettings, *, project_root: Path) -> None:
        """Args:
        settings: Vision configuration.
        project_root: Needed because ``Keyframe.image`` is project-relative - that
            relativity is what makes a plan portable - while opening a file needs an
            absolute path. Resolving here keeps the ``VisionProvider`` protocol free
            of filesystem concerns.
        """
        self._settings = settings
        self._root = project_root
        self._cascade: Any | None = None
        self._cascade_failed = False

    @property
    def name(self) -> str:
        return PROVIDER_NAME

    def describe(self, keyframes: tuple[Keyframe, ...], *, scene: TimeRange) -> SceneTags:
        """Tag a scene from its keyframes.

        Operates on the written stills rather than on video, so this provider - and any
        replacement - is a pure image-in, tags-out function with no decoder in it.
        """
        images = self._load_all(keyframes)
        if not images:
            # Nothing to look at. Report unknown rather than "nobody present".
            return SceneTags(provider=self.name, confidence=0.0)

        # The maximum, not the mean: Haar cascades miss faces on any given frame far more
        # often than they invent one, so the busiest frame is the best estimate of how
        # many people are actually in the shot.
        people = max(len(self.detect_faces(image)) for image in images)

        return SceneTags(
            provider=self.name,
            people_count=people,
            # Deliberately empty: see the module docstring. This provider cannot name
            # objects, actions or settings, and guessing would be worse than silence.
            objects=(),
            actions=(),
            setting=None,
            mood=None,
            caption=None,
            confidence=0.6 if people > 0 else 0.3,
        )

    def shot_type_for(self, keyframes: tuple[Keyframe, ...]) -> ShotType:
        """Best shot-type reading across a scene's keyframes.

        Takes the *tightest* framing seen: a scene containing one close-up is
        editorially a close-up scene even when other frames are wider.
        """
        ranking = {
            ShotType.CLOSE_UP: 3,
            ShotType.MEDIUM: 2,
            ShotType.WIDE: 1,
            ShotType.UNKNOWN: 0,
        }
        best = ShotType.UNKNOWN
        for image in self._load_all(keyframes):
            candidate = self.infer_shot_type(image)
            if ranking[candidate] > ranking[best]:
                best = candidate
        return best

    def infer_shot_type(self, image: np.ndarray) -> ShotType:
        """Classify framing from the largest face relative to the frame.

        Returns ``UNKNOWN`` when there is no face, which is the common case for B-roll.
        That is the honest answer: without a subject of known size there is no scale
        reference in a single 2D frame, and inventing one would mislabel most footage.
        """
        faces = self.detect_faces(image)
        if not faces:
            return ShotType.UNKNOWN

        frame_height = image.shape[0]
        if frame_height <= 0:
            return ShotType.UNKNOWN

        largest = max(faces, key=lambda face: face.area)
        ratio = largest.height / float(frame_height)
        settings = self._settings
        if ratio >= settings.close_up_face_ratio:
            return ShotType.CLOSE_UP
        if ratio >= settings.medium_face_ratio:
            return ShotType.MEDIUM
        return ShotType.WIDE

    def detect_faces(self, image: np.ndarray) -> list[FaceDetection]:
        """Detect frontal faces. Returns an empty list when detection is unavailable."""
        cascade = self._load_cascade()
        if cascade is None:
            return []

        grey = to_grayscale(image)
        height = grey.shape[0]
        minimum = max(16, int(height * self._settings.face_min_size_ratio))
        try:
            found = cascade.detectMultiScale(
                grey,
                scaleFactor=1.1,
                # A face must be found at several scales to count. Haar's false-positive
                # rate at minNeighbors=3 is high enough to inflate people_count on
                # textured footage such as foliage.
                minNeighbors=5,
                minSize=(minimum, minimum),
            )
        except Exception as exc:
            logger.debug("Face detection failed: %s", exc)
            return []

        return [
            FaceDetection(x=int(x), y=int(y), width=int(w), height=int(h)) for x, y, w, h in found
        ]

    # -- Cascade loading ----------------------------------------------------- #

    def _load_cascade(self) -> Any | None:
        """Load the bundled cascade, once, tolerating its absence.

        Cached including the *failure*, so a missing cascade costs one attempt rather
        than one per scene. Absence is survivable: the provider degrades to reporting
        nothing rather than aborting the analysis.
        """
        if self._cascade is not None:
            return self._cascade
        if self._cascade_failed:
            return None

        try:
            import cv2
        except ImportError:
            self._cascade_failed = True
            return None

        # OpenCV 5 removed CascadeClassifier entirely and ships no cascades; the
        # dependency is pinned below 5 for this reason, but a user with a mixed
        # environment should get a clear log line rather than a crash.
        if not hasattr(cv2, "CascadeClassifier"):
            logger.warning(
                "This OpenCV build has no CascadeClassifier, so faces cannot be counted. "
                'Install the 4.x line: pip install "opencv-python>=4.10,<5"'
            )
            self._cascade_failed = True
            return None

        path = Path(cv2.data.haarcascades) / self._settings.face_cascade  # type: ignore[attr-defined]
        if not path.is_file():
            logger.warning("Face cascade not found at %s; faces will not be counted", path)
            self._cascade_failed = True
            return None

        cascade = cv2.CascadeClassifier(str(path))
        if cascade.empty():
            logger.warning("Face cascade at %s could not be loaded", path)
            self._cascade_failed = True
            return None

        logger.debug("Loaded face cascade %s", path.name)
        self._cascade = cascade
        return cascade

    def _load_all(self, keyframes: tuple[Keyframe, ...]) -> list[np.ndarray]:
        """Read every keyframe that can be read, skipping any that cannot.

        A missing still is survivable - the scene is simply described from fewer frames
        - so this filters rather than raising.
        """
        images: list[np.ndarray] = []
        for keyframe in keyframes:
            try:
                path = keyframe.image.resolve_within(self._root)
            except ValueError:
                logger.debug("Skipping keyframe outside the project: %s", keyframe.image)
                continue
            image = read_image(path)
            if image is not None:
                images.append(image)
        return images


def read_image(path: Path) -> np.ndarray | None:
    """Load an image, or ``None`` if it is missing or unreadable."""
    try:
        import cv2
    except ImportError:  # pragma: no cover
        return None
    image = cv2.imread(str(path))
    return image if image is not None and image.size else None


__all__ = [
    "PROVIDER_NAME",
    "ClassicalCvVisionProvider",
    "FaceDetection",
    "read_image",
]
