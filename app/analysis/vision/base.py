"""Video and vision analysis boundaries (implemented in Phases 3 to 6).

Three separate protocols, because they have genuinely different costs and change
at genuinely different rates:

* :class:`SceneDetector` — cheap, stable, and solved. PySceneDetect does this well
  and there is little reason to replace it.
* :class:`QualityAnalyzer` — cheap classical CV. Blur, exposure, shake. The
  numbers may be retuned but the approach will not change.
* :class:`VisionProvider` — the expensive, fast-moving one. Today a classical-CV
  baseline that downloads no weights; later CLIP, YOLO, or a multimodal model.

Splitting them is what makes vision understanding replaceable without touching
scene detection, and lets a project mix providers - classical tags for forty clips,
a richer provider for the six that matter.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from app.models.common import MediaRef, TimeRange
from app.models.media import MediaProbe
from app.models.video import (
    ClipAnalysis,
    DuplicateGroup,
    FootageAnalysis,
    Keyframe,
    MotionStats,
    QualityScores,
    Scene,
    SceneTags,
)


@runtime_checkable
class MediaProber(Protocol):
    """Reads container metadata: duration, resolution, frame rate, streams.

    A distinct protocol because there are two viable backends with different
    trade-offs - the ``ffprobe`` binary, and PyAV in process - and the vendored
    FFmpeg wheel ships no ffprobe. See
    :mod:`app.services.ffmpeg_locator` for why that matters.
    """

    def probe(self, media: Path, *, ref: MediaRef) -> MediaProbe:
        """Read metadata for ``media``."""
        ...


@runtime_checkable
class SceneDetector(Protocol):
    """Finds shot boundaries within a single video file."""

    def detect(self, video: Path, *, probe: MediaProbe) -> tuple[TimeRange, ...]:
        """Return scene boundaries, in ascending order and covering no gaps.

        Implementations must respect ``vision.min_scene_duration`` by merging
        short detections into a neighbour rather than emitting them, because a
        0.2-second "scene" is a detector artefact, not a shot.
        """
        ...


@runtime_checkable
class KeyframeExtractor(Protocol):
    """Pulls representative stills out of a scene."""

    def extract(
        self,
        video: Path,
        *,
        scene: TimeRange,
        destination: Path,
        count: int,
    ) -> tuple[Keyframe, ...]:
        """Write up to ``count`` stills for ``scene`` into ``destination``."""
        ...


@runtime_checkable
class QualityAnalyzer(Protocol):
    """Scores the technical quality of a scene.

    Every returned score must be normalised so 1.0 is best, whatever the natural
    direction of the underlying measurement. Blur is the trap: raw Laplacian
    variance rises with sharpness, so it must be normalised here and not left for
    each consumer to remember.
    """

    def score(self, video: Path, *, scene: TimeRange, probe: MediaProbe) -> QualityScores:
        """Measure blur, brightness, exposure and stability across ``scene``."""
        ...

    def motion(self, video: Path, *, scene: TimeRange, probe: MediaProbe) -> MotionStats:
        """Measure movement and classify the camera move across ``scene``."""
        ...


@runtime_checkable
class VisionProvider(Protocol):
    """Describes the *content* of a scene: people, objects, actions, setting.

    The replaceable seam. Implementations declare themselves through
    :attr:`name`, which is recorded in
    :attr:`~app.models.video.SceneTags.provider`, so a plan can be traced back to
    whichever provider informed it.

    Implementations must degrade honestly. A classical-CV provider that can count
    faces but cannot name objects returns ``people_count`` set and ``objects``
    empty - never a guess. Consumers read an empty collection as "unknown", so a
    fabricated tag is worse than no tag.
    """

    @property
    def name(self) -> str:
        """Provider identity, e.g. ``classical_cv``, ``clip-vit-b32``."""
        ...

    def describe(self, keyframes: tuple[Keyframe, ...], *, scene: TimeRange) -> SceneTags:
        """Produce semantic tags for a scene from its keyframes.

        Keyframes rather than the video itself: it keeps this protocol free of
        video decoding, which means a future provider can be a pure
        image-in/tags-out function.
        """
        ...


@runtime_checkable
class DuplicateDetector(Protocol):
    """Groups scenes that are the same shot.

    Duplicate removal is what stops an automated edit from cutting between three
    takes of one sentence, which is the single most obvious tell that a human was
    not involved.
    """

    def find_duplicates(self, scenes: tuple[Scene, ...]) -> tuple[DuplicateGroup, ...]:
        """Group near-identical scenes, choosing the best of each group."""
        ...


@runtime_checkable
class FootageAnalyzer(Protocol):
    """Orchestrates the whole per-clip pipeline.

    The façade the CLI actually calls. It composes the protocols above; keeping it
    separate means the composition order is testable without any real media.
    """

    @property
    def version(self) -> str:
        """Pipeline version stamped into :attr:`~app.models.video.ClipAnalysis.analyzer_version`.

        Cached results from an older pipeline must be invalidated rather than
        trusted, and this is how that is detected.
        """
        ...

    def analyze_clip(self, video: Path, *, ref: MediaRef) -> ClipAnalysis:
        """Probe, detect scenes, extract keyframes, score, and tag one clip."""
        ...

    def analyze_project(
        self,
        clips: list[Path],
        *,
        cached: dict[MediaRef, ClipAnalysis] | None = None,
    ) -> FootageAnalysis:
        """Analyse every clip, then find duplicates across all of them.

        Part of the protocol rather than an implementation detail because duplicate
        detection is inherently cross-clip: adding one clip can create a duplicate of an
        existing one, so it cannot be done per clip and cached alongside it.

        ``cached`` supplies analyses to reuse, keyed by clip. Implementations must still
        run duplicate detection over the *whole* set, including reused entries.
        """
        ...


__all__ = [
    "DuplicateDetector",
    "FootageAnalyzer",
    "KeyframeExtractor",
    "MediaProber",
    "QualityAnalyzer",
    "SceneDetector",
    "VisionProvider",
]
