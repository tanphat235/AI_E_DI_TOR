"""Video and scene analysis results (Phase 3 to Phase 6 output).

The unit of editing is the **scene**, not the file. A five-minute raw take
contains a dozen usable moments and a lot of nothing; scene detection finds the
boundaries, quality scoring ranks what is inside them, and the AI director picks
from the survivors.

Every score in this module is normalised so that **1.0 is always better than
0.0**, whatever the underlying measurement's natural direction. Blur is the
obvious trap: the raw Laplacian variance goes *up* for sharper frames, so the
analyser inverts and normalises it before it ever reaches a model.
"""

from __future__ import annotations

from datetime import datetime
from typing import Self

from pydantic import Field, model_validator

from app.models.common import (
    AiveModel,
    CameraMove,
    MediaRef,
    MotionLevel,
    Score,
    Seconds,
    ShotType,
    TimeRange,
)
from app.models.media import MediaProbe


class Keyframe(AiveModel):
    """A representative still extracted from a scene.

    Keyframes exist so that a future vision provider - CLIP, YOLO, or a
    multimodal model - has something to look at without decoding video again, and
    so the desktop UI can show a filmstrip cheaply.
    """

    timestamp: Seconds = Field(description="Position in the *source clip*, in seconds.")
    image: MediaRef = Field(description="Extracted still, under the project cache dir.")


class QualityScores(AiveModel):
    """Per-scene technical quality, all normalised to 0.0 (worst) to 1.0 (best)."""

    blur: Score = Field(description="1.0 is tack sharp, 0.0 is unusably soft.")
    brightness: Score = Field(
        description="1.0 is well exposed; both dark and blown-out fall toward 0.0."
    )
    exposure: Score = Field(
        description="Histogram health - penalises clipped shadows and highlights."
    )
    stability: Score = Field(description="1.0 is locked off; 0.0 is unwatchable shake.")
    overall: Score = Field(
        description=(
            "Weighted aggregate the Rule Engine thresholds against. Stored rather "
            "than recomputed so that a plan reviewed today shows the same numbers "
            "tomorrow, even if the weighting is retuned."
        ),
    )
    blur_variance: float | None = Field(
        default=None,
        ge=0.0,
        description=(
            "Raw Laplacian variance, before normalisation. Kept because the blur "
            "*score* depends on a configured reference: retuning that reference would "
            "otherwise mean re-decoding every clip to recompute it."
        ),
    )


class MotionStats(AiveModel):
    """How much, and what kind of, movement a scene contains.

    Motion drives pacing decisions. A high-motion drone push holds attention for
    eight seconds; a static talking head over the same narration does not.
    """

    level: MotionLevel
    mean_magnitude: float = Field(
        ge=0.0,
        description="Mean optical-flow magnitude in pixels per frame at analysis scale.",
    )
    camera_move: CameraMove = CameraMove.UNKNOWN
    shake: Score = Field(
        default=0.0,
        description="High-frequency jitter, separated from intentional camera movement.",
    )


class SceneTags(AiveModel):
    """Semantic content of a scene.

    ``provider`` is the seam that makes vision understanding replaceable. A
    classical-CV baseline fills in ``people_count`` from a face detector and
    leaves ``objects`` empty; a later CLIP or multimodal provider populates all of
    it. Consumers never branch on which one ran - they read ``confidence`` and
    treat empty collections as "unknown", not "absent".
    """

    provider: str = Field(
        min_length=1,
        description="Which VisionProvider produced these tags, e.g. 'classical_cv'.",
    )
    people_count: int | None = Field(
        default=None,
        ge=0,
        description="None means not measured, 0 means measured and nobody present.",
    )
    objects: tuple[str, ...] = ()
    actions: tuple[str, ...] = ()
    setting: str | None = Field(default=None, description="e.g. 'garden', 'kitchen', 'street'.")
    mood: str | None = None
    caption: str | None = Field(
        default=None,
        description="Free-text description, when the provider can generate one.",
    )
    confidence: Score = Field(default=0.0, description="Provider's own confidence in these tags.")


class Scene(AiveModel):
    """A continuous shot within one raw clip - the atom the director selects."""

    clip: MediaRef
    index: int = Field(ge=0, description="Scene position within its clip.")
    range: TimeRange = Field(description="Bounds within the *source clip*, in seconds.")
    keyframes: tuple[Keyframe, ...] = ()
    quality: QualityScores
    motion: MotionStats
    tags: SceneTags
    shot_type: ShotType = ShotType.UNKNOWN
    phash: str | None = Field(
        default=None,
        description=(
            "Perceptual hash of the middle keyframe, as hex. Used for duplicate "
            "detection: two takes of the same shot hash close together even when "
            "their timestamps and exposure differ slightly."
        ),
    )

    @property
    def key(self) -> str:
        """Stable identifier, formatted ``<clip-stem>#<index>``, e.g. ``001#3``.

        Derived rather than stored so it can never drift out of sync with the
        clip and index it describes.
        """
        return f"{self.clip.stem}#{self.index}"


class ClipAnalysis(AiveModel):
    """Everything AIVE knows about one raw video file."""

    clip: MediaRef
    probe: MediaProbe
    scenes: tuple[Scene, ...] = ()
    analyzer_version: str = Field(
        min_length=1,
        description=(
            "Version of the analysis pipeline that produced this. Cached results "
            "from an older analyser must be invalidated, not trusted."
        ),
    )
    analyzed_at: datetime

    @model_validator(mode="after")
    def _validate_scenes(self) -> Self:
        previous: Scene | None = None
        for scene in self.scenes:
            if scene.clip != self.clip:
                msg = f"scene {scene.key} belongs to {scene.clip}, not {self.clip}"
                raise ValueError(msg)
            if previous is not None and scene.range.start < previous.range.start:
                msg = f"scenes must be in ascending time order: {scene.key} is out of order"
                raise ValueError(msg)
            previous = scene
        return self

    @property
    def usable_duration(self) -> float:
        """Total duration covered by detected scenes."""
        return sum(scene.range.duration for scene in self.scenes)


class DuplicateGroup(AiveModel):
    """A set of scenes judged to be the same shot.

    Only ``representative`` survives selection. Keeping the losers listed rather
    than deleting them matters: if the representative turns out to be unusable for
    an unrelated reason, the alternatives are still on the table.
    """

    representative: str = Field(description="Scene key that survives, e.g. '001#3'.")
    duplicates: tuple[str, ...] = Field(min_length=1, description="Suppressed scene keys.")
    similarity: Score = Field(description="Similarity between representative and duplicates.")

    @model_validator(mode="after")
    def _representative_not_duplicated(self) -> Self:
        if self.representative in self.duplicates:
            msg = f"representative {self.representative!r} must not appear in duplicates"
            raise ValueError(msg)
        return self


class ClipFailure(AiveModel):
    """A clip that could not be analysed, and why.

    Recorded rather than dropped. A footage document that silently omitted three of
    forty clips would read as "these are all your options", and the director would
    plan around footage it was never told about.
    """

    clip: MediaRef
    error: str = Field(min_length=1, description="What went wrong, for the user to act on.")


class FootageAnalysis(AiveModel):
    """The aggregate view of every raw clip in a project.

    This is the document the AI director reads before authoring a plan. It is
    written to the project cache in full and summarised to stdout as a digest,
    because the full form runs to tens of thousands of tokens on a real project.
    """

    clips: tuple[ClipAnalysis, ...] = ()
    duplicates: tuple[DuplicateGroup, ...] = ()
    failed: tuple[ClipFailure, ...] = Field(
        default=(),
        description="Clips that could not be analysed. Reported, never silently omitted.",
    )

    @property
    def scenes(self) -> tuple[Scene, ...]:
        """Every scene from every clip, clip order preserved."""
        return tuple(scene for clip in self.clips for scene in clip.scenes)

    def scene_by_key(self, key: str) -> Scene | None:
        """Look up a scene by its ``<clip-stem>#<index>`` key."""
        return next((scene for scene in self.scenes if scene.key == key), None)

    @property
    def suppressed_scene_keys(self) -> frozenset[str]:
        """Scene keys ruled out as duplicates of a better take."""
        return frozenset(key for group in self.duplicates for key in group.duplicates)


__all__ = [
    "ClipAnalysis",
    "DuplicateGroup",
    "FootageAnalysis",
    "Keyframe",
    "MotionStats",
    "QualityScores",
    "Scene",
    "SceneTags",
]
