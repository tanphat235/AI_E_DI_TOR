"""AIVE data models.

The layering is strict and one-directional:

* :mod:`app.models.common` - primitives (time, paths, enums). Depends on nothing.
* :mod:`app.models.media` - ``ffprobe`` facts. Depends on common.
* :mod:`app.models.speech`, :mod:`~app.models.video`, :mod:`~app.models.audio` -
  analysis output. Depend on common and media, never on each other.
* :mod:`app.models.edit_plan` - the contract. Depends only on common, so a
  renderer can consume a plan without importing a single analyser.

That last point is the whole architecture in one sentence: the Edit Plan is
decoupled from how it was produced.
"""

from __future__ import annotations

from app.models.audio import MusicLibrary, MusicTrack
from app.models.common import (
    AiveModel,
    AspectRatio,
    CameraMove,
    Issue,
    MediaKind,
    MediaRef,
    MotionLevel,
    MusicMood,
    Severity,
    ShotType,
    SubtitleFormat,
    TimeRange,
    TransitionKind,
)
from app.models.edit_plan import (
    EDIT_PLAN_SCHEMA_VERSION,
    DuckingSpec,
    EditPlan,
    EditPlanReport,
    FramingSpec,
    MusicCue,
    NarrationTrack,
    OutputSpec,
    SubtitleCue,
    TimelineClip,
    Transition,
)
from app.models.media import AudioStreamInfo, MediaProbe, VideoStreamInfo
from app.models.planning import (
    BeatCandidates,
    CoverageReport,
    PlanConstraints,
    PlanningBrief,
    SceneCandidate,
)
from app.models.project import MANIFEST_SCHEMA_VERSION, MediaEntry, ProjectManifest
from app.models.speech import (
    FillerSpan,
    NarrationBeat,
    RepetitionSpan,
    SilenceSpan,
    SpeechCleanupReport,
    Transcript,
    TranscriptSegment,
    Word,
)
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

__all__ = [
    "EDIT_PLAN_SCHEMA_VERSION",
    "MANIFEST_SCHEMA_VERSION",
    "AiveModel",
    "AspectRatio",
    "AudioStreamInfo",
    "BeatCandidates",
    "CameraMove",
    "ClipAnalysis",
    "CoverageReport",
    "DuckingSpec",
    "DuplicateGroup",
    "EditPlan",
    "EditPlanReport",
    "FillerSpan",
    "FootageAnalysis",
    "FramingSpec",
    "Issue",
    "Keyframe",
    "MediaEntry",
    "MediaKind",
    "MediaProbe",
    "MediaRef",
    "MotionLevel",
    "MotionStats",
    "MusicCue",
    "MusicLibrary",
    "MusicMood",
    "MusicTrack",
    "NarrationBeat",
    "NarrationTrack",
    "OutputSpec",
    "PlanConstraints",
    "PlanningBrief",
    "ProjectManifest",
    "QualityScores",
    "RepetitionSpan",
    "Scene",
    "SceneCandidate",
    "SceneTags",
    "Severity",
    "ShotType",
    "SilenceSpan",
    "SpeechCleanupReport",
    "SubtitleCue",
    "SubtitleFormat",
    "TimeRange",
    "TimelineClip",
    "Transcript",
    "TranscriptSegment",
    "Transition",
    "TransitionKind",
    "VideoStreamInfo",
    "Word",
]
