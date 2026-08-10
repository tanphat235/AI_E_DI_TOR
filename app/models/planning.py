"""The planning brief: everything the AI director needs to author a plan.

AIVE ships no planner. The director is Claude Code, running outside the app, and its job
is the one thing that genuinely needs judgement: deciding which shot serves which line.

So this module models the *input* to that decision rather than the decision itself. It
gathers what is otherwise scattered across three analysis documents and a config file
into one view: what is being said, which footage may be used, how well each shot suits
each line, and the constraints the finished plan must satisfy.

Two properties are deliberate.

**Candidates are ranked, never chosen.** A forty-beat project against two hundred scenes
is eight thousand pairs, which no reader holds at once. Ranking narrows that to a handful
per beat. But the score is scaffolding: it knows about duration, quality and repetition,
and nothing at all about whether a shot of hands in soil illustrates "prepare the bed".

**Every score is explained.** :attr:`SceneCandidate.reasons` records what raised and
lowered it, so the director can disagree with a ranking rather than either trusting or
ignoring it. An unexplained number would be worse than no number.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import Field

from app.models.common import (
    AiveModel,
    AspectRatio,
    Issue,
    MediaRef,
    MotionLevel,
    Score,
    ShotType,
    TimeRange,
    TransitionKind,
)


class SceneCandidate(AiveModel):
    """One scene offered as a possibility for one beat."""

    scene_key: str = Field(
        min_length=1, description="e.g. '001#3'. Put this in the clip's scene_key."
    )
    clip: MediaRef
    range: TimeRange = Field(description="The scene's bounds in its source file.")
    score: Score = Field(description="Suitability, 0 to 1. Scaffolding, not a verdict.")
    reasons: tuple[str, ...] = Field(
        default=(),
        description=(
            "What raised and lowered the score, in plain words. Present so the ranking can "
            "be argued with: a number with no explanation invites either blind trust or "
            "blanket dismissal, and both are worse than a reader who disagrees."
        ),
    )

    # Repeated from the scene so the brief stands alone. A director reading this should
    # not have to cross-reference footage.json to see whether a shot is sharp.
    quality: Score
    shot_type: ShotType = ShotType.UNKNOWN
    motion: MotionLevel = MotionLevel.STATIC

    @property
    def duration(self) -> float:
        return self.range.duration


class BeatCandidates(AiveModel):
    """One narration beat and the scenes that might cover it."""

    beat_index: int = Field(ge=0)
    text: str = Field(min_length=1, description="What is being said.")
    source_range: TimeRange = Field(description="Where the beat is in the original narration.")
    timeline_range: TimeRange | None = Field(
        default=None,
        description=(
            "Where the beat lands in the finished video. **This is the clock to place "
            "footage against.** None means the beat was cut as a retake or filler and "
            "needs no picture."
        ),
    )
    keywords: tuple[str, ...] = ()
    candidates: tuple[SceneCandidate, ...] = ()

    @property
    def needs_footage(self) -> bool:
        return self.timeline_range is not None

    @property
    def wanted_duration(self) -> float:
        """How long a clip covering this beat should be.

        Zero for a cut beat. Note that a plan need not match beats to clips one-for-one -
        a long beat may want two shots, and two short beats may share one - so this is a
        target rather than a requirement.
        """
        return 0.0 if self.timeline_range is None else self.timeline_range.duration

    @property
    def best(self) -> SceneCandidate | None:
        return self.candidates[0] if self.candidates else None


class PlanConstraints(AiveModel):
    """The rules a finished plan must satisfy.

    Echoed into the brief so the constraints sit beside the choices. Without this the
    director has to remember a separate ``config show``, and a plan that violates a
    threshold it was never shown is a failure of the tool, not of the director.
    """

    min_clip_duration: float
    max_clip_duration: float
    default_transition: TransitionKind
    transition_duration: float
    max_transition_ratio: float
    aspect_ratio: AspectRatio
    width: int
    height: int
    fps: float
    min_overall_quality: float = Field(
        description="Scenes below this were filtered out unless nothing else covers a beat."
    )


class CoverageReport(AiveModel):
    """Whether this project can be edited at all.

    Worth computing before any planning: if there is less usable footage than narration,
    no amount of good judgement produces a complete video, and saying so up front is far
    better than discovering it forty clips into a plan.
    """

    narration_duration: float = Field(
        ge=0.0, description="Length of the narration after cleanup - the video's natural length."
    )
    eligible_footage_duration: float = Field(
        ge=0.0, description="Total duration of scenes that passed the filters."
    )
    beats_total: int = Field(ge=0)
    beats_needing_footage: int = Field(ge=0)
    beats_with_candidates: int = Field(ge=0)
    eligible_scenes: int = Field(ge=0)
    rejected_scenes: int = Field(ge=0)
    issues: tuple[Issue, ...] = ()

    @property
    def feasible(self) -> bool:
        """Whether every beat that needs picture has at least one option.

        Deliberately not a judgement about *quality*. A feasible project can still be a
        poor edit; an infeasible one cannot be finished.
        """
        return (
            self.beats_needing_footage > 0
            and self.beats_with_candidates >= self.beats_needing_footage
        )

    @property
    def footage_ratio(self) -> float:
        """Usable footage divided by narration length.

        Below 1.0 the project cannot be covered without reusing shots. Around 3.0 or
        above gives a director real choice.
        """
        if self.narration_duration <= 0.0:
            return 0.0
        return self.eligible_footage_duration / self.narration_duration

    @property
    def uncovered_beats(self) -> int:
        return max(0, self.beats_needing_footage - self.beats_with_candidates)


class PlanningBrief(AiveModel):
    """The single document a director reads before writing a plan."""

    project_id: str = Field(min_length=1, pattern=r"^[A-Za-z0-9._-]+$")
    created_at: datetime
    brief_version: str = Field(min_length=1, description="Which planner build produced this.")

    coverage: CoverageReport
    constraints: PlanConstraints
    beats: tuple[BeatCandidates, ...] = ()
    unused_scenes: tuple[str, ...] = Field(
        default=(),
        description=(
            "Eligible scenes that were not a candidate for any beat. Usually B-roll with "
            "no keyword overlap - often the most interesting footage in the project, so "
            "it is listed rather than dropped."
        ),
    )
    narration: MediaRef | None = None

    @property
    def beats_needing_footage(self) -> tuple[BeatCandidates, ...]:
        """Beats that survived cleanup and therefore need picture."""
        return tuple(beat for beat in self.beats if beat.needs_footage)

    @property
    def uncovered(self) -> tuple[BeatCandidates, ...]:
        """Beats with no candidate at all - the ones that block a complete edit."""
        return tuple(beat for beat in self.beats_needing_footage if not beat.candidates)


__all__ = [
    "BeatCandidates",
    "CoverageReport",
    "PlanConstraints",
    "PlanningBrief",
    "SceneCandidate",
]
