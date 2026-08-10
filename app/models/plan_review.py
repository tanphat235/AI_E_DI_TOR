"""Reviewing a finished Edit Plan.

A plan is JSON, and JSON is not how anyone judges an edit. These models describe a plan the
way an editor would ask about it: how fast is it cutting, is the rhythm mechanical, does the
same framing repeat, how much of the available footage did it actually use, and does the
picture cover the narration.

The statistics are measurements. The **notes** are where the value is: each one is a pattern
that is invisible in a list of clips and obvious in a finished video. Twelve clips of
identical length reads as a metronome. Four consecutive wides read as laziness. Three scenes
used out of forty means the edit ignored the footage.

None of these are errors - the Rule Engine already covers admissibility. They are the
observations a second pair of eyes would make, which is exactly what a director doing a
second pass, or a user reviewing what the director did, needs.
"""

from __future__ import annotations

from pydantic import Field

from app.models.common import AiveModel, Issue, MediaRef, ShotType, TimeRange, TransitionKind


class TimelineEntry(AiveModel):
    """One clip as it appears on the timeline, for a readable rendering."""

    clip_id: str
    order: int = Field(ge=0)
    source: MediaRef
    source_range: TimeRange
    timeline_range: TimeRange | None = Field(
        default=None, description="None when the plan has not been normalised yet."
    )
    transition_in: TransitionKind = TransitionKind.CUT
    transition_duration: float = Field(default=0.0, ge=0.0)
    shot_type: ShotType = ShotType.UNKNOWN
    scene_key: str | None = None
    beat_index: int | None = None
    reason: str = Field(description="Why this clip is here, as the author wrote it.")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)

    @property
    def duration(self) -> float:
        return self.source_range.duration


class PlanStatistics(AiveModel):
    """Measurements of a plan's shape."""

    clip_count: int = Field(ge=0)
    total_duration: float = Field(ge=0.0)

    mean_clip_duration: float = Field(ge=0.0)
    median_clip_duration: float = Field(ge=0.0)
    shortest_clip: float = Field(ge=0.0)
    longest_clip: float = Field(ge=0.0)

    cuts_per_minute: float = Field(
        ge=0.0,
        description=(
            "Pacing, in cuts per minute. Roughly 6-12 reads as a relaxed vlog, 20-40 as "
            "energetic, above 60 as a montage. A number, not a target."
        ),
    )
    transition_counts: dict[str, int] = Field(
        default_factory=dict,
        description="How many of each transition kind, keyed by name. Hard cuts included.",
    )
    shot_type_counts: dict[str, int] = Field(default_factory=dict)

    distinct_sources: int = Field(ge=0, description="How many source files the plan draws on.")
    distinct_scenes: int = Field(ge=0, description="How many analysed scenes it uses.")
    longest_same_shot_run: int = Field(
        default=0,
        ge=0,
        description=(
            "Longest run of consecutive clips sharing one shot type. Three or more is the "
            "signature of an edit that stopped varying its framing."
        ),
    )

    narration_duration: float | None = Field(
        default=None, description="None when the plan has no narration track."
    )
    subtitle_count: int = Field(default=0, ge=0)

    @property
    def coverage_delta(self) -> float | None:
        """Picture duration minus narration duration.

        Negative means the video ends on black with someone still talking.
        """
        if self.narration_duration is None:
            return None
        return self.total_duration - self.narration_duration


class PlanReview(AiveModel):
    """A plan rendered for a human, plus the observations worth making about it."""

    project_id: str
    created_by: str
    is_placed: bool = Field(description="Whether clips have timeline positions yet.")
    statistics: PlanStatistics
    timeline: tuple[TimelineEntry, ...] = ()
    notes: tuple[Issue, ...] = Field(
        default=(),
        description=(
            "Editorial observations, never admissibility errors - the Rule Engine owns "
            "those. These are the things a second pair of eyes would mention."
        ),
    )
    plan_notes: str | None = Field(
        default=None, description="The author's own rationale, from the plan."
    )


class ClipChange(AiveModel):
    """How one clip differs between two plans."""

    clip_id: str
    fields: tuple[str, ...] = Field(
        min_length=1, description="Which fields differ, e.g. ('source_range', 'reason')."
    )
    before: str = Field(description="A short rendering of the old value.")
    after: str = Field(description="A short rendering of the new value.")


class PlanDiff(AiveModel):
    """What changed between two plans.

    Built for the director's second pass and for a user asking "what did it change?". Keyed
    on ``clip.id`` rather than position, because inserting one clip at the front would
    otherwise report every subsequent clip as modified.
    """

    added: tuple[str, ...] = Field(default=(), description="Clip ids only in the new plan.")
    removed: tuple[str, ...] = Field(default=(), description="Clip ids only in the old plan.")
    changed: tuple[ClipChange, ...] = ()
    reordered: bool = Field(
        default=False, description="Whether the surviving clips appear in a different order."
    )
    duration_before: float = Field(ge=0.0)
    duration_after: float = Field(ge=0.0)
    summary_fields: tuple[str, ...] = Field(
        default=(),
        description="Plan-level fields that differ, e.g. ('notes', 'output', 'subtitles').",
    )

    @property
    def is_identical(self) -> bool:
        return not (
            self.added or self.removed or self.changed or self.reordered or self.summary_fields
        )

    @property
    def duration_delta(self) -> float:
        return self.duration_after - self.duration_before


__all__ = [
    "ClipChange",
    "PlanDiff",
    "PlanReview",
    "PlanStatistics",
    "TimelineEntry",
]
