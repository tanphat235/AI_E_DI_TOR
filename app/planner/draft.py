"""A heuristic baseline plan.

This is **not** the AI Planner. It is a starting structure: one clip per surviving beat,
taking each beat's top-ranked candidate, trimmed to the beat's length. It exists for three
reasons.

**It guarantees the director never fights the schema.** A structurally valid plan to revise
beats an empty file, because the shape of the format stops being a question and the
director's whole attention goes to the editing.

**It is the zero-AI fallback.** A user who wants a rough assembly with no model involved
gets one, and it will render.

**It sets a floor to beat.** If a draft and a considered plan are indistinguishable, the
considered plan added nothing — which is a useful thing to be able to notice.

What it cannot do is the actual job. It has no idea whether a shot illustrates a line: it
matches on duration, quality and repetition, because those are the only things measurable
without semantic tags. Its ``reason`` fields say exactly that, so nobody mistakes a
generated justification for an editorial one, and ``created_by`` is ``"heuristic"`` so the
provenance is in the file itself.
"""

from __future__ import annotations

from datetime import UTC, datetime

from app.config.settings import AiveSettings
from app.models.common import MediaRef, TimeRange, TransitionKind
from app.models.edit_plan import EditPlan, NarrationTrack, TimelineClip, Transition
from app.models.planning import BeatCandidates, PlanningBrief
from app.models.speech import NarrationAnalysis
from app.utils.logging import get_logger

logger = get_logger(__name__)

DRAFT_VERSION = "draft/1"
CREATED_BY = "heuristic"
"""Recorded in the plan so its provenance travels with it.

A reviewer seeing ``created_by: "heuristic"`` knows the ``reason`` fields are mechanical.
Had this said ``claude-code`` it would be a lie encoded in the deliverable.
"""

_REASON = (
    "Heuristic pick: highest-ranked candidate by duration, quality and variety. "
    "No semantic matching - revise this."
)


class HeuristicDrafter:
    """Builds a baseline plan from a brief."""

    def __init__(self, settings: AiveSettings) -> None:
        self._settings = settings

    @property
    def version(self) -> str:
        return DRAFT_VERSION

    def draft(
        self,
        brief: PlanningBrief,
        *,
        narration: NarrationAnalysis | None = None,
    ) -> EditPlan:
        """Produce a baseline plan covering every beat that has a candidate.

        Raises:
            ValueError: when no beat has a candidate. An empty plan would fail model
                validation anyway, and a clear message beats a schema error.
        """
        beats = brief.beats_needing_footage
        wanted = self._wanted_durations(beats, brief.coverage.narration_duration)
        clips = [
            clip
            for position, beat in enumerate(beats)
            if (clip := self._clip_for(beat, position, wanted[position])) is not None
        ]
        if not clips:
            msg = (
                "no beat has a usable candidate, so no baseline plan can be built. "
                "Check `aive plan brief` for the coverage errors."
            )
            raise ValueError(msg)

        narration_track: NarrationTrack | None = None
        if narration is not None and narration.cleanup.kept_ranges:
            narration_track = NarrationTrack(
                source=narration.source,
                kept_ranges=narration.cleanup.kept_ranges,
            )

        logger.info("Drafted a baseline plan with %d clip(s)", len(clips))
        return EditPlan(
            project_id=brief.project_id,
            created_by=CREATED_BY,
            created_at=datetime.now(UTC),
            output=self._settings.output.to_output_spec(),
            narration=narration_track,
            clips=tuple(clips),
            notes=(
                "Heuristic baseline, not an edit. Clips were chosen by duration, quality and "
                "shot variety - nothing here reflects what the footage actually shows. "
                "Rewrite the selections and the reason fields before rendering."
            ),
        )

    def _wanted_durations(
        self, beats: tuple[BeatCandidates, ...], narration_duration: float
    ) -> list[float]:
        """How long each clip must be for the picture to run without gaps.

        Not simply each beat's own length. Beats are separated by the pauses that survived
        cleanup - anything shorter than ``min_silence_duration`` stays in the audio - so
        clips sized to the beats alone leave black on screen between every line. Each clip
        is therefore stretched to the *next beat's start*, and the last one to the end of
        the narration.

        Transitions are added back on top, because each one pulls its clip backwards over
        its predecessor and so shortens the total by its own duration.
        """
        if not beats:
            return []

        starts = [beat.timeline_range.start for beat in beats if beat.timeline_range is not None]
        boundaries = [*starts, max(narration_duration, starts[-1] if starts else 0.0)]

        wanted: list[float] = []
        for position in range(len(beats)):
            span = boundaries[position + 1] - boundaries[position]
            transition = self._transition(position)
            overlap = transition.duration if transition is not None else 0.0
            wanted.append(max(0.0, span) + overlap)
        return wanted

    def _clip_for(self, beat: BeatCandidates, position: int, wanted: float) -> TimelineClip | None:
        """One clip covering one beat, or ``None`` when there is nothing to use."""
        best = beat.best
        if best is None or beat.timeline_range is None:
            return None

        source_range = self._trim(best.range, wanted=wanted)
        if source_range is None:
            return None

        return TimelineClip(
            id=f"c{position:03d}",
            source=best.clip,
            source_range=source_range,
            # timeline_start is deliberately omitted: `aive rules normalize` places clips
            # correctly, including transition overlap, and duplicating that arithmetic here
            # would be a second implementation to keep in step.
            transition_in=self._transition(position),
            reason=_REASON,
            confidence=round(best.score, 3),
            scene_key=best.scene_key,
            beat_index=beat.beat_index,
        )

    def _trim(self, scene: TimeRange, *, wanted: float) -> TimeRange | None:
        """Trim a scene to the length a beat needs, centred within the scene.

        Centred rather than taken from the start, because the beginning of a shot is where
        a camera is still settling and an actor is still arriving. The middle is the part
        most likely to be usable.
        """
        rules = self._settings.rules
        target = max(rules.min_clip_duration, min(wanted, rules.max_clip_duration))

        if scene.duration <= target:
            # Shorter than wanted. Use all of it if it clears the floor, else skip it -
            # a clip below the minimum would only be rejected by validation.
            return scene if scene.duration >= rules.min_clip_duration else None

        margin = (scene.duration - target) / 2.0
        return TimeRange(start=scene.start + margin, end=scene.end - margin)

    def _transition(self, position: int) -> Transition | None:
        """The configured transition, except on the first clip.

        A hard cut is the default between shots, and this only applies the configured
        transition — which for a real edit is a decision per cut, not a global setting.
        That is precisely the kind of judgement a baseline cannot make.
        """
        if position == 0:
            return None
        rules = self._settings.rules
        if rules.default_transition is TransitionKind.CUT:
            return None
        return Transition(kind=rules.default_transition, duration=rules.transition_duration)


def sources_used(plan: EditPlan) -> tuple[MediaRef, ...]:
    """Distinct media a plan draws on."""
    return plan.sources


__all__ = ["CREATED_BY", "DRAFT_VERSION", "HeuristicDrafter"]
