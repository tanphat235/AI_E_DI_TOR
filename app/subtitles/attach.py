"""Putting subtitle cues into an Edit Plan.

This closes a hole in the contract. :attr:`app.models.edit_plan.EditPlan.subtitles` has
existed since Phase 1 and both the renderer and the CapCut exporter read it, but nothing
ever wrote to it: ``aive subtitle build`` produced ``.srt`` and ``.ass`` files and stopped
there. A plan that renders with subtitles and a plan that does not looked identical.

The distinction that matters is *whose clock* the cues are on. A plan carries its own
``narration.kept_ranges``, and those - not the cleanup report the transcript came with - are
what the finished video is cut to. When a director has trimmed the narration further, or
dropped a section entirely, the plan is authoritative and the cleanup report is stale. So
cues are always mapped against the plan.
"""

from __future__ import annotations

from app.config.settings import SubtitleSettings
from app.models.edit_plan import EditPlan
from app.models.speech import Transcript
from app.subtitles.builder import SubtitleBuilder
from app.utils.logging import get_logger

logger = get_logger(__name__)


def attach_subtitles(
    plan: EditPlan,
    transcript: Transcript,
    *,
    settings: SubtitleSettings,
    include_word_timings: bool | None = None,
) -> tuple[EditPlan, int]:
    """Return a copy of ``plan`` carrying subtitle cues, and how many were added.

    Args:
        plan: The plan to attach to. Its ``narration.kept_ranges`` define the timeline.
        transcript: Recognition result the cues come from.
        settings: Line width, cue duration bounds, and whether karaoke is wanted.
        include_word_timings: Attach per-word timing. Defaults to whether ``karaoke`` is
            enabled, since nothing else consumes it and building it is not free.

    Cues already on the plan are **replaced**, not merged. Two sets of cues for the same
    words would both render, and a director re-running this after editing the narration
    wants the new timing rather than both.
    """
    kept = plan.narration.kept_ranges if plan.narration is not None else None
    if kept is None:
        # No narration track. Source time is the only clock available, which is correct for
        # a music-only montage and a reasonable fallback otherwise.
        logger.debug("Plan has no narration track; timing cues against the transcript")

    wants_words = settings.karaoke if include_word_timings is None else include_word_timings
    cues = SubtitleBuilder(settings).build(
        transcript,
        kept_ranges=kept,
        include_word_timings=wants_words,
    )

    if plan.subtitles:
        logger.info("Replacing %d existing cue(s)", len(plan.subtitles))

    logger.info("Attached %d subtitle cue(s) to %s", len(cues), plan.project_id)
    return plan.model_copy(update={"subtitles": cues}), len(cues)


def cues_past_end(plan: EditPlan) -> tuple[int, ...]:
    """Indices of cues that start after the picture ends.

    The symptom of cues timed against narration time instead of timeline time: they drift
    later with every removed pause, so the last ones fall off the end. Reported here as
    well as by the Rule Engine so a caller attaching cues can notice immediately.
    """
    duration = plan.timeline_duration
    return tuple(
        index for index, cue in enumerate(plan.subtitles) if cue.range.start > duration + 0.001
    )


__all__ = ["attach_subtitles", "cues_past_end"]
