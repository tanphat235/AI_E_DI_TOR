"""Normalisation: turning an authored plan into a renderable one.

Normalisation is what lets the Edit Plan format be *forgiving to write*. The director
lists clips in order and omits the arithmetic; this module fills it in. Without it, every
plan would require an LLM to sum forty floating-point durations correctly, and the
failure mode - a one-frame gap at clip 31 - is invisible until render.

Two rules govern everything here.

**Every change is reported.** A silent fix-up is indistinguishable from a bug: the user
sees a video that does not match the plan they wrote, with nothing to explain why. So each
normaliser returns its edits as ``INFO`` issues, and the CLI prints them.

**Normalisation never invents editorial intent.** It places clips the director ordered,
clamps a range to a file's real length, separates colliding cues. It does not choose
footage, reorder shots, or delete a clip it dislikes - those are decisions, and decisions
belong to the director.
"""

from __future__ import annotations

import itertools
import math

from app.models.common import Issue, Severity
from app.models.edit_plan import EditPlan, SubtitleCue, TimelineClip
from app.rule_engine.context import RuleContext
from app.utils.logging import get_logger

logger = get_logger(__name__)


class ClipPlacementNormalizer:
    """Assigns ``timeline_start`` to every clip that omitted one.

    The most valuable thing the Rule Engine does. Clips are packed end to end in list
    order, and each incoming transition pulls its clip *back* over the previous one by
    exactly its duration - because that overlap is what a dissolve physically is.

    An explicit ``timeline_start`` is always respected: the director may need an insert to
    land on a beat, and overriding that would defeat the purpose of the field existing.
    """

    @property
    def name(self) -> str:
        return "clip_placement"

    def normalize(self, plan: EditPlan, context: RuleContext) -> tuple[EditPlan, tuple[Issue, ...]]:
        if plan.is_placed:
            return plan, ()

        issues: list[Issue] = []
        placed: list[TimelineClip] = []
        cursor = 0.0

        for position, clip in enumerate(plan.clips):
            if clip.timeline_start is not None:
                start = clip.timeline_start
                placed.append(clip)
            else:
                # The first clip's transition_in is a fade from black, not an overlap with
                # a previous clip, so it must not pull the start negative.
                overlap = (
                    clip.transition_in.duration
                    if position > 0 and clip.transition_in is not None
                    else 0.0
                )
                start = max(0.0, cursor - overlap)
                placed.append(clip.model_copy(update={"timeline_start": start}))
                issues.append(
                    Issue(
                        code="normalize.placed_clip",
                        severity=Severity.INFO,
                        message=f"{clip.id}: placed at {start:.3f}s",
                        location=f"clips[{position}]",
                    )
                )
            cursor = start + clip.timeline_duration

        logger.debug("Placed %d clip(s)", len(issues))
        return plan.model_copy(update={"clips": tuple(placed)}), tuple(issues)


class SourceRangeClampNormalizer:
    """Shortens source ranges that run past the end of the real file.

    A plausible authoring mistake with an unhelpful failure: FFmpeg either emits black or
    errors, depending on the codec. Clamping is safe because it preserves the director's
    intent - the shot they wanted, as much of it as exists.

    A range starting *beyond* the file is not clamped. There is nothing to salvage, and
    inventing a range would substitute a shot nobody chose; that stays an error for the
    director to fix.
    """

    @property
    def name(self) -> str:
        return "source_clamp"

    def normalize(self, plan: EditPlan, context: RuleContext) -> tuple[EditPlan, tuple[Issue, ...]]:
        issues: list[Issue] = []
        clips: list[TimelineClip] = []

        for position, clip in enumerate(plan.clips):
            duration = context.duration_of(clip.source)
            if (
                duration is None
                or clip.source_range.end <= duration
                or clip.source_range.start >= duration
            ):
                clips.append(clip)
                continue

            clamped = clip.source_range.model_copy(update={"end": duration})
            clips.append(clip.model_copy(update={"source_range": clamped}))
            issues.append(
                Issue(
                    code="normalize.clamped_source",
                    severity=Severity.INFO,
                    message=(
                        f"{clip.id}: source range shortened from "
                        f"{clip.source_range.end:.2f}s to {duration:.2f}s, "
                        f"the real length of {clip.source}"
                    ),
                    location=f"clips[{position}]",
                )
            )

        if not issues:
            return plan, ()
        return plan.model_copy(update={"clips": tuple(clips)}), tuple(issues)


class TransitionClampNormalizer:
    """Shortens transitions that would consume too much of the shots they join.

    Clamped rather than rejected because the *intent* - a dissolve here - is clearly right
    even when the duration is not, and the shortest admissible dissolve honours it. A
    transition that clamps below one frame is converted to a hard cut, since that is what
    it would render as anyway and saying so is more honest than emitting a filter that
    does nothing.
    """

    @property
    def name(self) -> str:
        return "transition_clamp"

    def normalize(self, plan: EditPlan, context: RuleContext) -> tuple[EditPlan, tuple[Issue, ...]]:
        from app.models.edit_plan import Transition

        rules = context.rules
        frame = plan.output.frame_duration
        issues: list[Issue] = []
        clips: list[TimelineClip] = []

        for position, clip in enumerate(plan.clips):
            transition = clip.transition_in
            if transition is None or transition.kind.is_instant:
                clips.append(clip)
                continue

            previous = plan.clips[position - 1] if position > 0 else None
            shorter = (
                clip.timeline_duration
                if previous is None
                else min(previous.timeline_duration, clip.timeline_duration)
            )
            allowed = shorter * rules.max_transition_ratio
            if transition.duration <= allowed:
                clips.append(clip)
                continue

            if allowed < frame:
                clips.append(clip.model_copy(update={"transition_in": Transition.cut()}))
                issues.append(
                    Issue(
                        code="normalize.transition_to_cut",
                        severity=Severity.INFO,
                        message=(
                            f"{clip.id}: the {transition.kind} became a hard cut - the "
                            f"clips are too short for any transition longer than a frame"
                        ),
                        location=f"clips[{position}]",
                    )
                )
            else:
                # Floored to milliseconds, never rounded. Rounding can land *above*
                # ``allowed``, so the next pass finds the transition still too long and
                # "clamps" it to the same value again - the chain then never reports zero
                # changes and normalisation runs until its pass limit, reporting a spurious
                # failure to converge.
                clamped = math.floor(allowed * 1000.0) / 1000.0
                if clamped >= transition.duration:
                    # Already within the limit once floored. Reporting a change here would
                    # be the same non-termination by another route.
                    clips.append(clip)
                    continue
                clips.append(
                    clip.model_copy(
                        update={"transition_in": Transition(kind=transition.kind, duration=clamped)}
                    )
                )
                issues.append(
                    Issue(
                        code="normalize.clamped_transition",
                        severity=Severity.INFO,
                        message=(
                            f"{clip.id}: {transition.kind} shortened from "
                            f"{transition.duration:.3f}s to {clamped:.3f}s"
                        ),
                        location=f"clips[{position}]",
                    )
                )

        if not issues:
            return plan, ()
        return plan.model_copy(update={"clips": tuple(clips)}), tuple(issues)


class SubtitleTimingNormalizer:
    """Separates colliding cues and enforces the on-screen duration bounds.

    Order matters and is the same as in the subtitle builder: extend short cues first,
    then resolve overlaps. Doing it the other way lets an extension re-create the overlap
    that was just removed.
    """

    @property
    def name(self) -> str:
        return "subtitle_timing"

    def normalize(self, plan: EditPlan, context: RuleContext) -> tuple[EditPlan, tuple[Issue, ...]]:
        if not plan.subtitles:
            return plan, ()

        settings = context.subtitle
        issues: list[Issue] = []
        cues = sorted(plan.subtitles, key=lambda cue: cue.range.start)
        adjusted: list[SubtitleCue] = []

        for position, cue in enumerate(cues):
            start, end = cue.range.start, cue.range.end
            if end - start < settings.min_cue_duration:
                end = start + settings.min_cue_duration
                issues.append(
                    Issue(
                        code="normalize.extended_cue",
                        severity=Severity.INFO,
                        message=f"cue {position}: held to {settings.min_cue_duration:.2f}s",
                        location=f"subtitles[{position}]",
                    )
                )
            elif end - start > settings.max_cue_duration:
                end = start + settings.max_cue_duration
                issues.append(
                    Issue(
                        code="normalize.shortened_cue",
                        severity=Severity.INFO,
                        message=f"cue {position}: capped at {settings.max_cue_duration:.2f}s",
                        location=f"subtitles[{position}]",
                    )
                )
            adjusted.append(
                cue.model_copy(update={"range": cue.range.model_copy(update={"end": end})})
                if end != cue.range.end
                else cue
            )

        resolved: list[SubtitleCue] = []
        for position, cue in enumerate(adjusted):
            if resolved:
                previous = resolved[-1]
                latest_end = cue.range.start - settings.cue_gap
                if previous.range.end > latest_end:
                    if latest_end <= previous.range.start:
                        # No room to shorten without inverting the range. Two cues this
                        # close are one utterance split in two, so dropping this one beats
                        # a one-frame flash.
                        issues.append(
                            Issue(
                                code="normalize.dropped_cue",
                                severity=Severity.INFO,
                                message=(
                                    f"cue {position} dropped: it begins before the previous "
                                    "cue could end"
                                ),
                                location=f"subtitles[{position}]",
                            )
                        )
                        continue
                    resolved[-1] = previous.model_copy(
                        update={"range": previous.range.model_copy(update={"end": latest_end})}
                    )
                    issues.append(
                        Issue(
                            code="normalize.separated_cues",
                            severity=Severity.INFO,
                            message=f"cue {position - 1}: shortened to clear cue {position}",
                            location=f"subtitles[{position - 1}]",
                        )
                    )
            resolved.append(cue)

        if not issues and tuple(resolved) == plan.subtitles:
            return plan, ()
        return plan.model_copy(update={"subtitles": tuple(resolved)}), tuple(issues)


class MusicFitNormalizer:
    """Shortens music cues that run past the end of the video or of their track."""

    @property
    def name(self) -> str:
        return "music_fit"

    def normalize(self, plan: EditPlan, context: RuleContext) -> tuple[EditPlan, tuple[Issue, ...]]:
        if not plan.music:
            return plan, ()

        video_duration = plan.timeline_duration
        issues: list[Issue] = []
        cues = []

        for position, cue in enumerate(plan.music):
            end = cue.timeline_range.end
            reason = ""

            if end > video_duration:
                end = video_duration
                reason = "the end of the video"

            track_duration = context.duration_of(cue.track)
            if track_duration is not None:
                available = track_duration - cue.source_offset
                if available > 0 and cue.timeline_range.start + available < end:
                    end = cue.timeline_range.start + available
                    reason = f"the end of {cue.track}"

            if end <= cue.timeline_range.start or end == cue.timeline_range.end:
                cues.append(cue)
                continue

            shortened = cue.timeline_range.model_copy(update={"end": end})
            # Fades must still fit, and the model enforces it - so scale them down rather
            # than letting a validation error escape from a *normaliser*.
            total_fade = cue.fade_in + cue.fade_out
            updates: dict[str, object] = {"timeline_range": shortened}
            if total_fade > shortened.duration:
                scale = shortened.duration / total_fade if total_fade > 0 else 0.0
                updates["fade_in"] = round(cue.fade_in * scale, 3)
                updates["fade_out"] = round(cue.fade_out * scale, 3)

            cues.append(cue.model_copy(update=updates))
            issues.append(
                Issue(
                    code="normalize.shortened_music",
                    severity=Severity.INFO,
                    message=(f"music cue {position}: shortened to {end:.2f}s to reach {reason}"),
                    location=f"music[{position}]",
                )
            )

        if not issues:
            return plan, ()
        return plan.model_copy(update={"music": tuple(cues)}), tuple(issues)


DEFAULT_NORMALIZERS: tuple[type, ...] = (
    SourceRangeClampNormalizer,
    TransitionClampNormalizer,
    ClipPlacementNormalizer,
    SubtitleTimingNormalizer,
    MusicFitNormalizer,
)
"""Normalisers in execution order, and the order is load-bearing.

Source ranges are clamped **first** because clamping changes a clip's timeline duration.
Transitions are clamped **second** because the cap is a fraction of that duration. Clips
are placed **third**, once both are final - placing before clamping would compute every
position from durations that are about to change. Subtitles and music come last, because
both are measured against the finished timeline length.
"""


def build_default_normalizers() -> tuple[object, ...]:
    """Instantiate the standard normaliser chain."""
    return tuple(normalizer() for normalizer in DEFAULT_NORMALIZERS)


def is_monotonic(plan: EditPlan) -> bool:
    """Whether placed clips are in ascending timeline order.

    Not enforced - a director may deliberately place an insert out of list order - but
    useful for reporting.
    """
    starts = [clip.timeline_start for clip in plan.clips if clip.timeline_start is not None]
    return all(earlier <= later for earlier, later in itertools.pairwise(starts))


__all__ = [
    "DEFAULT_NORMALIZERS",
    "ClipPlacementNormalizer",
    "MusicFitNormalizer",
    "SourceRangeClampNormalizer",
    "SubtitleTimingNormalizer",
    "TransitionClampNormalizer",
    "build_default_normalizers",
    "is_monotonic",
]
