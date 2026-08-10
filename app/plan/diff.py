"""Comparing two Edit Plans.

Built for two moments. A director revising its own work needs to see what actually changed
between drafts, and a user reviewing an automated edit needs to see what the tool did to the
plan they wrote — after ``rules normalize``, for instance.

The one design decision that matters: clips are matched by **id**, not by position.
Positional diffing reports every clip after an insertion as modified, which buries the one
real change in a wall of false ones. Ordering is then reported separately, as a single fact,
because a reordered edit is one decision rather than N.
"""

from __future__ import annotations

from app.models.edit_plan import EditPlan, TimelineClip
from app.models.plan_review import ClipChange, PlanDiff
from app.utils.logging import get_logger

logger = get_logger(__name__)

DIFF_VERSION = "diff/1"

_COMPARED_FIELDS: tuple[str, ...] = (
    "source",
    "source_range",
    "timeline_start",
    "speed",
    "transition_in",
    "transition_out",
    "framing",
    "mute_source_audio",
    "reason",
    "scene_key",
    "beat_index",
)
"""Clip fields worth reporting a change in.

``confidence`` is deliberately absent. It is the author's own estimate rather than a property
of the edit, and a plan re-drafted with slightly different weights would otherwise report
every clip as changed while the video is byte-identical.
"""

_FLOAT_TOLERANCE = 0.001
"""Below one millisecond, two timings are the same timing.

Placement is computed by summing floats, so re-normalising a plan can shift a value in its
last bit. Reporting that as a change would make every diff noisy.
"""


def diff_plans(before: EditPlan, after: EditPlan) -> PlanDiff:
    """Compare two plans."""
    old = {clip.id: clip for clip in before.clips}
    new = {clip.id: clip for clip in after.clips}

    added = tuple(clip.id for clip in after.clips if clip.id not in old)
    removed = tuple(clip.id for clip in before.clips if clip.id not in new)

    changed = tuple(
        change
        for clip_id in (clip.id for clip in after.clips if clip.id in old)
        if (change := _compare_clip(old[clip_id], new[clip_id])) is not None
    )

    diff = PlanDiff(
        added=added,
        removed=removed,
        changed=changed,
        reordered=_is_reordered(before, after),
        duration_before=before.timeline_duration,
        duration_after=after.timeline_duration,
        summary_fields=_summary_fields(before, after),
    )
    logger.info(
        "Diff: +%d -%d ~%d%s",
        len(diff.added),
        len(diff.removed),
        len(diff.changed),
        " reordered" if diff.reordered else "",
    )
    return diff


def _compare_clip(before: TimelineClip, after: TimelineClip) -> ClipChange | None:
    """What differs between two versions of one clip, or ``None`` if nothing does."""
    differing: list[str] = []
    for field in _COMPARED_FIELDS:
        if not _equal(getattr(before, field), getattr(after, field)):
            differing.append(field)

    if not differing:
        return None
    return ClipChange(
        clip_id=after.id,
        fields=tuple(differing),
        before=_render(before, differing),
        after=_render(after, differing),
    )


def _equal(first: object, second: object) -> bool:
    """Equality with a tolerance for floats, applied recursively through models."""
    if isinstance(first, float) and isinstance(second, float):
        return abs(first - second) <= _FLOAT_TOLERANCE
    if first is None or second is None:
        return first is second

    # Models compare field by field so a sub-millisecond timing difference inside a nested
    # TimeRange or Transition is tolerated too. Read off the *class*, not the instance:
    # pydantic deprecated instance access in 2.11 and removes it in 3.0.
    if type(first) is type(second):
        fields = getattr(type(first), "model_fields", None)
        if fields is not None:
            return all(_equal(getattr(first, name), getattr(second, name)) for name in fields)
    return bool(first == second)


def _render(clip: TimelineClip, fields: list[str]) -> str:
    """A short, readable rendering of just the fields that differ."""
    parts: list[str] = []
    for field in fields:
        value = getattr(clip, field)
        if field == "source_range":
            parts.append(f"src={value.start:.2f}-{value.end:.2f}")
        elif field == "timeline_start":
            parts.append(f"tl={value:.3f}" if value is not None else "tl=unplaced")
        elif field in {"transition_in", "transition_out"}:
            label = f"{value.kind.value}@{value.duration:.2f}s" if value else "none"
            parts.append(f"{field.replace('transition_', 't_')}={label}")
        elif field == "reason":
            text = str(value)
            parts.append(f'reason="{text[:40]}…"' if len(text) > 40 else f'reason="{text}"')
        elif field == "source":
            parts.append(f"source={value}")
        else:
            parts.append(f"{field}={value}")
    return " ".join(parts)


def _is_reordered(before: EditPlan, after: EditPlan) -> bool:
    """Whether the clips common to both plans appear in a different sequence.

    Compared over the intersection only: adding a clip in the middle is an addition, not a
    reordering, and conflating the two would report both for one edit.
    """
    shared = {clip.id for clip in before.clips} & {clip.id for clip in after.clips}
    old_order = [clip.id for clip in before.clips if clip.id in shared]
    new_order = [clip.id for clip in after.clips if clip.id in shared]
    return old_order != new_order


def _summary_fields(before: EditPlan, after: EditPlan) -> tuple[str, ...]:
    """Plan-level fields that differ, ignoring bookkeeping.

    ``created_at`` is excluded: it changes on every write and would make every diff
    non-empty, which would train a reader to ignore the field list.
    """
    differing: list[str] = []
    if before.output != after.output:
        differing.append("output")
    if before.narration != after.narration:
        differing.append("narration")
    if before.subtitles != after.subtitles:
        differing.append("subtitles")
    if before.music != after.music:
        differing.append("music")
    if (before.notes or "") != (after.notes or ""):
        differing.append("notes")
    if before.created_by != after.created_by:
        differing.append("created_by")
    return tuple(differing)


__all__ = ["DIFF_VERSION", "diff_plans"]
