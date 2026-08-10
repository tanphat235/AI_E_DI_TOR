"""Plan admissibility rules: checking what the director actually produced.

This is the load-bearing half of the Rule Engine, because an Edit Plan is **generated
text**. The Pydantic models already guarantee well-formedness - unique ids, positive
durations, a cut with no length. What they cannot know is whether the plan is *admissible*
against the real world: whether a clip is long enough to read, whether the source range
exists in the actual file, whether two clips claim the same second of timeline.

Every rule here is a plausible mistake, not a hypothetical one. A language model asked to
cut from a 30-second clip will occasionally write ``52.0`` as an end time; the number
looks reasonable and the model has no way to check it. That is what a probe is for.

Severity is chosen deliberately, and the distinction is the whole point:

* **Error** - the render would be wrong or would fail. Blocks.
* **Warning** - an editor would object, but the result is watchable. Does not block.

Getting that backwards in either direction is costly. Errors that should be warnings make
the tool refuse to work; warnings that should be errors ship a broken video.
"""

from __future__ import annotations

import itertools

from app.models.common import Issue, MediaRef, Severity, TimeRange
from app.models.edit_plan import EditPlan, TimelineClip
from app.rule_engine.context import RuleContext

_OVERLAP_TOLERANCE = 0.001
"""Slack when comparing timeline positions, in seconds.

Placement is computed by summing floats, so two values that should be identical can differ
in the last bit. A millisecond is far below one frame at any frame rate, so nothing
visible hides inside it.
"""


class SourceExistsRule:
    """Every referenced file must exist on disk.

    First rule to run, because it is the cheapest and its failure explains every other
    one. Discovering a missing file forty minutes into a render is unacceptable.
    """

    @property
    def name(self) -> str:
        return "source_exists"

    def check(self, plan: EditPlan, context: RuleContext) -> tuple[Issue, ...]:
        if context.project_root is None:
            # Without a project root there is nothing to resolve against. Skipping is
            # correct: a plan can legitimately be validated for shape alone.
            return ()

        issues: list[Issue] = []
        for ref in plan.sources:
            resolved = context.resolve(ref)
            if resolved is None or not resolved.is_file():
                issues.append(
                    Issue(
                        code="source.missing",
                        severity=Severity.ERROR,
                        message=f"{ref} does not exist in the project",
                        hint="check the path is relative to the project root and spelled correctly",
                        location=str(ref),
                    )
                )
        return tuple(issues)


class SourceBoundsRule:
    """A clip's source range must exist inside the real file.

    The rule that most needs a probe. ``source_range: {start: 45, end: 52}`` on a
    30-second clip is well-formed, plausible, and would render as either a black frame or
    an FFmpeg error depending on the codec.
    """

    @property
    def name(self) -> str:
        return "source_bounds"

    def check(self, plan: EditPlan, context: RuleContext) -> tuple[Issue, ...]:
        issues: list[Issue] = []
        for position, clip in enumerate(plan.clips):
            duration = context.duration_of(clip.source)
            if duration is None:
                # Unmeasured. Reported at INFO so the user knows the check was skipped
                # rather than passed - silence would imply the range was verified.
                issues.append(
                    Issue(
                        code="source.unverified",
                        severity=Severity.INFO,
                        message=f"{clip.id}: {clip.source} has not been analysed, so its "
                        "source range could not be checked",
                        hint="run: aive analyze video",
                        location=f"clips[{position}]",
                    )
                )
                continue

            if clip.source_range.start >= duration:
                issues.append(
                    Issue(
                        code="source.out_of_bounds",
                        severity=Severity.ERROR,
                        message=(
                            f"{clip.id}: starts at {clip.source_range.start:.2f}s but "
                            f"{clip.source} is only {duration:.2f}s long"
                        ),
                        hint="pick a start inside the clip, or choose a different scene",
                        location=f"clips[{position}]",
                    )
                )
            elif clip.source_range.end > duration + _OVERLAP_TOLERANCE:
                issues.append(
                    Issue(
                        code="source.out_of_bounds",
                        severity=Severity.ERROR,
                        message=(
                            f"{clip.id}: ends at {clip.source_range.end:.2f}s but "
                            f"{clip.source} is only {duration:.2f}s long"
                        ),
                        hint="shorten the range; `aive rules normalize` will clamp it for you",
                        location=f"clips[{position}]",
                    )
                )
        return tuple(issues)


class ClipDurationRule:
    """Clips must be long enough to read and short enough not to drag.

    Asymmetric on purpose. Too short is an **error**: below roughly a second a cut reads
    as a glitch rather than a shot, and no viewer can take anything from it. Too long is a
    **warning**: holding a shot for twelve seconds is unusual but sometimes exactly right,
    and refusing to render it would be the tool overruling an editorial choice.
    """

    @property
    def name(self) -> str:
        return "clip_duration"

    def check(self, plan: EditPlan, context: RuleContext) -> tuple[Issue, ...]:
        rules = context.rules
        issues: list[Issue] = []
        for position, clip in enumerate(plan.clips):
            duration = clip.timeline_duration
            if duration < rules.min_clip_duration:
                issues.append(
                    Issue(
                        code="clip.too_short",
                        severity=Severity.ERROR,
                        message=(
                            f"{clip.id}: {duration:.2f}s on the timeline, below the "
                            f"{rules.min_clip_duration:.2f}s minimum"
                        ),
                        hint=(
                            "extend the source range, drop the clip, or lower "
                            "rules.min_clip_duration"
                        ),
                        location=f"clips[{position}]",
                    )
                )
            elif duration > rules.max_clip_duration:
                issues.append(
                    Issue(
                        code="clip.too_long",
                        severity=Severity.WARNING,
                        message=(
                            f"{clip.id}: {duration:.2f}s on the timeline, above the "
                            f"{rules.max_clip_duration:.2f}s guideline"
                        ),
                        hint="a long hold can be right; split it if the shot has no movement",
                        location=f"clips[{position}]",
                    )
                )
        return tuple(issues)


class TimelineContinuityRule:
    """Placed clips must tile the timeline without overlapping or leaving gaps.

    Only meaningful once the plan is placed, so an unplaced plan is reported at INFO
    rather than failed - normalising is the fix, and telling the user to normalise is more
    useful than telling them the plan is broken.

    Clips *may* overlap by exactly the length of the second one's incoming transition:
    that overlap is what a dissolve is. Anything more is two clips claiming the same
    second, and anything less leaves black on screen.
    """

    @property
    def name(self) -> str:
        return "timeline_continuity"

    def check(self, plan: EditPlan, context: RuleContext) -> tuple[Issue, ...]:
        if not plan.is_placed:
            return (
                Issue(
                    code="plan.unplaced",
                    severity=Severity.INFO,
                    message="clips have no timeline positions, so continuity was not checked",
                    hint="run: aive rules normalize",
                ),
            )

        issues: list[Issue] = []
        for position, (earlier, later) in enumerate(itertools.pairwise(plan.clips)):
            expected_overlap = later.transition_in.duration if later.transition_in else 0.0
            actual_overlap = earlier.timeline_range.end - later.timeline_range.start

            if actual_overlap > expected_overlap + _OVERLAP_TOLERANCE:
                issues.append(
                    Issue(
                        code="clip.overlap",
                        severity=Severity.ERROR,
                        message=(
                            f"{earlier.id} and {later.id} overlap by {actual_overlap:.3f}s, "
                            f"but the transition between them is {expected_overlap:.3f}s"
                        ),
                        hint="re-place the clips; `aive rules normalize` does this correctly",
                        location=f"clips[{position + 1}]",
                    )
                )
            elif actual_overlap < expected_overlap - _OVERLAP_TOLERANCE:
                gap = expected_overlap - actual_overlap
                issues.append(
                    Issue(
                        code="clip.gap",
                        severity=Severity.WARNING,
                        message=(
                            f"{earlier.id} ends {gap:.3f}s before {later.id} begins, "
                            "leaving black on screen"
                        ),
                        hint="close the gap unless the black frame is deliberate",
                        location=f"clips[{position + 1}]",
                    )
                )

        first = plan.clips[0]
        if first.timeline_start is not None and first.timeline_start > _OVERLAP_TOLERANCE:
            issues.append(
                Issue(
                    code="plan.late_start",
                    severity=Severity.WARNING,
                    message=f"the first clip starts at {first.timeline_start:.2f}s, so the "
                    "video opens on black",
                    hint="set the first clip's timeline_start to 0, or omit it",
                    location="clips[0]",
                )
            )
        return tuple(issues)


class TransitionFitRule:
    """A transition must be short enough for the shots it joins.

    A dissolve consuming most of a two-second clip means the shot never resolves: the
    viewer sees a blend, not a picture. The cap is a *fraction* of the shorter clip rather
    than an absolute, because what matters is how much of the shot survives.
    """

    @property
    def name(self) -> str:
        return "transition_fit"

    def check(self, plan: EditPlan, context: RuleContext) -> tuple[Issue, ...]:
        rules = context.rules
        issues: list[Issue] = []

        for position, clip in enumerate(plan.clips):
            transition = clip.transition_in
            if transition is None or transition.kind.is_instant:
                continue

            previous = plan.clips[position - 1] if position > 0 else None
            shorter = (
                clip.timeline_duration
                if previous is None
                else min(previous.timeline_duration, clip.timeline_duration)
            )
            allowed = shorter * rules.max_transition_ratio
            if transition.duration > allowed + _OVERLAP_TOLERANCE:
                issues.append(
                    Issue(
                        code="transition.too_long",
                        severity=Severity.ERROR,
                        message=(
                            f"{clip.id}: a {transition.duration:.2f}s {transition.kind} "
                            f"exceeds {rules.max_transition_ratio:.0%} of the shorter "
                            f"clip ({shorter:.2f}s), leaving no time for the shot to read"
                        ),
                        hint=f"shorten it to at most {allowed:.2f}s, or lengthen the clips",
                        location=f"clips[{position}]",
                    )
                )
        return tuple(issues)


class DuplicateUsageRule:
    """The plan must not use a scene suppressed as a duplicate.

    The director is told which scenes are duplicates in the analysis digest, so using one
    is a decision rather than an accident - which is why this is a warning. It is also the
    single most visible failure of an automated edit, so it is never silent.
    """

    @property
    def name(self) -> str:
        return "duplicate_usage"

    def check(self, plan: EditPlan, context: RuleContext) -> tuple[Issue, ...]:
        if not context.suppressed_scene_keys:
            return ()

        issues: list[Issue] = []
        used: dict[str, str] = {}
        for position, clip in enumerate(plan.clips):
            if clip.scene_key is None:
                continue
            if clip.scene_key in context.suppressed_scene_keys:
                issues.append(
                    Issue(
                        code="scene.duplicate_used",
                        severity=Severity.WARNING,
                        message=f"{clip.id} uses {clip.scene_key}, a suppressed duplicate",
                        hint="use the surviving take instead; see the analysis digest",
                        location=f"clips[{position}]",
                    )
                )
            # Separately: the same scene used twice in one plan is a repeat, whatever the
            # duplicate groups say.
            if clip.scene_key in used:
                issues.append(
                    Issue(
                        code="scene.reused",
                        severity=Severity.WARNING,
                        message=(
                            f"{clip.id} reuses scene {clip.scene_key}, already used by "
                            f"{used[clip.scene_key]}"
                        ),
                        hint="the audience will notice the same shot twice",
                        location=f"clips[{position}]",
                    )
                )
            else:
                used[clip.scene_key] = clip.id
        return tuple(issues)


class NarrationCoverageRule:
    """Picture must last as long as the narration.

    If the narration outlives the footage the video ends on black with someone still
    talking, which is the most jarring failure a viewer can be shown. The reverse - picture
    outlasting narration - is a mild warning, because a few seconds of tail is a normal
    editorial choice.
    """

    @property
    def name(self) -> str:
        return "narration_coverage"

    def check(self, plan: EditPlan, context: RuleContext) -> tuple[Issue, ...]:
        if plan.narration is None:
            return ()

        narration_duration = plan.narration.timeline_duration
        picture_duration = plan.timeline_duration
        shortfall = narration_duration - picture_duration

        if shortfall > 0.5:
            return (
                Issue(
                    code="narration.uncovered",
                    severity=Severity.ERROR,
                    message=(
                        f"narration runs {narration_duration:.2f}s but the picture is only "
                        f"{picture_duration:.2f}s, leaving {shortfall:.2f}s over black"
                    ),
                    hint="add clips, or lengthen existing ones",
                ),
            )
        if shortfall < -2.0:
            return (
                Issue(
                    code="narration.trailing_picture",
                    severity=Severity.WARNING,
                    message=(f"the picture runs {-shortfall:.2f}s past the end of the narration"),
                    hint="fine as a deliberate tail; trim it otherwise",
                ),
            )
        return ()


class SubtitleRule:
    """Subtitle cues must be readable and must not collide.

    Cue times are in *timeline* space, so this also catches the classic authoring mistake:
    cues timed against the raw narration drift further out of sync with every removed
    pause, which shows up here as cues running past the end of the video.
    """

    @property
    def name(self) -> str:
        return "subtitles"

    def check(self, plan: EditPlan, context: RuleContext) -> tuple[Issue, ...]:
        if not plan.subtitles:
            return ()

        settings = context.subtitle
        issues: list[Issue] = []
        video_duration = plan.timeline_duration

        for position, cue in enumerate(plan.subtitles):
            location = f"subtitles[{position}]"
            duration = cue.range.duration

            if duration < settings.min_cue_duration:
                issues.append(
                    Issue(
                        code="subtitle.too_brief",
                        severity=Severity.WARNING,
                        message=(
                            f"cue {position} is on screen for {duration:.2f}s, too brief to read"
                        ),
                        hint=f"minimum is {settings.min_cue_duration:.2f}s",
                        location=location,
                    )
                )
            elif duration > settings.max_cue_duration:
                issues.append(
                    Issue(
                        code="subtitle.too_long",
                        severity=Severity.WARNING,
                        message=f"cue {position} is on screen for {duration:.2f}s",
                        hint=f"maximum is {settings.max_cue_duration:.2f}s; split it",
                        location=location,
                    )
                )

            if cue.range.start > video_duration + _OVERLAP_TOLERANCE:
                issues.append(
                    Issue(
                        code="subtitle.past_end",
                        severity=Severity.ERROR,
                        message=(
                            f"cue {position} starts at {cue.range.start:.2f}s but the video "
                            f"is {video_duration:.2f}s long"
                        ),
                        hint=(
                            "cue times are TIMELINE time, not narration time - "
                            "use `aive subtitle build` rather than mapping by hand"
                        ),
                        location=location,
                    )
                )

            longest = max((len(line) for line in cue.text.splitlines()), default=0)
            if longest > settings.max_chars_per_line:
                issues.append(
                    Issue(
                        code="subtitle.line_too_long",
                        severity=Severity.WARNING,
                        message=f"cue {position} has a {longest}-character line",
                        hint=f"wrap at {settings.max_chars_per_line}",
                        location=location,
                    )
                )
            if cue.line_count > settings.max_lines:
                issues.append(
                    Issue(
                        code="subtitle.too_many_lines",
                        severity=Severity.WARNING,
                        message=f"cue {position} has {cue.line_count} lines",
                        hint=f"maximum is {settings.max_lines}",
                        location=location,
                    )
                )

        for position, (earlier, later) in enumerate(itertools.pairwise(plan.subtitles)):
            if earlier.range.end > later.range.start + _OVERLAP_TOLERANCE:
                issues.append(
                    Issue(
                        code="subtitle.overlap",
                        severity=Severity.ERROR,
                        message=f"cues {position} and {position + 1} are on screen at once",
                        hint="two cues cannot share the screen; `rules normalize` separates them",
                        location=f"subtitles[{position + 1}]",
                    )
                )
        return tuple(issues)


class MusicRule:
    """Music cues must fit both the timeline and the track.

    ``source_offset`` is the field that goes wrong: it is a position inside the *music
    file*, and setting it past the end of a short track yields silence rather than an
    error from FFmpeg.
    """

    @property
    def name(self) -> str:
        return "music"

    def check(self, plan: EditPlan, context: RuleContext) -> tuple[Issue, ...]:
        if not plan.music:
            return ()

        issues: list[Issue] = []
        video_duration = plan.timeline_duration

        for position, cue in enumerate(plan.music):
            location = f"music[{position}]"

            if cue.timeline_range.start > video_duration + _OVERLAP_TOLERANCE:
                issues.append(
                    Issue(
                        code="music.past_end",
                        severity=Severity.WARNING,
                        message=(
                            f"music cue {position} starts at {cue.timeline_range.start:.2f}s, "
                            f"after the video ends at {video_duration:.2f}s"
                        ),
                        hint="it would never be heard",
                        location=location,
                    )
                )

            track_duration = context.duration_of(cue.track)
            if track_duration is None:
                continue

            if cue.source_offset >= track_duration:
                issues.append(
                    Issue(
                        code="music.offset_past_end",
                        severity=Severity.ERROR,
                        message=(
                            f"music cue {position} starts {cue.source_offset:.2f}s into "
                            f"{cue.track}, which is only {track_duration:.2f}s long"
                        ),
                        hint="the bed would be silent; lower source_offset",
                        location=location,
                    )
                )
            elif (
                cue.source_offset + cue.timeline_range.duration
                > track_duration + _OVERLAP_TOLERANCE
            ):
                available = track_duration - cue.source_offset
                issues.append(
                    Issue(
                        code="music.too_short",
                        severity=Severity.WARNING,
                        message=(
                            f"music cue {position} needs {cue.timeline_range.duration:.2f}s "
                            f"but only {available:.2f}s of {cue.track} remains after the offset"
                        ),
                        hint="the bed will run out; shorten the cue or pick a longer track",
                        location=location,
                    )
                )
        return tuple(issues)


class OutputSpecRule:
    """The output format must be consistent and renderable."""

    @property
    def name(self) -> str:
        return "output"

    def check(self, plan: EditPlan, context: RuleContext) -> tuple[Issue, ...]:
        output = plan.output
        issues: list[Issue] = []

        declared = output.aspect_ratio.ratio
        actual = output.actual_ratio
        if abs(declared - actual) > 0.02:
            issues.append(
                Issue(
                    code="output.aspect_mismatch",
                    severity=Severity.WARNING,
                    message=(
                        f"aspect_ratio says {output.aspect_ratio} ({declared:.3f}) but "
                        f"{output.width}x{output.height} is {actual:.3f}"
                    ),
                    hint="the renderer follows width and height; the label would mislead a reader",
                    location="output",
                )
            )

        # A transition shorter than a frame cannot be rendered as one.
        frame = output.frame_duration
        for position, clip in enumerate(plan.clips):
            for edge, transition in (("in", clip.transition_in), ("out", clip.transition_out)):
                if transition is None or transition.kind.is_instant:
                    continue
                if transition.duration < frame:
                    issues.append(
                        Issue(
                            code="transition.sub_frame",
                            severity=Severity.WARNING,
                            message=(
                                f"{clip.id}: the {edge} transition is {transition.duration:.3f}s, "
                                f"shorter than one frame at {output.fps:g}fps ({frame:.3f}s)"
                            ),
                            hint="it will render as a hard cut",
                            location=f"clips[{position}]",
                        )
                    )
        return tuple(issues)


def clip_source_range(clip: TimelineClip) -> TimeRange:
    """A clip's source range. Trivial, but keeps rule code reading declaratively."""
    return clip.source_range


def referenced_sources(plan: EditPlan) -> tuple[MediaRef, ...]:
    """Every file a plan needs."""
    return plan.sources


DEFAULT_RULES: tuple[type, ...] = (
    SourceExistsRule,
    SourceBoundsRule,
    ClipDurationRule,
    TimelineContinuityRule,
    TransitionFitRule,
    DuplicateUsageRule,
    NarrationCoverageRule,
    SubtitleRule,
    MusicRule,
    OutputSpecRule,
)
"""Rule classes in execution order.

Ordered cheapest-and-most-explanatory first: a missing source file explains every other
failure, so reporting it before a hundred out-of-bounds errors saves the reader from
chasing symptoms.
"""


def build_default_rules() -> tuple[object, ...]:
    """Instantiate the standard rule set."""
    return tuple(rule() for rule in DEFAULT_RULES)


__all__ = [
    "DEFAULT_RULES",
    "ClipDurationRule",
    "DuplicateUsageRule",
    "MusicRule",
    "NarrationCoverageRule",
    "OutputSpecRule",
    "SourceBoundsRule",
    "SourceExistsRule",
    "SubtitleRule",
    "TimelineContinuityRule",
    "TransitionFitRule",
    "build_default_rules",
]
