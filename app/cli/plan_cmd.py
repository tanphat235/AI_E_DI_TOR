"""``aive plan`` - the director's planning surface, before and after authoring.

Two commands lead into writing a plan:

``brief`` assembles everything needed to author one into a single document: what is said,
which footage may be used, how suitable each option is, and the constraints the result must
satisfy. Before it existed a director had to run three commands and cross-reference three
digests, holding the join in its head.

``draft`` produces a heuristic baseline to revise. It is not an edit and does not pretend to
be: ``created_by`` is ``"heuristic"`` and every ``reason`` says so.

Three follow after:

``show`` renders a plan the way an editor would ask about it - pacing, framing, coverage, and
every clip's ``reason`` - and flags the patterns that are invisible in JSON and obvious in a
finished video. ``diff`` says what a second pass or a normalisation actually changed.
``subtitles`` writes cues into ``plan.subtitles``, which is what the renderer and the CapCut
exporter read.

None of them decides anything. Deciding is the director's job, and it is the only part of
this pipeline that genuinely needs judgement.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from app.cli.output import ExitCode, emit_digest, emit_error, emit_json, note, write_json_file
from app.config.settings import AiveSettings, load_settings
from app.models.common import Severity
from app.models.edit_plan import EditPlan
from app.models.plan_review import PlanDiff, PlanReview
from app.models.planning import BeatCandidates, PlanningBrief
from app.models.speech import NarrationAnalysis
from app.models.video import FootageAnalysis
from app.plan.diff import diff_plans
from app.plan.loader import LoadedPlan, PlanLoadError, load_plan
from app.plan.review import build_review
from app.services.container import build_container
from app.services.paths import ProjectPaths
from app.subtitles.attach import attach_subtitles, cues_past_end
from app.utils.logging import get_logger

logger = get_logger(__name__)

_MARK = {Severity.ERROR: "E", Severity.WARNING: "W", Severity.INFO: "i"}

app = typer.Typer(no_args_is_help=True, help="Assemble what the AI director needs to plan.")


@app.command("brief")
def brief(
    project: Annotated[Path, typer.Argument(help="Project directory.")],
    full: Annotated[
        bool,
        typer.Option("--full", help="Emit the whole brief as JSON instead of a digest."),
    ] = False,
    candidates: Annotated[
        int | None,
        typer.Option("--candidates", help="Options to offer per beat. Defaults to config."),
    ] = None,
) -> None:
    """Assemble the planning brief: beats, candidate scenes, coverage and constraints.

    Read this once instead of cross-referencing three analysis digests. Exits `4` when the
    project cannot be covered - better to know before authoring forty clips than after.
    """
    paths = ProjectPaths.for_root(project)
    narration, footage = _load_analyses(paths)

    container = build_container(
        paths.root,
        settings=_with_candidate_override(paths.root, candidates) if candidates else None,
    )
    builder = container.brief
    if builder is None:  # pragma: no cover - always wired
        emit_error(
            "internal.not_wired",
            "the planner is not available in this build",
            exit_code=ExitCode.INTERNAL,
        )

    assembled = builder.build(
        project_id=_project_id(paths),
        narration=narration,
        footage=footage,
    )
    write_json_file(assembled, paths.planning_brief_file)
    _report_brief(assembled, paths, full=full)

    if not assembled.coverage.feasible:
        raise SystemExit(ExitCode.INVALID_INPUT.status)


@app.command("draft")
def draft(
    project: Annotated[Path, typer.Argument(help="Project directory.")],
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Where to write the plan. Defaults to edit_plan.json."),
    ] = None,
    full: Annotated[bool, typer.Option("--full", help="Emit the plan as JSON.")] = False,
) -> None:
    """Write a heuristic baseline plan, for the director to revise.

    **This is not an edit.** Clips are chosen by duration, quality and shot variety - nothing
    in it reflects what the footage actually shows. It exists so the director never has to
    fight the schema, and as a zero-AI fallback.

    The result is unplaced on purpose: run `aive rules normalize` to place the clips.
    """
    paths = ProjectPaths.for_root(project)
    narration, footage = _load_analyses(paths)
    container = build_container(paths.root)
    builder, drafter = container.brief, container.drafter
    if builder is None or drafter is None:  # pragma: no cover - always wired
        emit_error(
            "internal.not_wired",
            "the planner is not available in this build",
            exit_code=ExitCode.INTERNAL,
        )

    assembled = builder.build(project_id=_project_id(paths), narration=narration, footage=footage)
    try:
        plan = drafter.draft(assembled, narration=narration)
    except ValueError as exc:
        emit_error(
            "plan.not_draftable",
            str(exc),
            hint=f"run: aive plan brief {paths.root.as_posix()}",
            exit_code=ExitCode.INVALID_INPUT,
        )

    destination = output or paths.edit_plan_file
    write_json_file(plan, destination)

    note(f"Wrote a heuristic baseline to {destination}")
    note(f"  clips:  {len(plan.clips)}")
    note(f"  sources:{len(plan.sources)}")
    note("")
    note("  This is NOT an edit. Every clip was chosen by duration, quality and variety;")
    note("  none of it reflects what the footage shows. Rewrite the selections and the")
    note("  reason fields, then:")
    note(f"    aive rules normalize {destination.as_posix()} --in-place")

    if full:
        emit_json(plan)
        return
    emit_digest(_draft_digest(plan, destination))


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #


def _load_analyses(paths: ProjectPaths) -> tuple[NarrationAnalysis, FootageAnalysis]:
    """Both analysis documents, or a clear error naming the command that produces them.

    Both are required. A brief without narration has no beats to cover, and one without
    footage has nothing to cover them with; either way there is nothing to plan.
    """
    if not paths.exists():
        emit_error(
            "project.not_initialised",
            f"{paths.root} is not an AIVE project",
            hint=f"initialise it with: aive project init {paths.root.as_posix()}",
            exit_code=ExitCode.NOT_FOUND,
        )

    if not paths.narration_file.is_file():
        emit_error(
            "narration.not_analysed",
            f"no narration analysis at {paths.narration_file}",
            hint=f"run: aive analyze audio {paths.root.as_posix()}",
            exit_code=ExitCode.NOT_FOUND,
        )
    if not paths.footage_analysis_file.is_file():
        emit_error(
            "footage.not_analysed",
            f"no footage analysis at {paths.footage_analysis_file}",
            hint=f"run: aive analyze video {paths.root.as_posix()}",
            exit_code=ExitCode.NOT_FOUND,
        )

    try:
        narration = NarrationAnalysis.model_validate_json(
            paths.narration_file.read_text(encoding="utf-8")
        )
        footage = FootageAnalysis.model_validate_json(
            paths.footage_analysis_file.read_text(encoding="utf-8")
        )
    except (OSError, ValueError) as exc:
        emit_error(
            "analysis.unreadable",
            f"could not read the analysis documents: {exc}",
            hint="delete .aive/ and re-run the analyse commands",
            exit_code=ExitCode.INVALID_INPUT,
        )
    return narration, footage


def _with_candidate_override(project_root: Path, candidates: int) -> AiveSettings:
    """Settings with the shortlist size overridden by a CLI flag."""
    return load_settings(project_root, planner={"candidates_per_beat": candidates})


def _project_id(paths: ProjectPaths) -> str:
    import re

    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", paths.root.name).strip("-._")
    return cleaned or "project"


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #


def _report_brief(assembled: PlanningBrief, paths: ProjectPaths, *, full: bool) -> None:
    coverage = assembled.coverage
    note("")
    note(f"  narration:  {coverage.narration_duration:.1f}s in {coverage.beats_total} beat(s)")
    note(f"  needing footage: {coverage.beats_needing_footage}")
    note(
        f"  usable footage:  {coverage.eligible_footage_duration:.1f}s "
        f"({coverage.footage_ratio:.1f}x narration) "
        f"from {coverage.eligible_scenes} scene(s)"
    )
    note(f"  rejected scenes: {coverage.rejected_scenes}")
    note(f"  verdict:    {'PLANNABLE' if coverage.feasible else 'NOT PLANNABLE'}")
    note(f"  brief ->    {paths.planning_brief_file}")

    for issue in coverage.issues:
        note(f"\n  [{issue.severity.value}] {issue.message}")
        if issue.hint:
            note(f"     {issue.hint}")

    if assembled.unused_scenes:
        note(
            f"\n  {len(assembled.unused_scenes)} eligible scene(s) matched no beat - "
            "often the most interesting B-roll."
        )

    if full:
        emit_json(assembled)
        return
    emit_digest(_brief_digest(assembled, paths))


def _brief_digest(assembled: PlanningBrief, paths: ProjectPaths) -> list[str]:
    """The brief as a compact, line-oriented document.

    One ``B`` line per beat carrying its timeline position and text, then one indented
    candidate line per option. Reasons are included: a ranking a reader cannot argue with
    is one they will either over-trust or ignore.
    """
    coverage = assembled.coverage
    constraints = assembled.constraints
    lines = [
        f"# brief={paths.planning_brief_file.as_posix()} project={assembled.project_id} "
        f"feasible={str(coverage.feasible).lower()} "
        f"narration={coverage.narration_duration:.1f}s "
        f"beats={coverage.beats_total} need_footage={coverage.beats_needing_footage} "
        f"footage={coverage.eligible_footage_duration:.1f}s "
        f"ratio={coverage.footage_ratio:.1f}x scenes={coverage.eligible_scenes}",
        # The constraints belong beside the choices: a plan that breaks a threshold it was
        # never shown is the tool's failure, not the director's.
        f"# constraints "
        f"clip={constraints.min_clip_duration:.1f}-{constraints.max_clip_duration:.1f}s "
        f"transition={constraints.default_transition.value}@{constraints.transition_duration:.2f}s "
        f"max_transition_ratio={constraints.max_transition_ratio:.2f} "
        f"output={constraints.width}x{constraints.height}@{constraints.fps:g}fps "
        f"aspect={constraints.aspect_ratio.value} "
        f"quality_floor={constraints.min_overall_quality:.2f}",
    ]

    for issue in coverage.issues:
        lines.append(f"! {issue.code} {issue.message}")

    for beat in assembled.beats:
        lines.append(_beat_line(beat))
        for candidate in beat.candidates:
            reasons = "; ".join(candidate.reasons)
            lines.append(
                f"    {candidate.scene_key} {candidate.clip} "
                f"{candidate.range.start:.2f}-{candidate.range.end:.2f} "
                f"d={candidate.duration:.1f} score={candidate.score:.2f} "
                f"q={candidate.quality:.2f} shot={candidate.shot_type.value} "
                f"mot={candidate.motion.value}" + (f" | {reasons}" if reasons else "")
            )

    if assembled.unused_scenes:
        lines.append("# unused_eligible: " + " ".join(assembled.unused_scenes))
    return lines


def _beat_line(beat: BeatCandidates) -> str:
    timeline = (
        f"{beat.timeline_range.start:.2f}-{beat.timeline_range.end:.2f}"
        if beat.timeline_range is not None
        else "CUT"
    )
    keywords = ",".join(beat.keywords) or "-"
    return (
        f"B{beat.beat_index:03d} tl={timeline} want={beat.wanted_duration:.1f} "
        f"kw={keywords} cands={len(beat.candidates)} | {beat.text}"
    )


def _draft_digest(plan: EditPlan, destination: Path) -> list[str]:
    clips = plan.clips
    lines = [
        f"# draft={destination.as_posix()} created_by={plan.created_by} "
        f"clips={len(clips)} placed=false"
    ]
    lines.extend(
        f"{clip.id} {clip.source} {clip.source_range.start:.2f}-{clip.source_range.end:.2f} "
        f"d={clip.timeline_duration:.1f} scene={clip.scene_key or '-'} "
        f"beat={clip.beat_index if clip.beat_index is not None else '-'} "
        f"conf={clip.confidence:.2f}"
        for clip in clips
    )
    lines.append("# NOT an edit: revise the selections and the reason fields before rendering")
    return lines


# --------------------------------------------------------------------------- #
# Phase 6: working with a finished plan
# --------------------------------------------------------------------------- #


@app.command("show")
def show(
    plan_path: Annotated[Path, typer.Argument(help="Edit Plan to review.")],
    project: Annotated[
        Path | None,
        typer.Option("--project", help="Project directory. Defaults to the plan's folder."),
    ] = None,
    full: Annotated[bool, typer.Option("--full", help="Emit the review as JSON.")] = False,
) -> None:
    """Render a plan for review: its timeline, its pacing, and what it reveals.

    A plan is JSON, and JSON is not how anyone judges an edit. This is the surface for a
    second pass - and for a user asking what the director actually decided, since every
    clip's `reason` is printed beside it.

    The notes are **editorial observations, not errors** - `aive rules validate` owns
    admissibility. They flag the patterns that are invisible in a list of clips and obvious
    in a finished video: a mechanical rhythm, repeated framing, ignored footage.
    """
    paths, loaded = _load_plan(plan_path, project)
    footage = _load_footage_optional(paths)
    review = build_review(loaded.plan, footage=footage)

    if loaded.was_migrated:
        note(f"Migrated this plan from schema {loaded.migrated_from} on load.")
    if footage is None:
        note("No footage analysis found, so shot types and footage usage are unavailable.")

    statistics = review.statistics
    note("")
    note(f"  project:  {review.project_id}  (authored by {review.created_by})")
    note(f"  clips:    {statistics.clip_count} over {statistics.total_duration:.2f}s")
    note(f"  placed:   {'yes' if review.is_placed else 'no - run rules normalize'}")
    note(
        f"  pacing:   {statistics.cuts_per_minute:.1f} cuts/min, "
        f"clips {statistics.shortest_clip:.1f}-{statistics.longest_clip:.1f}s "
        f"(median {statistics.median_clip_duration:.1f}s)"
    )
    note(
        f"  sources:  {statistics.distinct_sources} file(s), {statistics.distinct_scenes} scene(s)"
    )
    if statistics.narration_duration is not None:
        delta = statistics.coverage_delta or 0.0
        note(
            f"  coverage: narration {statistics.narration_duration:.2f}s, "
            f"picture {statistics.total_duration:.2f}s ({delta:+.2f}s)"
        )
    note(f"  subtitles:{statistics.subtitle_count}")

    for issue in review.notes:
        note(f"\n  [{issue.severity.value}] {issue.message}")
        if issue.hint:
            note(f"     {issue.hint}")

    if full:
        emit_json(review)
        return
    emit_digest(_review_digest(review, plan_path))


@app.command("diff")
def diff(
    before_path: Annotated[Path, typer.Argument(help="The earlier plan.")],
    after_path: Annotated[Path, typer.Argument(help="The later plan.")],
    full: Annotated[bool, typer.Option("--full", help="Emit the diff as JSON.")] = False,
) -> None:
    """Compare two plans.

    Clips are matched by **id**, not position, so inserting one clip does not report every
    clip after it as modified. Reordering is reported once, as a single fact.

    Use it on a second pass, or to see what `rules normalize` did to a plan you wrote.
    """
    _, before = _load_plan(before_path, None)
    _, after = _load_plan(after_path, None)
    result = diff_plans(before.plan, after.plan)

    note("")
    if result.is_identical:
        note("  The two plans are identical.")
    else:
        note(f"  added:     {len(result.added)}")
        note(f"  removed:   {len(result.removed)}")
        note(f"  changed:   {len(result.changed)}")
        note(f"  reordered: {'yes' if result.reordered else 'no'}")
        note(
            f"  duration:  {result.duration_before:.2f}s -> "
            f"{result.duration_after:.2f}s ({result.duration_delta:+.2f}s)"
        )

    if full:
        emit_json(result)
        return
    emit_digest(_diff_digest(result, before_path, after_path))


@app.command("subtitles")
def subtitles(
    plan_path: Annotated[Path, typer.Argument(help="Edit Plan to attach cues to.")],
    project: Annotated[
        Path | None,
        typer.Option("--project", help="Project directory. Defaults to the plan's folder."),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Where to write. Defaults to stdout only."),
    ] = None,
    in_place: Annotated[bool, typer.Option("--in-place", help="Overwrite the input plan.")] = False,
    full: Annotated[bool, typer.Option("--full", help="Emit the plan as JSON.")] = False,
) -> None:
    """Put subtitle cues into the plan itself.

    `aive subtitle build` writes `.srt` and `.ass` files; this writes the cues into
    `plan.subtitles`, which is what the renderer and the CapCut exporter actually read. A
    plan without them renders without subtitles.

    Cues are timed against the **plan's** `narration.kept_ranges`, not the cleanup report -
    if the director trimmed the narration further, the plan is authoritative.
    """
    paths, loaded = _load_plan(plan_path, project)
    plan = loaded.plan

    if not paths.narration_file.is_file():
        emit_error(
            "narration.not_analysed",
            f"no narration analysis at {paths.narration_file}",
            hint=f"run: aive analyze audio {paths.root.as_posix()}",
            exit_code=ExitCode.NOT_FOUND,
        )
    try:
        narration = NarrationAnalysis.model_validate_json(
            paths.narration_file.read_text(encoding="utf-8")
        )
    except (OSError, ValueError) as exc:
        emit_error(
            "analysis.unreadable",
            f"could not read {paths.narration_file}: {exc}",
            hint="delete .aive/ and re-run: aive analyze audio",
            exit_code=ExitCode.INVALID_INPUT,
        )

    settings = load_settings(paths.root)
    updated, count = attach_subtitles(plan, narration.transcript, settings=settings.subtitle)

    if count == 0:
        emit_error(
            "subtitle.empty",
            "no cues survived: every recognised word falls inside a range the plan removed",
            hint="check the plan's narration.kept_ranges against the cleanup report",
            exit_code=ExitCode.INVALID_INPUT,
        )

    destination = plan_path if in_place else output
    if destination is not None:
        write_json_file(updated, destination)
        note(f"Wrote the plan with {count} cue(s) to {destination}")
    else:
        note(f"Attached {count} cue(s). Pass --in-place or -o to save.")

    stranded = cues_past_end(updated)
    if stranded:
        note(
            f"\n  WARNING: {len(stranded)} cue(s) start after the picture ends. The plan's "
            "narration is longer than its clips."
        )

    if full:
        emit_json(updated)
        return
    emit_digest(_subtitle_digest(updated, destination))


# --------------------------------------------------------------------------- #
# Phase 6 loading and reporting
# --------------------------------------------------------------------------- #


def _load_plan(plan_path: Path, project: Path | None) -> tuple[ProjectPaths, LoadedPlan]:
    """Read a plan through the migrating loader, and work out its project."""
    try:
        loaded = load_plan(plan_path)
    except PlanLoadError as exc:
        emit_error(
            "plan.invalid",
            str(exc),
            hint="run: aive schema show edit-plan",
            exit_code=(
                ExitCode.NOT_FOUND if "no Edit Plan at" in str(exc) else ExitCode.INVALID_INPUT
            ),
        )
    root = project if project is not None else plan_path.parent
    return ProjectPaths.for_root(root), loaded


def _load_footage_optional(paths: ProjectPaths) -> FootageAnalysis | None:
    """The footage analysis, or ``None``.

    Optional because a review is still useful without it - it simply cannot report shot
    types or how much of the footage went unused.
    """
    if not paths.footage_analysis_file.is_file():
        return None
    try:
        return FootageAnalysis.model_validate_json(
            paths.footage_analysis_file.read_text(encoding="utf-8")
        )
    except (OSError, ValueError):
        return None


def _review_digest(review: PlanReview, plan_path: Path) -> list[str]:
    """The review as a compact document: header, timeline, then the notes.

    Each clip's ``reason`` is on its line. It is the most token-expensive part and the whole
    point of the command: the reason is what a human reads to judge the edit.
    """
    statistics = review.statistics
    transitions = ",".join(
        f"{kind}:{count}" for kind, count in sorted(statistics.transition_counts.items())
    )
    shots = ",".join(
        f"{kind}:{count}" for kind, count in sorted(statistics.shot_type_counts.items())
    )
    lines = [
        f"# plan={plan_path.as_posix()} project={review.project_id} "
        f"by={review.created_by} placed={str(review.is_placed).lower()} "
        f"clips={statistics.clip_count} duration={statistics.total_duration:.2f} "
        f"cuts_per_min={statistics.cuts_per_minute:.1f} "
        f"sources={statistics.distinct_sources} scenes={statistics.distinct_scenes} "
        f"subtitles={statistics.subtitle_count}",
        f"# pacing median={statistics.median_clip_duration:.2f} "
        f"range={statistics.shortest_clip:.2f}-{statistics.longest_clip:.2f} "
        f"transitions={transitions or '-'} shots={shots or '-'} "
        f"longest_same_shot_run={statistics.longest_same_shot_run}",
    ]
    if statistics.narration_duration is not None:
        lines.append(
            f"# coverage narration={statistics.narration_duration:.2f} "
            f"picture={statistics.total_duration:.2f} "
            f"delta={statistics.coverage_delta or 0.0:+.2f}"
        )

    for entry in review.timeline:
        timeline = (
            f"{entry.timeline_range.start:.2f}-{entry.timeline_range.end:.2f}"
            if entry.timeline_range is not None
            else "unplaced"
        )
        transition = (
            f"{entry.transition_in.value}@{entry.transition_duration:.2f}"
            if entry.transition_duration > 0.0
            else entry.transition_in.value
        )
        lines.append(
            f"{entry.order:03d} {entry.clip_id} {entry.source} "
            f"src={entry.source_range.start:.2f}-{entry.source_range.end:.2f} "
            f"tl={timeline} d={entry.duration:.2f} in={transition} "
            f"shot={entry.shot_type.value} scene={entry.scene_key or '-'} "
            f"beat={entry.beat_index if entry.beat_index is not None else '-'} "
            f"| {entry.reason}"
        )

    lines.extend(
        f"{_MARK[issue.severity]} {issue.code} {issue.message}"
        + (f" | {issue.hint}" if issue.hint else "")
        for issue in review.notes
    )
    if review.plan_notes:
        lines.append(f"# author_notes: {review.plan_notes}")
    return lines


def _diff_digest(result: PlanDiff, before_path: Path, after_path: Path) -> list[str]:
    lines = [
        f"# diff before={before_path.name} after={after_path.name} "
        f"identical={str(result.is_identical).lower()} "
        f"added={len(result.added)} removed={len(result.removed)} "
        f"changed={len(result.changed)} reordered={str(result.reordered).lower()} "
        f"duration={result.duration_before:.2f}->{result.duration_after:.2f}"
        f"({result.duration_delta:+.2f})"
    ]
    lines.extend(f"+ {clip_id}" for clip_id in result.added)
    lines.extend(f"- {clip_id}" for clip_id in result.removed)
    lines.extend(
        f"~ {change.clip_id} [{','.join(change.fields)}] {change.before} -> {change.after}"
        for change in result.changed
    )
    if result.summary_fields:
        lines.append("# plan_level_changes: " + " ".join(result.summary_fields))
    return lines


def _subtitle_digest(plan: EditPlan, destination: Path | None) -> list[str]:
    target = destination.as_posix() if destination is not None else "(not saved)"
    lines = [f"# plan={target} cues={len(plan.subtitles)} duration={plan.timeline_duration:.2f}"]
    lines.extend(
        f"c{index:03d} {cue.range.start:.2f}-{cue.range.end:.2f} "
        f"lines={cue.line_count} words={len(cue.words)} "
        f"| {cue.text.replace(chr(10), ' | ')}"
        for index, cue in enumerate(plan.subtitles)
    )
    return lines


__all__ = ["app"]
