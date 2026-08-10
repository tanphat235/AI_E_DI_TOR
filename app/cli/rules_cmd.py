"""``aive rules`` - the guardrails around the AI director.

Three commands, corresponding to the two halves of the Rule Engine:

* ``scenes`` runs the *pre*-planning filters and reports which footage is usable, so the
  director knows what it may choose from before it chooses.
* ``normalize`` rewrites an authored plan into a renderable one and reports every change.
* ``validate`` checks a plan and changes nothing.

Normalise **before** validating. Normalisation fixes much of what validation would
otherwise reject, and it reports each fix, so running it first turns a wall of errors into
a short list of genuine editorial decisions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from app.cli.output import ExitCode, emit_digest, emit_error, emit_json, note, write_json_file
from app.config.settings import load_settings
from app.models.common import Issue, MediaRef, Severity
from app.models.edit_plan import EditPlan, EditPlanReport
from app.models.media import MediaProbe
from app.models.speech import NarrationAnalysis
from app.models.video import FootageAnalysis
from app.rule_engine.context import RuleContext
from app.rule_engine.engine import DefaultRuleEngine
from app.rule_engine.filters import EligibilityReport, filter_scenes
from app.services.paths import ProjectPaths
from app.utils.logging import get_logger

logger = get_logger(__name__)

app = typer.Typer(no_args_is_help=True, help="Validate and normalise Edit Plans.")

_SEVERITY_MARK = {Severity.ERROR: "E", Severity.WARNING: "W", Severity.INFO: "i"}


@app.command("validate")
def validate(
    plan_path: Annotated[Path, typer.Argument(help="Edit Plan to check.")],
    project: Annotated[
        Path | None,
        typer.Option("--project", help="Project directory. Defaults to the plan's folder."),
    ] = None,
    full: Annotated[bool, typer.Option("--full", help="Emit the report as JSON.")] = False,
) -> None:
    """Check a plan without changing it.

    Exits `4` when the plan has errors, so a script can branch on it. Warnings do not
    block: they are things an editor would question, not things that break a render.
    """
    paths, plan = _load(plan_path, project)
    context = _build_context(paths)

    report = DefaultRuleEngine().validate(plan, context)
    _report(plan_path, report, full=full, normalised=False)
    if not report.ok:
        raise SystemExit(ExitCode.INVALID_INPUT.status)


@app.command("normalize")
def normalize(
    plan_path: Annotated[Path, typer.Argument(help="Edit Plan to normalise.")],
    project: Annotated[
        Path | None,
        typer.Option("--project", help="Project directory. Defaults to the plan's folder."),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Where to write the result. Defaults to stdout only."),
    ] = None,
    in_place: Annotated[
        bool,
        typer.Option("--in-place", help="Overwrite the input plan."),
    ] = False,
    full: Annotated[bool, typer.Option("--full", help="Emit the normalised plan as JSON.")] = False,
) -> None:
    """Place clips, clamp what is out of bounds, then validate the result.

    Writes nothing unless asked. A normaliser that silently overwrote the plan would make
    the director's own record of its decisions unreadable, so saving is explicit.
    """
    paths, plan = _load(plan_path, project)
    context = _build_context(paths)

    normalised, report = DefaultRuleEngine().normalize(plan, context)

    destination = plan_path if in_place else output
    if destination is not None:
        write_json_file(normalised, destination)
        note(f"Wrote the normalised plan to {destination}")

    if full:
        emit_json(normalised)
        note("")
        _log_report(report, normalised=True)
        if not report.ok:
            raise SystemExit(ExitCode.INVALID_INPUT.status)
        return

    _report(plan_path, report, full=False, normalised=True)
    if not report.ok:
        raise SystemExit(ExitCode.INVALID_INPUT.status)


@app.command("scenes")
def scenes(
    project: Annotated[Path, typer.Argument(help="Project directory.")],
    full: Annotated[bool, typer.Option("--full", help="Emit the report as JSON.")] = False,
) -> None:
    """Report which scenes are eligible for selection, and why the rest are not.

    Run this before authoring a plan. It answers "what may I use?" once, instead of
    discovering the answer through rejected clips.
    """
    paths = ProjectPaths.for_root(project)
    footage = _load_footage(paths)
    if footage is None:
        emit_error(
            "footage.not_analysed",
            f"no footage analysis at {paths.footage_analysis_file}",
            hint=f"run: aive analyze video {paths.root.as_posix()}",
            exit_code=ExitCode.NOT_FOUND,
        )

    settings = load_settings(paths.root)
    report = filter_scenes(footage, rules=settings.rules)

    note("")
    note(f"  scenes:   {len(report.verdicts)}")
    note(f"  eligible: {len(report.eligible)} ({report.total_eligible_duration:.1f}s of footage)")
    note(f"  rejected: {len(report.rejected)}")
    if not report.eligible:
        note("\n  Nothing is usable. Lower the thresholds in [rules], or shoot more.")

    if full:
        emit_json(
            {
                "eligible": [scene.key for scene in report.eligible],
                "rejected": [
                    {
                        "scene": verdict.key,
                        "issues": [issue.model_dump(mode="json") for issue in verdict.issues],
                    }
                    for verdict in report.rejected
                ],
            }
        )
        return
    emit_digest(_scenes_digest(report))


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #


def _load(plan_path: Path, project: Path | None) -> tuple[ProjectPaths, EditPlan]:
    """Read the plan and work out which project it belongs to.

    The project defaults to the plan's own folder, because the convention is
    ``<project>/edit_plan.json``. Guessing correctly here matters: without a project root
    the source-existence and bounds checks silently do nothing.
    """
    if not plan_path.is_file():
        emit_error(
            "plan.missing",
            f"no Edit Plan at {plan_path}",
            hint="pass the path to a plan; `aive schema example` prints a valid one",
            exit_code=ExitCode.NOT_FOUND,
        )

    try:
        plan = EditPlan.model_validate_json(plan_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        emit_error(
            "plan.invalid",
            f"{plan_path.name} is not a valid Edit Plan: {exc}",
            hint="run: aive schema show edit-plan",
            exit_code=ExitCode.INVALID_INPUT,
        )

    root = project if project is not None else plan_path.parent
    return ProjectPaths.for_root(root), plan


def _build_context(paths: ProjectPaths) -> RuleContext:
    """Assemble a rule context from whatever analysis exists.

    Deliberately tolerant. Validating before running analysis still checks durations,
    overlaps and transitions; it just cannot check source bounds. Refusing outright would
    push users toward skipping validation, which is worse than a partial check.
    """
    settings = load_settings(paths.root)
    footage = _load_footage(paths)
    narration = _load_narration(paths)

    extra: dict[MediaRef, MediaProbe] = {}
    if narration is not None:
        # The narration duration comes from the transcript rather than a probe, which is
        # the honest source: it is what the recogniser actually decoded.
        extra[narration.source] = MediaProbe(
            source=narration.source,
            duration=narration.transcript.duration,
            size_bytes=0,
        )

    if footage is None:
        note("No footage analysis found, so source bounds cannot be checked.")
        note(f"  run: aive analyze video {paths.root.as_posix()}")

    return RuleContext.build(
        settings.rules,
        subtitle=settings.subtitle,
        project_root=paths.root if paths.root.is_dir() else None,
        footage=footage,
        narration=narration,
        extra_probes=extra,
    )


def _load_footage(paths: ProjectPaths) -> FootageAnalysis | None:
    if not paths.footage_analysis_file.is_file():
        return None
    try:
        return FootageAnalysis.model_validate_json(
            paths.footage_analysis_file.read_text(encoding="utf-8")
        )
    except (OSError, ValueError) as exc:
        logger.debug("Ignoring unreadable footage analysis: %s", exc)
        return None


def _load_narration(paths: ProjectPaths) -> NarrationAnalysis | None:
    if not paths.narration_file.is_file():
        return None
    try:
        return NarrationAnalysis.model_validate_json(
            paths.narration_file.read_text(encoding="utf-8")
        )
    except (OSError, ValueError) as exc:
        logger.debug("Ignoring unreadable narration analysis: %s", exc)
        return None


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #


def _report(plan_path: Path, report: EditPlanReport, *, full: bool, normalised: bool) -> None:
    _log_report(report, normalised=normalised)
    if full:
        emit_json(report)
        return
    emit_digest(_report_digest(plan_path, report, normalised=normalised))


def _log_report(report: EditPlanReport, *, normalised: bool) -> None:
    changes = [issue for issue in report.issues if issue.severity is Severity.INFO]
    note("")
    if normalised:
        note(f"  changes:  {len(changes)}")
    note(f"  errors:   {len(report.errors)}")
    note(f"  warnings: {len(report.warnings)}")
    note(f"  verdict:  {'RENDERABLE' if report.ok else 'BLOCKED'}")
    if report.errors and not normalised:
        note("\n  Try `aive rules normalize` first - it fixes much of this automatically.")


def _report_digest(plan_path: Path, report: EditPlanReport, *, normalised: bool) -> list[str]:
    """One line per finding: severity, code, location, message, then the hint.

    The hint is included rather than dropped for brevity. It is the field that lets the
    director correct itself instead of guessing, which is worth more than the tokens it
    costs.
    """
    lines = [
        f"# plan={plan_path.as_posix()} ok={str(report.ok).lower()} "
        f"errors={len(report.errors)} warnings={len(report.warnings)} "
        f"normalised={str(normalised).lower()}"
    ]
    lines.extend(_issue_line(issue) for issue in report.issues)
    return lines


def _issue_line(issue: Issue) -> str:
    mark = _SEVERITY_MARK[issue.severity]
    location = f" {issue.location}" if issue.location else ""
    hint = f" | {issue.hint}" if issue.hint else ""
    return f"{mark} {issue.code}{location} {issue.message}{hint}"


def _scenes_digest(report: EligibilityReport) -> list[str]:
    """``+`` for usable, ``-`` for rejected, with the reasons."""
    lines = [
        f"# scenes={len(report.verdicts)} eligible={len(report.eligible)} "
        f"rejected={len(report.rejected)} "
        f"eligible_duration={report.total_eligible_duration:.1f}"
    ]
    for verdict in report.verdicts:
        scene = verdict.scene
        mark = "+" if verdict.eligible else "-"
        reasons = " ".join(issue.code for issue in verdict.issues)
        lines.append(
            f"{mark} {scene.key} {scene.range.start:.2f}-{scene.range.end:.2f} "
            f"d={scene.range.duration:.1f} q={scene.quality.overall:.2f} "
            f"mot={scene.motion.level.value} shot={scene.shot_type.value}"
            + (f" {reasons}" if reasons else "")
        )
    return lines


__all__ = ["app"]
