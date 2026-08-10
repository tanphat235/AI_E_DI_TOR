"""``aive export`` - hand the edit to another editor.

Where `render` finishes a video, this hands over something still editable, with the AI's
decisions already laid out. It is the answer to "the edit is 90% right and I want to fix the
last 10% by hand" — which is most edits.

The command refuses to guess about two things, because both produce a failure the user
cannot diagnose:

**Where CapCut keeps drafts.** Detected, never invented. Creating the folder structure
ourselves would write a draft into a directory CapCut has never heard of, and the user would
hunt for a project that cannot appear.

**Whether the draft opens.** It cannot know. The format is undocumented and version-specific,
so the digest says which CapCut release the exporter targets and the command tells the user
plainly when that differs from what they have.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from app.cli.output import ExitCode, emit_digest, emit_error, emit_json, note
from app.config.settings import AiveSettings
from app.exporters.base import ExportRequest, ExportResult
from app.exporters.capcut.exporter import CapCutExportError, default_draft_dir
from app.exporters.capcut.locate import candidate_locations
from app.exporters.capcut.schema import TARGET_CAPCUT_VERSION
from app.models.edit_plan import EditPlan
from app.plan.loader import PlanLoadError, load_plan
from app.services.container import build_container
from app.services.paths import ProjectPaths
from app.utils.logging import get_logger

logger = get_logger(__name__)

app = typer.Typer(no_args_is_help=True, help="Export an Edit Plan to another editor.")


@app.command("capcut")
def capcut(
    plan_path: Annotated[Path, typer.Argument(help="Edit Plan to export.")],
    project: Annotated[
        Path | None,
        typer.Option("--project", help="Project directory. Defaults to the plan's folder."),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Draft directory. Defaults to CapCut's own."),
    ] = None,
    name: Annotated[
        str | None,
        typer.Option("--name", help="Draft name. Defaults to the plan's project_id."),
    ] = None,
    template: Annotated[
        Path | None,
        typer.Option("--template", help="An existing draft to inherit styling from."),
    ] = None,
    no_copy: Annotated[
        bool,
        typer.Option("--no-copy", help="Reference media in place instead of copying it."),
    ] = False,
    check: Annotated[
        bool, typer.Option("--check", help="Run preflight and stop. Writes nothing.")
    ] = False,
    full: Annotated[bool, typer.Option("--full", help="Emit the result as JSON.")] = False,
) -> None:
    """Write a CapCut draft the user can open and keep editing.

    Media is **copied into the draft** by default. A draft that references files elsewhere
    breaks the moment the user reorganises their footage, and CapCut's failure mode is a
    timeline of red placeholders. `--no-copy` is there for a stable layout.

    Pass `--template` pointing at one of the user's own drafts to inherit its fonts, colours
    and canvas settings; only the timeline is replaced.
    """
    paths, plan = _load(plan_path, project)
    container = build_container(paths.root)

    try:
        exporter = container.exporters.get("capcut")
    except KeyError as exc:  # pragma: no cover - always registered
        emit_error("internal.not_wired", str(exc), exit_code=ExitCode.INTERNAL)

    draft_root = output or _draft_root(container.settings)
    draft_name = name or plan.project_id
    destination = draft_root if output is not None else draft_root / draft_name

    request = ExportRequest(
        plan=plan,
        project_root=paths.root,
        destination=destination,
        template=template,
        copy_media=not no_copy and container.settings.capcut.copy_media,
        project_name=draft_name,
    )

    problems = exporter.preflight(request)
    if problems:
        emit_error(
            "export.preflight_failed",
            "; ".join(problems),
            hint="fix each problem above; most are a missing file or an unplaced plan",
            exit_code=ExitCode.INVALID_INPUT,
            details={"problems": list(problems)},
        )

    if check:
        note("Preflight passed. Nothing was written.")
        emit_digest(
            [
                f"# preflight plan={plan_path.as_posix()} ok=true "
                f"clips={len(plan.clips)} destination={destination.as_posix()} "
                f"targets_capcut={TARGET_CAPCUT_VERSION}"
            ]
        )
        return

    try:
        result = exporter.export(request)
    except CapCutExportError as exc:
        emit_error(
            "export.failed",
            str(exc),
            hint="check the draft directory is writable and no CapCut project is open",
            exit_code=ExitCode.INTERNAL,
        )

    _report(result, plan_path, plan, full=full)


@app.command("targets")
def targets() -> None:
    """Report where drafts would be written, and which CapCut version is targeted.

    Worth a command of its own: "it exported but I cannot find it" is the likeliest
    complaint, and it is almost always a second CapCut install or a moved folder.
    """
    locations = candidate_locations()
    note(f"This exporter targets CapCut {TARGET_CAPCUT_VERSION}.")
    note("")
    for location in locations:
        mark = "found" if location.exists else "absent"
        note(f"  [{mark:>6}] {location.product}: {location.path}")
    if not locations:
        note("  CapCut ships no build for this platform.")
    elif not any(location.exists for location in locations):
        note("")
        note("  No draft folder exists yet. Create one project in CapCut and re-run;")
        note("  AIVE will not invent the folder, because a draft written somewhere")
        note("  CapCut does not read cannot appear in its project list.")

    emit_digest(
        [
            f"# capcut target={TARGET_CAPCUT_VERSION} candidates={len(locations)} "
            f"found={sum(1 for item in locations if item.exists)}",
            *(
                f"{'+' if item.exists else '-'} {item.product} {item.path.as_posix()}"
                for item in locations
            ),
        ]
    )


def _load(plan_path: Path, project: Path | None) -> tuple[ProjectPaths, EditPlan]:
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
    if loaded.was_migrated:
        note(f"Migrated this plan from schema {loaded.migrated_from} on load.")
    root = project if project is not None else plan_path.parent
    return ProjectPaths.for_root(root), loaded.plan


def _draft_root(settings: AiveSettings) -> Path:
    found = default_draft_dir(settings)
    if found is None:
        emit_error(
            "capcut.not_found",
            "no CapCut draft folder was found on this machine",
            hint=(
                "run `aive export targets` to see where AIVE looked. Create one project "
                "in CapCut so the folder exists, or pass -o to write elsewhere."
            ),
            exit_code=ExitCode.NOT_FOUND,
        )
    return found


def _report(result: ExportResult, plan_path: Path, plan: EditPlan, *, full: bool) -> None:
    note("")
    note(f"  draft:  {result.project_dir}")
    note(f"  files:  {len(result.files_written)} written, {len(result.media_copied)} copied")
    for warning in result.warnings:
        note(f"  ! {warning}")
    if result.open_hint:
        note("")
        note(f"  {result.open_hint}")
    note("")
    note(f"  Built for CapCut {TARGET_CAPCUT_VERSION}. The draft format is undocumented,")
    note("  so a different CapCut version may not open this. If it does not, export with")
    note("  --template pointing at one of your own drafts.")

    if full:
        emit_json(
            {
                "project_dir": str(result.project_dir),
                "files_written": [str(path) for path in result.files_written],
                "media_copied": [str(path) for path in result.media_copied],
                "warnings": list(result.warnings),
                "targets_capcut": TARGET_CAPCUT_VERSION,
            }
        )
        return

    emit_digest(
        [
            f"# export plan={plan_path.as_posix()} draft={result.project_dir.as_posix()} "
            f"clips={len(plan.clips)} files={len(result.files_written)} "
            f"copied={len(result.media_copied)} warnings={len(result.warnings)} "
            f"targets_capcut={TARGET_CAPCUT_VERSION}",
            *(f"w {warning}" for warning in result.warnings),
        ]
    )


__all__ = ["app"]
