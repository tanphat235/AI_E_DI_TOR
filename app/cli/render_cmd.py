"""``aive render`` - turn an Edit Plan into a video file.

The last step, and the one where a mistake costs the most time, so the command is built
around not wasting it:

**Preflight runs first, always.** Every source file, the destination, the plan's placement.
Discovering a missing clip forty minutes into an encode is unacceptable, and `--check`
exposes that gate on its own so a director can verify a plan is renderable without
committing to the render.

**`--draft` is the default way to look at an edit.** Half resolution and a fast preset, which
is many times quicker and answers the only question a first pass asks: does this edit work?
Judge the cut on a draft and spend the full encode once.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from app.cli.output import ExitCode, emit_digest, emit_error, emit_json, note
from app.config.settings import AiveSettings
from app.models.common import SubtitleFormat
from app.models.edit_plan import EditPlan
from app.plan.loader import PlanLoadError, load_plan
from app.renderer.base import RenderRequest, RenderResult
from app.renderer.ffmpeg.graph import FilterGraphBuilder
from app.renderer.ffmpeg.renderer import RenderError
from app.services.container import build_container
from app.services.paths import ProjectPaths
from app.utils.logging import get_logger

logger = get_logger(__name__)


def render(
    plan_path: Annotated[Path, typer.Argument(help="Edit Plan to render.")],
    project: Annotated[
        Path | None,
        typer.Option("--project", help="Project directory. Defaults to the plan's folder."),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Where to write. Defaults to output/final.mp4."),
    ] = None,
    draft: Annotated[
        bool,
        typer.Option("--draft", help="Fast, half-size review encode instead of a master."),
    ] = False,
    subtitles: Annotated[
        list[str] | None,
        typer.Option("--subtitles", help="Also write a sidecar: srt, ass. Repeatable."),
    ] = None,
    burn_in: Annotated[
        bool,
        typer.Option("--burn-in", help="Burn subtitles into the picture, not just a sidecar."),
    ] = False,
    check: Annotated[
        bool,
        typer.Option("--check", help="Run preflight and stop. Encodes nothing."),
    ] = False,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Print the ffmpeg command and filter graph, then stop."),
    ] = False,
    full: Annotated[bool, typer.Option("--full", help="Emit the result as JSON.")] = False,
) -> None:
    """Encode a plan with FFmpeg.

    Runs preflight first and exits `4` on any blocking problem, naming each one. Use
    `--draft` while judging the edit and a full render once, at the end.
    """
    paths, plan = _load(plan_path, project)
    container = build_container(paths.root)
    renderer = container.renderer
    if renderer is None:  # pragma: no cover - always wired
        emit_error(
            "internal.not_wired",
            "the renderer is not available in this build",
            exit_code=ExitCode.INTERNAL,
        )

    destination = output or (paths.output / ("draft.mp4" if draft else paths.final_video.name))
    request = RenderRequest(
        plan=plan,
        project_root=paths.root,
        destination=destination,
        subtitle_formats=_parse_formats(subtitles),
        burn_in_subtitles=burn_in,
        draft=draft,
    )

    problems = renderer.preflight(request)
    if problems:
        emit_error(
            "render.preflight_failed",
            "; ".join(problems),
            hint="fix each problem above; most are a missing file or an unplaced plan",
            exit_code=ExitCode.INVALID_INPUT,
            details={"problems": list(problems)},
        )

    if check:
        note("Preflight passed. Nothing was encoded.")
        emit_digest(
            [
                f"# preflight plan={plan_path.as_posix()} ok=true "
                f"clips={len(plan.clips)} duration={plan.timeline_duration:.2f} "
                f"destination={destination.as_posix()}"
            ]
        )
        return

    if dry_run:
        _report_dry_run(container.settings, request, plan_path)
        return

    note(f"Rendering {len(plan.clips)} clip(s) to {destination}")
    if not draft:
        note("  This is a full-quality encode. --draft is several times faster for review.")

    try:
        result = renderer.render(request, on_progress=_progress)
    except RenderError as exc:
        emit_error(
            "render.failed",
            str(exc),
            hint="the ffmpeg log named above holds the filter-graph diagnostics",
            exit_code=ExitCode.INTERNAL,
            details={"log": str(exc.log_file) if exc.log_file else None},
        )

    _report(result, plan_path, draft=draft, full=full)


def _load(plan_path: Path, project: Path | None) -> tuple[ProjectPaths, EditPlan]:
    """Read the plan through the migrating loader and work out its project."""
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


def _parse_formats(raw: list[str] | None) -> tuple[SubtitleFormat, ...]:
    if not raw:
        return ()
    formats: list[SubtitleFormat] = []
    for name in raw:
        try:
            fmt = SubtitleFormat(name.lower().lstrip("."))
        except ValueError:
            emit_error(
                "subtitle.unknown_format",
                f"unknown subtitle format {name!r}",
                hint=f"available: {', '.join(f.value for f in SubtitleFormat)}",
                exit_code=ExitCode.USAGE,
            )
        if fmt not in formats:
            formats.append(fmt)
    return tuple(formats)


def _progress(fraction: float, detail: str) -> None:
    """Progress to stderr, never stdout.

    Rendering is the one command long enough that silence reads as a hang, and the one whose
    stdout a director is most likely to be parsing.
    """
    note(f"  {fraction * 100:5.1f}%  {detail}")


def _report_dry_run(settings: AiveSettings, request: RenderRequest, plan_path: Path) -> None:
    """Print the command and graph without running anything.

    Worth having as a first-class flag: a filter graph is the one part of this system a user
    might reasonably want to take away and run by hand, or paste into a bug report.
    """
    graph = FilterGraphBuilder(settings, draft=request.draft).build(
        request.plan, project_root=request.project_root, subtitle_file=None
    )
    note(f"Would render {len(request.plan.clips)} clip(s), {graph.duration:.2f}s")
    for warning in graph.warnings:
        note(f"  ! {warning}")
    emit_digest(
        [
            f"# dry_run plan={plan_path.as_posix()} inputs={len(graph.inputs)} "
            f"filters={len(graph.filters)} duration={graph.duration:.2f} "
            f"size={graph.width}x{graph.height} fps={graph.fps:g} "
            f"audio={'yes' if graph.audio_label else 'none'}",
            *(f"! {warning}" for warning in graph.warnings),
            f"filter_complex={graph.filter_complex()}",
        ]
    )


def _report(result: RenderResult, plan_path: Path, *, draft: bool, full: bool) -> None:
    note("")
    note(f"  video:    {result.video}")
    note(f"  duration: {result.duration:.2f}s")
    note(f"  took:     {result.elapsed:.1f}s")
    for path in result.subtitles:
        note(f"  subtitle: {path}")
    if draft:
        note("")
        note("  This was a draft. Re-run without --draft for the deliverable.")

    if full:
        emit_json(
            {
                "video": str(result.video),
                "subtitles": [str(path) for path in result.subtitles],
                "duration": result.duration,
                "elapsed": result.elapsed,
                "log_file": str(result.log_file) if result.log_file else None,
                "draft": draft,
            }
        )
        return

    emit_digest(
        [
            f"# render plan={plan_path.as_posix()} video={result.video.as_posix()} "
            f"draft={str(draft).lower()} duration={result.duration:.2f} "
            f"elapsed={result.elapsed:.1f} subtitles={len(result.subtitles)}",
            *(f"sub {path.as_posix()}" for path in result.subtitles),
        ]
    )


__all__ = ["render"]
