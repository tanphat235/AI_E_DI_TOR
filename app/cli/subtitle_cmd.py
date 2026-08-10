"""``aive subtitle`` - generate subtitle files.

Cues are timed against the **timeline**, not the source narration, so this command is
the sanctioned way to produce them: it applies the cleanup mapping that a
hand-authored plan would have to get right by arithmetic.

Two sources of timeline truth, in order of precedence:

* An Edit Plan's ``narration.kept_ranges``, when one is supplied. This is what a
  finished edit is actually cut to.
* Otherwise the cleanup report from ``aive analyze audio``, which is the same thing
  before any footage has been chosen.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from app.cli.output import ExitCode, emit_digest, emit_error, emit_json, note
from app.config.settings import load_settings
from app.models.common import SubtitleFormat, TimeRange
from app.models.edit_plan import EditPlan, SubtitleCue
from app.models.speech import NarrationAnalysis
from app.services.paths import ProjectPaths
from app.subtitles.builder import SubtitleBuilder
from app.subtitles.registry import writer_for
from app.utils.logging import get_logger

logger = get_logger(__name__)

app = typer.Typer(no_args_is_help=True, help="Generate subtitle files.")


@app.command("build")
def build(
    project: Annotated[Path, typer.Argument(help="Project directory.")],
    plan: Annotated[
        Path | None,
        typer.Option("--plan", help="Edit Plan to take timeline timing from."),
    ] = None,
    formats: Annotated[
        list[str] | None,
        typer.Option("--format", "-f", help="Repeatable: srt, ass. Defaults to config."),
    ] = None,
    raw: Annotated[
        bool,
        typer.Option(
            "--raw",
            help="Time against the original narration, ignoring cleanup. For review only.",
        ),
    ] = False,
    full: Annotated[
        bool,
        typer.Option("--full", help="Print the cues as JSON instead of a digest."),
    ] = False,
) -> None:
    """Build subtitles from the narration analysis."""
    paths = ProjectPaths.for_root(project)
    settings = load_settings(paths.root)

    analysis = _load_analysis(paths)
    requested = _resolve_formats(formats, settings.subtitle.formats)

    kept_ranges: tuple[TimeRange, ...] | None
    if raw:
        kept_ranges = None
        source = "raw narration (cleanup ignored)"
    elif plan is not None:
        kept_ranges = _kept_ranges_from_plan(plan)
        source = f"plan {plan.name}"
    else:
        kept_ranges = analysis.kept_ranges
        source = "cleanup report"

    # Word timings are only consumed by the ASS writer, so skip building them when
    # nothing asked for ASS.
    wants_word_timings = any(
        writer_for(item, settings.subtitle).supports_word_timings for item in requested
    )
    cues = SubtitleBuilder(settings.subtitle).build(
        analysis.transcript,
        kept_ranges=kept_ranges,
        include_word_timings=wants_word_timings,
    )

    if not cues:
        emit_error(
            "subtitle.empty",
            "no cues survived: every recognised word falls inside a removed range",
            hint="check the cleanup report, or re-run with --raw to ignore cleanup",
            exit_code=ExitCode.INVALID_INPUT,
        )

    written: list[Path] = []
    for subtitle_format in requested:
        writer = writer_for(subtitle_format, settings.subtitle)
        destination = paths.subtitle_file(subtitle_format.value)
        written.append(writer.write(cues, destination, style=settings.subtitle))

    note(f"Built {len(cues)} cue(s) from {source}")
    note(f"  timing:  {'timeline' if kept_ranges is not None else 'source'}")
    note(f"  karaoke: {'on' if settings.subtitle.karaoke and wants_word_timings else 'off'}")
    for path in written:
        note(f"  wrote {path}")

    if full:
        emit_json({"cues": [cue.model_dump(mode="json") for cue in cues]})
        return
    emit_digest(_digest(cues, written, karaoke=settings.subtitle.karaoke))


def _load_analysis(paths: ProjectPaths) -> NarrationAnalysis:
    if not paths.narration_file.is_file():
        emit_error(
            "narration.not_analysed",
            f"no narration analysis at {paths.narration_file}",
            hint=f"run: aive analyze audio {paths.root.as_posix()}",
            exit_code=ExitCode.NOT_FOUND,
        )
    try:
        return NarrationAnalysis.model_validate_json(
            paths.narration_file.read_text(encoding="utf-8")
        )
    except (OSError, ValueError) as exc:
        emit_error(
            "narration.unreadable",
            f"could not read {paths.narration_file}: {exc}",
            hint=f"delete it and re-run: aive analyze audio {paths.root.as_posix()}",
            exit_code=ExitCode.INVALID_INPUT,
        )


def _kept_ranges_from_plan(plan_path: Path) -> tuple[TimeRange, ...] | None:
    if not plan_path.is_file():
        emit_error(
            "plan.missing",
            f"no Edit Plan at {plan_path}",
            exit_code=ExitCode.NOT_FOUND,
        )
    try:
        parsed = EditPlan.model_validate_json(plan_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        emit_error(
            "plan.invalid",
            f"could not read {plan_path}: {exc}",
            hint="run: aive schema show edit-plan",
            exit_code=ExitCode.INVALID_INPUT,
        )
    if parsed.narration is None:
        # A music-only montage legitimately has no narration track, so this is not an
        # error - it just means there is no cleanup mapping to apply.
        note("Plan has no narration track; timing against the raw transcript")
        return None
    return parsed.narration.kept_ranges


def _resolve_formats(
    requested: list[str] | None, configured: tuple[SubtitleFormat, ...]
) -> tuple[SubtitleFormat, ...]:
    if not requested:
        return configured
    resolved: list[SubtitleFormat] = []
    for name in requested:
        try:
            resolved.append(SubtitleFormat(name.strip().lower().lstrip(".")))
        except ValueError:
            emit_error(
                "subtitle.unknown_format",
                f"unknown subtitle format {name!r}",
                hint=f"available: {', '.join(item.value for item in SubtitleFormat)}",
                exit_code=ExitCode.USAGE,
            )
    return tuple(dict.fromkeys(resolved))


def _digest(cues: tuple[SubtitleCue, ...], written: list[Path], *, karaoke: bool) -> list[str]:
    """One line per cue: index, timeline range, line count, then the text.

    Newlines inside a cue are shown as ``|`` so each cue stays on one line and the
    digest remains line-oriented.
    """
    lines = [
        f"# cues={len(cues)} karaoke={str(karaoke).lower()} "
        f"files={','.join(path.as_posix() for path in written)}"
    ]
    lines.extend(
        f"c{index:03d} {cue.range.start:.2f}-{cue.range.end:.2f} "
        f"lines={cue.line_count} | {cue.text.replace(chr(10), ' | ')}"
        for index, cue in enumerate(cues)
    )
    return lines


__all__ = ["app"]
