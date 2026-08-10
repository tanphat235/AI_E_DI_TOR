"""``aive`` - the command line entry point.

Only the commands Phase 1 actually implements are registered. Registering a
placeholder that exits with "not implemented" would be worse than leaving it out:
an AI director discovers capabilities by reading ``--help``, and a command that
exists but does nothing is a trap. What is not here yet is documented in
``docs/AGENT_TOOLS.md`` as a planned contract instead.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Annotated

import typer

from app.cli import (
    analyze_cmd,
    config_cmd,
    export_cmd,
    narrate_cmd,
    plan_cmd,
    project_cmd,
    rules_cmd,
    schema_cmd,
    subtitle_cmd,
)
from app.cli.doctor_cmd import doctor
from app.cli.output import ExitCode, emit_error, force_utf8
from app.cli.render_cmd import render
from app.config.settings import ConfigError
from app.utils.logging import configure_logging, get_logger

app = typer.Typer(
    name="aive",
    help=(
        "AIVE - local AI auto video editor.\n\n"
        "A toolbox driven by an AI director agent. Analysis writes JSON; the agent "
        "reads it, authors an Edit Plan, and asks AIVE to render or export it.\n\n"
        "stdout carries machine-readable output only. Logs go to stderr."
    ),
    no_args_is_help=True,
    add_completion=False,
    pretty_exceptions_enable=False,
)

app.add_typer(project_cmd.app, name="project")
app.add_typer(analyze_cmd.app, name="analyze")
app.add_typer(plan_cmd.app, name="plan")
app.add_typer(rules_cmd.app, name="rules")
app.add_typer(subtitle_cmd.app, name="subtitle")
app.add_typer(schema_cmd.app, name="schema")
app.add_typer(config_cmd.app, name="config")
app.add_typer(export_cmd.app, name="export")
app.command("narrate")(narrate_cmd.narrate_command)
app.command("render")(render)
app.command("doctor")(doctor)


@app.command("ui")
def ui(
    project: Annotated[
        Path | None,
        typer.Argument(help="Project directory to open. Optional."),
    ] = None,
) -> None:
    """Open the desktop window.

    A viewer and a launcher over the same pipeline these commands drive, not a separate
    product: every button calls the service its CLI command calls.
    """
    from app.ui.main import UiDependencyMissingError, run

    try:
        raise typer.Exit(run(project))
    except UiDependencyMissingError as exc:
        emit_error(
            "dependency.missing",
            str(exc),
            hint='install it with: pip install -e ".[ui]"',
            exit_code=ExitCode.ENVIRONMENT,
        )


logger = get_logger(__name__)


def _version_callback(value: bool) -> None:
    if value:
        from app import __version__

        # Straight to stdout: a version query is machine-readable output.
        sys.stdout.write(f"{__version__}\n")
        raise typer.Exit


@app.callback()
def main_callback(
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Log DEBUG to stderr."),
    ] = False,
    quiet: Annotated[
        bool,
        typer.Option("--quiet", "-q", help="Suppress human-facing stderr output."),
    ] = False,
    version: Annotated[
        bool,
        typer.Option(
            "--version", callback=_version_callback, is_eager=True, help="Print the version."
        ),
    ] = False,
) -> None:
    """Configure logging and stream encoding before any subcommand runs."""
    import os

    # Before anything can print. A cp1252 console cannot encode Vietnamese, and stdout is a
    # JSON contract, so the encoding is pinned rather than inherited.
    force_utf8()

    from app.config.settings import LogLevel

    if quiet:
        os.environ["AIVE_QUIET"] = "1"
    configure_logging(level=LogLevel.DEBUG if verbose else LogLevel.INFO)


def main() -> None:
    """Console-script entry point.

    Wraps the Typer app so that an unexpected exception still leaves a structured
    error on stdout. Without this, an internal failure would put a traceback on
    stderr and *nothing* on stdout, and the director would see an empty successful-
    looking read followed by a non-zero exit - the most confusing possible outcome.
    """
    try:
        app()
    except SystemExit:
        raise
    except KeyboardInterrupt:
        logger.warning("Interrupted by user")
        raise SystemExit(130) from None
    except ConfigError as exc:
        # A malformed aive.toml is user input, not an AIVE fault. Caught ahead of the
        # generic handler below because that one tells the user to report a bug, which
        # sends them looking in the wrong codebase for a typo in their own file.
        logger.debug("Configuration rejected", exc_info=exc)
        emit_error(
            "config.invalid",
            str(exc),
            hint="fix the setting named above, or delete the file to fall back to defaults",
            exit_code=ExitCode.INVALID_INPUT,
        )
    except Exception as exc:
        logger.exception("Unhandled error")
        emit_error(
            "internal.unhandled",
            f"{type(exc).__name__}: {exc}",
            hint="This is a bug in AIVE. Re-run with --verbose for a full traceback.",
            exit_code=ExitCode.INTERNAL,
        )


if __name__ == "__main__":
    main()
