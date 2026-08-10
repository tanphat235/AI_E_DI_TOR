"""``aive config`` - inspect the merged configuration.

Layered config is only trustworthy if you can see the result. This command answers
"what value is actually in effect, and which file set it?" without anyone having to
reason about precedence in their head.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

import typer

from app.cli.output import emit_json, note
from app.config.settings import AiveSettings, config_search_paths, load_settings

app = typer.Typer(no_args_is_help=True, help="Inspect AIVE configuration.")


@app.command("show")
def show(
    project: Annotated[
        Path | None,
        typer.Argument(help="Project directory, to include its aive.toml layer."),
    ] = None,
    section: Annotated[
        str | None,
        typer.Option("--section", help="Show only one section, e.g. rules."),
    ] = None,
) -> None:
    """Print the merged configuration."""
    settings = load_settings(project)
    payload = settings.model_dump(mode="json")

    if section is not None:
        if section not in payload:
            available = ", ".join(sorted(payload))
            note(f"No section named '{section}'. Available: {available}")
            raise SystemExit(2)
        payload = {section: payload[section]}

    layers = config_search_paths(project)
    note("Config layers, lowest priority first:")
    for index, path in enumerate(layers, start=1):
        note(f"  {index}. {path}")
    note("  then: AIVE_* environment variables, then CLI flags")

    emit_json(payload)


@app.command("layers")
def layers(
    project: Annotated[Path | None, typer.Argument(help="Project directory.")] = None,
) -> None:
    """List the config files that would be read, and whether each exists."""
    from app.config.settings import PACKAGED_DEFAULTS, PROJECT_CONFIG_NAME

    candidates: list[tuple[str, Path]] = [
        ("packaged defaults", PACKAGED_DEFAULTS),
        ("site override", Path.cwd() / "config" / PROJECT_CONFIG_NAME),
    ]
    if project is not None:
        candidates.append(("project override", project / PROJECT_CONFIG_NAME))

    applied = set(config_search_paths(project))
    payload: dict[str, Any] = {
        "layers": [
            {
                "kind": kind,
                "path": path.as_posix(),
                "exists": path.is_file(),
                "applied": path in applied,
            }
            for kind, path in candidates
        ],
        "env_prefix": "AIVE_",
        "env_nested_delimiter": "__",
    }
    emit_json(payload)


@app.command("defaults")
def defaults() -> None:
    """Print the field defaults, with no config *file* layer applied.

    Useful for telling a genuine default apart from a value some config file
    happened to set to the same thing. Note that ``AIVE_*`` environment variables
    still apply here, because they are read by the settings class itself rather
    than by a file source.
    """
    emit_json(AiveSettings().model_dump(mode="json"))


__all__ = ["app"]
