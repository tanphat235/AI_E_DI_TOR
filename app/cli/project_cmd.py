"""``aive project`` - create and inventory project folders."""

from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

import typer

from app.cli.output import ExitCode, emit_digest, emit_error, emit_json, note, write_json_file
from app.models.common import MediaKind
from app.models.project import MediaEntry, ProjectManifest
from app.services.paths import ProjectPaths, classify
from app.utils.logging import get_logger

logger = get_logger(__name__)

app = typer.Typer(no_args_is_help=True, help="Create and inspect AIVE projects.")

_SAFE_ID = re.compile(r"[^A-Za-z0-9._-]+")

GITIGNORE_BODY = """\
# Derived data. Everything here is regenerable from the inputs plus an Edit Plan.
.aive/
output/
"""


def _project_id(name: str) -> str:
    """A filesystem- and JSON-safe id derived from a folder name."""
    cleaned = _SAFE_ID.sub("-", name).strip("-._")
    return cleaned or "project"


@app.command("init")
def init(
    directory: Annotated[Path, typer.Argument(help="Project directory to create.")],
    name: Annotated[
        str | None, typer.Option("--name", help="Project name. Defaults to the folder name.")
    ] = None,
) -> None:
    """Create the project folder structure.

    Safe to re-run: only missing directories are created and an existing
    ``aive.toml`` is never overwritten, so this can be used to repair a project
    whose folders were partly deleted.
    """
    paths = ProjectPaths.for_root(directory)
    created = paths.ensure()

    config_written = False
    if not paths.config_file.is_file():
        paths.config_file.write_text(_starter_config(), encoding="utf-8")
        config_written = True

    gitignore = paths.root / ".gitignore"
    if not gitignore.is_file():
        gitignore.write_text(GITIGNORE_BODY, encoding="utf-8")

    project_name = name or paths.root.name
    note(f"Initialised project '{project_name}' at {paths.root}")
    if created:
        for directory_created in created:
            relative = (
                "."
                if directory_created == paths.root
                else directory_created.relative_to(paths.root).as_posix()
            )
            note(f"  created {relative}")
    else:
        note("  all directories already existed")
    note("\nNext: put narration.wav at the project root and footage in raw/, then run:")
    note(f"  aive project scan {paths.root}")

    emit_json(
        {
            "project_id": _project_id(project_name),
            "name": project_name,
            "root": paths.root.as_posix(),
            "created_directories": [path.as_posix() for path in created],
            "config_written": config_written,
            "next_command": f"aive project scan {paths.root.as_posix()}",
        }
    )


@app.command("scan")
def scan(
    directory: Annotated[Path, typer.Argument(help="Project directory to inventory.")],
    full: Annotated[
        bool,
        typer.Option("--full", help="Print the whole manifest instead of a digest."),
    ] = False,
) -> None:
    """Inventory the media in a project and write a manifest.

    The full manifest goes to ``.aive/manifest.json``; stdout gets a digest. See
    :mod:`app.cli.output` for why that split exists.
    """
    paths = ProjectPaths.for_root(directory)
    if not paths.root.is_dir():
        emit_error(
            "project.missing",
            f"{paths.root} does not exist",
            hint=f"create it with: aive project init {paths.root.as_posix()}",
            exit_code=ExitCode.NOT_FOUND,
        )
    if not paths.exists():
        emit_error(
            "project.not_initialised",
            f"{paths.root} is not an AIVE project (no raw/ directory)",
            hint=f"initialise it with: aive project init {paths.root.as_posix()}",
            exit_code=ExitCode.NOT_FOUND,
        )

    now = datetime.now(UTC)
    narration_path = paths.find_narration()
    narration = _entry(paths, narration_path, MediaKind.NARRATION) if narration_path else None

    raw_clips = tuple(_entry(paths, path, MediaKind.RAW_VIDEO) for path in paths.find_raw_clips())
    music = tuple(_entry(paths, path, MediaKind.MUSIC) for path in paths.find_music())
    templates = tuple(paths.to_ref(path) for path in paths.find_capcut_templates())

    project_name = paths.root.name
    manifest = ProjectManifest(
        project_id=_project_id(project_name),
        name=project_name,
        created_at=now,
        scanned_at=now,
        narration=narration,
        raw_clips=raw_clips,
        music=music,
        capcut_templates=templates,
    )

    destination = write_json_file(manifest, paths.manifest_file)

    note(f"Scanned {paths.root}")
    note(f"  narration:  {narration.ref if narration else 'MISSING'}")
    note(f"  raw clips:  {len(raw_clips)}")
    note(f"  music:      {len(music)}")
    note(f"  templates:  {len(templates)}")
    note(f"  manifest -> {destination}")
    if not manifest.is_editable:
        note(
            "\nThis project is not editable yet: it needs narration audio "
            "and at least one raw clip."
        )

    if full:
        emit_json(manifest)
        return

    emit_digest(_manifest_digest(manifest, destination))


def _manifest_digest(manifest: ProjectManifest, destination: Path) -> list[str]:
    """A compact, line-oriented summary.

    Format: ``kind  path  size_mb``. One line per file, plus a header the director
    can read for the summary without counting lines itself.
    """
    lines = [
        f"# manifest={destination.as_posix()} editable={str(manifest.is_editable).lower()} "
        f"clips={len(manifest.raw_clips)} music={len(manifest.music)} "
        f"templates={len(manifest.capcut_templates)}"
    ]
    for entry in manifest.all_entries():
        size_mb = entry.size_bytes / (1024 * 1024)
        lines.append(f"{entry.kind.value:<11} {entry.ref!s:<40} {size_mb:>9.1f}MB")
    for template in manifest.capcut_templates:
        lines.append(f"{'capcut_tmpl':<11} {template!s:<40} {'-':>11}")
    return lines


def _entry(paths: ProjectPaths, path: Path, fallback_kind: MediaKind) -> MediaEntry:
    """Build a manifest entry, preferring the extension-derived kind."""
    stat = path.stat()
    detected = classify(path)
    kind = detected if detected is not MediaKind.UNKNOWN else fallback_kind
    # A file in music/ is music even if its stem happens to be "voice".
    if fallback_kind is MediaKind.MUSIC:
        kind = MediaKind.MUSIC
    elif fallback_kind is MediaKind.NARRATION:
        kind = MediaKind.NARRATION
    return MediaEntry(
        ref=paths.to_ref(path),
        kind=kind,
        size_bytes=stat.st_size,
        modified_at=datetime.fromtimestamp(stat.st_mtime, tz=UTC),
    )


def _starter_config() -> str:
    """The per-project config written by ``init``.

    Ships commented out on purpose. An empty-but-documented file tells the user what
    is tunable without changing any behaviour, whereas a file full of active values
    would silently pin defaults and stop them tracking upgrades.
    """
    return """\
# Per-project overrides. Anything not set here falls back to the packaged
# defaults; see app/config/default.toml for the full list with explanations.
#
# Uncomment only what you want to change for THIS project.

# [output]
# aspect_ratio = "9:16"     # vertical, for shorts
# width = 1080
# height = 1920
# fps = 30.0

# [rules]
# min_clip_duration = 1.0   # faster cutting
# max_clip_duration = 5.0
# transition_duration = 0.25

# [speech]
# language = "en"           # skip autodetection
# model = "large-v3"        # slower, more accurate

# [subtitle]
# max_chars_per_line = 32
# font_size = 64
"""


__all__ = ["app"]
