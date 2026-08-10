"""``aive doctor`` - environment preflight.

The first command anyone runs, and the one that turns "the render failed" into "the
render was never going to work, and here is why". It resolves every external
dependency, reports what it found and where it came from, and distinguishes hard
failures from degraded-but-usable.
"""

from __future__ import annotations

import importlib.util
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any

import typer

from app.cli.output import ExitCode, emit_json, note
from app.config.settings import config_search_paths, load_settings
from app.services.ffmpeg_locator import FFmpegLocator, FFmpegNotFoundError
from app.utils.logging import get_logger

logger = get_logger(__name__)

OPTIONAL_MODULES: tuple[tuple[str, str, str], ...] = (
    ("faster_whisper", "speech", "speech recognition"),
    ("cv2", "video", "scene analysis"),
    ("scenedetect", "video", "scene detection"),
    ("numpy", "audio", "music features and video metrics"),
    ("av", "audio", "media probing and audio decoding"),
    ("PySide6", "ui", "desktop UI"),
)
"""Optional imports, as ``(module, extra, what it unlocks)``.

Checked with :func:`importlib.util.find_spec` rather than imported: importing
PySide6 or OpenCV costs hundreds of milliseconds and can fail for reasons that have
nothing to do with whether it is installed.
"""


@dataclass(frozen=True, slots=True)
class Check:
    """One diagnostic result."""

    name: str
    ok: bool
    detail: str
    required: bool = True

    @property
    def status(self) -> str:
        if self.ok:
            return "ok"
        return "missing" if self.required else "not installed"


def _python_check() -> Check:
    version = sys.version_info
    ok = (version.major, version.minor) >= (3, 12)
    return Check(
        name="python",
        ok=ok,
        detail=f"{platform.python_version()} at {sys.executable}"
        + ("" if ok else " (AIVE needs 3.12 or newer)"),
    )


def _ffmpeg_checks(project_dir: Path | None) -> list[Check]:
    settings = load_settings(project_dir)
    locator = FFmpegLocator(settings.media)
    checks: list[Check] = []

    try:
        tools = locator.locate()
    except FFmpegNotFoundError as exc:
        checks.append(Check(name="ffmpeg", ok=False, detail=str(exc).splitlines()[0]))
        checks.append(
            Check(name="ffprobe", ok=False, detail="not checked; ffmpeg missing", required=False)
        )
        return checks

    checks.append(
        Check(
            name="ffmpeg",
            ok=True,
            detail=f"{tools.ffmpeg.version()} [{tools.ffmpeg.source}] at {tools.ffmpeg.path}",
        )
    )

    if tools.ffprobe is not None:
        checks.append(
            Check(
                name="ffprobe",
                ok=True,
                detail=(
                    f"{tools.ffprobe.version()} [{tools.ffprobe.source}] at {tools.ffprobe.path}"
                ),
            )
        )
    else:
        # Not an error. The vendored imageio-ffmpeg wheel contains ffmpeg only, and
        # Phase 3 will probe with PyAV instead. Worth reporting so the gap is
        # visible now rather than surprising at render time.
        checks.append(
            Check(
                name="ffprobe",
                ok=False,
                required=False,
                detail=(
                    "not found; the vendored ffmpeg wheel ships no ffprobe. "
                    "Media probing will use PyAV (installed with the 'speech' extra). "
                    "Install a system FFmpeg for a native ffprobe."
                ),
            )
        )
    return checks


def _module_checks() -> list[Check]:
    checks: list[Check] = []
    for module, extra, purpose in OPTIONAL_MODULES:
        found = importlib.util.find_spec(module) is not None
        detail = f"{purpose}" if found else f'{purpose}; install with: pip install -e ".[{extra}]"'
        checks.append(Check(name=module, ok=found, detail=detail, required=False))
    return checks


def _config_check(project_dir: Path | None) -> Check:
    layers = config_search_paths(project_dir)
    rendered = " < ".join(path.name for path in layers)
    return Check(
        name="config",
        ok=bool(layers),
        detail=f"{len(layers)} layer(s), lowest first: {rendered}"
        if layers
        else "no config files found",
    )


def _project_check(project_dir: Path | None) -> Check | None:
    if project_dir is None:
        return None
    from app.services.paths import ProjectPaths

    paths = ProjectPaths.for_root(project_dir)
    if not paths.root.is_dir():
        return Check(name="project", ok=False, detail=f"{paths.root} does not exist")
    if not paths.exists():
        return Check(
            name="project",
            ok=False,
            detail=f"{paths.root} exists but is not initialised; run: aive project init",
        )
    narration = paths.find_narration()
    clips = paths.find_raw_clips()
    music = paths.find_music()
    return Check(
        name="project",
        ok=True,
        detail=(
            f"{paths.root} | narration: {narration.name if narration else 'none'} "
            f"| raw clips: {len(clips)} | music: {len(music)}"
        ),
    )


def _capcut_check() -> Check:
    """Whether a CapCut draft folder exists.

    Optional, and the detail matters more than the verdict: "it exported but I cannot find
    it" is the likeliest export complaint, and it is almost always this folder not existing
    because CapCut has never saved a project.
    """
    from app.exporters.capcut.locate import candidate_locations
    from app.exporters.capcut.schema import TARGET_CAPCUT_VERSION

    found = [location for location in candidate_locations() if location.exists]
    if found:
        return Check(
            name="capcut",
            ok=True,
            required=False,
            detail=f"{found[0].product} drafts at {found[0].path} "
            f"(exporter targets {TARGET_CAPCUT_VERSION})",
        )
    return Check(
        name="capcut",
        ok=False,
        required=False,
        detail=(
            "no draft folder found. Create one project in CapCut so the folder exists, "
            "or pass -o to `aive export capcut` to write elsewhere"
        ),
    )


def doctor(
    project: Annotated[
        Path | None,
        typer.Argument(help="Optional project directory to include in the report."),
    ] = None,
) -> None:
    """Check that everything AIVE needs is present and working."""
    checks: list[Check] = [_python_check(), _config_check(project)]
    checks.extend(_ffmpeg_checks(project))
    project_check = _project_check(project)
    if project_check is not None:
        checks.append(project_check)
    checks.extend(_module_checks())
    checks.append(_capcut_check())

    required_failures = [check for check in checks if check.required and not check.ok]
    optional_missing = [check for check in checks if not check.required and not check.ok]

    payload: dict[str, Any] = {
        "ok": not required_failures,
        "platform": f"{platform.system()} {platform.release()}",
        "aive_version": _aive_version(),
        "checks": [
            {
                "name": check.name,
                "status": check.status,
                "required": check.required,
                "detail": check.detail,
            }
            for check in checks
        ],
    }

    for check in checks:
        marker = "OK  " if check.ok else ("FAIL" if check.required else "--  ")
        note(f"  {marker} {check.name}: {check.detail}")
    if required_failures:
        note(f"\n{len(required_failures)} required check(s) failed.")
    elif optional_missing:
        note(
            f"\nAll required checks passed. {len(optional_missing)} "
            "optional component(s) not installed."
        )
    else:
        note("\nEverything is installed.")

    emit_json(payload)
    if required_failures:
        raise SystemExit(ExitCode.ENVIRONMENT.status)


def _aive_version() -> str:
    from app import __version__

    return __version__


__all__ = ["doctor"]
