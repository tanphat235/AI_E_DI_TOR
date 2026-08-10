"""Finding CapCut's draft directory.

Separate from the exporter because it is the one part that depends on the machine rather
than on the format, and because "where does CapCut keep drafts" is a question worth
answering once. ``aive doctor`` and the export command both ask it.

CapCut and its Chinese edition JianYing share a draft format but not a folder, and the
folder moved between releases. Rather than encode one answer, this checks the known
locations in preference order and reports which one it found — a user with both installed
should be told which they are about to write into.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

from app.utils.logging import get_logger

logger = get_logger(__name__)

DRAFT_FOLDER_NAME = "com.lveditor.draft"
"""CapCut's draft subfolder. The reverse-DNS name is CapCut's own."""


@dataclass(frozen=True, slots=True)
class DraftLocation:
    """A discovered draft directory, and which product it belongs to."""

    path: Path
    product: str
    """``CapCut`` or ``JianYing``, for reporting."""
    exists: bool


def candidate_locations() -> tuple[DraftLocation, ...]:
    """Every place a draft directory might be, most likely first.

    Returned whether or not they exist, so ``doctor`` can show a user what was looked for
    rather than only that nothing was found.
    """
    if sys.platform == "win32":
        roots = _windows_roots()
    elif sys.platform == "darwin":
        roots = _macos_roots()
    else:
        # CapCut ships no Linux desktop build. Returning nothing is the honest answer.
        return ()

    return tuple(
        DraftLocation(path=path, product=product, exists=path.is_dir()) for path, product in roots
    )


def _windows_roots() -> list[tuple[Path, str]]:
    local = os.environ.get("LOCALAPPDATA")
    if not local:  # pragma: no cover - LOCALAPPDATA is always set on Windows
        return []
    base = Path(local)
    return [
        (base / "CapCut" / "User Data" / "Projects" / DRAFT_FOLDER_NAME, "CapCut"),
        (base / "JianyingPro" / "User Data" / "Projects" / DRAFT_FOLDER_NAME, "JianYing"),
    ]


def _macos_roots() -> list[tuple[Path, str]]:
    home = Path.home()
    movies = home / "Movies"
    return [
        (movies / "CapCut" / "User Data" / "Projects" / DRAFT_FOLDER_NAME, "CapCut"),
        (movies / "JianyingPro" / "User Data" / "Projects" / DRAFT_FOLDER_NAME, "JianYing"),
    ]


def find_draft_dir() -> Path | None:
    """The first draft directory that exists, or ``None``.

    ``None`` rather than a guessed path: creating CapCut's folder structure ourselves would
    produce a draft directory the app has never heard of, and the user would be left looking
    for a project that cannot appear.
    """
    for location in candidate_locations():
        if location.exists:
            logger.debug("Found %s drafts at %s", location.product, location.path)
            return location.path
    return None


__all__ = ["DRAFT_FOLDER_NAME", "DraftLocation", "candidate_locations", "find_draft_dir"]
