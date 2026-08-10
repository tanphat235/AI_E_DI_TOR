"""Launching the desktop UI.

Kept apart from :mod:`app.ui.window` so that importing the window — in a test, say — does
not construct a ``QApplication`` as a side effect.

PySide6 is imported *inside* the function rather than at module scope. The UI is an optional
extra, and `aive --help` on an install without it must still work; a top-level import here
would make PySide6 a hard dependency of the CLI entry point.
"""

from __future__ import annotations

import sys
from pathlib import Path

from app.utils.logging import get_logger

logger = get_logger(__name__)


class UiDependencyMissingError(RuntimeError):
    """Raised when PySide6 is not installed.

    Distinct from any other failure: the fix is `pip install`, and the CLI maps it to exit
    5 (environment) rather than to a traceback.
    """


def run(project: Path | None = None) -> int:
    """Open the window and run the event loop. Returns the process exit code."""
    try:
        from PySide6.QtWidgets import QApplication
    except ImportError as exc:
        msg = 'the desktop UI needs PySide6. Install it with: pip install -e ".[ui]"'
        raise UiDependencyMissingError(msg) from exc

    from app.ui.window import MainWindow

    application = QApplication.instance() or QApplication(sys.argv)
    application.setApplicationName("AIVE")
    application.setOrganizationName("AIVE")

    window = MainWindow(project)
    window.show()
    logger.info("Desktop UI started")
    return int(application.exec())


def main() -> None:
    """Console-script entry point for ``aive-ui``."""
    project = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    try:
        raise SystemExit(run(project))
    except UiDependencyMissingError as exc:
        sys.stderr.write(f"{exc}\n")
        raise SystemExit(5) from None


__all__ = ["UiDependencyMissingError", "main", "run"]
