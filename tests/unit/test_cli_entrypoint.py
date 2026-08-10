"""Tests for the ``main()`` wrapper.

``main()`` exists for one reason: an unexpected exception must still leave a
structured error on stdout. Without it, an internal failure would print a traceback
to stderr and *nothing* to stdout, and the AI director would see an empty
successful-looking read followed by a non-zero exit - the most confusing possible
outcome, and the hardest to recover from automatically.
"""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

from app.cli import main as main_module
from app.cli.output import ExitCode


class TestTopLevelErrorBoundary:
    def test_an_unexpected_exception_becomes_a_structured_error(
        self, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def explode() -> None:
            raise RuntimeError("something went badly wrong")

        monkeypatch.setattr(main_module, "app", explode)

        with pytest.raises(SystemExit) as excinfo:
            main_module.main()
        assert excinfo.value.code == ExitCode.INTERNAL.status

        payload = json.loads(capsys.readouterr().out)
        assert payload["error"]["code"] == "internal.unhandled"
        assert "something went badly wrong" in payload["error"]["message"]
        assert "RuntimeError" in payload["error"]["message"]
        assert "--verbose" in payload["error"]["hint"]

    def test_a_deliberate_system_exit_passes_straight_through(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Typer's own exits must not be rewritten as internal errors."""

        def exit_cleanly() -> None:
            raise SystemExit(ExitCode.NOT_FOUND.status)

        monkeypatch.setattr(main_module, "app", exit_cleanly)
        with pytest.raises(SystemExit) as excinfo:
            main_module.main()
        assert excinfo.value.code == ExitCode.NOT_FOUND.status

    def test_interrupt_uses_the_conventional_status(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def interrupt() -> None:
            raise KeyboardInterrupt

        monkeypatch.setattr(main_module, "app", interrupt)
        with pytest.raises(SystemExit) as excinfo:
            main_module.main()
        assert excinfo.value.code == 130


class TestGlobalFlags:
    def test_quiet_silences_stderr_but_not_stdout(self) -> None:
        """An agent can mute the console without losing the machine-readable result."""
        completed = subprocess.run(
            [sys.executable, "-m", "app.cli.main", "--quiet", "schema", "list"],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        assert completed.returncode == 0
        assert json.loads(completed.stdout)["schemas"]
        assert completed.stderr == ""

    def test_verbose_adds_debug_output_without_touching_stdout(self) -> None:
        completed = subprocess.run(
            [sys.executable, "-m", "app.cli.main", "--verbose", "schema", "list"],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        assert completed.returncode == 0
        assert json.loads(completed.stdout)["schemas"]

    def test_module_execution_works(self) -> None:
        """`python -m app.cli.main` is how the subprocess tests reach the real CLI."""
        completed = subprocess.run(
            [sys.executable, "-m", "app.cli.main", "--version"],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        assert completed.returncode == 0
        from app import __version__

        assert completed.stdout.strip() == __version__
