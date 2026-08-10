"""Tests for the CLI - specifically, for the agent contract.

The tests that matter most here are the stream-separation ones. If a log line ever
reaches stdout, the AI director's parse of a successful command fails in a way that
is indistinguishable from a genuine error, and no amount of downstream care can
recover from it. So it is asserted, not assumed.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from app.cli.main import app
from app.models.edit_plan import EditPlan
from app.models.project import ProjectManifest
from app.services.paths import ProjectPaths

runner = CliRunner()


def _json_stdout(args: list[str], expect_exit: int = 0) -> dict:
    """Invoke the CLI and parse stdout as JSON, asserting the exit code."""
    result = runner.invoke(app, args)
    assert result.exit_code == expect_exit, f"stdout={result.stdout!r} stderr={result.stderr!r}"
    return json.loads(result.stdout)


class TestStdoutDiscipline:
    """stdout is a machine contract. Nothing human-facing may appear on it."""

    def test_schema_prints_nothing_but_json(self) -> None:
        result = runner.invoke(app, ["schema", "show", "edit-plan"])
        assert result.exit_code == 0
        schema = json.loads(result.stdout)
        assert schema["title"] == "EditPlan"
        assert result.stderr == ""

    def test_human_output_goes_to_stderr_only(self, populated_project: ProjectPaths) -> None:
        result = runner.invoke(app, ["project", "scan", str(populated_project.root)])
        assert result.exit_code == 0
        # The chatty summary is on stderr...
        assert "Scanned" in result.stderr
        # ...and never on stdout.
        assert "Scanned" not in result.stdout

    def test_errors_are_structured_on_stdout(self, tmp_path: Path) -> None:
        """The agent reads stdout for both success and failure."""
        blank = tmp_path / "blank"
        blank.mkdir()
        result = runner.invoke(app, ["project", "scan", str(blank)])
        assert result.exit_code == 3
        payload = json.loads(result.stdout)
        assert payload["error"]["code"] == "project.not_initialised"
        assert "aive project init" in payload["error"]["hint"]

    def test_stream_separation_holds_in_a_real_subprocess(self, tmp_path: Path) -> None:
        """CliRunner emulates the streams; a subprocess is the real thing.

        This is the test that would catch a library deciding to print to stdout
        behind our back.
        """
        completed = subprocess.run(
            [sys.executable, "-m", "app.cli.main", "doctor"],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
            cwd=tmp_path,
        )
        payload = json.loads(completed.stdout)
        assert "checks" in payload
        assert completed.stderr, "the human-readable report must reach stderr"


class TestSchemaCommand:
    @pytest.mark.parametrize(
        "document",
        ["edit-plan", "manifest", "transcript", "footage", "plan-report"],
    )
    def test_every_document_emits_a_usable_schema(self, document: str) -> None:
        schema = _json_stdout(["schema", "show", document])
        assert "properties" in schema
        assert "$defs" in schema or "properties" in schema

    def test_list(self) -> None:
        payload = _json_stdout(["schema", "list"])
        names = {item["name"] for item in payload["schemas"]}
        assert "edit-plan" in names

    def test_the_example_plan_actually_validates(self) -> None:
        """A worked example that does not validate is worse than none at all."""
        payload = _json_stdout(["schema", "example"])
        plan = EditPlan.model_validate(payload)
        assert plan.created_by == "claude-code"
        assert len(plan.clips) >= 2
        # Every clip must carry its rationale, since that is the point of the example.
        assert all(clip.reason for clip in plan.clips)

    def test_an_unknown_document_is_a_usage_error(self) -> None:
        result = runner.invoke(app, ["schema", "show", "nonsense"])
        assert result.exit_code == 2


class TestProjectInit:
    def test_creates_the_full_layout(self, tmp_path: Path) -> None:
        target = tmp_path / "myproject"
        payload = _json_stdout(["project", "init", str(target)])
        assert payload["config_written"] is True

        paths = ProjectPaths.for_root(target)
        for directory in paths.all_directories():
            assert directory.is_dir(), f"{directory} was not created"
        assert paths.config_file.is_file()
        assert (paths.root / ".gitignore").is_file()

    def test_is_idempotent(self, tmp_path: Path) -> None:
        """Re-running must repair, not fail or clobber."""
        target = tmp_path / "myproject"
        _json_stdout(["project", "init", str(target)])
        paths = ProjectPaths.for_root(target)
        paths.config_file.write_text("# user edits\n[output]\nfps = 24.0\n", encoding="utf-8")

        second = _json_stdout(["project", "init", str(target)])
        assert second["created_directories"] == []
        assert second["config_written"] is False
        assert "user edits" in paths.config_file.read_text(encoding="utf-8")

    def test_repairs_a_partly_deleted_project(self, tmp_path: Path) -> None:
        target = tmp_path / "myproject"
        _json_stdout(["project", "init", str(target)])
        paths = ProjectPaths.for_root(target)
        paths.raw.rmdir()

        payload = _json_stdout(["project", "init", str(target)])
        assert paths.raw.is_dir()
        assert any("raw" in path for path in payload["created_directories"])


class TestProjectScan:
    def test_an_empty_project_scans_to_a_valid_but_uneditable_manifest(
        self, paths: ProjectPaths
    ) -> None:
        result = runner.invoke(app, ["project", "scan", str(paths.root)])
        assert result.exit_code == 0
        assert "editable=false" in result.stdout
        manifest = ProjectManifest.model_validate_json(
            paths.manifest_file.read_text(encoding="utf-8")
        )
        assert not manifest.is_editable

    def test_finds_media_and_writes_a_manifest(self, populated_project: ProjectPaths) -> None:
        result = runner.invoke(app, ["project", "scan", str(populated_project.root)])
        assert result.exit_code == 0
        assert "editable=true" in result.stdout
        assert "clips=2" in result.stdout

        manifest = ProjectManifest.model_validate_json(
            populated_project.manifest_file.read_text(encoding="utf-8")
        )
        assert manifest.is_editable
        assert manifest.narration is not None
        assert manifest.narration.ref.name == "narration.wav"
        assert len(manifest.raw_clips) == 2
        assert len(manifest.music) == 1

    def test_the_digest_is_much_smaller_than_the_full_manifest(
        self, populated_project: ProjectPaths
    ) -> None:
        """Token frugality is a design requirement, so it gets a test."""
        digest = runner.invoke(app, ["project", "scan", str(populated_project.root)])
        full = runner.invoke(app, ["project", "scan", str(populated_project.root), "--full"])
        assert len(digest.stdout) < len(full.stdout) / 2

    def test_full_output_is_a_parseable_manifest(self, populated_project: ProjectPaths) -> None:
        payload = _json_stdout(["project", "scan", str(populated_project.root), "--full"])
        manifest = ProjectManifest.model_validate(payload)
        assert manifest.project_id

    def test_a_missing_directory_reports_not_found(self, tmp_path: Path) -> None:
        result = runner.invoke(app, ["project", "scan", str(tmp_path / "ghost")])
        assert result.exit_code == 3
        assert json.loads(result.stdout)["error"]["code"] == "project.missing"


class TestConfigCommand:
    def test_show_emits_every_section(self, isolated_cwd: Path) -> None:
        payload = _json_stdout(["config", "show"])
        assert {
            "app",
            "media",
            "speech",
            "vision",
            "rules",
            "subtitle",
            "music",
            "output",
            "capcut",
        } <= set(payload)

    def test_show_can_be_narrowed_to_one_section(self, isolated_cwd: Path) -> None:
        payload = _json_stdout(["config", "show", "--section", "rules"])
        assert set(payload) == {"rules"}
        assert payload["rules"]["min_clip_duration"] == 1.2

    def test_a_project_override_is_visible_in_the_merged_output(self, paths: ProjectPaths) -> None:
        paths.config_file.write_text("[output]\nfps = 24.0\n", encoding="utf-8")
        payload = _json_stdout(["config", "show", str(paths.root)])
        assert payload["output"]["fps"] == 24.0
        # A partial override must not wipe its siblings.
        assert payload["output"]["video_crf"] == 18

    def test_layers_reports_what_would_be_read(self, paths: ProjectPaths) -> None:
        payload = _json_stdout(["config", "layers", str(paths.root)])
        kinds = [layer["kind"] for layer in payload["layers"]]
        assert kinds == ["packaged defaults", "site override", "project override"]
        assert payload["layers"][0]["applied"] is True

    def test_defaults(self, isolated_cwd: Path) -> None:
        payload = _json_stdout(["config", "defaults"])
        assert payload["rules"]["min_clip_duration"] == 1.2


class TestDoctor:
    def test_reports_ok_in_this_environment(self, isolated_cwd: Path) -> None:
        """imageio-ffmpeg is a hard dependency, so doctor must pass on a clean install."""
        payload = _json_stdout(["doctor"])
        assert payload["ok"] is True
        by_name = {check["name"]: check for check in payload["checks"]}
        assert by_name["ffmpeg"]["status"] == "ok"
        assert by_name["python"]["status"] == "ok"

    def test_a_missing_optional_dependency_does_not_fail_the_run(self, isolated_cwd: Path) -> None:
        payload = _json_stdout(["doctor"])
        optional = [check for check in payload["checks"] if not check["required"]]
        assert optional, "the report should list optional components"
        assert payload["ok"] is True

    def test_includes_project_state_when_given_one(self, populated_project: ProjectPaths) -> None:
        payload = _json_stdout(["doctor", str(populated_project.root)])
        project = next(check for check in payload["checks"] if check["name"] == "project")
        assert project["status"] == "ok"
        assert "raw clips: 2" in project["detail"]

    def test_reports_an_uninitialised_project_as_a_failure(self, tmp_path: Path) -> None:
        blank = tmp_path / "blank"
        blank.mkdir()
        result = runner.invoke(app, ["doctor", str(blank)])
        assert result.exit_code == 5
        payload = json.loads(result.stdout)
        assert payload["ok"] is False


class TestRootCommand:
    def test_version_prints_only_the_version(self) -> None:
        from app import __version__

        result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        assert result.stdout.strip() == __version__

    def test_bare_invocation_shows_help(self) -> None:
        result = runner.invoke(app, [])
        assert "doctor" in result.output
        assert "project" in result.output

    def test_help_advertises_the_stdout_contract(self) -> None:
        """The director learns the contract from --help, so it must be stated there."""
        result = runner.invoke(app, ["--help"])
        assert "stdout" in result.output.lower()
