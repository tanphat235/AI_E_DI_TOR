"""Tests for the stdout/stderr contract primitives.

These functions are small, but they are the mechanism the whole agent integration
rests on, so their edge cases are worth pinning down: what lands on stdout, what
lands on stderr, and which exit code a given class of failure produces.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.cli.output import (
    ExitCode,
    emit_digest,
    emit_error,
    emit_json,
    note,
    write_json_file,
)
from app.models.common import MediaRef, Severity, TimeRange
from app.models.edit_plan import EditPlan


class TestExitCode:
    def test_success_is_zero(self) -> None:
        assert ExitCode.OK.status == 0

    @pytest.mark.parametrize(
        ("code", "status"),
        [
            (ExitCode.USAGE, 2),
            (ExitCode.NOT_FOUND, 3),
            (ExitCode.INVALID_INPUT, 4),
            (ExitCode.ENVIRONMENT, 5),
            (ExitCode.INTERNAL, 70),
        ],
    )
    def test_failure_codes_are_distinct_and_stable(self, code: ExitCode, status: int) -> None:
        """The agent branches on these, so they are API and must not drift."""
        assert code.status == status

    def test_every_member_has_a_status(self) -> None:
        assert all(isinstance(member.status, int) for member in ExitCode)


class TestEmitJson:
    def test_serialises_a_pydantic_model(
        self, capsys: pytest.CaptureFixture[str], minimal_plan: EditPlan
    ) -> None:
        emit_json(minimal_plan)
        captured = capsys.readouterr()
        assert captured.err == ""
        assert json.loads(captured.out)["project_id"] == "test"

    def test_serialises_a_plain_mapping(self, capsys: pytest.CaptureFixture[str]) -> None:
        emit_json({"a": 1, "b": [1, 2]})
        assert json.loads(capsys.readouterr().out) == {"a": 1, "b": [1, 2]}

    def test_output_is_a_single_trailing_newline_terminated_document(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        emit_json({"a": 1})
        out = capsys.readouterr().out
        assert out.endswith("\n")
        assert not out.endswith("\n\n")

    def test_coerces_types_json_cannot_handle(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Paths, enums, sets and models all appear in our payloads."""
        emit_json(
            {
                "path": Path("raw") / "001.mp4",
                "enum": Severity.WARNING,
                "set": {"b", "a"},
                "tuple": (1, 2),
                "model": TimeRange(start=0.0, end=1.0),
            }
        )
        payload = json.loads(capsys.readouterr().out)
        assert payload["path"] == "raw/001.mp4"
        assert payload["enum"] == "warning"
        assert sorted(payload["set"]) == ["a", "b"]
        assert payload["tuple"] == [1, 2]
        assert payload["model"] == {"start": 0.0, "end": 1.0}


class TestEmitDigest:
    def test_writes_one_line_per_record(self, capsys: pytest.CaptureFixture[str]) -> None:
        emit_digest(["001.mp4 #0 0.00-5.00", "001.mp4 #1 5.00-9.00"])
        out = capsys.readouterr().out
        assert out.splitlines() == ["001.mp4 #0 0.00-5.00", "001.mp4 #1 5.00-9.00"]

    def test_an_empty_digest_writes_nothing_at_all(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Not even a blank line: the agent may be counting lines."""
        emit_digest([])
        assert capsys.readouterr().out == ""

    def test_accepts_a_generator(self, capsys: pytest.CaptureFixture[str]) -> None:
        emit_digest(f"line {index}" for index in range(3))
        assert len(capsys.readouterr().out.splitlines()) == 3


class TestEmitError:
    def test_writes_a_structured_error_and_exits(self, capsys: pytest.CaptureFixture[str]) -> None:
        with pytest.raises(SystemExit) as excinfo:
            emit_error("thing.broke", "it broke", hint="unbreak it")
        assert excinfo.value.code == ExitCode.INVALID_INPUT.status

        payload = json.loads(capsys.readouterr().out)
        assert payload["error"] == {
            "code": "thing.broke",
            "message": "it broke",
            "hint": "unbreak it",
        }

    def test_the_hint_is_omitted_when_absent(self, capsys: pytest.CaptureFixture[str]) -> None:
        with pytest.raises(SystemExit):
            emit_error("thing.broke", "it broke")
        assert "hint" not in json.loads(capsys.readouterr().out)["error"]

    def test_details_are_coerced(self, capsys: pytest.CaptureFixture[str]) -> None:
        with pytest.raises(SystemExit):
            emit_error(
                "source.missing",
                "a source file is absent",
                details={"path": Path("raw/001.mp4"), "clips": 3},
                exit_code=ExitCode.NOT_FOUND,
            )
        details = json.loads(capsys.readouterr().out)["error"]["details"]
        assert details["path"] == "raw/001.mp4"
        assert details["clips"] == 3

    def test_the_error_goes_to_stdout_not_stderr(self, capsys: pytest.CaptureFixture[str]) -> None:
        """One place for the agent to look, whether the command worked or not."""
        with pytest.raises(SystemExit):
            emit_error("thing.broke", "it broke")
        captured = capsys.readouterr()
        assert captured.out
        assert captured.err == ""


class TestNote:
    def test_writes_to_stderr(self, capsys: pytest.CaptureFixture[str]) -> None:
        note("a message for a human")
        captured = capsys.readouterr()
        assert captured.out == ""
        assert "a message for a human" in captured.err

    def test_square_brackets_survive(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Rich would otherwise treat them as markup and delete them."""
        note('run: pip install -e ".[speech]"')
        assert '".[speech]"' in capsys.readouterr().err

    def test_quiet_suppresses_it(
        self, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("AIVE_QUIET", "1")
        note("should not appear")
        assert capsys.readouterr().err == ""

    @pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes"])
    def test_quiet_accepts_the_usual_truthy_spellings(
        self, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch, value: str
    ) -> None:
        monkeypatch.setenv("AIVE_QUIET", value)
        note("hidden")
        assert capsys.readouterr().err == ""

    def test_quiet_off_still_prints(
        self, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("AIVE_QUIET", "0")
        note("visible")
        assert "visible" in capsys.readouterr().err


class TestWriteJsonFile:
    def test_writes_a_model_and_creates_parents(
        self, tmp_path: Path, minimal_plan: EditPlan
    ) -> None:
        destination = tmp_path / "deep" / "nested" / "plan.json"
        written = write_json_file(minimal_plan, destination)
        assert written == destination
        assert EditPlan.model_validate_json(destination.read_text(encoding="utf-8")) == minimal_plan

    def test_writes_a_plain_dict(self, tmp_path: Path) -> None:
        destination = write_json_file({"a": 1}, tmp_path / "x.json")
        assert json.loads(destination.read_text(encoding="utf-8")) == {"a": 1}

    def test_writes_nothing_to_either_stream(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Full data goes to disk; only the digest goes to stdout."""
        write_json_file({"a": 1}, tmp_path / "x.json")
        captured = capsys.readouterr()
        assert captured.out == ""
        assert captured.err == ""

    def test_uses_utf8_regardless_of_the_system_codepage(self, tmp_path: Path) -> None:
        """Windows defaults to cp1252, which would mangle non-Latin narration."""
        destination = write_json_file({"text": "Trồng cây xanh"}, tmp_path / "vi.json")
        assert json.loads(destination.read_text(encoding="utf-8"))["text"] == "Trồng cây xanh"

    def test_paths_are_written_posix_style(self, tmp_path: Path) -> None:
        destination = write_json_file({"ref": MediaRef(path="raw/001.mp4")}, tmp_path / "r.json")
        assert "raw/001.mp4" in destination.read_text(encoding="utf-8")
