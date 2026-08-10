"""Tests for ``aive rules``.

The engine is real here - it is pure computation, so there is nothing to fake. What is
under test is the command surface: which exit code a director gets, whether the digest
carries the hints it needs to self-correct, and whether normalising writes a file only
when asked.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from typer.testing import CliRunner

from app.cli.main import app
from app.models.common import (
    CameraMove,
    MediaRef,
    MotionLevel,
    ShotType,
    TimeRange,
    TransitionKind,
)
from app.models.edit_plan import EditPlan, TimelineClip, Transition
from app.models.media import MediaProbe, VideoStreamInfo
from app.models.video import (
    ClipAnalysis,
    DuplicateGroup,
    FootageAnalysis,
    MotionStats,
    QualityScores,
    Scene,
    SceneTags,
)
from app.services.paths import ProjectPaths

runner = CliRunner()

CLIP_A = MediaRef(path="raw/001.mp4")
CLIP_B = MediaRef(path="raw/002.mp4")


def _scene(clip: MediaRef, index: int, *, overall: float = 0.85, duration: float = 4.0) -> Scene:
    return Scene(
        clip=clip,
        index=index,
        range=TimeRange(start=0.0, end=duration),
        quality=QualityScores(
            blur=0.9, brightness=0.6, exposure=0.9, stability=0.9, overall=overall
        ),
        motion=MotionStats(
            level=MotionLevel.LOW, mean_magnitude=1.0, camera_move=CameraMove.STATIC
        ),
        tags=SceneTags(provider="test"),
        shot_type=ShotType.MEDIUM,
    )


def _footage(*, duplicates: tuple = (), low_quality: bool = False) -> FootageAnalysis:
    return FootageAnalysis(
        clips=tuple(
            ClipAnalysis(
                clip=clip,
                probe=MediaProbe(
                    source=clip,
                    duration=30.0,
                    size_bytes=1024,
                    video=VideoStreamInfo(width=1920, height=1080, fps=30.0, codec="h264"),
                ),
                scenes=(_scene(clip, 0, overall=0.2 if low_quality else 0.85),),
                analyzer_version="test/1",
                analyzed_at=datetime.now(UTC),
            )
            for clip in (CLIP_A, CLIP_B)
        ),
        duplicates=duplicates,
    )


@pytest.fixture
def project(paths: ProjectPaths) -> ProjectPaths:
    """A project with real media files and a footage analysis on disk."""
    for name in ("001.mp4", "002.mp4"):
        (paths.raw / name).write_bytes(b"\0" * 1024)
    paths.footage_analysis_file.parent.mkdir(parents=True, exist_ok=True)
    paths.footage_analysis_file.write_text(_footage().model_dump_json(), encoding="utf-8")
    return paths


def _write_plan(paths: ProjectPaths, plan: EditPlan, name: str = "edit_plan.json") -> Path:
    destination = paths.root / name
    destination.write_text(plan.model_dump_json(), encoding="utf-8")
    return destination


def _good_plan() -> EditPlan:
    return EditPlan(
        project_id="test",
        created_by="pytest",
        clips=(
            TimelineClip(
                id="c1",
                source=CLIP_A,
                source_range=TimeRange(start=0.0, end=4.0),
                reason="first",
            ),
            TimelineClip(
                id="c2",
                source=CLIP_B,
                source_range=TimeRange(start=0.0, end=4.0),
                reason="second",
            ),
        ),
    )


class TestValidate:
    def test_a_good_plan_exits_zero(self, project: ProjectPaths) -> None:
        plan_path = _write_plan(project, _good_plan())
        result = runner.invoke(app, ["rules", "validate", str(plan_path)])
        assert result.exit_code == 0, result.stdout
        assert "ok=true" in result.stdout

    def test_errors_exit_four_so_a_script_can_branch(self, project: ProjectPaths) -> None:
        plan = _good_plan()
        broken = plan.model_copy(
            update={
                "clips": (
                    plan.clips[0].model_copy(
                        update={"source_range": TimeRange(start=45.0, end=52.0)}
                    ),
                    plan.clips[1],
                )
            }
        )
        plan_path = _write_plan(project, broken)
        result = runner.invoke(app, ["rules", "validate", str(plan_path)])
        assert result.exit_code == 4
        assert "source.out_of_bounds" in result.stdout

    def test_the_digest_carries_the_hint(self, project: ProjectPaths) -> None:
        """The hint is what lets the director self-correct instead of guessing."""
        plan = _good_plan()
        broken = plan.model_copy(
            update={
                "clips": (
                    plan.clips[0].model_copy(
                        update={"source_range": TimeRange(start=0.0, end=90.0)}
                    ),
                    plan.clips[1],
                )
            }
        )
        result = runner.invoke(app, ["rules", "validate", str(_write_plan(project, broken))])
        assert "| " in result.stdout
        assert "normalize" in result.stdout

    def test_severity_is_marked_per_line(self, project: ProjectPaths) -> None:
        plan_path = _write_plan(project, _good_plan())
        result = runner.invoke(app, ["rules", "validate", str(plan_path)])
        # An unplaced plan reports the continuity check as skipped, at INFO.
        assert any(line.startswith("i ") for line in result.stdout.splitlines())

    def test_validate_does_not_modify_the_plan(self, project: ProjectPaths) -> None:
        plan_path = _write_plan(project, _good_plan())
        before = plan_path.read_text(encoding="utf-8")
        runner.invoke(app, ["rules", "validate", str(plan_path)])
        assert plan_path.read_text(encoding="utf-8") == before

    def test_full_emits_the_report_as_json(self, project: ProjectPaths) -> None:
        plan_path = _write_plan(project, _good_plan())
        result = runner.invoke(app, ["rules", "validate", str(plan_path), "--full"])
        payload = json.loads(result.stdout)
        assert payload["plan_project_id"] == "test"

    def test_logs_stay_on_stderr(self, project: ProjectPaths) -> None:
        plan_path = _write_plan(project, _good_plan())
        result = runner.invoke(app, ["rules", "validate", str(plan_path)])
        assert "verdict:" in result.stderr
        assert "verdict:" not in result.stdout

    def test_duplicate_usage_is_reported(self, project: ProjectPaths) -> None:
        project.footage_analysis_file.write_text(
            _footage(
                duplicates=(
                    DuplicateGroup(representative="001#0", duplicates=("002#0",), similarity=0.98),
                )
            ).model_dump_json(),
            encoding="utf-8",
        )
        plan = _good_plan()
        using_duplicate = plan.model_copy(
            update={
                "clips": (
                    plan.clips[0],
                    plan.clips[1].model_copy(update={"scene_key": "002#0"}),
                )
            }
        )
        result = runner.invoke(
            app, ["rules", "validate", str(_write_plan(project, using_duplicate))]
        )
        assert "scene.duplicate_used" in result.stdout


class TestNormalize:
    def test_clips_are_placed(self, project: ProjectPaths) -> None:
        plan_path = _write_plan(project, _good_plan())
        result = runner.invoke(app, ["rules", "normalize", str(plan_path)])
        assert result.exit_code == 0, result.stdout
        assert "normalize.placed_clip" in result.stdout

    def test_nothing_is_written_unless_asked(self, project: ProjectPaths) -> None:
        """Overwriting silently would destroy the director's own record of its decisions."""
        plan_path = _write_plan(project, _good_plan())
        before = plan_path.read_text(encoding="utf-8")
        runner.invoke(app, ["rules", "normalize", str(plan_path)])
        assert plan_path.read_text(encoding="utf-8") == before

    def test_output_writes_a_valid_plan(self, project: ProjectPaths) -> None:
        plan_path = _write_plan(project, _good_plan())
        destination = project.root / "normalized.json"
        result = runner.invoke(app, ["rules", "normalize", str(plan_path), "-o", str(destination)])
        assert result.exit_code == 0
        written = EditPlan.model_validate_json(destination.read_text(encoding="utf-8"))
        assert written.is_placed

    def test_in_place_overwrites_the_input(self, project: ProjectPaths) -> None:
        plan_path = _write_plan(project, _good_plan())
        result = runner.invoke(app, ["rules", "normalize", str(plan_path), "--in-place"])
        assert result.exit_code == 0
        assert EditPlan.model_validate_json(plan_path.read_text(encoding="utf-8")).is_placed

    def test_an_out_of_bounds_range_is_clamped(self, project: ProjectPaths) -> None:
        plan = _good_plan()
        broken = plan.model_copy(
            update={
                "clips": (
                    plan.clips[0].model_copy(
                        update={"source_range": TimeRange(start=0.0, end=90.0)}
                    ),
                    plan.clips[1],
                )
            }
        )
        destination = project.root / "fixed.json"
        result = runner.invoke(
            app,
            ["rules", "normalize", str(_write_plan(project, broken)), "-o", str(destination)],
        )
        assert result.exit_code == 0
        written = EditPlan.model_validate_json(destination.read_text(encoding="utf-8"))
        assert written.clips[0].source_range.end == 30.0

    def test_an_oversized_transition_is_clamped(self, project: ProjectPaths) -> None:
        plan = _good_plan()
        broken = plan.model_copy(
            update={
                "clips": (
                    plan.clips[0],
                    plan.clips[1].model_copy(
                        update={
                            "transition_in": Transition(kind=TransitionKind.DISSOLVE, duration=3.0)
                        }
                    ),
                )
            }
        )
        destination = project.root / "fixed.json"
        runner.invoke(
            app,
            ["rules", "normalize", str(_write_plan(project, broken)), "-o", str(destination)],
        )
        written = EditPlan.model_validate_json(destination.read_text(encoding="utf-8"))
        transition = written.clips[1].transition_in
        assert transition is not None
        assert transition.duration == pytest.approx(1.0)

    def test_an_unfixable_error_still_exits_four(self, project: ProjectPaths) -> None:
        """Normalisation never invents editorial intent."""
        plan = _good_plan()
        broken = plan.model_copy(
            update={
                "clips": (
                    plan.clips[0].model_copy(
                        update={"source_range": TimeRange(start=0.0, end=0.3)}
                    ),
                    plan.clips[1],
                )
            }
        )
        result = runner.invoke(app, ["rules", "normalize", str(_write_plan(project, broken))])
        assert result.exit_code == 4
        assert "clip.too_short" in result.stdout

    def test_full_emits_the_normalised_plan(self, project: ProjectPaths) -> None:
        plan_path = _write_plan(project, _good_plan())
        result = runner.invoke(app, ["rules", "normalize", str(plan_path), "--full"])
        written = EditPlan.model_validate(json.loads(result.stdout))
        assert written.is_placed


class TestScenes:
    def test_eligible_and_rejected_are_listed(self, project: ProjectPaths) -> None:
        result = runner.invoke(app, ["rules", "scenes", str(project.root)])
        assert result.exit_code == 0, result.stdout
        assert result.stdout.startswith("# scenes=")
        assert any(line.startswith("+ ") for line in result.stdout.splitlines())

    def test_a_rejected_scene_names_its_reasons(self, project: ProjectPaths) -> None:
        project.footage_analysis_file.write_text(
            _footage(low_quality=True).model_dump_json(), encoding="utf-8"
        )
        result = runner.invoke(app, ["rules", "scenes", str(project.root)])
        assert any(line.startswith("- ") for line in result.stdout.splitlines())
        assert "quality.below_floor" in result.stdout

    def test_no_analysis_yet_points_at_the_fix(self, paths: ProjectPaths) -> None:
        result = runner.invoke(app, ["rules", "scenes", str(paths.root)])
        assert result.exit_code == 3
        error = json.loads(result.stdout)["error"]
        assert error["code"] == "footage.not_analysed"
        assert "aive analyze video" in error["hint"]

    def test_full_emits_json(self, project: ProjectPaths) -> None:
        result = runner.invoke(app, ["rules", "scenes", str(project.root), "--full"])
        payload = json.loads(result.stdout)
        assert "eligible" in payload
        assert "rejected" in payload


class TestErrorPaths:
    def test_a_missing_plan(self, tmp_path: Path) -> None:
        result = runner.invoke(app, ["rules", "validate", str(tmp_path / "absent.json")])
        assert result.exit_code == 3
        assert json.loads(result.stdout)["error"]["code"] == "plan.missing"

    def test_an_invalid_plan_points_at_the_schema(self, tmp_path: Path) -> None:
        broken = tmp_path / "broken.json"
        broken.write_text('{"project_id": "x"}', encoding="utf-8")
        result = runner.invoke(app, ["rules", "validate", str(broken)])
        assert result.exit_code == 4
        error = json.loads(result.stdout)["error"]
        assert error["code"] == "plan.invalid"
        assert "schema show edit-plan" in error["hint"]

    def test_a_plan_with_an_unknown_key_is_rejected(self, tmp_path: Path) -> None:
        """extra=forbid: a typo must fail loudly rather than change the edit."""
        broken = tmp_path / "typo.json"
        broken.write_text(
            json.dumps(
                {
                    "project_id": "x",
                    "created_by": "test",
                    "clips": [
                        {
                            "id": "c1",
                            "source": {"path": "raw/001.mp4"},
                            "source_range": {"start": 0.0, "end": 4.0},
                            "reason": "test",
                            "transiton_in": None,
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        result = runner.invoke(app, ["rules", "validate", str(broken)])
        assert result.exit_code == 4

    def test_validating_without_analysis_still_works(self, paths: ProjectPaths) -> None:
        """Refusing outright would push users toward skipping validation entirely."""
        plan_path = _write_plan(paths, _good_plan())
        (paths.raw / "001.mp4").write_bytes(b"\0")
        (paths.raw / "002.mp4").write_bytes(b"\0")
        result = runner.invoke(app, ["rules", "validate", str(plan_path)])
        # Source bounds cannot be checked, and the report says so rather than staying silent.
        assert "source.unverified" in result.stdout

    def test_a_project_override_is_honoured(self, project: ProjectPaths, tmp_path: Path) -> None:
        """A plan need not live inside the project it describes."""
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        plan_path = elsewhere / "plan.json"
        plan_path.write_text(_good_plan().model_dump_json(), encoding="utf-8")

        result = runner.invoke(
            app, ["rules", "validate", str(plan_path), "--project", str(project.root)]
        )
        # With the real project supplied, bounds are checkable and the files are found.
        assert "source.unverified" not in result.stdout
        assert "source.missing" not in result.stdout


class TestDiscoverability:
    def test_rules_is_advertised(self) -> None:
        result = runner.invoke(app, ["--help"])
        assert "rules" in result.output

    def test_all_three_subcommands_are_listed(self) -> None:
        result = runner.invoke(app, ["rules", "--help"])
        for name in ("validate", "normalize", "scenes"):
            assert name in result.output

    def test_every_registered_group_actually_does_something(self) -> None:
        """Was "export is still not registered" until Phase 8 implemented it.

        The rule it guarded - no command that exists but does nothing - is now carried by
        the exact-registry assertion below, which fails on an unexpected addition as well
        as on a removal.
        """
        registered = {group.name for group in app.registered_groups}
        assert "export" in registered

    def test_the_registry_lists_exactly_the_implemented_commands(self) -> None:
        """Exact, in both directions.

        A missing entry means a command was dropped; an extra one means a phase registered
        something before it worked. The second is the failure this project cares about, and
        it is why the assertion is equality rather than a subset check.
        """
        registered = {group.name for group in app.registered_groups} | {
            command.name for command in app.registered_commands
        }
        assert registered == {
            "project",
            "analyze",
            "plan",
            "rules",
            "subtitle",
            "schema",
            "config",
            "render",
            "export",
            "narrate",
            "ui",
            "doctor",
        }
