"""Tests for ``aive plan``.

Real planner, real Rule Engine — both are pure computation, so there is nothing to fake.
What is under test is the command surface: whether the brief a director reads actually
contains what it needs, whether infeasibility is signalled rather than buried, and whether
the baseline draft admits what it is.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from typer.testing import CliRunner

from app.cli.main import app
from app.models.common import CameraMove, MediaRef, MotionLevel, ShotType, TimeRange
from app.models.edit_plan import EditPlan
from app.models.media import MediaProbe, VideoStreamInfo
from app.models.planning import PlanningBrief
from app.models.speech import (
    NarrationAnalysis,
    NarrationBeat,
    SpeechCleanupReport,
    Transcript,
    TranscriptSegment,
)
from app.models.video import (
    ClipAnalysis,
    FootageAnalysis,
    MotionStats,
    QualityScores,
    Scene,
    SceneTags,
)
from app.services.paths import ProjectPaths

runner = CliRunner()

NARRATION = MediaRef(path="narration.wav")
CLIP_A = MediaRef(path="raw/001.mp4")
CLIP_B = MediaRef(path="raw/002.mp4")

# The synthetic scenes below are deliberately unremarkable, so the project config relaxes
# the quality floors that real footage would clear on its own.
PERMISSIVE_CONFIG = """\
[rules]
min_blur_score = 0.0
min_stability_score = 0.0
min_overall_quality = 0.20
min_clip_duration = 0.5
"""


def _scene(
    clip: MediaRef,
    index: int,
    *,
    start: float = 0.0,
    duration: float = 6.0,
    quality: float = 0.85,
    shot: ShotType = ShotType.MEDIUM,
) -> Scene:
    return Scene(
        clip=clip,
        index=index,
        range=TimeRange(start=start, end=start + duration),
        quality=QualityScores(
            blur=0.9, brightness=0.6, exposure=0.9, stability=0.9, overall=quality
        ),
        motion=MotionStats(
            level=MotionLevel.LOW, mean_magnitude=1.0, camera_move=CameraMove.STATIC
        ),
        tags=SceneTags(provider="test"),
        shot_type=shot,
    )


def _write_analyses(
    paths: ProjectPaths,
    *,
    beats: tuple[NarrationBeat, ...],
    scenes: tuple[Scene, ...],
    kept: float = 6.0,
) -> None:
    """Put both analysis documents on disk, as the analyse commands would."""
    narration = NarrationAnalysis(
        source=NARRATION,
        transcript=Transcript(
            source=NARRATION,
            language="en",
            duration=kept + 2.0,
            model_name="test/fake",
            segments=(
                TranscriptSegment(
                    index=0, range=TimeRange(start=0.0, end=kept), text="Prepare the soil."
                ),
            ),
        ),
        cleanup=SpeechCleanupReport(
            source=NARRATION,
            original_duration=kept + 2.0,
            kept_ranges=(TimeRange(start=0.0, end=kept),),
        ),
        beats=beats,
        analyzer_version="test/1",
        analyzed_at=datetime.now(UTC),
    )

    by_clip: dict[MediaRef, list[Scene]] = {}
    for scene in scenes:
        by_clip.setdefault(scene.clip, []).append(scene)
    footage = FootageAnalysis(
        clips=tuple(
            ClipAnalysis(
                clip=clip,
                probe=MediaProbe(
                    source=clip,
                    duration=60.0,
                    size_bytes=1024,
                    video=VideoStreamInfo(width=1920, height=1080, fps=30.0, codec="h264"),
                ),
                scenes=tuple(clip_scenes),
                analyzer_version="test/1",
                analyzed_at=datetime.now(UTC),
            )
            for clip, clip_scenes in by_clip.items()
        )
    )

    paths.narration_file.parent.mkdir(parents=True, exist_ok=True)
    paths.narration_file.write_text(narration.model_dump_json(), encoding="utf-8")
    paths.footage_analysis_file.write_text(footage.model_dump_json(), encoding="utf-8")


def _beat(
    index: int, *, start: float, duration: float, timeline_start: float | None
) -> NarrationBeat:
    timeline = (
        None
        if timeline_start is None
        else TimeRange(start=timeline_start, end=timeline_start + duration)
    )
    return NarrationBeat(
        index=index,
        range=TimeRange(start=start, end=start + duration),
        text=f"Beat number {index} of the narration.",
        keywords=("soil", "water"),
        timeline_range=timeline,
    )


@pytest.fixture
def planned(paths: ProjectPaths) -> ProjectPaths:
    """A project with media, both analyses, and relaxed quality floors."""
    for name in ("001.mp4", "002.mp4"):
        (paths.raw / name).write_bytes(b"\0" * 1024)
    (paths.root / "narration.wav").write_bytes(b"RIFF" + b"\0" * 64)
    paths.config_file.write_text(PERMISSIVE_CONFIG, encoding="utf-8")
    _write_analyses(
        paths,
        beats=(
            _beat(0, start=0.0, duration=2.0, timeline_start=0.0),
            _beat(1, start=3.0, duration=1.0, timeline_start=None),  # a cut retake
            _beat(2, start=5.0, duration=3.0, timeline_start=2.5),
        ),
        scenes=(
            _scene(CLIP_A, 0, shot=ShotType.WIDE),
            _scene(CLIP_A, 1, start=10.0, shot=ShotType.CLOSE_UP),
            _scene(CLIP_B, 0, start=0.0, shot=ShotType.MEDIUM),
        ),
    )
    return paths


class TestBrief:
    def test_the_brief_is_written_and_digested(self, planned: ProjectPaths) -> None:
        result = runner.invoke(app, ["plan", "brief", str(planned.root)])
        assert result.exit_code == 0, result.stdout
        assert planned.planning_brief_file.is_file()
        assert result.stdout.startswith("# brief=")
        restored = PlanningBrief.model_validate_json(
            planned.planning_brief_file.read_text(encoding="utf-8")
        )
        assert restored.coverage.feasible

    def test_the_digest_carries_the_constraints_beside_the_choices(
        self, planned: ProjectPaths
    ) -> None:
        """A plan that breaks a threshold it was never shown is the tool's failure."""
        result = runner.invoke(app, ["plan", "brief", str(planned.root)])
        constraints = next(
            line for line in result.stdout.splitlines() if line.startswith("# constraints")
        )
        for field in ("clip=", "transition=", "max_transition_ratio=", "output=", "quality_floor="):
            assert field in constraints

    def test_each_beat_shows_its_timeline_position_and_candidates(
        self, planned: ProjectPaths
    ) -> None:
        result = runner.invoke(app, ["plan", "brief", str(planned.root)])
        assert "B000 tl=0.00-2.00" in result.stdout
        assert "cands=" in result.stdout

    def test_a_cut_beat_is_marked_and_offered_nothing(self, planned: ProjectPaths) -> None:
        result = runner.invoke(app, ["plan", "brief", str(planned.root)])
        cut_line = next(line for line in result.stdout.splitlines() if line.startswith("B001"))
        assert "tl=CUT" in cut_line
        assert "cands=0" in cut_line

    def test_candidates_are_indented_under_their_beat_with_reasons(
        self, planned: ProjectPaths
    ) -> None:
        result = runner.invoke(app, ["plan", "brief", str(planned.root)])
        candidate_lines = [line for line in result.stdout.splitlines() if line.startswith("    ")]
        assert candidate_lines
        assert any("score=" in line and "|" in line for line in candidate_lines)

    def test_the_absence_of_semantic_tags_is_stated(self, planned: ProjectPaths) -> None:
        """The director must not read a low score as "wrong shot"."""
        result = runner.invoke(app, ["plan", "brief", str(planned.root)])
        assert "no semantic tags" in result.stdout

    def test_the_shortlist_size_can_be_overridden(self, planned: ProjectPaths) -> None:
        result = runner.invoke(app, ["plan", "brief", str(planned.root), "--candidates", "1"])
        assert result.exit_code == 0
        for line in result.stdout.splitlines():
            if line.startswith("B0") and "tl=CUT" not in line:
                assert "cands=1" in line

    def test_full_emits_the_brief_as_json(self, planned: ProjectPaths) -> None:
        result = runner.invoke(app, ["plan", "brief", str(planned.root), "--full"])
        assert PlanningBrief.model_validate(json.loads(result.stdout)).beats

    def test_logs_stay_on_stderr(self, planned: ProjectPaths) -> None:
        result = runner.invoke(app, ["plan", "brief", str(planned.root)])
        assert "verdict:" in result.stderr
        assert "verdict:" not in result.stdout

    def test_an_infeasible_project_exits_four(self, paths: ProjectPaths) -> None:
        """Better to learn this before authoring forty clips than after."""
        for name in ("001.mp4",):
            (paths.raw / name).write_bytes(b"\0" * 1024)
        paths.config_file.write_text("[rules]\nmin_overall_quality = 0.99\n", encoding="utf-8")
        _write_analyses(
            paths,
            beats=(_beat(0, start=0.0, duration=2.0, timeline_start=0.0),),
            scenes=(_scene(CLIP_A, 0, quality=0.3),),
        )
        result = runner.invoke(app, ["plan", "brief", str(paths.root)])
        assert result.exit_code == 4
        assert "feasible=false" in result.stdout
        assert "coverage.no_usable_footage" in result.stdout

    def test_unused_eligible_scenes_are_listed(self, planned: ProjectPaths) -> None:
        """Often the most interesting B-roll, so it is reported rather than dropped."""
        result = runner.invoke(app, ["plan", "brief", str(planned.root), "--candidates", "1"])
        assert "# unused_eligible:" in result.stdout


class TestDraft:
    def test_a_plan_is_written_to_edit_plan_json(self, planned: ProjectPaths) -> None:
        result = runner.invoke(app, ["plan", "draft", str(planned.root)])
        assert result.exit_code == 0, result.stdout
        assert planned.edit_plan_file.is_file()
        plan = EditPlan.model_validate_json(planned.edit_plan_file.read_text(encoding="utf-8"))
        assert plan.clips

    def test_the_provenance_says_heuristic(self, planned: ProjectPaths) -> None:
        runner.invoke(app, ["plan", "draft", str(planned.root)])
        plan = EditPlan.model_validate_json(planned.edit_plan_file.read_text(encoding="utf-8"))
        assert plan.created_by == "heuristic"

    def test_the_digest_and_stderr_both_say_it_is_not_an_edit(self, planned: ProjectPaths) -> None:
        result = runner.invoke(app, ["plan", "draft", str(planned.root)])
        assert "NOT an edit" in result.stdout
        assert "NOT an edit" in result.stderr

    def test_one_clip_per_surviving_beat(self, planned: ProjectPaths) -> None:
        """Three beats, one cut, so two clips."""
        runner.invoke(app, ["plan", "draft", str(planned.root)])
        plan = EditPlan.model_validate_json(planned.edit_plan_file.read_text(encoding="utf-8"))
        assert len(plan.clips) == 2

    def test_the_output_path_can_be_chosen(self, planned: ProjectPaths) -> None:
        destination = planned.root / "baseline.json"
        result = runner.invoke(app, ["plan", "draft", str(planned.root), "-o", str(destination)])
        assert result.exit_code == 0
        assert destination.is_file()
        assert not planned.edit_plan_file.exists()

    def test_the_draft_normalises_and_validates_cleanly(self, planned: ProjectPaths) -> None:
        """The whole point of a baseline: the director never fights the schema."""
        runner.invoke(app, ["plan", "draft", str(planned.root)])
        normalise = runner.invoke(
            app, ["rules", "normalize", str(planned.edit_plan_file), "--in-place"]
        )
        assert normalise.exit_code == 0, normalise.stdout
        validate = runner.invoke(app, ["rules", "validate", str(planned.edit_plan_file)])
        assert validate.exit_code == 0, validate.stdout

    def test_normalisation_of_the_draft_converges(self, planned: ProjectPaths) -> None:
        """Regression: rounding a clamped transition *up* made the chain never converge."""
        runner.invoke(app, ["plan", "draft", str(planned.root)])
        result = runner.invoke(
            app, ["rules", "normalize", str(planned.edit_plan_file), "--in-place"]
        )
        assert "normalize.not_converged" not in result.stdout

    def test_full_emits_the_plan(self, planned: ProjectPaths) -> None:
        result = runner.invoke(app, ["plan", "draft", str(planned.root), "--full"])
        assert EditPlan.model_validate(json.loads(result.stdout)).clips


class TestErrorPaths:
    def test_an_uninitialised_project(self, tmp_path: Path) -> None:
        blank = tmp_path / "blank"
        blank.mkdir()
        result = runner.invoke(app, ["plan", "brief", str(blank)])
        assert result.exit_code == 3
        assert json.loads(result.stdout)["error"]["code"] == "project.not_initialised"

    def test_missing_narration_analysis_names_the_command(self, paths: ProjectPaths) -> None:
        result = runner.invoke(app, ["plan", "brief", str(paths.root)])
        assert result.exit_code == 3
        error = json.loads(result.stdout)["error"]
        assert error["code"] == "narration.not_analysed"
        assert "aive analyze audio" in error["hint"]

    def test_missing_footage_analysis_names_the_command(self, paths: ProjectPaths) -> None:
        _write_analyses(
            paths,
            beats=(_beat(0, start=0.0, duration=2.0, timeline_start=0.0),),
            scenes=(_scene(CLIP_A, 0),),
        )
        paths.footage_analysis_file.unlink()
        result = runner.invoke(app, ["plan", "brief", str(paths.root)])
        assert result.exit_code == 3
        error = json.loads(result.stdout)["error"]
        assert error["code"] == "footage.not_analysed"
        assert "aive analyze video" in error["hint"]

    def test_a_corrupt_analysis_document(self, paths: ProjectPaths) -> None:
        paths.narration_file.parent.mkdir(parents=True, exist_ok=True)
        paths.narration_file.write_text("{not json", encoding="utf-8")
        paths.footage_analysis_file.write_text("{}", encoding="utf-8")
        result = runner.invoke(app, ["plan", "brief", str(paths.root)])
        assert result.exit_code == 4
        assert json.loads(result.stdout)["error"]["code"] == "analysis.unreadable"

    def test_nothing_draftable_points_at_the_brief(self, paths: ProjectPaths) -> None:
        (paths.raw / "001.mp4").write_bytes(b"\0")
        paths.config_file.write_text("[rules]\nmin_overall_quality = 0.99\n", encoding="utf-8")
        _write_analyses(
            paths,
            beats=(_beat(0, start=0.0, duration=2.0, timeline_start=0.0),),
            scenes=(_scene(CLIP_A, 0, quality=0.3),),
        )
        result = runner.invoke(app, ["plan", "draft", str(paths.root)])
        assert result.exit_code == 4
        error = json.loads(result.stdout)["error"]
        assert error["code"] == "plan.not_draftable"
        assert "aive plan brief" in error["hint"]


class TestDiscoverability:
    def test_plan_is_registered(self) -> None:
        registered = {group.name for group in app.registered_groups}
        assert "plan" in registered

    def test_both_subcommands_are_listed(self) -> None:
        result = runner.invoke(app, ["plan", "--help"])
        assert "brief" in result.output
        assert "draft" in result.output
