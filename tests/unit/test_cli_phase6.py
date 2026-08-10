"""Tests for ``aive plan show``, ``diff`` and ``subtitles``.

The commands themselves are thin — the logic lives in ``app/plan/`` and ``app/subtitles/``
and is tested directly in ``test_plan_tooling.py``. What is under test here is the part the
director actually depends on: that the digest on stdout carries the facts needed to act,
that a broken plan produces a structured error rather than a traceback, and that the two
commands which can write to disk write what they said they wrote.
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
from app.models.edit_plan import (
    EditPlan,
    NarrationTrack,
    TimelineClip,
    Transition,
)
from app.models.media import MediaProbe, VideoStreamInfo
from app.models.speech import (
    NarrationAnalysis,
    NarrationBeat,
    SpeechCleanupReport,
    Transcript,
    TranscriptSegment,
    Word,
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


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


def _clip(
    clip_id: str,
    *,
    source: MediaRef = CLIP_A,
    start: float = 0.0,
    end: float = 4.0,
    reason: str = "establishes the setting before any detail",
    **kwargs: object,
) -> TimelineClip:
    return TimelineClip(
        id=clip_id,
        source=source,
        source_range=TimeRange(start=start, end=end),
        reason=reason,
        **kwargs,  # type: ignore[arg-type]
    )


def _plan(*clips: TimelineClip, **kwargs: object) -> EditPlan:
    fields: dict[str, object] = {
        "project_id": "phase6",
        "created_by": "claude-code",
        "clips": clips
        or (
            _clip("c1", scene_key="001#0"),
            _clip(
                "c2",
                source=CLIP_B,
                start=1.0,
                end=6.0,
                scene_key="002#0",
                transition_in=Transition(kind=TransitionKind.DISSOLVE, duration=0.5),
            ),
        ),
        "narration": NarrationTrack(
            source=NARRATION,
            kept_ranges=(TimeRange(start=0.0, end=2.0), TimeRange(start=4.0, end=10.0)),
        ),
    }
    fields.update(kwargs)
    return EditPlan(**fields)  # type: ignore[arg-type]


def _scene(clip: MediaRef, index: int, *, shot: ShotType = ShotType.MEDIUM) -> Scene:
    return Scene(
        clip=clip,
        index=index,
        range=TimeRange(start=index * 8.0, end=index * 8.0 + 7.0),
        quality=QualityScores(blur=0.9, brightness=0.6, exposure=0.9, stability=0.9, overall=0.85),
        motion=MotionStats(
            level=MotionLevel.MEDIUM, mean_magnitude=2.0, camera_move=CameraMove.PAN
        ),
        tags=SceneTags(provider="classical_cv"),
        shot_type=shot,
    )


def _footage() -> FootageAnalysis:
    return FootageAnalysis(
        clips=tuple(
            ClipAnalysis(
                clip=clip,
                probe=MediaProbe(
                    source=clip,
                    duration=30.0,
                    size_bytes=2048,
                    video=VideoStreamInfo(width=1920, height=1080, fps=30.0, codec="h264"),
                ),
                scenes=(_scene(clip, 0, shot=shot),),
                analyzer_version="test/1",
                analyzed_at=datetime.now(UTC),
            )
            for clip, shot in ((CLIP_A, ShotType.WIDE), (CLIP_B, ShotType.CLOSE_UP))
        )
    )


def _narration_analysis() -> NarrationAnalysis:
    transcript = Transcript(
        source=NARRATION,
        language="en",
        duration=10.0,
        model_name="test/fake",
        segments=(
            TranscriptSegment(
                index=0,
                range=TimeRange(start=0.0, end=1.9),
                text="First, prepare the soil.",
                words=(
                    Word(text="First,", start=0.0, end=0.4),
                    Word(text="prepare", start=0.5, end=1.0),
                    Word(text="the", start=1.1, end=1.3),
                    Word(text="soil.", start=1.4, end=1.9),
                ),
            ),
            TranscriptSegment(
                index=1,
                range=TimeRange(start=4.0, end=5.6),
                text="Then water it well.",
                words=(
                    Word(text="Then", start=4.0, end=4.3),
                    Word(text="water", start=4.4, end=4.9),
                    Word(text="it", start=5.0, end=5.2),
                    Word(text="well.", start=5.3, end=5.6),
                ),
            ),
        ),
    )
    return NarrationAnalysis(
        source=NARRATION,
        transcript=transcript,
        cleanup=SpeechCleanupReport(
            source=NARRATION,
            original_duration=10.0,
            kept_ranges=(TimeRange(start=0.0, end=2.0), TimeRange(start=4.0, end=10.0)),
        ),
        beats=(
            NarrationBeat(
                index=0,
                range=TimeRange(start=0.0, end=1.9),
                timeline_range=TimeRange(start=0.0, end=1.9),
                text="First, prepare the soil.",
                segment_indices=(0,),
            ),
            NarrationBeat(
                index=1,
                range=TimeRange(start=4.0, end=5.6),
                timeline_range=TimeRange(start=2.0, end=3.6),
                text="Then water it well.",
                segment_indices=(1,),
            ),
        ),
        analyzer_version="test/1",
        analyzed_at=datetime.now(UTC),
    )


@pytest.fixture
def project(tmp_path: Path) -> ProjectPaths:
    """A project directory with a narration and footage analysis already cached."""
    paths = ProjectPaths.for_root(tmp_path / "proj")
    paths.ensure()
    paths.narration_file.write_text(_narration_analysis().model_dump_json(), encoding="utf-8")
    paths.footage_analysis_file.write_text(_footage().model_dump_json(), encoding="utf-8")
    return paths


def _write(paths: ProjectPaths, plan: EditPlan, name: str = "edit_plan.json") -> Path:
    path = paths.root / name
    path.write_text(plan.model_dump_json(indent=2), encoding="utf-8")
    return path


def _digest(output: str) -> list[str]:
    return [line for line in output.splitlines() if line.strip()]


def _shown(path: Path) -> str:
    """Run ``plan show`` and return stdout. Keeps the assertions below readable."""
    outcome = runner.invoke(app, ["plan", "show", str(path)])
    assert outcome.exit_code == 0, outcome.output
    return outcome.stdout


# --------------------------------------------------------------------------- #
# plan show
# --------------------------------------------------------------------------- #


class TestPlanShow:
    def test_the_header_carries_the_numbers_a_director_acts_on(self, project: ProjectPaths) -> None:
        path = _write(project, _plan())
        result = runner.invoke(app, ["plan", "show", str(path)])
        assert result.exit_code == 0, result.output

        header = _digest(result.stdout)[0]
        assert "project=phase6" in header
        assert "by=claude-code" in header
        assert "clips=2" in header
        assert "cuts_per_min=" in header

    def test_every_clip_reason_is_printed(self, project: ProjectPaths) -> None:
        """The reason is the whole point of the command: it is what a human judges."""
        path = _write(project, _plan(_clip("only", reason="the one deliberate choice here")))
        result = runner.invoke(app, ["plan", "show", str(path)])
        assert "the one deliberate choice here" in result.stdout

    def test_shot_types_come_from_the_cached_footage_analysis(self, project: ProjectPaths) -> None:
        path = _write(project, _plan())
        result = runner.invoke(app, ["plan", "show", str(path)])
        assert "shot=wide" in result.stdout
        assert "shot=close_up" in result.stdout

    def test_without_a_footage_analysis_the_review_says_less_rather_than_guessing(
        self, tmp_path: Path
    ) -> None:
        paths = ProjectPaths.for_root(tmp_path / "bare")
        paths.ensure()
        path = _write(paths, _plan())
        result = runner.invoke(app, ["plan", "show", str(path)])
        assert result.exit_code == 0, result.output
        assert "shot=unknown" in result.stdout
        assert "No footage analysis found" in result.stderr

    def test_an_unplaced_plan_is_reported_as_unplaced(self, project: ProjectPaths) -> None:
        path = _write(project, _plan())
        result = runner.invoke(app, ["plan", "show", str(path)])
        assert "placed=false" in result.stdout
        assert "tl=unplaced" in result.stdout

    def test_a_placed_plan_shows_timeline_positions(self, project: ProjectPaths) -> None:
        path = _write(project, _plan(_clip("c1", end=4.0, timeline_start=0.0)))
        result = runner.invoke(app, ["plan", "show", str(path)])
        assert "placed=true" in result.stdout
        assert "tl=0.00-4.00" in result.stdout

    def test_coverage_is_reported_against_the_narration(self, project: ProjectPaths) -> None:
        path = _write(project, _plan())
        line = next(line for line in _digest(_shown(path)) if line.startswith("# coverage"))
        assert "narration=8.00" in line

    def test_editorial_notes_appear_with_their_code(self, project: ProjectPaths) -> None:
        path = _write(project, _plan(_clip("c1", reason="ok")))
        result = runner.invoke(app, ["plan", "show", str(path)])
        assert "review.thin_reasons" in result.stdout

    def test_notes_are_marked_by_severity_not_left_ambiguous(self, project: ProjectPaths) -> None:
        path = _write(project, _plan(_clip("c1"), created_by="heuristic"))
        line = next(line for line in _digest(_shown(path)) if "review.heuristic_plan" in line)
        assert line.startswith("W ")

    def test_full_emits_the_review_as_json(self, project: ProjectPaths) -> None:
        path = _write(project, _plan())
        result = runner.invoke(app, ["plan", "show", str(path), "--full"])
        payload = json.loads(result.stdout)
        assert payload["project_id"] == "phase6"
        assert len(payload["timeline"]) == 2
        assert payload["timeline"][0]["reason"]

    def test_a_missing_plan_exits_not_found(self, tmp_path: Path) -> None:
        result = runner.invoke(app, ["plan", "show", str(tmp_path / "absent.json")])
        assert result.exit_code == 3
        assert json.loads(result.stdout)["error"]["code"] == "plan.invalid"

    def test_a_malformed_plan_exits_invalid_input_with_a_hint(self, tmp_path: Path) -> None:
        path = tmp_path / "broken.json"
        path.write_text("{not json", encoding="utf-8")
        result = runner.invoke(app, ["plan", "show", str(path)])
        assert result.exit_code == 4
        assert "schema show edit-plan" in json.loads(result.stdout)["error"]["hint"]

    def test_a_plan_from_the_future_is_refused(self, tmp_path: Path) -> None:
        """An unknown version must stop rather than be half-understood."""
        document = json.loads(_plan().model_dump_json())
        document["schema_version"] = "99.0"
        path = tmp_path / "future.json"
        path.write_text(json.dumps(document), encoding="utf-8")
        result = runner.invoke(app, ["plan", "show", str(path)])
        assert result.exit_code == 4
        assert "does not know" in json.loads(result.stdout)["error"]["message"]

    def test_the_project_can_be_pointed_at_explicitly(self, project: ProjectPaths) -> None:
        """A plan kept outside the project still needs the analysis beside it."""
        elsewhere = project.root.parent / "kept_apart.json"
        elsewhere.write_text(_plan().model_dump_json(), encoding="utf-8")
        result = runner.invoke(
            app, ["plan", "show", str(elsewhere), "--project", str(project.root)]
        )
        assert result.exit_code == 0, result.output
        assert "shot=wide" in result.stdout


# --------------------------------------------------------------------------- #
# plan diff
# --------------------------------------------------------------------------- #


class TestPlanDiff:
    def test_identical_plans(self, project: ProjectPaths) -> None:
        first = _write(project, _plan(), "a.json")
        second = _write(project, _plan(), "b.json")
        result = runner.invoke(app, ["plan", "diff", str(first), str(second)])
        assert result.exit_code == 0, result.output
        assert "identical=true" in result.stdout

    def test_an_addition_is_marked(self, project: ProjectPaths) -> None:
        before = _write(project, _plan(_clip("c1")), "a.json")
        after = _write(project, _plan(_clip("c1"), _clip("c2", start=5.0, end=9.0)), "b.json")
        result = runner.invoke(app, ["plan", "diff", str(before), str(after)])
        assert "added=1" in result.stdout
        assert "+ c2" in result.stdout

    def test_a_removal_is_marked(self, project: ProjectPaths) -> None:
        before = _write(project, _plan(_clip("c1"), _clip("c2", start=5.0, end=9.0)), "a.json")
        after = _write(project, _plan(_clip("c1")), "b.json")
        result = runner.invoke(app, ["plan", "diff", str(before), str(after)])
        assert "- c2" in result.stdout

    def test_a_modification_shows_both_values(self, project: ProjectPaths) -> None:
        before = _write(project, _plan(_clip("c1", start=0.0, end=4.0)), "a.json")
        after = _write(project, _plan(_clip("c1", start=2.0, end=8.0)), "b.json")
        result = runner.invoke(app, ["plan", "diff", str(before), str(after)])
        line = next(line for line in _digest(result.stdout) if line.startswith("~ c1"))
        assert "src=0.00-4.00" in line
        assert "src=2.00-8.00" in line

    def test_inserting_a_clip_does_not_report_the_others_as_changed(
        self, project: ProjectPaths
    ) -> None:
        """Positional diffing would bury the real change; this is the reason for id matching."""
        keep = (_clip("c1"), _clip("c2", start=5.0, end=9.0), _clip("c3", start=10.0, end=14.0))
        before = _write(project, _plan(*keep), "a.json")
        after = _write(project, _plan(_clip("cNEW", start=20.0, end=24.0), *keep), "b.json")
        result = runner.invoke(app, ["plan", "diff", str(before), str(after)])
        assert "added=1" in result.stdout
        assert "changed=0" in result.stdout

    def test_reordering_is_reported_once(self, project: ProjectPaths) -> None:
        first, second = _clip("c1"), _clip("c2", start=5.0, end=9.0)
        before = _write(project, _plan(first, second), "a.json")
        after = _write(project, _plan(second, first), "b.json")
        result = runner.invoke(app, ["plan", "diff", str(before), str(after)])
        assert "reordered=true" in result.stdout
        assert "changed=0" in result.stdout

    def test_plan_level_changes_are_listed_separately(self, project: ProjectPaths) -> None:
        before = _write(project, _plan(_clip("c1"), notes="first pass"), "a.json")
        after = _write(project, _plan(_clip("c1"), notes="second pass"), "b.json")
        result = runner.invoke(app, ["plan", "diff", str(before), str(after)])
        assert "# plan_level_changes: notes" in result.stdout

    def test_the_duration_delta_is_signed(self, project: ProjectPaths) -> None:
        before = _write(project, _plan(_clip("c1", end=4.0)), "a.json")
        after = _write(project, _plan(_clip("c1", end=10.0)), "b.json")
        result = runner.invoke(app, ["plan", "diff", str(before), str(after)])
        assert "duration=4.00->10.00(+6.00)" in result.stdout

    def test_full_emits_the_diff_as_json(self, project: ProjectPaths) -> None:
        before = _write(project, _plan(_clip("c1")), "a.json")
        after = _write(project, _plan(_clip("c1"), _clip("c2", start=5.0, end=9.0)), "b.json")
        result = runner.invoke(app, ["plan", "diff", str(before), str(after), "--full"])
        assert json.loads(result.stdout)["added"] == ["c2"]

    def test_a_missing_plan_on_either_side_exits_not_found(
        self, project: ProjectPaths, tmp_path: Path
    ) -> None:
        existing = _write(project, _plan(), "a.json")
        result = runner.invoke(app, ["plan", "diff", str(existing), str(tmp_path / "absent.json")])
        assert result.exit_code == 3


# --------------------------------------------------------------------------- #
# plan subtitles
# --------------------------------------------------------------------------- #


class TestPlanSubtitles:
    def test_cues_are_reported_on_stdout(self, project: ProjectPaths) -> None:
        path = _write(project, _plan())
        result = runner.invoke(app, ["plan", "subtitles", str(path)])
        assert result.exit_code == 0, result.output
        header = _digest(result.stdout)[0]
        assert "cues=2" in header
        assert "prepare the soil" in result.stdout

    def test_by_default_nothing_is_written(self, project: ProjectPaths) -> None:
        """Attaching cues rewrites the plan; doing that unasked would be surprising."""
        path = _write(project, _plan())
        original = path.read_bytes()
        result = runner.invoke(app, ["plan", "subtitles", str(path)])
        assert path.read_bytes() == original
        assert "(not saved)" in result.stdout

    def test_in_place_overwrites_the_input(self, project: ProjectPaths) -> None:
        path = _write(project, _plan())
        result = runner.invoke(app, ["plan", "subtitles", str(path), "--in-place"])
        assert result.exit_code == 0, result.output
        assert len(EditPlan.model_validate_json(path.read_text(encoding="utf-8")).subtitles) == 2

    def test_output_writes_elsewhere_and_leaves_the_input_alone(
        self, project: ProjectPaths
    ) -> None:
        path = _write(project, _plan())
        destination = project.root / "with_subs.json"
        result = runner.invoke(app, ["plan", "subtitles", str(path), "--output", str(destination)])
        assert result.exit_code == 0, result.output
        assert EditPlan.model_validate_json(destination.read_text(encoding="utf-8")).subtitles
        assert not EditPlan.model_validate_json(path.read_text(encoding="utf-8")).subtitles

    def test_cues_are_timed_against_the_plans_kept_ranges(self, project: ProjectPaths) -> None:
        """The plan is authoritative. Source 4.0s lands at 2.0s once 2.0-4.0 is removed."""
        path = _write(project, _plan())
        result = runner.invoke(app, ["plan", "subtitles", str(path)])
        second = next(line for line in _digest(result.stdout) if line.startswith("c001"))
        assert second.split()[1].startswith("2.00-")

    def test_a_plan_that_trimmed_further_than_cleanup_wins(self, project: ProjectPaths) -> None:
        """Cleanup kept 4.0-10.0; this plan drops it, so only the first sentence survives."""
        path = _write(
            project,
            _plan(
                _clip("c1", end=4.0),
                narration=NarrationTrack(
                    source=NARRATION, kept_ranges=(TimeRange(start=0.0, end=2.0),)
                ),
            ),
        )
        result = runner.invoke(app, ["plan", "subtitles", str(path)])
        assert "cues=1" in result.stdout

    def test_a_plan_that_removed_all_speech_is_an_error_not_an_empty_success(
        self, project: ProjectPaths
    ) -> None:
        path = _write(
            project,
            _plan(
                _clip("c1", end=4.0),
                narration=NarrationTrack(
                    source=NARRATION, kept_ranges=(TimeRange(start=20.0, end=30.0),)
                ),
            ),
        )
        result = runner.invoke(app, ["plan", "subtitles", str(path)])
        assert result.exit_code == 4
        assert json.loads(result.stdout)["error"]["code"] == "subtitle.empty"

    def test_without_a_narration_analysis_the_next_command_is_named(self, tmp_path: Path) -> None:
        paths = ProjectPaths.for_root(tmp_path / "bare")
        paths.ensure()
        path = _write(paths, _plan())
        result = runner.invoke(app, ["plan", "subtitles", str(path)])
        assert result.exit_code == 3
        error = json.loads(result.stdout)["error"]
        assert error["code"] == "narration.not_analysed"
        assert "analyze audio" in error["hint"]

    def test_a_corrupt_narration_analysis_is_reported_as_such(self, project: ProjectPaths) -> None:
        project.narration_file.write_text("{not json", encoding="utf-8")
        path = _write(project, _plan())
        result = runner.invoke(app, ["plan", "subtitles", str(path)])
        assert result.exit_code == 4
        assert json.loads(result.stdout)["error"]["code"] == "analysis.unreadable"

    def test_full_emits_the_updated_plan_as_json(self, project: ProjectPaths) -> None:
        path = _write(project, _plan())
        result = runner.invoke(app, ["plan", "subtitles", str(path), "--full"])
        payload = json.loads(result.stdout)
        assert len(payload["subtitles"]) == 2

    def test_the_result_round_trips_as_a_valid_plan(self, project: ProjectPaths) -> None:
        """A command that emits a plan must emit one the next command can read."""
        path = _write(project, _plan())
        result = runner.invoke(app, ["plan", "subtitles", str(path), "--full"])
        EditPlan.model_validate_json(result.stdout)

    def test_running_it_twice_is_idempotent(self, project: ProjectPaths) -> None:
        """Cues are replaced, not merged: two sets for the same words would both render."""
        path = _write(project, _plan())
        runner.invoke(app, ["plan", "subtitles", str(path), "--in-place"])
        runner.invoke(app, ["plan", "subtitles", str(path), "--in-place"])
        assert len(EditPlan.model_validate_json(path.read_text(encoding="utf-8")).subtitles) == 2


# --------------------------------------------------------------------------- #
# The stdout contract, on the new commands specifically
# --------------------------------------------------------------------------- #


class TestStdoutRemainsParseable:
    @pytest.mark.parametrize("command", [["show"], ["subtitles"]])
    def test_no_log_output_reaches_stdout(self, project: ProjectPaths, command: list[str]) -> None:
        path = _write(project, _plan())
        result = runner.invoke(app, ["plan", *command, str(path), "--full"])
        json.loads(result.stdout)  # would raise if a single log line leaked

    def test_a_digest_is_only_comments_and_records(self, project: ProjectPaths) -> None:
        path = _write(project, _plan())
        result = runner.invoke(app, ["plan", "show", str(path)])
        for line in _digest(result.stdout):
            assert line[0] in "#0123456789EWi", line
