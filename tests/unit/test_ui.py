"""Tests for the desktop UI.

Split the way the code is. :mod:`app.ui.tasks` imports no Qt, so the step catalogue,
preconditions and next-step logic are tested as plain Python — no display, no event loop,
no ``QApplication``. The Qt layer is tested separately and skips cleanly when PySide6 is
absent, because it is an optional extra and the suite must pass without it.

Qt tests run on the ``offscreen`` platform, so they need no display on CI or on a headless
Windows agent.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from app.models.common import MediaRef, TimeRange
from app.models.edit_plan import EditPlan, TimelineClip
from app.services.paths import ProjectPaths
from app.ui.tasks import (
    STEPS,
    STEPS_BY_ID,
    ProjectState,
    StepId,
    blockers,
    can_run,
    next_suggested,
)

# --------------------------------------------------------------------------- #
# Builders
# --------------------------------------------------------------------------- #


def _state(**overrides: object) -> ProjectState:
    fields: dict[str, object] = {
        "root": Path("/project"),
        "is_project": True,
        "has_narration": False,
        "has_footage": False,
        "has_music": False,
        "has_narration_analysis": False,
        "has_footage_analysis": False,
        "has_music_analysis": False,
        "has_plan": False,
        "plan": None,
    }
    fields.update(overrides)
    return ProjectState(**fields)  # type: ignore[arg-type]


def _plan(*, placed: bool = True, subtitles: tuple = ()) -> EditPlan:
    return EditPlan(
        project_id="ui",
        created_by="pytest",
        clips=(
            TimelineClip(
                id="c1",
                source=MediaRef(path="raw/001.mp4"),
                source_range=TimeRange(start=0.0, end=4.0),
                timeline_start=0.0 if placed else None,
                reason="a considered editorial reason",
            ),
        ),
        subtitles=subtitles,
    )


# --------------------------------------------------------------------------- #
# The step catalogue
# --------------------------------------------------------------------------- #


class TestSteps:
    def test_every_step_id_has_exactly_one_step(self) -> None:
        assert set(STEPS_BY_ID) == set(StepId)
        assert len(STEPS) == len(StepId)

    def test_every_step_has_a_label_and_a_description(self) -> None:
        """Both reach the user: the label is the button, the description its tooltip."""
        for step in STEPS:
            assert step.label.strip()
            assert step.description.strip()

    def test_only_the_genuinely_slow_steps_ask_for_confirmation(self) -> None:
        """A confirmation on every action is a confirmation nobody reads."""
        slow = {step.id for step in STEPS if step.is_slow}
        assert slow == {StepId.ANALYZE_AUDIO, StepId.ANALYZE_VIDEO, StepId.RENDER_FINAL}

    def test_the_steps_are_listed_in_pipeline_order(self) -> None:
        """The panel reads top to bottom, so analysis must precede planning and rendering."""
        order = [step.id for step in STEPS]
        assert order.index(StepId.ANALYZE_AUDIO) < order.index(StepId.BRIEF)
        assert order.index(StepId.BRIEF) < order.index(StepId.RENDER_DRAFT)
        assert order.index(StepId.RENDER_DRAFT) < order.index(StepId.EXPORT_CAPCUT)

    def test_every_precondition_is_one_blockers_can_actually_check(self) -> None:
        """A typo in a `requires` string would silently make a step permanently runnable."""
        known = {
            "a narration file",
            "footage in raw/",
            "audio in music/",
            "narration analysis",
            "footage analysis",
            "an edit plan",
        }
        for step in STEPS:
            assert set(step.requires) <= known, step.id


class TestBlockers:
    def test_a_non_project_blocks_everything(self) -> None:
        state = _state(is_project=False)
        for step in STEPS:
            assert not can_run(step, state)

    def test_a_blocker_says_what_is_missing(self) -> None:
        """So a disabled button can explain itself instead of prompting a support question."""
        problems = blockers(STEPS_BY_ID[StepId.ANALYZE_AUDIO], _state())
        assert problems == ("a narration file",)

    def test_a_step_with_its_requirement_met_can_run(self) -> None:
        assert can_run(STEPS_BY_ID[StepId.ANALYZE_AUDIO], _state(has_narration=True))

    def test_scan_needs_only_a_project(self) -> None:
        assert can_run(STEPS_BY_ID[StepId.SCAN], _state())

    def test_the_brief_needs_both_analyses(self) -> None:
        step = STEPS_BY_ID[StepId.BRIEF]
        assert len(blockers(step, _state())) == 2
        assert len(blockers(step, _state(has_narration_analysis=True))) == 1
        assert can_run(step, _state(has_narration_analysis=True, has_footage_analysis=True))

    def test_rendering_needs_a_plan(self) -> None:
        assert not can_run(STEPS_BY_ID[StepId.RENDER_FINAL], _state())
        assert can_run(STEPS_BY_ID[StepId.RENDER_FINAL], _state(has_plan=True))

    def test_subtitles_need_a_plan_and_the_narration_analysis(self) -> None:
        step = STEPS_BY_ID[StepId.SUBTITLES]
        assert not can_run(step, _state(has_plan=True))
        assert can_run(step, _state(has_plan=True, has_narration_analysis=True))


class TestNextSuggested:
    def test_a_non_project_suggests_nothing(self) -> None:
        assert next_suggested(_state(is_project=False)) is None

    def test_a_fresh_project_starts_with_a_scan(self) -> None:
        """Scan is cheap and answers "what did I actually give it?" before anything slow."""
        step = next_suggested(_state())
        assert step is not None
        assert step.id is StepId.SCAN

    def test_a_scanned_project_moves_on_to_the_footage(self) -> None:
        """Scan writes no artifact, so an existing analysis is what marks it done."""
        step = next_suggested(
            _state(has_narration=True, has_footage=True, has_narration_analysis=True)
        )
        assert step is not None
        assert step.id is StepId.ANALYZE_VIDEO

    def test_a_project_missing_its_media_suggests_nothing(self) -> None:
        """Honest: with no footage there is genuinely nothing to run. The window turns this
        into "Add footage in raw/ to continue" rather than leaving a stale message up."""
        assert next_suggested(_state(has_narration=True, has_narration_analysis=True)) is None

    def test_both_analyses_done_suggests_the_brief(self) -> None:
        step = next_suggested(
            _state(
                has_narration=True,
                has_footage=True,
                has_narration_analysis=True,
                has_footage_analysis=True,
            )
        )
        assert step is not None
        assert step.id is StepId.BRIEF

    def test_a_project_with_no_music_does_not_stall_on_music_analysis(self) -> None:
        """Music is optional. Suggesting a step that cannot run would dead-end the panel."""
        step = next_suggested(
            _state(
                has_narration=True,
                has_footage=True,
                has_narration_analysis=True,
                has_footage_analysis=True,
                has_music=False,
            )
        )
        assert step is not None
        assert step.id is not StepId.ANALYZE_MUSIC

    def test_an_unplaced_plan_suggests_normalising(self) -> None:
        step = next_suggested(
            _state(
                has_narration=True,
                has_footage=True,
                has_narration_analysis=True,
                has_footage_analysis=True,
                has_plan=True,
                plan=_plan(placed=False),
            )
        )
        assert step is not None
        assert step.id is StepId.NORMALIZE

    def test_a_placed_plan_without_cues_suggests_subtitles(self) -> None:
        step = next_suggested(
            _state(
                has_narration=True,
                has_footage=True,
                has_narration_analysis=True,
                has_footage_analysis=True,
                has_plan=True,
                plan=_plan(placed=True),
            )
        )
        assert step is not None
        assert step.id is StepId.SUBTITLES

    def test_a_finished_plan_suggests_rendering(self) -> None:
        from app.models.edit_plan import SubtitleCue

        step = next_suggested(
            _state(
                has_narration=True,
                has_footage=True,
                has_narration_analysis=True,
                has_footage_analysis=True,
                has_music=False,
                has_plan=True,
                plan=_plan(
                    placed=True,
                    subtitles=(SubtitleCue(range=TimeRange(start=0.0, end=1.0), text="hi"),),
                ),
            )
        )
        assert step is not None
        assert step.id is StepId.RENDER_DRAFT

    def test_the_suggestion_is_always_runnable(self) -> None:
        """A highlighted button that is greyed out is worse than no highlight."""
        for state in (
            _state(),
            _state(has_narration=True),
            _state(has_narration=True, has_footage=True),
            _state(has_plan=True, plan=_plan()),
        ):
            step = next_suggested(state)
            if step is not None:
                assert can_run(step, state)


# --------------------------------------------------------------------------- #
# Reading a project from disk
# --------------------------------------------------------------------------- #


class TestProjectState:
    def test_an_empty_directory_is_not_a_project(self, tmp_path: Path) -> None:
        assert ProjectState.read(tmp_path).is_project is False

    def test_an_initialised_project_is_recognised(self, tmp_path: Path) -> None:
        ProjectPaths.for_root(tmp_path).ensure()
        assert ProjectState.read(tmp_path).is_project is True

    def test_media_presence_is_read_from_disk(self, tmp_path: Path) -> None:
        paths = ProjectPaths.for_root(tmp_path)
        paths.ensure()
        (paths.root / "narration.wav").touch()
        (paths.raw / "001.mp4").touch()
        (paths.music / "bed.mp3").touch()

        state = ProjectState.read(tmp_path)
        assert state.has_narration
        assert state.has_footage
        assert state.has_music

    def test_a_plan_is_loaded_when_present(self, tmp_path: Path) -> None:
        paths = ProjectPaths.for_root(tmp_path)
        paths.ensure()
        paths.edit_plan_file.write_text(_plan().model_dump_json(), encoding="utf-8")

        state = ProjectState.read(tmp_path)
        assert state.has_plan
        assert state.plan is not None

    def test_an_unreadable_plan_is_a_display_state_not_a_crash(self, tmp_path: Path) -> None:
        """A user part-way through hand-editing a plan must not crash the app showing it."""
        paths = ProjectPaths.for_root(tmp_path)
        paths.ensure()
        paths.edit_plan_file.write_text("{ broken", encoding="utf-8")

        state = ProjectState.read(tmp_path)
        assert state.has_plan is True
        assert state.plan is None

    def test_state_reflects_the_filesystem_rather_than_a_cache(self, tmp_path: Path) -> None:
        """The CLI is the primary interface; a user will change things in a terminal."""
        paths = ProjectPaths.for_root(tmp_path)
        paths.ensure()
        assert ProjectState.read(tmp_path).has_footage is False
        (paths.raw / "001.mp4").touch()
        assert ProjectState.read(tmp_path).has_footage is True


# --------------------------------------------------------------------------- #
# The Qt layer
# --------------------------------------------------------------------------- #

pytest.importorskip("PySide6", reason="the desktop UI is an optional extra")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(scope="module")
def qt_app():
    """One QApplication for the module. Qt permits exactly one per process."""
    from PySide6.QtWidgets import QApplication

    application = QApplication.instance() or QApplication([])
    yield application


@pytest.fixture
def project(tmp_path: Path) -> Path:
    paths = ProjectPaths.for_root(tmp_path)
    paths.ensure()
    paths.edit_plan_file.write_text(_plan().model_dump_json(indent=2), encoding="utf-8")
    return paths.root


class TestTimelineModel:
    def test_an_empty_model_has_no_rows(self, qt_app: object) -> None:
        from app.ui.models import TimelineModel

        assert TimelineModel().rowCount() == 0

    def test_one_row_per_clip(self, qt_app: object) -> None:
        from app.plan.review import build_review
        from app.ui.models import TimelineModel

        model = TimelineModel()
        model.set_review(build_review(_plan()))
        assert model.rowCount() == 1
        assert model.columnCount() == 8

    def test_the_reason_is_the_last_column(self, qt_app: object) -> None:
        """It is why the table exists; everything else can be read off the video."""
        from PySide6.QtCore import Qt

        from app.plan.review import build_review
        from app.ui.models import TimelineModel

        model = TimelineModel()
        model.set_review(build_review(_plan()))
        value = model.data(model.index(0, 7), Qt.ItemDataRole.DisplayRole)
        assert value == "a considered editorial reason"

    def test_the_full_reason_is_available_as_a_tooltip(self, qt_app: object) -> None:
        """The column elides; the tooltip must not."""
        from PySide6.QtCore import Qt

        from app.plan.review import build_review
        from app.ui.models import TimelineModel

        model = TimelineModel()
        model.set_review(build_review(_plan()))
        tooltip = model.data(model.index(0, 0), Qt.ItemDataRole.ToolTipRole)
        assert tooltip == "a considered editorial reason"

    def test_an_unplaced_clip_says_so_rather_than_showing_a_blank(self, qt_app: object) -> None:
        from PySide6.QtCore import Qt

        from app.plan.review import build_review
        from app.ui.models import TimelineModel

        model = TimelineModel()
        model.set_review(build_review(_plan(placed=False)))
        assert model.data(model.index(0, 3), Qt.ItemDataRole.DisplayRole) == "unplaced"

    def test_clearing_the_review_empties_the_table(self, qt_app: object) -> None:
        from app.plan.review import build_review
        from app.ui.models import TimelineModel

        model = TimelineModel()
        model.set_review(build_review(_plan()))
        model.set_review(None)
        assert model.rowCount() == 0

    def test_headers_are_labelled(self, qt_app: object) -> None:
        from PySide6.QtCore import Qt

        from app.ui.models import TimelineModel

        model = TimelineModel()
        header = model.headerData(7, Qt.Orientation.Horizontal, Qt.ItemDataRole.DisplayRole)
        assert header == "Reason"


class TestNotesModel:
    def test_notes_are_listed_with_their_hint(self, qt_app: object) -> None:
        from PySide6.QtCore import Qt

        from app.plan.review import build_review
        from app.ui.models import NotesModel

        model = NotesModel()
        model.set_review(build_review(_plan()))
        assert model.rowCount() >= 1
        assert model.columnCount() == 2
        assert model.data(model.index(0, 0), Qt.ItemDataRole.DisplayRole)

    def test_severity_is_colour_not_markup(self, qt_app: object) -> None:
        from PySide6.QtCore import Qt

        from app.plan.review import build_review
        from app.ui.models import NotesModel

        model = NotesModel()
        model.set_review(build_review(_plan()))
        colour = model.data(model.index(0, 0), Qt.ItemDataRole.ForegroundRole)
        assert colour is not None

    def test_the_code_is_available_as_a_tooltip(self, qt_app: object) -> None:
        """The message is for reading; the code is for searching the docs."""
        from PySide6.QtCore import Qt

        from app.plan.review import build_review
        from app.ui.models import NotesModel

        model = NotesModel()
        model.set_review(build_review(_plan()))
        assert str(model.data(model.index(0, 0), Qt.ItemDataRole.ToolTipRole)).startswith("review.")


class TestStepRunner:
    def test_it_starts_idle(self, qt_app: object) -> None:
        from app.ui.workers import StepRunner

        assert StepRunner().is_busy is False

    def test_a_second_step_is_refused_while_one_runs(self, qt_app: object, project: Path) -> None:
        """Serialised on purpose: a render started while its footage analysis is still
        running would produce a video from a plan that no longer matches the project."""
        from app.ui.tasks import StepId
        from app.ui.workers import StepRunner

        runner = StepRunner()
        assert runner.start(StepId.SCAN, project) is True
        assert runner.start(StepId.SCAN, project) is False
        runner.wait()

    def test_a_failing_step_becomes_a_signal_not_a_crash(
        self, qt_app: object, tmp_path: Path
    ) -> None:
        """The top of a thread: an uncaught exception would leave the window waiting for a
        `finished` that never arrives."""
        from PySide6.QtCore import QEventLoop, QTimer

        from app.ui.tasks import StepId
        from app.ui.workers import StepRunner

        runner = StepRunner()
        failures: list[str] = []
        loop = QEventLoop()
        runner.failed.connect(lambda headline, _detail: (failures.append(headline), loop.quit()))
        runner.finished.connect(lambda _result: loop.quit())

        # No plan here, so the render step raises before it reaches FFmpeg.
        runner.start(StepId.RENDER_DRAFT, tmp_path)
        QTimer.singleShot(20_000, loop.quit)
        loop.exec()
        runner.wait()

        assert failures
        assert runner.is_busy is False

    def test_a_completed_step_reports_a_result_and_clears_busy(
        self, qt_app: object, project: Path
    ) -> None:
        from PySide6.QtCore import QEventLoop, QTimer

        from app.ui.tasks import StepId, StepResult
        from app.ui.workers import StepRunner

        runner = StepRunner()
        results: list[StepResult] = []
        loop = QEventLoop()
        runner.finished.connect(lambda result: (results.append(result), loop.quit()))
        runner.failed.connect(lambda _h, _d: loop.quit())

        runner.start(StepId.SCAN, project)
        QTimer.singleShot(20_000, loop.quit)
        loop.exec()
        runner.wait()

        assert results
        assert results[0].summary
        assert runner.is_busy is False


class TestMainWindow:
    def test_it_builds_without_a_project(self, qt_app: object) -> None:
        from app.ui.window import MainWindow

        window = MainWindow()
        assert window.windowTitle() == "AIVE"
        window.close()

    def test_opening_a_project_populates_the_table(self, qt_app: object, project: Path) -> None:
        from app.ui.window import MainWindow

        window = MainWindow(project)
        assert window._timeline_model.rowCount() == 1
        assert project.name in window.windowTitle()
        window.close()

    def test_a_disabled_button_explains_itself_in_its_tooltip(
        self, qt_app: object, tmp_path: Path
    ) -> None:
        from app.ui.tasks import StepId
        from app.ui.window import MainWindow

        ProjectPaths.for_root(tmp_path).ensure()
        window = MainWindow(tmp_path)
        button = window._buttons[StepId.RENDER_FINAL]
        assert button.isEnabled() is False
        assert "Needs" in button.toolTip()
        window.close()

    def test_the_suggested_step_is_the_default_button(self, qt_app: object, project: Path) -> None:
        from app.ui.window import MainWindow

        window = MainWindow(project)
        defaults = [sid for sid, button in window._buttons.items() if button.isDefault()]
        assert len(defaults) == 1
        window.close()

    def test_a_project_with_nothing_runnable_says_what_is_missing(
        self, qt_app: object, tmp_path: Path
    ) -> None:
        """A window with no highlighted button and a stale status bar looks broken."""
        from app.ui.window import MainWindow

        paths = ProjectPaths.for_root(tmp_path)
        paths.ensure()
        # An analysis exists but its media is gone - the project was moved, or the media
        # folder was cleaned out. Scan counts as done, and nothing else can run.
        paths.narration_file.write_text("{}", encoding="utf-8")

        window = MainWindow(tmp_path)
        message = window.statusBar().currentMessage()
        assert "narration" in message
        assert "raw/" in message
        window.close()

    def test_refresh_picks_up_a_change_made_outside_the_window(
        self, qt_app: object, project: Path
    ) -> None:
        """A user will run CLI commands in a terminal beside this window."""
        from app.ui.window import MainWindow

        window = MainWindow(project)
        assert window._state is not None
        assert window._state.has_footage is False

        (ProjectPaths.for_root(project).raw / "001.mp4").touch()
        window.refresh()
        assert window._state.has_footage is True
        window.close()
