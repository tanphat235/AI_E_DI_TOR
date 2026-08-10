"""The main window.

Deliberately thin. Everything worth testing — which steps exist, what each needs, what
running one does — is in :mod:`app.ui.tasks`, which imports no Qt. This module lays out
widgets and connects signals, and that is all it should ever do.

The layout follows the pipeline because the pipeline is what a user is trying to get
through: steps on the left in the order the docs recommend, the plan on the right, a log at
the bottom. One step is highlighted as the suggested next action, because eleven equally
weighted buttons is a workflow nobody finishes.

**This is a viewer and a launcher, not an editor.** It shows the plan and runs the steps;
it does not let a user drag a clip. That boundary is honest rather than reluctant — the
editing surface for an AIVE plan is the director, or CapCut after an export, and a
half-featured timeline editor would be worse than either.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QAction, QFont
from PySide6.QtWidgets import (
    QFileDialog,
    QGroupBox,
    QHeaderView,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSplitter,
    QTableView,
    QVBoxLayout,
    QWidget,
)

from app.models.video import FootageAnalysis
from app.plan.review import build_review
from app.services.paths import ProjectPaths
from app.ui.models import NotesModel, TimelineModel
from app.ui.tasks import STEPS, ProjectState, Step, StepId, StepResult, blockers, next_suggested
from app.ui.workers import StepRunner
from app.utils.logging import get_logger

logger = get_logger(__name__)

WINDOW_TITLE = "AIVE"


class MainWindow(QMainWindow):
    """A window over one project."""

    def __init__(self, project: Path | None = None) -> None:
        super().__init__()
        self.setWindowTitle(WINDOW_TITLE)
        self.resize(1280, 800)

        self._root: Path | None = None
        self._state: ProjectState | None = None
        self._buttons: dict[StepId, QPushButton] = {}

        self._runner = StepRunner(self)
        self._runner.progress.connect(self._on_progress)
        self._runner.finished.connect(self._on_finished)
        self._runner.failed.connect(self._on_failed)
        self._runner.busy_changed.connect(self._on_busy_changed)

        self._build_ui()
        self._build_menu()

        if project is not None:
            self.open_project(project)
        else:
            self._set_placeholder()

    # -- Construction -------------------------------------------------------- #

    def _build_ui(self) -> None:
        splitter = QSplitter(Qt.Orientation.Horizontal, self)
        splitter.addWidget(self._build_steps_panel())
        splitter.addWidget(self._build_plan_panel())
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([340, 940])
        self.setCentralWidget(splitter)

        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setVisible(False)
        self._progress.setMaximumWidth(240)
        self.statusBar().addPermanentWidget(self._progress)
        self.statusBar().showMessage("Open a project to begin.")

    def _build_steps_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)

        self._project_label = QLabel("No project open")
        self._project_label.setWordWrap(True)
        font = self._project_label.font()
        font.setBold(True)
        self._project_label.setFont(font)
        layout.addWidget(self._project_label)

        self._facts_label = QLabel("")
        self._facts_label.setWordWrap(True)
        layout.addWidget(self._facts_label)

        group = QGroupBox("Pipeline")
        steps_layout = QVBoxLayout(group)
        for step in STEPS:
            button = QPushButton(step.label)
            button.setToolTip(step.description)
            button.setEnabled(False)
            button.clicked.connect(lambda _checked=False, item=step: self._run(item))
            steps_layout.addWidget(button)
            self._buttons[step.id] = button
        steps_layout.addStretch(1)
        layout.addWidget(group, 1)

        self._open_button = QPushButton("Open last output")
        self._open_button.setEnabled(False)
        self._open_button.clicked.connect(self._open_artifact)
        layout.addWidget(self._open_button)

        return panel

    def _build_plan_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)

        self._summary_label = QLabel("")
        self._summary_label.setWordWrap(True)
        layout.addWidget(self._summary_label)

        self._timeline = QTableView()
        self._timeline_model = TimelineModel(self)
        self._timeline.setModel(self._timeline_model)
        self._timeline.setSelectionBehavior(QTableView.SelectionBehavior.SelectRows)
        self._timeline.setAlternatingRowColors(True)
        self._timeline.verticalHeader().setVisible(False)
        # The reason column takes whatever is left: it is the column the user is here for.
        self._timeline.horizontalHeader().setStretchLastSection(True)
        self._timeline.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents
        )
        layout.addWidget(self._timeline, 3)

        notes_group = QGroupBox("Review notes")
        notes_layout = QVBoxLayout(notes_group)
        self._notes = QTableView()
        self._notes_model = NotesModel(self)
        self._notes.setModel(self._notes_model)
        self._notes.verticalHeader().setVisible(False)
        self._notes.horizontalHeader().setStretchLastSection(True)
        notes_layout.addWidget(self._notes)
        layout.addWidget(notes_group, 1)

        self._log = QPlainTextEdit()
        self._log.setReadOnly(True)
        self._log.setMaximumBlockCount(500)
        self._log.setFont(QFont("Consolas", 9))
        layout.addWidget(self._log, 1)

        return panel

    def _build_menu(self) -> None:
        file_menu = self.menuBar().addMenu("&File")

        open_action = QAction("&Open project...", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self._choose_project)
        file_menu.addAction(open_action)

        refresh_action = QAction("&Refresh", self)
        refresh_action.setShortcut("F5")
        refresh_action.triggered.connect(self.refresh)
        file_menu.addAction(refresh_action)

        file_menu.addSeparator()
        quit_action = QAction("&Quit", self)
        quit_action.setShortcut("Ctrl+Q")
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

    # -- Project ------------------------------------------------------------- #

    @Slot()
    def _choose_project(self) -> None:
        chosen = QFileDialog.getExistingDirectory(self, "Open AIVE project")
        if chosen:
            self.open_project(Path(chosen))

    def open_project(self, root: Path) -> None:
        self._root = root
        self.setWindowTitle(f"{WINDOW_TITLE} - {root.name}")
        self.refresh()

    @Slot()
    def refresh(self) -> None:
        """Re-read the project from disk and update every widget.

        Read rather than remembered, because the CLI is the primary interface and a user
        will run commands in a terminal beside this window. A cached view would be
        confidently stale.
        """
        if self._root is None:
            return

        self._state = ProjectState.read(self._root)
        state = self._state

        self._project_label.setText(str(state.root))
        if not state.is_project:
            self._facts_label.setText(
                "Not an AIVE project. Run `aive project init` here, or choose another folder."
            )
        else:
            self._facts_label.setText(
                f"narration: {_mark(state.has_narration)}   "
                f"footage: {_mark(state.has_footage)}   "
                f"music: {_mark(state.has_music)}\n"
                f"analysed - audio: {_mark(state.has_narration_analysis)}   "
                f"video: {_mark(state.has_footage_analysis)}   "
                f"music: {_mark(state.has_music_analysis)}\n"
                f"plan: {_mark(state.has_plan)}"
            )

        suggested = next_suggested(state)
        for step in STEPS:
            button = self._buttons[step.id]
            problems = blockers(step, state)
            button.setEnabled(not problems and not self._runner.is_busy)
            # The tooltip carries the *reason* a button is disabled. "Greyed out" is a
            # support question; "needs a narration file" is not.
            button.setToolTip(f"Needs: {', '.join(problems)}" if problems else step.description)
            is_next = suggested is not None and step.id == suggested.id
            button.setDefault(is_next)
            font = button.font()
            font.setBold(is_next)
            button.setFont(font)

        self._refresh_plan(state)
        if suggested is not None:
            self.statusBar().showMessage(f"Next: {suggested.label}")
        else:
            # Nothing runnable is a state with a cause, and leaving the previous message up
            # reads as a stale window. Name what is missing, or say the pipeline is done.
            self.statusBar().showMessage(_nothing_to_do(state))

    def _refresh_plan(self, state: ProjectState) -> None:
        if state.plan is None:
            self._timeline_model.set_review(None)
            self._notes_model.set_review(None)
            self._summary_label.setText("No edit plan yet.")
            return

        paths = ProjectPaths.for_root(state.root)
        footage = None
        if paths.footage_analysis_file.is_file():
            try:
                footage = FootageAnalysis.model_validate_json(
                    paths.footage_analysis_file.read_text(encoding="utf-8")
                )
            except (OSError, ValueError):
                # A stale analysis costs shot types, not the whole view.
                footage = None

        review = build_review(state.plan, footage=footage)
        self._timeline_model.set_review(review)
        self._notes_model.set_review(review)

        statistics = review.statistics
        self._summary_label.setText(
            f"{statistics.clip_count} clips over {statistics.total_duration:.2f}s   "
            f"{statistics.cuts_per_minute:.1f} cuts/min   "
            f"{statistics.distinct_sources} source(s)   "
            f"{statistics.subtitle_count} subtitle(s)   "
            f"placed: {'yes' if review.is_placed else 'no'}   "
            f"by {review.created_by}"
        )

    # -- Running ------------------------------------------------------------- #

    def _run(self, step: Step) -> None:
        if self._root is None or self._state is None:
            return
        if step.is_slow and not self._confirm_slow(step):
            return
        if not self._runner.start(step.id, self._root):
            self.statusBar().showMessage("A step is already running.")
            return
        self._append(f"--- {step.label} ---")

    def _confirm_slow(self, step: Step) -> bool:
        """Ask before a step that takes the machine for minutes.

        Only for the genuinely slow ones. A confirmation on every action is a confirmation
        nobody reads.
        """
        answer = QMessageBox.question(
            self,
            step.label,
            f"{step.description}\n\nThis can take several minutes. Start it?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        )
        return answer == QMessageBox.StandardButton.Yes

    @Slot(float, str)
    def _on_progress(self, fraction: float, message: str) -> None:
        self._progress.setValue(int(fraction * 100))
        self.statusBar().showMessage(message)

    @Slot(object)
    def _on_finished(self, result: StepResult) -> None:
        self._append(result.summary)
        if result.detail:
            self._append(result.detail)
        self._last_artifact = result.artifact
        self._open_button.setEnabled(result.artifact is not None)
        self.statusBar().showMessage(result.summary)
        self.refresh()

    @Slot(str, str)
    def _on_failed(self, headline: str, detail: str) -> None:
        self._append(f"FAILED: {headline}")
        logger.debug("Step failure detail:\n%s", detail)
        # The headline goes in the dialog and the traceback into the log, rather than a
        # modal wall of text the user has to dismiss before they can read anything.
        QMessageBox.warning(self, "Step failed", headline)
        self.statusBar().showMessage("Failed. See the log.")
        self.refresh()

    @Slot(bool)
    def _on_busy_changed(self, busy: bool) -> None:
        self._progress.setVisible(busy)
        if not busy:
            self._progress.setValue(0)
        for button in self._buttons.values():
            button.setEnabled(button.isEnabled() and not busy)
        if not busy:
            self.refresh()

    # -- Output -------------------------------------------------------------- #

    _last_artifact: Path | None = None

    @Slot()
    def _open_artifact(self) -> None:
        """Reveal the last output in the platform file manager."""
        target = self._last_artifact
        if target is None:
            return
        folder = target if target.is_dir() else target.parent
        try:
            if sys.platform == "win32":
                os.startfile(folder)
            elif sys.platform == "darwin":
                subprocess.run(["open", str(folder)], check=False)
            else:
                subprocess.run(["xdg-open", str(folder)], check=False)
        except OSError as exc:
            self._append(f"Could not open {folder}: {exc}")

    def _append(self, text: str) -> None:
        self._log.appendPlainText(text)

    def closeEvent(self, event: object) -> None:  # noqa: N802 - Qt's name
        """Wait for a running step before the process exits.

        A half-written analysis document is worse than no document: the next run would read
        it, trust it, and plan from truncated JSON.
        """
        if self._runner.is_busy:
            self.statusBar().showMessage("Finishing the current step...")
            self._runner.wait()
        super().closeEvent(event)  # type: ignore[arg-type]

    def _set_placeholder(self) -> None:
        self._summary_label.setText("Open a project with Ctrl+O.")


def _mark(value: bool) -> str:
    return "yes" if value else "-"


def _nothing_to_do(state: ProjectState) -> str:
    """Why no step is suggested.

    Three genuinely different situations, and telling them apart is the difference between
    a window that looks broken and one that tells the user what to add.
    """
    if not state.is_project:
        return "Not an AIVE project. Run `aive project init` here."
    missing = [
        name
        for name, present in (
            ("a narration file", state.has_narration),
            ("footage in raw/", state.has_footage),
        )
        if not present
    ]
    if missing:
        return f"Add {' and '.join(missing)} to continue."
    return "Everything is up to date. Render or export when you are ready."


__all__ = ["WINDOW_TITLE", "MainWindow"]
