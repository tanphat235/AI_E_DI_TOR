"""Running pipeline steps off the GUI thread.

Analysis and rendering take minutes. Run them on the GUI thread and the window stops
repainting, Windows greys it out and labels it "Not Responding", and the user kills an
encode that was working. So every step runs on a worker.

The rule that makes Qt threading safe is simple and absolute: **widgets are only ever
touched on the GUI thread.** A worker therefore never receives a widget and never calls
back into one. It emits signals; Qt delivers them to the GUI thread; the window updates
itself there. The worker's whole knowledge of the UI is three signals.

:class:`StepWorker` deliberately holds no reference to the window that started it.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal, Slot

from app.ui.tasks import StepId, StepResult, run_step
from app.utils.logging import get_logger

logger = get_logger(__name__)


class WorkerSignals(QObject):
    """The only channel from a worker back to the window.

    A separate ``QObject`` because :class:`QRunnable` is not one and cannot carry signals
    itself — a Qt detail, not a design choice.
    """

    progress = Signal(float, str)
    finished = Signal(object)
    """Carries a :class:`~app.ui.tasks.StepResult`."""
    failed = Signal(str, str)
    """``(headline, detail)``. Split so the window can show one and offer the other."""


class StepWorker(QRunnable):
    """Runs one pipeline step on a thread-pool thread."""

    def __init__(self, step_id: StepId, root: Path) -> None:
        super().__init__()
        self._step_id = step_id
        self._root = root
        self.signals = WorkerSignals()

    @Slot()
    def run(self) -> None:
        """Execute the step, reporting through signals.

        Catches ``Exception`` broadly and on purpose. This is the top of a thread: an
        uncaught exception here would be printed to stderr by the interpreter and the
        window would wait forever for a ``finished`` that is never coming. Every failure
        has to become a signal.
        """
        try:
            result = run_step(
                self._step_id,
                self._root,
                on_progress=lambda fraction, message: self.signals.progress.emit(fraction, message),
            )
        except Exception as exc:
            logger.exception("Step %s failed", self._step_id.value)
            self.signals.failed.emit(f"{type(exc).__name__}: {exc}", _traceback())
            return
        self.signals.finished.emit(result)


def _traceback() -> str:
    import traceback

    return traceback.format_exc()


class StepRunner(QObject):
    """Owns the thread pool and enforces one step at a time.

    Serialised deliberately. The steps have real dependencies — a brief needs both
    analyses, a render needs a placed plan — and letting a user start a render while the
    footage analysis that will invalidate it is still running produces a video from a plan
    that no longer matches the project.
    """

    progress = Signal(float, str)
    finished = Signal(object)
    failed = Signal(str, str)
    busy_changed = Signal(bool)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._pool = QThreadPool(self)
        self._pool.setMaxThreadCount(1)
        self._busy = False

    @property
    def is_busy(self) -> bool:
        return self._busy

    def start(self, step_id: StepId, root: Path) -> bool:
        """Start a step. Returns ``False`` when one is already running."""
        if self._busy:
            logger.debug("Ignoring %s: a step is already running", step_id.value)
            return False

        worker = StepWorker(step_id, root)
        worker.signals.progress.connect(self.progress)
        worker.signals.finished.connect(self._on_finished)
        worker.signals.failed.connect(self._on_failed)

        self._set_busy(True)
        self._pool.start(worker)
        return True

    def wait(self, timeout_ms: int = 30_000) -> bool:
        """Block until the running step finishes. For tests and for a clean shutdown."""
        return self._pool.waitForDone(timeout_ms)

    @Slot(object)
    def _on_finished(self, result: StepResult) -> None:
        self._set_busy(False)
        self.finished.emit(result)

    @Slot(str, str)
    def _on_failed(self, headline: str, detail: str) -> None:
        self._set_busy(False)
        self.failed.emit(headline, detail)

    def _set_busy(self, value: bool) -> None:
        self._busy = value
        self.busy_changed.emit(value)


__all__ = ["StepRunner", "StepWorker", "WorkerSignals"]
