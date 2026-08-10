"""Qt models over an Edit Plan.

The window shows a plan as a table, and the interesting column is **Reason**. Everything
else — source, timing, framing — a user could reconstruct by watching the video. The reason
is the only place the director's judgement is written down, and putting it on screen beside
the clip is the whole point of having required it in the schema.

The rows are built from :class:`~app.models.plan_review.PlanReview` rather than from the
plan directly, so the table and ``aive plan show`` display the same numbers computed by the
same code. A UI that formats durations its own way is a UI that eventually disagrees with
the CLI about how long the video is.
"""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import QAbstractTableModel, QModelIndex, QObject, QPersistentModelIndex, Qt
from PySide6.QtGui import QColor

from app.models.common import Severity
from app.models.plan_review import PlanReview

Index = QModelIndex | QPersistentModelIndex
"""Qt hands either kind to a model, so an override that names only ``QModelIndex`` is
narrower than the base class and mypy is right to reject it."""

_COLUMNS = ("#", "Clip", "Source", "Timeline", "Length", "In", "Shot", "Reason")

_SEVERITY_COLOURS = {
    Severity.ERROR: QColor(200, 60, 60),
    Severity.WARNING: QColor(200, 140, 40),
    Severity.INFO: QColor(120, 120, 120),
}


class TimelineModel(QAbstractTableModel):
    """One row per clip, with the author's reason in the last column."""

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._review: PlanReview | None = None

    def set_review(self, review: PlanReview | None) -> None:
        """Replace the contents. ``beginResetModel`` because every row changes at once."""
        self.beginResetModel()
        self._review = review
        self.endResetModel()

    @property
    def review(self) -> PlanReview | None:
        return self._review

    def rowCount(self, parent: Index | None = None) -> int:  # noqa: N802 - Qt's name
        if parent is not None and parent.isValid():
            return 0
        return len(self._review.timeline) if self._review else 0

    def columnCount(self, parent: Index | None = None) -> int:  # noqa: N802 - Qt's name
        if parent is not None and parent.isValid():
            return 0
        return len(_COLUMNS)

    def headerData(  # noqa: N802 - Qt's name
        self, section: int, orientation: Qt.Orientation, role: int = Qt.ItemDataRole.DisplayRole
    ) -> Any:
        if role != Qt.ItemDataRole.DisplayRole or orientation != Qt.Orientation.Horizontal:
            return None
        return _COLUMNS[section] if 0 <= section < len(_COLUMNS) else None

    def data(self, index: Index, role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        if not index.isValid() or self._review is None:
            return None
        entry = self._review.timeline[index.row()]

        if role == Qt.ItemDataRole.ToolTipRole:
            # The full reason, because the column will be elided and the reason is the one
            # thing a reviewer must be able to read in full.
            return entry.reason

        if role != Qt.ItemDataRole.DisplayRole:
            return None

        timeline = (
            f"{entry.timeline_range.start:.2f}"
            if entry.timeline_range is not None
            # Said plainly rather than shown as a blank cell: an unplaced plan is a state
            # with a fix, and the window's status bar names the step that fixes it.
            else "unplaced"
        )
        transition = (
            f"{entry.transition_in.value} {entry.transition_duration:.2f}s"
            if entry.transition_duration > 0.0
            else entry.transition_in.value
        )
        return (
            str(entry.order),
            entry.clip_id,
            f"{entry.source} {entry.source_range.start:.2f}-{entry.source_range.end:.2f}",
            timeline,
            f"{entry.duration:.2f}s",
            transition,
            entry.shot_type.value,
            entry.reason,
        )[index.column()]


class NotesModel(QAbstractTableModel):
    """The review's editorial notes, colour-coded by severity.

    Kept as a model rather than concatenated into a label so the severity colour is data
    rather than markup, and so a long list scrolls instead of pushing the timeline off screen.
    """

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._review: PlanReview | None = None

    def set_review(self, review: PlanReview | None) -> None:
        self.beginResetModel()
        self._review = review
        self.endResetModel()

    def rowCount(self, parent: Index | None = None) -> int:  # noqa: N802 - Qt's name
        if parent is not None and parent.isValid():
            return 0
        return len(self._review.notes) if self._review else 0

    def columnCount(self, parent: Index | None = None) -> int:  # noqa: N802 - Qt's name
        if parent is not None and parent.isValid():
            return 0
        return 2

    def headerData(  # noqa: N802 - Qt's name
        self, section: int, orientation: Qt.Orientation, role: int = Qt.ItemDataRole.DisplayRole
    ) -> Any:
        if role != Qt.ItemDataRole.DisplayRole or orientation != Qt.Orientation.Horizontal:
            return None
        return ("Note", "What to do")[section]

    def data(self, index: Index, role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        if not index.isValid() or self._review is None:
            return None
        issue = self._review.notes[index.row()]

        if role == Qt.ItemDataRole.ForegroundRole:
            return _SEVERITY_COLOURS.get(issue.severity)
        if role == Qt.ItemDataRole.ToolTipRole:
            return issue.code
        if role != Qt.ItemDataRole.DisplayRole:
            return None
        return (issue.message, issue.hint or "")[index.column()]


__all__ = ["NotesModel", "TimelineModel"]
