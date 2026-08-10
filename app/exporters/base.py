"""Exporter boundary and registry (CapCut lands in Phase 8).

An exporter converts an :class:`~app.models.edit_plan.EditPlan` into another
editor's project format, so the user can carry on editing by hand. Like a renderer,
it consumes only the plan.

The registry exists because exporters are the part of AIVE most likely to
proliferate and to break through no fault of ours. CapCut's ``draft_content.json``
is undocumented and changes between versions; Premiere XML, DaVinci Resolve, Final
Cut XML and EDL are all plausible additions. Registration by name keeps each one
isolated: a CapCut format change is repaired in one file, and a new exporter is
added without editing anything that already works.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

from app.models.edit_plan import EditPlan


@dataclass(frozen=True, slots=True)
class ExportRequest:
    """Everything an exporter needs beyond the plan."""

    plan: EditPlan
    project_root: Path
    destination: Path
    """Where to write. For CapCut this is the draft directory to create."""

    template: Path | None = None
    """An existing project to clone, so the user's styling and settings survive.

    Preferred over generating a project from scratch: a template carries fonts,
    colour settings and canvas configuration that no exporter can reliably invent.
    """

    copy_media: bool = True
    """Copy media into the exported project rather than referencing it in place.

    Costs disk, but a project that references media outside itself breaks silently
    the moment the user reorganises their footage.
    """

    project_name: str | None = None


@dataclass(frozen=True, slots=True)
class ExportResult:
    """What an export produced."""

    project_dir: Path
    files_written: tuple[Path, ...] = ()
    media_copied: tuple[Path, ...] = ()
    warnings: tuple[str, ...] = field(default=())
    open_hint: str | None = None
    """How the user opens the result, e.g. "restart CapCut and look under Drafts"."""


@runtime_checkable
class Exporter(Protocol):
    """Converts an Edit Plan into another editor's project format."""

    @property
    def name(self) -> str:
        """Registry key, e.g. ``capcut``."""
        ...

    @property
    def display_name(self) -> str:
        """Human-facing label, e.g. ``CapCut Desktop``."""
        ...

    def preflight(self, request: ExportRequest) -> tuple[str, ...]:
        """Check the request without writing anything.

        Returns blocking problems; empty means good to go. For CapCut this is where
        an unrecognised template version is caught, before a half-written draft
        makes the user's project list unusable.
        """
        ...

    def export(self, request: ExportRequest) -> ExportResult:
        """Write the project."""
        ...


class ExporterRegistry:
    """Name-to-exporter lookup.

    Deliberately a plain object built at startup rather than an import-time global.
    Import-time registration means importing an exporter has side effects, which
    makes test isolation awkward and hides which exporters a given run actually had.
    """

    def __init__(self) -> None:
        self._exporters: dict[str, Exporter] = {}

    def register(self, exporter: Exporter) -> None:
        """Add an exporter. Raises on a duplicate name rather than overwriting."""
        if exporter.name in self._exporters:
            msg = f"an exporter named {exporter.name!r} is already registered"
            raise ValueError(msg)
        self._exporters[exporter.name] = exporter

    def get(self, name: str) -> Exporter:
        """Look up an exporter by name.

        Raises:
            KeyError: with the available names, because a typo in ``--format`` is
                the likeliest cause and listing the options resolves it instantly.
        """
        try:
            return self._exporters[name]
        except KeyError:
            available = ", ".join(sorted(self._exporters)) or "none registered"
            msg = f"unknown exporter {name!r}; available: {available}"
            raise KeyError(msg) from None

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._exporters))

    def __contains__(self, name: object) -> bool:
        return name in self._exporters

    def __len__(self) -> int:
        return len(self._exporters)


__all__ = [
    "ExportRequest",
    "ExportResult",
    "Exporter",
    "ExporterRegistry",
]
