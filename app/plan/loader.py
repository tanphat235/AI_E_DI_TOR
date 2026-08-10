"""Loading an Edit Plan from disk, with migration.

One function, and it is the only place a plan should be read. It exists because loading a
plan is not just ``model_validate_json``: a document written by an older AIVE has to be
migrated first, and doing that at every call site would mean forgetting it at one of them.

The migration is reported rather than silent. A user whose file was upgraded should be told,
because the version on disk is now behind what they are working with.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from app.models.edit_plan import EditPlan
from app.models.migrations import PlanMigrationError, migrate, needs_migration
from app.utils.logging import get_logger

logger = get_logger(__name__)


class PlanLoadError(RuntimeError):
    """Raised when a plan cannot be read, migrated or validated."""


@dataclass(frozen=True, slots=True)
class LoadedPlan:
    """A plan and what had to be done to load it."""

    plan: EditPlan
    migrated_from: str | None = None
    """The version on disk, when it differed from the current one. ``None`` otherwise."""

    @property
    def was_migrated(self) -> bool:
        return self.migrated_from is not None


def load_plan(path: Path) -> LoadedPlan:
    """Read, migrate and validate a plan.

    Raises:
        PlanLoadError: for a missing file, malformed JSON, an unmigratable version, or a
            document that fails validation. All four are user-fixable, and the message says
            which one happened - "not valid JSON" and "missing a required field" call for
            completely different responses.
    """
    if not path.is_file():
        msg = f"no Edit Plan at {path}"
        raise PlanLoadError(msg)

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        msg = f"could not read {path.name}: {exc}"
        raise PlanLoadError(msg) from exc
    except json.JSONDecodeError as exc:
        msg = f"{path.name} is not valid JSON: {exc}"
        raise PlanLoadError(msg) from exc

    if not isinstance(raw, dict):
        msg = f"{path.name} must contain a JSON object, not {type(raw).__name__}"
        raise PlanLoadError(msg)

    original_version = str(raw.get("schema_version") or "")
    migrated_from: str | None = None
    if needs_migration(raw):
        try:
            raw, chain = migrate(raw)
        except PlanMigrationError as exc:
            raise PlanLoadError(str(exc)) from exc
        migrated_from = original_version or chain[0]
        logger.info("Loaded %s after migrating from %s", path.name, migrated_from)

    try:
        plan = EditPlan.model_validate(raw)
    except ValueError as exc:
        msg = f"{path.name} is not a valid Edit Plan: {exc}"
        raise PlanLoadError(msg) from exc

    return LoadedPlan(plan=plan, migrated_from=migrated_from)


__all__ = ["LoadedPlan", "PlanLoadError", "load_plan"]
