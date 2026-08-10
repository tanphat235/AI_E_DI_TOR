"""Upgrading Edit Plans written by an older version of AIVE.

:data:`app.models.edit_plan.EDIT_PLAN_SCHEMA_VERSION` is pinned as a ``Literal``, which makes
an old build **refuse** a newer plan rather than misread fields it does not understand. That
is the right failure. But it also means a *newer* build refuses an *older* plan, and a user
whose plan stops loading after an upgrade has lost work.

This is the other half of that contract, and it is built now, while there is exactly one
version and therefore nothing to migrate. Retrofitting a migration path after plans exist in
the wild means guessing what those plans contain.

**Migrations run on raw dictionaries, before validation.** They have to: the models describe
the *current* shape, so a 1.0 document cannot be loaded as a model in order to be upgraded to
2.0. Each migration takes the dict one version forward, and they compose in sequence.

To add one, when ``1.0`` becomes ``1.1``::

    @register("1.0", "1.1")
    def _add_colour_grade(document: dict[str, Any]) -> dict[str, Any]:
        for clip in document.get("clips", []):
            clip.setdefault("colour_grade", None)
        return document

Nothing else changes. The chain is found automatically.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from app.models.edit_plan import EDIT_PLAN_SCHEMA_VERSION
from app.utils.logging import get_logger

logger = get_logger(__name__)

Migration = Callable[[dict[str, Any]], dict[str, Any]]
"""Takes a plan document one version forward. Must not mutate its input in place."""

_MIGRATIONS: dict[str, tuple[str, Migration]] = {}
"""From-version to (to-version, migration). One step per version, so the chain is linear."""


class PlanMigrationError(RuntimeError):
    """Raised when a plan cannot be brought to the current version."""


def register(from_version: str, to_version: str) -> Callable[[Migration], Migration]:
    """Register a migration from one schema version to the next."""

    def decorator(migration: Migration) -> Migration:
        if from_version in _MIGRATIONS:
            msg = f"a migration from {from_version} is already registered"
            raise ValueError(msg)
        _MIGRATIONS[from_version] = (to_version, migration)
        return migration

    return decorator


def needs_migration(document: dict[str, Any]) -> bool:
    """Whether a raw plan document is from an older schema version."""
    return _version_of(document) != EDIT_PLAN_SCHEMA_VERSION


def migrate(document: dict[str, Any]) -> tuple[dict[str, Any], tuple[str, ...]]:
    """Bring a raw plan document to the current schema version.

    Returns the upgraded document and the chain of versions it passed through, so a caller
    can report what happened rather than silently rewriting the user's file.

    Raises:
        PlanMigrationError: when the document is *newer* than this build understands, or when
            no migration exists for its version. Both are refusals rather than guesses: a
            half-understood plan renders a subtly wrong video, which is worse than not
            rendering.
    """
    current = dict(document)
    version = _version_of(current)
    chain: list[str] = [version]

    # A plan from the future. Guessing at fields we do not know would produce a video that
    # is confidently wrong, so this stops.
    if version not in _MIGRATIONS and version != EDIT_PLAN_SCHEMA_VERSION:
        msg = (
            f"this plan declares schema_version {version!r}, which this build of AIVE does "
            f"not know (it understands {EDIT_PLAN_SCHEMA_VERSION!r} and can upgrade "
            f"{sorted(_MIGRATIONS) or 'nothing'}). Upgrade AIVE, or re-author the plan."
        )
        raise PlanMigrationError(msg)

    # Bounded by the number of registered migrations: a cycle would otherwise hang.
    for _step in range(len(_MIGRATIONS) + 1):
        if version == EDIT_PLAN_SCHEMA_VERSION:
            break
        target, migration = _MIGRATIONS[version]
        logger.info("Migrating plan from %s to %s", version, target)
        current = migration(dict(current))
        current["schema_version"] = target
        version = target
        chain.append(version)
    else:
        msg = f"migrating this plan did not reach {EDIT_PLAN_SCHEMA_VERSION}; chain was {chain}"
        raise PlanMigrationError(msg)

    return current, tuple(chain)


def _version_of(document: dict[str, Any]) -> str:
    """The declared version, defaulting to the current one.

    A document with no ``schema_version`` is treated as current rather than rejected: the
    field has a default on the model precisely so a hand-authored plan need not carry it.
    """
    raw = document.get("schema_version")
    return EDIT_PLAN_SCHEMA_VERSION if raw is None else str(raw)


def registered_versions() -> tuple[str, ...]:
    """Versions this build can upgrade from."""
    return tuple(sorted(_MIGRATIONS))


__all__ = [
    "Migration",
    "PlanMigrationError",
    "migrate",
    "needs_migration",
    "register",
    "registered_versions",
]
