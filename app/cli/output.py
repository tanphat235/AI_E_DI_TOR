"""stdout discipline - the mechanics of AIVE's contract with the AI director.

Three rules, and this module is the only sanctioned way to satisfy them.

**1. stdout is machine-readable only.** Everything a human reads - logs, tables,
progress, warnings - goes to stderr. The director parses stdout, and a stray
progress bar there is indistinguishable from a malformed result.

**2. Full data to disk, a digest to stdout.** Analysis output is written whole to
``<project>/.aive/`` and summarised to stdout. This is not a nicety: a twenty-minute
project with two hundred scenes is roughly 80k tokens as raw JSON and about 12k as a
digest. The director pays for every token it reads, so the default must be the
digest and ``--full`` must be the opt-in.

**3. Errors are structured.** A non-zero exit plus a JSON error object carrying a
stable ``code`` and an actionable ``hint``, so the agent can correct itself instead
of pattern-matching a traceback.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Iterable, Mapping
from enum import StrEnum
from pathlib import Path
from typing import Any, Final, NoReturn

from pydantic import BaseModel

from app.utils.logging import is_quiet, stderr_console

JSON_INDENT: Final = 2
"""Indent for stdout JSON.

Indented rather than compact: an indented document costs a few percent more tokens
but is far easier for a human to read in a terminal, and diffs cleanly when a
director writes a plan to disk and revises it.
"""


class ExitCode(StrEnum):
    """Stable exit codes.

    A string enum with integer values attached, so both the shell and the agent get
    something meaningful. Distinct codes let the director branch without parsing:
    a missing project is recoverable by running ``init``, a validation failure is
    recoverable by rewriting the plan, an environment failure is not recoverable at
    all and should stop the workflow.
    """

    OK = "ok"
    USAGE = "usage"
    NOT_FOUND = "not_found"
    INVALID_INPUT = "invalid_input"
    ENVIRONMENT = "environment"
    INTERNAL = "internal"

    @property
    def status(self) -> int:
        return _EXIT_STATUS[self]


_EXIT_STATUS: Final[dict[ExitCode, int]] = {
    ExitCode.OK: 0,
    ExitCode.USAGE: 2,
    ExitCode.NOT_FOUND: 3,
    ExitCode.INVALID_INPUT: 4,
    ExitCode.ENVIRONMENT: 5,
    ExitCode.INTERNAL: 70,
}


def emit_json(payload: Any) -> None:
    """Write one JSON document to stdout.

    Accepts a pydantic model, a mapping, or anything ``json`` can serialise. Models
    are dumped in JSON mode so that enums become strings and ``MediaRef`` paths come
    out POSIX-style, matching what a plan file on disk looks like.
    """
    if isinstance(payload, BaseModel):
        text = payload.model_dump_json(indent=JSON_INDENT)
    else:
        text = json.dumps(payload, indent=JSON_INDENT, ensure_ascii=False, default=_fallback)
    sys.stdout.write(text + "\n")
    sys.stdout.flush()


def emit_digest(lines: Iterable[str]) -> None:
    """Write a compact, line-oriented summary to stdout.

    One record per line, fields separated by whitespace. Chosen over JSON for
    digests because it is roughly a third of the tokens for the same information and
    stays readable to both a human and a language model.
    """
    text = "\n".join(lines)
    if text:
        sys.stdout.write(text + "\n")
        sys.stdout.flush()


def emit_error(
    code: str,
    message: str,
    *,
    hint: str | None = None,
    exit_code: ExitCode = ExitCode.INVALID_INPUT,
    details: Mapping[str, Any] | None = None,
) -> NoReturn:
    """Write a structured error to stdout and exit non-zero.

    On stdout, not stderr, and deliberately so. The director is reading stdout; an
    error object there means one place to look for both success and failure. The
    human-readable rendering of the same failure still goes to stderr via the
    logger.

    Args:
        code: Stable slug the agent may branch on, e.g. ``project.not_initialised``.
        message: What went wrong.
        hint: What to do about it. Almost always worth supplying.
        exit_code: Which class of failure this is.
        details: Extra machine-readable context.
    """
    payload: dict[str, Any] = {"error": {"code": code, "message": message}}
    if hint is not None:
        payload["error"]["hint"] = hint
    if details:
        # Pass values through untouched and let `emit_json`'s serialiser hook coerce
        # only what it must. Mapping `_fallback` over everything eagerly would turn
        # an int into a string, and the agent may well be comparing it numerically.
        payload["error"]["details"] = dict(details)
    emit_json(payload)
    raise SystemExit(exit_code.status)


def note(message: str) -> None:
    """Write a human-facing line to **stderr**.

    Use for the conversational output a person wants and a parser must not see.
    Suppressed by ``AIVE_QUIET``.
    """
    if not is_quiet():
        stderr_console().print(message)


def write_json_file(payload: Any, destination: Path) -> Path:
    """Write a JSON document to disk, creating parent directories as needed.

    This is the "full data" half of rule 2. Returns the path so a command can
    report where the complete result landed while printing only a digest.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(payload, BaseModel):
        text = payload.model_dump_json(indent=JSON_INDENT)
    else:
        text = json.dumps(payload, indent=JSON_INDENT, ensure_ascii=False, default=_fallback)
    destination.write_text(text + "\n", encoding="utf-8")
    return destination


def _fallback(value: Any) -> Any:
    """Serialise the types ``json`` does not handle natively."""
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, StrEnum):
        return value.value
    if isinstance(value, set | frozenset | tuple):
        return list(value)
    return str(value)


__all__ = [
    "JSON_INDENT",
    "ExitCode",
    "emit_digest",
    "emit_error",
    "emit_json",
    "note",
    "write_json_file",
]
