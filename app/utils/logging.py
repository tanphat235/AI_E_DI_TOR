"""Structured logging.

One rule dominates this module: **nothing here may ever write to stdout.**

stdout is a machine-readable contract consumed by the AI director. A single stray
progress bar on it turns a parseable JSON document into a parse error, and the
agent has no way to tell the difference between that and a real failure. So the
Rich console is constructed with ``stderr=True``, and the sanctioned way to emit
data is :mod:`app.cli.output`.

Each invocation gets a ``run_id`` bound to every record, so one render's logs can
be isolated from a day's worth of them.
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Final

from rich.console import Console
from rich.logging import RichHandler

from app.config.settings import LogLevel

LOGGER_NAME: Final = "aive"

_RESERVED_RECORD_KEYS: Final = frozenset(
    {
        "args",
        "asctime",
        "created",
        "exc_info",
        "exc_text",
        "filename",
        "funcName",
        "levelname",
        "levelno",
        "lineno",
        "message",
        "module",
        "msecs",
        "msg",
        "name",
        "pathname",
        "process",
        "processName",
        "relativeCreated",
        "stack_info",
        "taskName",
        "thread",
        "threadName",
    }
)
"""Attributes the stdlib puts on every record. Anything else was added by us and
is therefore structured context worth serialising."""


def new_run_id() -> str:
    """A short, sortable identifier for one invocation.

    Time prefix then random suffix: logs sort chronologically by filename, and two
    runs started in the same second still cannot collide.
    """
    return f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"


def stderr_console() -> Console:
    """The one console AIVE is allowed to print human-facing output to.

    Markup and highlighting are off. Our messages routinely contain square
    brackets - ``pip install -e ".[speech]"``, ``clips[3]`` - and Rich would parse
    those as style tags and silently delete them. Anything that genuinely needs
    styling should pass ``style=`` explicitly.
    """
    return Console(stderr=True, soft_wrap=False, markup=False, highlight=False)


class RunIdFilter(logging.Filter):
    """Attaches the current ``run_id`` to every record."""

    def __init__(self, run_id: str) -> None:
        super().__init__()
        self.run_id = run_id

    def filter(self, record: logging.LogRecord) -> bool:
        record.run_id = self.run_id
        return True


class JsonLinesFormatter(logging.Formatter):
    """Formats records as one JSON object per line.

    Machine-greppable post-mortems: ``jq 'select(.level=="ERROR")'`` over a render
    beats scrolling a wall of console text. Extra keyword context passed to a log
    call is preserved rather than flattened into the message.
    """

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(record.created))
            + f".{int(record.msecs):03d}",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        run_id = getattr(record, "run_id", None)
        if run_id is not None:
            payload["run_id"] = run_id
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        extras = {
            key: _jsonable(value)
            for key, value in record.__dict__.items()
            if key not in _RESERVED_RECORD_KEYS and key not in {"run_id"}
        }
        if extras:
            payload["context"] = extras
        return json.dumps(payload, ensure_ascii=False, default=str)


def _jsonable(value: Any) -> Any:
    """Coerce a context value into something ``json.dumps`` accepts."""
    if isinstance(value, str | int | float | bool | type(None)):
        return value
    if isinstance(value, Path):
        return value.as_posix()
    return str(value)


def configure_logging(
    *,
    level: LogLevel = LogLevel.INFO,
    run_id: str | None = None,
    log_file: Path | None = None,
) -> str:
    """Install AIVE's handlers and return the ``run_id`` in use.

    Idempotent: existing handlers on the ``aive`` logger are removed first, so
    repeated calls in a long-lived process (the desktop UI) do not duplicate every
    line.

    Args:
        level: Console verbosity. The JSON file always records DEBUG and up,
            because the cost of a verbose file is nothing next to re-running an
            hour-long render to reproduce a bug.
        run_id: Reuse an existing id; a new one is generated when omitted.
        log_file: Destination for JSON-lines output. Skipped when ``None`` or when
            the path cannot be created.
    """
    resolved_run_id = run_id or new_run_id()

    logger = logging.getLogger(LOGGER_NAME)
    for handler in tuple(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
    for existing in tuple(logger.filters):
        logger.removeFilter(existing)

    logger.setLevel(logging.DEBUG)
    # Do not let records reach the root logger, whose default handler writes to
    # stderr with a different format - and, if anyone calls basicConfig, possibly
    # to stdout.
    logger.propagate = False

    # The run-id filter is attached to the *handlers*, not to the logger. A filter
    # on a logger only sees records logged directly to it; records that propagate
    # up from a child such as `aive.services.paths` bypass it entirely and would
    # arrive with no run_id. Handler filters see everything that reaches them.
    run_id_filter = RunIdFilter(resolved_run_id)

    console_handler = RichHandler(
        console=stderr_console(),
        show_path=False,
        rich_tracebacks=True,
        markup=False,
        omit_repeated_times=False,
    )
    console_handler.setLevel(getattr(logging, level.value))
    console_handler.addFilter(run_id_filter)
    logger.addHandler(console_handler)

    if log_file is not None:
        try:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file, encoding="utf-8")
        except OSError as exc:
            # A missing log destination must never abort the actual work.
            logger.warning("Could not open log file %s: %s", log_file, exc)
        else:
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(JsonLinesFormatter())
            file_handler.addFilter(run_id_filter)
            logger.addHandler(file_handler)

    return resolved_run_id


def get_logger(name: str | None = None) -> logging.Logger:
    """A child of the ``aive`` logger.

    Always use this rather than ``logging.getLogger(__name__)``, so every record
    inherits the stderr-only handlers and the ``run_id``.
    """
    if name is None:
        return logging.getLogger(LOGGER_NAME)
    suffix = name.removeprefix("app.").removeprefix(f"{LOGGER_NAME}.")
    return logging.getLogger(f"{LOGGER_NAME}.{suffix}")


@contextmanager
def stage(logger: logging.Logger, description: str, **context: Any) -> Iterator[None]:
    """Log the start, duration and outcome of a unit of work.

    Video work is slow and mostly opaque. Knowing that scene detection took four
    minutes and vision tagging took forty seconds is the difference between
    optimising the right thing and guessing.
    """
    started = time.perf_counter()
    logger.info("%s: start", description, extra=context)
    try:
        yield
    except Exception:
        elapsed = time.perf_counter() - started
        logger.exception("%s: failed after %.2fs", description, elapsed, extra=context)
        raise
    else:
        elapsed = time.perf_counter() - started
        logger.info(
            "%s: done in %.2fs", description, elapsed, extra={**context, "elapsed": elapsed}
        )


def is_quiet() -> bool:
    """True when the caller asked for no human-facing console output.

    Honours ``AIVE_QUIET`` so an agent can silence the console without also
    silencing the JSON log file.
    """
    return os.environ.get("AIVE_QUIET", "").strip().lower() in {"1", "true", "yes"}


__all__ = [
    "LOGGER_NAME",
    "JsonLinesFormatter",
    "RunIdFilter",
    "configure_logging",
    "get_logger",
    "is_quiet",
    "new_run_id",
    "stage",
    "stderr_console",
]
