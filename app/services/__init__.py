"""Infrastructure services: filesystem layout, binary discovery, dependency wiring.

These are the pieces that know about the machine AIVE is running on. Everything else is
kept ignorant of it, which is what lets the rest of the codebase be tested without media
files or external binaries.

**This package deliberately does not re-export the DI container.** Import it from
:mod:`app.services.container` directly.

The reason is a circular import that is easy to recreate. Analysis modules legitimately
need ``app.services.paths`` and ``app.services.ffmpeg_locator``; importing either
executes this ``__init__``; and the container imports those same analysis modules to wire
them up. Re-exporting the container here closes that loop, and the failure is nasty
because it depends on import *order*: everything works until some process imports an
analysis module first, at which point it fails with a bare ImportError.

Guarded by ``tests/unit/test_architecture.py::TestEveryModuleImportsStandalone``, which
imports every module in a fresh interpreter so ordering cannot hide it.
"""

from __future__ import annotations

from app.services.ffmpeg_locator import (
    BinarySource,
    FFmpegLocator,
    FFmpegNotFoundError,
    FFmpegTools,
    ResolvedBinary,
)
from app.services.paths import ProjectPaths, classify

__all__ = [
    "BinarySource",
    "FFmpegLocator",
    "FFmpegNotFoundError",
    "FFmpegTools",
    "ProjectPaths",
    "ResolvedBinary",
    "classify",
]
