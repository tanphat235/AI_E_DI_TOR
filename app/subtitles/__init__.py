"""Subtitle generation (Phase 2).

Cue *building* maps a transcript onto timeline time and wraps it into readable lines;
cue *writing* serialises the result. They are separate because the hard part is the
mapping, and it is worth testing without a file format in the way.
"""

from __future__ import annotations

from app.subtitles.base import SubtitleWriter
from app.subtitles.builder import SubtitleBuilder
from app.subtitles.registry import available_formats, writer_for

__all__ = ["SubtitleBuilder", "SubtitleWriter", "available_formats", "writer_for"]
