"""Choosing a subtitle writer by format.

A function rather than a mutable registry class. Unlike exporters — where third
parties will plausibly add formats and each needs isolating — the subtitle formats
worth supporting are a closed set of two, and a lookup table says that plainly.
"""

from __future__ import annotations

from app.config.settings import SubtitleSettings
from app.models.common import SubtitleFormat
from app.subtitles.ass_writer import AssWriter
from app.subtitles.base import SubtitleWriter
from app.subtitles.srt_writer import SrtWriter


def writer_for(subtitle_format: SubtitleFormat, settings: SubtitleSettings) -> SubtitleWriter:
    """The writer for ``subtitle_format``.

    ``settings`` is taken here rather than at write time because a writer's *behaviour*
    can depend on config — the ASS writer needs to know whether karaoke is wanted
    before it is handed any cues.
    """
    match subtitle_format:
        case SubtitleFormat.SRT:
            return SrtWriter()
        case SubtitleFormat.ASS:
            return AssWriter(karaoke=settings.karaoke)


def available_formats() -> tuple[SubtitleFormat, ...]:
    """Every format that can be written."""
    return tuple(SubtitleFormat)


__all__ = ["available_formats", "writer_for"]
