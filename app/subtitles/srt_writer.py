"""SRT output.

Hand-rolled rather than delegated to pysubs2. SRT is four lines per cue and a
timestamp format; writing it directly costs nothing, keeps the dependency optional
rather than required, and — the actual reason — gives exact control over the two
things players disagree about: the comma decimal separator and CRLF line endings.

UTF-8 with a BOM. Without it, several Windows players still guess the system codepage
and render Vietnamese narration as mojibake.
"""

from __future__ import annotations

from pathlib import Path

from app.config.settings import SubtitleSettings
from app.models.common import SubtitleFormat
from app.models.edit_plan import SubtitleCue
from app.utils.logging import get_logger

logger = get_logger(__name__)


class SrtWriter:
    """A :class:`~app.subtitles.base.SubtitleWriter` producing SubRip files."""

    @property
    def format(self) -> SubtitleFormat:
        return SubtitleFormat.SRT

    @property
    def supports_word_timings(self) -> bool:
        """SRT has no karaoke syntax, so word timings are dropped."""
        return False

    def write(
        self,
        cues: tuple[SubtitleCue, ...],
        destination: Path,
        *,
        style: SubtitleSettings,
    ) -> Path:
        """Write ``cues`` as SRT. ``style`` is unused: SRT carries no styling."""
        destination.parent.mkdir(parents=True, exist_ok=True)
        blocks = [
            "\r\n".join(
                (
                    str(number),
                    f"{format_timestamp(cue.range.start)} --> {format_timestamp(cue.range.end)}",
                    cue.text.replace("\n", "\r\n"),
                )
            )
            for number, cue in enumerate(cues, start=1)
        ]
        # A trailing blank line after the final cue; some parsers need the terminator.
        body = "\r\n\r\n".join(blocks) + "\r\n\r\n" if blocks else ""
        destination.write_text(body, encoding="utf-8-sig", newline="")
        logger.debug("Wrote %d SRT cue(s) to %s", len(cues), destination)
        return destination


def format_timestamp(seconds: float) -> str:
    """Format seconds as ``HH:MM:SS,mmm``.

    Milliseconds are truncated, not rounded: rounding up can push a cue's end past the
    next cue's start and produce an overlap the builder deliberately removed.
    """
    clamped = max(0.0, seconds)
    total_ms = int(clamped * 1000)
    hours, remainder = divmod(total_ms, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


__all__ = ["SrtWriter", "format_timestamp"]
