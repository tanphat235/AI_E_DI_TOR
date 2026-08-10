"""Building subtitle cues from a transcript.

This is the module that owns the single easiest thing to get wrong in the whole
project: **cue times are timeline time, transcript times are source time.** Cleanup
removes silence and filler, so a word spoken at 41.2 s in ``narration.wav`` lands at
33.8 s in the finished video. Every cue leaving here has already been mapped.

The mapping is done **per word**, not per segment, which is what makes it correct
rather than approximately correct. A sentence that straddles a removed pause has to
become two cues, and only word-level positions reveal where the seam is. When a
transcript has no word timings the builder falls back to clamping segments against the
kept ranges — cruder, and the reason ``speech.word_timestamps`` defaults to true.

Line wrapping is the other half. Two lines of 42 characters is roughly the most a
viewer reads comfortably while also watching the picture; anything longer gets split
into consecutive cues rather than shrunk.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.config.settings import SubtitleSettings
from app.models.common import TimeRange, map_to_timeline
from app.models.edit_plan import SubtitleCue
from app.models.speech import Transcript, TranscriptSegment, Word
from app.utils.logging import get_logger

logger = get_logger(__name__)

BUILDER_VERSION = "subtitles/1"

_MAX_GAP_WITHIN_CUE = 0.6
"""A pause longer than this inside one sentence starts a new cue.

Without it, a cue whose middle was cut would stretch across the join and sit on screen
during unrelated footage. Six-tenths of a second is long enough to survive ordinary
speech rhythm and short enough to catch a real edit.
"""


@dataclass(frozen=True, slots=True)
class _TimedRun:
    """A run of words that will become one cue, already in timeline time."""

    words: tuple[Word, ...]
    timeline: tuple[tuple[float, float], ...]

    @property
    def text(self) -> str:
        return " ".join(word.text for word in self.words)

    @property
    def range(self) -> TimeRange:
        return TimeRange(start=self.timeline[0][0], end=self.timeline[-1][1])


class SubtitleBuilder:
    """Turns a transcript into timeline-timed, line-wrapped cues."""

    def __init__(self, settings: SubtitleSettings) -> None:
        self._settings = settings

    def build(
        self,
        transcript: Transcript,
        *,
        kept_ranges: tuple[TimeRange, ...] | None = None,
        include_word_timings: bool = True,
    ) -> tuple[SubtitleCue, ...]:
        """Build cues for ``transcript``.

        Args:
            transcript: The recognition result.
            kept_ranges: Narration ranges surviving cleanup, in source time. ``None``
                means nothing was removed, so source time *is* timeline time.
            include_word_timings: Attach per-word karaoke timing. Only the ASS writer
                consumes it, so a caller emitting SRT alone can skip the work.
        """
        if transcript.has_word_timings:
            cues = self._build_from_words(
                transcript,
                kept_ranges=kept_ranges,
                include_word_timings=include_word_timings,
            )
        else:
            logger.warning(
                "Transcript has no word timings; subtitle timing will be approximate. "
                "Enable speech.word_timestamps for exact cues."
            )
            cues = self._build_from_segments(transcript, kept_ranges=kept_ranges)

        return self._enforce_durations(cues)

    # -- Word-level path (the normal one) ----------------------------------- #

    def _build_from_words(
        self,
        transcript: Transcript,
        *,
        kept_ranges: tuple[TimeRange, ...] | None,
        include_word_timings: bool,
    ) -> list[SubtitleCue]:
        cues: list[SubtitleCue] = []
        for segment in transcript.segments:
            for run in self._runs_for_segment(segment, kept_ranges):
                cues.extend(self._wrap_run(run, include_word_timings=include_word_timings))
        return cues

    def _runs_for_segment(
        self,
        segment: TranscriptSegment,
        kept_ranges: tuple[TimeRange, ...] | None,
    ) -> list[_TimedRun]:
        """Map a segment's words to timeline time, splitting where the timeline jumps.

        Words falling inside a removed gap simply vanish, and a discontinuity in the
        surviving positions is where the cue must be cut in two.
        """
        words: list[Word] = []
        stamps: list[tuple[float, float]] = []
        runs: list[_TimedRun] = []

        def flush() -> None:
            if words:
                runs.append(_TimedRun(words=tuple(words), timeline=tuple(stamps)))
            words.clear()
            stamps.clear()

        for word in segment.words:
            mapped = self._map_word(word, kept_ranges)
            if mapped is None:
                # The word was cut. Its neighbours may still be contiguous, so this is
                # not automatically a split - the gap check below decides.
                continue
            if stamps and mapped[0] - stamps[-1][1] > _MAX_GAP_WITHIN_CUE:
                flush()
            words.append(word)
            stamps.append(mapped)

        flush()
        return runs

    @staticmethod
    def _map_word(
        word: Word, kept_ranges: tuple[TimeRange, ...] | None
    ) -> tuple[float, float] | None:
        """A word's timeline start and end, or ``None`` if it was cut."""
        if kept_ranges is None:
            return word.start, word.end

        start = map_to_timeline(word.start, kept_ranges)
        if start is None:
            # The word's onset was cut. Probing its middle would let a half-cut word
            # produce a cue, which reads as a stutter on screen; drop it instead.
            return None
        # Preserve the word's own duration rather than mapping its end separately: a
        # word never spans a cut, so its length is unchanged by one.
        return start, start + (word.end - word.start)

    # -- Segment fallback --------------------------------------------------- #

    def _build_from_segments(
        self,
        transcript: Transcript,
        *,
        kept_ranges: tuple[TimeRange, ...] | None,
    ) -> list[SubtitleCue]:
        cues: list[SubtitleCue] = []
        for segment in transcript.segments:
            span = self._map_segment(segment.range, kept_ranges)
            if span is None:
                continue
            for line_block in self._wrap_text(segment.text):
                cues.append(SubtitleCue(range=span, text=line_block))
                # Every block of a multi-block segment would share one range, which is
                # wrong; without word timings there is nothing better, so emit the
                # first and drop the rest rather than stack duplicates on screen.
                break
        return cues

    @staticmethod
    def _map_segment(
        span: TimeRange, kept_ranges: tuple[TimeRange, ...] | None
    ) -> TimeRange | None:
        if kept_ranges is None:
            return span
        start = map_to_timeline(span.start, kept_ranges)
        if start is None:
            return None
        for kept in kept_ranges:
            overlap = kept.intersection(span)
            if overlap is not None:
                return TimeRange(start=start, end=start + overlap.duration)
        return None

    # -- Wrapping ----------------------------------------------------------- #

    def _wrap_run(self, run: _TimedRun, *, include_word_timings: bool) -> list[SubtitleCue]:
        """Split one run into cues that fit the line budget."""
        settings = self._settings
        budget = settings.max_chars_per_line * settings.max_lines

        chunks = _chunk_by_budget(list(zip(run.words, run.timeline, strict=True)), budget)
        cues: list[SubtitleCue] = []
        for chunk in chunks:
            words = tuple(word for word, _ in chunk)
            stamps = [stamp for _, stamp in chunk]
            text = "\n".join(self._lines(" ".join(word.text for word in words)))
            karaoke: tuple[tuple[str, float, float], ...] = ()
            if include_word_timings:
                karaoke = tuple(
                    (word.text, stamp[0], stamp[1])
                    for word, stamp in zip(words, stamps, strict=True)
                )
            cues.append(
                SubtitleCue(
                    range=TimeRange(start=stamps[0][0], end=stamps[-1][1]),
                    text=text,
                    words=karaoke,
                )
            )
        return cues

    def _wrap_text(self, text: str) -> list[str]:
        """Wrap plain text into blocks of at most ``max_lines`` lines."""
        lines = self._lines(text)
        limit = self._settings.max_lines
        return ["\n".join(lines[index : index + limit]) for index in range(0, len(lines), limit)]

    def _lines(self, text: str) -> list[str]:
        """Greedy word wrap at ``max_chars_per_line``.

        Greedy rather than balanced. A balanced split reads marginally better but
        changes line breaks as text is edited, which makes subtitle diffs unreadable;
        predictability wins.
        """
        width = self._settings.max_chars_per_line
        lines: list[str] = []
        current = ""
        for word in text.split():
            candidate = f"{current} {word}".strip()
            if current and len(candidate) > width:
                lines.append(current)
                current = word
            else:
                current = candidate
        if current:
            lines.append(current)
        return lines or [text]

    # -- Duration hygiene --------------------------------------------------- #

    def _enforce_durations(self, cues: list[SubtitleCue]) -> tuple[SubtitleCue, ...]:
        """Apply the minimum and maximum on-screen time, and keep cues from colliding.

        Order matters: extend short cues first, then trim overlaps. Doing it the other
        way lets an extension re-create the overlap that was just removed.
        """
        settings = self._settings
        if not cues:
            return ()

        ordered = sorted(cues, key=lambda cue: cue.range.start)
        adjusted: list[SubtitleCue] = []

        for cue in ordered:
            start, end = cue.range.start, cue.range.end
            if end - start < settings.min_cue_duration:
                end = start + settings.min_cue_duration
            if end - start > settings.max_cue_duration:
                end = start + settings.max_cue_duration
            adjusted.append(cue.model_copy(update={"range": TimeRange(start=start, end=end)}))

        result: list[SubtitleCue] = []
        for cue in adjusted:
            if result:
                previous = result[-1]
                latest_end = cue.range.start - settings.cue_gap
                if previous.range.end > latest_end:
                    if latest_end <= previous.range.start:
                        # No room to shorten the previous cue without inverting it. Two
                        # cues this close are one utterance the recogniser split, so
                        # dropping this one is better than a one-frame flash.
                        continue
                    result[-1] = previous.model_copy(
                        update={"range": TimeRange(start=previous.range.start, end=latest_end)}
                    )
            result.append(cue)

        return tuple(result)


def _chunk_by_budget(
    items: list[tuple[Word, tuple[float, float]]], budget: int
) -> list[list[tuple[Word, tuple[float, float]]]]:
    """Group words into chunks whose joined text fits ``budget`` characters."""
    if not items:
        return []
    chunks: list[list[tuple[Word, tuple[float, float]]]] = [[]]
    length = 0
    for item in items:
        word_length = len(item[0].text) + (1 if length else 0)
        if length and length + word_length > budget:
            chunks.append([])
            length = 0
            word_length = len(item[0].text)
        chunks[-1].append(item)
        length += word_length
    return [chunk for chunk in chunks if chunk]


__all__ = ["BUILDER_VERSION", "SubtitleBuilder"]
