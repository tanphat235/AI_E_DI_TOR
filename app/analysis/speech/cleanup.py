"""Narration cleanup: deciding which parts of the recording survive.

Three kinds of removal, in increasing order of how much judgement they need and how
badly they can go wrong:

**Silence** is measured, not guessed, and is the safe one. FFmpeg reports the quiet
spans; each is shrunk by a padding margin so a cut never clips the consonant either
side of it.

**Filler words** need word-level timings — without them, excising "um" means losing
the whole segment. They are split into hesitation sounds, which are never meaningful
and are always removed, and discourse markers like "like" and "actually", which are
also ordinary words and therefore need an explicit opt-in.

**Repetitions** are retakes the speaker left in. The *later* occurrence is kept,
because a restart means the speaker is correcting themselves.

Everything here is pure computation over a transcript plus a list of spans, with the
one audio-touching step delegated to a
:class:`~app.analysis.speech.base.SilenceDetector`. That is what lets the whole module
be tested with no audio file: retune a threshold and re-run in milliseconds instead of
re-transcribing.
"""

from __future__ import annotations

import itertools
import unicodedata
from collections import defaultdict
from pathlib import Path

from app.analysis.speech.base import SilenceDetector
from app.config.rules import RuleSettings
from app.models.common import MediaRef, TimeRange, map_to_timeline
from app.models.speech import (
    FillerSpan,
    RepetitionSpan,
    SilenceSpan,
    SpeechCleanupReport,
    Transcript,
    Word,
)
from app.utils.logging import get_logger

logger = get_logger(__name__)

CLEANUP_VERSION = "cleanup/1"

_EDGE_EPSILON = 0.05
"""Tolerance for deciding a silence touches the start or end of the file.

Leading and trailing silence is trimmed flush rather than padded: keeping 80 ms of
room tone before the first word is not breathing room, it is a late start.
"""


class NarrationCleanupService:
    """A :class:`~app.analysis.speech.base.NarrationCleaner` over measured silence."""

    def __init__(self, detector: SilenceDetector, rules: RuleSettings) -> None:
        self._detector = detector
        self._rules = rules

    def clean(self, transcript: Transcript, *, audio: Path) -> SpeechCleanupReport:
        """Propose removals for ``transcript`` and compute what survives."""
        rules = self._rules
        duration = transcript.duration

        silences = self._detector.detect(
            audio,
            threshold_db=rules.silence_threshold_db,
            min_duration=rules.min_silence_duration,
        )
        fillers = self.find_fillers(transcript)
        repetitions = self.find_repetitions(transcript)

        removals = [
            *(self._padded(span, duration) for span in silences),
            *(span.range for span in fillers),
            *(span.range for span in repetitions),
        ]
        kept = self._complement(
            [span for span in removals if span is not None],
            duration=duration,
        )

        logger.info(
            "Cleanup: %d silence, %d filler, %d repetition -> %.2fs of %.2fs kept",
            len(silences),
            len(fillers),
            len(repetitions),
            sum(item.duration for item in kept),
            duration,
        )

        return SpeechCleanupReport(
            source=transcript.source,
            original_duration=duration,
            silences=silences,
            fillers=fillers,
            repetitions=repetitions,
            kept_ranges=kept,
        )

    # -- Silence ------------------------------------------------------------ #

    def _padded(self, span: SilenceSpan, duration: float) -> TimeRange | None:
        """Shrink a silence by the keep-padding, or drop it if nothing is left.

        Padding is applied only on interior edges. A silence that begins at the very
        start of the file, or runs to the very end, is trimmed flush.
        """
        padding = self._rules.silence_keep_padding
        at_start = span.range.start <= _EDGE_EPSILON
        at_end = span.range.end >= duration - _EDGE_EPSILON

        start = span.range.start if at_start else span.range.start + padding
        end = span.range.end if at_end else span.range.end - padding
        if end - start <= 0.0:
            # The whole span was padding, so there is nothing to cut.
            return None
        return TimeRange(start=start, end=min(end, duration))

    # -- Fillers ------------------------------------------------------------ #

    def find_fillers(self, transcript: Transcript) -> tuple[FillerSpan, ...]:
        """Locate filler words and phrases.

        Returns empty when the transcript carries no word timings: there is no safe
        way to remove one word from a segment timed only at segment level, and
        removing the whole segment to lose an "um" is obviously worse than keeping it.
        """
        words = transcript.words()
        if not words:
            if transcript.segments:
                logger.warning(
                    "Skipping filler removal: transcript has no word timings. "
                    "Enable speech.word_timestamps to use it."
                )
            return ()

        rules = self._rules
        phrases: list[tuple[tuple[str, ...], bool]] = [
            (tuple(_normalise(part) for part in phrase.split()), False)
            for phrase in rules.filler_words
        ]
        if rules.remove_ambiguous_fillers:
            phrases.extend(
                (tuple(_normalise(part) for part in phrase.split()), True)
                for phrase in rules.ambiguous_filler_words
            )
        # Longest phrases first, so "you know" wins over a bare "know" would-be match
        # and each word is consumed by the most specific pattern available.
        phrases.sort(key=lambda item: len(item[0]), reverse=True)

        normalised = [_normalise(word.text) for word in words]
        consumed: set[int] = set()
        found: list[FillerSpan] = []

        for tokens, ambiguous in phrases:
            length = len(tokens)
            if length == 0:
                continue
            for index in range(len(normalised) - length + 1):
                window = range(index, index + length)
                if any(position in consumed for position in window):
                    continue
                if tuple(normalised[index : index + length]) != tokens:
                    continue
                consumed.update(window)
                found.append(
                    FillerSpan(
                        range=TimeRange(
                            start=words[index].start,
                            end=words[index + length - 1].end,
                        ),
                        text=" ".join(tokens),
                        ambiguous=ambiguous,
                    )
                )

        found.sort(key=lambda span: span.range.start)
        return tuple(found)

    # -- Repetitions -------------------------------------------------------- #

    def find_repetitions(self, transcript: Transcript) -> tuple[RepetitionSpan, ...]:
        """Locate retakes: the same phrase said twice in quick succession.

        Searches longest phrase first, so "then water it. then water it well."
        contributes one three-word removal rather than three one-word ones.
        """
        words = transcript.words()
        rules = self._rules
        if len(words) < rules.min_repetition_words * 2:
            return ()

        normalised = [_normalise(word.text) for word in words]
        consumed: set[int] = set()
        found: list[RepetitionSpan] = []

        longest = min(rules.max_repetition_words, len(words) // 2)
        for length in range(longest, rules.min_repetition_words - 1, -1):
            buckets: dict[tuple[str, ...], list[int]] = defaultdict(list)
            for index in range(len(normalised) - length + 1):
                buckets[tuple(normalised[index : index + length])].append(index)

            for tokens, positions in buckets.items():
                if len(positions) < 2 or not all(tokens):
                    continue
                for earlier, later in itertools.pairwise(positions):
                    if later - earlier < length:
                        # Overlapping occurrences are one stuttered phrase, not two.
                        continue
                    if _overlaps_consumed(earlier, length, consumed) or _overlaps_consumed(
                        later, length, consumed
                    ):
                        continue
                    gap = words[later].start - words[earlier + length - 1].end
                    if gap > rules.max_repetition_gap:
                        continue
                    # Keep the later take: a restart means the speaker is correcting.
                    consumed.update(range(earlier, earlier + length))
                    found.append(
                        RepetitionSpan(
                            range=TimeRange(
                                start=words[earlier].start,
                                end=words[earlier + length - 1].end,
                            ),
                            text=" ".join(tokens),
                            kept_range=TimeRange(
                                start=words[later].start,
                                end=words[later + length - 1].end,
                            ),
                        )
                    )

        found.sort(key=lambda span: span.range.start)
        return tuple(found)

    # -- Keeping ------------------------------------------------------------ #

    def _complement(self, removals: list[TimeRange], *, duration: float) -> tuple[TimeRange, ...]:
        """Everything in ``[0, duration)`` that is not being removed."""
        merged = merge_ranges(removals)
        kept: list[TimeRange] = []
        cursor = 0.0
        for span in merged:
            if span.start > cursor:
                kept.append(TimeRange(start=cursor, end=min(span.start, duration)))
            cursor = max(cursor, span.end)
            if cursor >= duration:
                break
        if cursor < duration:
            kept.append(TimeRange(start=cursor, end=duration))

        floor = self._rules.min_kept_duration
        return tuple(span for span in kept if span.duration >= floor)


def merge_ranges(ranges: list[TimeRange]) -> tuple[TimeRange, ...]:
    """Sort and coalesce overlapping or touching ranges.

    Shared by cleanup and, later, by the Rule Engine. Touching ranges are merged as
    well as overlapping ones, because two removals that abut exactly are one cut.
    """
    if not ranges:
        return ()
    ordered = sorted(ranges, key=lambda span: (span.start, span.end))
    merged: list[TimeRange] = [ordered[0]]
    for span in ordered[1:]:
        last = merged[-1]
        if span.start <= last.end:
            if span.end > last.end:
                merged[-1] = TimeRange(start=last.start, end=span.end)
        else:
            merged.append(span)
    return tuple(merged)


def _overlaps_consumed(start: int, length: int, consumed: set[int]) -> bool:
    return any(position in consumed for position in range(start, start + length))


def _normalise(text: str) -> str:
    """Lowercase and strip punctuation, so "Um," matches "um".

    Unicode-aware: the category check drops full-width and typographic punctuation
    that a plain ``str.strip(".,!?")`` would leave behind, which matters as soon as
    the narration is not English.
    """
    folded = unicodedata.normalize("NFKC", text).casefold()
    return "".join(char for char in folded if not unicodedata.category(char).startswith("P"))


def word_span(words: tuple[Word, ...]) -> TimeRange | None:
    """The range covered by a run of words, or ``None`` when empty."""
    if not words:
        return None
    return TimeRange(start=words[0].start, end=words[-1].end)


def source_ref(transcript: Transcript) -> MediaRef:
    """The narration reference a cleanup report should carry."""
    return transcript.source


__all__ = [
    "CLEANUP_VERSION",
    "NarrationCleanupService",
    "map_to_timeline",
    "merge_ranges",
    "word_span",
]
