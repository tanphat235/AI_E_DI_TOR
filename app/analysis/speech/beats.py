"""Narration beats: the units the AI director actually reasons about.

A beat is "the thing being said right now" — usually one sentence. It is the level at
which an editor thinks: *this* line needs *that* shot. Recogniser segments are the
wrong granularity, being whatever length Whisper felt like emitting, so they are
re-cut at sentence boundaries and glued back together when too short to stand alone.

Each beat carries two clocks. ``range`` is in original narration time, so a human can
scrub to it in the raw recording. ``timeline_range`` is where it lands after cleanup
removes the gaps, which is the clock footage is placed against. Carrying both is what
lets a director reason about the finished video while still being able to check its
work against the source.

Keywords are extracted with a stopword list, not a language model. That is a
deliberate floor rather than an ambition: it is predictable, instant, and needs no
weights. When a real vision provider arrives in a later phase, matching should move to
embeddings — the field stays the same.
"""

from __future__ import annotations

import re
import unicodedata

from app.models.common import TimeRange, map_to_timeline
from app.models.speech import NarrationBeat, SpeechCleanupReport, Transcript, TranscriptSegment

BEATS_VERSION = "beats/1"

_SENTENCE_END = re.compile(r"(?<=[.!?…])\s+")
"""Split on whitespace *after* terminal punctuation.

A lookbehind rather than a capture-and-rejoin, so the punctuation stays attached to
the sentence it ends. Abbreviations ("Mr. Smith") will split wrongly; that is accepted
because the failure is one short beat rather than a wrong cut, and the alternative is
a sentence tokeniser and its model.
"""

MIN_BEAT_WORDS = 3
"""Below this a fragment is glued onto its neighbour rather than standing alone.

"And then." is not a beat an editor can find a shot for.
"""

_EN_STOPWORDS = (
    "a an and are as at be been but by can could did do does for from had has have he"
    " her here hers him his how i if in into is it its just me my no not of on one only"
    " or our out over own said same she should so some such than that the their them"
    " then there these they this those to too up us very was we were what when where"
    " which while who why will with would you your"
)

# Vietnamese is included because this project's author narrates in Vietnamese. Keyword
# extraction is only as good as its stopword list, and an English-only list would return
# every Vietnamese function word as a "keyword", making matching useless.
_VI_STOPWORDS = (
    "và là của có được cho nhưng thì mà này đó các những một hai với để từ trong trên"
    " dưới khi nếu vì nên rồi đã sẽ đang bị bởi cũng rất quá lắm không chưa tôi bạn anh"
    " chị em họ nó mình chúng ta ở về ra vào lên xuống nữa hơn nhất như sau trước giữa"
    " cùng theo hay hoặc tại do nào gì sao ai đâu"
)

STOPWORDS: dict[str, frozenset[str]] = {
    "en": frozenset(_EN_STOPWORDS.split()),
    "vi": frozenset(_VI_STOPWORDS.split()),
}

_MIN_KEYWORD_LENGTH = 3
_MAX_KEYWORDS = 8
"""Cap on keywords per beat. Beyond a handful they stop discriminating between
scenes, and every one costs the director tokens to read."""


class NarrationBeatBuilder:
    """Turns a transcript plus a cleanup report into beats."""

    def __init__(self, *, min_beat_words: int = MIN_BEAT_WORDS) -> None:
        self._min_beat_words = min_beat_words

    def build(
        self,
        transcript: Transcript,
        cleanup: SpeechCleanupReport | None = None,
    ) -> tuple[NarrationBeat, ...]:
        """Build beats, mapping each into timeline time when ``cleanup`` is given.

        With no cleanup report the two clocks are identical, which is the correct
        behaviour for a project that is not removing anything.
        """
        language = transcript.language.split("-")[0].lower()
        stopwords = STOPWORDS.get(language, STOPWORDS["en"])
        kept = cleanup.kept_ranges if cleanup is not None else None

        beats: list[NarrationBeat] = []
        for segment in transcript.segments:
            for text, span in self._split_segment(segment):
                beats.append(
                    NarrationBeat(
                        index=len(beats),
                        range=span,
                        text=text,
                        keywords=extract_keywords(text, stopwords),
                        segment_indices=(segment.index,),
                        timeline_range=self._timeline_range(span, kept),
                    )
                )
        return tuple(beats)

    # -- Splitting ---------------------------------------------------------- #

    def _split_segment(self, segment: TranscriptSegment) -> list[tuple[str, TimeRange]]:
        """Cut one segment into sentence-sized beats with their own timings.

        With word timings each sentence gets exact bounds. Without them, time is
        apportioned across the segment by character count — crude, but a better
        approximation than treating a forty-word segment as a single instant.
        """
        sentences = _split_sentences(segment.text)
        if len(sentences) <= 1:
            return [(segment.text, segment.range)]

        if segment.has_word_timings:
            timed = self._align_with_words(segment, sentences)
            if timed is not None:
                return timed
        return self._apportion_by_length(segment, sentences)

    def _align_with_words(
        self, segment: TranscriptSegment, sentences: list[str]
    ) -> list[tuple[str, TimeRange]] | None:
        """Assign exact bounds by walking the word list alongside the sentences.

        Returns ``None`` if the two disagree — a mismatch means the recogniser's word
        list and its text diverged, and apportioning by length is a safer fallback
        than a confidently wrong timing.
        """
        words = list(segment.words)
        position = 0
        result: list[tuple[str, TimeRange]] = []

        for sentence in sentences:
            needed = len(sentence.split())
            if needed == 0:
                continue
            if position + needed > len(words):
                return None
            run = words[position : position + needed]
            result.append((sentence, TimeRange(start=run[0].start, end=run[-1].end)))
            position += needed

        if position != len(words):
            # Leftover words mean the split did not line up; do not guess.
            return None
        return self._merge_short(result)

    def _apportion_by_length(
        self, segment: TranscriptSegment, sentences: list[str]
    ) -> list[tuple[str, TimeRange]]:
        total_chars = sum(len(sentence) for sentence in sentences)
        if total_chars == 0:
            return [(segment.text, segment.range)]

        result: list[tuple[str, TimeRange]] = []
        cursor = segment.range.start
        available = segment.range.duration
        for position, sentence in enumerate(sentences):
            share = available * (len(sentence) / total_chars)
            # Pin the final end to the segment's own end so rounding cannot leave a gap.
            end = segment.range.end if position == len(sentences) - 1 else cursor + share
            if end <= cursor:
                continue
            result.append((sentence, TimeRange(start=cursor, end=end)))
            cursor = end
        return self._merge_short(result or [(segment.text, segment.range)])

    def _merge_short(self, beats: list[tuple[str, TimeRange]]) -> list[tuple[str, TimeRange]]:
        """Glue fragments below the word floor onto their neighbour."""
        merged: list[tuple[str, TimeRange]] = []
        for text, span in beats:
            too_short = len(text.split()) < self._min_beat_words
            if too_short and merged:
                previous_text, previous_span = merged[-1]
                merged[-1] = (
                    f"{previous_text} {text}".strip(),
                    TimeRange(start=previous_span.start, end=span.end),
                )
            else:
                merged.append((text, span))
        return merged

    # -- Timeline mapping --------------------------------------------------- #

    @staticmethod
    def _timeline_range(span: TimeRange, kept: tuple[TimeRange, ...] | None) -> TimeRange | None:
        """Where a source range lands on the timeline.

        Without a cleanup report the clocks coincide. With one, the beat's own start
        and end may both sit inside removed gaps even though its middle survives, so
        the mapping probes inward from both ends rather than trusting the endpoints.
        """
        if kept is None:
            return span

        start = _first_surviving(span, kept)
        end = _last_surviving(span, kept)
        if start is None or end is None or end <= start:
            return None
        return TimeRange(start=start, end=end)


def _first_surviving(span: TimeRange, kept: tuple[TimeRange, ...]) -> float | None:
    """Timeline position of the earliest surviving instant within ``span``."""
    for candidate in kept:
        overlap = candidate.intersection(span)
        if overlap is not None:
            return map_to_timeline(overlap.start, kept)
    return None


def _last_surviving(span: TimeRange, kept: tuple[TimeRange, ...]) -> float | None:
    """Timeline position of the latest surviving instant within ``span``."""
    for candidate in reversed(kept):
        overlap = candidate.intersection(span)
        if overlap is None:
            continue
        # `map_to_timeline` is half-open, so the exact end is never "contained";
        # map a hair inside and add the remainder back.
        interior = map_to_timeline(max(overlap.start, overlap.end - 1e-6), kept)
        if interior is None:
            continue
        return interior + min(1e-6, overlap.duration)
    return None


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences, discarding empties."""
    return [part.strip() for part in _SENTENCE_END.split(text.strip()) if part.strip()]


def extract_keywords(text: str, stopwords: frozenset[str]) -> tuple[str, ...]:
    """Salient terms from a beat, for matching against scene tags.

    Order is first-appearance rather than frequency: a beat is one sentence, so
    frequency carries no signal, whereas the subject usually arrives early.
    """
    seen: dict[str, None] = {}
    for token in _tokenise(text):
        if len(token) < _MIN_KEYWORD_LENGTH or token in stopwords or token.isdigit():
            continue
        seen.setdefault(token, None)
        if len(seen) >= _MAX_KEYWORDS:
            break
    return tuple(seen)


def _tokenise(text: str) -> list[str]:
    """Lowercase word tokens, punctuation removed, accents preserved.

    Accents are kept deliberately: stripping them would merge distinct Vietnamese
    words and turn keyword matching into noise.
    """
    folded = unicodedata.normalize("NFKC", text).casefold()
    cleaned = "".join(
        char if not unicodedata.category(char).startswith("P") else " " for char in folded
    )
    return cleaned.split()


__all__ = [
    "BEATS_VERSION",
    "MIN_BEAT_WORDS",
    "STOPWORDS",
    "NarrationBeatBuilder",
    "extract_keywords",
]
