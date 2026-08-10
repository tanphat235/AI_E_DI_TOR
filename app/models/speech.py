"""Speech recognition and narration cleanup results (Phase 2 / Phase 3 output).

The narration is the spine of the whole edit: footage is chosen to serve it,
subtitles are derived from it, and music is ducked under it. So these models are
deliberately richer than "text plus timestamps" - they carry the confidence and
word-level detail that later phases need in order to make judgement calls.
"""

from __future__ import annotations

from datetime import datetime
from typing import Self

from pydantic import Field, model_validator

from app.models.common import AiveModel, MediaRef, Score, Seconds, TimeRange


class Word(AiveModel):
    """A single recognised word with its own timing.

    Word-level timing is what makes karaoke-style ASS subtitles and precise
    filler-word excision possible. Without it, removing "um" means cutting a whole
    segment and losing the words around it.
    """

    text: str = Field(min_length=1)
    start: Seconds
    end: float = Field(gt=0.0)
    probability: Score = Field(
        default=1.0,
        description="Recogniser confidence for this word.",
    )

    @model_validator(mode="after")
    def _validate_order(self) -> Self:
        if self.end <= self.start:
            msg = f"word {self.text!r}: end ({self.end}) must exceed start ({self.start})"
            raise ValueError(msg)
        return self

    @property
    def range(self) -> TimeRange:
        return TimeRange(start=self.start, end=self.end)


class TranscriptSegment(AiveModel):
    """One recogniser segment - roughly a phrase or sentence."""

    index: int = Field(ge=0, description="Position within the transcript.")
    range: TimeRange
    text: str = Field(description="Segment text, already whitespace-stripped.")
    words: tuple[Word, ...] = Field(
        default=(),
        description="Word-level timings; empty when the recogniser did not emit them.",
    )
    avg_logprob: float | None = Field(
        default=None,
        description="Mean token log-probability. More negative means less certain.",
    )
    no_speech_prob: Score | None = Field(
        default=None,
        description=(
            "Probability the segment is not speech at all. High values usually mean "
            "the recogniser hallucinated text over music or room tone."
        ),
    )
    compression_ratio: float | None = Field(
        default=None,
        description=(
            "Text-to-token compression. Values above roughly 2.4 indicate the "
            "degenerate repetition loop Whisper falls into on unclear audio."
        ),
    )

    @property
    def has_word_timings(self) -> bool:
        return len(self.words) > 0


class Transcript(AiveModel):
    """The complete recognition result for one narration file."""

    source: MediaRef
    language: str = Field(min_length=2, description="BCP-47-ish code, e.g. 'en' or 'vi'.")
    language_probability: Score | None = None
    duration: float = Field(gt=0.0, description="Duration of the audio analysed.")
    model_name: str = Field(
        min_length=1,
        description="Recogniser identity, e.g. 'faster-whisper/medium'.",
    )
    segments: tuple[TranscriptSegment, ...] = ()

    @property
    def text(self) -> str:
        """The whole narration as one string."""
        return " ".join(segment.text for segment in self.segments).strip()

    @property
    def has_word_timings(self) -> bool:
        """True only when *every* segment carries word timings."""
        return bool(self.segments) and all(seg.has_word_timings for seg in self.segments)

    def words(self) -> tuple[Word, ...]:
        """Every word across every segment, in order."""
        return tuple(word for segment in self.segments for word in segment.words)


# --------------------------------------------------------------------------- #
# Cleanup (Phase 3)
# --------------------------------------------------------------------------- #


class SilenceSpan(AiveModel):
    """A stretch of narration quiet enough to remove."""

    range: TimeRange
    mean_db: float | None = Field(
        default=None,
        description=(
            "Mean level across the span in dBFS, when the detector measured it. "
            "FFmpeg's silencedetect reports only that a span fell below the threshold, "
            "not how far below, and an extra decode pass per span is not worth the "
            "number - so this is honestly None rather than a restated threshold."
        ),
    )


class FillerSpan(AiveModel):
    """A filler word or phrase that should be excised."""

    range: TimeRange
    text: str = Field(min_length=1, description="The matched filler, e.g. 'um'.")
    ambiguous: bool = Field(
        default=False,
        description=(
            "True when the match came from the ambiguous list - a word that is also "
            "ordinary English. Recorded so a reviewer can audit exactly the removals "
            "most likely to be wrong."
        ),
    )


class RepetitionSpan(AiveModel):
    """A repeated phrase - almost always a retake the speaker left in."""

    range: TimeRange
    text: str = Field(min_length=1)
    kept_range: TimeRange = Field(
        description="The occurrence being kept, so a reviewer can hear the choice.",
    )


class SpeechCleanupReport(AiveModel):
    """What cleanup proposes to remove from the narration, and what survives.

    ``kept_ranges`` is the authoritative output. The three span lists exist to
    explain *why* each gap appeared, which is what lets a human - or the AI
    director - overrule a specific cut instead of disabling cleanup wholesale.
    """

    source: MediaRef
    original_duration: float = Field(gt=0.0)
    silences: tuple[SilenceSpan, ...] = ()
    fillers: tuple[FillerSpan, ...] = ()
    repetitions: tuple[RepetitionSpan, ...] = ()
    kept_ranges: tuple[TimeRange, ...] = Field(
        default=(),
        description="Ranges of the original audio to retain, in ascending order.",
    )

    @model_validator(mode="after")
    def _validate_kept_ranges_ordered(self) -> Self:
        previous: TimeRange | None = None
        for current in self.kept_ranges:
            if previous is not None and current.start < previous.end:
                msg = (
                    f"kept_ranges must be ascending and non-overlapping: "
                    f"{current} starts before {previous} ends"
                )
                raise ValueError(msg)
            previous = current
        return self

    @property
    def kept_duration(self) -> float:
        return sum(item.duration for item in self.kept_ranges)

    @property
    def removed_duration(self) -> float:
        return self.original_duration - self.kept_duration


class NarrationBeat(AiveModel):
    """A semantic unit of narration that footage must cover.

    This is the object the AI director actually reasons about. One beat is "the
    thing being said right now" - typically a sentence - and the director's job is
    to choose the clip that best illustrates it.

    ``range`` is in *original* narration time. Mapping to timeline time happens
    once, when the Edit Plan is assembled, so that a beat stays traceable back to
    the audio a human can listen to.
    """

    index: int = Field(ge=0)
    range: TimeRange
    text: str = Field(min_length=1)
    keywords: tuple[str, ...] = Field(
        default=(),
        description="Salient terms for matching against scene tags.",
    )
    segment_indices: tuple[int, ...] = Field(
        default=(),
        description="Transcript segments this beat was assembled from.",
    )
    timeline_range: TimeRange | None = Field(
        default=None,
        description=(
            "Where this beat lands in the finished video, once cleanup gaps are "
            "removed. None when the beat was itself cut. This is the field the "
            "director uses to place footage, because it is in the same clock as "
            "the timeline."
        ),
    )

    @property
    def survives_cleanup(self) -> bool:
        """True when this beat still appears in the finished video."""
        return self.timeline_range is not None


class NarrationAnalysis(AiveModel):
    """Everything Phase 2 learns about the narration, as one document.

    Bundled rather than split across three files because the three parts are only
    meaningful together: beats are derived from the transcript *and* the cleanup, and
    a transcript paired with a stale cleanup report would silently mis-time every
    subtitle.
    """

    source: MediaRef
    transcript: Transcript
    cleanup: SpeechCleanupReport
    beats: tuple[NarrationBeat, ...] = ()
    analyzer_version: str = Field(
        min_length=1,
        description="Pipeline version, so cached results from an older build are not trusted.",
    )
    analyzed_at: datetime

    @model_validator(mode="after")
    def _validate_consistent_source(self) -> Self:
        if self.transcript.source != self.source:
            msg = f"transcript is for {self.transcript.source}, not {self.source}"
            raise ValueError(msg)
        if self.cleanup.source != self.source:
            msg = f"cleanup report is for {self.cleanup.source}, not {self.source}"
            raise ValueError(msg)
        return self

    @property
    def surviving_beats(self) -> tuple[NarrationBeat, ...]:
        """Beats that still appear in the finished video."""
        return tuple(beat for beat in self.beats if beat.survives_cleanup)

    @property
    def kept_ranges(self) -> tuple[TimeRange, ...]:
        """The narration ranges that survive, ready for an Edit Plan's narration track.

        Exposed here so the CLI can assemble a ``NarrationTrack`` without this module
        importing the Edit Plan. That import direction is the one thing keeping the
        plan a contract rather than a coupling.
        """
        return self.cleanup.kept_ranges


__all__ = [
    "FillerSpan",
    "NarrationAnalysis",
    "NarrationBeat",
    "RepetitionSpan",
    "SilenceSpan",
    "SpeechCleanupReport",
    "Transcript",
    "TranscriptSegment",
    "Word",
]
