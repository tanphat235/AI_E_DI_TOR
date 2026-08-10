"""Tests for narration cleanup: silence, filler and repetition removal.

Every test here runs with a fake silence detector and no audio file. That is the whole
point of putting the one audio-touching step behind a protocol: cleanup is where all
the judgement lives, and judgement should be testable in milliseconds.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.analysis.speech.cleanup import (
    NarrationCleanupService,
    map_to_timeline,
    merge_ranges,
)
from app.config.rules import RuleSettings
from app.models.common import MediaRef, TimeRange
from app.models.speech import SilenceSpan, Transcript, TranscriptSegment, Word

NO_AUDIO = Path("does-not-exist.wav")


class FakeSilenceDetector:
    """Returns a canned list of spans. Never touches the filesystem."""

    def __init__(self, *spans: tuple[float, float]) -> None:
        self._spans = tuple(
            SilenceSpan(range=TimeRange(start=start, end=end)) for start, end in spans
        )

    @property
    def name(self) -> str:
        return "fake"

    def detect(
        self, audio: Path, *, threshold_db: float, min_duration: float
    ) -> tuple[SilenceSpan, ...]:
        return self._spans


def _word(text: str, start: float, end: float) -> Word:
    return Word(text=text, start=start, end=end)


def _transcript(*segments: TranscriptSegment, duration: float = 20.0) -> Transcript:
    return Transcript(
        source=MediaRef(path="narration.wav"),
        language="en",
        duration=duration,
        model_name="test/fake",
        segments=segments,
    )


def _segment(index: int, text: str, words: tuple[Word, ...]) -> TranscriptSegment:
    return TranscriptSegment(
        index=index,
        range=TimeRange(start=words[0].start, end=words[-1].end),
        text=text,
        words=words,
    )


def _service(
    detector: FakeSilenceDetector | None = None, **rule_overrides: object
) -> NarrationCleanupService:
    return NarrationCleanupService(
        detector or FakeSilenceDetector(),
        RuleSettings(**rule_overrides),  # type: ignore[arg-type]
    )


class TestMergeRanges:
    def test_empty(self) -> None:
        assert merge_ranges([]) == ()

    def test_disjoint_ranges_are_left_alone(self) -> None:
        ranges = [TimeRange(start=0.0, end=1.0), TimeRange(start=2.0, end=3.0)]
        assert merge_ranges(ranges) == tuple(ranges)

    def test_overlapping_ranges_merge(self) -> None:
        merged = merge_ranges([TimeRange(start=0.0, end=2.0), TimeRange(start=1.0, end=3.0)])
        assert merged == (TimeRange(start=0.0, end=3.0),)

    def test_touching_ranges_merge(self) -> None:
        """Two removals that abut exactly are one cut."""
        merged = merge_ranges([TimeRange(start=0.0, end=2.0), TimeRange(start=2.0, end=3.0)])
        assert merged == (TimeRange(start=0.0, end=3.0),)

    def test_a_contained_range_is_absorbed(self) -> None:
        merged = merge_ranges([TimeRange(start=0.0, end=9.0), TimeRange(start=2.0, end=3.0)])
        assert merged == (TimeRange(start=0.0, end=9.0),)

    def test_unsorted_input_is_handled(self) -> None:
        merged = merge_ranges([TimeRange(start=5.0, end=6.0), TimeRange(start=0.0, end=1.0)])
        assert merged[0].start == 0.0


class TestMapToTimeline:
    KEPT = (TimeRange(start=0.0, end=4.0), TimeRange(start=6.0, end=10.0))

    def test_maps_within_the_first_range(self) -> None:
        assert map_to_timeline(2.0, self.KEPT) == pytest.approx(2.0)

    def test_maps_within_a_later_range_by_subtracting_the_gap(self) -> None:
        assert map_to_timeline(7.0, self.KEPT) == pytest.approx(5.0)

    def test_a_removed_moment_has_no_position(self) -> None:
        assert map_to_timeline(5.0, self.KEPT) is None

    def test_beyond_the_end_has_no_position(self) -> None:
        assert map_to_timeline(99.0, self.KEPT) is None

    def test_agrees_with_the_edit_plan_model(self) -> None:
        """The free function and the model method must never diverge."""
        from app.models.edit_plan import NarrationTrack

        track = NarrationTrack(source=MediaRef(path="narration.wav"), kept_ranges=self.KEPT)
        for moment in (0.0, 2.0, 3.9, 5.0, 6.0, 9.5):
            assert map_to_timeline(moment, self.KEPT) == track.source_to_timeline(moment)


class TestSilencePadding:
    def test_interior_silence_keeps_padding_on_both_sides(self) -> None:
        service = _service(FakeSilenceDetector((5.0, 8.0)), silence_keep_padding=0.1)
        report = service.clean(_transcript(duration=20.0), audio=NO_AUDIO)
        # 5.0-8.0 becomes 5.1-7.9, so the kept ranges break there.
        assert report.kept_ranges == (
            TimeRange(start=0.0, end=5.1),
            TimeRange(start=7.9, end=20.0),
        )

    def test_leading_silence_is_trimmed_flush(self) -> None:
        """80 ms of room tone before the first word is a late start, not breathing room."""
        service = _service(FakeSilenceDetector((0.0, 3.0)), silence_keep_padding=0.1)
        report = service.clean(_transcript(duration=20.0), audio=NO_AUDIO)
        assert report.kept_ranges == (TimeRange(start=2.9, end=20.0),)

    def test_trailing_silence_is_trimmed_flush(self) -> None:
        service = _service(FakeSilenceDetector((17.0, 20.0)), silence_keep_padding=0.1)
        report = service.clean(_transcript(duration=20.0), audio=NO_AUDIO)
        assert report.kept_ranges == (TimeRange(start=0.0, end=17.1),)

    def test_a_silence_shorter_than_its_padding_is_not_cut(self) -> None:
        """Otherwise padding would invert the range and cut audible speech."""
        service = _service(FakeSilenceDetector((5.0, 5.1)), silence_keep_padding=0.2)
        report = service.clean(_transcript(duration=20.0), audio=NO_AUDIO)
        assert report.kept_ranges == (TimeRange(start=0.0, end=20.0),)

    def test_zero_padding_cuts_the_whole_silence(self) -> None:
        service = _service(FakeSilenceDetector((5.0, 8.0)), silence_keep_padding=0.0)
        report = service.clean(_transcript(duration=20.0), audio=NO_AUDIO)
        assert report.kept_ranges == (
            TimeRange(start=0.0, end=5.0),
            TimeRange(start=8.0, end=20.0),
        )

    def test_slivers_below_the_floor_are_discarded(self) -> None:
        """A fragment between two cuts holds only padding, and plays as a click."""
        service = _service(
            FakeSilenceDetector((0.0, 5.0), (5.1, 10.0)),
            silence_keep_padding=0.0,
            min_kept_duration=0.5,
        )
        report = service.clean(_transcript(duration=10.0), audio=NO_AUDIO)
        # The 5.0-5.1 sliver is dropped rather than kept.
        assert report.kept_ranges == ()


class TestFillerRemoval:
    def test_removes_a_hesitation_sound(self) -> None:
        transcript = _transcript(
            _segment(
                0,
                "First, um, prepare the soil.",
                (
                    _word("First,", 0.0, 0.4),
                    _word("um,", 0.5, 0.8),
                    _word("prepare", 1.0, 1.5),
                    _word("the", 1.55, 1.65),
                    _word("soil.", 1.7, 2.2),
                ),
            )
        )
        fillers = _service().find_fillers(transcript)
        assert len(fillers) == 1
        assert fillers[0].text == "um"
        assert fillers[0].range == TimeRange(start=0.5, end=0.8)
        assert fillers[0].ambiguous is False

    def test_punctuation_and_case_do_not_prevent_a_match(self) -> None:
        transcript = _transcript(
            _segment(
                0,
                "Uh... okay then.",
                (_word("Uh...", 0.0, 0.4), _word("okay", 0.5, 0.9), _word("then.", 1.0, 1.4)),
            )
        )
        assert [span.text for span in _service().find_fillers(transcript)] == ["uh"]

    def test_ambiguous_words_are_left_alone_by_default(self) -> None:
        """Cutting "like" out of "I like gardening" would ruin the sentence."""
        transcript = _transcript(
            _segment(
                0,
                "I like gardening.",
                (_word("I", 0.0, 0.2), _word("like", 0.3, 0.6), _word("gardening.", 0.7, 1.4)),
            )
        )
        assert _service().find_fillers(transcript) == ()

    def test_ambiguous_words_are_removed_when_opted_in(self) -> None:
        transcript = _transcript(
            _segment(
                0,
                "It is, like, big.",
                (
                    _word("It", 0.0, 0.2),
                    _word("is,", 0.3, 0.5),
                    _word("like,", 0.6, 0.9),
                    _word("big.", 1.0, 1.4),
                ),
            )
        )
        service = _service(remove_ambiguous_fillers=True)
        fillers = service.find_fillers(transcript)
        assert [span.text for span in fillers] == ["like"]
        assert fillers[0].ambiguous is True

    def test_multi_word_fillers_are_matched_whole(self) -> None:
        transcript = _transcript(
            _segment(
                0,
                "It is, you know, fine.",
                (
                    _word("It", 0.0, 0.2),
                    _word("is,", 0.3, 0.5),
                    _word("you", 0.6, 0.8),
                    _word("know,", 0.9, 1.2),
                    _word("fine.", 1.3, 1.7),
                ),
            )
        )
        fillers = _service(remove_ambiguous_fillers=True).find_fillers(transcript)
        assert [span.text for span in fillers] == ["you know"]
        # One span covering both words, not two separate ones.
        assert fillers[0].range == TimeRange(start=0.6, end=1.2)

    def test_filler_removal_is_skipped_without_word_timings(self) -> None:
        """Losing a whole segment to excise an "um" is worse than keeping the "um"."""
        transcript = _transcript(
            TranscriptSegment(index=0, range=TimeRange(start=0.0, end=3.0), text="Um, hello there.")
        )
        assert _service().find_fillers(transcript) == ()

    def test_an_empty_transcript_yields_nothing(self) -> None:
        assert _service().find_fillers(_transcript()) == ()


class TestRepetitionRemoval:
    def _retake(self) -> Transcript:
        # "Then water it. Then water it well."
        return _transcript(
            _segment(
                0,
                "Then water it. Then water it well.",
                (
                    _word("Then", 5.0, 5.25),
                    _word("water", 5.3, 5.7),
                    _word("it.", 5.75, 5.95),
                    _word("Then", 6.3, 6.55),
                    _word("water", 6.6, 7.0),
                    _word("it", 7.05, 7.2),
                    _word("well.", 7.25, 7.7),
                ),
            )
        )

    def test_keeps_the_later_take(self) -> None:
        """A restart means the speaker is correcting themselves."""
        repetitions = _service().find_repetitions(self._retake())
        assert len(repetitions) == 1
        span = repetitions[0]
        assert span.text == "then water it"
        assert span.range == TimeRange(start=5.0, end=5.95)
        assert span.kept_range == TimeRange(start=6.3, end=7.2)

    def test_a_long_gap_means_deliberate_repetition_not_a_retake(self) -> None:
        transcript = _transcript(
            _segment(
                0,
                "Then water it. Then water it well.",
                (
                    _word("Then", 5.0, 5.25),
                    _word("water", 5.3, 5.7),
                    _word("it.", 5.75, 5.95),
                    # A five-second gap: rhetorical repetition, not a stumble.
                    _word("Then", 11.0, 11.25),
                    _word("water", 11.3, 11.7),
                    _word("it", 11.75, 11.9),
                    _word("well.", 11.95, 12.4),
                ),
            )
        )
        assert _service(max_repetition_gap=1.5).find_repetitions(transcript) == ()

    def test_single_repeated_words_are_kept_by_default(self) -> None:
        """At min_repetition_words=1, "very very good" would silently lose a word."""
        transcript = _transcript(
            _segment(
                0,
                "It is very very good.",
                (
                    _word("It", 0.0, 0.2),
                    _word("is", 0.25, 0.4),
                    _word("very", 0.45, 0.7),
                    _word("very", 0.75, 1.0),
                    _word("good.", 1.05, 1.5),
                ),
            )
        )
        assert _service().find_repetitions(transcript) == ()

    def test_single_words_are_caught_when_configured(self) -> None:
        transcript = _transcript(
            _segment(
                0,
                "It is very very good.",
                (
                    _word("It", 0.0, 0.2),
                    _word("is", 0.25, 0.4),
                    _word("very", 0.45, 0.7),
                    _word("very", 0.75, 1.0),
                    _word("good.", 1.05, 1.5),
                ),
            )
        )
        found = _service(min_repetition_words=1).find_repetitions(transcript)
        assert [span.text for span in found] == ["very"]

    def test_prefers_the_longest_matching_phrase(self) -> None:
        """One three-word removal, not three one-word removals."""
        found = _service(min_repetition_words=1).find_repetitions(self._retake())
        assert len(found) == 1
        assert found[0].text == "then water it"

    def test_too_few_words_to_repeat(self) -> None:
        transcript = _transcript(_segment(0, "Hello.", (_word("Hello.", 0.0, 0.5),)))
        assert _service().find_repetitions(transcript) == ()

    def test_no_word_timings_means_no_detection(self) -> None:
        transcript = _transcript(
            TranscriptSegment(index=0, range=TimeRange(start=0.0, end=3.0), text="a b a b")
        )
        assert _service().find_repetitions(transcript) == ()


class TestCleanReport:
    def test_combines_every_removal_kind(self) -> None:
        transcript = _transcript(
            _segment(
                0,
                "First, um, prepare the soil.",
                (
                    _word("First,", 0.0, 0.4),
                    _word("um,", 0.5, 0.8),
                    _word("prepare", 1.0, 1.5),
                    _word("the", 1.55, 1.65),
                    _word("soil.", 1.7, 2.2),
                ),
            ),
            duration=9.0,
        )
        service = _service(FakeSilenceDetector((2.2, 5.0)), silence_keep_padding=0.05)
        report = service.clean(transcript, audio=NO_AUDIO)

        assert len(report.silences) == 1
        assert len(report.fillers) == 1
        assert report.original_duration == 9.0
        assert report.kept_duration < report.original_duration
        assert report.removed_duration > 0.0
        # kept_ranges must remain ascending and non-overlapping - the model enforces it,
        # so a bug here would surface as a ValidationError rather than a wrong edit.
        assert report.kept_ranges == tuple(sorted(report.kept_ranges, key=lambda r: r.start))

    def test_nothing_to_remove_keeps_everything(self) -> None:
        report = _service().clean(_transcript(duration=12.0), audio=NO_AUDIO)
        assert report.kept_ranges == (TimeRange(start=0.0, end=12.0),)
        assert report.removed_duration == pytest.approx(0.0)

    def test_source_is_carried_through(self) -> None:
        report = _service().clean(_transcript(duration=5.0), audio=NO_AUDIO)
        assert report.source == MediaRef(path="narration.wav")

    def test_a_fully_silent_file_keeps_nothing(self) -> None:
        service = _service(FakeSilenceDetector((0.0, 10.0)), silence_keep_padding=0.0)
        report = service.clean(_transcript(duration=10.0), audio=NO_AUDIO)
        assert report.kept_ranges == ()
        assert report.kept_duration == pytest.approx(0.0)
