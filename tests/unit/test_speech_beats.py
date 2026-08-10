"""Tests for beat building and keyword extraction.

Beats are what the AI director reasons about, and they carry two clocks. The tests that
matter most are the timeline-mapping ones: a beat whose ``timeline_range`` is wrong
sends footage to the wrong moment in the finished video.
"""

from __future__ import annotations

import pytest

from app.analysis.speech.beats import STOPWORDS, NarrationBeatBuilder, extract_keywords
from app.models.common import MediaRef, TimeRange
from app.models.speech import SpeechCleanupReport, Transcript, TranscriptSegment, Word


def _word(text: str, start: float, end: float) -> Word:
    return Word(text=text, start=start, end=end)


def _transcript(
    *segments: TranscriptSegment, duration: float = 20.0, language: str = "en"
) -> Transcript:
    return Transcript(
        source=MediaRef(path="narration.wav"),
        language=language,
        duration=duration,
        model_name="test/fake",
        segments=segments,
    )


def _cleanup(*kept: tuple[float, float], duration: float = 20.0) -> SpeechCleanupReport:
    return SpeechCleanupReport(
        source=MediaRef(path="narration.wav"),
        original_duration=duration,
        kept_ranges=tuple(TimeRange(start=start, end=end) for start, end in kept),
    )


class TestSentenceSplitting:
    def test_a_single_sentence_stays_one_beat(self) -> None:
        transcript = _transcript(
            TranscriptSegment(
                index=0, range=TimeRange(start=0.0, end=3.0), text="Prepare the soil well."
            )
        )
        beats = NarrationBeatBuilder().build(transcript)
        assert len(beats) == 1
        assert beats[0].range == TimeRange(start=0.0, end=3.0)

    def test_a_multi_sentence_segment_splits(self) -> None:
        transcript = _transcript(
            TranscriptSegment(
                index=0,
                range=TimeRange(start=0.0, end=6.0),
                text="Prepare the soil well. Then water the plant.",
            )
        )
        beats = NarrationBeatBuilder().build(transcript)
        assert len(beats) == 2
        assert beats[0].text == "Prepare the soil well."
        assert beats[1].text == "Then water the plant."

    def test_split_beats_tile_the_segment_without_gaps(self) -> None:
        """Apportioning must not lose or duplicate time."""
        transcript = _transcript(
            TranscriptSegment(
                index=0,
                range=TimeRange(start=2.0, end=8.0),
                text="One sentence here. Another sentence there.",
            )
        )
        beats = NarrationBeatBuilder().build(transcript)
        assert beats[0].range.start == 2.0
        assert beats[-1].range.end == 8.0
        assert beats[0].range.end == pytest.approx(beats[1].range.start)

    def test_word_timings_give_exact_bounds(self) -> None:
        transcript = _transcript(
            TranscriptSegment(
                index=0,
                range=TimeRange(start=0.0, end=4.0),
                text="Dig the soil. Water it now.",
                words=(
                    _word("Dig", 0.0, 0.3),
                    _word("the", 0.35, 0.5),
                    _word("soil.", 0.55, 1.0),
                    _word("Water", 2.0, 2.4),
                    _word("it", 2.45, 2.6),
                    _word("now.", 2.65, 3.1),
                ),
            )
        )
        beats = NarrationBeatBuilder().build(transcript)
        assert beats[0].range == TimeRange(start=0.0, end=1.0)
        assert beats[1].range == TimeRange(start=2.0, end=3.1)

    def test_a_short_fragment_is_glued_to_its_neighbour(self) -> None:
        """ "And then." is not something an editor can find a shot for."""
        transcript = _transcript(
            TranscriptSegment(
                index=0,
                range=TimeRange(start=0.0, end=6.0),
                text="We prepared the soil carefully. And then.",
            )
        )
        beats = NarrationBeatBuilder().build(transcript)
        assert len(beats) == 1
        assert beats[0].text.endswith("And then.")

    def test_question_and_exclamation_marks_also_split(self) -> None:
        transcript = _transcript(
            TranscriptSegment(
                index=0,
                range=TimeRange(start=0.0, end=6.0),
                text="Is the soil ready? Yes it is now!",
            )
        )
        assert len(NarrationBeatBuilder().build(transcript)) == 2

    def test_beat_indices_are_contiguous_across_segments(self) -> None:
        transcript = _transcript(
            TranscriptSegment(
                index=0, range=TimeRange(start=0.0, end=3.0), text="First sentence here."
            ),
            TranscriptSegment(
                index=1, range=TimeRange(start=3.0, end=6.0), text="Second sentence here."
            ),
        )
        beats = NarrationBeatBuilder().build(transcript)
        assert [beat.index for beat in beats] == [0, 1]
        assert [beat.segment_indices for beat in beats] == [(0,), (1,)]


class TestTimelineMapping:
    def test_without_cleanup_the_clocks_coincide(self) -> None:
        transcript = _transcript(
            TranscriptSegment(
                index=0, range=TimeRange(start=1.0, end=4.0), text="Prepare the soil."
            )
        )
        beats = NarrationBeatBuilder().build(transcript, None)
        assert beats[0].timeline_range == beats[0].range

    def test_a_beat_after_a_removed_gap_shifts_earlier(self) -> None:
        transcript = _transcript(
            TranscriptSegment(index=0, range=TimeRange(start=6.0, end=9.0), text="Water it well.")
        )
        # 0-4 kept, 4-6 removed, 6-20 kept: everything after 6.0 shifts back by 2s.
        beats = NarrationBeatBuilder().build(transcript, _cleanup((0.0, 4.0), (6.0, 20.0)))
        assert beats[0].timeline_range is not None
        assert beats[0].timeline_range.start == pytest.approx(4.0)

    def test_a_fully_removed_beat_has_no_timeline_position(self) -> None:
        """This is how the director learns a retake was dropped."""
        transcript = _transcript(
            TranscriptSegment(index=0, range=TimeRange(start=4.5, end=5.5), text="Water it.")
        )
        beats = NarrationBeatBuilder().build(transcript, _cleanup((0.0, 4.0), (6.0, 20.0)))
        assert beats[0].timeline_range is None
        assert beats[0].survives_cleanup is False

    def test_a_partially_removed_beat_survives(self) -> None:
        """A beat straddling a cut still needs footage for the part that remains."""
        transcript = _transcript(
            TranscriptSegment(
                index=0, range=TimeRange(start=3.0, end=7.0), text="Water it well now."
            )
        )
        beats = NarrationBeatBuilder().build(transcript, _cleanup((0.0, 4.0), (6.0, 20.0)))
        assert beats[0].survives_cleanup is True

    def test_surviving_beats_helper(self) -> None:
        from datetime import UTC, datetime

        from app.models.speech import NarrationAnalysis

        transcript = _transcript(
            TranscriptSegment(
                index=0, range=TimeRange(start=0.0, end=2.0), text="Kept sentence here."
            ),
            TranscriptSegment(
                index=1, range=TimeRange(start=4.2, end=5.5), text="Cut sentence here."
            ),
        )
        cleanup = _cleanup((0.0, 4.0), (6.0, 20.0))
        analysis = NarrationAnalysis(
            source=MediaRef(path="narration.wav"),
            transcript=transcript,
            cleanup=cleanup,
            beats=NarrationBeatBuilder().build(transcript, cleanup),
            analyzer_version="test/1",
            analyzed_at=datetime.now(UTC),
        )
        assert len(analysis.beats) == 2
        assert len(analysis.surviving_beats) == 1
        assert analysis.kept_ranges == cleanup.kept_ranges


class TestKeywords:
    def test_stopwords_and_short_tokens_are_dropped(self) -> None:
        keywords = extract_keywords("First, prepare the soil in a bed.", STOPWORDS["en"])
        assert "prepare" in keywords
        assert "soil" in keywords
        assert "the" not in keywords
        assert "in" not in keywords
        assert "a" not in keywords

    def test_punctuation_is_stripped_but_order_preserved(self) -> None:
        assert extract_keywords("Soil, water, mulch!", STOPWORDS["en"]) == (
            "soil",
            "water",
            "mulch",
        )

    def test_duplicates_collapse(self) -> None:
        assert extract_keywords("Soil soil soil.", STOPWORDS["en"]) == ("soil",)

    def test_digits_are_not_keywords(self) -> None:
        assert "2024" not in extract_keywords("Planted 2024 tomatoes.", STOPWORDS["en"])

    def test_keyword_count_is_capped(self) -> None:
        text = " ".join(f"word{index}" for index in range(30))
        assert len(extract_keywords(text, STOPWORDS["en"])) <= 8

    def test_vietnamese_stopwords_are_applied(self) -> None:
        """Without a Vietnamese list, every function word would come back as a keyword."""
        keywords = extract_keywords("Tôi trồng cây xanh trong vườn của tôi.", STOPWORDS["vi"])
        assert "tôi" not in keywords
        assert "của" not in keywords
        assert "trồng" in keywords

    def test_accents_are_preserved(self) -> None:
        """Stripping them would merge distinct Vietnamese words into noise."""
        assert extract_keywords("Trồng cây", STOPWORDS["vi"]) == ("trồng", "cây")

    def test_language_selects_the_stopword_list(self) -> None:
        transcript = _transcript(
            TranscriptSegment(
                index=0, range=TimeRange(start=0.0, end=3.0), text="Tôi trồng cây xanh."
            ),
            language="vi",
        )
        beats = NarrationBeatBuilder().build(transcript)
        assert "tôi" not in beats[0].keywords

    def test_an_unknown_language_falls_back_to_english(self) -> None:
        transcript = _transcript(
            TranscriptSegment(
                index=0, range=TimeRange(start=0.0, end=3.0), text="The soil is ready."
            ),
            language="xx",
        )
        beats = NarrationBeatBuilder().build(transcript)
        assert "the" not in beats[0].keywords

    def test_a_regional_language_tag_is_reduced_to_its_base(self) -> None:
        transcript = _transcript(
            TranscriptSegment(
                index=0, range=TimeRange(start=0.0, end=3.0), text="The soil is ready."
            ),
            language="en-GB",
        )
        beats = NarrationBeatBuilder().build(transcript)
        assert "the" not in beats[0].keywords


class TestEmptyInput:
    def test_no_segments_yields_no_beats(self) -> None:
        assert NarrationBeatBuilder().build(_transcript()) == ()
