"""Tests for subtitle cue building and the two writers.

The load-bearing property is that **cue times are timeline time**. A cue timed against
the raw narration would drift further out of sync with every removed pause, so the
mapping tests here are the ones that matter.
"""

from __future__ import annotations

import itertools
from pathlib import Path

import pytest

from app.config.settings import SubtitleSettings
from app.models.common import MediaRef, SubtitleFormat, TimeRange
from app.models.edit_plan import SubtitleCue
from app.models.speech import Transcript, TranscriptSegment, Word
from app.subtitles.ass_writer import AssWriter, escape_text, karaoke_text
from app.subtitles.ass_writer import format_timestamp as ass_timestamp
from app.subtitles.base import SubtitleWriter
from app.subtitles.builder import SubtitleBuilder
from app.subtitles.registry import available_formats, writer_for
from app.subtitles.srt_writer import SrtWriter
from app.subtitles.srt_writer import format_timestamp as srt_timestamp


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


def _timed_segment(index: int, text: str, words: tuple[Word, ...]) -> TranscriptSegment:
    return TranscriptSegment(
        index=index,
        range=TimeRange(start=words[0].start, end=words[-1].end),
        text=text,
        words=words,
    )


def _cue(start: float, end: float, text: str = "hello", **kwargs: object) -> SubtitleCue:
    return SubtitleCue(range=TimeRange(start=start, end=end), text=text, **kwargs)  # type: ignore[arg-type]


@pytest.fixture
def style() -> SubtitleSettings:
    return SubtitleSettings()


class TestTimestampFormatting:
    @pytest.mark.parametrize(
        ("seconds", "expected"),
        [
            (0.0, "00:00:00,000"),
            (1.5, "00:00:01,500"),
            (61.25, "00:01:01,250"),
            (3661.125, "01:01:01,125"),
        ],
    )
    def test_srt(self, seconds: float, expected: str) -> None:
        assert srt_timestamp(seconds) == expected

    @pytest.mark.parametrize(
        ("seconds", "expected"),
        [
            (0.0, "0:00:00.00"),
            (1.5, "0:00:01.50"),
            (61.25, "0:01:01.25"),
            (3661.125, "1:01:01.12"),
        ],
    )
    def test_ass_uses_centiseconds_and_an_unpadded_hour(
        self, seconds: float, expected: str
    ) -> None:
        assert ass_timestamp(seconds) == expected

    def test_sub_millisecond_values_truncate_rather_than_round(self) -> None:
        """Rounding up could push a cue past its successor and re-create an overlap."""
        assert srt_timestamp(1.9999) == "00:00:01,999"
        assert ass_timestamp(1.999) == "0:00:01.99"

    def test_negative_times_are_clamped(self) -> None:
        assert srt_timestamp(-5.0) == "00:00:00,000"


class TestCueBuildingWithoutCleanup:
    def test_source_time_is_timeline_time(self) -> None:
        transcript = _transcript(
            _timed_segment(
                0,
                "Dig the soil.",
                (_word("Dig", 1.0, 1.3), _word("the", 1.35, 1.5), _word("soil.", 1.55, 2.0)),
            )
        )
        cues = SubtitleBuilder(SubtitleSettings()).build(transcript)
        assert len(cues) == 1
        assert cues[0].range.start == pytest.approx(1.0)
        assert cues[0].text == "Dig the soil."


class TestCueBuildingWithCleanup:
    def test_cues_shift_earlier_by_the_removed_amount(self) -> None:
        transcript = _transcript(
            _timed_segment(
                0,
                "Water it well.",
                (_word("Water", 6.0, 6.4), _word("it", 6.45, 6.6), _word("well.", 6.65, 7.1)),
            )
        )
        kept = (TimeRange(start=0.0, end=4.0), TimeRange(start=6.0, end=20.0))
        cues = SubtitleBuilder(SubtitleSettings()).build(transcript, kept_ranges=kept)
        # 6.0 in source is 4.0 on the timeline: the 4-6 gap is gone.
        assert cues[0].range.start == pytest.approx(4.0)

    def test_a_removed_word_is_dropped_from_the_text(self) -> None:
        """The filler must vanish from the subtitle as well as the audio."""
        transcript = _transcript(
            _timed_segment(
                0,
                "First, um, prepare.",
                (
                    _word("First,", 0.0, 0.4),
                    _word("um,", 0.5, 0.8),
                    _word("prepare.", 1.0, 1.6),
                ),
            )
        )
        # 0.5-0.8 (the "um") is cut out.
        kept = (TimeRange(start=0.0, end=0.5), TimeRange(start=0.8, end=20.0))
        cues = SubtitleBuilder(SubtitleSettings()).build(transcript, kept_ranges=kept)
        assert "um" not in cues[0].text
        assert cues[0].text == "First, prepare."

    def test_removing_a_gap_makes_the_words_one_continuous_cue(self) -> None:
        """A removed pause is *gone*, so the words either side become adjacent.

        This is the correct outcome, not a bug: in the finished video the audio plays
        continuously, so one cue covering it is exactly what a viewer needs. A cue is
        only split where a pause **survives** - see the test below.
        """
        transcript = _transcript(
            _timed_segment(
                0,
                "Before the cut after the cut.",
                (
                    _word("Before", 0.0, 0.4),
                    _word("the", 0.45, 0.6),
                    _word("cut", 0.65, 0.9),
                    # Everything from here is 10s later in the source.
                    _word("after", 11.0, 11.4),
                    _word("the", 11.45, 11.6),
                    _word("cut.", 11.65, 12.0),
                ),
            )
        )
        kept = (TimeRange(start=0.0, end=1.0), TimeRange(start=11.0, end=20.0))
        cues = SubtitleBuilder(SubtitleSettings()).build(transcript, kept_ranges=kept)
        assert len(cues) == 1
        assert cues[0].text == "Before the cut after the cut."
        # The 10-second source gap contributes nothing: 12s of source becomes 2s of
        # timeline, which is the sum of the surviving words alone.
        assert cues[0].range.duration == pytest.approx(2.0)

    def test_a_surviving_pause_splits_the_cue(self) -> None:
        """A pause left in the audio must not have a subtitle sitting across it."""
        transcript = _transcript(
            _timed_segment(
                0,
                "Before the pause after the pause.",
                (
                    _word("Before", 0.0, 0.4),
                    _word("the", 0.45, 0.6),
                    _word("pause", 0.65, 0.9),
                    # A two-second pause that cleanup did NOT remove.
                    _word("after", 2.9, 3.3),
                    _word("the", 3.35, 3.5),
                    _word("pause.", 3.55, 3.9),
                ),
            )
        )
        cues = SubtitleBuilder(SubtitleSettings()).build(transcript)
        assert len(cues) == 2
        assert cues[0].text == "Before the pause"
        assert cues[1].text == "after the pause."

    def test_a_fully_removed_segment_yields_no_cue(self) -> None:
        transcript = _transcript(
            _timed_segment(
                0, "Gone entirely.", (_word("Gone", 4.5, 4.8), _word("entirely.", 4.85, 5.4))
            )
        )
        kept = (TimeRange(start=0.0, end=4.0), TimeRange(start=6.0, end=20.0))
        assert SubtitleBuilder(SubtitleSettings()).build(transcript, kept_ranges=kept) == ()


class TestLineWrapping:
    def test_long_text_wraps_at_the_character_budget(self) -> None:
        words = tuple(_word(f"word{index}", index * 0.3, index * 0.3 + 0.25) for index in range(12))
        transcript = _transcript(_timed_segment(0, " ".join(w.text for w in words), words))
        settings = SubtitleSettings(max_chars_per_line=20, max_lines=2)
        cues = SubtitleBuilder(settings).build(transcript)
        for cue in cues:
            for line in cue.text.splitlines():
                assert len(line) <= 20, f"line too long: {line!r}"
            assert cue.line_count <= 2

    def test_text_beyond_the_line_budget_becomes_extra_cues(self) -> None:
        words = tuple(_word(f"word{index}", index * 0.3, index * 0.3 + 0.25) for index in range(30))
        transcript = _transcript(_timed_segment(0, " ".join(w.text for w in words), words))
        settings = SubtitleSettings(max_chars_per_line=20, max_lines=2)
        cues = SubtitleBuilder(settings).build(transcript)
        assert len(cues) > 1

    def test_a_single_word_longer_than_the_line_is_not_dropped(self) -> None:
        long_word = "a" * 60
        transcript = _transcript(_timed_segment(0, long_word, (_word(long_word, 0.0, 1.0),)))
        cues = SubtitleBuilder(SubtitleSettings(max_chars_per_line=20)).build(transcript)
        assert long_word in cues[0].text


class TestDurationHygiene:
    def test_a_very_short_cue_is_extended_to_the_minimum(self) -> None:
        transcript = _transcript(_timed_segment(0, "Hi.", (_word("Hi.", 0.0, 0.1),)))
        settings = SubtitleSettings(min_cue_duration=0.8)
        cues = SubtitleBuilder(settings).build(transcript)
        assert cues[0].range.duration == pytest.approx(0.8)

    def test_a_very_long_cue_is_capped(self) -> None:
        transcript = _transcript(_timed_segment(0, "Long.", (_word("Long.", 0.0, 30.0),)))
        settings = SubtitleSettings(max_cue_duration=5.0)
        cues = SubtitleBuilder(settings).build(transcript)
        assert cues[0].range.duration == pytest.approx(5.0)

    def test_cues_never_overlap_after_extension(self) -> None:
        """Extending a short cue must not swallow the next one's start."""
        transcript = _transcript(
            _timed_segment(0, "One.", (_word("One.", 0.0, 0.1),)),
            _timed_segment(1, "Two.", (_word("Two.", 0.4, 0.6),)),
        )
        settings = SubtitleSettings(min_cue_duration=1.0, cue_gap=0.04)
        cues = SubtitleBuilder(settings).build(transcript)
        for earlier, later in itertools.pairwise(cues):
            assert earlier.range.end <= later.range.start, "cues overlap"

    def test_cues_are_returned_in_ascending_order(self) -> None:
        transcript = _transcript(
            _timed_segment(0, "Second.", (_word("Second.", 5.0, 5.6),)),
            _timed_segment(1, "First.", (_word("First.", 1.0, 1.6),)),
        )
        cues = SubtitleBuilder(SubtitleSettings()).build(transcript)
        starts = [cue.range.start for cue in cues]
        assert starts == sorted(starts)


class TestSegmentFallback:
    def test_cues_are_still_produced_without_word_timings(self) -> None:
        transcript = _transcript(
            TranscriptSegment(
                index=0, range=TimeRange(start=1.0, end=4.0), text="Prepare the soil."
            )
        )
        cues = SubtitleBuilder(SubtitleSettings()).build(transcript)
        assert len(cues) == 1
        assert cues[0].range.start == pytest.approx(1.0)

    def test_no_karaoke_data_is_invented(self) -> None:
        transcript = _transcript(
            TranscriptSegment(
                index=0, range=TimeRange(start=1.0, end=4.0), text="Prepare the soil."
            )
        )
        cues = SubtitleBuilder(SubtitleSettings()).build(transcript)
        assert cues[0].words == ()

    def test_cleanup_mapping_still_applies(self) -> None:
        transcript = _transcript(
            TranscriptSegment(index=0, range=TimeRange(start=6.0, end=9.0), text="Water it well.")
        )
        kept = (TimeRange(start=0.0, end=4.0), TimeRange(start=6.0, end=20.0))
        cues = SubtitleBuilder(SubtitleSettings()).build(transcript, kept_ranges=kept)
        assert cues[0].range.start == pytest.approx(4.0)


class TestWordTimings:
    def test_karaoke_data_is_attached_when_requested(self) -> None:
        transcript = _transcript(
            _timed_segment(0, "Dig soil.", (_word("Dig", 0.0, 0.3), _word("soil.", 0.4, 0.9)))
        )
        cues = SubtitleBuilder(SubtitleSettings()).build(transcript, include_word_timings=True)
        assert len(cues[0].words) == 2
        assert cues[0].words[0][0] == "Dig"

    def test_it_can_be_skipped_when_only_srt_is_wanted(self) -> None:
        transcript = _transcript(
            _timed_segment(0, "Dig soil.", (_word("Dig", 0.0, 0.3), _word("soil.", 0.4, 0.9)))
        )
        cues = SubtitleBuilder(SubtitleSettings()).build(transcript, include_word_timings=False)
        assert cues[0].words == ()

    def test_word_timings_are_in_timeline_time_too(self) -> None:
        transcript = _transcript(
            _timed_segment(0, "Water it.", (_word("Water", 6.0, 6.4), _word("it.", 6.45, 6.8)))
        )
        kept = (TimeRange(start=0.0, end=4.0), TimeRange(start=6.0, end=20.0))
        cues = SubtitleBuilder(SubtitleSettings()).build(transcript, kept_ranges=kept)
        assert cues[0].words[0][1] == pytest.approx(4.0)


class TestSrtWriter:
    def test_structure(self, tmp_path: Path, style: SubtitleSettings) -> None:
        destination = SrtWriter().write(
            (_cue(0.0, 2.0, "First line."), _cue(3.0, 5.0, "Second line.")),
            tmp_path / "out.srt",
            style=style,
        )
        # Read bytes, not text: `read_text` applies universal-newline translation and
        # would report CRLF as LF, hiding whether the file is actually CRLF.
        raw = destination.read_bytes().removeprefix(b"\xef\xbb\xbf").decode("utf-8")
        assert raw.startswith("1\r\n")
        assert "00:00:00,000 --> 00:00:02,000" in raw
        assert "\r\n2\r\n" in raw

    def test_utf8_bom_is_written(self, tmp_path: Path, style: SubtitleSettings) -> None:
        """Without it, Windows players guess the codepage and mangle Vietnamese."""
        destination = SrtWriter().write(
            (_cue(0.0, 2.0, "Trồng cây"),), tmp_path / "vi.srt", style=style
        )
        assert destination.read_bytes().startswith(b"\xef\xbb\xbf")
        assert "Trồng cây" in destination.read_text(encoding="utf-8-sig")

    def test_multi_line_cues_use_crlf(self, tmp_path: Path, style: SubtitleSettings) -> None:
        destination = SrtWriter().write(
            (_cue(0.0, 2.0, "Line one\nLine two"),), tmp_path / "o.srt", style=style
        )
        assert b"Line one\r\nLine two" in destination.read_bytes()

    def test_it_declares_no_word_timing_support(self) -> None:
        assert SrtWriter().supports_word_timings is False
        assert SrtWriter().format is SubtitleFormat.SRT

    def test_no_cues_writes_an_empty_file_rather_than_failing(
        self, tmp_path: Path, style: SubtitleSettings
    ) -> None:
        destination = SrtWriter().write((), tmp_path / "empty.srt", style=style)
        assert destination.is_file()

    def test_parent_directories_are_created(self, tmp_path: Path, style: SubtitleSettings) -> None:
        destination = SrtWriter().write(
            (_cue(0.0, 1.0),), tmp_path / "deep" / "nested" / "o.srt", style=style
        )
        assert destination.is_file()


class TestAssWriter:
    def test_required_sections_are_present(self, tmp_path: Path, style: SubtitleSettings) -> None:
        destination = AssWriter().write((_cue(0.0, 2.0, "Hello"),), tmp_path / "o.ass", style=style)
        text = destination.read_text(encoding="utf-8")
        assert "[Script Info]" in text
        assert "[V4+ Styles]" in text
        assert "[Events]" in text
        assert "ScriptType: v4.00+" in text

    def test_style_settings_reach_the_file(self, tmp_path: Path) -> None:
        settings = SubtitleSettings(font_name="Inter", font_size=72, margin_v=100, alignment=8)
        destination = AssWriter().write((_cue(0.0, 2.0),), tmp_path / "o.ass", style=settings)
        style_line = next(
            line
            for line in destination.read_text(encoding="utf-8").splitlines()
            if line.startswith("Style:")
        )
        assert "Inter" in style_line
        assert "72" in style_line
        assert style_line.endswith(",100,1")

    def test_a_fixed_coordinate_space_keeps_text_size_stable(
        self, tmp_path: Path, style: SubtitleSettings
    ) -> None:
        """Font size must not change meaning when output resolution changes."""
        destination = AssWriter().write((_cue(0.0, 2.0),), tmp_path / "o.ass", style=style)
        text = destination.read_text(encoding="utf-8")
        assert "PlayResX: 1920" in text
        assert "PlayResY: 1080" in text

    def test_karaoke_tags_are_emitted(self, tmp_path: Path, style: SubtitleSettings) -> None:
        cue = _cue(0.0, 1.0, "Dig soil", words=(("Dig", 0.0, 0.3), ("soil", 0.4, 1.0)))
        destination = AssWriter(karaoke=True).write((cue,), tmp_path / "o.ass", style=style)
        dialogue = next(
            line
            for line in destination.read_text(encoding="utf-8").splitlines()
            if line.startswith("Dialogue:")
        )
        assert "{\\k30}Dig" in dialogue
        # "soil" spans 0.4-1.0, but the highlight starts where "Dig" ended at 0.3, so
        # the 0.1s gap is absorbed: 0.7s, not 0.6s. That keeps the sweep continuous.
        assert "{\\k70}soil" in dialogue

    def test_karaoke_can_be_disabled(self, tmp_path: Path, style: SubtitleSettings) -> None:
        cue = _cue(0.0, 1.0, "Dig soil", words=(("Dig", 0.0, 0.3), ("soil", 0.4, 1.0)))
        destination = AssWriter(karaoke=False).write((cue,), tmp_path / "o.ass", style=style)
        assert "\\k" not in destination.read_text(encoding="utf-8")

    def test_newlines_become_hard_breaks(self, tmp_path: Path, style: SubtitleSettings) -> None:
        destination = AssWriter().write(
            (_cue(0.0, 2.0, "Line one\nLine two"),), tmp_path / "o.ass", style=style
        )
        assert "Line one\\NLine two" in destination.read_text(encoding="utf-8")

    def test_it_declares_word_timing_support(self) -> None:
        assert AssWriter().supports_word_timings is True
        assert AssWriter().format is SubtitleFormat.ASS


class TestAssEscaping:
    def test_braces_are_neutralised(self) -> None:
        """An unescaped brace opens an override block and eats the rest of the line."""
        assert escape_text("{bold}") == "\\{bold\\}"

    def test_backslashes_are_doubled(self) -> None:
        assert escape_text("a\\b") == "a\\\\b"

    def test_both_newline_conventions_become_hard_breaks(self) -> None:
        assert escape_text("a\r\nb") == "a\\Nb"
        assert escape_text("a\nb") == "a\\Nb"

    def test_karaoke_of_a_cue_with_no_words_is_empty(self) -> None:
        assert karaoke_text(_cue(0.0, 1.0, "text")) == ""

    def test_a_zero_length_word_still_gets_a_visible_duration(self) -> None:
        """A \\k0 tag makes the word never highlight."""
        cue = _cue(0.0, 1.0, "x", words=(("x", 0.0, 0.0),))
        assert "{\\k1}" in karaoke_text(cue)


class TestRegistry:
    def test_srt_lookup(self) -> None:
        assert isinstance(writer_for(SubtitleFormat.SRT, SubtitleSettings()), SrtWriter)

    def test_ass_lookup(self) -> None:
        assert isinstance(writer_for(SubtitleFormat.ASS, SubtitleSettings()), AssWriter)

    def test_karaoke_config_reaches_the_writer(self, tmp_path: Path) -> None:
        writer = writer_for(SubtitleFormat.ASS, SubtitleSettings(karaoke=True))
        cue = _cue(0.0, 1.0, "Dig", words=(("Dig", 0.0, 1.0),))
        destination = writer.write((cue,), tmp_path / "o.ass", style=SubtitleSettings(karaoke=True))
        assert "\\k" in destination.read_text(encoding="utf-8")

    def test_every_format_has_a_writer(self) -> None:
        for subtitle_format in available_formats():
            assert writer_for(subtitle_format, SubtitleSettings()) is not None

    def test_writers_satisfy_the_protocol_without_inheriting(self) -> None:
        assert isinstance(SrtWriter(), SubtitleWriter)
        assert isinstance(AssWriter(), SubtitleWriter)
