"""Tests for script-to-speech narration.

The split follows the code. :mod:`app.analysis.speech.synth.script` is pure string work and is
tested directly. The narrator is tested against a **fake synthesizer** that writes silence, so
the document-building logic — which is where the value is — needs neither a network call nor a
PowerShell subprocess. The real backends are exercised in Phase verification, not here: a unit
suite that needs the internet is a unit suite that gets skipped.

One test in here guards a bug that all 1290 other tests missed, and it is worth reading first:
``TestUtf8Stdout``. AIVE could not print its own output for any project that was not in
English, because a Windows console hands Python cp1252 and ``CliRunner`` captures in UTF-8 —
so the tests never touched the encoding that broke.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from app.analysis.speech.synth.base import (
    SynthesizedLine,
    Voice,
    VoiceNotFoundError,
    WordTiming,
)
from app.analysis.speech.synth.edge import DEFAULT_VOICES, EdgeSynthesizer, _rate_string
from app.analysis.speech.synth.narrator import narrator_version
from app.analysis.speech.synth.sapi import SapiSynthesizer, _rate_to_sapi
from app.analysis.speech.synth.script import (
    ScriptError,
    load_script,
    parse_script,
)

VIETNAMESE = "Đầu tiên, chuẩn bị đất cho thật tơi và sạch cỏ."


# --------------------------------------------------------------------------- #
# Script parsing
# --------------------------------------------------------------------------- #


class TestParseScript:
    def test_a_blank_line_separates_beats(self) -> None:
        """A beat is what footage gets matched against, so this is an editorial boundary."""
        lines = parse_script("First line.\n\nSecond line.\n\nThird line.")
        assert [item.text for item in lines] == ["First line.", "Second line.", "Third line."]

    def test_a_single_newline_is_a_soft_wrap_not_a_beat(self) -> None:
        """Editors wrap at column 80; the writer did not mean a cut there."""
        lines = parse_script("This sentence was\nwrapped by an editor.")
        assert len(lines) == 1
        assert lines[0].text == "This sentence was wrapped by an editor."

    def test_comments_are_not_spoken(self) -> None:
        """A script needs somewhere to put a stage direction."""
        lines = parse_script("# camera pushes in here\nSpoken text.\n# another note")
        assert [item.text for item in lines] == ["Spoken text."]

    def test_runs_of_whitespace_collapse(self) -> None:
        """A double space is invisible in an editor and audible as a stumble in some voices."""
        assert parse_script("Two  spaces   here.")[0].text == "Two spaces here."

    def test_several_blank_lines_do_not_create_empty_beats(self) -> None:
        lines = parse_script("One.\n\n\n\n\nTwo.")
        assert len(lines) == 2

    def test_beats_are_indexed_in_order(self) -> None:
        lines = parse_script("A.\n\nB.\n\nC.")
        assert [item.index for item in lines] == [0, 1, 2]

    def test_vietnamese_survives_intact(self) -> None:
        """The whole point of the feature; a mangled diacritic is a mispronounced word."""
        assert parse_script(VIETNAMESE)[0].text == VIETNAMESE

    def test_word_count_is_reported(self) -> None:
        assert parse_script("One two three four.")[0].word_count == 4

    def test_an_empty_script_is_refused(self) -> None:
        with pytest.raises(ScriptError, match="no speakable text"):
            parse_script("   \n\n  \n")

    def test_a_comment_only_script_is_refused(self) -> None:
        """Silently producing zero beats would look like a successful empty narration."""
        with pytest.raises(ScriptError, match="no speakable text"):
            parse_script("# just a note\n# and another")


class TestLongParagraphSplitting:
    def test_splitting_is_off_by_default(self) -> None:
        """One paragraph is exactly one beat unless the user asks otherwise."""
        long_text = " ".join(f"word{index}." for index in range(60))
        assert len(parse_script(long_text)) == 1

    def test_a_long_paragraph_splits_at_sentence_boundaries(self) -> None:
        text = "First sentence here. Second sentence here. Third sentence here."
        lines = parse_script(text, max_words_per_line=4)
        assert len(lines) == 3
        assert lines[0].text == "First sentence here."

    def test_sentences_are_recombined_greedily(self) -> None:
        """Splitting into three beats when two fit cuts more than the writing asked for."""
        text = "One two. Three four. Five six."
        # Each sentence is two words: 2+2 fits under 4, 4+2 does not, so two beats.
        lines = parse_script(text, max_words_per_line=4)
        assert [item.text for item in lines] == ["One two. Three four.", "Five six."]
        # Raise the limit and all three fit in one.
        assert len(parse_script(text, max_words_per_line=6)) == 1

    def test_a_paragraph_under_the_limit_is_untouched(self) -> None:
        assert len(parse_script("Short enough.", max_words_per_line=10)) == 1

    def test_one_very_long_sentence_is_left_whole(self) -> None:
        """A cut inside a clause is worse than one long shot."""
        text = " ".join(f"word{index}" for index in range(50)) + "."
        assert len(parse_script(text, max_words_per_line=5)) == 1

    def test_an_abbreviation_does_not_end_a_sentence(self) -> None:
        text = "Ask Dr. Smith about it. Then water it well."
        lines = parse_script(text, max_words_per_line=5)
        assert lines[0].text == "Ask Dr. Smith about it."

    def test_vietnamese_sentences_split_correctly(self) -> None:
        text = "Đầu tiên chuẩn bị đất. Sau đó tưới nước cho kỹ."
        lines = parse_script(text, max_words_per_line=5)
        assert len(lines) == 2
        assert lines[1].text == "Sau đó tưới nước cho kỹ."


class TestLoadScript:
    def test_a_file_is_read_and_parsed(self, tmp_path: Path) -> None:
        path = tmp_path / "script.txt"
        path.write_text("One.\n\nTwo.", encoding="utf-8")
        assert len(load_script(path)) == 2

    def test_a_missing_file_names_the_path(self, tmp_path: Path) -> None:
        with pytest.raises(ScriptError, match="no script at"):
            load_script(tmp_path / "absent.txt")

    def test_a_notepad_bom_is_not_pronounced(self, tmp_path: Path) -> None:
        """An unstripped BOM becomes an invisible first character the voice tries to say."""
        path = tmp_path / "script.txt"
        path.write_bytes(b"\xef\xbb\xbf" + VIETNAMESE.encode("utf-8"))
        assert load_script(path)[0].text == VIETNAMESE

    def test_utf8_vietnamese_round_trips_from_disk(self, tmp_path: Path) -> None:
        path = tmp_path / "script.txt"
        path.write_text(VIETNAMESE, encoding="utf-8")
        assert load_script(path)[0].text == VIETNAMESE


# --------------------------------------------------------------------------- #
# The narrator, against a fake backend
# --------------------------------------------------------------------------- #


class FakeSynthesizer:
    """Writes a fixed length of real silence per line.

    Real audio, not a stub file: the narrator *measures* what it produced rather than
    trusting a reported duration, so a fake that wrote nothing would exercise none of the
    interesting code.
    """

    def __init__(self, *, seconds: float = 1.0, words: bool = True) -> None:
        self.seconds = seconds
        self.emit_words = words
        self.spoken: list[str] = []

    @property
    def name(self) -> str:
        return "fake"

    @property
    def requires_network(self) -> bool:
        return False

    def voices(self) -> tuple[Voice, ...]:
        return (Voice(name="fake-vi", locale="vi-VN"), Voice(name="fake-en", locale="en-US"))

    def default_voice(self, language: str) -> str:
        for voice in self.voices():
            if voice.language == language.split("-")[0].lower():
                return voice.name
        raise VoiceNotFoundError(language)

    def speak(
        self, text: str, *, voice: str, destination: Path, rate: float = 1.0
    ) -> SynthesizedLine:
        import subprocess

        import imageio_ffmpeg

        self.spoken.append(text)
        destination.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                imageio_ffmpeg.get_ffmpeg_exe(),
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "lavfi",
                "-i",
                f"anullsrc=r=48000:cl=mono:d={self.seconds}",
                str(destination),
            ],
            check=True,
            capture_output=True,
        )
        words = (
            tuple(
                WordTiming(text=word, start=index * 0.1, end=index * 0.1 + 0.09)
                for index, word in enumerate(text.split())
            )
            if self.emit_words
            else ()
        )
        return SynthesizedLine(index=0, text=text, audio=destination, duration=0.0, words=words)


@pytest.fixture
def locator():
    from app.config.settings import AiveSettings
    from app.services.ffmpeg_locator import FFmpegLocator

    return FFmpegLocator(AiveSettings().media)


def _narrate(tmp_path: Path, locator, text: str, **kwargs):
    from app.analysis.speech.synth.narrator import narrate

    synthesizer = kwargs.pop("synthesizer", None) or FakeSynthesizer()
    return narrate(
        parse_script(text),
        synthesizer=synthesizer,
        voice="fake-vi",
        destination=tmp_path / "narration.wav",
        locator=locator,
        language="vi",
        **kwargs,
    )


@pytest.mark.integration
class TestNarrator:
    """Needs FFmpeg to make and measure silence, hence ``integration``.

    Run with ``pytest -m integration``. FFmpeg ships with the package, so this is not
    normally skipped in practice.
    """

    def test_every_line_is_spoken_once(self, tmp_path: Path, locator) -> None:
        fake = FakeSynthesizer()
        _narrate(tmp_path, locator, "One.\n\nTwo.\n\nThree.", synthesizer=fake)
        assert fake.spoken == ["One.", "Two.", "Three."]

    def test_the_joined_audio_exists_and_has_length(self, tmp_path: Path, locator) -> None:
        result = _narrate(tmp_path, locator, "One.\n\nTwo.", gap=0.0)
        assert result.audio.is_file()
        assert result.duration == pytest.approx(2.0, abs=0.15)

    def test_the_gap_is_added_between_lines(self, tmp_path: Path, locator) -> None:
        """A script read with no pause between sentences sounds hurried."""
        without = _narrate(tmp_path / "a", locator, "One.\n\nTwo.", gap=0.0)
        with_gap = _narrate(tmp_path / "b", locator, "One.\n\nTwo.", gap=0.5)
        assert with_gap.duration - without.duration == pytest.approx(1.0, abs=0.15)

    def test_timeline_time_equals_source_time(self, tmp_path: Path, locator) -> None:
        """The one place in AIVE where these two clocks coincide, because nothing was cut.

        Everywhere else the divergence is the single easiest thing to get wrong, so the
        absence of it here is asserted rather than assumed.
        """
        result = _narrate(tmp_path, locator, "One.\n\nTwo.\n\nThree.")
        for beat in result.analysis.beats:
            assert beat.timeline_range == beat.range

    def test_nothing_is_reported_as_removed(self, tmp_path: Path, locator) -> None:
        """Synthesised speech has no fillers, no retakes and no dead air to cut."""
        cleanup = _narrate(tmp_path, locator, "One.\n\nTwo.").analysis.cleanup
        assert cleanup.silences == ()
        assert cleanup.fillers == ()
        assert cleanup.repetitions == ()
        assert len(cleanup.kept_ranges) == 1

    def test_the_text_is_exact_rather_than_recognised(self, tmp_path: Path, locator) -> None:
        """The feature's real advantage: a recogniser mishears; this cannot."""
        result = _narrate(tmp_path, locator, VIETNAMESE)
        assert result.analysis.beats[0].text == VIETNAMESE

    def test_word_timings_are_shifted_onto_the_joined_clock(self, tmp_path: Path, locator) -> None:
        """Backends report offsets relative to their own line, not to the whole narration."""
        result = _narrate(tmp_path, locator, "One two.\n\nThree four.", gap=0.0)
        second = result.analysis.transcript.segments[1]
        assert second.words
        assert second.words[0].start >= second.range.start

    def test_no_word_escapes_its_own_segment(self, tmp_path: Path, locator) -> None:
        """A service occasionally reports a final word running past the audio it produced,
        and a Word outside its segment fails model validation."""
        result = _narrate(tmp_path, locator, "One two three four five six.")
        for segment in result.analysis.transcript.segments:
            for word in segment.words:
                assert segment.range.start <= word.start
                assert word.end <= segment.range.end

    def test_a_backend_without_word_timings_still_works(self, tmp_path: Path, locator) -> None:
        """SAPI reports none. Everything except karaoke works from durations alone."""
        result = _narrate(
            tmp_path, locator, "One.\n\nTwo.", synthesizer=FakeSynthesizer(words=False)
        )
        assert result.analysis.transcript.has_word_timings is False
        assert len(result.analysis.beats) == 2

    def test_the_analysis_validates_as_the_real_document(self, tmp_path: Path, locator) -> None:
        """It must be the same NarrationAnalysis `plan brief` reads, not a lookalike."""
        from app.models.speech import NarrationAnalysis

        result = _narrate(tmp_path, locator, "One.\n\nTwo.")
        assert NarrationAnalysis.model_validate_json(result.analysis.model_dump_json())

    def test_the_backend_is_recorded_in_the_version(self, tmp_path: Path, locator) -> None:
        """Voices differ in pace, so a cached analysis from another backend is not reusable."""
        result = _narrate(tmp_path, locator, "One.")
        assert "fake" in result.analysis.analyzer_version

    def test_per_line_audio_is_kept_for_redoing_one_line(self, tmp_path: Path, locator) -> None:
        result = _narrate(tmp_path, locator, "One.\n\nTwo.")
        assert len(result.lines) == 2
        assert all(line.audio.is_file() for line in result.lines)

    def test_an_empty_script_is_refused(self, tmp_path: Path, locator) -> None:
        from app.analysis.speech.synth.base import SynthesisError
        from app.analysis.speech.synth.narrator import narrate

        with pytest.raises(SynthesisError, match="no script lines"):
            narrate(
                (),
                synthesizer=FakeSynthesizer(),
                voice="fake-vi",
                destination=tmp_path / "n.wav",
                locator=locator,
            )


class TestNarratorVersion:
    def test_it_names_the_backend(self) -> None:
        assert "edge" in narrator_version("edge")
        assert "sapi" in narrator_version("sapi")

    def test_it_differs_per_backend(self) -> None:
        """So re-narrating with another voice invalidates the cached analysis."""
        assert narrator_version("edge") != narrator_version("sapi")


# --------------------------------------------------------------------------- #
# Backend details, without calling them
# --------------------------------------------------------------------------- #


class TestEdgeBackend:
    def test_it_declares_that_it_calls_the_network(self) -> None:
        """AIVE's core promise is that it makes none, so this must be askable."""
        assert EdgeSynthesizer().requires_network is True

    def test_vietnamese_has_a_default_voice(self) -> None:
        assert EdgeSynthesizer().default_voice("vi") == "vi-VN-HoaiMyNeural"

    def test_a_locale_tag_is_accepted_as_well_as_a_language(self) -> None:
        assert EdgeSynthesizer().default_voice("vi-VN") == DEFAULT_VOICES["vi"]

    def test_an_unlisted_language_refuses_rather_than_guessing(self) -> None:
        """Picking the alphabetically first xx-* voice would be arbitrary and occasionally
        absurd, and the user would only find out by listening."""
        with pytest.raises(VoiceNotFoundError, match="no default voice"):
            EdgeSynthesizer().default_voice("sw")

    @pytest.mark.parametrize(
        ("rate", "expected"), [(1.0, "+0%"), (1.25, "+25%"), (0.8, "-20%"), (2.0, "+100%")]
    )
    def test_the_rate_carries_a_mandatory_sign(self, rate: float, expected: str) -> None:
        """Edge rejects a bare ``25%``; the sign is not optional."""
        assert _rate_string(rate) == expected


class TestSapiBackend:
    def test_it_makes_no_network_call(self) -> None:
        assert SapiSynthesizer().requires_network is False

    def test_it_knows_whether_it_can_run(self) -> None:
        assert SapiSynthesizer().available == (sys.platform == "win32")

    @pytest.mark.parametrize(
        ("rate", "expected"), [(1.0, 0), (1.33, 1), (0.75, -1), (100.0, 10), (0.001, -10)]
    )
    def test_the_rate_maps_onto_sapis_clamped_integer_scale(
        self, rate: float, expected: int
    ) -> None:
        assert _rate_to_sapi(rate) == expected

    def test_a_nonsense_rate_does_not_raise(self) -> None:
        assert _rate_to_sapi(0.0) == 0
        assert _rate_to_sapi(-1.0) == 0

    @pytest.mark.skipif(sys.platform != "win32", reason="System.Speech is Windows-only")
    def test_it_reports_the_voices_windows_actually_has(self) -> None:
        """Reported rather than assumed: a stock Windows has English only, and a user
        narrating Vietnamese needs to find that out before a render, not after."""
        voices = SapiSynthesizer().voices()
        assert all(item.name for item in voices)

    @pytest.mark.skipif(sys.platform != "win32", reason="System.Speech is Windows-only")
    def test_a_language_with_no_installed_voice_refuses_with_instructions(self) -> None:
        """Reading Vietnamese with an American voice is confidently unusable, so falling
        back to English would waste a whole render."""
        synthesizer = SapiSynthesizer()
        installed = {item.language for item in synthesizer.voices()}
        missing = next((code for code in ("vi", "ja", "th", "hu") if code not in installed), None)
        if missing is None:  # pragma: no cover - a very well-equipped machine
            pytest.skip("every candidate language is installed")
        with pytest.raises(VoiceNotFoundError, match="Add voices"):
            synthesizer.default_voice(missing)


# --------------------------------------------------------------------------- #
# The bug the whole suite missed
# --------------------------------------------------------------------------- #


class TestUtf8Stdout:
    """AIVE could not print its own output for a non-English project.

    A Windows console gives Python ``cp1252``, which cannot encode Vietnamese, so
    ``sys.stdout.write`` raised ``UnicodeEncodeError`` and the command exited 70 with the
    message "This is a bug in AIVE". It was — and 1290 tests passed straight over it, because
    ``CliRunner`` captures output in memory as UTF-8 and never touches a real console.

    The lesson is narrow and worth keeping: a test harness that replaces a stream does not
    test that stream's encoding.
    """

    def test_force_utf8_pins_stdout(self) -> None:
        import io

        from app.cli.output import force_utf8

        original = sys.stdout
        try:
            sys.stdout = io.TextIOWrapper(io.BytesIO(), encoding="cp1252")
            force_utf8()
            assert sys.stdout.encoding.lower().replace("-", "") == "utf8"
        finally:
            sys.stdout = original

    def test_a_stream_that_cannot_be_reconfigured_is_left_alone(self) -> None:
        """A pytest-captured or pipe-wrapped stream may lack ``reconfigure``; that must not
        crash the CLI before it has printed anything."""
        import io

        from app.cli.output import force_utf8

        original = sys.stdout
        try:
            sys.stdout = io.StringIO()  # no reconfigure attribute
            force_utf8()  # must not raise
        finally:
            sys.stdout = original

    def test_vietnamese_survives_a_cp1252_console(self) -> None:
        """The end-to-end property, exercised on the real byte stream rather than a capture."""
        import io

        from app.cli.output import emit_digest, force_utf8

        buffer = io.BytesIO()
        original = sys.stdout
        try:
            sys.stdout = io.TextIOWrapper(buffer, encoding="cp1252")
            force_utf8()
            emit_digest([f"b000 | {VIETNAMESE}"])
            sys.stdout.flush()
            # Read before restoring: reassigning sys.stdout drops the last reference to the
            # wrapper, which closes the BytesIO underneath it.
            written = buffer.getvalue()
        finally:
            sys.stdout = original

        assert VIETNAMESE in written.decode("utf-8")

    def test_cp1252_really_cannot_encode_vietnamese(self) -> None:
        """Pins the premise. Without this the tests above could pass for the wrong reason."""
        with pytest.raises(UnicodeEncodeError):
            VIETNAMESE.encode("cp1252")
