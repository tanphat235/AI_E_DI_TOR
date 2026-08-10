"""Tests for the recogniser and the FFmpeg silence detector.

No model is ever downloaded and no audio is ever decoded. The recogniser's real work is
delegated to faster-whisper, which does not need re-testing; what needs testing is
everything *around* it — device selection, hallucination filtering, and the CPU
fallback — because that is where the failures a user actually hits come from.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from app.analysis.speech.silence import (
    FFmpegSilenceDetector,
    SilenceDetectionError,
    parse_silencedetect,
    trailing_silence_start,
)
from app.analysis.speech.whisper_recognizer import (
    FasterWhisperRecognizer,
    _is_cuda_environment_failure,
)
from app.config.settings import MediaSettings, SpeechSettings, WhisperDevice
from app.models.common import MediaRef
from app.services.ffmpeg_locator import FFmpegLocator

MODULE = "app.analysis.speech.whisper_recognizer"


# --------------------------------------------------------------------------- #
# Fakes standing in for faster-whisper's own objects
# --------------------------------------------------------------------------- #


@dataclass
class FakeWord:
    word: str
    start: float
    end: float
    probability: float = 0.9


@dataclass
class FakeSegment:
    text: str
    start: float
    end: float
    words: list[FakeWord] | None = None
    avg_logprob: float = -0.2
    no_speech_prob: float = 0.05
    compression_ratio: float = 1.5


@dataclass
class FakeInfo:
    language: str = "en"
    language_probability: float = 0.99
    duration: float = 10.0


class TestDeviceResolution:
    def test_explicit_cpu_is_respected(self) -> None:
        recognizer = FasterWhisperRecognizer(
            SpeechSettings(device=WhisperDevice.CPU, compute_type="int8")
        )
        assert recognizer._resolve_device() == ("cpu", "int8")

    def test_auto_pairs_cpu_with_int8(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """float16 on a CPU is unsupported or slower, so the pairing matters."""
        monkeypatch.setattr(FasterWhisperRecognizer, "_cuda_available", staticmethod(lambda: False))
        recognizer = FasterWhisperRecognizer(SpeechSettings())
        assert recognizer._resolve_device() == ("cpu", "int8")

    def test_auto_pairs_cuda_with_float16(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(FasterWhisperRecognizer, "_cuda_available", staticmethod(lambda: True))
        recognizer = FasterWhisperRecognizer(SpeechSettings())
        assert recognizer._resolve_device() == ("cuda", "float16")

    def test_an_explicit_compute_type_overrides_the_pairing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(FasterWhisperRecognizer, "_cuda_available", staticmethod(lambda: True))
        recognizer = FasterWhisperRecognizer(SpeechSettings(compute_type="int8_float16"))
        assert recognizer._resolve_device() == ("cuda", "int8_float16")

    def test_a_forced_device_wins_over_everything(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Set after a GPU failure, so later calls do not rediscover it."""
        monkeypatch.setattr(FasterWhisperRecognizer, "_cuda_available", staticmethod(lambda: True))
        recognizer = FasterWhisperRecognizer(SpeechSettings(device=WhisperDevice.CUDA))
        recognizer._forced_device = "cpu"
        assert recognizer._resolve_device()[0] == "cpu"

    def test_name_includes_the_model_size(self) -> None:
        """A tiny-model transcript must be distinguishable from a large-model one."""
        assert FasterWhisperRecognizer(SpeechSettings(model="large-v3")).name == (
            "faster-whisper/large-v3"
        )


class TestCudaProbe:
    def test_a_device_with_no_runtime_libraries_is_not_usable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The bug this guards: a device count above zero is not "CUDA works".

        A machine with an NVIDIA card and no CUDA toolkit counts one device and then
        dies mid-transcription on a missing cuBLAS.
        """
        import ctranslate2

        monkeypatch.setattr(ctranslate2, "get_cuda_device_count", lambda: 1)
        monkeypatch.setattr(f"{MODULE}._can_load_library", lambda _name: False)
        assert FasterWhisperRecognizer._cuda_available() is False

    def test_a_device_with_its_libraries_is_usable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import ctranslate2

        monkeypatch.setattr(ctranslate2, "get_cuda_device_count", lambda: 1)
        monkeypatch.setattr(f"{MODULE}._can_load_library", lambda _name: True)
        assert FasterWhisperRecognizer._cuda_available() is True

    def test_no_device_at_all(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import ctranslate2

        monkeypatch.setattr(ctranslate2, "get_cuda_device_count", lambda: 0)
        assert FasterWhisperRecognizer._cuda_available() is False

    def test_a_probe_that_raises_is_treated_as_no_gpu(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import ctranslate2

        def explode() -> int:
            raise RuntimeError("driver mismatch")

        monkeypatch.setattr(ctranslate2, "get_cuda_device_count", explode)
        assert FasterWhisperRecognizer._cuda_available() is False

    @pytest.mark.parametrize(
        "message",
        [
            "Library cublas64_12.dll is not found or cannot be loaded",
            "cuDNN failed to initialize",
            "CUDA out of memory",
            "no kernel image is available for execution on the device",
        ],
    )
    def test_cuda_failures_are_recognised(self, message: str) -> None:
        assert _is_cuda_environment_failure(RuntimeError(message)) is True

    @pytest.mark.parametrize(
        "message",
        ["list index out of range", "invalid literal for int()", "unexpected keyword argument"],
    )
    def test_ordinary_bugs_are_not_mistaken_for_cuda_failures(self, message: str) -> None:
        """A false positive here would retry on the CPU and hide a real defect."""
        assert _is_cuda_environment_failure(RuntimeError(message)) is False


class TestTranscription:
    @pytest.fixture
    def audio(self, tmp_path: Path) -> Path:
        path = tmp_path / "narration.wav"
        path.write_bytes(b"RIFF" + b"\0" * 64)
        return path

    def _recognizer(
        self,
        monkeypatch: pytest.MonkeyPatch,
        segments: list[FakeSegment],
        *,
        info: FakeInfo | None = None,
        settings: SpeechSettings | None = None,
    ) -> FasterWhisperRecognizer:
        recognizer = FasterWhisperRecognizer(settings or SpeechSettings())
        monkeypatch.setattr(
            recognizer,
            "_attempt",
            lambda _audio: (recognizer._collect_segments(segments), info or FakeInfo()),
        )
        return recognizer

    def test_a_missing_file_fails_before_loading_a_model(self, tmp_path: Path) -> None:
        recognizer = FasterWhisperRecognizer(SpeechSettings())
        with pytest.raises(FileNotFoundError, match="narration audio not found"):
            recognizer.transcribe(tmp_path / "absent.wav", ref=MediaRef(path="absent.wav"))

    def test_segments_and_words_are_converted(
        self, monkeypatch: pytest.MonkeyPatch, audio: Path
    ) -> None:
        recognizer = self._recognizer(
            monkeypatch,
            [
                FakeSegment(
                    text="  Prepare the soil.  ",
                    start=0.0,
                    end=2.0,
                    words=[
                        FakeWord("Prepare", 0.0, 0.6),
                        FakeWord(" the", 0.65, 0.8),
                        FakeWord(" soil.", 0.85, 2.0),
                    ],
                )
            ],
        )
        transcript = recognizer.transcribe(audio, ref=MediaRef(path="narration.wav"))
        assert transcript.segments[0].text == "Prepare the soil."
        assert transcript.has_word_timings
        # Leading spaces that Whisper attaches to each word must be stripped.
        assert [word.text for word in transcript.segments[0].words] == ["Prepare", "the", "soil."]

    def test_empty_and_zero_length_segments_are_discarded(
        self, monkeypatch: pytest.MonkeyPatch, audio: Path
    ) -> None:
        recognizer = self._recognizer(
            monkeypatch,
            [
                FakeSegment(text="   ", start=0.0, end=1.0),
                FakeSegment(text="Real speech.", start=1.0, end=2.0),
                FakeSegment(text="Zero length.", start=3.0, end=3.0),
            ],
        )
        transcript = recognizer.transcribe(audio, ref=MediaRef(path="narration.wav"))
        assert [segment.text for segment in transcript.segments] == ["Real speech."]

    def test_zero_width_words_are_discarded(
        self, monkeypatch: pytest.MonkeyPatch, audio: Path
    ) -> None:
        recognizer = self._recognizer(
            monkeypatch,
            [
                FakeSegment(
                    text="Two words.",
                    start=0.0,
                    end=2.0,
                    words=[
                        FakeWord("Two", 0.0, 0.5),
                        FakeWord("glitch", 1.0, 1.0),
                        FakeWord("words.", 1.2, 2.0),
                    ],
                )
            ],
        )
        transcript = recognizer.transcribe(audio, ref=MediaRef(path="narration.wav"))
        assert len(transcript.segments[0].words) == 2

    def test_indices_are_renumbered_contiguously(
        self, monkeypatch: pytest.MonkeyPatch, audio: Path
    ) -> None:
        """A dropped segment must not leave a hole in the numbering."""
        recognizer = self._recognizer(
            monkeypatch,
            [
                FakeSegment(text="Keep this.", start=0.0, end=1.0),
                FakeSegment(text="Hallucinated.", start=1.0, end=2.0, no_speech_prob=0.99),
                FakeSegment(text="Keep this too.", start=2.0, end=3.0),
            ],
        )
        transcript = recognizer.transcribe(audio, ref=MediaRef(path="narration.wav"))
        assert [segment.index for segment in transcript.segments] == [0, 1]

    def test_out_of_range_probabilities_are_clamped(
        self, monkeypatch: pytest.MonkeyPatch, audio: Path
    ) -> None:
        """Losing a 20-minute transcription to a rounding artefact would be absurd."""
        recognizer = self._recognizer(
            monkeypatch,
            [
                FakeSegment(
                    text="Word.",
                    start=0.0,
                    end=1.0,
                    words=[FakeWord("Word.", 0.0, 1.0, probability=1.0000001)],
                )
            ],
        )
        transcript = recognizer.transcribe(audio, ref=MediaRef(path="narration.wav"))
        assert transcript.segments[0].words[0].probability == 1.0

    def test_duration_falls_back_to_the_last_segment(
        self, monkeypatch: pytest.MonkeyPatch, audio: Path
    ) -> None:
        recognizer = self._recognizer(
            monkeypatch,
            [FakeSegment(text="Speech.", start=0.0, end=7.5)],
            info=FakeInfo(duration=0.0),
        )
        transcript = recognizer.transcribe(audio, ref=MediaRef(path="narration.wav"))
        assert transcript.duration == pytest.approx(7.5)

    def test_no_determinable_duration_is_an_error(
        self, monkeypatch: pytest.MonkeyPatch, audio: Path
    ) -> None:
        recognizer = self._recognizer(monkeypatch, [], info=FakeInfo(duration=0.0))
        with pytest.raises(ValueError, match="could not determine a duration"):
            recognizer.transcribe(audio, ref=MediaRef(path="narration.wav"))


class TestHallucinationFiltering:
    @pytest.fixture
    def audio(self, tmp_path: Path) -> Path:
        path = tmp_path / "narration.wav"
        path.write_bytes(b"RIFF")
        return path

    def _transcribe(
        self, monkeypatch: pytest.MonkeyPatch, audio: Path, segments: list[FakeSegment]
    ) -> Any:
        recognizer = FasterWhisperRecognizer(SpeechSettings())
        monkeypatch.setattr(
            recognizer,
            "_attempt",
            lambda _a: (recognizer._collect_segments(segments), FakeInfo()),
        )
        return recognizer.transcribe(audio, ref=MediaRef(path="narration.wav"))

    def test_high_no_speech_probability_is_dropped(
        self, monkeypatch: pytest.MonkeyPatch, audio: Path
    ) -> None:
        """Whisper invents "Thank you for watching" over room tone."""
        transcript = self._transcribe(
            monkeypatch,
            audio,
            [
                FakeSegment(text="Real speech.", start=0.0, end=1.0, no_speech_prob=0.05),
                FakeSegment(
                    text="Thank you for watching.", start=1.0, end=2.0, no_speech_prob=0.95
                ),
            ],
        )
        assert [segment.text for segment in transcript.segments] == ["Real speech."]

    def test_a_repetition_loop_is_dropped(
        self, monkeypatch: pytest.MonkeyPatch, audio: Path
    ) -> None:
        transcript = self._transcribe(
            monkeypatch,
            audio,
            [
                FakeSegment(text="Real speech.", start=0.0, end=1.0, compression_ratio=1.4),
                FakeSegment(text="la la la la la la", start=1.0, end=2.0, compression_ratio=8.0),
            ],
        )
        assert len(transcript.segments) == 1

    def test_borderline_segments_are_kept(
        self, monkeypatch: pytest.MonkeyPatch, audio: Path
    ) -> None:
        """Exactly at the threshold is not over it: keep it and let cleanup judge."""
        transcript = self._transcribe(
            monkeypatch,
            audio,
            [FakeSegment(text="Borderline.", start=0.0, end=1.0, no_speech_prob=0.75)],
        )
        assert len(transcript.segments) == 1

    def test_missing_metrics_do_not_cause_a_drop(
        self, monkeypatch: pytest.MonkeyPatch, audio: Path
    ) -> None:
        recognizer = FasterWhisperRecognizer(SpeechSettings())

        class Bare:
            text = "No metrics here."
            start = 0.0
            end = 1.0
            words = None

        monkeypatch.setattr(
            recognizer, "_attempt", lambda _a: (recognizer._collect_segments([Bare()]), FakeInfo())
        )
        transcript = recognizer.transcribe(audio, ref=MediaRef(path="narration.wav"))
        assert len(transcript.segments) == 1


class TestCpuFallback:
    @pytest.fixture
    def audio(self, tmp_path: Path) -> Path:
        path = tmp_path / "narration.wav"
        path.write_bytes(b"RIFF")
        return path

    def test_a_cuda_failure_retries_on_the_cpu(
        self, monkeypatch: pytest.MonkeyPatch, audio: Path
    ) -> None:
        """A broken CUDA install should cost speed, not the whole run."""
        recognizer = FasterWhisperRecognizer(SpeechSettings(device=WhisperDevice.CUDA))
        attempts: list[str] = []

        def attempt(_audio: Path) -> tuple[list[Any], FakeInfo]:
            device, _ = recognizer._resolve_device()
            attempts.append(device)
            if device == "cuda":
                raise RuntimeError("Library cublas64_12.dll is not found or cannot be loaded")
            return recognizer._collect_segments([FakeSegment("Recovered.", 0.0, 1.0)]), FakeInfo()

        monkeypatch.setattr(recognizer, "_attempt", attempt)
        transcript = recognizer.transcribe(audio, ref=MediaRef(path="narration.wav"))

        assert attempts == ["cuda", "cpu"]
        assert transcript.segments[0].text == "Recovered."

    def test_a_genuine_error_is_not_swallowed(
        self, monkeypatch: pytest.MonkeyPatch, audio: Path
    ) -> None:
        recognizer = FasterWhisperRecognizer(SpeechSettings(device=WhisperDevice.CUDA))

        def attempt(_audio: Path) -> tuple[list[Any], FakeInfo]:
            raise RuntimeError("list index out of range")

        monkeypatch.setattr(recognizer, "_attempt", attempt)
        with pytest.raises(RuntimeError, match="list index out of range"):
            recognizer.transcribe(audio, ref=MediaRef(path="narration.wav"))

    def test_a_cpu_failure_does_not_retry(
        self, monkeypatch: pytest.MonkeyPatch, audio: Path
    ) -> None:
        """Retrying the CPU on the CPU would loop."""
        recognizer = FasterWhisperRecognizer(SpeechSettings(device=WhisperDevice.CPU))
        calls = 0

        def attempt(_audio: Path) -> tuple[list[Any], FakeInfo]:
            nonlocal calls
            calls += 1
            raise RuntimeError("cublas is not found or cannot be loaded")

        monkeypatch.setattr(recognizer, "_attempt", attempt)
        with pytest.raises(RuntimeError):
            recognizer.transcribe(audio, ref=MediaRef(path="narration.wav"))
        assert calls == 1


# --------------------------------------------------------------------------- #
# Silence detection
# --------------------------------------------------------------------------- #


class TestSilenceParsing:
    SAMPLE = """
ffmpeg version 7.1 Copyright (c) 2000-2024 the FFmpeg developers
[silencedetect @ 0x55f1a] silence_start: 2.20015
[silencedetect @ 0x55f1a] silence_end: 5.00842 | silence_duration: 2.80827
[silencedetect @ 0x55f1a] silence_start: 7.70125
[silencedetect @ 0x55f1a] silence_end: 9.11 | silence_duration: 1.40875
size=N/A time=00:00:11.57 bitrate=N/A speed=  85x
"""

    def test_pairs_are_extracted(self) -> None:
        spans = parse_silencedetect(self.SAMPLE)
        assert len(spans) == 2
        assert spans[0].range.start == pytest.approx(2.20015)
        assert spans[0].range.end == pytest.approx(5.00842)

    def test_mean_db_is_left_unmeasured(self) -> None:
        """silencedetect reports only that a span was below the threshold."""
        assert parse_silencedetect(self.SAMPLE)[0].mean_db is None

    def test_surrounding_ffmpeg_noise_is_ignored(self) -> None:
        assert len(parse_silencedetect(self.SAMPLE)) == 2

    def test_no_silence_yields_nothing(self) -> None:
        assert parse_silencedetect("ffmpeg version 7.1\nsize=N/A time=00:00:10.00\n") == ()

    def test_an_unterminated_final_silence_is_not_invented(self) -> None:
        """The parser cannot know the file's duration, so it must not guess an end."""
        text = "[silencedetect @ 0x1] silence_start: 8.5\n"
        assert parse_silencedetect(text) == ()
        assert trailing_silence_start(text) == pytest.approx(8.5)

    def test_a_terminated_silence_reports_no_trailing_start(self) -> None:
        assert trailing_silence_start(self.SAMPLE) is None

    def test_a_negative_start_is_clamped_to_zero(self) -> None:
        text = (
            "[silencedetect @ 0x1] silence_start: -0.00012\n"
            "[silencedetect @ 0x1] silence_end: 1.5 | silence_duration: 1.5\n"
        )
        assert parse_silencedetect(text)[0].range.start == 0.0

    def test_an_end_before_its_start_is_discarded(self) -> None:
        text = (
            "[silencedetect @ 0x1] silence_start: 5.0\n"
            "[silencedetect @ 0x1] silence_end: 4.0 | silence_duration: -1.0\n"
        )
        assert parse_silencedetect(text) == ()

    def test_spans_are_ascending(self) -> None:
        spans = parse_silencedetect(self.SAMPLE)
        assert [span.range.start for span in spans] == sorted(span.range.start for span in spans)


class TestFFmpegSilenceDetector:
    def test_a_missing_file_is_reported_before_running_ffmpeg(self, tmp_path: Path) -> None:
        detector = FFmpegSilenceDetector(FFmpegLocator(MediaSettings()))
        with pytest.raises(FileNotFoundError, match="audio file not found"):
            detector.detect(tmp_path / "absent.wav", threshold_db=-35.0, min_duration=0.3)

    def test_output_is_parsed_from_stderr(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        audio = tmp_path / "narration.wav"
        audio.write_bytes(b"RIFF")

        def fake_run(command: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
            assert "silencedetect" in " ".join(command)
            return subprocess.CompletedProcess(
                command,
                0,
                stdout="",
                stderr=(
                    "[silencedetect @ 0x1] silence_start: 1.0\n"
                    "[silencedetect @ 0x1] silence_end: 2.5 | silence_duration: 1.5\n"
                ),
            )

        monkeypatch.setattr(subprocess, "run", fake_run)
        detector = FFmpegSilenceDetector(FFmpegLocator(MediaSettings()))
        spans = detector.detect(audio, threshold_db=-35.0, min_duration=0.3)
        assert len(spans) == 1

    def test_the_threshold_and_duration_reach_the_filter(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        audio = tmp_path / "narration.wav"
        audio.write_bytes(b"RIFF")
        captured: list[str] = []

        def fake_run(command: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
            captured.extend(command)
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

        monkeypatch.setattr(subprocess, "run", fake_run)
        FFmpegSilenceDetector(FFmpegLocator(MediaSettings())).detect(
            audio, threshold_db=-42.0, min_duration=0.75
        )
        assert "silencedetect=noise=-42.0dB:d=0.75" in " ".join(captured)

    def test_a_nonzero_exit_raises(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        audio = tmp_path / "narration.wav"
        audio.write_bytes(b"RIFF")

        def fake_run(command: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
            return subprocess.CompletedProcess(command, 1, stdout="", stderr="Invalid data found")

        monkeypatch.setattr(subprocess, "run", fake_run)
        detector = FFmpegSilenceDetector(FFmpegLocator(MediaSettings()))
        with pytest.raises(SilenceDetectionError, match="Invalid data found"):
            detector.detect(audio, threshold_db=-35.0, min_duration=0.3)

    def test_a_failure_to_launch_raises(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        audio = tmp_path / "narration.wav"
        audio.write_bytes(b"RIFF")

        def fake_run(command: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
            raise OSError("permission denied")

        monkeypatch.setattr(subprocess, "run", fake_run)
        detector = FFmpegSilenceDetector(FFmpegLocator(MediaSettings()))
        with pytest.raises(SilenceDetectionError, match="failed to run ffmpeg"):
            detector.detect(audio, threshold_db=-35.0, min_duration=0.3)

    def test_it_satisfies_the_protocol(self) -> None:
        from app.analysis.speech.base import SilenceDetector

        detector = FFmpegSilenceDetector(FFmpegLocator(MediaSettings()))
        assert isinstance(detector, SilenceDetector)
        assert detector.name == "ffmpeg-silencedetect"


@pytest.mark.integration
class TestRealFFmpegSilenceDetection:
    """Exercises the real binary on generated audio. Runs only with -m integration."""

    def test_detects_a_generated_silence(self, tmp_path: Path) -> None:
        locator = FFmpegLocator(MediaSettings())
        ffmpeg = locator.locate().ffmpeg
        audio = tmp_path / "tone.wav"
        # One second of tone, two of silence, one of tone.
        subprocess.run(
            [
                str(ffmpeg.path),
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "lavfi",
                "-i",
                "sine=frequency=440:duration=1",
                "-f",
                "lavfi",
                "-i",
                "anullsrc=r=44100:cl=mono:d=2",
                "-f",
                "lavfi",
                "-i",
                "sine=frequency=440:duration=1",
                "-filter_complex",
                "[0][1][2]concat=n=3:v=0:a=1",
                str(audio),
            ],
            check=True,
            capture_output=True,
            timeout=120,
        )
        spans = FFmpegSilenceDetector(locator).detect(audio, threshold_db=-35.0, min_duration=0.3)
        assert len(spans) >= 1
        assert spans[0].range.start == pytest.approx(1.0, abs=0.2)
        assert spans[0].range.end == pytest.approx(3.0, abs=0.2)
