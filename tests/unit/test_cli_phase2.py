"""Tests for the Phase 2 CLI: ``analyze audio`` and ``subtitle build``.

The recogniser is always faked. Downloading model weights in a test suite would make it
slow, network-dependent, and non-deterministic — and the recogniser's own behaviour is
covered in :mod:`tests.unit.test_whisper_recognizer`. What is under test here is the
command surface: the stdout contract, the digest, caching, and the error paths.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from app.cli import analyze_cmd
from app.cli.main import app
from app.models.common import MediaRef, TimeRange
from app.models.speech import NarrationAnalysis, Transcript, TranscriptSegment, Word
from app.services.container import Container, build_container
from app.services.paths import ProjectPaths

runner = CliRunner()


def _word(text: str, start: float, end: float) -> Word:
    return Word(text=text, start=start, end=end)


def _fake_transcript(ref: MediaRef, model: str = "medium") -> Transcript:
    """A transcript with a filler, a retake, and a gap between sentences."""
    return Transcript(
        source=ref,
        language="en",
        duration=9.0,
        # Must reflect the configured model, exactly as a real recogniser does: the
        # cache is invalidated on a model change, and a fake that always claims one
        # size would make that check untestable.
        model_name=f"faster-whisper/{model}",
        segments=(
            TranscriptSegment(
                index=0,
                range=TimeRange(start=0.0, end=2.2),
                text="First, um, prepare the soil.",
                words=(
                    _word("First,", 0.0, 0.4),
                    _word("um,", 0.5, 0.8),
                    _word("prepare", 1.0, 1.5),
                    _word("the", 1.55, 1.65),
                    _word("soil.", 1.7, 2.2),
                ),
                no_speech_prob=0.02,
                compression_ratio=1.4,
            ),
            TranscriptSegment(
                index=1,
                range=TimeRange(start=5.0, end=7.7),
                text="Then water it. Then water it well.",
                words=(
                    _word("Then", 5.0, 5.25),
                    _word("water", 5.3, 5.7),
                    _word("it.", 5.75, 5.95),
                    _word("Then", 6.3, 6.55),
                    _word("water", 6.6, 7.0),
                    _word("it", 7.05, 7.2),
                    _word("well.", 7.25, 7.7),
                ),
                no_speech_prob=0.03,
                compression_ratio=1.5,
            ),
        ),
    )


class FakeRecognizer:
    """Returns a canned transcript. Loads no model and reads no audio."""

    def __init__(self, transcript_for: object = None) -> None:
        self.calls = 0
        self.model = "medium"
        """Set by the patched container from the resolved settings, so ``--model``
        changes the reported name just as it would with a real recogniser."""
        self._override = transcript_for

    @property
    def name(self) -> str:
        return f"faster-whisper/{self.model}"

    def transcribe(self, audio: Path, *, ref: MediaRef) -> Transcript:
        self.calls += 1
        if callable(self._override):
            return self._override(ref)
        return _fake_transcript(ref, self.model)


class FakeSilence:
    """Silence between the sentences, plus trailing silence."""

    @property
    def name(self) -> str:
        return "fake"

    def detect(self, audio: Path, *, threshold_db: float, min_duration: float) -> tuple:
        from app.models.speech import SilenceSpan

        return (
            SilenceSpan(range=TimeRange(start=2.2, end=5.0)),
            SilenceSpan(range=TimeRange(start=7.7, end=9.0)),
        )


@pytest.fixture
def narration_project(paths: ProjectPaths) -> ProjectPaths:
    (paths.root / "narration.wav").write_bytes(b"RIFF" + b"\0" * 512)
    (paths.raw / "001.mp4").write_bytes(b"\0" * 1024)
    return paths


@pytest.fixture
def fake_speech(monkeypatch: pytest.MonkeyPatch) -> FakeRecognizer:
    """Replace the real recogniser and silence detector inside the CLI's container."""
    recognizer = FakeRecognizer()

    def patched(project_dir: Path | None = None, *, settings: object = None) -> Container:
        from app.analysis.speech.cleanup import NarrationCleanupService

        real = build_container(project_dir, settings=settings)  # type: ignore[arg-type]
        silence = FakeSilence()
        recognizer.model = real.settings.speech.model
        return Container(
            settings=real.settings,
            paths=real.paths,
            ffmpeg=real.ffmpeg,
            exporters=real.exporters,
            recognizer=recognizer,
            silence=silence,
            cleaner=NarrationCleanupService(silence, real.settings.rules),
        )

    monkeypatch.setattr(analyze_cmd, "build_container", patched)
    return recognizer


class TestAnalyzeAudio:
    def test_writes_a_document_and_prints_a_digest(
        self, narration_project: ProjectPaths, fake_speech: FakeRecognizer
    ) -> None:
        result = runner.invoke(app, ["analyze", "audio", str(narration_project.root)])
        assert result.exit_code == 0, result.stderr

        assert narration_project.narration_file.is_file()
        analysis = NarrationAnalysis.model_validate_json(
            narration_project.narration_file.read_text(encoding="utf-8")
        )
        assert analysis.transcript.language == "en"
        assert len(analysis.beats) >= 2

        # Digest, not JSON, on stdout.
        assert result.stdout.startswith("# narration=")
        assert "wordtimings=true" in result.stdout

    def test_the_digest_carries_beat_text_and_both_clocks(
        self, narration_project: ProjectPaths, fake_speech: FakeRecognizer
    ) -> None:
        """The director reads this instead of the full document."""
        result = runner.invoke(app, ["analyze", "audio", str(narration_project.root)])
        assert "b000 src=" in result.stdout
        assert " tl=" in result.stdout
        assert "prepare the soil" in result.stdout
        # The removed retake is reported as cut rather than silently missing.
        assert "tl=CUT" in result.stdout

    def test_cuts_are_summarised(
        self, narration_project: ProjectPaths, fake_speech: FakeRecognizer
    ) -> None:
        result = runner.invoke(app, ["analyze", "audio", str(narration_project.root)])
        assert "# cuts:" in result.stdout
        assert "filler:" in result.stdout
        assert "retake:" in result.stdout

    def test_full_emits_the_whole_document_as_json(
        self, narration_project: ProjectPaths, fake_speech: FakeRecognizer
    ) -> None:
        result = runner.invoke(app, ["analyze", "audio", str(narration_project.root), "--full"])
        payload = json.loads(result.stdout)
        assert NarrationAnalysis.model_validate(payload).source.name == "narration.wav"

    def test_the_digest_is_much_smaller_than_the_document(
        self, narration_project: ProjectPaths, fake_speech: FakeRecognizer
    ) -> None:
        digest = runner.invoke(app, ["analyze", "audio", str(narration_project.root)])
        full = runner.invoke(
            app, ["analyze", "audio", str(narration_project.root), "--full", "--force"]
        )
        assert len(digest.stdout) < len(full.stdout) / 2

    def test_no_cleanup_keeps_the_whole_narration(
        self, narration_project: ProjectPaths, fake_speech: FakeRecognizer
    ) -> None:
        result = runner.invoke(
            app, ["analyze", "audio", str(narration_project.root), "--no-cleanup"]
        )
        assert result.exit_code == 0
        analysis = NarrationAnalysis.model_validate_json(
            narration_project.narration_file.read_text(encoding="utf-8")
        )
        assert analysis.cleanup.kept_ranges == (TimeRange(start=0.0, end=9.0),)
        assert analysis.cleanup.removed_duration == pytest.approx(0.0)
        # With nothing removed, both clocks coincide.
        assert all(beat.survives_cleanup for beat in analysis.beats)

    def test_logs_go_to_stderr_only(
        self, narration_project: ProjectPaths, fake_speech: FakeRecognizer
    ) -> None:
        result = runner.invoke(app, ["analyze", "audio", str(narration_project.root)])
        assert "language:" in result.stderr
        assert "language:" not in result.stdout


class TestCaching:
    def test_a_second_run_reuses_the_cache(
        self, narration_project: ProjectPaths, fake_speech: FakeRecognizer
    ) -> None:
        """Transcription costs minutes, so this matters."""
        runner.invoke(app, ["analyze", "audio", str(narration_project.root)])
        assert fake_speech.calls == 1
        result = runner.invoke(app, ["analyze", "audio", str(narration_project.root)])
        assert result.exit_code == 0
        assert fake_speech.calls == 1, "the recogniser was called again"
        assert "Reusing cached analysis" in result.stderr

    def test_force_re_runs(
        self, narration_project: ProjectPaths, fake_speech: FakeRecognizer
    ) -> None:
        runner.invoke(app, ["analyze", "audio", str(narration_project.root)])
        runner.invoke(app, ["analyze", "audio", str(narration_project.root), "--force"])
        assert fake_speech.calls == 2

    def test_newer_narration_invalidates_the_cache(
        self, narration_project: ProjectPaths, fake_speech: FakeRecognizer
    ) -> None:
        """Reusing a stale result would silently mis-time every subtitle."""
        import os
        import time

        runner.invoke(app, ["analyze", "audio", str(narration_project.root)])
        assert fake_speech.calls == 1

        narration = narration_project.root / "narration.wav"
        future = time.time() + 60
        os.utime(narration, (future, future))

        runner.invoke(app, ["analyze", "audio", str(narration_project.root)])
        assert fake_speech.calls == 2

    def test_a_different_model_invalidates_the_cache(
        self, narration_project: ProjectPaths, fake_speech: FakeRecognizer
    ) -> None:
        runner.invoke(app, ["analyze", "audio", str(narration_project.root)])
        result = runner.invoke(
            app, ["analyze", "audio", str(narration_project.root), "--model", "small"]
        )
        assert result.exit_code == 0
        assert fake_speech.calls == 2

    def test_a_corrupt_cache_is_ignored_rather_than_fatal(
        self, narration_project: ProjectPaths, fake_speech: FakeRecognizer
    ) -> None:
        narration_project.narration_file.parent.mkdir(parents=True, exist_ok=True)
        narration_project.narration_file.write_text("{not json", encoding="utf-8")
        result = runner.invoke(app, ["analyze", "audio", str(narration_project.root)])
        assert result.exit_code == 0
        assert fake_speech.calls == 1

    def test_a_version_bump_invalidates_the_cache(
        self,
        narration_project: ProjectPaths,
        fake_speech: FakeRecognizer,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        runner.invoke(app, ["analyze", "audio", str(narration_project.root)])
        monkeypatch.setattr(analyze_cmd, "ANALYSIS_VERSION", "something-else/99")
        runner.invoke(app, ["analyze", "audio", str(narration_project.root)])
        assert fake_speech.calls == 2


class TestAnalyzeAudioErrors:
    def test_an_uninitialised_project(self, tmp_path: Path) -> None:
        blank = tmp_path / "blank"
        blank.mkdir()
        result = runner.invoke(app, ["analyze", "audio", str(blank)])
        assert result.exit_code == 3
        assert json.loads(result.stdout)["error"]["code"] == "project.not_initialised"

    def test_missing_narration_names_the_accepted_filenames(self, paths: ProjectPaths) -> None:
        result = runner.invoke(app, ["analyze", "audio", str(paths.root)])
        assert result.exit_code == 3
        error = json.loads(result.stdout)["error"]
        assert error["code"] == "narration.missing"
        assert "narration.wav" in error["hint"]

    def test_no_recognised_speech_is_an_input_error_not_a_crash(
        self, narration_project: ProjectPaths, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def empty(ref: MediaRef) -> Transcript:
            return Transcript(
                source=ref,
                language="en",
                duration=9.0,
                model_name="faster-whisper/tiny",
                segments=(),
            )

        def patched(project_dir: Path | None = None, *, settings: object = None) -> Container:
            real = build_container(project_dir, settings=settings)  # type: ignore[arg-type]
            return Container(
                settings=real.settings,
                paths=real.paths,
                ffmpeg=real.ffmpeg,
                exporters=real.exporters,
                recognizer=FakeRecognizer(empty),
                silence=FakeSilence(),
                cleaner=real.cleaner,
            )

        monkeypatch.setattr(analyze_cmd, "build_container", patched)
        result = runner.invoke(app, ["analyze", "audio", str(narration_project.root)])
        assert result.exit_code == 4
        error = json.loads(result.stdout)["error"]
        assert error["code"] == "narration.no_speech"
        assert error["details"]["duration"] == 9.0


class TestSubtitleBuild:
    def _analyse(self, paths: ProjectPaths) -> None:
        runner.invoke(app, ["analyze", "audio", str(paths.root)])

    def test_writes_both_formats_by_default(
        self, narration_project: ProjectPaths, fake_speech: FakeRecognizer
    ) -> None:
        self._analyse(narration_project)
        result = runner.invoke(app, ["subtitle", "build", str(narration_project.root)])
        assert result.exit_code == 0, result.stderr
        assert narration_project.subtitle_file("srt").is_file()
        assert narration_project.subtitle_file("ass").is_file()

    def test_cues_are_in_timeline_time(
        self, narration_project: ProjectPaths, fake_speech: FakeRecognizer
    ) -> None:
        """The first cue must start at 0, not at the narration's first word."""
        self._analyse(narration_project)
        runner.invoke(app, ["subtitle", "build", str(narration_project.root)])
        srt = narration_project.subtitle_file("srt").read_text(encoding="utf-8-sig")
        assert "00:00:00,000 -->" in srt

    def test_the_filler_is_absent_from_the_subtitle(
        self, narration_project: ProjectPaths, fake_speech: FakeRecognizer
    ) -> None:
        """It was cut from the audio, so it must be cut from the text too."""
        self._analyse(narration_project)
        runner.invoke(app, ["subtitle", "build", str(narration_project.root)])
        srt = narration_project.subtitle_file("srt").read_text(encoding="utf-8-sig")
        assert "um" not in srt.lower().replace("number", "")

    def test_a_single_format_can_be_selected(
        self, narration_project: ProjectPaths, fake_speech: FakeRecognizer
    ) -> None:
        self._analyse(narration_project)
        result = runner.invoke(app, ["subtitle", "build", str(narration_project.root), "-f", "srt"])
        assert result.exit_code == 0
        assert narration_project.subtitle_file("srt").is_file()
        assert not narration_project.subtitle_file("ass").is_file()

    def test_raw_timing_uses_source_time(
        self, narration_project: ProjectPaths, fake_speech: FakeRecognizer
    ) -> None:
        self._analyse(narration_project)
        result = runner.invoke(
            app, ["subtitle", "build", str(narration_project.root), "--raw", "-f", "srt"]
        )
        assert result.exit_code == 0
        assert "cleanup ignored" in result.stderr
        srt = narration_project.subtitle_file("srt").read_text(encoding="utf-8-sig")
        # The retake survives when cleanup is ignored.
        assert srt.lower().count("then water it") >= 2

    def test_the_digest_lists_every_cue(
        self, narration_project: ProjectPaths, fake_speech: FakeRecognizer
    ) -> None:
        self._analyse(narration_project)
        result = runner.invoke(app, ["subtitle", "build", str(narration_project.root)])
        assert result.stdout.startswith("# cues=")
        assert "c000 " in result.stdout

    def test_full_emits_cue_json(
        self, narration_project: ProjectPaths, fake_speech: FakeRecognizer
    ) -> None:
        self._analyse(narration_project)
        result = runner.invoke(app, ["subtitle", "build", str(narration_project.root), "--full"])
        payload = json.loads(result.stdout)
        assert payload["cues"]
        assert "range" in payload["cues"][0]

    def test_karaoke_follows_project_config(
        self, narration_project: ProjectPaths, fake_speech: FakeRecognizer
    ) -> None:
        self._analyse(narration_project)
        narration_project.config_file.write_text("[subtitle]\nkaraoke = true\n", encoding="utf-8")
        runner.invoke(app, ["subtitle", "build", str(narration_project.root), "-f", "ass"])
        assert "\\k" in narration_project.subtitle_file("ass").read_text(encoding="utf-8")

    def test_a_plan_supplies_the_timeline_when_given(
        self, narration_project: ProjectPaths, fake_speech: FakeRecognizer
    ) -> None:
        from app.models.edit_plan import EditPlan, NarrationTrack, TimelineClip

        self._analyse(narration_project)
        plan = EditPlan(
            project_id="test",
            created_by="pytest",
            narration=NarrationTrack(
                source=MediaRef(path="narration.wav"),
                # Keep only the first sentence.
                kept_ranges=(TimeRange(start=0.0, end=2.2),),
            ),
            clips=(
                TimelineClip(
                    id="c1",
                    source=MediaRef(path="raw/001.mp4"),
                    source_range=TimeRange(start=0.0, end=2.2),
                    reason="only clip",
                ),
            ),
        )
        plan_path = narration_project.root / "edit_plan.json"
        plan_path.write_text(plan.model_dump_json(), encoding="utf-8")

        result = runner.invoke(
            app,
            [
                "subtitle",
                "build",
                str(narration_project.root),
                "--plan",
                str(plan_path),
                "-f",
                "srt",
            ],
        )
        assert result.exit_code == 0
        assert "plan edit_plan.json" in result.stderr
        srt = narration_project.subtitle_file("srt").read_text(encoding="utf-8-sig")
        # Everything after 2.2s was excluded by the plan.
        assert "water" not in srt.lower()


class TestSubtitleBuildErrors:
    def test_no_analysis_yet(self, narration_project: ProjectPaths) -> None:
        result = runner.invoke(app, ["subtitle", "build", str(narration_project.root)])
        assert result.exit_code == 3
        error = json.loads(result.stdout)["error"]
        assert error["code"] == "narration.not_analysed"
        assert "aive analyze audio" in error["hint"]

    def test_an_unknown_format(
        self, narration_project: ProjectPaths, fake_speech: FakeRecognizer
    ) -> None:
        runner.invoke(app, ["analyze", "audio", str(narration_project.root)])
        result = runner.invoke(app, ["subtitle", "build", str(narration_project.root), "-f", "vtt"])
        assert result.exit_code == 2
        error = json.loads(result.stdout)["error"]
        assert error["code"] == "subtitle.unknown_format"
        assert "srt" in error["hint"]

    def test_a_corrupt_analysis_document(self, narration_project: ProjectPaths) -> None:
        narration_project.narration_file.parent.mkdir(parents=True, exist_ok=True)
        narration_project.narration_file.write_text('{"bad": true}', encoding="utf-8")
        result = runner.invoke(app, ["subtitle", "build", str(narration_project.root)])
        assert result.exit_code == 4
        assert json.loads(result.stdout)["error"]["code"] == "narration.unreadable"

    def test_a_missing_plan(
        self, narration_project: ProjectPaths, fake_speech: FakeRecognizer
    ) -> None:
        runner.invoke(app, ["analyze", "audio", str(narration_project.root)])
        result = runner.invoke(
            app,
            [
                "subtitle",
                "build",
                str(narration_project.root),
                "--plan",
                str(narration_project.root / "absent.json"),
            ],
        )
        assert result.exit_code == 3
        assert json.loads(result.stdout)["error"]["code"] == "plan.missing"


class TestPhase2CommandsAreDiscoverable:
    def test_analyze_and_subtitle_appear_in_help(self) -> None:
        """An AI director learns what exists from --help."""
        result = runner.invoke(app, ["--help"])
        assert "analyze" in result.output
        assert "subtitle" in result.output

    def test_the_audio_analyser_is_advertised(self) -> None:
        """``--help`` is where a director discovers what AIVE can do.

        This began as an assertion that ``music`` was *absent*, guarding the rule that a
        command which exists but does nothing is a trap. Phase 7 implemented it, so the
        rule is now enforced by the exact-registry test in ``test_cli_phase4``, and what is
        left to check here is that Phase 2's own command is discoverable.
        """
        result = runner.invoke(app, ["analyze", "--help"])
        assert "audio" in result.output
