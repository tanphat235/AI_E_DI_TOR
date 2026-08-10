"""Tests for project layout, media discovery, logging and the DI container."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from app.config.settings import AiveSettings, LogLevel
from app.exporters.base import ExporterRegistry, ExportRequest
from app.models.common import MediaKind, MediaRef
from app.services.container import build_container
from app.services.paths import ProjectPaths, classify
from app.utils.logging import LOGGER_NAME, configure_logging, get_logger, new_run_id, stage


class TestClassify:
    @pytest.mark.parametrize(
        ("filename", "expected"),
        [
            ("001.mp4", MediaKind.RAW_VIDEO),
            ("clip.MOV", MediaKind.RAW_VIDEO),
            ("drone.mkv", MediaKind.RAW_VIDEO),
            ("narration.wav", MediaKind.NARRATION),
            ("voice.mp3", MediaKind.NARRATION),
            ("vo.m4a", MediaKind.NARRATION),
            ("calm_forest.mp3", MediaKind.MUSIC),
            ("notes.txt", MediaKind.UNKNOWN),
            ("thumbnail.png", MediaKind.UNKNOWN),
        ],
    )
    def test_classification(self, filename: str, expected: MediaKind) -> None:
        assert classify(Path(filename)) is expected


class TestProjectPaths:
    def test_layout(self, tmp_path: Path) -> None:
        paths = ProjectPaths.for_root(tmp_path)
        assert paths.raw == tmp_path / "raw"
        assert paths.cache == tmp_path / ".aive"
        assert paths.logs == tmp_path / "output" / "logs"
        assert paths.keyframes == tmp_path / ".aive" / "keyframes"
        assert paths.final_video == tmp_path / "output" / "final.mp4"

    def test_subtitle_file_accepts_either_form(self, tmp_path: Path) -> None:
        paths = ProjectPaths.for_root(tmp_path)
        assert paths.subtitle_file("srt").name == "subtitle.srt"
        assert paths.subtitle_file(".ass").name == "subtitle.ass"

    def test_ensure_is_idempotent(self, tmp_path: Path) -> None:
        paths = ProjectPaths.for_root(tmp_path)
        first = paths.ensure()
        # tmp_path already exists, so the root is not reported as newly created.
        assert set(first) == set(paths.all_directories()) - {paths.root}
        assert all(directory.is_dir() for directory in paths.all_directories())
        assert paths.ensure() == ()

    def test_exists_requires_the_raw_directory(self, tmp_path: Path) -> None:
        paths = ProjectPaths.for_root(tmp_path)
        assert not paths.exists()
        paths.ensure()
        assert paths.exists()

    def test_to_ref_and_back(self, tmp_path: Path) -> None:
        paths = ProjectPaths.for_root(tmp_path)
        paths.ensure()
        clip = paths.raw / "001.mp4"
        clip.touch()
        ref = paths.to_ref(clip)
        assert ref == MediaRef(path="raw/001.mp4")
        assert paths.resolve(ref) == clip.resolve()

    def test_resolve_refuses_to_escape_the_project(self, tmp_path: Path) -> None:
        """Defence in depth.

        ``MediaRef`` validation already rejects ``..``, so this constructs one via
        ``model_construct`` to bypass it and prove the *second* barrier holds. A
        plan is untrusted input, so one check is not enough.
        """
        paths = ProjectPaths.for_root(tmp_path)
        escaping = MediaRef.model_construct(path=Path("../escape.mp4"))
        with pytest.raises(ValueError, match="resolves outside the project root"):
            paths.resolve(escaping)


class TestDiscovery:
    def test_narration_at_the_project_root(self, paths: ProjectPaths) -> None:
        (paths.root / "narration.wav").touch()
        found = paths.find_narration()
        assert found is not None
        assert found.name == "narration.wav"

    def test_narration_in_the_audio_folder(self, paths: ProjectPaths) -> None:
        """The brief showed both layouts, so both are supported."""
        paths.audio.mkdir()
        (paths.audio / "voice.wav").touch()
        found = paths.find_narration()
        assert found is not None
        assert found.name == "voice.wav"

    def test_a_root_narration_wins_over_the_audio_folder(self, paths: ProjectPaths) -> None:
        (paths.root / "narration.wav").touch()
        paths.audio.mkdir()
        (paths.audio / "something.wav").touch()
        found = paths.find_narration()
        assert found is not None
        assert found.name == "narration.wav"

    def test_stem_preference_order_is_honoured(self, paths: ProjectPaths) -> None:
        (paths.root / "voice.wav").touch()
        (paths.root / "narration.wav").touch()
        found = paths.find_narration()
        assert found is not None
        assert found.name == "narration.wav"

    def test_no_narration_returns_none(self, paths: ProjectPaths) -> None:
        assert paths.find_narration() is None

    def test_raw_clips_are_sorted_by_name(self, paths: ProjectPaths) -> None:
        """Users number footage in shooting order, which is an editorial hint."""
        for name in ("003.mp4", "001.mp4", "002.mp4"):
            (paths.raw / name).touch()
        assert [path.name for path in paths.find_raw_clips()] == ["001.mp4", "002.mp4", "003.mp4"]

    def test_raw_discovery_recurses_and_ignores_non_video(self, paths: ProjectPaths) -> None:
        (paths.raw / "drone").mkdir()
        (paths.raw / "drone" / "aerial.mp4").touch()
        (paths.raw / "notes.txt").touch()
        (paths.raw / "001.mp4").touch()
        found = [path.name for path in paths.find_raw_clips()]
        assert set(found) == {"001.mp4", "aerial.mp4"}

    def test_music_discovery(self, paths: ProjectPaths) -> None:
        (paths.music / "calm.mp3").touch()
        (paths.music / "upbeat.wav").touch()
        (paths.music / "cover.jpg").touch()
        assert len(paths.find_music()) == 2

    def test_capcut_templates_need_the_marker_file(self, paths: ProjectPaths) -> None:
        """A CapCut draft is a directory identified by draft_content.json."""
        real = paths.capcut / "my_template"
        real.mkdir()
        (real / "draft_content.json").write_text("{}", encoding="utf-8")
        (paths.capcut / "just_a_folder").mkdir()

        found = paths.find_capcut_templates()
        assert len(found) == 1
        assert found[0].name == "my_template"

    def test_no_capcut_folder_is_not_an_error(self, tmp_path: Path) -> None:
        assert ProjectPaths.for_root(tmp_path).find_capcut_templates() == ()


class TestLogging:
    def test_run_ids_are_unique_and_sortable(self) -> None:
        first, second = new_run_id(), new_run_id()
        assert first != second
        assert len(first.split("-")) == 3

    def test_get_logger_is_always_under_the_aive_namespace(self) -> None:
        assert get_logger("app.services.paths").name == f"{LOGGER_NAME}.services.paths"
        assert get_logger().name == LOGGER_NAME

    def test_nothing_is_written_to_stdout(
        self, capsys: pytest.CaptureFixture[str], tmp_path: Path
    ) -> None:
        """The single most important property in this module."""
        configure_logging(level=LogLevel.DEBUG, log_file=tmp_path / "run.jsonl")
        logger = get_logger("test")
        logger.info("an informational message")
        logger.warning("a warning")
        logger.error("an error")

        captured = capsys.readouterr()
        assert captured.out == "", f"logging leaked to stdout: {captured.out!r}"
        assert "an informational message" in captured.err

    def test_jsonl_file_is_machine_readable(self, tmp_path: Path) -> None:
        log_file = tmp_path / "logs" / "run.jsonl"
        run_id = configure_logging(level=LogLevel.INFO, log_file=log_file)
        get_logger("test").info("hello", extra={"clip": "001.mp4", "scene": 3})
        logging.getLogger(LOGGER_NAME).handlers[-1].flush()

        records = [json.loads(line) for line in log_file.read_text(encoding="utf-8").splitlines()]
        assert records
        record = records[-1]
        assert record["message"] == "hello"
        assert record["run_id"] == run_id
        assert record["context"]["clip"] == "001.mp4"

    def test_debug_reaches_the_file_even_when_the_console_is_quiet(self, tmp_path: Path) -> None:
        log_file = tmp_path / "run.jsonl"
        configure_logging(level=LogLevel.ERROR, log_file=log_file)
        get_logger("test").debug("a detail worth keeping")
        logging.getLogger(LOGGER_NAME).handlers[-1].flush()
        assert "a detail worth keeping" in log_file.read_text(encoding="utf-8")

    def test_configure_is_idempotent(self, tmp_path: Path) -> None:
        """The desktop UI configures repeatedly; handlers must not accumulate."""
        for _ in range(3):
            configure_logging(level=LogLevel.INFO, log_file=tmp_path / "run.jsonl")
        assert len(logging.getLogger(LOGGER_NAME).handlers) == 2

    def test_an_unwritable_log_file_does_not_abort_the_run(self, tmp_path: Path) -> None:
        """A missing log destination must never stop the actual work."""
        blocker = tmp_path / "blocked"
        blocker.write_text("i am a file, not a directory", encoding="utf-8")
        configure_logging(level=LogLevel.INFO, log_file=blocker / "sub" / "run.jsonl")
        get_logger("test").info("still working")

    def test_stage_logs_success_with_a_duration(self, capsys: pytest.CaptureFixture[str]) -> None:
        configure_logging(level=LogLevel.INFO)
        logger = get_logger("test")
        with stage(logger, "scene detection", clip="001.mp4"):
            pass
        err = capsys.readouterr().err
        assert "scene detection: start" in err
        assert "scene detection: done" in err

    def test_stage_reports_a_failure_and_reraises(self, capsys: pytest.CaptureFixture[str]) -> None:
        configure_logging(level=LogLevel.INFO)
        logger = get_logger("test")
        with pytest.raises(RuntimeError, match="boom"), stage(logger, "risky work"):
            raise RuntimeError("boom")
        assert "risky work: failed" in capsys.readouterr().err


class TestContainer:
    def test_builds_without_a_project(self) -> None:
        container = build_container()
        assert container.paths is None
        assert container.ffmpeg is not None
        # Phase 8 registers CapCut. Registration happens in build_container rather
        # than at import time, so a test can see exactly which exporters a run had.
        assert container.exporters.names() == ("capcut",)

    def test_phase_2_components_are_wired(self) -> None:
        """Constructing them must stay free: `doctor` builds a container too.

        The recogniser imports ctranslate2 and fetches weights only on its first
        `transcribe`, so wiring it here costs nothing.
        """
        container = build_container()
        assert container.recognizer is not None
        assert container.cleaner is not None
        assert container.silence is not None

    def test_the_prober_needs_no_project(self) -> None:
        """Probing reads a file and writes nothing, so it works without a project."""
        assert build_container().prober is not None

    def test_video_analysis_needs_a_project(self) -> None:
        """It writes keyframes into the project cache, so it cannot be built without one.

        This is why `doctor` and `schema` can run anywhere while `analyze video` cannot.
        """
        assert build_container().footage is None
        assert build_container().vision is None

    def test_video_analysis_is_wired_when_a_project_is_given(self, tmp_path: Path) -> None:
        container = build_container(tmp_path)
        assert container.footage is not None
        assert container.vision is not None

    def test_the_rule_engine_needs_no_project(self) -> None:
        """It judges a plan against a context the caller supplies, not against a folder."""
        assert build_container().rules is not None

    def test_phase_seven_components_are_wired(self, tmp_path: Path) -> None:
        """Constructing both is free - the renderer resolves FFmpeg lazily inside preflight
        and the analyser imports numpy only when it decodes - so neither needs to stay None
        to keep ``aive doctor`` cheap."""
        container = build_container(tmp_path)
        assert container.renderer is not None
        assert container.music is not None

    def test_require_paths_fails_clearly_without_a_project(self) -> None:
        with pytest.raises(ValueError, match="needs a project directory"):
            build_container().require_paths()

    def test_require_paths_returns_them_when_present(self, tmp_path: Path) -> None:
        container = build_container(tmp_path)
        assert container.require_paths().root == tmp_path.resolve()

    def test_accepts_injected_settings(self, tmp_path: Path) -> None:
        """Tests must be able to bypass the config files entirely."""
        settings = AiveSettings(rules={"min_clip_duration": 7.0})
        container = build_container(tmp_path, settings=settings)
        assert container.settings.rules.min_clip_duration == 7.0

    def test_is_frozen(self, tmp_path: Path) -> None:
        container = build_container(tmp_path)
        with pytest.raises((AttributeError, TypeError)):
            container.settings = AiveSettings()  # type: ignore[misc]


class TestExporterRegistry:
    class _Fake:
        """A stand-in exporter. Deliberately does not inherit from anything."""

        def __init__(self, name: str = "fake") -> None:
            self._name = name

        @property
        def name(self) -> str:
            return self._name

        @property
        def display_name(self) -> str:
            return "Fake Exporter"

        def preflight(self, request: ExportRequest) -> tuple[str, ...]:
            return ()

        def export(self, request: ExportRequest) -> object:
            raise NotImplementedError

    def test_register_and_get(self) -> None:
        registry = ExporterRegistry()
        exporter = self._Fake()
        registry.register(exporter)
        assert registry.get("fake") is exporter
        assert registry.names() == ("fake",)
        assert "fake" in registry

    def test_duplicate_registration_is_refused(self) -> None:
        """Overwriting silently would make which exporter ran depend on import order."""
        registry = ExporterRegistry()
        registry.register(self._Fake())
        with pytest.raises(ValueError, match="already registered"):
            registry.register(self._Fake())

    def test_unknown_name_lists_the_alternatives(self) -> None:
        registry = ExporterRegistry()
        registry.register(self._Fake("capcut"))
        with pytest.raises(KeyError, match="capcut"):
            registry.get("premiere")

    def test_a_plain_object_satisfies_the_protocol(self) -> None:
        """Protocols, not base classes: an implementation need not import us."""
        from app.exporters.base import Exporter

        assert isinstance(self._Fake(), Exporter)
