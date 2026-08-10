"""Tests for ``aive analyze video``.

The analyser itself is faked so these run in milliseconds. What is under test is the
command surface: the digest a director reads, the per-clip cache, the error paths, and the
stdout contract. Real decoding is covered in ``test_video_pipeline.py``.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from typer.testing import CliRunner

from app.cli import analyze_cmd
from app.cli.main import app
from app.models.common import CameraMove, MediaRef, MotionLevel, ShotType, TimeRange
from app.models.media import MediaProbe, VideoStreamInfo
from app.models.video import (
    ClipAnalysis,
    DuplicateGroup,
    FootageAnalysis,
    MotionStats,
    QualityScores,
    Scene,
    SceneTags,
)
from app.services.container import Container, build_container
from app.services.paths import ProjectPaths

runner = CliRunner()

FAKE_VERSION = "fake-footage/1"


def _scene(
    clip: MediaRef,
    index: int,
    *,
    quality: float = 0.8,
    start: float = 0.0,
    end: float = 3.0,
) -> Scene:
    return Scene(
        clip=clip,
        index=index,
        range=TimeRange(start=start, end=end),
        quality=QualityScores(
            blur=quality, brightness=0.7, exposure=0.85, stability=0.9, overall=quality
        ),
        motion=MotionStats(
            level=MotionLevel.LOW, mean_magnitude=1.2, camera_move=CameraMove.PAN, shake=0.1
        ),
        tags=SceneTags(provider="classical_cv", people_count=1, confidence=0.6),
        shot_type=ShotType.MEDIUM,
        phash=f"{index:028x}",
    )


def _clip_analysis(clip: MediaRef, *, scenes: int = 2, quality: float = 0.8) -> ClipAnalysis:
    return ClipAnalysis(
        clip=clip,
        probe=MediaProbe(
            source=clip,
            duration=6.0,
            size_bytes=2048,
            format_name="mov,mp4",
            video=VideoStreamInfo(width=1920, height=1080, fps=30.0, codec="h264"),
        ),
        scenes=tuple(
            _scene(clip, index, quality=quality, start=index * 3.0, end=index * 3.0 + 3.0)
            for index in range(scenes)
        ),
        analyzer_version=FAKE_VERSION,
        analyzed_at=datetime.now(UTC),
    )


class FakeFootageAnalyzer:
    """Returns canned analyses. Decodes nothing."""

    def __init__(self, paths: ProjectPaths, *, low_quality: set[str] | None = None) -> None:
        self._paths = paths
        self._low = low_quality or set()
        self.analysed: list[str] = []

    @property
    def version(self) -> str:
        return FAKE_VERSION

    def analyze_clip(self, video: Path, *, ref: MediaRef) -> ClipAnalysis:
        self.analysed.append(ref.name)
        quality = 0.2 if ref.name in self._low else 0.8
        return _clip_analysis(ref, quality=quality)

    def analyze_project(
        self,
        clips: list[Path],
        *,
        cached: dict[MediaRef, ClipAnalysis] | None = None,
    ) -> FootageAnalysis:
        reusable = cached or {}
        analyses = []
        for video in clips:
            ref = self._paths.to_ref(video)
            existing = reusable.get(ref)
            analyses.append(existing if existing is not None else self.analyze_clip(video, ref=ref))

        footage = FootageAnalysis(clips=tuple(analyses))
        # Declare the second clip's first scene a duplicate of the first clip's, so the
        # digest's duplicate reporting is exercised.
        duplicates: tuple[DuplicateGroup, ...] = ()
        if len(analyses) >= 2 and analyses[0].scenes and analyses[1].scenes:
            duplicates = (
                DuplicateGroup(
                    representative=analyses[0].scenes[0].key,
                    duplicates=(analyses[1].scenes[0].key,),
                    similarity=0.97,
                ),
            )
        return footage.model_copy(update={"duplicates": duplicates})


@pytest.fixture
def footage_project(paths: ProjectPaths) -> ProjectPaths:
    for name in ("001.mp4", "002.mp4"):
        (paths.raw / name).write_bytes(b"\0" * 2048)
    return paths


@pytest.fixture
def fake_footage(
    monkeypatch: pytest.MonkeyPatch, footage_project: ProjectPaths
) -> FakeFootageAnalyzer:
    analyzer = FakeFootageAnalyzer(footage_project)

    def patched(project_dir: Path | None = None, *, settings: object = None) -> Container:
        real = build_container(project_dir, settings=settings)  # type: ignore[arg-type]
        return Container(
            settings=real.settings,
            paths=real.paths,
            ffmpeg=real.ffmpeg,
            exporters=real.exporters,
            footage=analyzer,  # type: ignore[arg-type]
        )

    monkeypatch.setattr(analyze_cmd, "build_container", patched)
    return analyzer


class TestAnalyzeVideo:
    def test_writes_a_document_and_prints_a_digest(
        self, footage_project: ProjectPaths, fake_footage: FakeFootageAnalyzer
    ) -> None:
        result = runner.invoke(app, ["analyze", "video", str(footage_project.root)])
        assert result.exit_code == 0, result.stderr

        assert footage_project.footage_analysis_file.is_file()
        footage = FootageAnalysis.model_validate_json(
            footage_project.footage_analysis_file.read_text(encoding="utf-8")
        )
        assert len(footage.clips) == 2
        assert result.stdout.startswith("# footage=")

    def test_the_digest_groups_scenes_under_their_clip(
        self, footage_project: ProjectPaths, fake_footage: FakeFootageAnalyzer
    ) -> None:
        """The source path is written once, not on every scene line."""
        result = runner.invoke(app, ["analyze", "video", str(footage_project.root)])
        assert "@ raw/001.mp4 1920x1080 30.00fps" in result.stdout
        assert "001#0 0.00-3.00" in result.stdout
        # The clip path must not repeat on scene lines.
        scene_lines = [line for line in result.stdout.splitlines() if line.startswith("001#")]
        assert all("raw/001.mp4" not in line for line in scene_lines)

    def test_the_digest_carries_every_selection_signal(
        self, footage_project: ProjectPaths, fake_footage: FakeFootageAnalyzer
    ) -> None:
        result = runner.invoke(app, ["analyze", "video", str(footage_project.root)])
        line = next(line for line in result.stdout.splitlines() if line.startswith("001#0"))
        for field in ("q=", "blur=", "br=", "ex=", "st=", "mot=", "cam=", "shot=", "ppl="):
            assert field in line, f"{field} missing from the digest"

    def test_duplicates_are_flagged_and_summarised(
        self, footage_project: ProjectPaths, fake_footage: FakeFootageAnalyzer
    ) -> None:
        """This is how the director learns not to reuse a shot."""
        result = runner.invoke(app, ["analyze", "video", str(footage_project.root)])
        assert " DUP" in result.stdout
        assert "# duplicates: 002#0<-001#0" in result.stdout

    def test_scenes_below_the_quality_floor_are_flagged(
        self, footage_project: ProjectPaths, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        analyzer = FakeFootageAnalyzer(footage_project, low_quality={"002.mp4"})

        def patched(project_dir: Path | None = None, *, settings: object = None) -> Container:
            real = build_container(project_dir, settings=settings)  # type: ignore[arg-type]
            return Container(
                settings=real.settings,
                paths=real.paths,
                ffmpeg=real.ffmpeg,
                exporters=real.exporters,
                footage=analyzer,  # type: ignore[arg-type]
            )

        monkeypatch.setattr(analyze_cmd, "build_container", patched)
        result = runner.invoke(app, ["analyze", "video", str(footage_project.root)])
        assert " LOW" in result.stdout
        assert "below the quality floor" in result.stderr

    def test_full_emits_the_whole_document(
        self, footage_project: ProjectPaths, fake_footage: FakeFootageAnalyzer
    ) -> None:
        result = runner.invoke(app, ["analyze", "video", str(footage_project.root), "--full"])
        payload = json.loads(result.stdout)
        assert FootageAnalysis.model_validate(payload).clips

    def test_the_digest_is_much_smaller_than_the_document(
        self, footage_project: ProjectPaths, fake_footage: FakeFootageAnalyzer
    ) -> None:
        digest = runner.invoke(app, ["analyze", "video", str(footage_project.root)])
        full = runner.invoke(
            app, ["analyze", "video", str(footage_project.root), "--full", "--force"]
        )
        assert len(digest.stdout) < len(full.stdout) / 3

    def test_logs_stay_on_stderr(
        self, footage_project: ProjectPaths, fake_footage: FakeFootageAnalyzer
    ) -> None:
        result = runner.invoke(app, ["analyze", "video", str(footage_project.root)])
        assert "clips:" in result.stderr
        assert "clips:" not in result.stdout

    def test_a_single_clip_can_be_selected(
        self, footage_project: ProjectPaths, fake_footage: FakeFootageAnalyzer
    ) -> None:
        result = runner.invoke(
            app, ["analyze", "video", str(footage_project.root), "--clip", "002.mp4"]
        )
        assert result.exit_code == 0
        assert "clips=1" in result.stdout
        assert "raw/001.mp4" not in result.stdout

    def test_a_clip_can_be_selected_by_stem(
        self, footage_project: ProjectPaths, fake_footage: FakeFootageAnalyzer
    ) -> None:
        result = runner.invoke(
            app, ["analyze", "video", str(footage_project.root), "--clip", "002"]
        )
        assert result.exit_code == 0
        assert "clips=1" in result.stdout


class TestCaching:
    def test_a_second_run_reuses_every_clip(
        self, footage_project: ProjectPaths, fake_footage: FakeFootageAnalyzer
    ) -> None:
        runner.invoke(app, ["analyze", "video", str(footage_project.root)])
        assert len(fake_footage.analysed) == 2

        result = runner.invoke(app, ["analyze", "video", str(footage_project.root)])
        assert result.exit_code == 0
        assert len(fake_footage.analysed) == 2, "clips were re-analysed"
        assert "Reusing 2 cached" in result.stderr

    def test_only_a_new_clip_is_analysed(
        self, footage_project: ProjectPaths, fake_footage: FakeFootageAnalyzer
    ) -> None:
        """Per-clip caching: adding one clip to forty must cost one analysis."""
        runner.invoke(app, ["analyze", "video", str(footage_project.root)])
        (footage_project.raw / "003.mp4").write_bytes(b"\0" * 2048)

        runner.invoke(app, ["analyze", "video", str(footage_project.root)])
        assert fake_footage.analysed == ["001.mp4", "002.mp4", "003.mp4"]

    def test_force_re_analyses_everything(
        self, footage_project: ProjectPaths, fake_footage: FakeFootageAnalyzer
    ) -> None:
        runner.invoke(app, ["analyze", "video", str(footage_project.root)])
        runner.invoke(app, ["analyze", "video", str(footage_project.root), "--force"])
        assert len(fake_footage.analysed) == 4

    def test_a_touched_clip_is_re_analysed(
        self, footage_project: ProjectPaths, fake_footage: FakeFootageAnalyzer
    ) -> None:
        """Reusing an analysis of footage that has changed would misinform the director."""
        import os
        import time

        runner.invoke(app, ["analyze", "video", str(footage_project.root)])
        future = time.time() + 60
        os.utime(footage_project.raw / "001.mp4", (future, future))

        runner.invoke(app, ["analyze", "video", str(footage_project.root)])
        assert fake_footage.analysed.count("001.mp4") == 2
        assert fake_footage.analysed.count("002.mp4") == 1

    def test_a_version_bump_invalidates_the_cache(
        self, footage_project: ProjectPaths, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        analyzer = FakeFootageAnalyzer(footage_project)

        def patched(project_dir: Path | None = None, *, settings: object = None) -> Container:
            real = build_container(project_dir, settings=settings)  # type: ignore[arg-type]
            return Container(
                settings=real.settings,
                paths=real.paths,
                ffmpeg=real.ffmpeg,
                exporters=real.exporters,
                footage=analyzer,  # type: ignore[arg-type]
            )

        monkeypatch.setattr(analyze_cmd, "build_container", patched)
        runner.invoke(app, ["analyze", "video", str(footage_project.root)])
        assert len(analyzer.analysed) == 2

        monkeypatch.setattr(FakeFootageAnalyzer, "version", property(lambda self: "other/9"))
        runner.invoke(app, ["analyze", "video", str(footage_project.root)])
        assert len(analyzer.analysed) == 4

    def test_a_corrupt_cache_is_ignored(
        self, footage_project: ProjectPaths, fake_footage: FakeFootageAnalyzer
    ) -> None:
        footage_project.footage_analysis_file.parent.mkdir(parents=True, exist_ok=True)
        footage_project.footage_analysis_file.write_text("{not json", encoding="utf-8")
        result = runner.invoke(app, ["analyze", "video", str(footage_project.root)])
        assert result.exit_code == 0
        assert len(fake_footage.analysed) == 2


class TestAnalyzeVideoErrors:
    def test_an_uninitialised_project(self, tmp_path: Path) -> None:
        blank = tmp_path / "blank"
        blank.mkdir()
        result = runner.invoke(app, ["analyze", "video", str(blank)])
        assert result.exit_code == 3
        assert json.loads(result.stdout)["error"]["code"] == "project.not_initialised"

    def test_no_footage_names_the_accepted_formats(self, paths: ProjectPaths) -> None:
        result = runner.invoke(app, ["analyze", "video", str(paths.root)])
        assert result.exit_code == 3
        error = json.loads(result.stdout)["error"]
        assert error["code"] == "footage.missing"
        assert "mp4" in error["hint"]

    def test_an_unknown_clip_lists_the_alternatives(
        self, footage_project: ProjectPaths, fake_footage: FakeFootageAnalyzer
    ) -> None:
        result = runner.invoke(
            app, ["analyze", "video", str(footage_project.root), "--clip", "nope.mp4"]
        )
        assert result.exit_code == 3
        error = json.loads(result.stdout)["error"]
        assert error["code"] == "clip.not_found"
        assert "001.mp4" in error["hint"]

    def test_a_missing_video_dependency_is_an_environment_error(
        self, footage_project: ProjectPaths, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Exit 5 tells the director to stop rather than retry."""
        from app.analysis.vision.probe import VideoDependencyMissingError

        class Broken(FakeFootageAnalyzer):
            def analyze_project(
                self, clips: list[Path], *, cached: object = None
            ) -> FootageAnalysis:
                raise VideoDependencyMissingError("opencv-python")

        analyzer = Broken(footage_project)

        def patched(project_dir: Path | None = None, *, settings: object = None) -> Container:
            real = build_container(project_dir, settings=settings)  # type: ignore[arg-type]
            return Container(
                settings=real.settings,
                paths=real.paths,
                ffmpeg=real.ffmpeg,
                exporters=real.exporters,
                footage=analyzer,  # type: ignore[arg-type]
            )

        monkeypatch.setattr(analyze_cmd, "build_container", patched)
        result = runner.invoke(app, ["analyze", "video", str(footage_project.root)])
        assert result.exit_code == 5
        assert json.loads(result.stdout)["error"]["code"] == "dependency.missing"

    def test_unreadable_footage_is_an_input_error(
        self, footage_project: ProjectPaths, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from app.analysis.vision.probe import ProbeError

        class Broken(FakeFootageAnalyzer):
            def analyze_project(
                self, clips: list[Path], *, cached: object = None
            ) -> FootageAnalysis:
                raise ProbeError("could not read 001.mp4: InvalidDataError")

        analyzer = Broken(footage_project)

        def patched(project_dir: Path | None = None, *, settings: object = None) -> Container:
            real = build_container(project_dir, settings=settings)  # type: ignore[arg-type]
            return Container(
                settings=real.settings,
                paths=real.paths,
                ffmpeg=real.ffmpeg,
                exporters=real.exporters,
                footage=analyzer,  # type: ignore[arg-type]
            )

        monkeypatch.setattr(analyze_cmd, "build_container", patched)
        result = runner.invoke(app, ["analyze", "video", str(footage_project.root)])
        assert result.exit_code == 4
        assert json.loads(result.stdout)["error"]["code"] == "footage.unreadable"


class TestDiscoverability:
    def test_video_is_advertised_alongside_audio(self) -> None:
        result = runner.invoke(app, ["analyze", "--help"])
        assert "audio" in result.output
        assert "video" in result.output

    def test_the_video_analyser_is_advertised(self) -> None:
        """Was "music is still not advertised" until Phase 7 implemented it.

        The no-empty-commands rule it guarded now lives in the exact-registry test in
        ``test_cli_phase4``, which fails on an *extra* command as well as a missing one.
        """
        result = runner.invoke(app, ["analyze", "--help"])
        assert "video" in result.output
