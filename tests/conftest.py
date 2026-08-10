"""Shared test fixtures.

Two principles run through these:

* **No real media, no external binaries.** Every fixture builds objects in memory
  or writes tiny placeholder files. A test suite that needs a video file is a test
  suite nobody runs.
* **No dependence on the developer's machine.** Config is constructed explicitly
  and environment variables are scrubbed, so a stray ``AIVE_*`` in someone's shell
  cannot change a result.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import pytest

from app.config.settings import AiveSettings, MediaSettings
from app.models.common import MediaRef, TimeRange
from app.models.edit_plan import EditPlan, NarrationTrack, TimelineClip
from app.models.media import AudioStreamInfo, MediaProbe, VideoStreamInfo
from app.services.paths import ProjectPaths


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove any ``AIVE_*`` variables for the duration of a test.

    Autouse because a single leaked override in a developer's shell would
    otherwise produce failures that reproduce for nobody else.
    """
    for key in list(os.environ):
        if key.startswith("AIVE_"):
            monkeypatch.delenv(key, raising=False)


@pytest.fixture
def project_root(tmp_path: Path) -> Path:
    """An initialised, empty project folder."""
    ProjectPaths.for_root(tmp_path).ensure()
    return tmp_path


@pytest.fixture
def paths(project_root: Path) -> ProjectPaths:
    return ProjectPaths.for_root(project_root)


@pytest.fixture
def populated_project(paths: ProjectPaths) -> ProjectPaths:
    """A project with placeholder media.

    The bytes are meaningless. Nothing in Phase 1 decodes media - ``scan`` only
    stats files - so a real encode would add minutes of fixture setup for no
    additional coverage.
    """
    (paths.root / "narration.wav").write_bytes(b"RIFF" + b"\0" * 64)
    (paths.raw / "001.mp4").write_bytes(b"\0" * 2048)
    (paths.raw / "002.mp4").write_bytes(b"\0" * 4096)
    (paths.music / "calm_forest.mp3").write_bytes(b"\0" * 1024)
    return paths


@pytest.fixture
def settings() -> AiveSettings:
    """Bare settings with no file layer, for tests that must not read the disk."""
    return AiveSettings()


@pytest.fixture
def empty_media_settings() -> MediaSettings:
    """Media settings with nothing configured, so the locator auto-resolves."""
    return MediaSettings()


# --------------------------------------------------------------------------- #
# Model fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture
def video_probe() -> MediaProbe:
    """A plausible 30-second 1080p clip."""
    return MediaProbe(
        source=MediaRef(path="raw/001.mp4"),
        duration=30.0,
        size_bytes=12_345_678,
        format_name="mov,mp4,m4a",
        video=VideoStreamInfo(width=1920, height=1080, fps=30.0, codec="h264"),
        audio=AudioStreamInfo(codec="aac", sample_rate=48000, channels=2),
    )


@pytest.fixture
def minimal_plan() -> EditPlan:
    """The smallest valid Edit Plan: one clip, no narration, no music."""
    return EditPlan(
        project_id="test",
        created_by="pytest",
        clips=(
            TimelineClip(
                id="c001",
                source=MediaRef(path="raw/001.mp4"),
                source_range=TimeRange(start=0.0, end=4.0),
                reason="the only clip",
            ),
        ),
    )


@pytest.fixture
def narration_track() -> NarrationTrack:
    """Narration with a removed gap between 4.0 and 6.0 seconds.

    The gap is the point: it is what makes source time and timeline time diverge,
    which is the behaviour most worth testing.
    """
    return NarrationTrack(
        source=MediaRef(path="narration.wav"),
        kept_ranges=(TimeRange(start=0.0, end=4.0), TimeRange(start=6.0, end=10.0)),
    )


@pytest.fixture
def now() -> datetime:
    return datetime(2026, 8, 6, 12, 0, 0, tzinfo=UTC)


@pytest.fixture(scope="session")
def generated_videos(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """Real H.264 clips with known properties, built once per session.

    Session-scoped because encoding six clips takes a few seconds and every integration
    test wants the same ones. See ``tests/conftest_video.py`` for what each demonstrates.
    """
    from tests.conftest_video import generate_clips

    return generate_clips(tmp_path_factory.mktemp("generated_video"))


@pytest.fixture
def video_project(paths: ProjectPaths, generated_videos: dict[str, Path]) -> ProjectPaths:
    """An initialised project with the generated clips copied into ``raw/``."""
    import shutil

    for name, source in generated_videos.items():
        shutil.copy(source, paths.raw / name)
    return paths


@pytest.fixture
def isolated_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """Run with the working directory set to a temp folder.

    Needed because the site-override config layer is resolved relative to the cwd,
    so a test asserting on config layers must not see the repo's own ``config/``.
    """
    monkeypatch.chdir(tmp_path)
    yield tmp_path
