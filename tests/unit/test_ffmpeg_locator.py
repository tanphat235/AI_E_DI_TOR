"""Tests for FFmpeg discovery.

Every tier is exercised with the environment monkeypatched, so results do not
depend on what happens to be installed on the machine running the suite. That is
the whole reason :class:`FFmpegLocator` is a class taking settings rather than a
module-level function reading globals.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.config.settings import MediaSettings
from app.services.ffmpeg_locator import (
    BinarySource,
    FFmpegLocator,
    FFmpegNotFoundError,
    ResolvedBinary,
)

LOCATOR_MODULE = "app.services.ffmpeg_locator"


@pytest.fixture
def fake_binaries(tmp_path: Path) -> tuple[Path, Path]:
    """A directory containing both binaries, as a real install would."""
    ffmpeg = tmp_path / "bin" / "ffmpeg.exe"
    ffprobe = tmp_path / "bin" / "ffprobe.exe"
    ffmpeg.parent.mkdir(parents=True)
    ffmpeg.touch()
    ffprobe.touch()
    return ffmpeg, ffprobe


@pytest.fixture
def no_path_lookup(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pretend nothing is on PATH."""
    monkeypatch.setattr(f"{LOCATOR_MODULE}.shutil.which", lambda _name: None)


@pytest.fixture
def no_vendored(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pretend imageio-ffmpeg is not installed."""
    monkeypatch.setattr(FFmpegLocator, "_vendored_ffmpeg", staticmethod(lambda: None))


class TestConfiguredPath:
    def test_uses_an_explicit_path(self, fake_binaries: tuple[Path, Path]) -> None:
        ffmpeg, ffprobe = fake_binaries
        locator = FFmpegLocator(MediaSettings(ffmpeg_path=ffmpeg, ffprobe_path=ffprobe))
        tools = locator.locate()
        assert tools.ffmpeg == ResolvedBinary(path=ffmpeg, source=BinarySource.CONFIGURED)
        assert tools.ffprobe == ResolvedBinary(path=ffprobe, source=BinarySource.CONFIGURED)

    def test_a_configured_path_that_does_not_exist_is_an_error(self, tmp_path: Path) -> None:
        """Silently falling back would hide a typo in the user's config."""
        locator = FFmpegLocator(MediaSettings(ffmpeg_path=tmp_path / "nope.exe"))
        with pytest.raises(FFmpegNotFoundError, match="configured ffmpeg_path does not exist"):
            locator.locate()

    def test_an_empty_string_means_auto_detect(self) -> None:
        """TOML has no null, so `ffmpeg_path = ""` must not become Path(".")."""
        settings = MediaSettings.model_validate({"ffmpeg_path": "", "ffprobe_path": "   "})
        assert settings.ffmpeg_path is None
        assert settings.ffprobe_path is None


class TestPathLookup:
    def test_prefers_a_system_build_over_the_vendored_one(
        self, monkeypatch: pytest.MonkeyPatch, fake_binaries: tuple[Path, Path]
    ) -> None:
        ffmpeg, ffprobe = fake_binaries
        found = {"ffmpeg": str(ffmpeg), "ffprobe": str(ffprobe)}
        monkeypatch.setattr(f"{LOCATOR_MODULE}.shutil.which", lambda name: found.get(name))

        tools = FFmpegLocator(MediaSettings()).locate()
        assert tools.ffmpeg.source is BinarySource.SYSTEM_PATH
        assert tools.ffprobe is not None
        assert tools.ffprobe.source is BinarySource.SYSTEM_PATH


class TestVendoredFallback:
    def test_falls_back_to_the_wheel(
        self, monkeypatch: pytest.MonkeyPatch, no_path_lookup: None, tmp_path: Path
    ) -> None:
        vendored = tmp_path / "ffmpeg-win-x86_64-v7.1.exe"
        vendored.touch()
        monkeypatch.setattr(FFmpegLocator, "_vendored_ffmpeg", staticmethod(lambda: vendored))

        tools = FFmpegLocator(MediaSettings()).locate()
        assert tools.ffmpeg == ResolvedBinary(path=vendored, source=BinarySource.VENDORED)

    def test_vendored_ffmpeg_yields_no_ffprobe(
        self, monkeypatch: pytest.MonkeyPatch, no_path_lookup: None, tmp_path: Path
    ) -> None:
        """The wheel ships ffmpeg alone; looking beside it would find nothing."""
        vendored = tmp_path / "ffmpeg-win-x86_64-v7.1.exe"
        vendored.touch()
        monkeypatch.setattr(FFmpegLocator, "_vendored_ffmpeg", staticmethod(lambda: vendored))

        tools = FFmpegLocator(MediaSettings()).locate()
        assert tools.ffprobe is None
        assert not tools.has_ffprobe


class TestSiblingLookup:
    def test_finds_ffprobe_next_to_a_configured_ffmpeg(
        self,
        monkeypatch: pytest.MonkeyPatch,
        fake_binaries: tuple[Path, Path],
        no_path_lookup: None,
    ) -> None:
        ffmpeg, ffprobe = fake_binaries
        tools = FFmpegLocator(MediaSettings(ffmpeg_path=ffmpeg)).locate()
        assert tools.ffprobe == ResolvedBinary(path=ffprobe, source=BinarySource.SIBLING)

    def test_no_sibling_means_no_ffprobe(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, no_path_lookup: None
    ) -> None:
        lonely = tmp_path / "ffmpeg.exe"
        lonely.touch()
        tools = FFmpegLocator(MediaSettings(ffmpeg_path=lonely)).locate()
        assert tools.ffprobe is None


class TestTotalFailure:
    def test_raises_with_actionable_instructions(
        self, no_path_lookup: None, no_vendored: None
    ) -> None:
        """The message must tell the user how to fix it, not just that it broke."""
        with pytest.raises(FFmpegNotFoundError) as excinfo:
            FFmpegLocator(MediaSettings()).locate()
        message = str(excinfo.value)
        assert "pip install imageio-ffmpeg" in message
        assert "winget install" in message


class TestVersionReporting:
    def test_version_of_a_non_executable_never_raises(self, tmp_path: Path) -> None:
        """`doctor` exists to report a broken environment, not to crash inside it."""
        bogus = tmp_path / "not-really-ffmpeg.exe"
        bogus.write_text("this is not a binary", encoding="utf-8")
        result = ResolvedBinary(path=bogus, source=BinarySource.CONFIGURED).version()
        assert isinstance(result, str)
        assert result

    def test_version_of_a_missing_file_never_raises(self, tmp_path: Path) -> None:
        result = ResolvedBinary(
            path=tmp_path / "gone.exe", source=BinarySource.CONFIGURED
        ).version()
        assert "unavailable" in result


class TestRealEnvironment:
    def test_the_vendored_wheel_is_actually_present(self) -> None:
        """Sanity check on the real environment: imageio-ffmpeg must be installed.

        This is the one test here that touches the machine, and deliberately so -
        it is what proves the "works on a clean install" promise rather than
        assuming it.
        """
        vendored = FFmpegLocator._vendored_ffmpeg()
        assert vendored is not None, "imageio-ffmpeg is a hard dependency and must ship a binary"
        assert vendored.is_file()
