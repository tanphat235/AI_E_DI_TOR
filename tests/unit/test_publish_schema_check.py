"""scripts/publish: publish.json validation and the pre-upload checks."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from pydantic import ValidationError
from scripts.publish.check import VideoInfo, check_meta
from scripts.publish.config import load_limits
from scripts.publish.schema import PublishMeta, compose

NOW = datetime(2026, 10, 1, 12, 0, tzinfo=UTC)


def _meta(**overrides: object) -> PublishMeta:
    payload: dict[str, object] = {
        "video": "clip.mp4",
        "publish_at": (NOW + timedelta(hours=2)).isoformat(),
        "youtube": {"title": "Title", "description": "desc", "hashtags": ["#Shorts"]},
        "facebook": {"description": "fb text", "hashtags": ["#phatphap"]},
        "tiktok": {"caption": "tt text", "hashtags": ["#fyp"]},
    }
    payload.update(overrides)
    return PublishMeta.model_validate(payload)


def _video(tmp_path: Path) -> Path:
    path = tmp_path / "clip.mp4"
    path.write_bytes(b"\0" * 64)
    return path


GOOD_INFO = VideoInfo(duration=45.0, width=1080, height=1920, fps=30.0)


class TestSchema:
    def test_a_named_platform_needs_its_section(self) -> None:
        with pytest.raises(ValidationError, match="no section"):
            _meta(tiktok=None)

    def test_a_platform_left_out_needs_no_section(self) -> None:
        meta = _meta(platforms=["facebook"], youtube=None, tiktok=None)
        assert meta.platforms == ("facebook",)

    def test_a_misspelt_key_fails_loudly(self) -> None:
        with pytest.raises(ValidationError):
            _meta(facebook={"descripton": "typo"})

    def test_hashtags_must_be_one_word_with_a_hash(self) -> None:
        with pytest.raises(ValidationError):
            _meta(tiktok={"caption": "x", "hashtags": ["#two words"]})
        with pytest.raises(ValidationError):
            _meta(tiktok={"caption": "x", "hashtags": ["nohash"]})

    def test_publish_at_must_carry_an_offset(self) -> None:
        with pytest.raises(ValidationError):
            _meta(publish_at="2026-10-01T19:00:00")

    def test_compose_puts_hashtags_on_their_own_line(self) -> None:
        assert compose("Body  \n", ("#a", "#b")) == "Body\n\n#a #b"
        assert compose("Body", ()) == "Body"
        assert compose("", ("#a",)) == "#a"


class TestCheck:
    def _run(self, meta: PublishMeta, tmp_path: Path, **kw: object) -> list[str]:
        issues = check_meta(
            meta,
            load_limits(),
            video_path=kw.get("video", _video(tmp_path)),  # type: ignore[arg-type]
            now=NOW,
            info=kw.get("info", GOOD_INFO),  # type: ignore[arg-type]
            require_future=bool(kw.get("require_future", True)),
        )
        return [f"{i.level}:{i.platform}:{i.message}" for i in issues]

    def test_a_clean_job_has_no_errors(self, tmp_path: Path) -> None:
        assert not [i for i in self._run(_meta(), tmp_path) if i.startswith("error")]

    def test_placeholders_block_posting(self, tmp_path: Path) -> None:
        issues = self._run(_meta(tiktok={"caption": "TODO"}), tmp_path)
        assert any(i.startswith("error:tiktok") and "TODO" in i for i in issues)

    def test_youtube_title_over_the_limit(self, tmp_path: Path) -> None:
        issues = self._run(_meta(youtube={"title": "x" * 101}), tmp_path)
        assert any(i.startswith("error:youtube") and "title" in i for i in issues)

    def test_angle_brackets_are_rejected_for_youtube(self, tmp_path: Path) -> None:
        issues = self._run(_meta(youtube={"title": "a <b>"}), tmp_path)
        assert any("< or >" in i for i in issues)

    def test_a_past_time_fails_only_when_scheduling(self, tmp_path: Path) -> None:
        past = _meta(publish_at=(NOW - timedelta(minutes=1)).isoformat())
        assert any("schedule" in i for i in self._run(past, tmp_path))
        assert not any("schedule" in i for i in self._run(past, tmp_path, require_future=False))

    def test_tiktok_frame_rate_bounds(self, tmp_path: Path) -> None:
        slow = VideoInfo(duration=30.0, width=1080, height=1920, fps=15.0)
        assert any(
            i.startswith("error:tiktok") and "fps" in i
            for i in self._run(_meta(), tmp_path, info=slow)
        )

    def test_a_long_video_warns_on_facebook_and_youtube(self, tmp_path: Path) -> None:
        long = VideoInfo(duration=200.0, width=1080, height=1920, fps=30.0)
        issues = self._run(_meta(), tmp_path, info=long)
        assert any(i.startswith("warn:facebook") for i in issues)
        assert any(i.startswith("warn:youtube") and "regular video" in i for i in issues)

    def test_missing_video_is_an_error(self, tmp_path: Path) -> None:
        issues = self._run(_meta(), tmp_path, video=tmp_path / "gone.mp4")
        assert any(i.startswith("error:video") for i in issues)

    def test_without_pyav_the_media_checks_are_skipped_not_passed(self, tmp_path: Path) -> None:
        issues = self._run(_meta(), tmp_path, info=None)
        assert any(i.startswith("warn:video") and "PyAV" in i for i in issues)
