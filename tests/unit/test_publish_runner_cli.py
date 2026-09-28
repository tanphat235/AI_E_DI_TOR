"""scripts/publish: the ledger, the job lock, and the commands built for the server."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from scripts.publish import cli, remote
from scripts.publish.check import VideoInfo
from scripts.publish.errors import PublishError
from scripts.publish.platforms.base import PostResult, Publisher
from scripts.publish.runner import LOCK_FILE, post_job, summary
from scripts.publish.schema import Platform, PublishMeta, load_ledger, load_meta, write_meta

FIXED = datetime(2026, 10, 1, 12, tzinfo=UTC)


def _job(tmp_path: Path, *, at: datetime | None = None) -> Path:
    job = tmp_path / "clip_001"
    job.mkdir()
    (job / "clip_001.mp4").write_bytes(b"\0" * 32)
    write_meta(
        job,
        {
            "video": "clip_001.mp4",
            "publish_at": (at or FIXED).isoformat(),
            "youtube": {"title": "T", "hashtags": ["#Shorts"]},
            "facebook": {"description": "F"},
            "tiktok": {"caption": "C"},
        },
    )
    return job


class FakePublisher:
    def __init__(self, name: Platform, fail: bool = False) -> None:
        self._name: Platform = name
        self.fail = fail
        self.calls = 0

    @property
    def name(self) -> Platform:
        return self._name

    def post(self, video: Path, meta: PublishMeta) -> PostResult:
        self.calls += 1
        if self.fail:
            raise PublishError(f"{self._name}_upload_failed", "boom")
        return PostResult(f"{self._name}-id", f"https://{self._name}/x", "")


def _factories(fakes: dict[Platform, FakePublisher]) -> dict[Platform, object]:
    return {name: (lambda f=fake: f) for name, fake in fakes.items()}


class TestRunner:
    def test_a_rerun_retries_only_what_failed(self, tmp_path: Path) -> None:
        job = _job(tmp_path)
        fakes: dict[Platform, FakePublisher] = {
            "facebook": FakePublisher("facebook"),
            "youtube": FakePublisher("youtube", fail=True),
            "tiktok": FakePublisher("tiktok"),
        }
        _, ledger = post_job(job, _factories(fakes), now=lambda: FIXED)  # type: ignore[arg-type]
        assert ledger.posted("facebook") and ledger.posted("tiktok")
        assert ledger.entries["youtube"].status == "failed"
        assert "boom" in ledger.entries["youtube"].error

        fakes["youtube"].fail = False
        _, ledger = post_job(job, _factories(fakes), now=lambda: FIXED)  # type: ignore[arg-type]
        assert all(ledger.posted(p) for p in ("facebook", "youtube", "tiktok"))
        assert [fakes[p].calls for p in ("facebook", "youtube", "tiktok")] == [1, 2, 1]
        assert load_ledger(job) == ledger

    def test_a_platform_already_posted_never_builds_its_publisher(self, tmp_path: Path) -> None:
        job = _job(tmp_path)
        fakes: dict[Platform, FakePublisher] = {
            p: FakePublisher(p) for p in ("facebook", "youtube", "tiktok")
        }
        post_job(job, _factories(fakes), now=lambda: FIXED)  # type: ignore[arg-type]

        def explode() -> Publisher:
            raise AssertionError("should not be built")

        post_job(job, dict.fromkeys(fakes, explode), now=lambda: FIXED)

    def test_only_limits_the_platforms(self, tmp_path: Path) -> None:
        job = _job(tmp_path)
        fakes: dict[Platform, FakePublisher] = {
            p: FakePublisher(p) for p in ("facebook", "youtube", "tiktok")
        }
        _, ledger = post_job(
            job,
            _factories(fakes),
            now=lambda: FIXED,
            only=("facebook",),  # type: ignore[arg-type]
        )
        assert set(ledger.entries) == {"facebook"}

    def test_a_concurrent_post_is_refused_and_the_lock_is_released(self, tmp_path: Path) -> None:
        job = _job(tmp_path)
        (job / LOCK_FILE).write_text("123")
        with pytest.raises(PublishError, match="job_locked"):
            post_job(job, {}, now=lambda: FIXED)
        (job / LOCK_FILE).unlink()
        post_job(
            job,
            _factories({p: FakePublisher(p) for p in ("facebook", "youtube", "tiktok")}),  # type: ignore[arg-type]
            now=lambda: FIXED,
        )
        assert not (job / LOCK_FILE).exists()

    def test_summary_names_every_platform(self, tmp_path: Path) -> None:
        job = _job(tmp_path)
        _, ledger = post_job(
            job,
            _factories(
                {
                    "facebook": FakePublisher("facebook"),  # type: ignore[arg-type]
                    "youtube": FakePublisher("youtube", fail=True),
                    "tiktok": FakePublisher("tiktok"),
                }
            ),
            now=lambda: FIXED,
        )
        text = summary(load_meta(job), ledger, job.name)
        assert "facebook: posted https://facebook/x" in text
        assert "youtube: FAILED" in text


class TestRemote:
    def test_schedule_hands_the_time_and_the_post_command_to_openclaw(self, tmp_path: Path) -> None:
        job = _job(tmp_path)
        target = remote.Remote(host="me@srv", root="/srv/aive-publish", python="/v/bin/python")
        commands = remote.schedule(
            job, load_meta(job), target, timeout_seconds=1800, extra=["--announce"]
        )
        assert commands[0][:2] == ["ssh", "me@srv"]
        assert commands[1][:2] == ["scp", "-r"]
        assert commands[1][-1] == "me@srv:/srv/aive-publish/jobs/"
        create = commands[-1][2]
        assert "openclaw automations create --at 2026-10-01T12:00:00+00:00" in create
        assert "--timeout-seconds 1800 --announce" in create
        assert "--command-cwd /srv/aive-publish" in create
        argv_json = create.split("--command-argv ", 1)[1].split(" --command-cwd")[0]
        assert json.loads(argv_json.strip("'")) == [
            "/v/bin/python",
            "-m",
            "scripts.publish.cli",
            "post",
            "jobs/clip_001",
        ]

    def test_schedule_never_deletes_the_remote_job(self, tmp_path: Path) -> None:
        job = _job(tmp_path)
        commands = remote.schedule(
            job, load_meta(job), remote.Remote("h", "/r"), timeout_seconds=60, extra=[]
        )
        assert not any("rm" in " ".join(c) for c in commands)


class TestCli:
    def test_draft_then_check_rejects_the_placeholders(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        video = tmp_path / "clip_009.mp4"
        video.write_bytes(b"\0" * 16)
        assert cli.main(["draft", str(video), "--at", "2099-01-01T19:00:00+07:00"]) == 0
        job = tmp_path / "publish" / "clip_009"
        assert (job / "clip_009.mp4").is_file()
        capsys.readouterr()
        assert cli.main(["check", str(job)]) == cli.EXIT_CHECK
        out = capsys.readouterr().out
        assert "TODO" in out
        assert "cannot be decoded" in out or "PyAV not installed" in out

    def test_draft_refuses_to_overwrite(self, tmp_path: Path) -> None:
        video = tmp_path / "c.mp4"
        video.write_bytes(b"\0")
        assert cli.main(["draft", str(video)]) == 0
        assert cli.main(["draft", str(video)]) == cli.EXIT_CONFIG

    def test_schedule_dry_run_prints_and_runs_nothing(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            cli,
            "probe_video",
            lambda _p: VideoInfo(duration=30.0, width=1080, height=1920, fps=30.0),
        )
        job = _job(tmp_path, at=datetime.now(UTC) + timedelta(days=1))
        args = cli.build_parser().parse_args(
            ["schedule", str(job), "--host", "me@srv", "--dry-run"]
        )
        ran: list[list[str]] = []
        assert cli.cmd_schedule(args, run=lambda argv: ran.append(argv) or 0) == 0
        assert ran == []
        out = capsys.readouterr().out
        assert "openclaw automations create" in out
        assert "scp -r -q" in out

    def test_schedule_refuses_a_time_in_the_past(self, tmp_path: Path) -> None:
        job = _job(tmp_path, at=datetime.now(UTC) - timedelta(minutes=5))
        args = cli.build_parser().parse_args(["schedule", str(job), "--host", "h", "--dry-run"])
        assert cli.cmd_schedule(args) == cli.EXIT_CHECK

    def test_post_without_apps_toml_fails_as_configuration(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("AIVE_PUBLISH_HOME", str(tmp_path / "empty-home"))
        monkeypatch.setattr(
            cli,
            "probe_video",
            lambda _p: VideoInfo(duration=30.0, width=1080, height=1920, fps=30.0),
        )
        job = _job(tmp_path)
        assert cli.main(["post", str(job)]) == cli.EXIT_PARTIAL
        ledger = load_ledger(job)
        assert "missing_apps" in ledger.entries["youtube"].error
        assert "not_authorized" in ledger.entries["facebook"].error
