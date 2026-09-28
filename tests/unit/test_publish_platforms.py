"""scripts/publish uploaders against a faked network. Nothing here leaves the process."""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import pytest
from scripts.publish.config import TokenStore, load_limits
from scripts.publish.errors import PublishError
from scripts.publish.platforms.facebook import FacebookPublisher
from scripts.publish.platforms.tiktok import TikTokPublisher, plan_chunks
from scripts.publish.platforms.youtube import YouTubePublisher
from scripts.publish.schema import PublishMeta

Handler = Callable[[httpx.Request], httpx.Response]
VIDEO_BYTES = bytes(range(256)) * 4 + b"tail!"  # 1029 bytes


@pytest.fixture
def video(tmp_path: Path) -> Path:
    path = tmp_path / "clip.mp4"
    path.write_bytes(VIDEO_BYTES)
    return path


@pytest.fixture
def tokens(tmp_path: Path) -> TokenStore:
    return TokenStore(tmp_path / "home")


@pytest.fixture
def meta() -> PublishMeta:
    return PublishMeta.model_validate(
        {
            "video": "clip.mp4",
            "publish_at": datetime(2026, 10, 1, 19, tzinfo=UTC).isoformat(),
            "youtube": {"title": "T", "description": "D", "tags": ["a"], "hashtags": ["#Shorts"]},
            "facebook": {"description": "FB", "hashtags": ["#x"]},
            "tiktok": {"caption": "Cap", "hashtags": ["#y"]},
        }
    )


def _limits(**tiktok: int) -> dict[str, Any]:
    limits = load_limits()
    limits["tiktok"].update(tiktok)
    limits["upload"]["read_block_bytes"] = 100
    return limits


class Recorder:
    def __init__(self, handler: Handler) -> None:
        self.handler = handler
        self.requests: list[tuple[httpx.Request, bytes]] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = request.read()
        self.requests.append((request, body))
        return self.handler(request)

    def client(self) -> httpx.Client:
        return httpx.Client(transport=httpx.MockTransport(self))


def _no_sleep(_seconds: float) -> None:
    return


class TestYouTube:
    def test_resumes_after_a_503_from_the_byte_google_reports(
        self, video: Path, tokens: TokenStore, meta: PublishMeta
    ) -> None:
        tokens.save("youtube", {"access_token": "at", "refresh_token": "rt", "expires_at": 1e12})
        puts = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            if request.method == "POST":
                assert request.headers["X-Upload-Content-Length"] == str(len(VIDEO_BYTES))
                body = json.loads(request.content)
                assert body["snippet"]["description"] == "D\n\n#Shorts"
                return httpx.Response(200, headers={"Location": "https://up.example/s1"})
            if request.headers.get("Content-Range", "").startswith("bytes */"):
                return httpx.Response(308, headers={"Range": "bytes=0-99"})
            puts["n"] += 1
            if puts["n"] == 1:
                return httpx.Response(503)
            return httpx.Response(201, json={"id": "vid123"})

        rec = Recorder(handler)
        result = YouTubePublisher(
            rec.client(),
            tokens,
            {"client_id": "c", "client_secret": "s"},
            _limits(),
            sleep=_no_sleep,
            clock=lambda: 0.0,
        ).post(video, meta)

        assert result.remote_id == "vid123"
        assert result.url == "https://youtube.com/shorts/vid123"
        assert "private" in result.note
        resumed_request, resumed_body = rec.requests[-1]
        assert resumed_request.headers["Content-Range"] == f"bytes 100-1028/{len(VIDEO_BYTES)}"
        assert resumed_body == VIDEO_BYTES[100:]

    def test_refreshes_an_expired_access_token(
        self, video: Path, tokens: TokenStore, meta: PublishMeta
    ) -> None:
        tokens.save("youtube", {"access_token": "old", "refresh_token": "rt", "expires_at": 0})

        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.host == "oauth2.googleapis.com":
                return httpx.Response(200, json={"access_token": "new", "expires_in": 3600})
            if request.method == "POST":
                assert request.headers["Authorization"] == "Bearer new"
                return httpx.Response(200, headers={"Location": "https://up.example/s1"})
            return httpx.Response(200, json={"id": "v"})

        YouTubePublisher(
            Recorder(handler).client(),
            tokens,
            {"client_id": "c", "client_secret": "s"},
            _limits(),
            sleep=_no_sleep,
            clock=lambda: 1000.0,
        ).post(video, meta)
        assert tokens.load("youtube")["access_token"] == "new"

    def test_a_4xx_is_not_retried(self, video: Path, tokens: TokenStore, meta: PublishMeta) -> None:
        tokens.save("youtube", {"access_token": "at", "refresh_token": "rt", "expires_at": 1e12})

        def handler(request: httpx.Request) -> httpx.Response:
            if request.method == "POST":
                return httpx.Response(200, headers={"Location": "https://up.example/s1"})
            return httpx.Response(400, json={"error": "bad"})

        with pytest.raises(PublishError, match="youtube_upload_failed"):
            YouTubePublisher(
                Recorder(handler).client(),
                tokens,
                {"client_id": "c", "client_secret": "s"},
                _limits(),
                sleep=_no_sleep,
            ).post(video, meta)


class TestFacebook:
    def _handler(self, status: dict[str, Any]) -> Handler:
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.host == "rupload.facebook.com":
                assert request.headers["Authorization"] == "OAuth pt"
                assert request.headers["file_size"] == str(len(VIDEO_BYTES))
                return httpx.Response(200, json={"success": True})
            if request.method == "GET":
                return httpx.Response(200, json={"status": status})
            form = dict(httpx.QueryParams(request.content.decode()))
            if form["upload_phase"] == "start":
                return httpx.Response(200, json={"video_id": "r1"})
            assert form["description"] == "FB\n\n#x"
            assert form["video_state"] == "PUBLISHED"
            return httpx.Response(200, json={"success": True})

        return handler

    def test_start_upload_finish(self, video: Path, tokens: TokenStore, meta: PublishMeta) -> None:
        tokens.save("facebook", {"page_id": "p1", "page_token": "pt", "page_name": "Page"})
        rec = Recorder(self._handler({"publishing_phase": {"status": "complete"}}))
        result = FacebookPublisher(rec.client(), tokens, _limits(), sleep=_no_sleep).post(
            video, meta
        )
        assert result.remote_id == "r1"
        assert result.url == "https://www.facebook.com/reel/r1"
        assert result.note == "published"
        uploaded = [body for req, body in rec.requests if req.url.host == "rupload.facebook.com"]
        assert uploaded == [VIDEO_BYTES]

    def test_a_processing_error_is_reported(
        self, video: Path, tokens: TokenStore, meta: PublishMeta
    ) -> None:
        tokens.save("facebook", {"page_id": "p1", "page_token": "pt", "page_name": "Page"})
        status = {"processing_phase": {"status": "error", "error": {"message": "too low"}}}
        with pytest.raises(PublishError, match="too low"):
            FacebookPublisher(
                Recorder(self._handler(status)).client(), tokens, _limits(), sleep=_no_sleep
            ).post(video, meta)

    def test_without_a_token_it_says_to_authorise(
        self, video: Path, tokens: TokenStore, meta: PublishMeta
    ) -> None:
        with pytest.raises(PublishError, match="auth facebook"):
            FacebookPublisher(
                Recorder(self._handler({})).client(), tokens, _limits(), sleep=_no_sleep
            ).post(video, meta)


class TestTikTokChunks:
    def test_a_file_that_fits_goes_whole(self) -> None:
        plan = plan_chunks(4_194_304, load_limits()["tiktok"])
        assert plan.chunk_size == 4_194_304
        assert plan.ranges == ((0, 4_194_303),)

    def test_the_trailing_bytes_join_the_last_chunk(self) -> None:
        size = 150_000_123
        plan = plan_chunks(size, load_limits()["tiktok"])
        assert plan.chunk_size == 32_000_000
        assert len(plan.ranges) == size // 32_000_000
        assert plan.ranges[-1][1] == size - 1
        last = plan.ranges[-1][1] - plan.ranges[-1][0] + 1
        assert last < 128_000_000
        starts = [r[0] for r in plan.ranges]
        assert starts == sorted(starts) and starts[0] == 0


class TestTikTok:
    def test_inbox_upload_in_chunks_returns_the_caption(
        self, video: Path, tokens: TokenStore, meta: PublishMeta
    ) -> None:
        tokens.save("tiktok", {"access_token": "old", "refresh_token": "rt", "expires_at": 0})

        def handler(request: httpx.Request) -> httpx.Response:
            path = request.url.path
            if path == "/v2/oauth/token/":
                return httpx.Response(
                    200, json={"access_token": "new", "expires_in": 86400, "refresh_token": "rt2"}
                )
            if path == "/v2/post/publish/inbox/video/init/":
                assert request.headers["Authorization"] == "Bearer new"
                info = json.loads(request.content)["source_info"]
                assert info == {
                    "source": "FILE_UPLOAD",
                    "video_size": 1029,
                    "chunk_size": 400,
                    "total_chunk_count": 2,
                }
                return httpx.Response(
                    200,
                    json={
                        "data": {"publish_id": "pub1", "upload_url": "https://up.tiktok/u?x=1"},
                        "error": {"code": "ok"},
                    },
                )
            if request.method == "PUT":
                return httpx.Response(206)
            return httpx.Response(200, json={"data": {"status": "SEND_TO_USER_INBOX"}})

        rec = Recorder(handler)
        result = TikTokPublisher(
            rec.client(),
            tokens,
            {"client_key": "k", "client_secret": "s"},
            _limits(max_chunk_bytes=1000, chunk_bytes=400),
            sleep=_no_sleep,
            clock=lambda: 0.0,
        ).post(video, meta)

        puts = [
            (req.headers["Content-Range"], body)
            for req, body in rec.requests
            if req.method == "PUT"
        ]
        assert puts == [
            ("bytes 0-399/1029", VIDEO_BYTES[:400]),
            ("bytes 400-1028/1029", VIDEO_BYTES[400:]),
        ]
        assert result.remote_id == "pub1"
        assert "Cap\n\n#y" in result.note
        assert tokens.load("tiktok")["refresh_token"] == "rt2"

    def test_a_failed_status_raises(
        self, video: Path, tokens: TokenStore, meta: PublishMeta
    ) -> None:
        tokens.save("tiktok", {"access_token": "at", "refresh_token": "rt", "expires_at": 1e12})

        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path.endswith("/init/"):
                return httpx.Response(
                    200,
                    json={
                        "data": {"publish_id": "p", "upload_url": "https://up.tiktok/u"},
                        "error": {"code": "ok"},
                    },
                )
            if request.method == "PUT":
                return httpx.Response(201)
            return httpx.Response(
                200, json={"data": {"status": "FAILED", "fail_reason": "file_format_check_failed"}}
            )

        with pytest.raises(PublishError, match="file_format_check_failed"):
            TikTokPublisher(
                Recorder(handler).client(),
                tokens,
                {"client_key": "k", "client_secret": "s"},
                _limits(),
                sleep=_no_sleep,
                clock=lambda: 0.0,
            ).post(video, meta)

    def test_the_pending_share_cap_surfaces_as_an_error(
        self, video: Path, tokens: TokenStore, meta: PublishMeta
    ) -> None:
        tokens.save("tiktok", {"access_token": "at", "refresh_token": "rt", "expires_at": 1e12})

        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(403, json={"error": {"code": "spam_risk_too_many_pending_share"}})

        with pytest.raises(PublishError, match="spam_risk_too_many_pending_share"):
            TikTokPublisher(
                Recorder(handler).client(),
                tokens,
                {"client_key": "k", "client_secret": "s"},
                _limits(),
                sleep=_no_sleep,
                clock=lambda: 0.0,
            ).post(video, meta)
