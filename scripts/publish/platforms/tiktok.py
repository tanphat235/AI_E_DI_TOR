"""TikTok Content Posting API, upload-to-inbox.

The video lands in the creator's TikTok inbox and is posted from the app. Inbox
upload takes no caption, so the caption is returned in the note for pasting.

https://developers.tiktok.com/doc/content-posting-api-reference-upload-video
https://developers.tiktok.com/doc/content-posting-api-media-transfer-guide
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import httpx
from scripts.publish.config import TokenStore
from scripts.publish.errors import PublishError
from scripts.publish.platforms.base import PostResult, Sleep, api_error, iter_file
from scripts.publish.schema import PublishMeta, compose

API = "https://open.tiktokapis.com"
AUTHORIZE_URL = "https://www.tiktok.com/v2/auth/authorize/"
TOKEN_URL = f"{API}/v2/oauth/token/"
SCOPE = "video.upload"
DONE = frozenset({"SEND_TO_USER_INBOX", "PUBLISH_COMPLETE"})


@dataclass(frozen=True, slots=True)
class ChunkPlan:
    chunk_size: int
    ranges: tuple[tuple[int, int], ...]  # inclusive (first, last) byte offsets


def plan_chunks(size: int, limits: dict[str, Any]) -> ChunkPlan:
    """Whole upload when it fits in one chunk; otherwise the trailing bytes join the last."""
    if size <= 0:
        raise PublishError("tiktok_empty_video", "video is empty")
    if size <= int(limits["max_chunk_bytes"]):
        return ChunkPlan(size, ((0, size - 1),))
    chunk = int(limits["chunk_bytes"])
    count = size // chunk
    if count > int(limits["max_chunks"]):
        raise PublishError("tiktok_too_many_chunks", f"{count} chunks exceeds the maximum")
    ranges = [(i * chunk, (i + 1) * chunk - 1) for i in range(count)]
    ranges[-1] = (ranges[-1][0], size - 1)
    return ChunkPlan(chunk, tuple(ranges))


def refresh_access_token(
    client: httpx.Client, tokens: TokenStore, app: dict[str, str], *, now: float
) -> str:
    token = tokens.load("tiktok")
    if token.get("access_token") and float(token.get("expires_at", 0)) > now + 60:
        return str(token["access_token"])
    response = client.post(
        TOKEN_URL,
        data={
            "client_key": app["client_key"],
            "client_secret": app["client_secret"],
            "grant_type": "refresh_token",
            "refresh_token": token.get("refresh_token", ""),
        },
    )
    data = response.json() if response.status_code == 200 else {}
    if "access_token" not in data:
        raise api_error("tiktok", "token_refresh", response)
    token.update(
        access_token=data["access_token"],
        expires_at=now + float(data.get("expires_in", 86400)),
        refresh_token=data.get("refresh_token", token.get("refresh_token")),
        refresh_expires_at=now + float(data.get("refresh_expires_in", 0)),
    )
    tokens.save("tiktok", token)
    return str(data["access_token"])


class TikTokPublisher:
    name: Literal["tiktok"] = "tiktok"

    def __init__(
        self,
        client: httpx.Client,
        tokens: TokenStore,
        app: dict[str, str],
        limits: dict[str, Any],
        *,
        sleep: Sleep = time.sleep,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.client = client
        self.tokens = tokens
        self.app = app
        self.limits = limits
        self.sleep = sleep
        self.clock = clock

    def post(self, video: Path, meta: PublishMeta) -> PostResult:
        tt = meta.tiktok
        if tt is None:
            raise PublishError("invalid_meta", "publish.json has no tiktok section")
        token = refresh_access_token(self.client, self.tokens, self.app, now=self.clock())
        auth = {"Authorization": f"Bearer {token}"}
        size = video.stat().st_size
        plan = plan_chunks(size, self.limits["tiktok"])

        init = self.client.post(
            f"{API}/v2/post/publish/inbox/video/init/",
            headers={**auth, "Content-Type": "application/json; charset=UTF-8"},
            json={
                "source_info": {
                    "source": "FILE_UPLOAD",
                    "video_size": size,
                    "chunk_size": plan.chunk_size,
                    "total_chunk_count": len(plan.ranges),
                }
            },
        )
        try:
            body = init.json()
        except ValueError:
            body = {}
        if init.status_code != 200 or (body.get("error") or {}).get("code") != "ok":
            raise api_error("tiktok", "upload_init", init)
        publish_id = str(body["data"]["publish_id"])
        upload_url = str(body["data"]["upload_url"])

        block = int(self.limits["upload"]["read_block_bytes"])
        attempts = int(self.limits["upload"]["max_attempts"])
        for first, last in plan.ranges:
            self._put_chunk(upload_url, video, first, last, size, block, attempts)

        caption = compose(tt.caption, tt.hashtags)
        state = self._wait(publish_id, auth)
        note = f"{state}. Open the TikTok inbox notification to post it. Caption:\n{caption}"
        return PostResult(publish_id, None, note)

    def _put_chunk(
        self, url: str, video: Path, first: int, last: int, size: int, block: int, attempts: int
    ) -> None:
        for attempt in range(attempts):
            try:
                response = self.client.put(
                    url,
                    headers={
                        "Content-Type": "video/mp4",
                        "Content-Length": str(last - first + 1),
                        "Content-Range": f"bytes {first}-{last}/{size}",
                    },
                    content=iter_file(video, first, last + 1, block),
                )
            except httpx.TransportError:
                self.sleep(min(2**attempt, 60))
                continue
            if response.status_code in (200, 201, 206):
                return
            if response.status_code < 500:
                raise api_error("tiktok", "upload", response)
            self.sleep(min(2**attempt, 60))
        raise PublishError("tiktok_upload_incomplete", f"chunk {first}-{last} failed {attempts}x")

    def _wait(self, publish_id: str, auth: dict[str, str]) -> str:
        lim = self.limits["tiktok"]
        status = "PROCESSING_UPLOAD"
        for _ in range(int(lim["status_poll_attempts"])):
            response = self.client.post(
                f"{API}/v2/post/publish/status/fetch/",
                headers={**auth, "Content-Type": "application/json; charset=UTF-8"},
                json={"publish_id": publish_id},
            )
            if response.status_code != 200:
                raise api_error("tiktok", "status", response)
            data = response.json().get("data") or {}
            status = str(data.get("status", status))
            if status == "FAILED":
                raise PublishError(
                    "tiktok_processing_failed", str(data.get("fail_reason", "unknown reason"))
                )
            if status in DONE:
                return "in your TikTok inbox"
            self.sleep(float(lim["status_poll_seconds"]))
        return f"uploaded; TikTok still reports {status}"
