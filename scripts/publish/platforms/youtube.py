"""YouTube Data API v3 resumable upload.

https://developers.google.com/youtube/v3/guides/using_resumable_upload_protocol
"""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

import httpx
from scripts.publish.config import TokenStore
from scripts.publish.errors import PublishError
from scripts.publish.platforms.base import PostResult, Sleep, api_error, iter_file
from scripts.publish.schema import PublishMeta, compose

TOKEN_URL = "https://oauth2.googleapis.com/token"
UPLOAD_URL = "https://www.googleapis.com/upload/youtube/v3/videos"
SCOPE = "https://www.googleapis.com/auth/youtube.upload"
RETRYABLE = frozenset({500, 502, 503, 504})

LOCKED_NOTE = (
    "If the Google project has not passed the YouTube API audit, YouTube locks this "
    "upload to private: make it public in YouTube Studio."
)


def refresh_access_token(
    client: httpx.Client, tokens: TokenStore, app: dict[str, str], *, now: float
) -> str:
    token = tokens.load("youtube")
    if token.get("access_token") and float(token.get("expires_at", 0)) > now + 60:
        return str(token["access_token"])
    if not token.get("refresh_token"):
        raise PublishError(
            "not_authorized", "youtube token has no refresh_token", hint="re-run auth youtube"
        )
    response = client.post(
        TOKEN_URL,
        data={
            "client_id": app["client_id"],
            "client_secret": app["client_secret"],
            "refresh_token": token["refresh_token"],
            "grant_type": "refresh_token",
        },
    )
    if response.status_code != 200:
        raise api_error("youtube", "token_refresh", response)
    data = response.json()
    token["access_token"] = data["access_token"]
    token["expires_at"] = now + float(data.get("expires_in", 3600))
    tokens.save("youtube", token)
    return str(data["access_token"])


class YouTubePublisher:
    name: Literal["youtube"] = "youtube"

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
        yt = meta.youtube
        if yt is None:
            raise PublishError("invalid_meta", "publish.json has no youtube section")
        token = refresh_access_token(self.client, self.tokens, self.app, now=self.clock())
        size = video.stat().st_size
        body = {
            "snippet": {
                "title": yt.title,
                "description": compose(yt.description, yt.hashtags),
                "tags": list(yt.tags),
                "categoryId": yt.category_id or self.limits["youtube"]["category_id"],
            },
            "status": {
                "privacyStatus": yt.privacy,
                "selfDeclaredMadeForKids": yt.made_for_kids,
            },
        }
        auth = {"Authorization": f"Bearer {token}"}
        init = self.client.post(
            UPLOAD_URL,
            params={"uploadType": "resumable", "part": "snippet,status"},
            headers={
                **auth,
                "X-Upload-Content-Length": str(size),
                "X-Upload-Content-Type": "video/*",
            },
            json=body,
        )
        if init.status_code != 200 or "location" not in init.headers:
            raise api_error("youtube", "upload_init", init)
        session = init.headers["location"]

        upload = self.limits["upload"]
        offset = 0
        for attempt in range(int(upload["max_attempts"])):
            response: httpx.Response | None
            try:
                response = self.client.put(
                    session,
                    headers={
                        **auth,
                        "Content-Length": str(size - offset),
                        "Content-Type": "video/*",
                        **(
                            {"Content-Range": f"bytes {offset}-{size - 1}/{size}"} if offset else {}
                        ),
                    },
                    content=iter_file(video, offset, size, int(upload["read_block_bytes"])),
                )
            except httpx.TransportError:
                response = None
            if response is not None and response.status_code in (200, 201):
                return self._result(response, yt.privacy)
            if response is not None and response.status_code not in RETRYABLE:
                raise api_error("youtube", "upload", response)
            self.sleep(min(2**attempt, 60))
            status = self._query(session, size, auth)
            if status.status_code in (200, 201):
                return self._result(status, yt.privacy)
            offset = _next_offset(status)
        raise PublishError(
            "youtube_upload_incomplete",
            f"gave up after {upload['max_attempts']} attempts",
            hint="re-run post; the ledger keeps the other platforms",
        )

    def _query(self, session: str, size: int, auth: dict[str, str]) -> httpx.Response:
        response = self.client.put(
            session,
            headers={**auth, "Content-Length": "0", "Content-Range": f"bytes */{size}"},
        )
        if response.status_code not in (200, 201, 308):
            raise api_error("youtube", "upload_status", response)
        return response

    def _result(self, response: httpx.Response, requested: str) -> PostResult:
        video_id = str(response.json()["id"])
        note = LOCKED_NOTE if requested != "private" else ""
        return PostResult(video_id, f"https://youtube.com/shorts/{video_id}", note)


def _next_offset(status: httpx.Response) -> int:
    """A 308's ``Range: bytes=0-N`` means N+1 bytes are stored; no header means none."""
    value = status.headers.get("range", "")
    if not value:
        return 0
    return int(value.rsplit("-", 1)[1]) + 1
