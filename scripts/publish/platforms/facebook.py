"""Facebook Page Reels: start a session, send the bytes to rupload, finish.

https://developers.facebook.com/docs/video-api/guides/reels-publishing/
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Literal

import httpx
from scripts.publish.config import TokenStore
from scripts.publish.errors import PublishError
from scripts.publish.platforms.base import PostResult, Sleep, api_error, iter_file
from scripts.publish.schema import PublishMeta, compose


class FacebookPublisher:
    name: Literal["facebook"] = "facebook"

    def __init__(
        self,
        client: httpx.Client,
        tokens: TokenStore,
        limits: dict[str, Any],
        *,
        sleep: Sleep = time.sleep,
    ) -> None:
        self.client = client
        self.tokens = tokens
        self.limits = limits
        self.sleep = sleep

    def post(self, video: Path, meta: PublishMeta) -> PostResult:
        fb = meta.facebook
        if fb is None:
            raise PublishError("invalid_meta", "publish.json has no facebook section")
        token = self.tokens.load("facebook")
        page_id, page_token = str(token["page_id"]), str(token["page_token"])
        lim = self.limits["facebook"]
        version = lim["graph_version"]
        graph = f"https://graph.facebook.com/{version}"
        size = video.stat().st_size

        start = self.client.post(
            f"{graph}/{page_id}/video_reels",
            data={"upload_phase": "start", "access_token": page_token},
        )
        if start.status_code != 200:
            raise api_error("facebook", "upload_start", start)
        started = start.json()
        video_id = str(started["video_id"])
        upload_url = started.get("upload_url") or (
            f"https://rupload.facebook.com/video-upload/{version}/{video_id}"
        )

        # rupload wants "OAuth", not "Bearer" - Bearer fails without naming the cause.
        sent = self.client.post(
            upload_url,
            headers={
                "Authorization": f"OAuth {page_token}",
                "offset": "0",
                "file_size": str(size),
                "Content-Type": "application/octet-stream",
                "Content-Length": str(size),
            },
            content=iter_file(video, 0, size, int(self.limits["upload"]["read_block_bytes"])),
        )
        if sent.status_code != 200 or not sent.json().get("success"):
            raise api_error("facebook", "upload", sent)

        fields: dict[str, str] = {
            "upload_phase": "finish",
            "video_id": video_id,
            "video_state": "PUBLISHED",
            "description": compose(fb.description, fb.hashtags),
            "access_token": page_token,
        }
        if fb.title:
            fields["title"] = fb.title
        finish = self.client.post(f"{graph}/{page_id}/video_reels", data=fields)
        if finish.status_code != 200 or not finish.json().get("success"):
            raise api_error("facebook", "upload_finish", finish)

        url = f"https://www.facebook.com/reel/{video_id}"
        note = self._wait_published(graph, video_id, page_token)
        return PostResult(video_id, url, note)

    def _wait_published(self, graph: str, video_id: str, page_token: str) -> str:
        lim = self.limits["facebook"]
        for _ in range(int(lim["status_poll_attempts"])):
            response = self.client.get(
                f"{graph}/{video_id}",
                params={"fields": "status", "access_token": page_token},
            )
            if response.status_code != 200:
                raise api_error("facebook", "status", response)
            status: dict[str, Any] = response.json().get("status", {})
            for phase in ("uploading_phase", "processing_phase", "publishing_phase"):
                part = status.get(phase) or {}
                if part.get("status") == "error":
                    message = (part.get("error") or {}).get("message", "unknown error")
                    raise PublishError("facebook_processing_failed", f"{phase}: {message}")
            if (status.get("publishing_phase") or {}).get("status") == "complete":
                return "published"
            self.sleep(float(lim["status_poll_seconds"]))
        return "accepted; Facebook was still processing when polling stopped"
