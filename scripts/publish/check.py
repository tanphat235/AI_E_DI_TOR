"""Catch what each platform would reject before spending an upload on it."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from scripts.publish.schema import PublishMeta, compose

PLACEHOLDER = "TODO"


@dataclass(frozen=True, slots=True)
class Issue:
    level: Literal["error", "warn"]
    platform: str
    message: str

    def __str__(self) -> str:
        return f"[{self.level}] {self.platform}: {self.message}"


@dataclass(frozen=True, slots=True)
class VideoInfo:
    duration: float
    width: int
    height: int
    fps: float


class UnreadableVideoError(Exception):
    """PyAV is installed but could not decode the file, or it has no video stream."""


def probe_video(path: Path) -> VideoInfo | None:
    """Read duration, size and fps with PyAV. None when PyAV is not installed."""
    try:
        import av
    except ImportError:
        return None
    try:
        with av.open(str(path)) as container:
            if not container.streams.video:
                raise UnreadableVideoError(f"{path.name} has no video stream")
            stream = container.streams.video[0]
            rate = stream.average_rate or stream.guessed_rate
            return VideoInfo(
                duration=(container.duration or 0) / 1_000_000,
                width=int(stream.width),
                height=int(stream.height),
                fps=float(rate) if rate else 0.0,
            )
    except av.error.FFmpegError as exc:
        raise UnreadableVideoError(f"{path.name} cannot be decoded: {exc}") from exc


def check_meta(
    meta: PublishMeta,
    limits: dict[str, Any],
    *,
    video_path: Path,
    now: datetime,
    info: VideoInfo | None,
    require_future: bool,
) -> list[Issue]:
    issues: list[Issue] = []

    def err(platform: str, message: str) -> None:
        issues.append(Issue("error", platform, message))

    def warn(platform: str, message: str) -> None:
        issues.append(Issue("warn", platform, message))

    if not video_path.is_file():
        err("video", f"{video_path} not found")
    elif video_path.stat().st_size == 0:
        err("video", f"{video_path} is empty")
    if require_future and meta.publish_at <= now:
        err("schedule", f"publish_at {meta.publish_at.isoformat()} is not in the future")
    if info is None:
        warn("video", "PyAV not installed; duration, size and fps were not checked")

    for platform in meta.platforms:
        section = getattr(meta, platform)
        if PLACEHOLDER in section.model_dump_json():
            err(platform, f"still contains a {PLACEHOLDER} placeholder")

    if meta.youtube is not None and "youtube" in meta.platforms:
        lim = limits["youtube"]
        yt = meta.youtube
        if len(yt.title) > lim["title_max"]:
            err("youtube", f"title is {len(yt.title)} chars, max {lim['title_max']}")
        text = compose(yt.description, yt.hashtags)
        if len(text) > lim["description_max"]:
            err("youtube", f"description is {len(text)} chars, max {lim['description_max']}")
        if any(c in yt.title + text for c in "<>"):
            err("youtube", "title or description contains < or >, which YouTube rejects")
        tags_len = sum(len(t) + 2 * (" " in t) for t in yt.tags) + max(len(yt.tags) - 1, 0)
        if tags_len > lim["tags_total_max"]:
            err("youtube", f"tags total {tags_len} chars, max {lim['tags_total_max']}")
        if len(yt.hashtags) > lim["hashtags_max"]:
            err(
                "youtube",
                f"{len(yt.hashtags)} hashtags; above {lim['hashtags_max']} all are ignored",
            )
        if "#shorts" not in {h.lower() for h in yt.hashtags}:
            warn("youtube", "no #Shorts hashtag")
        if info is not None:
            if info.duration > lim["shorts_max_seconds"]:
                warn(
                    "youtube",
                    f"{info.duration:.0f}s is longer than {lim['shorts_max_seconds']}s; "
                    "it will be a regular video, not a Short",
                )
            if info.width >= info.height:
                warn("youtube", f"{info.width}x{info.height} is not vertical; not a Short")

    if meta.facebook is not None and "facebook" in meta.platforms:
        lim = limits["facebook"]
        text = compose(meta.facebook.description, meta.facebook.hashtags)
        if len(text) > lim["description_max"]:
            warn(
                "facebook",
                f"description is {len(text)} chars (soft limit {lim['description_max']})",
            )
        if info is not None:
            if not lim["reels_min_seconds"] <= info.duration <= lim["reels_max_seconds"]:
                warn(
                    "facebook",
                    f"{info.duration:.0f}s is outside the Reels range "
                    f"{lim['reels_min_seconds']}-{lim['reels_max_seconds']}s",
                )
            if min(info.width, info.height) < lim["min_short_side"]:
                err("facebook", f"{info.width}x{info.height} is below {lim['min_short_side']}p")

    if meta.tiktok is not None and "tiktok" in meta.platforms:
        lim = limits["tiktok"]
        text = compose(meta.tiktok.caption, meta.tiktok.hashtags)
        if len(text) > lim["caption_max"]:
            err("tiktok", f"caption is {len(text)} chars, max {lim['caption_max']}")
        if info is not None:
            if info.duration > lim["max_seconds"]:
                err("tiktok", f"{info.duration:.0f}s is longer than {lim['max_seconds']}s")
            if info.fps and not lim["min_fps"] <= info.fps <= lim["max_fps"]:
                err("tiktok", f"{info.fps:.2f} fps is outside {lim['min_fps']}-{lim['max_fps']}")
            if min(info.width, info.height) < lim["min_side"]:
                err("tiktok", f"{info.width}x{info.height} is below {lim['min_side']}px")
            if max(info.width, info.height) > lim["max_side"]:
                err("tiktok", f"{info.width}x{info.height} is above {lim['max_side']}px")

    return issues
