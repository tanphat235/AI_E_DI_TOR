"""publish.json and ledger.json.

Frozen with ``extra="forbid"`` like every AIVE model: publish.json is written by hand
or by an agent, and a misspelt key has to fail rather than silently post without it.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Literal

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    model_validator,
)
from scripts.publish.errors import PublishError

Platform = Literal["youtube", "facebook", "tiktok"]
ALL_PLATFORMS: tuple[Platform, ...] = ("facebook", "youtube", "tiktok")

META_FILE = "publish.json"
LEDGER_FILE = "ledger.json"

Hashtag = Annotated[str, StringConstraints(pattern=r"^#[^\s#]+$")]


class _Strict(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class YouTubeMeta(_Strict):
    title: str = Field(min_length=1)
    description: str = ""
    tags: tuple[str, ...] = ()
    hashtags: tuple[Hashtag, ...] = ()
    privacy: Literal["public", "unlisted", "private"] = "public"
    made_for_kids: bool = False
    category_id: str | None = None


class FacebookMeta(_Strict):
    description: str = Field(min_length=1)
    hashtags: tuple[Hashtag, ...] = ()
    title: str | None = None


class TikTokMeta(_Strict):
    caption: str = Field(min_length=1)
    hashtags: tuple[Hashtag, ...] = ()


class PublishMeta(_Strict):
    video: str = Field(min_length=1)
    publish_at: AwareDatetime
    platforms: tuple[Platform, ...] = ALL_PLATFORMS
    youtube: YouTubeMeta | None = None
    facebook: FacebookMeta | None = None
    tiktok: TikTokMeta | None = None

    @model_validator(mode="after")
    def _sections_for_platforms(self) -> PublishMeta:
        if not self.platforms:
            raise ValueError("platforms is empty")
        if len(set(self.platforms)) != len(self.platforms):
            raise ValueError("platforms lists a platform twice")
        missing = [p for p in self.platforms if getattr(self, p) is None]
        if missing:
            raise ValueError(f"platforms names {missing} but publish.json has no section for it")
        return self


def compose(body: str, hashtags: tuple[str, ...]) -> str:
    """Body text, then the hashtags on a line of their own."""
    text = body.rstrip()
    if not hashtags:
        return text
    tags = " ".join(hashtags)
    return f"{text}\n\n{tags}" if text else tags


class LedgerEntry(_Strict):
    status: Literal["posted", "failed"]
    at: AwareDatetime
    remote_id: str | None = None
    url: str | None = None
    note: str = ""
    error: str = ""


class Ledger(_Strict):
    entries: dict[Platform, LedgerEntry] = Field(default_factory=dict)

    def posted(self, platform: Platform) -> bool:
        entry = self.entries.get(platform)
        return entry is not None and entry.status == "posted"

    def with_entry(self, platform: Platform, entry: LedgerEntry) -> Ledger:
        return Ledger(entries={**self.entries, platform: entry})


def load_meta(job_dir: Path) -> PublishMeta:
    path = job_dir / META_FILE
    if not path.is_file():
        raise PublishError("missing_meta", f"{path} not found", hint="run `draft` first")
    try:
        return PublishMeta.model_validate_json(path.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise PublishError("invalid_meta", f"{path}: {exc}") from exc


def load_ledger(job_dir: Path) -> Ledger:
    path = job_dir / LEDGER_FILE
    if not path.is_file():
        return Ledger()
    return Ledger.model_validate_json(path.read_text(encoding="utf-8"))


def save_ledger(job_dir: Path, ledger: Ledger) -> None:
    path = job_dir / LEDGER_FILE
    tmp = path.with_suffix(".tmp")
    tmp.write_text(ledger.model_dump_json(indent=2), encoding="utf-8")
    tmp.replace(path)


def write_meta(job_dir: Path, payload: dict[str, object]) -> Path:
    """Write a draft that is allowed to be incomplete (TODO placeholders)."""
    path = job_dir / META_FILE
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def now_utc() -> datetime:
    return datetime.now(UTC)
