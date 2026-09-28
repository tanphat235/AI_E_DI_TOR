from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import httpx
from scripts.publish.errors import PublishError
from scripts.publish.schema import Platform, PublishMeta

Sleep = Callable[[float], None]


@dataclass(frozen=True, slots=True)
class PostResult:
    remote_id: str
    url: str | None
    note: str = ""


class Publisher(Protocol):
    @property
    def name(self) -> Platform: ...

    def post(self, video: Path, meta: PublishMeta) -> PostResult: ...


def api_error(platform: str, step: str, response: httpx.Response) -> PublishError:
    body = response.text[:600].replace("\n", " ")
    hint = ""
    if response.status_code in (401, 403):
        hint = f"token rejected; re-run: python -m scripts.publish.cli auth {platform}"
    return PublishError(
        f"{platform}_{step}_failed",
        f"HTTP {response.status_code} at {step}: {body}",
        hint=hint,
    )


def iter_file(path: Path, start: int, end: int, block: int) -> Iterator[bytes]:
    """Bytes ``start`` up to, not including, ``end``."""
    with path.open("rb") as handle:
        handle.seek(start)
        remaining = end - start
        while remaining > 0:
            chunk = handle.read(min(block, remaining))
            if not chunk:
                return
            remaining -= len(chunk)
            yield chunk
