"""Post one job to every platform it names, recording each outcome as it happens."""

from __future__ import annotations

import os
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import httpx
from scripts.publish.errors import PublishError
from scripts.publish.platforms.base import Publisher
from scripts.publish.schema import (
    Ledger,
    LedgerEntry,
    Platform,
    PublishMeta,
    load_ledger,
    load_meta,
    save_ledger,
)

LOCK_FILE = ".post.lock"


@contextmanager
def job_lock(job_dir: Path) -> Iterator[None]:
    """Refuse a second concurrent ``post`` of the same job - it would double-upload."""
    path = job_dir / LOCK_FILE
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise PublishError(
            "job_locked",
            f"{path} exists: another post of this job is running",
            hint="if no post is running, delete the lock file",
        ) from exc
    try:
        os.write(fd, str(os.getpid()).encode("ascii"))
        os.close(fd)
        yield
    finally:
        path.unlink(missing_ok=True)


def post_job(
    job_dir: Path,
    publishers: Mapping[Platform, Callable[[], Publisher]],
    *,
    now: Callable[[], datetime],
    only: tuple[Platform, ...] = (),
) -> tuple[PublishMeta, Ledger]:
    """Publishers are factories so a platform already posted never needs its token."""
    meta = load_meta(job_dir)
    video = job_dir / meta.video
    ledger = load_ledger(job_dir)
    targets = [p for p in meta.platforms if not only or p in only]
    with job_lock(job_dir):
        for platform in targets:
            if ledger.posted(platform):
                continue
            try:
                result = publishers[platform]().post(video, meta)
            except (PublishError, httpx.HTTPError, KeyError, ValueError) as exc:
                entry = LedgerEntry(status="failed", at=now(), error=f"{type(exc).__name__}: {exc}")
            else:
                entry = LedgerEntry(
                    status="posted",
                    at=now(),
                    remote_id=result.remote_id,
                    url=result.url,
                    note=result.note,
                )
            ledger = ledger.with_entry(platform, entry)
            save_ledger(job_dir, ledger)
    return meta, ledger


def summary(meta: PublishMeta, ledger: Ledger, job_name: str) -> str:
    lines = [f"publish {job_name} ({meta.video})"]
    for platform in meta.platforms:
        entry = ledger.entries.get(platform)
        if entry is None:
            lines.append(f"- {platform}: not attempted")
        elif entry.status == "posted":
            lines.append(f"- {platform}: posted {entry.url or entry.remote_id}")
            if entry.note:
                lines.append("  " + entry.note.replace("\n", "\n  "))
        else:
            lines.append(f"- {platform}: FAILED {entry.error}")
    return "\n".join(lines)
