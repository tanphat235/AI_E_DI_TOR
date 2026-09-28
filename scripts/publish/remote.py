"""Commands that move a job to the server and hand its timing to OpenClaw.

Only builds argv lists; ``cli`` runs them, so the exact commands are testable and
``--dry-run`` can print them.
"""

from __future__ import annotations

import json
import posixpath
import shlex
from dataclasses import dataclass
from pathlib import Path

from scripts.publish.schema import PublishMeta


@dataclass(frozen=True, slots=True)
class Remote:
    host: str
    root: str
    python: str = "python3"
    openclaw: str = "openclaw"

    def jobs(self) -> str:
        return posixpath.join(self.root, "jobs")


def ssh(remote: Remote, argv: list[str]) -> list[str]:
    return ["ssh", remote.host, shlex.join(argv)]


def scp(local: Path, remote: Remote, dest: str) -> list[str]:
    return ["scp", "-r", "-q", str(local), f"{remote.host}:{dest}"]


def push_code(code_dir: Path, remote: Remote) -> list[list[str]]:
    scripts_dir = posixpath.join(remote.root, "scripts")
    return [
        ssh(remote, ["mkdir", "-p", scripts_dir, remote.jobs()]),
        ssh(remote, ["rm", "-rf", posixpath.join(scripts_dir, "publish")]),
        scp(code_dir, remote, scripts_dir + "/"),
    ]


def push_auth(home: Path, remote: Remote, remote_home: str) -> list[list[str]]:
    return [
        ssh(remote, ["mkdir", "-p", remote_home]),
        ssh(remote, ["chmod", "700", remote_home]),
        scp(home / "apps.toml", remote, remote_home + "/"),
        scp(home / "tokens", remote, remote_home + "/"),
        ssh(remote, ["chmod", "-R", "go-rwx", remote_home]),
    ]


def schedule(
    job_dir: Path,
    meta: PublishMeta,
    remote: Remote,
    *,
    timeout_seconds: int,
    extra: list[str],
) -> list[list[str]]:
    job = job_dir.name
    post_argv = [remote.python, "-m", "scripts.publish.cli", "post", f"jobs/{job}"]
    create = [
        remote.openclaw,
        "automations",
        "create",
        "--at",
        meta.publish_at.isoformat(),
        "--name",
        f"aive-publish {job}",
        "--command-argv",
        json.dumps(post_argv),
        "--command-cwd",
        remote.root,
        "--timeout-seconds",
        str(timeout_seconds),
        *extra,
    ]
    # No rm of the remote job first: scp only overwrites what it sends, so a ledger
    # written by an earlier server-side post survives a reschedule.
    return [
        ssh(remote, ["mkdir", "-p", remote.jobs()]),
        scp(job_dir, remote, remote.jobs() + "/"),
        ssh(remote, create),
    ]
