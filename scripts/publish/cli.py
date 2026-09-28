"""python -m scripts.publish.cli <command> - see README.md in this folder.

Human-readable summary on stdout (OpenClaw's --announce forwards it as-is);
``--json`` switches to a machine-readable object. Errors go to stderr.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from collections.abc import Callable, Sequence
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
from scripts.publish import auth, remote
from scripts.publish.check import Issue, UnreadableVideoError, check_meta, probe_video
from scripts.publish.config import TokenStore, app_section, load_apps, load_limits, publish_home
from scripts.publish.errors import PublishError
from scripts.publish.platforms.base import Publisher
from scripts.publish.platforms.facebook import FacebookPublisher
from scripts.publish.platforms.tiktok import TikTokPublisher
from scripts.publish.platforms.youtube import YouTubePublisher
from scripts.publish.runner import post_job, summary
from scripts.publish.schema import (
    ALL_PLATFORMS,
    META_FILE,
    Platform,
    load_ledger,
    load_meta,
    now_utc,
    write_meta,
)

EXIT_OK, EXIT_CONFIG, EXIT_CHECK, EXIT_PARTIAL = 0, 2, 3, 4
CODE_DIR = Path(__file__).resolve().parent
REMOTE_HOME = "~/.aive-publish"

Run = Callable[[list[str]], int]


def _out(text: str) -> None:
    sys.stdout.write(text + "\n")
    sys.stdout.flush()


def _err(text: str) -> None:
    sys.stderr.write(text + "\n")
    sys.stderr.flush()


def _run(argv: list[str]) -> int:
    return subprocess.run(argv, check=False).returncode


def _platforms(raw: str | None) -> tuple[Platform, ...]:
    if not raw:
        return ()
    names = tuple(p.strip() for p in raw.split(",") if p.strip())
    bad = [n for n in names if n not in ALL_PLATFORMS]
    if bad:
        raise PublishError("bad_platform", f"unknown platform(s): {', '.join(bad)}")
    return names  # type: ignore[return-value]


def _client(limits: dict[str, Any]) -> httpx.Client:
    timeout = float(limits["upload"]["timeout_seconds"])
    return httpx.Client(timeout=httpx.Timeout(timeout, connect=30.0), follow_redirects=False)


# --------------------------------------------------------------------------- commands


def cmd_auth(args: argparse.Namespace) -> int:
    home = publish_home()
    apps, limits, tokens = load_apps(home), load_limits(), TokenStore(home)

    def ask(prompt: str) -> str:
        sys.stderr.write(prompt)
        sys.stderr.flush()
        return sys.stdin.readline()

    common: dict[str, Any] = {"port": args.port, "paste": args.paste, "say": _err, "ask": ask}
    with _client(limits) as client:
        if args.platform == "youtube":
            app = app_section(apps, "youtube", "client_id", "client_secret")
            result = auth.auth_youtube(client, tokens, app, **common)
        elif args.platform == "facebook":
            app = app_section(apps, "facebook", "app_id", "app_secret")
            result = auth.auth_facebook(client, tokens, app, limits, page_id=args.page_id, **common)
        else:
            app = app_section(apps, "tiktok", "client_key", "client_secret")
            result = auth.auth_tiktok(client, tokens, app, **common)
    _out(json.dumps({**result, "saved": str(tokens.path(args.platform))}, ensure_ascii=False))
    return EXIT_OK


def cmd_draft(args: argparse.Namespace) -> int:
    video = Path(args.video).resolve()
    if not video.is_file():
        raise PublishError("missing_video", f"{video} not found")
    job_dir = Path(args.job_dir) if args.job_dir else video.parent / "publish" / video.stem
    if (job_dir / META_FILE).exists() and not args.force:
        raise PublishError(
            "draft_exists", f"{job_dir / META_FILE} exists", hint="edit it, or pass --force"
        )
    job_dir.mkdir(parents=True, exist_ok=True)
    target = job_dir / video.name
    if not target.exists() or args.force:
        shutil.copy2(video, target)
    if args.at:
        publish_at = datetime.fromisoformat(args.at)
        if publish_at.tzinfo is None:
            raise PublishError("naive_time", "--at needs an offset, e.g. 2026-10-01T19:00:00+07:00")
    else:
        publish_at = (datetime.now().astimezone() + timedelta(hours=1)).replace(
            minute=0, second=0, microsecond=0
        )
    title = args.title or "TODO"
    payload: dict[str, object] = {
        "video": video.name,
        "publish_at": publish_at.isoformat(),
        "platforms": list(ALL_PLATFORMS),
        "youtube": {
            "title": title,
            "description": "TODO",
            "tags": [],
            "hashtags": ["#Shorts"],
        },
        "facebook": {"description": "TODO", "hashtags": []},
        "tiktok": {"caption": "TODO", "hashtags": []},
    }
    path = write_meta(job_dir, payload)
    _out(json.dumps({"job_dir": str(job_dir), "meta": str(path)}, ensure_ascii=False))
    return EXIT_OK


def _issues(job_dir: Path, *, require_future: bool) -> list[Issue]:
    meta = load_meta(job_dir)
    video = job_dir / meta.video
    unreadable: list[Issue] = []
    info = None
    if video.is_file():
        try:
            info = probe_video(video)
        except UnreadableVideoError as exc:
            unreadable.append(Issue("error", "video", str(exc)))
    issues = check_meta(
        meta,
        load_limits(),
        video_path=video,
        now=now_utc(),
        info=info,
        require_future=require_future,
    )
    if unreadable:
        issues = unreadable + [i for i in issues if not i.message.startswith("PyAV not")]
    return issues


def cmd_check(args: argparse.Namespace) -> int:
    issues = _issues(Path(args.job_dir), require_future=not args.now)
    errors = [i for i in issues if i.level == "error"]
    if args.json:
        rows = [{"level": i.level, "platform": i.platform, "message": i.message} for i in issues]
        _out(json.dumps(rows, ensure_ascii=False))
    else:
        _out("\n".join(str(i) for i in issues) or "ok")
    return EXIT_CHECK if errors else EXIT_OK


def build_publishers(
    client: httpx.Client, home: Path, limits: dict[str, Any]
) -> dict[Platform, Callable[[], Publisher]]:
    tokens = TokenStore(home)

    def apps() -> dict[str, Any]:
        return load_apps(home)

    return {
        "youtube": lambda: YouTubePublisher(
            client,
            tokens,
            app_section(apps(), "youtube", "client_id", "client_secret"),
            limits,
        ),
        "facebook": lambda: FacebookPublisher(client, tokens, limits),
        "tiktok": lambda: TikTokPublisher(
            client,
            tokens,
            app_section(apps(), "tiktok", "client_key", "client_secret"),
            limits,
        ),
    }


def cmd_post(args: argparse.Namespace) -> int:
    job_dir = Path(args.job_dir)
    errors = [i for i in _issues(job_dir, require_future=False) if i.level == "error"]
    if errors:
        _out("\n".join(str(i) for i in errors))
        return EXIT_CHECK
    limits = load_limits()
    with _client(limits) as client:
        post_job(
            job_dir,
            build_publishers(client, publish_home(), limits),
            now=now_utc,
            only=_platforms(args.only),
        )
    return _report(job_dir, args.json)


def _report(job_dir: Path, as_json: bool) -> int:
    meta, ledger = load_meta(job_dir), load_ledger(job_dir)
    if as_json:
        _out(ledger.model_dump_json(indent=2))
    else:
        _out(summary(meta, ledger, job_dir.name))
    failed = [p for p in meta.platforms if not ledger.posted(p)]
    return EXIT_PARTIAL if failed else EXIT_OK


def cmd_status(args: argparse.Namespace) -> int:
    return _report(Path(args.job_dir), args.json)


def _remote(args: argparse.Namespace) -> remote.Remote:
    return remote.Remote(host=args.host, root=args.root, python=args.python, openclaw=args.openclaw)


def _execute(commands: list[list[str]], *, dry_run: bool, run: Run) -> int:
    for argv in commands:
        if dry_run:
            _out(subprocess.list2cmdline(argv))
            continue
        _err("$ " + subprocess.list2cmdline(argv))
        code = run(argv)
        if code != 0:
            _err(f"command failed with exit code {code}")
            return EXIT_CONFIG
    return EXIT_OK


def cmd_push_code(args: argparse.Namespace, run: Run = _run) -> int:
    return _execute(remote.push_code(CODE_DIR, _remote(args)), dry_run=args.dry_run, run=run)


def cmd_push_auth(args: argparse.Namespace, run: Run = _run) -> int:
    home = publish_home()
    for needed in (home / "apps.toml", home / "tokens"):
        if not needed.exists():
            raise PublishError("missing_auth", f"{needed} not found", hint="run auth first")
    commands = remote.push_auth(home, _remote(args), args.remote_home)
    return _execute(commands, dry_run=args.dry_run, run=run)


def cmd_schedule(args: argparse.Namespace, run: Run = _run) -> int:
    job_dir = Path(args.job_dir)
    issues = _issues(job_dir, require_future=True)
    errors = [i for i in issues if i.level == "error"]
    if errors:
        _out("\n".join(str(i) for i in errors))
        return EXIT_CHECK
    meta = load_meta(job_dir)
    commands = remote.schedule(
        job_dir,
        meta,
        _remote(args),
        timeout_seconds=int(load_limits()["schedule"]["timeout_seconds"]),
        extra=list(args.oc_arg) if args.oc_arg else ["--announce"],
    )
    return _execute(commands, dry_run=args.dry_run, run=run)


# --------------------------------------------------------------------------- parser


def _add_remote(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--host", required=True, help="ssh target, e.g. user@server")
    parser.add_argument("--root", default="/srv/aive-publish", help="server working dir")
    parser.add_argument("--python", default="python3", help="python on the server")
    parser.add_argument("--openclaw", default="openclaw", help="openclaw binary on the server")
    parser.add_argument("--dry-run", action="store_true", help="print commands, run nothing")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m scripts.publish.cli")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("auth", help="one-time OAuth for a platform")
    p.add_argument("platform", choices=ALL_PLATFORMS)
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--paste", action="store_true", help="paste the redirect URL by hand")
    p.add_argument("--page-id", default=None, help="facebook: which Page")
    p.set_defaults(func=cmd_auth)

    p = sub.add_parser("draft", help="create a job folder with a publish.json skeleton")
    p.add_argument("video")
    p.add_argument("--job-dir", default=None)
    p.add_argument("--at", default=None, help="ISO time with offset")
    p.add_argument("--title", default=None)
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_draft)

    p = sub.add_parser("check", help="validate a job against limits.toml")
    p.add_argument("job_dir")
    p.add_argument("--now", action="store_true", help="skip the publish_at-in-future check")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_check)

    p = sub.add_parser("post", help="post a job now; re-running retries only failures")
    p.add_argument("job_dir")
    p.add_argument("--only", default=None, help="comma list, e.g. facebook,tiktok")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_post)

    p = sub.add_parser("status", help="what the ledger says")
    p.add_argument("job_dir")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("push-code", help="copy scripts/publish to the server")
    _add_remote(p)
    p.set_defaults(func=cmd_push_code)

    p = sub.add_parser("push-auth", help="copy apps.toml and tokens/ to the server")
    _add_remote(p)
    p.add_argument("--remote-home", default=REMOTE_HOME)
    p.set_defaults(func=cmd_push_auth)

    p = sub.add_parser("schedule", help="copy a job to the server and create an OpenClaw job")
    p.add_argument("job_dir")
    _add_remote(p)
    p.add_argument(
        "--oc-arg",
        action="append",
        help="extra arg for `openclaw automations create` (repeatable); "
        "replaces the default --announce",
    )
    p.set_defaults(func=cmd_schedule)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")
    args = build_parser().parse_args(argv)
    try:
        code: int = args.func(args)
    except PublishError as exc:
        _err(f"error: {exc}")
        return EXIT_CONFIG
    return code


if __name__ == "__main__":
    raise SystemExit(main())
