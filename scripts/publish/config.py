"""Where publish state lives, and the files it reads from there."""

from __future__ import annotations

import json
import os
import tomllib
from pathlib import Path
from typing import Any

from scripts.publish.errors import PublishError

HOME_ENV = "AIVE_PUBLISH_HOME"
LIMITS_FILE = Path(__file__).with_name("limits.toml")


def publish_home() -> Path:
    raw = os.environ.get(HOME_ENV)
    return Path(raw).expanduser() if raw else Path.home() / ".aive-publish"


def load_limits(path: Path = LIMITS_FILE) -> dict[str, Any]:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def load_apps(home: Path) -> dict[str, Any]:
    path = home / "apps.toml"
    if not path.is_file():
        raise PublishError(
            "missing_apps",
            f"{path} not found",
            hint="copy scripts/publish/apps.example.toml there and fill it in",
        )
    with path.open("rb") as handle:
        return tomllib.load(handle)


def app_section(apps: dict[str, Any], platform: str, *keys: str) -> dict[str, str]:
    section = apps.get(platform) or {}
    missing = [key for key in keys if not str(section.get(key, "")).strip()]
    if missing:
        raise PublishError(
            "missing_app_keys",
            f"[{platform}] in apps.toml lacks {', '.join(missing)}",
            hint="see scripts/publish/README.md, One-time setup",
        )
    return {key: str(value) for key, value in section.items()}


class TokenStore:
    """One JSON file per platform under ``<home>/tokens``."""

    def __init__(self, home: Path) -> None:
        self.dir = home / "tokens"

    def path(self, platform: str) -> Path:
        return self.dir / f"{platform}.json"

    def load(self, platform: str) -> dict[str, Any]:
        path = self.path(platform)
        if not path.is_file():
            raise PublishError(
                "not_authorized",
                f"no token for {platform}",
                hint=f"run: python -m scripts.publish.cli auth {platform}",
            )
        data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        return data

    def save(self, platform: str, data: dict[str, Any]) -> Path:
        self.dir.mkdir(parents=True, exist_ok=True)
        path = self.path(platform)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        if os.name != "nt":
            tmp.chmod(0o600)
        tmp.replace(path)
        return path
