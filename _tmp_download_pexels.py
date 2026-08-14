"""Download free Pexels videos for AIVE stock footage.

Requires a free API key: https://www.pexels.com/api/
PowerShell:
  $env:PEXELS_API_KEY = 'your_key'
  .\\.venv\\Scripts\\python.exe _tmp_download_pexels.py

Current profile: drone / aerial moving nature (mountains, forest, sea, river,
sunrise/sunset). Skips metadata that suggests people, animals, or buildings.
Optionally copies local reference clips listed in REF_COPIES when present.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

DEST = Path(__file__).resolve().parent / "projects" / "stock-pexels" / "raw"
COUNT = 28
QUERIES = (
    "drone aerial mountain",
    "drone forest aerial flyover",
    "aerial ocean coastline drone",
    "drone river canyon aerial",
    "aerial sunrise mountain drone",
    "drone sunset seascape aerial",
    "aerial waterfall drone",
    "drone lake landscape flyover",
    "aerial cliff coast drone",
    "drone rice terrace aerial",
    "aerial green hills flyover",
    "drone fjord aerial landscape",
)
BLOCK = (
    "people",
    "person",
    "man",
    "woman",
    "human",
    "crowd",
    "portrait",
    "selfie",
    "animal",
    "dog",
    "cat",
    "bird",
    "wildlife",
    "horse",
    "fish",
    "house",
    "home",
    "building",
    "city",
    "street",
    "car",
    "urban",
    "office",
    "room",
    "timelapse static",
)
MIN_DURATION = 8
MIN_WIDTH = 1920
TIMEOUT = 180
# Set True to wipe existing pexels_*.mp4 before downloading.
CLEAR_EXISTING = False

REF_COPIES = (
    Path(r"C:\Users\phat.phamt\Downloads\15935001_1920_1080_24fps.mp4"),
    Path(r"C:\Users\phat.phamt\Downloads\16466882_2160_3840_50fps.mp4"),
)


def _request(url: str, api_key: str) -> dict:
    req = urllib.request.Request(
        url,
        headers={"Authorization": api_key, "User-Agent": "aive-stock-fetch/1.2"},
        method="GET",
    )
    with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _blob(video: dict) -> str:
    parts = [str(video.get("url") or ""), str(video.get("image") or "")]
    user = video.get("user") or {}
    if isinstance(user, dict):
        parts.append(str(user.get("name") or ""))
    for tag in video.get("tags") or []:
        parts.append(str(tag.get("name") if isinstance(tag, dict) else tag))
    return " ".join(parts).lower()


def _blocked(video: dict) -> bool:
    return any(word in _blob(video) for word in BLOCK)


def _looks_aerial(video: dict) -> bool:
    blob = _blob(video)
    return any(k in blob for k in ("drone", "aerial", "flyover", "fpv", "from above"))


def _best_file(video: dict) -> dict | None:
    files = [f for f in video.get("video_files") or [] if f.get("link") and f.get("width")]
    files = [
        f
        for f in files
        if max(int(f.get("width") or 0), int(f.get("height") or 0)) >= MIN_WIDTH
    ]
    if not files:
        return None
    files.sort(
        key=lambda f: (
            abs(int(f.get("width") or 0) - 1920) + abs(int(f.get("height") or 0) - 1080),
            -int(f.get("width") or 0) * int(f.get("height") or 0),
        )
    )
    return files[0]


def _download(url: str, dest: Path) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": "aive-stock-fetch/1.2"})
    with urllib.request.urlopen(req, timeout=TIMEOUT) as resp, dest.open("wb") as out:
        while True:
            chunk = resp.read(1024 * 256)
            if not chunk:
                break
            out.write(chunk)


def main() -> int:
    api_key = os.environ.get("PEXELS_API_KEY", "").strip()
    if not api_key:
        print(
            "Missing PEXELS_API_KEY.\n"
            "1) https://www.pexels.com/api/\n"
            "2) $env:PEXELS_API_KEY='your_key'\n"
            "3) Re-run this script.",
            file=sys.stderr,
        )
        return 2

    DEST.mkdir(parents=True, exist_ok=True)
    if CLEAR_EXISTING:
        for old in DEST.glob("pexels_*.mp4"):
            old.unlink(missing_ok=True)

    saved: list[Path] = []
    for ref in REF_COPIES:
        if ref.is_file():
            dest = DEST / f"ref_{ref.name}"
            if not dest.exists():
                shutil.copy2(ref, dest)
                print(f"copied reference {dest.name}")
            saved.append(dest)

    seen: set[int] = set()
    for path in DEST.glob("pexels_*.mp4"):
        try:
            seen.add(int(path.stem.split("_", 1)[1]))
        except (IndexError, ValueError):
            pass

    target = COUNT
    for query in QUERIES:
        if len(saved) >= target:
            break
        params = urllib.parse.urlencode({"query": query, "per_page": 15})
        try:
            payload = _request(f"https://api.pexels.com/videos/search?{params}", api_key)
        except urllib.error.HTTPError as exc:
            print(f"API error {query!r}: {exc}", file=sys.stderr)
            continue

        videos = sorted(
            payload.get("videos") or [],
            key=lambda v: (-int(v.get("duration") or 0), -int(v.get("width") or 0)),
        )
        for video in videos:
            if len(saved) >= target:
                break
            vid = int(video.get("id") or 0)
            if not vid or vid in seen:
                continue
            duration = int(video.get("duration") or 0)
            if duration < MIN_DURATION:
                continue
            if _blocked(video):
                continue
            chosen = _best_file(video)
            if chosen is None:
                continue
            dest = DEST / f"pexels_{vid}.mp4"
            if dest.exists() and dest.stat().st_size > 0:
                seen.add(vid)
                saved.append(dest)
                print(f"skip existing {dest.name}")
                continue
            tag = "aerial" if _looks_aerial(video) else "candidate"
            print(
                f"download {dest.name} [{query}|{tag}] "
                f"{duration}s {chosen.get('width')}x{chosen.get('height')}"
            )
            try:
                _download(str(chosen["link"]), dest)
            except Exception as exc:  # noqa: BLE001
                print(f"  failed: {exc}", file=sys.stderr)
                dest.unlink(missing_ok=True)
                continue
            seen.add(vid)
            saved.append(dest)
            print(f"  ok {dest.stat().st_size // 1024} KiB")

    print(f"Done: {len(saved)} clip(s) in {DEST}")
    return 0 if len(saved) >= 15 else 1


if __name__ == "__main__":
    raise SystemExit(main())
