"""Synthesise the narration and derive caption timings from the script itself.

This is the step that makes the captions correct. Every caption problem in the
other projects came from running speech recognition over someone else's audio
and getting back non-words. Here the text exists before the audio: edge-tts
reports a WordBoundary for each word it speaks, so a line's start and end are
measured, not guessed, and the caption is the exact text that was written.

One request for the whole script, not one per line: per-line requests get rate
limited. The lines are then recovered by walking the word-boundary stream --
and the boundaries are **not** one-to-one with the written words, so they are
matched by consuming normalised characters rather than by counting tokens.

Writes:
  .aive/audio/<id>.mp3      the narration
  .aive/audio/<id>.json     per-line start/end in seconds, plus the words
"""

from __future__ import annotations

import asyncio
import json
import re
import sys
import unicodedata
from pathlib import Path

import edge_tts

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

TICKS = 10_000_000  # edge-tts reports offsets in 100-nanosecond units


def work_dir(script_path: str) -> Path:
    """Where this video's artefacts live: the project holding the script.

    The tool sits in `scripts/`, which is source and is tracked; everything it
    produces belongs beside its own script under `projects/<name>/`, which is
    project data and is not. So `projects/x/scripts/y.json` writes to
    `projects/x/`.
    """
    return Path(script_path).resolve().parents[1]


def norm(text: str) -> str:
    """Letters and digits only, lowercased -- what survives on both sides."""
    text = unicodedata.normalize("NFC", text).lower()
    return re.sub(r"[^0-9a-zà-ỹ]+", "", text)


async def speak(text: str, voice: str, mp3: Path) -> list[dict]:
    """Write the mp3 and return every timing mark the service reported.

    Both kinds are collected. The Vietnamese voices emit **SentenceBoundary**,
    not WordBoundary -- asking only for words returns an empty stream and the
    whole script lands at 0.00s, which is how this was found. Sentences are the
    better unit here anyway, since a caption line is a sentence.
    """
    marks: list[dict] = []
    comm = edge_tts.Communicate(text, voice)
    with mp3.open("wb") as fh:
        async for chunk in comm.stream():
            kind = chunk.get("type")
            if kind == "audio":
                fh.write(chunk["data"])
            elif kind in ("WordBoundary", "SentenceBoundary"):
                marks.append(
                    {
                        "kind": kind,
                        "text": chunk["text"],
                        "start": chunk["offset"] / TICKS,
                        "end": (chunk["offset"] + chunk["duration"]) / TICKS,
                    }
                )
    return marks


def split_lines(lines: list[str], words: list[dict]) -> list[dict]:
    """Give each written line the span of the marks that spoke it.

    Matched on normalised characters, not by counting marks: one written line
    can hold several sentences ("Không có thuốc. Không có chợ.") and a number
    can be read as several words, so a count would drift. Consuming characters
    until the line is covered does not.
    """
    out: list[dict] = []
    w = 0
    for line in lines:
        want = norm(line)
        got = ""
        first = w
        while w < len(words) and len(got) < len(want):
            got += norm(words[w]["text"])
            w += 1
        if w == first:  # nothing matched; keep the timeline moving
            prev_end = out[-1]["end"] if out else 0.0
            out.append({"text": line, "start": prev_end, "end": prev_end, "words": 0})
            continue
        out.append(
            {
                "text": line,
                "start": round(words[first]["start"], 3),
                "end": round(words[w - 1]["end"], 3),
                "words": w - first,
                "matched": len(got) >= len(want),
            }
        )
    return out


async def main() -> int:
    if len(sys.argv) < 2:
        print("usage: narrate.py <script.json>")
        return 2
    doc = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    lines = [x["text"] for x in doc["lines"]]
    out_dir = work_dir(sys.argv[1]) / ".aive" / "audio"
    out_dir.mkdir(parents=True, exist_ok=True)
    mp3 = out_dir / f"{doc['id']}.mp3"

    words = await speak(" ".join(lines), doc["voice"], mp3)
    timed = split_lines(lines, words)
    (out_dir / f"{doc['id']}.json").write_text(
        json.dumps({"lines": timed, "words": words}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    kinds = ", ".join(sorted({w["kind"] for w in words})) or "none"
    print(f"voice {doc['voice']}   {len(words)} timing marks ({kinds})")
    unmatched = [t for t in timed if not t.get("matched")]
    for t in timed:
        flag = "" if t.get("matched") else "   <- ran out of marks"
        print(f"  {t['start']:6.2f}-{t['end']:6.2f}  ({t['words']:2d}m) {t['text'][:52]}{flag}")
    print(f"\naudio {mp3}  {mp3.stat().st_size / 1024:.0f} KB")
    print(f"narration ends at {timed[-1]['end']:.2f}s")
    if unmatched:
        print(f"WARNING: {len(unmatched)} line(s) did not match the timing stream")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
