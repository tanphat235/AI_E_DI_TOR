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
import subprocess
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


def compress_pauses(mp3: Path, marks: list[dict], max_pause: float,
                    pitch_ratio: float = 1.0) -> float:
    """Shorten every silence longer than ``max_pause`` to ``max_pause``.

    The service puts about a second between sentences -- measured, 51 pauses
    in two minutes at ~1.0 s each, a third of the running time silent -- which
    reads as slow and mechanical. Each long pause keeps half of ``max_pause``
    at either edge and loses its middle, so a sentence still lands on a breath.

    Cut on decoded PCM with exact sample indices, not with an ffmpeg filter:
    aselect works on whole frames (~26 ms), and over two hundred cuts that
    rounding would drift the captions by seconds. The same cut list remaps the
    timing marks in place, so audio and captions move together. Returns the
    seconds removed.
    """
    import wave

    import imageio_ffmpeg
    import numpy as np

    ff = imageio_ffmpeg.get_ffmpeg_exe()
    wav = mp3.with_suffix(".full.wav")
    decode = [ff, "-y", "-loglevel", "error", "-i", str(mp3), "-ac", "1"]
    if pitch_ratio != 1.0:
        # Deeper voice, done here rather than with the service's own pitch
        # option: that option made requests intermittently return no audio,
        # and a long script is several requests, so one always failed.
        # asetrate lowers pitch and tempo together, atempo restores the tempo,
        # so every timing mark stays where it was. The real sample rate is
        # needed -- a guessed one changes the duration.
        info = subprocess.run([ff, "-hide_banner", "-i", str(mp3)], capture_output=True,
                              text=True, encoding="utf-8", errors="replace").stderr
        sr = int(re.search(r"(\d+) Hz", info).group(1))
        decode += ["-af", f"asetrate={sr * pitch_ratio:.0f},aresample={sr},"
                          f"atempo={1 / pitch_ratio:.5f}"]
    subprocess.run(decode + [str(wav)], check=True)
    log = subprocess.run(
        [ff, "-hide_banner", "-nostats", "-i", str(wav), "-af",
         f"silencedetect=noise=-40dB:d={max_pause}", "-f", "null", "-"],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    ).stderr
    starts = [max(0.0, float(v)) for v in re.findall(r"silence_start: (-?[\d.]+)", log)]
    ends = [float(v) for v in re.findall(r"silence_end: ([\d.]+)", log)]
    half = max_pause / 2
    cuts = [(s + half, e - half) for s, e in zip(starts, ends, strict=False) if e - s > max_pause]
    if not cuts:
        subprocess.run([ff, "-y", "-loglevel", "error", "-i", str(wav), "-c:a", "libmp3lame",
                        "-b:a", "128k", str(mp3)], check=True)
        wav.unlink()
        return 0.0

    with wave.open(str(wav)) as w:
        sr, params = w.getframerate(), w.getparams()
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    keep, pos = [], 0
    for a, b in cuts:
        keep.append(pcm[pos:int(a * sr)])
        pos = int(b * sr)
    keep.append(pcm[pos:])
    out_wav = mp3.with_suffix(".cut.wav")
    with wave.open(str(out_wav), "wb") as w:
        w.setparams(params)
        w.writeframes(np.concatenate(keep).tobytes())
    subprocess.run([ff, "-y", "-loglevel", "error", "-i", str(out_wav), "-c:a", "libmp3lame",
                    "-b:a", "128k", str(mp3)], check=True)
    wav.unlink()
    out_wav.unlink()

    def remap(t: float) -> float:
        removed = 0.0
        for a, b in cuts:
            if t >= b:
                removed += b - a
            elif t > a:
                return a - removed  # inside a cut: pin to where it now is
            else:
                break
        return t - removed

    for m in marks:
        m["start"], m["end"] = remap(m["start"]), remap(m["end"])
    return sum(b - a for a, b in cuts)


async def speak(text: str, voice: str, mp3: Path, rate: str = "+0%", pitch: str = "+0Hz") -> list[dict]:
    """Write the mp3 and return every timing mark the service reported.

    Both kinds are collected. The Vietnamese voices emit **SentenceBoundary**,
    not WordBoundary -- asking only for words returns an empty stream and the
    whole script lands at 0.00s, which is how this was found. Sentences are the
    better unit here anyway, since a caption line is a sentence.
    """
    marks: list[dict] = []
    comm = edge_tts.Communicate(text, voice, rate=rate, pitch=pitch)
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
    """Give each written line a span, by character position in the mark stream.

    A mark need not line up with a written line in either direction: one line
    can hold several sentences ("Không có thuốc. Không có chợ."), and the
    service can fold two written lines into one sentence -- it does that across
    "..." -- so consuming whole marks per line let one merged mark push every
    later caption one line out of step. Instead each line's start and end are
    placed by its cumulative character offset, interpolated inside whichever
    mark covers that offset. A line that shares a mark gets a proportional
    slice of it, and an error cannot carry past the next mark boundary.
    """
    marks = [(norm(m["text"]), m["start"], m["end"]) for m in words]
    marks = [m for m in marks if m[0]]
    if not marks:
        return [{"text": line, "start": 0.0, "end": 0.0, "matched": False} for line in lines]

    mark_total = sum(len(m[0]) for m in marks)
    line_total = sum(len(norm(line)) for line in lines) or 1
    # The two sides rarely normalise to the exact same length (a mark may drop
    # a quote or read a symbol); scale one onto the other rather than let the
    # tail run past the last mark.
    scale = mark_total / line_total

    def time_at(pos: float) -> float:
        acc = 0
        for text, start, end in marks:
            if pos <= acc + len(text):
                frac = (pos - acc) / len(text)
                return start + max(0.0, min(1.0, frac)) * (end - start)
            acc += len(text)
        return marks[-1][2]

    out: list[dict] = []
    pos = 0
    for line in lines:
        n = len(norm(line))
        start, end = time_at(pos * scale), time_at((pos + n) * scale)
        out.append({"text": line, "start": round(start, 3), "end": round(end, 3),
                    "matched": abs(scale - 1.0) < 0.05})
        pos += n

    # Offset 0 of a mark is also offset len() of the one before, so a line that
    # begins exactly on a boundary would start at the previous mark's end --
    # inside the pause. Snap it to the start of the mark it actually begins.
    starts = []
    acc = 0
    for text, s, _ in marks:
        starts.append((acc, s))
        acc += len(text)
    pos = 0
    for item, line in zip(out, lines, strict=True):
        for off, s in starts:
            if abs(off - pos * scale) < 0.5:
                item["start"] = round(s, 3)
                break
        pos += len(norm(line))
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

    # The service intermittently answers with no audio at all (NoAudioReceived)
    # for a request that succeeds unchanged a moment later -- measured on a
    # one-sentence probe -- so retry rather than fail a twelve-minute script.
    for attempt in range(1, 4):
        try:
            words = await speak(" ".join(lines), doc["voice"], mp3,
                                rate=doc.get("rate", "+0%"), pitch=doc.get("pitch", "+0Hz"))
            break
        except edge_tts.exceptions.NoAudioReceived:
            if attempt == 3:
                raise
            print(f"no audio from the service (attempt {attempt}); retrying")
            await asyncio.sleep(10 * attempt)
    if doc.get("max_pause") or doc.get("pitch_ratio"):
        # A max_pause longer than any real pause makes this a pitch pass only.
        removed = compress_pauses(mp3, words, float(doc.get("max_pause") or 99),
                                  pitch_ratio=float(doc.get("pitch_ratio", 1.0)))
        print(f"pitch x{doc.get('pitch_ratio', 1.0)}, pauses capped at "
              f"{doc.get('max_pause')}s: {removed:.1f}s of silence removed")
    timed = split_lines(lines, words)
    (out_dir / f"{doc['id']}.json").write_text(
        json.dumps({"lines": timed, "words": words}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    kinds = ", ".join(sorted({w["kind"] for w in words})) or "none"
    print(f"voice {doc['voice']}   {len(words)} timing marks ({kinds})")
    for t in timed:
        print(f"  {t['start']:7.2f}-{t['end']:7.2f}  {t['text'][:60]}")
    print(f"\naudio {mp3}  {mp3.stat().st_size / 1024:.0f} KB")
    print(f"narration ends at {timed[-1]['end']:.2f}s")
    if timed and not timed[0].get("matched"):
        print("WARNING: written text and timing marks differ by more than 5% in length; "
              "caption timing is scaled and may drift -- check the printed spans")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
