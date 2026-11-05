"""Narrate a script with VieNeu-TTS, one sentence at a time.

Microsoft edge-tts is not used here. Each written line is synthesized on its
own, so the caption times are the synthesis times.

Two modes, chosen by the script:

* **Plain** (no ``pauses`` key): lines are joined by a fixed 0.2 s gap and each
  line is one caption. This is what the earlier videos use.
* **Paced** (script has ``pauses``): the silence after a line depends on what
  follows it -- a breath between sentences, a longer rest before a new picture,
  a long one around a chapter card -- and every line is cut into short
  *phrases*, each with its own start and end, so a caption can change as the
  voice moves on instead of printing the whole sentence and waiting.

Phrase timing: the sentence is synthesised whole (so its intonation stays
natural), the phrase borders are placed by character share and then snapped to
the quietest moment within +-0.35 s, which is where the voice actually
breathes at a comma. A short rest is inserted there only if the voice really
does pause at that point, never in the middle of a word.

Writes:
  .aive/audio/<id>.mp3
  .aive/audio/<id>.json     per-line and (paced mode) per-phrase start/end
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
os.environ.setdefault("HF_HOME", str(REPO / ".hf-cache"))

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Plain mode: short enough that the edit does not sound like the old neural
# voice, long enough that two sentences do not collide.
GAP_S = 0.20
SR = 48000
FFMPEG = (
    REPO / ".venv-sdxl" / "Lib" / "site-packages" / "imageio_ffmpeg"
    / "binaries" / "ffmpeg-win-x86_64-v7.1.exe"
)

# Paced mode defaults; every one can be overridden by the script's "pauses".
DEFAULT_PAUSES = {
    "phrase": 0.20,       # inside a sentence, where the voice already rests
    "sentence": 0.55,     # between sentences of one picture
    "paragraph": 1.10,    # before a line that opens a new picture
    "card_before": 1.60,  # before a chapter card
    "card_after": 1.50,   # after the card's own announcement
    "section": 2.40,      # before the first line of a new part (a ``section`` line)
    "inner_max": 0.30,    # longest silence kept inside a line; 0 = keep them all
    "lead_in": 0.80,      # silence before the first word
}
# A caption phrase longer than this is split at a space near its middle.
MAX_PHRASE_CHARS = 60
# Level (dB under the line's peak) below which a frame counts as silence.
SILENCE_DB = -40.0
# Function words that bind to what follows, so a caption cannot end on one.
_BINDS_FORWARD = frozenset(
    ["những", "các", "của", "và", "với", "một", "giữa", "trong", "để", "cho", "là",
     "mà", "thì", "ở", "từ", "về", "theo", "như", "tứ", "ân", "ơn", "đất", "tam",
     "phật", "hòa", "huỳnh", "phú", "đức", "ủy", "việt", "nam", "dân", "chủ", "xã"]
)
# Silence kept on each side when a line's own edge padding is trimmed.
EDGE_KEEP_S = 0.06


def work_dir(script_path: str) -> Path:
    """The project that holds the script. Artefacts are written beside it."""
    return Path(script_path).resolve().parents[1]


def _envelope_db(audio, hop: int):
    """Per-hop RMS level in dB, relative to the loudest hop."""
    import numpy as np

    n = len(audio) // hop
    if n == 0:
        return np.zeros(0)
    frames = audio[: n * hop].reshape(n, hop)
    rms = np.sqrt((frames.astype(np.float64) ** 2).mean(axis=1) + 1e-12)
    return 20 * np.log10(rms / rms.max())


def trim_edges(audio, sr: int):
    """Drop the synthesiser's leading and trailing silence, keep a short margin.

    Without this every line brings ~0.1-0.3 s of its own padding, so the real
    pause between two lines is whatever the script asked for *plus* a random
    amount, and the rests the script specifies cannot be told apart.
    """
    import numpy as np

    hop = int(sr * 0.01)
    env = _envelope_db(audio, hop)
    voiced = np.where(env > SILENCE_DB)[0]
    if len(voiced) == 0:
        return audio
    keep = int(EDGE_KEEP_S * sr)
    a = max(0, voiced[0] * hop - keep)
    b = min(len(audio), (voiced[-1] + 1) * hop + keep)
    return audio[a:b]


def split_phrases(text: str, max_chars: int = MAX_PHRASE_CHARS) -> list[str]:
    """Cut a line into caption phrases at sentence ends and commas.

    Pieces are merged back greedily up to ``max_chars`` so a list such as
    "ơn cha mẹ, ơn chúng sinh, ..." does not become five one-second captions,
    and a piece that is still too long is split at the space nearest its middle.
    """
    text = " ".join(text.split())
    pieces = [p for p in re.findall(r"[^,;:.?!]+[,;:.?!]*", text) if p.strip()]
    merged: list[str] = []
    for piece in (p.strip() for p in pieces):
        if merged and len(merged[-1]) + 1 + len(piece) <= max_chars:
            merged[-1] = f"{merged[-1]} {piece}"
        else:
            merged.append(piece)
    out: list[str] = []
    for piece in merged:
        while len(piece) > max_chars:
            cut = piece.rfind(" ", 0, max_chars)
            mid = piece.find(" ", len(piece) // 2 - 8)
            at = mid if 0 < mid < max_chars + 10 else cut
            if at <= 0:
                break
            left, right = piece[:at].strip(), piece[at:].strip()
            # A caption must not end on a word that belongs to the next one.
            words = left.split(" ")
            while len(words) > 3 and words[-1].lower() in _BINDS_FORWARD:
                right = f"{words.pop()} {right}"
            out.append(" ".join(words))
            piece = right
        out.append(piece)
    return out


_ONES = ["không", "một", "hai", "ba", "bốn", "năm", "sáu", "bảy", "tám", "chín"]


def number_syllables(n: int) -> int:
    """How many syllables Vietnamese needs to say ``n`` (0..9999).

    Only the count matters -- it is the weight of a number when a line's time is
    shared out between phrases. "1945" is four characters but seven syllables
    ("một nghìn chín trăm bốn mươi lăm"), and weighting it as four put the
    caption borders of every line with a year a second or more off the voice.
    """
    if n < 10:
        return 1
    if n < 20:
        return 1 if n == 10 else 2
    if n < 100:
        return 2 if n % 10 == 0 else 3
    if n < 1000:
        rest = n % 100
        return 2 + (0 if rest == 0 else (2 if rest < 10 else number_syllables(rest)))
    rest = n % 1000
    tail = 0 if rest == 0 else (2 + number_syllables(rest) if rest < 100 else number_syllables(rest))
    return 2 + tail


def spoken_syllables(text: str) -> int:
    """Syllables a line takes to say: one per word, numbers spelled out."""
    count = 0
    for token in re.findall(r"\d+|[^\W\d_]+", text):
        if token.isdigit():
            count += number_syllables(int(token)) if len(token) <= 4 else len(token)
        else:
            count += 1
    return max(1, count)


def tighten(audio, sr: int, inner_max: float):
    """Cap every silence inside a line at ``inner_max`` seconds.

    The voice rests 0.4-0.7 s at some commas and around some numbers, which
    reads as the narrator stalling. The middle of any silence longer than the
    cap is cut out; silence is below -40 dB of the line's peak, so the cut is
    inaudible. Returns the audio and the seconds removed.
    """
    import numpy as np

    hop = int(sr * 0.01)
    env = _envelope_db(audio, hop)
    voiced = np.where(env > SILENCE_DB)[0]
    if len(voiced) == 0 or inner_max <= 0:
        return audio, 0.0
    first, last = int(voiced[0]), int(voiced[-1])
    cap = max(1, int(round(inner_max / 0.01)))
    keep = np.ones(len(env), dtype=bool)
    k = first
    while k <= last:
        if env[k] <= SILENCE_DB:
            j = k
            while j <= last and env[j] <= SILENCE_DB:
                j += 1
            if j - k > cap:
                keep[k + cap // 2: j - (cap - cap // 2)] = False
            k = j
        else:
            k += 1
    mask = np.ones(len(audio), dtype=bool)
    mask[: len(env) * hop] = np.repeat(keep, hop)
    removed = (len(audio) - int(mask.sum())) / sr
    return audio[mask], removed


def pause_centres(env, hop_s: float, min_len: float = 0.08, floor_db: float = -35.0) -> list[int]:
    """Frame index at the middle of every quiet stretch inside the speech."""
    import numpy as np

    voiced = np.where(env > floor_db)[0]
    if len(voiced) == 0:
        return []
    first, last = int(voiced[0]), int(voiced[-1])
    need = max(1, int(round(min_len / hop_s)))
    out: list[int] = []
    k = first
    while k <= last:
        if env[k] <= floor_db:
            j = k
            while j <= last and env[j] <= floor_db:
                j += 1
            if j - k >= need:
                out.append((k + j) // 2)
            k = j
        else:
            k += 1
    return out


def phrase_borders(audio, sr: int, phrases: list[str]) -> list[int]:
    """Sample index of each border between caption phrases.

    The voice breathes at commas, so a border is put at a real pause in the
    audio when one is near where the syllable count says the phrase should
    end; otherwise at the quietest frame near that estimate. After each border
    the rest of the line is re-estimated from where the voice actually is, so
    an error early in a long line does not carry to its end.
    """
    import numpy as np

    if len(phrases) < 2:
        return []
    weights = [spoken_syllables(p) for p in phrases]
    hop = int(sr * 0.01)
    hop_s = hop / sr
    env = _envelope_db(audio, hop)
    n = len(env)
    pauses = pause_centres(env, hop_s)
    voiced = np.where(env > SILENCE_DB)[0]
    end_frame = int(voiced[-1]) if len(voiced) else n
    window = int(0.35 / hop_s)
    min_gap = int(0.25 / hop_s)
    reach = int(0.6 / hop_s)
    borders: list[int] = []
    pos = int(voiced[0]) if len(voiced) else 0
    for idx in range(len(weights) - 1):
        remaining = sum(weights[idx:])
        guess = pos + int((end_frame - pos) * weights[idx] / remaining)
        lo = pos + min_gap
        hi = end_frame - min_gap * (len(weights) - 1 - idx)
        usable = [p for p in pauses if lo <= p <= hi and abs(p - guess) <= reach]
        if usable:
            best = min(usable, key=lambda p: abs(p - guess))
        else:
            a, b = max(lo, guess - window), min(max(lo + 1, hi), guess + window + 1)
            if b <= a:
                best = min(max(guess, lo), n - 1)
            else:
                seg = env[a:b]
                best = a + int(np.argmin(seg + 0.002 * np.abs(np.arange(a, b) - guess)))
        borders.append(best * hop)
        pos = best
    return borders


def gap_after(lines: list[dict], i: int, pauses: dict) -> float:
    """Seconds of silence after line ``i``, from what comes next."""
    line = lines[i]
    if line.get("pause") is not None:
        return float(line["pause"])
    if line.get("card"):
        return pauses["card_after"]
    nxt = lines[i + 1]
    if nxt.get("section"):
        # A new part of the video: a clear rest, not a spoken heading.
        return pauses["section"]
    if nxt.get("card"):
        return pauses["card_before"]
    if nxt.get("image") or nxt.get("file"):
        return pauses["paragraph"]
    return pauses["sentence"]


def legacy_index(audio_dir: Path) -> dict[str, Path]:
    """Lines synthesised by earlier runs, found by their text.

    Earlier runs named a line's file by its number, so inserting or removing a
    line made every later one look stale. Those files are still good audio.
    """
    found: dict[str, Path] = {}
    for note in audio_dir.glob("*-vieneu/[0-9][0-9][0-9].json"):
        wav = note.with_suffix(".wav")
        if wav.is_file():
            try:
                found.setdefault(json.loads(note.read_text(encoding="utf-8"))["text"], wav)
            except (KeyError, ValueError):
                continue
    return found


def synth_line(tts, voice: str, text: str, cache_dir: Path, legacy: dict[str, Path]):
    """One line's audio, cached on its voice and text (not on its position)."""
    import hashlib

    import numpy as np
    import soundfile as sf

    key = hashlib.sha1(f"{voice}\0{text}".encode()).hexdigest()[:12]
    part = cache_dir / f"{key}.wav"
    note = cache_dir / f"{key}.json"
    if part.is_file():
        audio, sr = sf.read(part, dtype="float32")
        if sr != SR:
            raise SystemExit(f"{part.name} is {sr} Hz, expected {SR}")
        return audio
    if text in legacy:
        audio, sr = sf.read(legacy[text], dtype="float32")
        if sr == SR:
            return audio
    raw = np.asarray(tts.infer(text, voice=voice))
    audio = raw.astype(np.float32, copy=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 2.0:
        audio = audio / 32768.0
    sf.write(part, audio, SR)
    note.write_text(json.dumps({"text": text}, ensure_ascii=False), encoding="utf-8")
    return audio


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: narrate_vieneu.py <script.json>")
        return 2
    doc = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8-sig"))
    doc_lines = doc["lines"]
    lines = [x["text"] for x in doc_lines]
    voice = doc["voice"]
    paced = "pauses" in doc
    pauses = {**DEFAULT_PAUSES, **(doc.get("pauses") or {})}

    import numpy as np
    import soundfile as sf
    from vieneu import Vieneu

    # ONNX stays on the CPU. The GPU is reserved for the image pass, and this
    # environment has no torch.
    tts = Vieneu(backend="onnx")
    presets = tts.list_preset_voices()
    chosen = next((label for label, voice_id in presets if voice_id == voice), None)
    if chosen is None:
        print(f"voice {voice!r} is not a VieNeu preset")
        for label, voice_id in presets:
            print(f"  {voice_id}  —  {label}")
        return 2
    mode = "paced" if paced else "plain"
    print(f"VieNeu ONNX  {chosen}  {len(lines)} lines  ({mode})", flush=True)

    out_dir = work_dir(sys.argv[1]) / ".aive" / "audio"
    part_dir = out_dir / "_lines-vieneu"   # shared by every script, keyed by text
    out_dir.mkdir(parents=True, exist_ok=True)
    part_dir.mkdir(parents=True, exist_ok=True)
    legacy = legacy_index(out_dir)
    trimmed_total = 0.0

    pieces: list[np.ndarray] = []
    timed: list[dict] = []
    phrases_out: list[dict] = []
    cursor = 0.0
    if paced and pauses["lead_in"] > 0:
        pieces.append(np.zeros(int(SR * pauses["lead_in"]), dtype=np.float32))
        cursor += pauses["lead_in"]

    for i, text in enumerate(lines):
        audio = synth_line(tts, voice, text, part_dir, legacy)
        if not paced:
            duration = len(audio) / SR
            timed.append({"text": text, "start": round(cursor, 3),
                          "end": round(cursor + duration, 3), "matched": True})
            pieces.append(audio)
            cursor += duration
            if i < len(lines) - 1:
                pieces.append(np.zeros(int(SR * GAP_S), dtype=np.float32))
                cursor += GAP_S
            print(f"  {i + 1:3d}/{len(lines)}  {duration:5.2f}s  {text[:56]}", flush=True)
            continue

        audio = trim_edges(audio, SR)
        audio, removed = tighten(audio, SR, float(pauses["inner_max"]))
        trimmed_total += removed
        line_start = cursor
        is_card = bool(doc_lines[i].get("card"))
        phrases = [text] if is_card else split_phrases(text)
        borders = phrase_borders(audio, SR, phrases)
        env = _envelope_db(audio, int(SR * 0.01))

        prev = 0
        for k, phrase in enumerate(phrases):
            end = borders[k] if k < len(borders) else len(audio)
            chunk = audio[prev:end]
            phrases_out.append({
                "text": phrase, "line": i, "card": is_card,
                "start": round(cursor, 3),
                "end": round(cursor + len(chunk) / SR, 3),
            })
            pieces.append(chunk)
            cursor += len(chunk) / SR
            if k < len(borders):
                # Rest here only if the voice already rests here.
                at = min(len(env) - 1, max(0, borders[k] // int(SR * 0.01)))
                if env[at] < SILENCE_DB + 8 and pauses["phrase"] > 0:
                    pieces.append(np.zeros(int(SR * pauses["phrase"]), dtype=np.float32))
                    cursor += pauses["phrase"]
            prev = end
        # Each caption holds until the next one starts, within a line.
        mine = [p for p in phrases_out if p["line"] == i]
        for a, b in zip(mine, mine[1:], strict=False):
            a["end"] = b["start"]

        timed.append({"text": text, "start": round(line_start, 3),
                      "end": round(cursor, 3), "matched": True, "card": is_card})
        if i < len(lines) - 1:
            gap = gap_after(doc_lines, i, pauses)
            pieces.append(np.zeros(int(SR * gap), dtype=np.float32))
            cursor += gap
        print(f"  {i + 1:3d}/{len(lines)}  {timed[-1]['end'] - line_start:5.2f}s  "
              f"{len(phrases)}ph  {text[:50]}", flush=True)

    wav = out_dir / f"{doc['id']}.wav"
    mp3 = out_dir / f"{doc['id']}.mp3"
    speed = float(doc.get("speed") or 1.0)
    if speed <= 0:
        raise SystemExit(f"speed must be positive, got {speed}")
    if abs(speed - 1.0) > 1e-6:
        for row in timed:
            row["start"] = round(row["start"] / speed, 3)
            row["end"] = round(row["end"] / speed, 3)
        for row in phrases_out:
            row["start"] = round(row["start"] / speed, 3)
            row["end"] = round(row["end"] / speed, 3)
        print(f"playback speed {speed:g}x", flush=True)
        raw = np.concatenate(pieces)
        tmp = out_dir / f"{doc['id']}_raw.wav"
        sf.write(tmp, raw, SR)
        subprocess.run(
            [str(FFMPEG), "-y", "-loglevel", "error", "-i", str(tmp),
             "-filter:a", f"atempo={speed:.4f}", str(wav)],
            check=True,
        )
        tmp.unlink(missing_ok=True)
    else:
        sf.write(wav, np.concatenate(pieces), SR)
    if not FFMPEG.is_file():
        raise SystemExit(f"ffmpeg not found at {FFMPEG}")
    subprocess.run(
        [str(FFMPEG), "-y", "-loglevel", "error", "-i", str(wav),
         "-codec:a", "libmp3lame", "-q:a", "2", str(mp3)],
        check=True,
    )
    result = {"lines": timed, "voice": voice, "engine": "vieneu"}
    if paced:
        result["phrases"] = phrases_out
        result["pauses"] = pauses
    (out_dir / f"{doc['id']}.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if paced:
        print(f"\nlong silences inside lines shortened: {trimmed_total:.1f}s removed in total")
    print(f"audio {mp3}  {mp3.stat().st_size / 1024:.0f} KB")
    print(f"narration ends at {timed[-1]['end']:.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
