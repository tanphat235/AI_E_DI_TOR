"""Snap hand-given cut times onto real word boundaries.

A time written down by a person ("from 0:18 to 2:45") is a rough place, not a
word edge. Cut exactly there and the clip opens on half a word or ends on the
start of the next one. The standing rule is: if a word is clipped, widen the
cut (up to --max-move seconds); if a stray word is left over, trim it.

How it decides, per edge:

1. Transcribe only a window around the edge with word timestamps (the
   whole-talk transcript is segment-level and 2 s coarse -- not enough).
2. The gaps between words are the only legal cut places. Take the gaps within
   --max-move of the requested time and score each one: a longer pause is
   better (a sentence break beats a breath), a gap closer to the request is
   better (a pause counts for at most 0.4 s, distance costs 1 s per second). The best
   gap wins.
3. Inside the winning gap the cut goes where the audio is quietest. In a real
   pause it sits with a little room: --pre before the next word for a start,
   --post after the last word for an end. In fast speech there is no pause --
   Whisper's word end and next word start touch -- so the cut goes at the
   quietest frame within +-0.08 s of that boundary.

   (Refining each word's own start and end against the audio was tried and
   dropped: in continuous speech the energy never dips between words, so the
   "refined" ends ran 0.3-0.5 s into the next word.)

It reports; the words either side of each edge are printed so the choice can
be read. A pause is not proof the sentence is complete -- read them, and pass
an exact time back in if a sentence needs finishing.

Output: JSON on stdout (one object per span), human text on stderr.

    python scripts/shorts/find_edges.py --work-dir projects/<job> \\
        --span 1115 1262 --span 1286 1454
"""

from __future__ import annotations

import argparse
import json
import sys
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np

FRAME = 0.01  # seconds per energy frame


@dataclass
class Word:
    text: str
    start: float
    end: float


def _read_window(wav_path: Path, t0: float, t1: float) -> tuple[np.ndarray, int]:
    with wave.open(str(wav_path), "rb") as w:
        if w.getsampwidth() != 2 or w.getnchannels() != 1:
            raise SystemExit(f"{wav_path}: expected 16-bit mono PCM")
        rate = w.getframerate()
        a = max(0, int(t0 * rate))
        b = min(w.getnframes(), int(t1 * rate))
        w.setpos(a)
        raw = w.readframes(b - a)
    return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0, rate


def _rms_db(x: np.ndarray, rate: int) -> np.ndarray:
    n = max(1, int(FRAME * rate))
    frames = len(x) // n
    if frames == 0:
        return np.full(1, -90.0)
    v = x[: frames * n].reshape(frames, n)
    return 20 * np.log10(np.sqrt((v**2).mean(axis=1)) + 1e-7)


class Finder:
    def __init__(self, wav_path: Path, model_name: str, language: str) -> None:
        from faster_whisper import WhisperModel

        self.wav = wav_path
        print(f"loading whisper {model_name}...", file=sys.stderr)
        self.model = WhisperModel(model_name, device="cpu", compute_type="float32")
        self.language = language

    def window(self, centre: float, half: float) -> tuple[list[Word], np.ndarray, float]:
        """Words (absolute time), energy in dB per FRAME, and the window start."""
        t0 = max(0.0, centre - half)
        x, rate = _read_window(self.wav, t0, centre + half)
        segs, _ = self.model.transcribe(
            x,
            language=self.language,
            word_timestamps=True,
            vad_filter=False,
            beam_size=5,
            condition_on_previous_text=False,
        )
        words: list[Word] = []
        for s in segs:
            for w in s.words or []:
                words.append(Word(w.word.strip(), t0 + w.start, t0 + w.end))
        return words, _rms_db(x, rate), t0


def energy_pauses(db: np.ndarray, t0: float, min_len: float = 0.2) -> list[tuple[float, float]]:
    """Real silences, from the audio itself: runs of frames well under the
    window's speech level. Steadier than Whisper's word times, which move by
    0.3-0.5 s between two runs over slightly different windows."""
    thr = float(np.percentile(db, 95)) - 15.0
    out: list[tuple[float, float]] = []
    run = None
    for i, quiet in enumerate(db < thr):
        if quiet and run is None:
            run = i
        if (not quiet or i == len(db) - 1) and run is not None:
            end = i if not quiet else i + 1
            if (end - run) * FRAME >= min_len:
                out.append((t0 + run * FRAME, t0 + end * FRAME))
            run = None
    return out


def snap_edge(
    words: list[Word],
    db: np.ndarray,
    t0: float,
    t: float,
    *,
    kind: str,
    max_move: float,
    pre: float,
    post: float,
) -> tuple[float, dict]:
    """Best cut time near `t`. `kind` is "start" or "end"."""
    best = None
    # (a, b, prev word, next word); a real silence first, then word boundaries.
    cands: list[tuple[float, float, Word | None, Word | None]] = []
    for a, b in energy_pauses(db, t0):
        # A word that is still sounding at the dip (a soft word-final syllable
        # drops under the threshold before the word ends) is not in the pause:
        # push the pause's edges out to that word's own end / start.
        prev = next((w for w in reversed(words) if w.start <= a), None)
        nxt = next((w for w in words if w.end >= b), None)
        if prev is not None:
            a = max(a, prev.end)
        if nxt is not None:
            b = min(b, nxt.start)
        if b <= a:
            continue
        cands.append((a, b, prev, nxt))
    for prev, nxt in zip([None, *words], [*words, None]):
        if prev is not None and nxt is not None:
            cands.append((prev.end, nxt.start, prev, nxt))
    for a, b, prev, nxt in cands:
        length = max(0.0, b - a)
        lo, hi = (a, b) if b >= a else (b, a)
        dist = abs(min(max(t, lo), hi) - t)
        if dist > max_move:
            continue
        # A boundary with no silence in it is where Whisper THINKS two words
        # meet, and Whisper sometimes drops a word or shifts it (it lost "não"
        # from "phiền não" and the cut landed inside it). Prefer a real
        # pause nearby; keep a tight boundary as the fallback.
        score = min(length, 0.4) - 1.0 * dist - (0.4 if length < 0.2 else 0.0)
        if best is None or score > best[0]:
            best = (score, a, b, prev, nxt, length)
    if best is None:
        return t, {"note": "no pause or word boundary within reach; left as given"}
    _, ga, gb, prev, nxt, length = best

    if length >= 0.2:
        cut = gb - min(pre, length / 2) if kind == "start" else ga + min(post, length / 2)
    else:
        mid = (ga + gb) / 2
        i0 = max(0, int(round((mid - 0.08 - t0) / FRAME)))
        i1 = min(len(db) - 1, int(round((mid + 0.08 - t0) / FRAME)))
        k = i0 + int(np.argmin(db[i0 : i1 + 1]))
        cut = t0 + (k + 0.5) * FRAME
    # Last line of defence. Whisper's word times can be wrong or missing, which
    # put a start inside the tail of the previous sentence ("...giới vậy | Cho
    # nên": the cut fell in "vậy" and left a stray fragment). The audio is the
    # judge: a cut must sit in a quiet frame. If it does not, move it into the
    # nearest real silence within reach.
    quiet_thr = float(np.percentile(db, 95)) - 15.0

    def _loud(x: float) -> bool:
        i0 = max(0, int(round((x - 0.04 - t0) / FRAME)))
        i1 = min(len(db) - 1, int(round((x + 0.04 - t0) / FRAME)))
        return bool(db[i0 : i1 + 1].mean() > quiet_thr)

    fixed_note = None
    if _loud(cut):
        near = [
            (a, b) for a, b in energy_pauses(db, t0)
            if min(abs(a - t), abs(b - t)) <= max_move + 0.5 or a <= t <= b
        ]
        if near:
            a, b = min(near, key=lambda ab: 0.0 if ab[0] <= t <= ab[1] else min(abs(ab[0] - t), abs(ab[1] - t)))
            room = min(pre if kind == "start" else post, (b - a) / 2)
            new = (b - room) if kind == "start" else (a + room)
            fixed_note = f"cut {cut:.2f} was inside speech; moved into the silence {a:.2f}-{b:.2f}"
            cut = new
    info = {
        "pause": round(length, 2),
        "moved": round(cut - t, 2),
        "before": prev.text if prev else None,
        "after": nxt.text if nxt else None,
    }
    if fixed_note:
        info["note"] = fixed_note
    return cut, info


def _context(words: list[Word], t: float, n: int = 6) -> tuple[str, str]:
    before = [w.text for w in words if w.end <= t][-n:]
    after = [w.text for w in words if w.start >= t][:n]
    return " ".join(before), " ".join(after)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--work-dir", type=Path, required=True, help="Job folder with .aive/audio.wav")
    p.add_argument("--span", type=float, nargs=2, action="append", metavar=("START", "END"),
                   required=True, help="Requested span in source seconds; repeatable.")
    p.add_argument("--model", default="small")
    p.add_argument("--language", default="vi")
    p.add_argument("--half-window", type=float, default=5.0, help="Seconds of audio each side.")
    p.add_argument("--max-move", type=float, default=1.0,
                   help="Furthest an edge may move from the request (seconds).")
    p.add_argument("--pre", type=float, default=0.15, help="Room before the first word.")
    p.add_argument("--post", type=float, default=0.25, help="Room after the last word.")
    p.add_argument("--verify", action="store_true",
                   help="Re-transcribe the first and last 3 s of each snapped span.")
    args = p.parse_args()

    wav = args.work_dir / ".aive" / "audio.wav"
    if not wav.is_file():
        raise SystemExit(f"missing {wav}")
    finder = Finder(wav, args.model, args.language)

    results = []
    for req_s, req_e in args.span:
        row: dict = {"requested": [req_s, req_e]}
        for kind, t in (("start", req_s), ("end", req_e)):
            words, db, t0 = finder.window(t, args.half_window)
            cut, info = snap_edge(
                words, db, t0, t, kind=kind, max_move=args.max_move, pre=args.pre, post=args.post
            )
            before, after = _context(words, cut)
            info["cut"] = round(cut, 2)
            info["context_before"] = before
            info["context_after"] = after
            row[kind] = info
            print(
                f"{kind:5s} asked {t:8.2f} -> {cut:8.2f} (moved {cut - t:+.2f}, "
                f"pause {info.get('pause', '?')})\n"
                f"        ...{before} | {after}...",
                file=sys.stderr,
            )
        row["span"] = [row["start"]["cut"], row["end"]["cut"]]
        results.append(row)

    if args.verify:
        print("\nverify (fresh transcription of the exact cut, 3 s each side):", file=sys.stderr)
        for row in results:
            a, b = row["span"]
            for label, lo, hi in (("opens ", a, a + 3.0), ("closes", b - 3.0, b)):
                x, _ = _read_window(wav, lo, hi)
                segs, _ = finder.model.transcribe(
                    x, language=args.language, word_timestamps=True, vad_filter=False,
                    beam_size=5, condition_on_previous_text=False,
                )
                ws = [w.word.strip() for sg in segs for w in (sg.words or [])]
                ws = ws[:6] if label == "opens " else ws[-6:]
                print(f"  {a if label == 'opens ' else b:9.2f} {label}: {' '.join(ws)}", file=sys.stderr)

    sys.stdout.write(json.dumps(results, ensure_ascii=False, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
