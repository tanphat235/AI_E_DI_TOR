"""Locate a channel's transition sting wherever it was inserted into a talk.

Re-uploads often carry another channel's transition effect, dropped in at each
question boundary. The insert is loud -- around -22 dBFS on the giesu-ducphat
talk, as loud as speech -- so ``silencedetect`` does not report it as a pause,
and a segmenter that trusts silence alone will happily cut straight into it.

Because the same audio is pasted each time, normalised cross-correlation of the
raw waveform identifies it exactly: real copies scored 0.97-1.00 on that talk
while the whole-file median was 0.00 and the best non-copy was 0.45. Envelope
correlation cannot separate the two -- loud speech scores just as well -- so it
is not used here.

Writes ``.aive/stings.json`` with, per occurrence, the span to drop plus the cut
points either side. ``segment_qa.py`` reads that file and treats every sting as a
mandatory boundary, which is also editorially right: the insert marks where the
next question starts.

Example:
  .\\.venv\\Scripts\\python.exe scripts\\shorts\\find_stings.py ^
    --work-dir projects\\myjob --template 3412.72,3413.60
"""

from __future__ import annotations

import argparse
import json
import sys
import wave
from pathlib import Path

import numpy as np


def load_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path)) as handle:
        sr = handle.getframerate()
        raw = handle.readframes(handle.getnframes())
    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    return audio, sr


def correlate_valid(x: np.ndarray, tpl: np.ndarray) -> np.ndarray:
    """sum(x[k:k+L] * tpl) for every k, by overlap-save FFT."""
    length = len(tpl)
    n = 1 << 16
    block = n - length + 1
    if block <= 0:
        raise SystemExit("template longer than the FFT block; shorten --template")
    tpl_f = np.fft.rfft(tpl[::-1].astype(np.float64), n)
    out = np.empty(len(x) - length + 1, dtype=np.float64)
    pos = 0
    while pos < len(out):
        seg = x[pos : pos + n]
        if len(seg) < n:
            seg = np.pad(seg, (0, n - len(seg)))
        y = np.fft.irfft(np.fft.rfft(seg.astype(np.float64), n) * tpl_f, n)
        take = min(block, len(out) - pos)
        out[pos : pos + take] = y[length - 1 : length - 1 + take]
        pos += block
    return out


def normalised_correlation(audio: np.ndarray, tpl: np.ndarray) -> np.ndarray:
    centred = tpl.astype(np.float64) - tpl.mean()
    tpl_norm = np.sqrt((centred**2).sum())
    length = len(centred)
    num = correlate_valid(audio, centred)
    wide = audio.astype(np.float64)
    cs = np.concatenate([[0.0], np.cumsum(wide)])
    cs2 = np.concatenate([[0.0], np.cumsum(wide**2)])
    k = np.arange(len(num))
    total = cs[k + length] - cs[k]
    energy = cs2[k + length] - cs2[k]
    var = np.maximum(energy - total * total / length, 1e-12)
    return num / (np.sqrt(var) * tpl_norm)


def measure_length(
    audio: np.ndarray, sr: int, seed: int, others: list[int], *, floor: float
) -> float:
    """Grow the template until copies stop matching; that is the insert length."""

    def ncc(start: int, length: int) -> float:
        tpl = audio[seed : seed + length].astype(np.float64)
        tpl = tpl - tpl.mean()
        win = audio[start : start + length].astype(np.float64)
        win = win - win.mean()
        denom = np.sqrt((tpl**2).sum() * (win**2).sum()) + 1e-12
        return float((tpl * win).sum() / denom)

    best = 0.5
    trial = 0.5
    while trial <= 8.0:
        length = int(trial * sr)
        if seed + length >= len(audio):
            break
        scores = [ncc(o, length) for o in others]
        if scores and min(scores) < floor:
            break
        best = trial
        trial += 0.05
    return round(best, 3)


def speech_end_before(
    audio: np.ndarray, sr: int, onset: float, *, look: float, guard: float, pad: float
) -> float:
    """Where speech last stopped before the insert.

    Not the quietest moment in the run-up -- a dip mid-sentence is often quieter
    than the pause that follows it, and cutting there truncated real words on
    three of the nine occurrences. Instead take the last frame that is clearly
    above the local noise floor and cut just after it.
    """
    step = 0.02
    start = max(0.0, onset - look)
    limit = max(start + step, onset - guard)
    times = np.arange(start, limit, step)
    if len(times) == 0:
        return max(0.0, onset - guard)
    levels = np.array(
        [
            20 * np.log10(np.sqrt((audio[int(t * sr) : int((t + step) * sr)] ** 2).mean() + 1e-12))
            for t in times
        ]
    )
    floor = float(np.percentile(levels, 15))
    loud = np.nonzero(levels > floor + 8.0)[0]
    if len(loud) == 0:
        return float(times[0])
    return float(min(times[loud[-1]] + step + pad, onset - guard))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Find inserted transition stings.")
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument(
        "--template",
        required=True,
        help='"start,end" in seconds of one clean occurrence.',
    )
    parser.add_argument(
        "--threshold", type=float, default=0.80, help="Minimum normalised correlation for a copy."
    )
    parser.add_argument(
        "--length-floor",
        type=float,
        default=0.95,
        help="Correlation a grown template must keep to count.",
    )
    parser.add_argument(
        "--min-gap", type=float, default=3.0, help="Seconds between distinct occurrences."
    )
    args = parser.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    cache = args.work_dir / ".aive"
    wav = cache / "audio.wav"
    if not wav.is_file():
        raise SystemExit(f"missing {wav}; run transcribe.py first")
    try:
        t_start, t_end = (float(x) for x in args.template.split(","))
    except ValueError as exc:
        raise SystemExit('--template wants "start,end", e.g. 3412.72,3413.60') from exc

    audio, sr = load_wav(wav)
    tpl = audio[int(t_start * sr) : int(t_end * sr)]
    if len(tpl) < sr // 10:
        raise SystemExit("template is too short")
    ncc = normalised_correlation(audio, tpl)

    order = np.argsort(-ncc)
    picked: list[int] = []
    for idx in order:
        if ncc[idx] < args.threshold:
            break
        if all(abs(int(idx) - p) > args.min_gap * sr for p in picked):
            picked.append(int(idx))
    picked.sort()
    if not picked:
        raise SystemExit(f"no occurrence scored >= {args.threshold}")
    if len(picked) < 2:
        # A channel insert is pasted in repeatedly; a template matching only
        # itself is a one-off sound, and calling it a sting would drop good
        # audio. Emit nothing rather than a bogus span.
        raise SystemExit(
            f"template matches only itself at {picked[0] / sr:.2f}s, so it does "
            "not repeat and is not an inserted sting; nothing written"
        )

    seed = picked[int(np.argmax([ncc[p] for p in picked]))]
    others = [p for p in picked if p != seed]
    length = measure_length(audio, sr, seed, others, floor=args.length_floor)

    rows = []
    for idx in picked:
        onset = idx / sr
        end = onset + length
        rows.append(
            {
                "start": round(onset, 3),
                "end": round(end, 3),
                "ncc": round(float(ncc[idx]), 4),
                # cut the outgoing clip where speech stopped, not at the onset
                "cut_before": round(
                    speech_end_before(audio, sr, onset, look=2.5, guard=0.05, pad=0.15),
                    3,
                ),
                # the insert is a replacement, so original audio resumes at its end
                "cut_after": round(end + 0.03, 3),
            }
        )

    out = cache / "stings.json"
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    below = [ncc[i] for i in order if ncc[i] < args.threshold][:1]
    summary = (
        f"matches={len(rows)} insert_length={length:.2f}s "
        f"ncc_range={min(r['ncc'] for r in rows):.4f}..{max(r['ncc'] for r in rows):.4f}"
    )
    if below:
        summary += f" best_non_match={below[0]:.4f}"
    print(summary)
    for r in rows:
        print(
            f"  {r['start']:8.2f}-{r['end']:8.2f}s  ncc {r['ncc']:.4f}  "
            f"cut_before {r['cut_before']:8.2f}  cut_after {r['cut_after']:8.2f}"
        )
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
