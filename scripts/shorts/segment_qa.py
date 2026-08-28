"""Segment a talk into standalone Q&A blocks that never cut mid-sentence.

Unlike transcript-gap segmentation, boundaries come from acoustic silence
measured on the audio itself (ffmpeg ``silencedetect``). Whisper's segment
timestamps abut one another -- on a 65 min talk only 6 transcript gaps
exceeded 0.5s, so they carry no usable pause signal. Silence detection on the
same audio found 949.

A clip therefore ends only at a real pause. When a topic block runs past
``--max-sec`` it is cut at the best available pause rather than chopped at a
fixed offset, so no clip ever ends in the middle of a sentence.

If ``.aive/stings.json`` exists (see ``find_stings.py``) every transition sting
another channel inserted becomes a mandatory boundary and its own span is
dropped. That matters twice over: the insert is as loud as speech, so silence
detection never flags it and the packer would otherwise cut straight into it;
and the insert sits exactly where the next question begins, so honouring it is
what stops a clip from carrying half of one answer and half of the next.

Example:
  .\\.venv\\Scripts\\python.exe scripts\\shorts\\segment_qa.py ^
    --work-dir projects\\myjob --target-sec 120
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path

# Recurring faster-whisper mishears on Vietnamese Buddhist vocabulary. Titles
# are read by humans, so fix the words that recur. The audio is untouched.
TITLE_FIXES = (
    ("Phật Đảng", "Phật Đản"),
    ("phật đảng", "Phật Đản"),
    ("Đảng xanh", "đản sanh"),
    ("đảng xanh", "đản sanh"),
    ("Môn Nhi", "Mâu Ni"),
    ("môn nhi", "Mâu Ni"),
    ("camo ni", "Mâu Ni"),
    ("Đại trúng", "đại chúng"),
    ("đại trúng", "đại chúng"),
    ("Giày Su", "Giê-su"),
    ("Thứ đại chúng", "Thưa đại chúng"),
)

# A cut is welcome in front of these -- they open a new question or topic.
OPEN_CUES = (
    "cau hoi",
    "thua thay",
    "kinh bach",
    "co nguoi hoi",
    "co vi hoi",
    "nguoi ta hoi",
    "thua dai chung",
    "thua quy vi",
    "vay thi",
    "tiep theo",
    "bay gio",
    "hom nay",
    "truoc het",
    "ke den",
    "thu hai",
    "cau nay",
)

# A cut in front of these would sever a clause from what it depends on.
CONTINUATION = (
    "va ",
    "ma ",
    "nhung ",
    "thi ",
    "cho nen",
    "tai vi",
    "boi vi",
    "roi ",
    "nen ",
    "hoac ",
    "voi ",
    "cua ",
    "de ",
    "vi ",
    "con ",
)


def _ffmpeg() -> Path:
    try:
        import imageio_ffmpeg

        return Path(imageio_ffmpeg.get_ffmpeg_exe())
    except Exception as exc:
        raise SystemExit(f"ffmpeg not found via imageio-ffmpeg: {exc}") from exc


def _strip_accents(text: str) -> str:
    decomposed = unicodedata.normalize("NFD", text.lower())
    return "".join(c for c in decomposed if unicodedata.category(c) != "Mn")


@dataclass(frozen=True, slots=True)
class Silence:
    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


def detect_silences(wav: Path, ffmpeg: Path, *, noise_db: float, min_dur: float) -> list[Silence]:
    """Measure silence on the audio itself -- the only reliable pause signal."""
    proc = subprocess.run(
        [
            str(ffmpeg),
            "-hide_banner",
            "-nostats",
            "-i",
            str(wav),
            "-af",
            f"silencedetect=noise={noise_db}dB:d={min_dur}",
            "-f",
            "null",
            "-",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    log = (proc.stderr or "") + (proc.stdout or "")
    starts = [float(x) for x in re.findall(r"silence_start: ([\d.]+)", log)]
    ends = [float(x) for x in re.findall(r"silence_end: ([\d.]+)", log)]
    if not starts:
        raise SystemExit("silencedetect found no silence; check the audio file")
    return [Silence(a, b) for a, b in zip(starts, ends, strict=False)]


def _text_between(segments: list[dict], start: float, end: float) -> str:
    parts = [s["text"] for s in segments if s["end"] > start and s["start"] < end]
    return re.sub(r"\s+", " ", " ".join(parts)).strip()


def _words_after(segments: list[dict], t: float, n: int = 12) -> str:
    parts: list[str] = []
    for s in segments:
        if s["end"] > t:
            parts.append(s["text"])
            if len(" ".join(parts).split()) > n:
                break
    return " ".join(" ".join(parts).split()[:n])


def score_boundary(sil: Silence, segments: list[dict]) -> float:
    """How much a cut here looks like a real topic or question boundary."""
    score = min(sil.duration, 5.0) * 2.0
    after = _strip_accents(_words_after(segments, sil.end))
    if any(after.startswith(cue) or f" {cue}" in after[:40] for cue in OPEN_CUES):
        score += 3.0
    if any(after.startswith(cue) for cue in CONTINUATION):
        score -= 2.5
    return score


def _cut_out(sil: Silence) -> float:
    """Where a clip ends: just inside the pause, so the last word survives."""
    return sil.start + min(0.45, sil.duration * 0.4)


def _cut_in(sil: Silence) -> float:
    """Where the next clip starts: just before speech resumes."""
    return sil.end - min(0.35, sil.duration * 0.3)


def _pack_block(
    lo: float,
    hi: float,
    scored: list[tuple[Silence, float]],
    *,
    min_sec: float,
    target_sec: float,
    max_sec: float,
) -> list[tuple[float, float]]:
    """Split one question into clips at silences, never at a fixed offset."""
    inner = [(s, sc) for s, sc in scored if lo + min_sec <= _cut_out(s) and _cut_in(s) <= hi]
    blocks: list[tuple[float, float]] = []
    cur = lo
    used = 0
    while used < len(inner):
        if hi - cur <= max_sec:
            break  # what is left already fits; no need to cut again
        window = [
            (i, s, sc)
            for i, (s, sc) in enumerate(inner[used:], start=used)
            if min_sec <= _cut_out(s) - cur <= max_sec
        ]
        if window:
            best = max(
                window,
                key=lambda t, at=cur: (
                    t[2] - abs((_cut_out(t[1]) - at) - target_sec) / target_sec * 4.0
                ),
            )
        else:
            # Nothing inside the window: take the first pause past min_sec
            # rather than chopping mid-sentence at max_sec.
            beyond = [
                (i, s, sc)
                for i, (s, sc) in enumerate(inner[used:], start=used)
                if _cut_out(s) - cur >= min_sec
            ]
            if not beyond:
                break
            best = beyond[0]
        idx, sil, _ = best
        blocks.append((cur, _cut_out(sil)))
        cur = _cut_in(sil)
        used = idx + 1

    if hi - cur > 0.5:
        blocks.append((cur, hi))
    elif blocks:
        blocks[-1] = (blocks[-1][0], hi)
    return blocks


def build_segments(
    silences: list[Silence],
    segments: list[dict],
    *,
    duration: float,
    min_sec: float,
    target_sec: float,
    max_sec: float,
    min_boundary: float,
    title_fallback: str,
    stings: list[dict] | None = None,
) -> list[dict]:
    cands = [s for s in silences if s.duration >= min_boundary]
    if not cands:
        raise SystemExit(f"no silence >= {min_boundary}s; lower --min-boundary")
    scored = [(s, score_boundary(s, segments)) for s in cands]

    start_at = _cut_in(silences[0]) if silences[0].start <= 0.05 else 0.0

    # A transition sting marks where the next question begins, so it is a
    # mandatory boundary and its own span is dropped. Without this the packer
    # cuts by duration alone and lands mid-question -- or inside the sting,
    # which is loud enough that silence detection never flags it.
    questions: list[tuple[float, float]] = []
    cur = max(0.0, start_at)
    for sting in sorted(stings or [], key=lambda s: s["start"]):
        if sting["cut_before"] > cur + 0.5:
            questions.append((cur, sting["cut_before"]))
        cur = sting["cut_after"]
    if duration - cur > 0.5:
        questions.append((cur, duration))

    blocks: list[tuple[float, float]] = []
    for lo, hi in questions:
        blocks.extend(
            _pack_block(lo, hi, scored, min_sec=min_sec, target_sec=target_sec, max_sec=max_sec)
        )

    out: list[dict] = []
    for i, (a, b) in enumerate(blocks, start=1):
        text = _text_between(segments, a, b)
        out.append(
            {
                "id": f"clip_{i:03d}",
                "start": round(a, 3),
                "end": round(b, 3),
                "duration": round(b - a, 3),
                "text": text,
                "title": _title_from(text, index=i, fallback=title_fallback),
            }
        )
    return out


def _title_from(text: str, *, index: int, fallback: str) -> str:
    clean = text
    for wrong, right in TITLE_FIXES:
        clean = clean.replace(wrong, right)
    clean = re.sub(r"\s+", " ", clean).strip(" .,;:-")
    for sep in ("? ", ". ", "! ", ", "):
        if sep in clean[:90]:
            clean = clean.split(sep, 1)[0]
            break
    if len(clean) < 12:
        clean = f"{fallback} #{index:02d}"
    if len(clean) > 52:
        clean = clean[:52].rsplit(" ", 1)[0] + "..."
    return clean


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Cut a talk into whole Q&A blocks.")
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--min-sec", type=float, default=45.0)
    parser.add_argument("--target-sec", type=float, default=120.0)
    parser.add_argument("--max-sec", type=float, default=175.0)
    parser.add_argument(
        "--min-boundary",
        type=float,
        default=1.5,
        help="Shortest silence allowed to end a clip.",
    )
    parser.add_argument("--noise-db", type=float, default=-30.0)
    parser.add_argument("--min-silence", type=float, default=0.45)
    parser.add_argument("--title-fallback", default="Phap thoai")
    args = parser.parse_args(argv)

    # Titles are Vietnamese; a cp1252 console would raise on the summary.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    cache = args.work_dir / ".aive"
    wav = cache / "audio.wav"
    tpath = cache / "transcript.json"
    if not wav.is_file():
        raise SystemExit(f"missing {wav}; run transcribe.py first")
    if not tpath.is_file():
        raise SystemExit(f"missing {tpath}; run transcribe.py first")

    doc = json.loads(tpath.read_text(encoding="utf-8"))
    if doc.get("partial"):
        raise SystemExit("transcript is still partial; wait for transcribe.py")

    sting_path = cache / "stings.json"
    stings = json.loads(sting_path.read_text(encoding="utf-8")) if sting_path.is_file() else []

    sil = detect_silences(wav, _ffmpeg(), noise_db=args.noise_db, min_dur=args.min_silence)
    (cache / "silences.json").write_text(
        json.dumps(
            [{"start": s.start, "end": s.end, "duration": round(s.duration, 3)} for s in sil],
            indent=2,
        ),
        encoding="utf-8",
    )
    segs = build_segments(
        sil,
        doc["segments"],
        duration=float(doc["duration"]),
        min_sec=args.min_sec,
        target_sec=args.target_sec,
        max_sec=args.max_sec,
        min_boundary=args.min_boundary,
        title_fallback=args.title_fallback,
        stings=stings,
    )
    out = cache / "answer_segments.json"
    out.write_text(json.dumps(segs, ensure_ascii=False, indent=2), encoding="utf-8")

    total = sum(s["duration"] for s in segs)
    print(
        f"silences={len(sil)} segments={len(segs)} covered={total:.0f}s of {doc['duration']:.0f}s"
    )
    for seg in segs:
        print(
            f"  {seg['id']} {seg['start']:7.1f}-{seg['end']:7.1f} "
            f"({seg['duration']:5.1f}s) {seg['title']}"
        )
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
