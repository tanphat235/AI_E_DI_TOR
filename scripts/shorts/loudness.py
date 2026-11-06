"""Bring a finished clip to one fixed loudness, up or down.

Why this exists: the centre format used to only pull a clip DOWN to a ceiling
(-15.3 LUFS). A clip that rendered quieter was left alone, so shorts cut from a
distant camera (a wide hall shot, a softer voice on the recording) came out
1.5-2 dB under the close-up ones from the same talk: -16.7 to -17.4 LUFS
against -15.3. A viewer swiping between shorts hears that as "this one is
quiet". The target is now a fixed number and the correction goes both ways.

What is and is not touched, so the output quality does not change:

* Gain only. One linear ``volume`` step on the finished mix, so the balance of
  voice to music bed that the render measured is exactly the same. No
  compressor, no EQ, no loudnorm rewrite (loudnorm's own normalise pass
  squashes dynamics; here it is used only to MEASURE).
* The same peak ceiling the render uses, applied after the gain
  (``alimiter``, ``level=disabled``), so a boost cannot clip.
* The video stream is copied bit for bit. Audio is re-encoded once from the
  ORIGINAL file each attempt (never from a previous attempt), with the render's
  own AAC 128k settings, so there is no generational loss.
* A clip already within ``tolerance`` of the target is left completely alone.

Run on existing files (writes next to nothing; give --out-dir to keep the
originals):

    python scripts/shorts/loudness.py clip.mp4 other.mp4 --out-dir fixed/
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

# House loudness for centre-format shorts, in integrated LUFS. Measured on the
# established close-up shorts (-15.3 to -15.4 LUFS) and the same number the old
# ceiling was set to. center_format.LOUDNESS_TARGET passes it to build_shorts.
TARGET_LUFS = -15.3
# Within this many LU of the target counts as on target; also the stopping
# point of the correction loop. 0.15 LU is below what an ear can tell apart.
TOLERANCE_LU = 0.15
# Sample-peak ceiling after the gain, dBFS. -1.5 rather than the render's own -1.0: the AAC
# encode that follows adds 0.5-1 dB of overshoot, and with -1.0 a clip measured -0.1 dBFS
# true peak (2026-11-06), too close to clipping on some players.
PEAK_CEILING_DBFS = -1.5
# True-peak limit of the FINISHED file (after the AAC encode, inter-sample peaks included).
# The sample-peak limiter above cannot see either, so the result is measured and, if it is
# over, the ceiling is lowered by the excess and the clip re-encoded from the original.
TRUE_PEAK_MAX_DBTP = -1.0
MAX_PASSES = 4


def _ffmpeg() -> Path:
    import imageio_ffmpeg

    return Path(imageio_ffmpeg.get_ffmpeg_exe())


def measure(ffmpeg: Path, path: Path) -> tuple[float | None, float | None]:
    """Integrated loudness (LUFS) and true peak (dBTP) of a file's audio, one read-only pass."""
    proc = subprocess.run(
        [str(ffmpeg), "-i", str(path), "-vn", "-af", "loudnorm=print_format=json", "-f", "null", "-"],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    i = re.search(r'"input_i"\s*:\s*"(-?[\d.]+)"', proc.stderr or "")
    tp = re.search(r'"input_tp"\s*:\s*"(-?[\d.]+)"', proc.stderr or "")
    return (float(i.group(1)) if i else None, float(tp.group(1)) if tp else None)


def measure_lufs(ffmpeg: Path, path: Path) -> float | None:
    """Integrated loudness of a file's audio, in LUFS (read-only pass)."""
    return measure(ffmpeg, path)[0]


def _encode(ffmpeg: Path, src: Path, dst: Path, gain_db: float, ceiling_dbfs: float) -> bool:
    limit = 10.0 ** (ceiling_dbfs / 20.0)
    cmd = [
        str(ffmpeg), "-y", "-i", str(src),
        "-af", f"volume={gain_db:.3f}dB,alimiter=limit={limit:.4f}:level=disabled",
        "-c:v", "copy", "-c:a", "aac", "-b:a", "128k",
        "-movflags", "+faststart", str(dst),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    return proc.returncode == 0 and dst.is_file()


def match_loudness(
    ffmpeg: Path,
    src: Path,
    dst: Path | None = None,
    *,
    target: float = TARGET_LUFS,
    tolerance: float = TOLERANCE_LU,
    ceiling_dbfs: float = PEAK_CEILING_DBFS,
    tp_max: float = TRUE_PEAK_MAX_DBTP,
) -> tuple[float | None, float | None]:
    """Bring `src` to `target` LUFS and under `tp_max` dBTP. Returns (before, after) LUFS.

    `dst` None means in place. A clip already on target AND under the true-peak
    limit is not rewritten (with a `dst` it is copied as is). The limiter can hold
    a boost short of the target on a very peaky clip; the loop then stops at the
    best it reached and says so rather than pretending.
    """
    in_place = dst is None
    out = src if in_place else dst
    before, tp_before = measure(ffmpeg, src)
    if before is None:
        print(f"  WARNING loudness not measurable; {src.name} left as rendered")
        return None, None
    on_target = abs(target - before) <= tolerance
    peak_ok = tp_before is None or tp_before <= tp_max
    if on_target and peak_ok:
        if not in_place:
            shutil.copy2(src, out)
        print(f"  loudness {before:.1f} LUFS: on target ({target:.1f}), untouched")
        return before, before

    gain = target - before
    ceiling = ceiling_dbfs
    tmp = out.with_name(out.stem + ".loudfix" + out.suffix)
    best: tuple[float, float, float] | None = None  # (score, gain, ceiling): closest attempt
    after, tp = before, tp_before
    for _ in range(MAX_PASSES):
        if not _encode(ffmpeg, src, tmp, gain, ceiling):
            tmp.unlink(missing_ok=True)
            print(f"  WARNING loudness fix failed; keeping {before:.1f} LUFS as rendered")
            return before, before
        after, tp = measure(ffmpeg, tmp)
        if after is None:
            break
        err = target - after
        over = max(0.0, (tp - tp_max)) if tp is not None else 0.0
        score = abs(err) + 4.0 * over  # a peak over the limit counts more than a loudness miss
        if best is None or score < best[0]:
            best = (score, gain, ceiling)
        if abs(err) <= tolerance and over <= 0.0:
            break
        if over > 0.0:
            ceiling -= over + 0.05  # the encode overshot: pull the limiter down by the excess
        if abs(err) > tolerance:
            gain += err  # the limiter ate some of it; ask for the shortfall again
    if best is not None and (after is None or best[1] != gain or best[2] != ceiling):
        _encode(ffmpeg, src, tmp, best[1], best[2])
        after, tp = measure(ffmpeg, tmp)
        gain, ceiling = best[1], best[2]
    tmp.replace(out)
    notes = []
    if after is not None and abs(target - after) > tolerance:
        notes.append(f"peak limit held it short of {target:.1f}")
    if tp is not None and tp > tp_max:
        notes.append(f"true peak {tp:.1f} dBTP, over {tp_max:.1f}")
    note = ("  (" + "; ".join(notes) + ")") if notes else ""
    tp_txt = f", {tp:.1f} dBTP" if tp is not None else ""
    print(f"  loudness {before:.1f} -> {after:.1f} LUFS{tp_txt} (target {target:.1f}, {gain:+.1f} dB){note}")
    return before, after


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("files", nargs="+", type=Path)
    p.add_argument("--target", type=float, default=TARGET_LUFS)
    p.add_argument("--out-dir", type=Path, help="Write fixed copies here; originals stay as they are.")
    p.add_argument("--measure-only", action="store_true")
    args = p.parse_args()
    ff = _ffmpeg()
    if args.out_dir:
        args.out_dir.mkdir(parents=True, exist_ok=True)
    for f in args.files:
        print(f.name)
        if args.measure_only:
            print(f"  {measure_lufs(ff, f)} LUFS")
            continue
        match_loudness(ff, f, (args.out_dir / f.name) if args.out_dir else None, target=args.target)
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
