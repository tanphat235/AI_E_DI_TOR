"""Measure where a re-upload's burned-in text, logos and borders sit.

Re-uploads carry channel furniture baked into the picture: a watermark in a
corner, a decorative pillarbox, a scrolling promo banner, a title card. It has
to be cropped out of every short, and eyeballing a grid frame per video is slow
and easy to get wrong by a few pixels.

**This reports; it does not decide.** Two signal designs were tried as full
auto-detectors and both failed on real material, so the numbers below are for a
human to read, not to act on blindly:

- *Absolute* near-zero temporal variance finds only fully opaque overlays. On
  the song-tot talk the corner watermark measured std 3.22 against a plain wall
  at 10.85 -- lower than its surroundings but far from frozen, because it is
  semi-transparent and inherits motion from the footage beneath. Only 0.24% of
  that frame was near-frozen, scattered everywhere, i.e. flat compressed blocks.
- *Local* variance ratio finds that same corner watermark (0.49x the frame
  median) but misses a second, fainter mark on the same frame (1.33x, i.e.
  above the median), while flagging 174 of 576 blocks of ordinary low-motion
  ceiling.

So a fixed-camera talk is full of still, sharp, legitimate scene content that
looks exactly like an overlay to any cheap test. An auto-crop built on either
signal threw away a third of the usable picture.

What this does instead: dump a diagnostic PNG-ish montage (variance map, edge
map, and the candidate mask over a real frame) plus per-region statistics and a
*suggested* crop of the requested aspect. Look at the montage, then set
--src-crop yourself.

Example:
  .\\.venv\\Scripts\\python.exe scripts\\shorts\\find_crop.py ^
    --source talk.mp4 --aspect 1080:960 --out-dir scratch\\crop
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


def _ffmpeg() -> Path:
    try:
        import imageio_ffmpeg

        return Path(imageio_ffmpeg.get_ffmpeg_exe())
    except Exception as exc:
        raise SystemExit(f"ffmpeg not found via imageio-ffmpeg: {exc}") from exc


def _probe(path: Path) -> tuple[int, int, float]:
    import av

    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        dur = (
            float(container.duration) / 1_000_000.0
            if container.duration
            else float(stream.duration * stream.time_base)
        )
        return int(stream.width), int(stream.height), dur


def sample_grey(
    source: Path, ffmpeg: Path, *, width: int, height: int, times: list[float]
) -> np.ndarray:
    frames = []
    for t in times:
        proc = subprocess.run(
            [
                str(ffmpeg),
                "-y",
                "-ss",
                f"{t:.3f}",
                "-i",
                str(source),
                "-frames:v",
                "1",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "gray",
                "-",
            ],
            capture_output=True,
        )
        if len(proc.stdout) < width * height:
            continue
        frames.append(
            np.frombuffer(proc.stdout[: width * height], dtype=np.uint8).reshape(height, width)
        )
    if len(frames) < 4:
        raise SystemExit(f"only decoded {len(frames)} frames; need at least 4")
    return np.stack(frames).astype(np.float32)


def edge_map(frame: np.ndarray) -> np.ndarray:
    gx = np.zeros_like(frame)
    gy = np.zeros_like(frame)
    gx[:, 1:-1] = frame[:, 2:] - frame[:, :-2]
    gy[1:-1, :] = frame[2:, :] - frame[:-2, :]
    return np.abs(gx) + np.abs(gy)


def block_stats(std: np.ndarray, edges: np.ndarray, block: int) -> tuple[np.ndarray, np.ndarray]:
    h, w = std.shape
    gh, gw = h // block, w // block
    s = std[: gh * block, : gw * block].reshape(gh, block, gw, block).mean(axis=(1, 3))
    e = edges[: gh * block, : gw * block].reshape(gh, block, gw, block).mean(axis=(1, 3))
    return s, e


def suggest_crop(
    mask: np.ndarray, *, aspect_w: int, aspect_h: int, step: int
) -> tuple[tuple[int, int, int, int], int]:
    """Largest window of the given aspect holding the fewest flagged pixels."""
    height, width = mask.shape
    integral = np.zeros((height + 1, width + 1), dtype=np.int64)
    integral[1:, 1:] = np.cumsum(np.cumsum(mask.astype(np.int64), axis=0), axis=1)

    def bad(x: int, y: int, w: int, h: int) -> int:
        return int(
            integral[y + h, x + w] - integral[y, x + w] - integral[y + h, x] + integral[y, x]
        )

    best: tuple[tuple[int, int, int, int], int] | None = None
    h = height
    while h >= height // 3:
        w = round(h * aspect_w / aspect_h)
        if w <= width:
            for y in range(0, height - h + 1, step):
                for x in range(0, width - w + 1, step):
                    c = bad(x, y, w, h)
                    if best is None or c < best[1]:
                        best = ((w, h, x, y), c)
                    if c == 0:
                        return best
        h -= step
    if best is None:
        raise SystemExit("no window fits the requested aspect")
    return best


def write_montage(
    out_dir: Path,
    ffmpeg: Path,
    *,
    frame: np.ndarray,
    std: np.ndarray,
    edges: np.ndarray,
    mask: np.ndarray,
    crop: tuple[int, int, int, int],
) -> Path:
    """Four panels: a real frame, variance, edges, and the flagged mask on it."""
    out_dir.mkdir(parents=True, exist_ok=True)
    h, w = std.shape

    def norm(x: np.ndarray) -> np.ndarray:
        top = np.percentile(x, 99) or 1.0
        return np.clip(x / top, 0, 1)

    grey = frame / 255.0
    panels = [grey, norm(std), norm(edges)]
    overlay = np.stack([grey, grey, grey], axis=-1)
    overlay[mask] = [1.0, 0.15, 0.15]
    cw, ch, cx, cy = crop
    box = np.zeros_like(mask)
    box[cy : cy + ch, cx : cx + 2] = True
    box[cy : cy + ch, cx + cw - 2 : cx + cw] = True
    box[cy : cy + 2, cx : cx + cw] = True
    box[cy + ch - 2 : cy + ch, cx : cx + cw] = True
    overlay[box] = [0.2, 1.0, 0.2]

    rgb_rows = []
    for p in panels:
        rgb_rows.append((np.stack([p, p, p], axis=-1) * 255).astype(np.uint8))
    rgb_rows.append((overlay * 255).astype(np.uint8))
    top_row = np.concatenate(rgb_rows[:2], axis=1)
    bottom_row = np.concatenate(rgb_rows[2:], axis=1)
    sheet = np.concatenate([top_row, bottom_row], axis=0)

    raw = out_dir / "crop_diag.rgb"
    sheet.tofile(raw)
    png = out_dir / "crop_diag.png"
    subprocess.run(
        [
            str(ffmpeg),
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{w * 2}x{h * 2}",
            "-i",
            str(raw),
            "-frames:v",
            "1",
            "-update",
            "1",
            "-vf",
            f"scale={w}:{h}",
            str(png),
        ],
        capture_output=True,
    )
    raw.unlink(missing_ok=True)
    return png


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Measure burned-in furniture and suggest a crop (for review)."
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--aspect", default="1080:960")
    parser.add_argument("--span", default="", help='Limit sampling to "start,end" seconds.')
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--block", type=int, default=40, help="Block size for the report grid.")
    parser.add_argument(
        "--ratio-max",
        type=float,
        default=0.55,
        help="Flag blocks whose variance is below this fraction of the median.",
    )
    parser.add_argument(
        "--edge-min",
        type=float,
        default=3.0,
        help="...and whose edge energy is at least this, i.e. text not blank wall.",
    )
    parser.add_argument(
        "--banner-mult",
        type=float,
        default=3.0,
        help="Flag rows whose variance exceeds this multiple of the median.",
    )
    parser.add_argument("--step", type=int, default=2)
    parser.add_argument("--out-dir", type=Path, default=None, help="Where to write the montage.")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if not args.source.is_file():
        raise SystemExit(f"source not found: {args.source}")
    aspect_w, aspect_h = (int(x) for x in args.aspect.split(":"))

    width, height, total = _probe(args.source)
    lo, hi = (0.0, total) if not args.span else tuple(float(x) for x in args.span.split(","))
    pad = (hi - lo) * 0.02
    times = list(np.linspace(lo + pad, hi - pad, args.samples))

    ffmpeg = _ffmpeg()
    stack = sample_grey(args.source, ffmpeg, width=width, height=height, times=times)
    std = stack.std(axis=0)
    median_frame = np.median(stack, axis=0)
    edges = edge_map(median_frame)

    bstd, bedge = block_stats(std, edges, args.block)
    gmed = float(np.median(bstd))
    ratio = bstd / (gmed or 1.0)

    flagged_blocks = (ratio < args.ratio_max) & (bedge >= args.edge_min)
    mask = np.zeros_like(std, dtype=bool)
    for by, bx in zip(*np.nonzero(flagged_blocks), strict=False):
        mask[
            by * args.block : (by + 1) * args.block,
            bx * args.block : (bx + 1) * args.block,
        ] = True

    row_var = std.mean(axis=1)
    banner_rows = row_var > np.median(row_var) * args.banner_mult
    mask[banner_rows, :] = True

    crop, bad = suggest_crop(mask, aspect_w=aspect_w, aspect_h=aspect_h, step=args.step)
    cw, ch, cx, cy = crop

    regions = []
    for by, bx in zip(*np.nonzero(flagged_blocks), strict=False):
        regions.append(
            {
                "x": int(bx * args.block),
                "y": int(by * args.block),
                "w": args.block,
                "h": args.block,
                "var_ratio": round(float(ratio[by, bx]), 2),
                "edge": round(float(bedge[by, bx]), 1),
            }
        )

    result = {
        "source": str(args.source),
        "frame": f"{width}x{height}",
        "samples": int(stack.shape[0]),
        "median_block_variance": round(gmed, 2),
        "flagged_blocks": len(regions),
        "banner_rows": int(banner_rows.sum()),
        "suggested_crop": f"{cw}:{ch}:{cx}:{cy}",
        "kept_pct": round(100.0 * cw * ch / (width * height), 1),
        "flagged_pixels_inside": int(bad),
        "regions": regions[:40],
    }

    montage = None
    if args.out_dir is not None:
        montage = write_montage(
            args.out_dir,
            ffmpeg,
            frame=median_frame,
            std=std,
            edges=edges,
            mask=mask,
            crop=crop,
        )
        result["montage"] = str(montage)

    if args.json:
        print(json.dumps(result, indent=2))
        return 0

    print(f"source            {args.source.name}")
    print(f"frame             {width}x{height}, {stack.shape[0]} samples over {lo:.0f}-{hi:.0f}s")
    print(f"median block var  {gmed:.2f}")
    print(f"flagged           {len(regions)} blocks, {int(banner_rows.sum())} banner rows")
    for r in regions[:12]:
        print(f"   x {r['x']:4d} y {r['y']:4d}  var {r['var_ratio']:4.2f}x  edge {r['edge']:5.1f}")
    if len(regions) > 12:
        print(f"   ... and {len(regions) - 12} more")
    print(
        f"suggested crop    {cw}:{ch}:{cx}:{cy}  keeps {result['kept_pct']}% "
        f"({bad} flagged pixels inside)"
    )
    if montage:
        print(f"montage           {montage}")
    print()
    print("This is a measurement, not a decision. Look at the montage before")
    print("setting --src-crop: a still, sharp piece of the real scene looks the")
    print("same as an overlay to this test.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
