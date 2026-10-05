"""The "where are we" map: one reference picture of the whole cosmology, with
the region being talked about lit and everything else dimmed.

Two renderings of the same thing:

* ``render_full`` -- the map as a full-frame shot, for the lines that are *about*
  the structure (the six realms one by one, where the Pure Land sits).
* ``render_mini`` -- a small copy for a corner of every generated picture, so a
  shot of a prince in a palace still tells the viewer which part of the map the
  story is in.

Regions are declared in the script, in the map image's own pixels, as one of
``circle [cx, cy, r]``, ``wedge [cx, cy, r_in, r_out, deg_from, deg_to]``
(degrees counter-clockwise from +x, y up), ``ellipse [x0, y0, x1, y1]``,
``box [x0, y0, x1, y1]`` or ``all``. The shapes are drawn, not detected: the
map is a hand-made illustration and its layout is known.

Labels are drawn here in Segoe UI Black, never asked of the picture model --
no diffusion model renders Vietnamese diacritics.
"""

from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont

FONT = Path(r"C:/Windows/Fonts/seguibl.ttf")
GOLD = (255, 210, 74)
# How much light the parts of the map not being talked about keep.
DIM = 0.32


def _mask(spec: dict, size: tuple[int, int], scale: float) -> Image.Image:
    m = Image.new("L", size, 0)
    d = ImageDraw.Draw(m)
    if spec.get("all"):
        d.rectangle([0, 0, size[0], size[1]], fill=255)
    elif "circle" in spec:
        cx, cy, r = (v * scale for v in spec["circle"])
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=255)
    elif "ellipse" in spec:
        d.ellipse([v * scale for v in spec["ellipse"]], fill=255)
    elif "box" in spec:
        x0, y0, x1, y1 = (v * scale for v in spec["box"])
        d.rounded_rectangle([x0, y0, x1, y1], radius=int(12 * scale), fill=255)
    elif "wedge" in spec:
        cx, cy, r_in, r_out, a0, a1 = spec["wedge"]
        cx, cy, r_in, r_out = cx * scale, cy * scale, r_in * scale, r_out * scale
        steps = 24
        angles = [math.radians(a0 + (a1 - a0) * k / steps) for k in range(steps + 1)]
        outer = [(cx + r_out * math.cos(a), cy - r_out * math.sin(a)) for a in angles]
        inner = [(cx + r_in * math.cos(a), cy - r_in * math.sin(a)) for a in reversed(angles)]
        d.polygon(outer + inner, fill=255)
    else:
        raise ValueError(f"region has no shape: {spec}")
    return m


def _lit(src: Image.Image, spec: dict, scale: float) -> Image.Image:
    """The map with ``spec`` lit, the rest dimmed, and a gold rim round it."""
    img = src.convert("RGB")
    if spec.get("all"):
        return img
    mask = _mask(spec, img.size, scale)
    dark = Image.eval(img, lambda v: int(v * DIM))
    out = Image.composite(img, dark, mask)
    # The rim: the mask's edge, thickened and softened into a glow.
    width = max(3, int(3 * scale))
    edge = ImageChops.subtract(mask.filter(ImageFilter.MaxFilter(2 * width + 1)), mask)
    glow = edge.filter(ImageFilter.GaussianBlur(width))
    rim = Image.new("RGB", img.size, GOLD)
    out = Image.composite(rim, out, ImageChops.lighter(edge, glow))
    return out


def _chip(draw: ImageDraw.ImageDraw, text: str, xy: tuple[int, int], size: int,
          anchor_right: bool = False) -> None:
    font = ImageFont.truetype(str(FONT), size)
    x0, y0, x1, y1 = draw.textbbox((0, 0), text, font=font)
    w, h = x1 - x0, y1 - y0
    pad = size // 3
    x, y = xy
    if anchor_right:
        x -= w + 2 * pad
    draw.rounded_rectangle([x, y, x + w + 2 * pad, y + h + 2 * pad], radius=size // 3,
                           fill=(0, 0, 0, 190), outline=GOLD, width=max(2, size // 14))
    draw.text((x + pad - x0, y + pad - y0), text, font=font, fill=GOLD)


def render_full(map_path: Path, spec: dict, out: Path, frame: tuple[int, int] = (1920, 1080)) -> Path:
    """The whole map filling a 16:9 frame, the region lit, its name top-left."""
    src = Image.open(map_path).convert("RGB")
    fw, fh = frame
    scale = min(fw / src.width, fh / src.height)
    big = src.resize((round(src.width * scale), round(src.height * scale)), Image.LANCZOS)
    lit = _lit(big, spec, scale)
    # Letterbox with a blurred, darkened copy of the map rather than flat black,
    # so the bars read as part of the picture.
    bg = src.resize(frame, Image.LANCZOS).filter(ImageFilter.GaussianBlur(24))
    bg = Image.eval(bg, lambda v: int(v * 0.4))
    bg.paste(lit, ((fw - lit.width) // 2, (fh - lit.height) // 2))
    if not spec.get("all") and spec.get("label"):
        # Top-left, under the map's own title band -- at the bottom it sat on
        # the caption plate.
        _chip(ImageDraw.Draw(bg, "RGBA"), spec["label"], (40, 100), 44)
    out.parent.mkdir(parents=True, exist_ok=True)
    bg.save(out)
    return out


def render_mini(map_path: Path, spec: dict, out: Path, width: int = 460) -> Path:
    """A corner-sized map, the region lit, with its name on a chip beneath."""
    src = Image.open(map_path).convert("RGB")
    scale = width / src.width
    small = src.resize((width, round(src.height * scale)), Image.LANCZOS)
    lit = _lit(small, spec, scale)
    chip_h = 58
    canvas = Image.new("RGBA", (width + 8, lit.height + 8 + chip_h), (0, 0, 0, 0))
    d = ImageDraw.Draw(canvas, "RGBA")
    d.rounded_rectangle([0, 0, width + 7, lit.height + 7], radius=10, fill=GOLD + (255,))
    canvas.paste(lit, (4, 4))
    if spec.get("label"):
        _chip(d, spec["label"], (width + 8, lit.height + 14), 30, anchor_right=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out)
    return out
