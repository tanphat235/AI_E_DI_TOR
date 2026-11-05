"""Generated assets for the old-film look: nothing here is copied from anywhere.

* ``make_overlay``  -- a looping RGBA clip of dust, scratches, hairs and drifting
  embers, laid over the finished picture. Grain, flicker, vignette and the colour
  grade are ffmpeg filters in ``assemble_film``; only the things that *move
  independently* of the picture need a clip.
* ``make_sfx``      -- the small mechanical sounds that go with a picture change,
  synthesised from noise and decaying tones.
* ``make_band``     -- the cream slide-mount strip that scrolls through between
  two pictures in the projector transition.
* ``render_card``   -- a chapter card: gold-on-red backdrop, kicker, title, sub.

Every function writes a file and returns its path, and skips the work when the
file already exists.
"""

from __future__ import annotations

import subprocess
import wave
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H = 1920, 1080
FPS = 30
SR = 48000
FONT_DIR = Path(r"C:/Windows/Fonts")

# ---------------------------------------------------------------------------
# Dust, scratches, hairs, embers
# ---------------------------------------------------------------------------


def make_overlay(ffmpeg: str, out: Path, seconds: int = 12, seed: int = 7) -> Path:
    """Render the looping dust-and-scratch clip (QuickTime RLE, with alpha)."""
    if out.is_file() and out.stat().st_size > 10_000:
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    frames = seconds * FPS

    scratches: list[dict] = []
    hairs: list[dict] = []
    # Embers persist across frames and drift upward; they wrap around, so the
    # loop point shows no jump -- they simply re-enter from below.
    embers = [
        {
            "x": float(rng.uniform(0, W)), "y": float(rng.uniform(0, H)),
            "vy": float(rng.uniform(0.5, 1.6)), "sway": float(rng.uniform(0.01, 0.04)),
            "ph": float(rng.uniform(0, 6.28)), "r": float(rng.uniform(1.5, 3.4)),
        }
        for _ in range(16)
    ]

    proc = subprocess.Popen(
        [ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgba",
         "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "qtrle", "-pix_fmt", "argb",
         str(out)],
        stdin=subprocess.PIPE,
    )
    assert proc.stdin is not None
    for n in range(frames):
        layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        d = ImageDraw.Draw(layer)

        # Vertical scratches: thin, pale, a few frames each, now and then a
        # long-lived one that stays put the way a real gate scratch does.
        if rng.random() < 0.10:
            scratches.append({
                "x": int(rng.integers(0, W)), "y0": int(rng.integers(0, H // 3)),
                "y1": int(rng.integers(2 * H // 3, H)), "w": int(rng.integers(1, 3)),
                "a": int(rng.integers(55, 130)), "life": int(rng.integers(2, 7)),
            })
        if rng.random() < 0.012:
            scratches.append({
                "x": int(rng.integers(0, W)), "y0": 0, "y1": H, "w": 1,
                "a": int(rng.integers(35, 75)), "life": int(rng.integers(10, 22)),
            })
        for s in scratches:
            x = s["x"] + int(rng.integers(-1, 2))
            d.line([(x, s["y0"]), (x, s["y1"])], fill=(236, 228, 208, s["a"]), width=s["w"])
            s["life"] -= 1
        scratches = [s for s in scratches if s["life"] > 0]

        # Dust: new specks every frame, a minority dark.
        for _ in range(int(rng.poisson(6))):
            x, y = int(rng.integers(0, W)), int(rng.integers(0, H))
            r = int(rng.choice([1, 1, 2, 2, 3]))
            if rng.random() < 0.4:
                col = (28, 18, 10, int(rng.integers(110, 190)))
            else:
                col = (240, 232, 212, int(rng.integers(110, 200)))
            d.ellipse([x - r, y - r, x + r, y + r], fill=col)
        if rng.random() < 0.05:
            x, y = int(rng.integers(0, W)), int(rng.integers(0, H))
            r = int(rng.integers(4, 8))
            d.ellipse([x - r, y - r, x + r, y + r], fill=(30, 20, 12, 55))

        # Hairs and fibres.
        if rng.random() < 0.015:
            px, py = float(rng.uniform(100, W - 100)), float(rng.uniform(100, H - 100))
            pts = [(px, py)]
            for _ in range(5):
                px += float(rng.uniform(-18, 18))
                py += float(rng.uniform(4, 22))
                pts.append((px, py))
            hairs.append({
                "pts": pts, "life": int(rng.integers(4, 12)),
                "col": (30, 20, 12, 150) if rng.random() < 0.6 else (235, 228, 210, 140),
            })
        for hr in hairs:
            d.line(hr["pts"], fill=hr["col"], width=1)
            hr["life"] -= 1
        hairs = [hr for hr in hairs if hr["life"] > 0]

        # Embers: a soft glow under a hot core, flickering.
        for e in embers:
            e["y"] -= e["vy"]
            if e["y"] < -10:
                e["y"] = H + 10
                e["x"] = float(rng.uniform(0, W))
            x = e["x"] + 14 * np.sin(e["ph"] + n * e["sway"])
            y = e["y"]
            flick = 0.6 + 0.4 * float(rng.random())
            g = e["r"] * 3
            d.ellipse([x - g, y - g, x + g, y + g], fill=(255, 90, 30, int(34 * flick)))
            r = e["r"]
            d.ellipse([x - r, y - r, x + r, y + r], fill=(255, 150, 70, int(190 * flick)))

        proc.stdin.write(layer.tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise SystemExit("ffmpeg failed while writing the film overlay")
    return out


# ---------------------------------------------------------------------------
# Sound effects
# ---------------------------------------------------------------------------


def _fir_band(x: np.ndarray, lo: float | None, hi: float | None, taps: int = 513) -> np.ndarray:
    """Windowed-sinc band filter. ``lo``/``hi`` None means no edge on that side."""
    n = np.arange(taps) - (taps - 1) / 2

    def sinc_lp(fc: float) -> np.ndarray:
        h = np.sinc(2 * fc / SR * n) * 2 * fc / SR
        return h * np.hamming(taps)

    if lo is None and hi is None:
        return x
    if lo is None:
        h = sinc_lp(hi)
    elif hi is None:
        h = -sinc_lp(lo)
        h[(taps - 1) // 2] += 1.0
    else:
        h = sinc_lp(hi) - sinc_lp(lo)
    return np.convolve(x, h, mode="same")


def _place(buf: np.ndarray, sig: np.ndarray, t: float, gain: float = 1.0) -> None:
    i = int(t * SR)
    j = min(len(buf), i + len(sig))
    if i < len(buf):
        buf[i:j] += sig[: j - i] * gain


def _thunk(rng: np.random.Generator, f: float, tau: float, length: float = 0.25) -> np.ndarray:
    t = np.arange(int(length * SR)) / SR
    tone = np.sin(2 * np.pi * f * t) * np.exp(-t / tau)
    body = _fir_band(rng.standard_normal(len(t)), None, 600.0) * np.exp(-t / (tau * 0.5))
    return tone + 0.5 * body / (np.abs(body).max() + 1e-9)


def _tick(rng: np.random.Generator, length: float = 0.04) -> np.ndarray:
    t = np.arange(int(length * SR)) / SR
    return _fir_band(rng.standard_normal(len(t)), 1800.0, 6500.0) * np.exp(-t / 0.006)


def _peak(x: np.ndarray) -> np.ndarray:
    return x / (np.abs(x).max() + 1e-9)


def sfx_clack(seed: int = 1) -> np.ndarray:
    """Projector slide change: a thunk, a tick, a paper slide, a settling clack."""
    rng = np.random.default_rng(seed)
    out = np.zeros(int(0.95 * SR))
    _place(out, _thunk(rng, 95.0, 0.045), 0.0, 0.9)
    _place(out, _tick(rng), 0.055, 0.55)
    # The slide moving: band-limited noise with a smooth swell.
    n = int(0.40 * SR)
    sweep = _fir_band(rng.standard_normal(n), 500.0, 2400.0) * np.hanning(n)
    _place(out, sweep / (np.abs(sweep).max() + 1e-9), 0.10, 0.28)
    _place(out, _thunk(rng, 120.0, 0.035), 0.58, 0.6)
    _place(out, _tick(rng), 0.62, 0.4)
    return _peak(out)


def sfx_swish(seed: int = 2) -> np.ndarray:
    """A quick air sweep, for slides and wipes."""
    rng = np.random.default_rng(seed)
    n = int(0.50 * SR)
    noise = rng.standard_normal(n)
    low = _fir_band(noise, 300.0, 1400.0)
    high = _fir_band(noise, 1400.0, 5200.0)
    mix = np.linspace(0.0, 1.0, n)
    sig = (low * (1 - mix) + high * mix) * np.sin(np.linspace(0, np.pi, n)) ** 1.6
    out = np.zeros(int(0.6 * SR))
    _place(out, _peak(sig), 0.0, 1.0)
    return _peak(out)


def sfx_soft(seed: int = 3) -> np.ndarray:
    """A faint low breath, for dissolves and fades."""
    rng = np.random.default_rng(seed)
    n = int(0.55 * SR)
    sig = _fir_band(rng.standard_normal(n), None, 700.0)
    env = np.sin(np.linspace(0, np.pi, n)) ** 2.2
    out = np.zeros(int(0.7 * SR))
    _place(out, _peak(sig * env), 0.0, 0.8)
    _place(out, _thunk(rng, 70.0, 0.05, 0.2), 0.0, 0.35)
    return _peak(out)


def write_wav(path: Path, x: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = (np.clip(x, -1, 1) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())
    return path


SFX_MAKERS = {"clack": sfx_clack, "swish": sfx_swish, "soft": sfx_soft}


def sfx_for(transition: str) -> str:
    """Which sound goes with which picture change."""
    if transition == "proj":
        return "clack"
    if transition in ("fade", "dissolve", "fadeblack"):
        return "soft"
    return "swish"


# ---------------------------------------------------------------------------
# Slide-mount band and chapter cards
# ---------------------------------------------------------------------------


def _font(name: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(FONT_DIR / name), size)


def make_band(out: Path, height: int = 340) -> Path:
    """The cream mount that scrolls between two pictures in the projector change."""
    if out.is_file():
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(11)
    base = np.zeros((height, W, 3), dtype=np.float32)
    base[:] = (233, 222, 196)
    base += rng.normal(0, 5.0, (height, W, 1))  # paper grain
    yy = np.linspace(-1, 1, height)[:, None, None]
    base *= 1.0 - 0.07 * yy**2  # a touch darker at the cut edges
    img = Image.fromarray(np.clip(base, 0, 255).astype("uint8"))
    d = ImageDraw.Draw(img)
    red = (150, 52, 44)
    f1, f2 = _font("georgiab.ttf", 40), _font("georgia.ttf", 30)
    for text, font, y in (("THIS SIDE", f1, 92), ("TOWARD SCREEN", f1, 142), ("TRANSPARENCY", f2, 214)):
        w = d.textlength(text, font=font)
        d.text(((W - w) / 2, y), text, font=font, fill=red)
    d.line([(W / 2 - 120, 196), (W / 2 + 120, 196)], fill=red, width=2)
    img.save(out)
    return out


def _spaced(draw: ImageDraw.ImageDraw, xy: tuple[float, float], text: str,
            font: ImageFont.FreeTypeFont, fill, spacing: float) -> None:
    x, y = xy
    for ch in text:
        draw.text((x, y), ch, font=font, fill=fill)
        x += draw.textlength(ch, font=font) + spacing


def _spaced_width(draw: ImageDraw.ImageDraw, text: str, font, spacing: float) -> float:
    return sum(draw.textlength(ch, font=font) + spacing for ch in text) - spacing


def _wrap_to(draw: ImageDraw.ImageDraw, text: str, font, width: int) -> list[str]:
    lines: list[str] = []
    cur = ""
    for word in text.split():
        trial = f"{cur} {word}".strip()
        if draw.textlength(trial, font=font) <= width or not cur:
            cur = trial
        else:
            lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    return lines


def render_photo(src: Path, out: Path, max_h: int = 900, max_w: int = 1720) -> Path:
    """Compose a supplied screenshot at its own colours on a dark field.

    Web captures stay in RGB so highlighted article text stays readable. The
    surround is a neutral dark field, not a warm grade that would pull the
    page toward the painted shots.
    """
    from PIL import ImageFilter

    out.parent.mkdir(parents=True, exist_ok=True)
    photo = Image.open(src).convert("RGB")
    scale = min(max_h / photo.height, max_w / photo.width, 1.0)
    if scale < 1.0:
        photo = photo.resize(
            (int(photo.width * scale), int(photo.height * scale)), Image.LANCZOS
        )
    photo = photo.filter(ImageFilter.UnsharpMask(radius=1.2, percent=40, threshold=2))

    back = Image.new("RGB", (W, H), (18, 18, 20))
    border = 8
    framed = Image.new("RGB", (photo.width + 2 * border, photo.height + 2 * border), (240, 240, 240))
    framed.paste(photo, (border, border))
    x = (W - framed.width) // 2
    y = (H - framed.height) // 2 + 20
    back.paste(framed, (x, y))
    back.save(out)
    return out


def render_section_hint(section: dict, out: Path) -> tuple[Path, int, int]:
    """A transparent overlay naming the part that is starting.

    Returns the file and the (x, y) where its top-left corner goes in the frame.

    A dark plaque behind gold kicker and white title, so the name stays
    readable on a pale stick-figure picture. Nothing is spoken.
    """

    out.parent.mkdir(parents=True, exist_ok=True)
    kicker, title = section.get("kicker", ""), section.get("title", "")
    gold, cream = (242, 206, 96, 255), (255, 255, 255, 255)
    kf = _font("palab.ttf", 38)
    size = 70
    probe = ImageDraw.Draw(Image.new("RGB", (4, 4)))
    while True:
        tf = _font("palab.ttf", size)
        lines = _wrap_to(probe, title, tf, W - 2 * 280)
        if len(lines) == 1 or size <= 44:
            break
        size -= 2
    if len(lines) > 1:  # balance two lines rather than leave one word alone
        words = title.split()
        best = min(range(1, len(words)), key=lambda c: abs(
            probe.textlength(" ".join(words[:c]), font=tf)
            - probe.textlength(" ".join(words[c:]), font=tf)))
        lines = [" ".join(words[:best]), " ".join(words[best:])]
    line_h = int(size * 1.2)
    top = 56
    kw = _spaced_width(probe, kicker, kf, 11) if kicker else 0
    block_h = (58 if kicker else 0) + len(lines) * line_h
    plaque_pad_x, plaque_pad_y = 72, 36
    plaque = [
        W / 2 - max(kw, max((probe.textlength(ln, font=tf) for ln in lines), default=0)) / 2 - plaque_pad_x,
        top - plaque_pad_y,
        W / 2 + max(kw, max((probe.textlength(ln, font=tf) for ln in lines), default=0)) / 2 + plaque_pad_x,
        top + block_h + plaque_pad_y,
    ]

    text = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(text)
    d.rounded_rectangle(plaque, radius=18, fill=(10, 10, 12, 210))
    d.rounded_rectangle(plaque, radius=18, outline=(242, 206, 96, 230), width=3)
    y = top
    if kicker:
        _spaced(d, ((W - kw) / 2, y), kicker, kf, gold, 11)
        y += 58
    for line in lines:
        w = d.textlength(line, font=tf)
        d.text(((W - w) / 2, y), line, font=tf, fill=cream)
        y += line_h
    for ly in (top - 22, top + block_h + 10):
        d.line([(W / 2 - 140, ly), (W / 2 + 140, ly)], fill=gold, width=2)

    full = text
    # Keep only the lettering's box: a full-frame layer per part is 8 MB a frame
    # and seven of them looping at once exhausted ffmpeg's memory.
    box = full.getchannel("A").point(lambda v: 255 if v > 2 else 0).getbbox() or (0, 0, W, H)
    full.crop(box).save(out)
    return out, box[0], box[1]


def render_card(bg: Path | None, card: dict, out: Path) -> Path:
    """One chapter card, drawn once per chapter."""
    out.parent.mkdir(parents=True, exist_ok=True)
    if bg is not None and bg.is_file():
        img = Image.open(bg).convert("RGB").resize((W, H), Image.LANCZOS)
    else:
        img = Image.new("RGB", (W, H), (110, 22, 18))
    # A dark plaque in the middle keeps the title legible on any backdrop.
    plaque = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    pd = ImageDraw.Draw(plaque)
    x0, y0, x1, y1 = 250, 250, W - 250, H - 250
    pd.rounded_rectangle([x0, y0, x1, y1], radius=26, fill=(38, 8, 6, 175))
    pd.rounded_rectangle([x0 + 14, y0 + 14, x1 - 14, y1 - 14], radius=18,
                         outline=(214, 170, 84, 210), width=3)
    img = Image.alpha_composite(img.convert("RGBA"), plaque).convert("RGB")
    d = ImageDraw.Draw(img)

    gold, cream = (222, 180, 92), (244, 232, 204)
    kicker, title, sub = card.get("kicker", ""), card.get("title", ""), card.get("sub", "")

    size = 104
    while True:
        tf = _font("palab.ttf", size)
        lines = _wrap_to(d, title, tf, W - 2 * 330)
        if len(lines) <= 3 or size <= 56:
            break
        size -= 6
    line_h = int(size * 1.22)
    # Georgia lacks the Vietnamese stacked-diacritic glyphs; Palatino has them.
    kf, sf = _font("palab.ttf", 42), _font("palai.ttf", 46)
    block = 62 + len(lines) * line_h + (70 if sub else 0)
    y = (H - block) // 2
    if kicker:
        kw = _spaced_width(d, kicker, kf, 12)
        _spaced(d, ((W - kw) / 2, y), kicker, kf, gold, 12)
    y += 62
    for line in lines:
        w = d.textlength(line, font=tf)
        # A dark offset under the cream gives the engraved weight.
        d.text(((W - w) / 2 + 3, y + 3), line, font=tf, fill=(20, 4, 2))
        d.text(((W - w) / 2, y), line, font=tf, fill=cream)
        y += line_h
    if sub:
        w = d.textlength(sub, font=sf)
        d.text(((W - w) / 2, y + 12), sub, font=sf, fill=gold)
    img.save(out)
    return out
