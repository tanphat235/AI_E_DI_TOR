"""Generate one still per script line with SDXL base 1.0.

Run with .venv-sdxl, not the main .venv: CUDA torch is kept out of the
environment that drives faster-whisper and PyAV for the cutting pipeline.

SSD-1B, not SDXL base 1.0, and fp32, not fp16. Both are forced by the card.

The T1000 is a TU117 and its fp16 path makes NaN: SDXL and SD 1.5 alike decode
to a pure black frame, measured, latents already NaN before the VAE. So every
model here must run fp32 -- and SDXL's fp32 UNet is 10.3 GB against 8 GB of
VRAM, while on the CPU it measured 233 s/step, 34 h for one video.

SSD-1B is SDXL distilled to a 1.3B UNet: 5.2 GB at fp32, which fits, and it
keeps SDXL's prompt adherence. Apache 2.0, so the channel may monetise it.
Measured 80-113 s per frame at 1024x576. Do not "optimise" this back to fp16.

The style anchor and the style lock come from the user's own specification and
are wrapped around every scene description, so the whole video holds one look.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
# C: has ~18 GB free and the models run to tens of GB, so the cache goes on D:
# before torch or huggingface_hub is imported and reads the variable.
os.environ.setdefault("HF_HOME", str(REPO / ".hf-cache"))

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

MODEL = "segmind/SSD-1B"
# True 16:9 at SDXL's scale. 1344x768 is 1.75 and left a visible seam of
# duplicated sky at the edges.
WIDTH, HEIGHT = 1024, 576
STEPS = 28
GUIDANCE = 7.0

ANCHOR = (
    "Hand-drawn 2D doodle cartoon animation, flat solid colors, bold black "
    "hand-drawn outlines, slightly wobbly imperfect marker lines, "
)
# Short on purpose. CLIP stops at 77 tokens and truncates the *tail*, so a long
# style suffix is silently dropped and the frame loses the look -- the first run
# lost exactly this. Everything phrased as "no X" lives in NEGATIVE instead,
# where it costs nothing from the prompt's budget.
LOCK = ", simple educational explainer doodle"
NEGATIVE = (
    "photorealistic, photograph, 3d render, gradient, drop shadow, texture, "
    "shading, cross-hatching, realistic face, anime, manga, watermark, "
    "signature, lettering, blurry, cluttered, duplicated moon, two suns"
)
# CLIP's own limit. Past it the scene itself starts getting cut.
TOKEN_BUDGET = 77


def build_prompt(scene: str) -> str:
    return f"{ANCHOR}{scene.strip().rstrip('.')}{LOCK}"


def work_dir(script_path: str) -> Path:
    """Where this video's artefacts live: the project holding the script.

    The tool sits in `scripts/`, which is source and is tracked; everything it
    produces belongs beside its own script under `projects/<name>/`, which is
    project data and is not. So `projects/x/scripts/y.json` writes to
    `projects/x/`, the same convention `build_shorts.py --work-dir` follows.
    """
    return Path(script_path).resolve().parents[1]


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: generate_images.py <script.json> [--limit N]")
        return 2
    doc = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    limit = 0
    if "--limit" in sys.argv:
        limit = int(sys.argv[sys.argv.index("--limit") + 1])

    out_dir = work_dir(sys.argv[1]) / ".aive" / "images" / doc["id"]
    out_dir.mkdir(parents=True, exist_ok=True)

    import numpy as np
    import torch
    from diffusers import StableDiffusionXLPipeline

    if not torch.cuda.is_available():
        print("no CUDA device; on the CPU this measured 233 s/step")
        return 1
    free, total = (x / 2**30 for x in torch.cuda.mem_get_info())
    print(f"{torch.cuda.get_device_name(0)}  {free:.1f} of {total:.1f} GB free")
    print(f"cache {os.environ['HF_HOME']}")

    pipe = StableDiffusionXLPipeline.from_pretrained(
        MODEL, dtype=torch.float32, use_safetensors=True
    )
    # The two text encoders plus the UNet do not fit at fp32 together; offloading
    # each stage once it has run leaves the UNet's 5.2 GB resident, which does.
    pipe.enable_model_cpu_offload()
    # diffusers 0.40 moved this onto the VAE itself; the pipeline-level
    # enable_vae_slicing() was removed and raises AttributeError.
    pipe.vae.enable_slicing()
    pipe.set_progress_bar_config(disable=True)
    tokenizer = pipe.tokenizer

    lines = doc["lines"][:limit] if limit else doc["lines"]
    for i, line in enumerate(lines, 1):
        prompt = build_prompt(line["image"])
        # The filename carries the prompt's hash, so editing a scene
        # regenerates only that frame and the rest are reused.
        n_tok = len(tokenizer(prompt).input_ids)
        if n_tok > TOKEN_BUDGET:
            print(f"  [{i:2d}] WARNING {n_tok} tokens; the last "
                  f"{n_tok - TOKEN_BUDGET} are dropped -- shorten this scene")
        tag = hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:8]
        dst = out_dir / f"{i:03d}_{tag}.png"
        if dst.is_file():
            print(f"  [{i:2d}/{len(lines)}] reuse {dst.name}")
            continue
        for old in out_dir.glob(f"{i:03d}_*.png"):
            old.unlink()
        gen = torch.Generator(device="cuda").manual_seed(1000 + i)
        t0 = time.time()
        image = pipe(
            prompt=prompt,
            negative_prompt=NEGATIVE,
            width=WIDTH,
            height=HEIGHT,
            num_inference_steps=STEPS,
            guidance_scale=GUIDANCE,
            generator=gen,
        ).images[0]
        # A flat frame means the decode overflowed, not that the prompt was
        # dull. Say so instead of writing it and letting it reach the edit.
        spread = float(np.asarray(image, dtype=np.float32).std())
        if spread < 2.0:
            print(f"  [{i:2d}/{len(lines)}] BLANK (std {spread:.2f}) -- VAE overflow")
            return 1
        image.save(dst)
        print(
            f"  [{i:2d}/{len(lines)}] {dst.name}  {time.time() - t0:5.1f}s"
            f"  {line['text'][:40]}"
        )

    print(f"\n{len(list(out_dir.glob('*.png')))} images in {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
