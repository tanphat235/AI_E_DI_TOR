"""Generate one still per shot with FLUX.1-schnell.

Run with .venv-sdxl, not the main .venv: CUDA torch is kept out of the
environment that drives faster-whisper and PyAV for the cutting pipeline.

A shot is a script line that carries an ``image``; the lines after it without
one are spoken over the same picture. So a scene can have short, sentence-level
captions and still hold one frame for as long as the scene needs.

FLUX.1-schnell, on an RTX 5060 (8 GB, Blackwell sm_120). Chosen over SDXL for
prompt adherence on multi-figure scenes and because its T5 encoder reads ~256
tokens, where SDXL's CLIP silently truncates the tail at 77. Apache 2.0, so the
channel may monetise it -- unlike FLUX.1-dev, which the README rejects.

The full model is 24 GB, so it runs in two phases that never share the card:

1. Encode every prompt with CLIP + T5 on the **CPU** (T5-XXL alone is 9.5 GB,
   more than the whole card) and cache the embeddings per prompt hash.
2. Free the encoders, load the transformer as a Q3_K_S GGUF (~5.2 GB) and
   denoise from the cached embeddings.

The previous card, a T1000, made NaN in fp16 and forced SSD-1B at fp32; see the
README for that history. Blackwell needs a CUDA 12.8+ torch build.
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
# C: is the small drive and the models run to tens of GB; set before torch or
# huggingface_hub is imported and reads the variable.
os.environ.setdefault("HF_HOME", str(REPO / ".hf-cache"))
# FLUX.1-schnell is gated. `hf auth login` writes the token under the *default*
# HF home, and moving HF_HOME would otherwise make huggingface_hub look for it
# beside the cache instead and report the repo as inaccessible.
os.environ.setdefault("HF_TOKEN_PATH", str(Path.home() / ".cache" / "huggingface" / "token"))

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

MODEL = "black-forest-labs/FLUX.1-schnell"
GGUF_REPO = "city96/FLUX.1-schnell-gguf"
# Q3_K_S, not Q4_K_S. Q4 peaked at 7.88 GB with the transformer on the card,
# and Windows holds ~0.6 GB for the desktop, so the driver silently spilled into
# shared system memory instead of raising OOM: 90 s/step, 560 s per image.
# Q3_K_S peaks at 6.96 GB and fits: 3.5 s/step, about 20 s per image, measured
# at 1360x768 on the RTX 5060.
GGUF_FILE = "flux1-schnell-Q3_K_S.gguf"
# 16:9 in multiples of 16, about one megapixel -- FLUX's native scale.
WIDTH, HEIGHT = 1360, 768
# schnell is distilled for 1-4 steps and ignores guidance.
STEPS = 4
MAX_SEQ = 256

# Only what the two phases load. The repo root also holds a 23 GB single-file
# checkpoint and the transformer weights, which the GGUF replaces.
ALLOW = [
    "model_index.json", "scheduler/*", "tokenizer/*", "tokenizer_2/*",
    "text_encoder/*", "text_encoder_2/*", "vae/*", "transformer/config.json",
]

# Used only when a script sets no "style". FLUX takes no negative prompt, so
# every constraint has to be phrased as what *is* wanted.
DEFAULT_STYLE = (
    "hand-drawn 2D doodle cartoon, flat solid colors, bold black wobbly marker "
    "outlines, simple educational explainer doodle"
)


def build_prompt(scene: str, style: str) -> str:
    return f"{scene.strip().rstrip('.')}. {style.strip().rstrip('.')}."


def prompt_tag(prompt: str) -> str:
    """The picture's filename. The hash alone, not a shot number: editing one
    scene redraws only that picture, and adding or removing a shot elsewhere
    renumbers nothing, so every other picture is reused."""
    return hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:8]


def image_path(doc: dict, work: Path, scene: str) -> Path:
    """Where the picture for ``scene`` is (or will be) written."""
    tag = prompt_tag(build_prompt(scene, doc.get("style") or DEFAULT_STYLE))
    return work / ".aive" / "images" / doc["id"] / f"{tag}.png"


def work_dir(script_path: str) -> Path:
    """Where this video's artefacts live: the project holding the script.

    `projects/x/scripts/y.json` writes to `projects/x/`, the same convention
    `build_shorts.py --work-dir` follows.
    """
    return Path(script_path).resolve().parents[1]


def shots(doc: dict) -> list[str]:
    """The scene description of every line that starts a new picture."""
    return [line["image"] for line in doc["lines"] if line.get("image")]


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: generate_images.py <script.json> [--limit N]")
        return 2
    doc = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else 0
    style = doc.get("style") or DEFAULT_STYLE

    work = work_dir(sys.argv[1])
    out_dir = work / ".aive" / "images" / doc["id"]
    emb_dir = work / ".aive" / "prompt_embeds"
    out_dir.mkdir(parents=True, exist_ok=True)
    emb_dir.mkdir(parents=True, exist_ok=True)

    scenes = shots(doc)
    if limit:
        scenes = scenes[:limit]
    prompts = [build_prompt(s, style) for s in scenes]
    tags = [prompt_tag(p) for p in prompts]
    # Files from before the name was the hash alone were "<n>_<tag>.png"; take
    # them over rather than paying ~20 s each to draw the same picture again.
    for old in out_dir.glob("[0-9][0-9][0-9]_*.png"):
        new = out_dir / f"{old.stem.split('_', 1)[1]}.png"
        if not new.exists():
            old.rename(new)
    todo = [i for i, t in enumerate(tags) if not (out_dir / f"{t}.png").is_file()]
    print(f"{len(scenes)} shots, {len(scenes) - len(todo)} reused, {len(todo)} to render")
    if not todo:
        return 0

    import numpy as np
    import torch
    from diffusers import FluxPipeline, FluxTransformer2DModel, GGUFQuantizationConfig
    from huggingface_hub import hf_hub_download, snapshot_download

    if not torch.cuda.is_available():
        print("no CUDA device")
        return 1
    free, total = (x / 2**30 for x in torch.cuda.mem_get_info())
    print(f"{torch.cuda.get_device_name(0)}  {free:.1f} of {total:.1f} GB free")
    print(f"cache {os.environ['HF_HOME']}")

    base = snapshot_download(MODEL, allow_patterns=ALLOW)

    # Phase 1: prompt embeddings on the CPU, cached so a re-run skips T5.
    need = [i for i in todo if not (emb_dir / f"{tags[i]}.pt").is_file()]
    if need:
        t0 = time.time()
        enc = FluxPipeline.from_pretrained(
            base, transformer=None, vae=None, torch_dtype=torch.bfloat16
        )
        for n, i in enumerate(need, 1):
            with torch.no_grad():
                pe, ppe, _ = enc.encode_prompt(
                    prompt=prompts[i], prompt_2=None, device="cpu",
                    num_images_per_prompt=1, max_sequence_length=MAX_SEQ,
                )
            torch.save({"pe": pe, "ppe": ppe}, emb_dir / f"{tags[i]}.pt")
            print(f"  encode {n}/{len(need)}")
        del enc
        gc.collect()
        print(f"encoded {len(need)} prompts on CPU in {time.time() - t0:.0f}s")

    # Phase 2: the quantised transformer on the GPU, no text encoders loaded.
    gguf = hf_hub_download(GGUF_REPO, GGUF_FILE)
    transformer = FluxTransformer2DModel.from_single_file(
        gguf,
        quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
        torch_dtype=torch.bfloat16,
        config=base,
        subfolder="transformer",
    )
    pipe = FluxPipeline.from_pretrained(
        base, transformer=transformer, text_encoder=None, text_encoder_2=None,
        tokenizer=None, tokenizer_2=None, torch_dtype=torch.bfloat16,
    )
    # Resident on the card, not enable_model_cpu_offload(): at Q3 everything
    # fits, and offloading only adds a 5 GB PCIe transfer per image.
    pipe.to("cuda")
    pipe.vae.enable_tiling()
    pipe.set_progress_bar_config(disable=True)

    for i in todo:
        dst = out_dir / f"{tags[i]}.png"
        emb = torch.load(emb_dir / f"{tags[i]}.pt")
        gen = torch.Generator(device="cpu").manual_seed(1000 + i + 1)
        t0 = time.time()
        image = pipe(
            prompt_embeds=emb["pe"].to("cuda", torch.bfloat16),
            pooled_prompt_embeds=emb["ppe"].to("cuda", torch.bfloat16),
            width=WIDTH, height=HEIGHT, num_inference_steps=STEPS,
            guidance_scale=0.0, max_sequence_length=MAX_SEQ, generator=gen,
        ).images[0]
        # A flat frame means the decode overflowed, not that the prompt was
        # dull. Stop instead of letting it reach the edit.
        if float(np.asarray(image, dtype=np.float32).std()) < 2.0:
            print(f"  [{i + 1:3d}] BLANK frame -- decode overflow")
            return 1
        image.save(dst)
        print(f"  [{i + 1:3d}/{len(scenes)}] {dst.name}  {time.time() - t0:5.1f}s  {scenes[i][:50]}")

    print(f"\n{len(list(out_dir.glob('*.png')))} images in {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
