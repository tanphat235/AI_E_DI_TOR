# aivideo — a video generated end to end

Nothing here reuses another channel's footage. The script is written from
scratch, the pictures are generated, the voice is synthesised, and the captions
come from the script itself.

Three tools, run in order. Each takes the path to a written script and puts its
output beside that script's project, the same split `scripts/shorts` uses:
**the tool is source and is tracked, what it produces is project data and is not.**

```powershell
.\.venv-sdxl\Scripts\python.exe scripts\aivideo\generate_images.py projects\<name>\scripts\<id>.json
.\.venv\Scripts\python.exe      scripts\aivideo\narrate.py         projects\<name>\scripts\<id>.json
.\.venv\Scripts\python.exe      scripts\aivideo\assemble.py        projects\<name>\scripts\<id>.json
```

`projects/<name>/scripts/<id>.json` therefore writes to `projects/<name>/`:

```
.aive/images/<id>/   generated stills, one per line
.aive/audio/<id>.*   narration mp3 + per-line timings
output/<id>.mp4      the finished video
```

## The script file

One JSON per video. `text` is spoken **and** shown as the caption. A line with
an `image` starts a new picture; the lines after it without one are spoken over
the same picture. So captions stay sentence-sized while a scene holds one frame
for as long as it needs. (An image on every line still works: one shot per line.)

```json
{
  "id": "su_that_vu_tru",
  "voice": "vi-VN-NamMinhNeural",
  "style": "Epic spiritual cinematic illustration, ... soft golden divine light",
  "end_card": {"lines": ["TITLE", "subtitle"], "seconds": 6},
  "lines": [
    {"text": "Sau khi chết, chúng ta sẽ đi về đâu?", "image": "the vast universe..."},
    {"text": "Next sentence, same picture."}
  ]
}
```

- `style` is appended to every scene so the whole video holds one look. Omit it
  to get the older doodle-cartoon style.
- `end_card` holds the last shot for `seconds` after the narration. With
  `lines` it also dims the picture and draws them over it.
- `rate` / `pitch` go to the voice as-is (`"+6%"`, `"-10Hz"`). `max_pause`
  shortens every silence longer than that many seconds down to it -- the
  service leaves ~1 s between sentences, a third of the running time.
- `music` (a path relative to the project) is laid under the voice at
  `music_under_db` dB, measured against the narration rather than a raw gain.

### The map: telling the viewer *where* a scene is

`map.image` is one reference picture of the whole subject (for a cosmology
video: the wheel of six realms, the Pure Land, the mountain). `map.regions`
names its parts in that image's own pixels -- `circle [cx,cy,r]`,
`wedge [cx,cy,r_in,r_out,deg_from,deg_to]`, `ellipse`, `box` or `all`, each with
a `label`. Then per line:

- `"map": "<region>"` instead of `image` makes the shot the map itself, that
  region lit, the rest dimmed, its name on a chip. Map shots are held still: a
  list like "Cõi Trời. Cõi Người. …" lights each realm in turn over one map,
  and a restarting push would stutter.
- `"where": "<region>"` on an `image` line puts a small copy of the map in the
  top-right corner with that region lit, for the whole shot.

`locator.py` draws both. The shapes are declared, not detected -- the map is a
hand-made illustration with a known layout. Check a new map's regions by
rendering them all once and looking before rendering the video.

Rules for the text:

- **Every `text` must end at a sentence boundary** (`.` `?` `!`). The voice
  reports sentence timings only; a caption that ends mid-sentence has no
  timing of its own. A trailing `:` does not count -- join it to the next line.
- **Never ask for text in the picture.** No diffusion model renders Vietnamese
  diacritics; it produces shapes that look like letters and are not. Words are
  drawn by `assemble.py` in Segoe UI Black, which does.
- Captions wrap at about 70 characters a line. A caption that needs a third
  line is drawn in full and reported, never truncated -- split it if it reads
  badly.

## Why the captions are right

Every caption problem in the cutting projects came from running speech
recognition over someone else's recording and getting `niếc bàn`, `bồ tác`,
`lục chung`. Here the text exists **before** the audio, so each caption is the
exact text that was written. No ASR, nothing to hand-correct.

`edge-tts` is asked once for the whole script, not once per line — per-line
requests get rate limited. Note that the Vietnamese voices emit
**SentenceBoundary**, not WordBoundary; asking only for words returns an empty
stream and every line lands at 0.00s.

## The picture model: FLUX.1-schnell on the RTX 5060

The machine now has an **RTX 5060, 8 GB** (Blackwell, `sm_120`), which needs a
**CUDA 12.8+ torch build** (`--index-url https://download.pytorch.org/whl/cu128`;
2.11.0+cu128 measured working). FLUX.1-schnell replaced SSD-1B: it follows
multi-figure scenes far better, its T5 encoder reads ~256 tokens instead of
CLIP's 77, and it is **Apache 2.0**, so the monetised channel may use it (unlike
FLUX.1-dev, rejected below).

It does not fit 8 GB whole (24 GB), so `generate_images.py` runs two phases:
CLIP + T5 encode every prompt **on the CPU** (T5-XXL alone is 9.5 GB; ~37 s a
prompt, cached per prompt hash), then a **Q3_K_S GGUF** transformer from
`city96/FLUX.1-schnell-gguf` denoises on the card.

Q3, not Q4, is a measurement, not a guess:

| transformer | peak VRAM | per step | per image |
|---|---|---|---|
| Q4_K_S, 1360x768 | 7.88 GB | ~90 s | **560 s** |
| Q3_K_S, 1360x768 | 6.96 GB | 3.5 s | **~20 s** |

Q4 does not raise out-of-memory: Windows keeps ~0.6 GB of VRAM for the desktop,
and past the card's limit the driver silently spills into shared system memory
and runs 25x slower. If a render suddenly takes minutes per frame, suspect this
first.

**The model is gated** despite the open licence. Once per machine: accept the
terms at huggingface.co/black-forest-labs/FLUX.1-schnell, then
`.venv-sdxl\Scripts\hf.exe auth login`. The script points `HF_TOKEN_PATH` at the
default token location, because it moves `HF_HOME` to `<repo>/.hf-cache` and the
token would otherwise not be found.

### History: the T1000

The machine's previous GPU was an **NVIDIA T1000**, a TU117. Its fp16 path produces NaN:
measured directly, `latents: nan=True` for SDXL *and* for SD 1.5, which decodes
to a pure black frame. It is not slowness and not a bad prompt — a faster card
with the same fault would also return black.

The cure for that fault is fp32, which doubles the weights, and there the 8 GB
of VRAM runs out. What was measured:

| | result |
|---|---|
| SDXL fp16, GPU | NaN → black frame |
| SD 1.5 fp16, GPU | NaN → black frame |
| SDXL fp32, GPU | will not fit — 10.3 GB UNet against 8 GB |
| SDXL fp32, CPU | 233 s/step → **34.5 h** for one video |
| SD 1.5 fp32, GPU | runs, 31 s/frame, but loses both the style and the scene |
| **SSD-1B fp32, GPU** | **80–113 s/frame, style and scene both right** |

So: **SSD-1B**, SDXL distilled to a 1.3B UNet. 5.2 GB at fp32, which fits, and
it keeps SDXL's prompt adherence.

**On a card with working fp16 this reverts to SDXL base 1.0**, which is better
still: at fp16 SDXL needs about 7 GB and fits in the same 8 GB. Change `MODEL`
and pass `dtype=torch.float16, variant="fp16"`. One trap on Blackwell (an RTX
50-series is `sm_120`): PyTorch must be a **CUDA 12.8 or newer** build, or every
call fails with `no kernel image is available for execution on the device`.

## Licensing, because this channel is monetised

| piece | what is used | licence |
|---|---|---|
| pictures | **FLUX.1-schnell** (Q3_K_S GGUF) | Apache 2.0 — commercial use permitted |
| pictures, fallback | **SDXL base 1.0** | CreativeML OpenRAIL++-M — permitted |
| pictures, on the old T1000 | **SSD-1B** | Apache 2.0 — permitted |
| voice | `edge-tts` | Microsoft's read-aloud service. Its terms cover Edge's own feature; commercial use is a grey area. **Piper TTS (MIT) is the clean fallback** and has Vietnamese voices. |
| music | the project's existing bed | unchanged from the other projects |

SDXL-Turbo, FLUX.1-dev and SD3 were all rejected: fast and good, but their
licences forbid or restrict commercial use, which most tutorials omit.

## Install

`.venv-sdxl` is a **separate** environment, like `.venv-demucs`. The main
`.venv` drives faster-whisper and PyAV for the whole cutting pipeline, and a
CUDA torch install there risks dragging numpy with it.

The model cache is forced to `<repo>/.hf-cache` — C: has 18 GB free and the
models run to tens of GB.
