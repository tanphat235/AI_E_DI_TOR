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

One JSON per video. `text` is spoken **and** shown as the caption; `image` is
the scene given to the picture model.

```json
{
  "id": "lich_su_loai_nguoi",
  "voice": "vi-VN-NamMinhNeural",
  "lines": [{"text": "Loài người đã có mặt...", "image": "a small planet earth..."}]
}
```

Two rules for `image`, both learned the hard way:

- **Never ask for text in the picture.** No diffusion model renders Vietnamese
  diacritics; it produces shapes that look like letters and are not. Words are
  drawn by `assemble.py` in Segoe UI Black, which does.
- **Keep it under 77 tokens with the style anchor.** CLIP truncates the *tail*,
  so an over-long scene silently loses the style lock and the frame stops
  matching the rest. `generate_images.py` counts and warns.

## Why the captions are right

Every caption problem in the cutting projects came from running speech
recognition over someone else's recording and getting `niếc bàn`, `bồ tác`,
`lục chung`. Here the text exists **before** the audio, so each caption is the
exact text that was written. No ASR, nothing to hand-correct.

`edge-tts` is asked once for the whole script, not once per line — per-line
requests get rate limited. Note that the Vietnamese voices emit
**SentenceBoundary**, not WordBoundary; asking only for words returns an empty
stream and every line lands at 0.00s.

## The picture model, and why it is not SDXL

This machine's GPU is an **NVIDIA T1000**, a TU117. Its fp16 path produces NaN:
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
| pictures | **SSD-1B** | Apache 2.0 — commercial use permitted |
| pictures, on better hardware | **SDXL base 1.0** | CreativeML OpenRAIL++-M — permitted |
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
