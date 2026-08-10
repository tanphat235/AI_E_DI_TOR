# AIVE — AI Auto Video Editor

A local, offline video editor that cuts raw footage against a narration track.
You supply the narration, the raw clips and a music folder; AIVE produces either a
finished `final.mp4` or a CapCut project you can keep editing by hand.

## The idea

AIVE is a **toolbox, not an autonomous application**. It ships no language model,
makes no network calls, and needs no API key.

The intelligence comes from an AI director agent — in practice, Claude Code —
running *outside* the app. The agent invokes AIVE's CLI, reads the analysis it
produces, decides the edit, and writes an **Edit Plan**. AIVE then renders that
plan deterministically.

```
You: "Cut this into a chill vlog"
  │
  ▼
Claude Code  ── the director: decides what to use, in what order, and why
  │  invokes CLI tools, reads JSON
  ├──▶ aive analyze audio     faster-whisper  →  transcript, word timings
  ├──▶ aive analyze video     OpenCV + PySceneDetect  →  scenes, quality, motion
  ├──▶ aive analyze music     PyAV + FFmpeg  →  tempo, energy, EBU R128 loudness
  │
  ▼
EDIT PLAN  ── the contract: strongly typed, versioned, renderer-agnostic
  │
  ├──▶ aive render          FFmpeg   →  final.mp4 + subtitles
  └──▶ aive export capcut            →  a CapCut draft you can open and tweak
```

Why the split? Because the director is the part you want to swap, and the
renderer is the part you want to trust. Keeping an Edit Plan between them means
the reasoning can improve without touching a single FFmpeg argument, and the
renderer can be rewritten without teaching it anything about storytelling.

## Status

**Complete — all 10 phases. v1.0.0.**

| Phase | Scope | State |
|---|---|---|
| 1 | Architecture, data models, config, CLI skeleton | done |
| 2 | Speech recognition, cleanup, subtitles | done |
| 3 | Video and scene analysis | done |
| 4 | Rule Engine | done |
| 5 | AI Planner support (brief, candidates, baseline) | done |
| 6 | Edit Plan tooling (review, diff, cues, migration) | done |
| 7 | FFmpeg renderer, music analysis, ducking | done |
| 8 | CapCut exporter | done |
| 9 | PySide6 desktop UI | done |
| 10 | Packaging and docs | done |

Working today: `aive doctor`, `aive project init`, `aive project scan`,
`aive analyze audio/video/music`,
`aive plan brief/draft/show/diff/subtitles`,
`aive rules validate/normalize/scenes`, `aive subtitle build`, `aive render`,
`aive export capcut/targets`, `aive schema`, `aive config`, `aive ui`.
Every documented command is implemented.

Raw media goes in and a finished video comes out:

```powershell
aive project scan ./project
aive analyze audio ./project     # transcribe, cut silence/fillers/retakes, find beats
aive analyze video ./project     # detect scenes, score quality, find duplicate takes
aive analyze music ./project     # tempo, energy, loudness, where each intro ends
aive plan brief ./project        # beats + candidate scenes + constraints, in one doc
                                 # ... you (or Claude) write edit_plan.json ...
aive rules normalize edit_plan.json --in-place   # place clips, clamp what is out of bounds
aive rules validate edit_plan.json              # confirm it is admissible
aive plan subtitles edit_plan.json --in-place   # put the cues in the plan itself
aive plan show edit_plan.json                   # read the edit back: pacing, framing, reasons
aive render edit_plan.json --draft              # fast review encode
aive render edit_plan.json                      # the deliverable
aive export capcut edit_plan.json               # ...or hand it back for manual polish
```

Or drive the same pipeline from a window:

```powershell
aive ui ./project     # or the `aive-ui` shortcut
```

The window is a **viewer and a launcher**, not a timeline editor: it runs the steps, shows
the plan with every clip's reason beside it, and surfaces the same review notes
`aive plan show` prints. Editing belongs to the director, or to CapCut after an export.
Long steps run on a worker thread, so the window keeps repainting through a render.

The pipeline is end to end. What remains is packaging (Phase 10).

## Install

Requires Python 3.12 or newer. On this machine, pin the interpreter with `py` —
bare `python` may resolve to a different installation.

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"

aive doctor          # verifies the ffmpeg binary resolves
```

FFmpeg needs no separate install: a static binary ships with `imageio-ffmpeg`. A
system FFmpeg on `PATH` is used in preference when present.

Media dependencies are grouped per phase so you install only what you use:

```powershell
pip install -e ".[speech]"   # faster-whisper
pip install -e ".[video]"    # opencv, scenedetect, numpy
pip install -e ".[audio]"    # numpy + PyAV (music analysis)
pip install -e ".[ui]"       # PySide6
pip install -e ".[all]"      # everything
```

Rendering and CapCut export need **no** Python library: the renderer builds an FFmpeg
command line itself, and the subtitle writers are hand-rolled. A core install can already
render, given footage someone else analysed.

## Project layout

```
project/
    narration.wav        your voiceover
    raw/                 raw footage: 001.mp4, 002.mp4, ...
    music/               background music library
    capcut/              optional CapCut template project
    output/              renders, subtitles, logs
    .aive/               analysis cache (regenerable, not for git)
    aive.toml            per-project config overrides
```

Create one with `aive project init ./project`.

## Documentation

- [docs/WORKED_EXAMPLE.md](docs/WORKED_EXAMPLE.md) — one project end to end, with real output
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — layers, boundaries, and why they sit where they do
- [docs/EDIT_PLAN.md](docs/EDIT_PLAN.md) — the Edit Plan format, with a worked example
- [docs/AGENT_TOOLS.md](docs/AGENT_TOOLS.md) — the CLI contract an AI director codes against
- [CLAUDE.md](CLAUDE.md) — operating manual for the director agent
- [CHANGELOG.md](CHANGELOG.md) — what shipped, and which decisions measurement overturned

## Design rules

These are load-bearing, not stylistic:

1. **Everything depends on the Edit Plan.** No analyser knows a renderer exists.
2. **The AI never generates FFmpeg commands or CapCut files.** It generates a plan.
3. **stdout is a machine contract.** Tool output is JSON; all logs go to stderr.
4. **No hardcoded thresholds.** Every tunable lives in `config/default.toml`.
5. **Analysis results are immutable values.** Transformations return new objects,
   so a normalisation is always inspectable as a before and an after.

## Licence

MIT.
