# Architecture

## The one idea

Everything in AIVE exists to keep **decision-making** separate from **execution**.

The AI director decides *what* the video should be. AIVE executes it. Between them
sits one document — the **Edit Plan** — and neither side knows anything about the
other.

```
                    ┌──────────────────────────┐
                    │   AI Director (Claude)   │   decides
                    │   runs OUTSIDE the app   │
                    └────────────┬─────────────┘
             reads JSON          │          writes a plan
        ┌───────────────────────┴───────────────────────┐
        ▼                                               ▼
┌───────────────────┐                          ┌─────────────────┐
│     ANALYSIS      │                          │    EDIT PLAN    │
│  speech · video   │─────── facts ──────────▶ │  the contract   │
│  audio · quality  │                          │  Pydantic, v1.0 │
└───────────────────┘                          └────────┬────────┘
                                                        │
                                        ┌───────────────┼───────────────┐
                                        ▼               ▼               ▼
                                 ┌────────────┐  ┌────────────┐  ┌────────────┐
                                 │ RULE ENGINE│  │  RENDERER  │  │  EXPORTERS │
                                 │ validate & │  │   FFmpeg   │  │   CapCut   │
                                 │  normalise │  │  → mp4     │  │  → draft   │
                                 └────────────┘  └────────────┘  └────────────┘
```

Why bother? Because the director is the part you will want to change constantly,
and the renderer is the part you need to trust absolutely. Coupling them means every
improvement to editorial judgement risks the encoder, and every encoder fix risks
the edit. Keeping a plan between them means:

- The renderer can be rewritten without teaching it anything about storytelling.
- The director can be replaced — different model, different prompt, or a plain
  heuristic — without touching a single FFmpeg argument.
- A render is reproducible. Same plan plus same media equals the same video, forever.
- A plan is reviewable. A human can read it, and edit it, before anything is encoded.

## Layers

The dependency graph is strictly one-directional. Nothing below depends on anything
above it.

| Layer | Package | Depends on | Never depends on |
|---|---|---|---|
| Primitives | `app/models/common.py` | nothing | anything |
| Probe facts | `app/models/media.py` | common | analysis, plan |
| Analysis results | `app/models/{speech,video,audio}.py` | common, media | each other, plan |
| **The contract** | `app/models/edit_plan.py` | common only | analysis, renderer, exporters |
| Config | `app/config/` | models | analysis, renderer |
| Analysis | `app/analysis/{speech,vision,audio}/` | models, config | plan, renderer, exporters |
| Rule Engine | `app/rule_engine/` | models, config | renderer, exporters |
| Planner support | `app/planner/` | models, config | renderer, exporters |
| Plan tooling | `app/plan/` | models, config | renderer, exporters |
| Renderer | `app/renderer/` | **plan only** | analysis |
| Exporters | `app/exporters/` | **plan only** | analysis, renderer |
| Services | `app/services/` | models, config | analysis, renderer |
| CLI | `app/cli/` | everything | — |
| UI | `app/ui/` | services, models | analysis internals |

The rule worth defending in code review: **`app/models/edit_plan.py` imports only
from `app/models/common.py`.** The moment it imports a transcript or a scene, the
Edit Plan stops being a contract and becomes a coupling.

## Where the intelligence is

The shipped package contains **no language model, no API key, and no network call**.
Verify it:

```powershell
grep -ri "anthropic\|openai\|requests\|httpx" app/    # no hits
```

The director is Claude Code, running outside the app and invoking the CLI. That is
what reconciles the two apparently contradictory project requirements — "the
finished application must not depend on the Claude API" and "use Claude to analyse".
Claude does the reasoning; the API is never a dependency of the artefact.

Consequences worth internalising, because they shaped several designs:

**1. The plan is untrusted input.** It is generated text. So:
- `extra="forbid"` on every model — a typo'd key fails loudly instead of silently
  changing the edit.
- `MediaRef` refuses absolute paths and `..` traversal — a plan cannot name
  `C:/Windows/System32`.
- `schema_version` is a `Literal` — an old build refuses a newer plan rather than
  misreading it.
- The Rule Engine validates *admissibility* against real probe data, because a plan
  can be perfectly well-formed and still cut from 45s of a 30s clip.

**2. Tokens are a real cost.** The director reads analysis output and pays per
token. So analysis writes the full document to `.aive/` and prints a **digest** to
stdout — roughly 12k tokens instead of 80k for a 200-scene project. `--full` is the
opt-in, never the default.

**3. Plans must be easy to author, strict to accept.** `timeline_start` is optional:
list clips in order and the Rule Engine packs them. Asking a language model to sum
forty float durations correctly is how you get a one-frame gap at clip 31.

## Validation, in two halves

This split is load-bearing and easy to get wrong.

| | Pydantic models | Rule Engine |
|---|---|---|
| Question | Is it *well-formed*? | Is it *admissible*? |
| Needs | nothing but the document | a `RuleContext`: thresholds + probe data + duplicate groups |
| Examples | unique clip ids; `end > start`; a cut has zero duration | clip too short to read; source range past the file's real end; two shots are the same take; transition longer than the shot |
| Failure | `ValidationError` at parse time | `Issue` in an `EditPlanReport` |

A plan can pass the first and fail the second. Only the second knows the difference
between valid and good.

## Extension seams

Every boundary is a `typing.Protocol`, so an implementation never inherits from or
imports AIVE, and every component can be tested against a hand-written fake with no
media and no models.

| Seam | Protocol | Default (later phase) | Future |
|---|---|---|---|
| Speech | `SpeechRecognizer` | **faster-whisper** | any recogniser |
| Probing | `MediaProber` | PyAV, or system ffprobe | — |
| Scenes | `SceneDetector` | PySceneDetect | — |
| Quality | `QualityAnalyzer` | classical OpenCV | — |
| **Vision** | `VisionProvider` | **classical CV, no weights** | CLIP, YOLO, multimodal |
| Rules | `PlanRule`, `PlanNormalizer` | **10 rules, 5 normalisers** | custom styles |
| Render | `Renderer` | FFmpeg | NVENC, proxy |
| Export | `Exporter` | CapCut | Premiere XML, Resolve, FCP, EDL |

`VisionProvider` is the one designed hardest for replacement, because it is both the
most expensive and the fastest-moving. Its output — `SceneTags` — carries a
`provider` field, so a project can mix a classical baseline over forty clips with a
richer provider over the six that matter, and nothing downstream has to care which
produced a given tag. Providers must degrade *honestly*: a provider that can count
faces but not name objects returns `people_count` set and `objects` empty. An empty
collection reads as "unknown"; a fabricated tag is worse than no tag.

Exporters additionally go through a `ExporterRegistry` keyed by name. CapCut's
`draft_content.json` is undocumented and changes between versions, so isolation
matters: a format change is repaired in one file, and a new exporter is added without
editing anything that already works.

## Cross-cutting rules

**stdout is a machine contract.** JSON or a digest on stdout; every log, table and
progress bar on stderr. Enforced by `app/cli/output.py` (the only module permitted to
`print`) and `app/utils/logging.py` (Rich console pinned to stderr, `propagate=False`
so nothing escapes to the root logger). Tested in
`tests/unit/test_cli.py::TestStdoutDiscipline`, including in a real subprocess.

**No hardcoded values.** Every threshold is a field on a settings model, merged from
six layers (defaults → packaged TOML → site → project → env → CLI). `extra="forbid"`
here too, so a misspelled key in a config file is an error rather than a setting that
never applies.

**Analysis results are immutable.** Every model is `frozen=True`. Transformations
return new objects, which is what makes the Rule Engine's normalise step auditable:
you always hold the before and the after.

**Errors are structured.** Non-zero exit plus `{"error": {"code", "message",
"hint"}}` on stdout. Stable `code` slugs so the director can branch; a `hint` so it
can self-correct. Exit codes are API: `2` usage, `3` not found, `4` invalid input,
`5` environment, `70` internal.

## Project folder

```
project/
    narration.wav      or audio/voice.wav — both layouts supported
    raw/               footage; name order is treated as shooting order
    music/             user-supplied library
    capcut/            template projects (identified by draft_content.json)
    output/            final.mp4, subtitles, logs/, preview/
    .aive/             analysis cache — regenerable, gitignored
        manifest.json  transcript.json  footage.json  keyframes/  analysis/
    aive.toml          per-project config overrides
    edit_plan.json     the plan
```

`.aive/` versus `output/`: everything in `.aive/` is a derived *fact* about the
inputs and is invalidated by `analyzer_version` and file mtime; everything in
`output/` is a *deliverable*. Both are reproducible, neither belongs in git.

## Phase map

Phase 1 built the skeleton and the contracts. Later phases fill in implementations
behind protocols that already exist, which is the point of having built it in this
order.

| Phase | Adds | Behind which seam |
|---|---|---|
| ~~2~~ | **done** - faster-whisper, cleanup, beats, SRT/ASS | `SpeechRecognizer`, `SilenceDetector`, `SubtitleWriter` |
| ~~3~~ | **done** - probing, scenes, quality, motion, faces, dedup | `MediaProber`, `SceneDetector`, `QualityAnalyzer`, `VisionProvider`, `DuplicateDetector` |
| ~~4~~ | **done** - 3 scene filters, 10 plan rules, 5 normalisers | `SceneFilter`, `PlanRule`, `PlanNormalizer` |
| ~~5~~ | **done** - candidate ranking, planning brief, heuristic baseline | `app/planner/` |
| ~~6~~ | **done** - review, diff, cue attachment, schema migration | `app/plan/`, `app/models/migrations.py` |
| ~~7~~ | **done** - filter graph, music analysis, exact ducking | `Renderer`, `MusicAnalyzer`, `LoudnessMeter` |
| ~~8~~ | **done** - draft_content.json, media copy, template clone | `Exporter` |
| ~~9~~ | **done** - step catalogue, worker threads, plan table | `app/ui/` |
| ~~10~~ | **done** - wheel, extras, CI, manifest tests | — |

The future features in the brief — auto B-roll, face tracking, auto zoom, 9:16
repurposing, multi-language subtitles, viral shorts — need no schema change. Zoom and
crop already have a home in `FramingSpec`; `beat_index` and `scene_key` already carry
the provenance a B-roll inserter needs.

## See also

- [EDIT_PLAN.md](EDIT_PLAN.md) — the contract in detail, with a worked example
- [AGENT_TOOLS.md](AGENT_TOOLS.md) — the CLI contract
- [../CLAUDE.md](../CLAUDE.md) — the director's operating manual
