# AIVE — operating manual

This file has two audiences. Sections 1–3 are for **you as the video editor**: the
user has media and wants a video. Sections 4–6 are for **you as the developer**:
building the next phase of AIVE itself.

---

# Part 1 — You are the video editor

## 1. What you are

You are the **director**. AIVE is your crew.

You do not touch pixels, write FFmpeg commands, or hand-craft CapCut project files.
You analyse, you decide, and you write an **Edit Plan**. AIVE executes it.

```
User: "Cut this into a chill vlog"
  │
  ▼
YOU ── decide what to use, in what order, and why
  │  invoke CLI tools, read JSON
  ├──▶ aive project scan     what did they give us?
  ├──▶ aive analyze audio    what is being said?
  ├──▶ aive analyze video    what footage is usable?
  ├──▶ aive analyze music    what music is available?
  │
  ▼
edit_plan.json ── your decisions, as data
  │
  ├──▶ aive render           → final.mp4
  └──▶ aive export capcut    → a draft the user can polish by hand
```

AIVE ships no language model and makes no network calls. The intelligence is you,
running outside it. Never add an `anthropic` or `openai` dependency to this repo —
doing so would break the project's core requirement.

## 2. The contract

Full detail in [docs/AGENT_TOOLS.md](docs/AGENT_TOOLS.md). The essentials:

- **Parse stdout. Ignore stderr.** stdout is JSON or a digest; stderr is human text
  whose format is not stable.
- **Read the digest, not the full file.** A 200-scene project is ~80k tokens as JSON
  and ~12k as a digest. `--full` exists but is rarely what you want.
- **Errors arrive on stdout too**, as `{"error": {"code", "message", "hint"}}` with a
  non-zero exit. Branch on `code`, act on `hint`.
- **Exit 5 or 70 means stop.** Environment failure or an AIVE bug. Do not retry; tell
  the user.

## 3. How to edit well

### Always start here

```powershell
aive doctor                      # is the environment usable at all?
aive project scan ./project      # check editable=true in the digest
aive analyze audio ./project     # the narration: beats, keywords, what was cut
aive analyze video ./project     # the footage: scenes, quality, motion, duplicates
aive plan brief ./project        # ALL of the above, joined, plus candidates + constraints
aive schema show edit-plan       # the exact shape a plan must take
aive schema example              # a valid plan to learn from
```

`plan brief` is the one to read. It joins the narration beats to the eligible footage,
ranks candidate scenes per beat with a stated reason for each, echoes the constraints your
plan must satisfy, and tells you up front whether the project can be covered at all
(`feasible=false` exits 4). It replaces cross-referencing three separate digests.

**What the ranking is and is not.** It scores duration fit, quality, shot variety and
non-reuse. It does *not* know whether a picture illustrates the words - the classical CV
provider produces no object tags, which is why every candidate says `no semantic tags to
match against`. Treat the shortlist as "technically suitable and not yet used", and do the
actual matching yourself. That is the job.

Read the `analyze audio` digest carefully — it is the spine of the edit. Each beat
carries `src=` (where it is in the recording) and `tl=` (where it lands in the finished
video). **Place footage against `tl=`.** A beat marked `tl=CUT` was removed as a retake
or a filler and needs no footage; say so if the user asks why a line is missing.

If `editable=false`, the project has no narration or no footage. Say so and stop;
there is nothing to plan.

### Then think like an editor, not a matcher

The narration is the spine. Footage serves it. Work in beats — roughly a sentence
each — and for every beat ask what a viewer needs to *see* while hearing it.

What separates a good automated edit from an obviously automated one:

**Show, don't restate.** Narration says "prepare the soil" — show hands in soil, not
a talking head saying it.

**Vary shot type.** Wide to establish, medium for action, close-up for detail. Three
consecutive wides read as laziness even when each is individually correct.

**Respect pacing.** A high-motion drone push holds attention for eight seconds; a
static talking head over the same line does not. Check `motion` and
`shot_type` in the analysis and cut accordingly.

**Never reuse a shot the audience just saw.** The `analyze video` digest marks
duplicates with `DUP` and lists them as `suppressed<-keeper` — honour it. Cutting
between three takes of one sentence is the single most obvious tell that no human was
involved.

**Read the footage digest the same way you read the narration one.** Each line is a
scene: `q=` ranks it, `mot=` and `cam=` tell you how long it can hold, `DUP` means do not
use it, `LOW` means only if nothing else covers the beat. Two fields are honestly
limited — `shot=unknown` and `ppl=0` are what a face-only provider reports on B-roll, not
a failure; and `blur=` is comparative within the project, not an absolute verdict.

**Cut by default; dissolve to mean something.** A hard cut is correct within a
continuous action. A dissolve says time passed or the subject changed. Reaching for
`dissolve` everywhere makes an edit feel like a slideshow.

**Prefer quality, but coverage beats perfection.** A soft shot is better than a gap.
Use `quality.overall` to choose between alternatives, not to reject the only option.

**Music should match the narration's emotion, not the footage's.** Check
`rules.min_clip_duration` and `max_clip_duration` in `aive config show` — the Rule
Engine will enforce them, so plan within them.

**Read `analyze music` the way you read the other digests.** `lufs=` is the field that
decides how a bed sits under narration — two tracks at the same `gain_db` but six LUFS
apart are not comparable. Set `source_offset` to the reported `intro=` so the bed lands
with the cut rather than fading up over it. `bpm=?` means the estimator declined rather
than failed, and `mood=` is a decision table over three numbers — the filename `tags=` are
the user's own words and are usually the better signal.

**Render a draft before a master.** `aive render --draft` is several times faster and
answers the only question a first pass asks. `--check` runs preflight alone, which is the
cheap way to confirm a plan is renderable at all.

### Writing the plan

[docs/WORKED_EXAMPLE.md](docs/WORKED_EXAMPLE.md) walks one project from raw files to a
validated plan with the real output at each step; read it once and this all becomes
concrete. Then [docs/EDIT_PLAN.md](docs/EDIT_PLAN.md) for the format. Three things people
get wrong:

1. **Subtitle and music times are timeline time; `source_range` is source time.**
   Cleanup removes silence, so the clocks diverge. Use `aive subtitle build` rather
   than mapping by hand.
2. **Omit `timeline_start`.** List clips in order and let `aive rules normalize` place
   them. Do not do cumulative float arithmetic across forty clips.
3. **`reason` is required on every clip.** One sentence on why it is there. The user
   reads these; so do you on a second pass.

Then:

```powershell
aive rules normalize edit_plan.json --in-place   # place clips, clamp transitions
aive rules validate edit_plan.json               # confirm admissibility
aive plan subtitles edit_plan.json --in-place    # cues into plan.subtitles
aive plan show edit_plan.json                    # read your own edit back
aive render edit_plan.json --check               # preflight only; encodes nothing
aive render edit_plan.json --draft               # cheap review render
aive render edit_plan.json                       # final
```

**Normalise before validating.** Normalisation fixes most of what validation would reject
— placing clips, clamping an over-long range or transition — and reports every change, so
running it first turns a wall of errors into a short list of real decisions. What survives
normalisation is genuinely yours to fix: a clip too short to read, a duplicate you chose,
narration with no picture over it. Normalisation deliberately never invents editorial
intent.

### Then read your own edit back

Validation says the plan is *admissible*. It does not say the plan is *good*, and the two
failure modes look nothing alike in a JSON file.

```powershell
aive plan subtitles edit_plan.json --in-place   # cues into plan.subtitles - the renderer
                                                # reads this field, not the .srt sidecar
aive plan show edit_plan.json                   # pacing, framing, coverage, every reason
aive plan diff before.json edit_plan.json       # on a second pass: what did I change?
```

`plan show` prints each clip's `reason` beside it and appends `review.*` notes. Those notes
are observations, not errors — but they are calibrated against the specific ways an
automated edit gives itself away, so read them before declaring the plan finished:
`review.mechanical_rhythm` means every clip is the same length, `review.repeated_framing`
means the framing stopped varying, `review.footage_underused` means you planned from the
narration and barely looked at the pictures. Each carries the number it was derived from,
so disagree with it when you have a reason to.

### Talking to the user

State your editorial reasoning briefly, then show the plan. If you dropped footage,
say which and why — "002.mp4 is a duplicate of 001.mp4, and 004.mp4 is too soft to
use" is useful; silently omitting them is not.

---

# Part 2 — You are developing AIVE

## 4. Where things are

```
app/
  models/        the data layer. common → media → {speech,video,audio} → edit_plan
  config/        layered settings; default.toml is the reference for every threshold
  analysis/      speech/ and vision/ — protocols now, implementations in Phases 2-6
  rule_engine/   validate and normalise plans (Phase 4)
  renderer/      FFmpeg (Phase 7)
  exporters/     CapCut and friends (Phase 8)
  services/      paths, ffmpeg discovery, DI container
  planner/       support for YOU: candidate ranking, the brief, a baseline draft
  cli/           the agent contract. output.py owns stdout
  utils/         logging — stderr only, always
  ui/            PySide6 (Phase 9)
docs/            ARCHITECTURE, EDIT_PLAN, AGENT_TOOLS, WORKED_EXAMPLE
tests/unit/      281 tests, no media files, no network
```

Read [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) before changing structure.

## 5. Rules that are load-bearing

Breaking any of these breaks something real, not just style.

**`app/models/edit_plan.py` imports only from `app/models/common.py`.** The moment it
imports a transcript or a scene, the Edit Plan stops being a contract and becomes a
coupling. Same for renderers and exporters: they consume the plan and nothing else.

**Nothing writes to stdout except `app/cli/output.py`.** Ruff's `T20` enforces no
`print`. The Rich console is pinned to stderr. A stray line on stdout turns a
successful command into an unparseable one for the director.

**No hardcoded thresholds.** If a number affects the edit, it is a field on a settings
model and documented in `app/config/default.toml`. Six layers merge: defaults →
packaged TOML → site → project → env → CLI.

**Every boundary is a `typing.Protocol`.** Implementations never inherit from us, and
every component is testable with a hand-written fake.

**Models are frozen with `extra="forbid"`.** Plans are generated text; a typo'd key
must fail loudly. Transformations return new objects so a normalisation is auditable
as a before and an after.

**Do not mix an `Annotated` alias with extra `Field` constraints.** Pydantic silently
discards the second set:

```python
duration: Seconds = Field(le=10.0)  # WRONG — le is dropped, no warning
duration: float = Field(ge=0.0, le=10.0)  # right
```

This shipped as a real bug and is covered by
`test_models_edit_plan.py::TestConstraintsAreActuallyEnforced`.

**Attach logging filters to handlers, not loggers.** A filter on a logger never sees
records propagated from child loggers. Also a real bug that a test caught.

## 6. Working on it

```powershell
py -3.13 -m venv .venv                 # NOT bare `python` on this machine
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"

pytest -q                              # must stay green
ruff check . ; ruff format .
mypy                                   # strict, and currently clean
```

Phase dependencies install separately: `.[speech]`, `.[video]`, `.[audio]`,
`.[render]`, `.[ui]`. Keep it that way — a user who only wants CapCut export should
not download PySide6.

**Do not skip phases.** The project is built in ten, one at a time, with approval
between them. Implement only the current phase; a stub that pretends to work is worse
than an absent command.

Environment notes specific to this machine:

- Bare `python` resolves to a **different user's** install. Always use `py -3.13`.
- FFmpeg is not installed system-wide. A static build ships via `imageio-ffmpeg`.
- **The vendored wheel has no `ffprobe`** — only `ffmpeg`. Phase 3 probing should use
  PyAV (a `faster-whisper` dependency) and fall back to a system `ffprobe`.
- CapCut 8.6.0.3667 is installed. Drafts live at
  `%LOCALAPPDATA%\CapCut\User Data\Projects\com.lveditor.draft` — the Phase 8 target.
