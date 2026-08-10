# The agent tool contract

This is the interface an AI director codes against. It is a contract, not a
suggestion: the rules below are enforced in `app/cli/output.py` and
`app/utils/logging.py`, and tested in `tests/unit/test_cli.py`.

## Three rules

### 1. stdout is machine-readable. stderr is for humans.

Every command writes either a JSON document or a line-oriented digest to stdout, and
nothing else. Logs, progress, tables and summaries go to stderr.

```powershell
aive schema show edit-plan 1> plan.schema.json 2> log.txt
# plan.schema.json is parseable JSON; log.txt holds everything else
```

Parse stdout unconditionally. Never parse stderr; its format is not stable.

### 2. Full data goes to disk. A digest goes to stdout.

Analysis writes its complete result under `<project>/.aive/` and prints a compact
digest. This is a cost decision, not a cosmetic one: a 20-minute project with 200
scenes is roughly **80k tokens as raw JSON and about 12k as a digest**.

Read the digest. Reach for the file, or `--full`, only when you need a specific
detail the digest omits.

```
# manifest=.../manifest.json editable=true clips=2 music=1 templates=0
narration    narration.wav                                  4.2MB
raw_video    raw/001.mp4                                  182.4MB
raw_video    raw/002.mp4                                  241.9MB
music        music/calm_forest.mp3                          3.1MB
```

A digest is whitespace-separated fields, one record per line, with a `#` header
carrying the summary so you never have to count lines.

### 3. Failures are structured.

Non-zero exit, plus an error object on **stdout** — the same stream as success, so
there is one place to look.

```json
{
  "error": {
    "code": "project.not_initialised",
    "message": "D:/work/demo is not an AIVE project (no raw/ directory)",
    "hint": "initialise it with: aive project init D:/work/demo"
  }
}
```

Branch on `code`. Act on `hint`. Both are meant to let you self-correct rather than
retry blindly.

### Exit codes

| Code | Meaning | Recoverable? |
|---|---|---|
| 0 | success | — |
| 2 | usage error — bad arguments | yes, fix the invocation |
| 3 | not found — missing project or file | yes, often via `init` |
| 4 | invalid input — a plan failed validation | yes, rewrite the plan |
| 5 | environment — a dependency is missing | no, tell the user |
| 70 | internal — a bug in AIVE | no, report it |
| 130 | interrupted | — |

`5` and `70` mean stop. Do not retry.

## Global flags

| Flag | Effect |
|---|---|
| `--quiet` / `-q` | silence stderr; stdout unaffected. Useful once you trust a step |
| `--verbose` / `-v` | DEBUG to stderr |
| `--version` | version to stdout, nothing else |

`AIVE_QUIET=1` does the same as `--quiet`. Note that it silences the *console* only —
the JSON log file still records everything.

## Available now (Phases 1-5)

### `aive doctor [PROJECT]`

Environment preflight. Run it first; it is the difference between "the render failed"
and "the render was never going to work".

```json
{
  "ok": true,
  "platform": "Windows 11",
  "aive_version": "0.1.0",
  "checks": [
    { "name": "ffmpeg", "status": "ok", "required": true, "detail": "ffmpeg version 7.1-..." },
    { "name": "cv2", "status": "not installed", "required": false, "detail": "... pip install -e \".[video]\"" }
  ]
}
```

Check `ok`. Exits `5` if a **required** check failed. Optional components reporting
`not installed` are normal — they gate later phases, and their `detail` carries the
exact install command.

### `aive project init DIRECTORY [--name NAME]`

Creates the folder structure. Idempotent and safe to re-run: it only creates what is
missing and never overwrites an existing `aive.toml`, so it also repairs a project
whose folders were partly deleted.

### `aive project scan DIRECTORY [--full]`

Inventories the media and writes `.aive/manifest.json`. Digest on stdout.

Check `editable=` in the header: `false` means the project lacks narration or footage
and there is nothing to plan yet. Exits `3` if the directory is missing or
uninitialised.

### `aive schema show DOCUMENT`

JSON Schema for `edit-plan`, `manifest`, `transcript`, `footage`, or `plan-report`.
Generated from the Pydantic models, so it cannot drift from what the validator
accepts.

**Fetch `edit-plan` before authoring your first plan.** It is the difference between
writing a valid plan once and discovering the format through a sequence of validation
errors.

### `aive schema example`

A minimal but valid Edit Plan, constructed from real models and therefore guaranteed
to validate. Faster to learn from than the schema, because it shows which fields are
genuinely optional and what a plausible `reason` reads like.

### `aive schema list`

The documents that have a schema.

### `aive config show [PROJECT] [--section NAME]`

The merged configuration. Use it to read the thresholds you are planning against —
`rules.min_clip_duration` and `rules.max_clip_duration` in particular, since the Rule
Engine will hold you to them.

### `aive config layers [PROJECT]` · `aive config defaults`

Which config files apply, and the values with no file layer.

### `aive analyze audio PROJECT`

Transcribes the narration, decides what to cut, and splits it into **beats**. Writes
`.aive/narration.json`; digest on stdout.

```
# narration=narration.wav lang=en model=faster-whisper/medium dur=11.57 kept=5.68 beats=4 surviving=3 wordtimings=true doc=...
b000 src=0.00-3.02 tl=0.00-1.90 kw=first,prepare,soil | First, um, prepare the soil.
b001 src=4.06-5.10 tl=CUT       kw=water              | Then water it?
b002 src=5.82-7.22 tl=2.03-3.16 kw=water,well         | Then water it well.
# cuts: silence:0.66-1.17 filler:1.28=um retake:4.06-5.10
```

This digest is the most important thing you read all session. Per line:

| Field | Meaning |
|---|---|
| `src=` | where the beat is in `narration.wav` — for a human to scrub to |
| `tl=` | where it lands in the **finished video**, after cuts. **Place footage against this.** |
| `tl=CUT` | the beat was removed — a retake or a filler. Do not give it footage. |
| `kw=` | keywords, for matching against scene tags |
| after `\|` | the text itself |

`surviving=` in the header is how many beats actually need footage. The `# cuts:` line
explains every removal, so you can tell the user *why* something disappeared.

Options: `--model` (tiny…large-v3), `--language` (skip autodetection — worth it on a
short or noisy recording), `--no-cleanup` (keep every second), `--force` (re-transcribe
rather than reuse the cache), `--full`.

Results are cached and reused unless the audio, the model, or the analyser version
changed. Transcription costs minutes, so do not pass `--force` reflexively.

Exits `3` if there is no narration, `4` if no speech was recognised, `5` if
`faster-whisper` is not installed.

### `aive analyze video PROJECT`

Detects scenes in the raw footage and measures each one. Writes `.aive/footage.json`;
digest on stdout. **The slowest command in AIVE** — it decodes every clip — so results
are cached per clip and adding one new clip re-analyses only that clip.

```
# footage=.../footage.json clips=5 scenes=7 dup_groups=1 suppressed=1 quality_floor=0.45
@ raw/001.mp4 1920x1080 30.00fps 6.00s scenes=3
001#0 0.00-2.00 d=2.0 q=0.52 blur=0.00 br=0.60 ex=1.00 st=1.00 mot=static cam=unknown shot=unknown ppl=0
001#2 4.00-6.00 d=2.0 q=0.96 blur=1.00 br=0.98 ex=0.78 st=1.00 mot=static cam=static shot=unknown ppl=0
@ raw/006.mp4 1920x1080 30.00fps 2.00s scenes=1
006#0 0.00-2.00 d=2.0 q=0.52 blur=0.00 br=0.60 ex=1.00 st=1.00 mot=static cam=unknown shot=unknown ppl=0 DUP
# duplicates: 006#0<-001#0
```

An `@` line introduces a clip; the lines under it are its scenes. The clip path is
written once, which on a 200-scene project saves several thousand tokens.

| Field | Meaning |
|---|---|
| `001#0` | scene key — clip stem, then scene index. Use it in a plan's `scene_key`. |
| `0.00-2.00` | the scene's range **in that source file** |
| `q=` | overall quality, 0–1. Rank alternatives with this. |
| `blur=` `br=` `ex=` `st=` | blur, brightness, exposure, stability — all 0–1, 1 is best |
| `mot=` | static / low / medium / high — drives pacing |
| `cam=` | static, pan, tilt, zoom, handheld, unknown |
| `shot=` | wide / medium / close_up, or `unknown` |
| `ppl=` | faces detected; `?` means not measured |
| `DUP` | **suppressed as a duplicate of a better take. Do not use it.** |
| `LOW` | below `rules.min_overall_quality`. Usable only if nothing else covers the beat. |

`# duplicates:` reads `suppressed<-keeper`. Honour it: cutting between two takes of one
shot is the most obvious sign no human was involved.

Two honest limits worth knowing before you rely on a field:

- **`shot` and `ppl` come from face detection only.** No face means `shot=unknown` and
  `ppl=0`, which is normal for B-roll — not a failure. There is no scale reference in a
  single frame without a subject of known size.
- **`blur` is comparative, not absolute.** It is a normalised Laplacian variance, which
  depends on content as much as focus: a sharp shot of a plain wall scores low. Use it to
  rank alternatives *within a project*, not as a verdict.

Options: `--clip NAME` (analyse one clip, by filename or stem), `--force` (ignore the
cache), `--full`.

Exits `3` if there is no footage, `4` if a file is unreadable, `5` if the `[video]` extra
is not installed.

### `aive plan brief PROJECT`

**The one document to read before authoring a plan.** Joins the narration beats to the
eligible footage, ranks candidate scenes per beat, echoes the constraints the plan must
satisfy, and says whether the project can be covered at all. Writes `.aive/brief.json`.

```
# brief=.../brief.json project=demo feasible=true narration=5.7s beats=4 need_footage=3 footage=13.0s ratio=2.3x scenes=5
# constraints clip=0.8-8.0s transition=dissolve@0.40s max_transition_ratio=0.25 output=1920x1080@30fps aspect=16:9 quality_floor=0.30
B000 tl=0.00-1.90 want=1.9 kw=first,prepare,soil cands=5 | First, um, prepare the soil.
    004#0 raw/004.mp4 0.00-3.00 d=3.0 score=0.57 q=0.97 shot=unknown mot=high | no semantic tags to match against; 3.0s covers the 1.9s beat; good quality (0.97)
B001 tl=CUT want=0.0 kw=water cands=0 | Then water it?
```

`B` lines are beats; the indented lines under each are its candidates.

| Field | Meaning |
|---|---|
| `feasible=` | whether every beat needing picture has an option. `false` exits `4` |
| `ratio=` | usable footage over narration length. Below `1.0x` cannot be covered; `2.0x`+ gives real choice |
| `want=` | how long a clip covering this beat should be |
| `score=` | suitability, 0-1. **Scaffolding, not a verdict** |
| after `\|` | why it scored that way - argue with it |

**What the score knows:** duration fit, quality, shot variety, and whether the shot was
already offered elsewhere. **What it does not know:** whether the picture illustrates the
words. The classical CV provider emits no object tags, so every candidate carries `no
semantic tags to match against`. The shortlist means *"technically suitable and not yet
used"* - the actual matching is yours, and it is the only step in this pipeline that needs
a director.

Options: `--candidates N` (shortlist size), `--full`.

### `aive plan draft PROJECT [-o OUT]`

A **heuristic baseline**, not an edit. One clip per surviving beat, taking each beat's
top-ranked candidate, stretched to cover the pauses between beats so the picture runs
without gaps. Written to `edit_plan.json` unless `-o` says otherwise.

Stamped `created_by: "heuristic"`, and every `reason` says so, because it chose on duration
and quality alone. It exists so you never have to fight the schema, and as a zero-AI
fallback. **Revise the selections and the reasons before rendering.**

The result is deliberately unplaced - run `aive rules normalize` next.

### `aive rules scenes PROJECT`

Which scenes are eligible for selection, and why the rest are not. Run it **before**
authoring a plan: it answers "what may I use?" once, instead of your discovering the
answer through rejected clips.

```
# scenes=6 eligible=4 rejected=2 eligible_duration=14.0
+ 001#2 4.00-6.00 d=2.0 q=0.96 mot=static shot=unknown
- 003#0 0.00-2.00 d=2.0 q=0.42 mot=static shot=unknown quality.below_floor
- 006#0 0.00-2.00 d=2.0 q=0.52 mot=static shot=unknown scene.duplicate
```

`+` is usable, `-` is not, and the trailing codes say why. Rejected scenes are **not
deleted** — coverage beats perfection, so reach for one if nothing else covers a beat. A
gap is worse than a soft shot.

### `aive rules normalize PLAN [-o OUT | --in-place]`

Places clips, clamps what is out of bounds, then validates the result. **Run this before
`validate`** — it fixes much of what validation would reject, and reports each fix.

```
# plan=edit_plan.json ok=true errors=0 warnings=1 normalised=true
i normalize.clamped_source clips[1] c2: source range shortened from 12.00s to 4.00s, the real length of raw/002.mp4
i normalize.clamped_transition clips[1] c2: dissolve shortened from 0.90s to 0.43s
i normalize.placed_clip clips[0] c1: placed at 0.000s
i normalize.placed_clip clips[1] c2: placed at 1.275s
W scene.duplicate_used clips[2] c3 uses 006#0, a suppressed duplicate | use the surviving take instead
```

This is what makes omitting `timeline_start` safe. List clips in order; normalisation packs
them end to end and pulls each one back over its predecessor by exactly the length of its
incoming transition — which is what a dissolve physically is.

It **writes nothing unless asked**: pass `-o PATH` or `--in-place`. And it never invents
editorial intent, so a too-short clip or a duplicate stays for you to decide.

### `aive rules validate PLAN`

Checks a plan and changes nothing. Exits `4` when there are errors, so a script can branch.

```
E source.out_of_bounds clips[1] c2: ends at 12.00s but raw/002.mp4 is only 4.00s long | shorten the range; `aive rules normalize` will clamp it for you
W clip.too_long clips[1] c2: 11.00s on the timeline, above the 8.00s guideline | a long hold can be right
i plan.unplaced clips have no timeline positions, so continuity was not checked | run: aive rules normalize
```

`E` blocks the render, `W` does not, `i` is informational. Everything after `|` is the
hint — act on it rather than guessing.

Both commands take `--project DIR`; it defaults to the plan's own folder, since the
convention is `<project>/edit_plan.json`. Without a project, source-bounds and
file-existence checks are skipped and say so (`source.unverified`) rather than staying
silent.

**The errors worth knowing about**, all of them mistakes that look reasonable when written:

| Code | What went wrong |
|---|---|
| `source.out_of_bounds` | cutting past the real end of a file — needs `analyze video` to detect |
| `clip.too_short` | below `rules.min_clip_duration`; reads as a glitch, not a shot |
| `clip.overlap` | two clips claim the same second, beyond their transition |
| `transition.too_long` | the dissolve leaves the shot no time to read |
| `narration.uncovered` | the video ends on black with someone still talking |
| `subtitle.past_end` | cues timed against narration time instead of timeline time |
| `music.offset_past_end` | `source_offset` past the end of the track — yields silence |
| `scene.duplicate_used` | using a take the analysis suppressed (warning) |

### `aive subtitle build PROJECT`

Builds cues and writes `output/subtitle.srt` and `output/subtitle.ass`.

```
# cues=3 karaoke=false files=.../subtitle.srt,.../subtitle.ass
c000 0.00-1.90 lines=1 | First, prepare the soil.
c001 2.03-3.27 lines=1 | Then water it well.
```

**Use this rather than hand-writing cues into a plan.** It applies the source-to-timeline
mapping, which is the single easiest thing to get wrong, and it drops words that cleanup
removed — so an excised "um" vanishes from the subtitle as well as the audio.

Options: `--format srt|ass` (repeatable), `--plan PLAN` (take timing from a plan's
narration track instead of the cleanup report — use this once you have a plan), `--raw`
(time against the original narration, for review only), `--full`.

### `aive plan subtitles PLAN [-o OUT | --in-place]`

Puts the cues **into the plan**, at `plan.subtitles`.

```
# plan=edit_plan.json cues=3 duration=5.68
c000 0.00-1.90 lines=1 words=0 | First, prepare the soil.
c001 2.03-3.27 lines=1 words=0 | Then water it well.
```

`subtitle build` writes `.srt` and `.ass` files; **this is the one the renderer and the
CapCut exporter read.** A plan without cues renders without subtitles, however many
sidecar files sit next to it.

Cues are timed against the **plan's** `narration.kept_ranges`, not the cleanup report. If
you trimmed the narration further than cleanup did, the plan wins — it describes the video
being made, and the cleanup report describes a video you decided against.

Existing cues are **replaced, not merged**, so running it twice is safe. It writes nothing
unless you pass `-o PATH` or `--in-place`. Exits `3` without a narration analysis, `4` if
every recognised word falls inside a range the plan removed (an empty cue list would look
like success).

### `aive plan show PLAN`

**Read an edit back.** A plan is JSON, and JSON is not how anyone judges an edit.

```
# plan=edit_plan.json project=demo by=claude-code placed=true clips=3 duration=5.68 cuts_per_min=31.7 sources=3 scenes=3 subtitles=3
# pacing median=2.03 range=1.68-2.77 transitions=cut:1,dissolve:2 shots=wide:1,medium:2 longest_same_shot_run=1
# coverage narration=5.66 picture=5.68 delta=+0.02
000 c1 raw/001.mp4 src=0.00-2.00 tl=0.00-2.00 d=2.00 in=cut shot=wide scene=001#0 beat=0 | Wide establishes the bed before any detail
W review.mechanical_rhythm all 6 clips are within 4% of each other in length | a constant shot length reads as a metronome
```

Use it on your own work before declaring it finished, and to answer "what did it decide?".
Every clip's `reason` is on its line — that is the point of the command, and the reason
those fields are mandatory.

The `review.*` notes are **editorial observations, never errors** — `rules validate` owns
admissibility, and this exit code is always `0` on a readable plan. They flag what is
invisible in a list of clips and unmissable in a finished video:

| Code | What it saw |
|---|---|
| `review.mechanical_rhythm` | clip lengths within 15% of each other — computed, not chosen |
| `review.repeated_framing` | three or more consecutive clips sharing a shot type |
| `review.every_cut_is_a_transition` | no hard cuts at all; reads as a slideshow |
| `review.no_transitions` | all hard cuts (informational — often correct) |
| `review.footage_underused` | under a quarter of the usable scenes appear |
| `review.heuristic_plan` | `created_by: heuristic` — this was never actually edited |
| `review.thin_reasons` | a `reason` too short to be one |
| `review.no_provenance` | no clip records a `scene_key`, so nothing links to the analysis |
| `review.narration_uncovered` | the picture ends before the voice does |
| `review.long_tail` | the picture runs well past the narration |

Shot types and footage usage need `.aive/footage.json`; without it the review says so and
reports `shot=unknown` rather than guessing. Takes `--project DIR` and `--full`.

### `aive plan diff BEFORE AFTER`

What changed between two plans.

```
# diff before=before.json after=edit_plan.json identical=false added=0 removed=0 changed=1 reordered=false duration=5.68->7.20(+1.52)
~ c2 [source_range,reason] src=1.00-3.00 reason="filler" -> src=1.00-4.50 reason="holds on the hands…"
# plan_level_changes: subtitles
```

Clips are matched **by `id`, not position**, so inserting one clip does not report every
clip after it as modified. Reordering is reported once, as a single fact. `confidence` and
`created_at` are ignored — they change without the video changing.

Use it on a second pass, or to see exactly what `rules normalize` did to a plan you wrote.

### A note on `schema_version`

Every command that reads a plan runs it through the same loader, which upgrades an older
`schema_version` before validating and tells you on stderr that it did. A plan declaring a
version **newer** than the build understands is refused with exit `4` rather than
half-understood — a partially-read plan renders a confidently wrong video.

### `aive analyze music PROJECT`

Describes the tracks in `music/` so a bed can be chosen deliberately. Writes `.aive/music.json`.

```
# music=.../music.json tracks=2 failed=0 total=44.0
music/beat-120-uplifting.wav d=20.0 bpm=120 energy=0.23 lufs=-31.9 intro=- mood=calm,melancholic tags=beat,uplifting
music/slow-intro-calm.wav d=24.0 bpm=? energy=0.40 lufs=-23.8 intro=6.0 mood=neutral,dramatic tags=slow,intro,calm
```

| Field | Meaning |
|---|---|
| `bpm=` | tempo. **`?` means it declined to answer** — sustained material has no beat to find, and a wrong tempo is worse than none because you would pace cuts to it |
| `energy=` | perceived intensity, 0-1, from level and spectral brightness |
| `lufs=` | integrated loudness, measured by FFmpeg's EBU R128 meter. **The field that matters for placing a bed**: two tracks at the same gain but six LUFS apart sit completely differently under the same narration |
| `intro=` | where the opening quiet section ends. Set `source_offset` to this and the bed lands with the cut instead of fading up over it |
| `mood=` | a ranked *pair*, not a verdict — see below |
| `tags=` | words from the filename. The **user's** words, so trust them over `mood` |

**Nothing here selects music.** `mood` is a decision table over three numbers, and every
threshold is config (`aive config show`). Match the bed to the narration's emotion, not the
footage's — that is an editorial call the digest informs rather than makes.

Options: `--force` (ignore the per-track cache), `--full`. Exits `3` with no `music/` folder,
`4` if nothing could be analysed, `5` if the `[audio]` extra is missing.

### `aive render PLAN [--draft]`

Encodes the plan with FFmpeg. Writes `output/final.mp4`, or `output/draft.mp4` with `--draft`.

```
# render plan=edit_plan.json video=.../output/draft.mp4 draft=true duration=5.68 elapsed=0.5 subtitles=2
sub .../output/draft.srt
sub .../output/draft.ass
```

**Preflight runs first, always**, and exits `4` naming every problem: a missing source, an
unplaced plan, an unwritable destination, burn-in requested with no cues. Discovering a
missing clip forty minutes into an encode is unacceptable.

**Use `--draft` while judging the edit.** Half size and a fast preset, several times quicker,
and it answers the only question a first pass asks. Spend the full encode once.

| Option | Effect |
|---|---|
| `--check` | run preflight and stop. Encodes nothing |
| `--dry-run` | print the filter graph and stop. Useful in a bug report |
| `--subtitles srt\|ass` | write a sidecar. Repeatable |
| `--burn-in` | burn cues into the picture. Needs `aive plan subtitles` first |
| `-o PATH` | somewhere other than `output/` |
| `--project DIR` | defaults to the plan's own folder |

Progress goes to **stderr**; stdout stays parseable. The full FFmpeg log is always kept at
`output/logs/render-{draft,final}.log`, and a failure quotes its last lines.

**What the renderer will not do**, reported as a warning rather than silently dropped:
`FramingSpec` (zoom, crop, Ken Burns) is described in the schema but not applied, and
`zoom_out` has no FFmpeg equivalent so the configured substitute is used.

### `aive export capcut PLAN`

Writes a CapCut draft the user can open and keep editing. This is the handoff for "the edit
is 90% right and I want to fix the last 10% by hand", which is most edits.

```
# export plan=edit_plan.json draft=.../com.lveditor.draft/p5final clips=3 files=2 copied=5 warnings=2 targets_capcut=8.6.0.3667
w clip c000: the 'Dissolve' transition is a labelled placeholder - CapCut resolves transitions from its own library, so pick it again in the UI to make it render
```

The draft is a **directory** containing `draft_content.json`, `draft_meta_info.json` and (by
default) a `media/` copy of every source file. Media is copied because a draft referencing
files elsewhere breaks the moment the user reorganises their footage, and CapCut's failure
mode is a timeline of red placeholders with no explanation.

| Option | Effect |
|---|---|
| `-o PATH` | write here instead of CapCut's own draft folder |
| `--name` | draft name. Defaults to the plan's `project_id` |
| `--template DIR` | inherit fonts, colours and canvas settings from one of the user's own drafts. Only the timeline is replaced |
| `--no-copy` | reference media in place. For a stable layout |
| `--check` | preflight only, writes nothing |

Narration and music land on **separate audio tracks**, because rebalancing voice against
music is the first thing a user does and it is a two-slider job only if they are not
interleaved. Each narration kept-range becomes its own segment, so a single cleanup cut can
be undone without undoing all of them.

**Read the warnings.** Transitions export as *labelled placeholders*: CapCut resolves them
from its own downloadable library by effect id, which AIVE cannot know, so the cut is hard
until the user picks the named transition again in the UI. It is visible and one click to
fix — which is the point of naming it rather than omitting it.

### `aive export targets`

Where drafts would be written, and which CapCut version the exporter targets.

```
# capcut target=8.6.0.3667 candidates=2 found=1
+ CapCut C:/Users/.../AppData/Local/CapCut/User Data/Projects/com.lveditor.draft
- JianYing C:/Users/.../AppData/Local/JianyingPro/User Data/Projects/com.lveditor.draft
```

Run this first when "it exported but I cannot find it" — almost always a second install or a
draft folder that does not exist yet. AIVE will **not** create CapCut's folder: a draft
written where CapCut does not read cannot appear in its project list.

### A caveat on the CapCut format

`draft_content.json` is undocumented, ships as compiled code, and changes between releases.
The exporter targets **8.6.0.3667** and is a best-effort reconstruction. If a draft does not
open, export again with `--template` pointing at one of the user's own working drafts.

### `aive ui [PROJECT]`

Opens the desktop window. Not part of the agent contract — it writes nothing to stdout and
blocks until closed — but listed here so a director asked "is there a GUI?" can answer.

The window drives the same services these commands drive; there is no second implementation
of anything. It is a viewer and a launcher, not a timeline editor.

## The intended workflow

```powershell
aive doctor                              # 1. environment
aive project scan ./project              # 2. what did the user give us?
aive analyze audio ./project             # 3. what is being said?
aive analyze video ./project             # 4. what footage is usable?
aive analyze music ./project             # 5. what music is available?
aive schema show edit-plan               # 6. what shape must a plan be?
aive plan brief ./project                # 6b. beats + candidates + constraints
                                         # 7. decide the edit; write edit_plan.json
aive rules normalize edit_plan.json      # 8. place clips, clamp transitions
aive rules validate edit_plan.json       # 9. confirm it is admissible
aive plan subtitles edit_plan.json       # 10. put the cues in the plan
aive plan show edit_plan.json            # 11. read the edit back before committing to it
aive render edit_plan.json --draft       # 12. cheap render to review
aive render edit_plan.json               # 13. final
aive export capcut edit_plan.json        # 14. hand off for manual polish   [Phase 8]
```

Step 7 is the only one requiring judgement. Everything else is deterministic, which
is the entire point of the architecture.

Step 11 is the one most easily skipped and most worth keeping. `validate` says the plan is
*admissible*; `show` is where you find out it is a metronome of identical two-second clips.
If a second pass follows, `aive plan diff` on the two versions is how you confirm you
changed what you meant to and nothing else.

## Working efficiently

**Read digests, not files.** The digest exists so you do not have to load a
200-scene JSON document to learn that clip 4 is unusably shaky.

**Get the schema before authoring, not after failing.** One `schema show` costs far
less than three validation round-trips.

**Omit `timeline_start`.** Let `rules normalize` place the clips. Cumulative float
arithmetic across forty clips is not a good use of your attention.

**Normalise before validating.** Normalisation can fix what validation would have
rejected, and it reports every change it made.

**Draft-render before final-render.** `--draft` trades quality for speed. Reviewing
an edit does not need a visually lossless encode.

**Say why.** Fill in `reason` on every clip and `notes` on the plan. A human is going
to read these, and so are you on a second pass.
