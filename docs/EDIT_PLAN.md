# The Edit Plan

The Edit Plan is a complete, renderer-agnostic description of a finished video. The
AI director writes it; the Rule Engine checks it; the FFmpeg renderer and the CapCut
exporter read it. Nothing else passes between those stages.

Get the machine-readable schema and a worked example straight from the code, so
neither can drift from what the validator accepts:

```powershell
aive schema show edit-plan > plan.schema.json
aive schema example > my_plan.json
```

## The three things people get wrong

Read these before writing a plan by hand.

### 1. Subtitle and music times are in **timeline** time. Source ranges are not.

Narration cleanup removes silence and filler, so the two clocks diverge. A word
spoken at 41.2 s in `narration.wav` may land at 33.8 s in the finished video.

- `clips[].source_range` — time inside the **source file**
- `clips[].timeline_start` — time in the **finished video**
- `subtitles[].range` — **timeline**
- `music[].timeline_range` — **timeline**
- `music[].source_offset` — inside the **music file**
- `narration.kept_ranges` — inside the **narration file**

`NarrationTrack.source_to_timeline()` performs the mapping and returns `None` for a
moment that fell in a removed gap. Prefer `aive subtitle build` over doing it
yourself.

### 2. `timeline_start` is optional — usually omit it.

Omit it on every clip and the Rule Engine packs them end to end, subtracting
transition overlap. List the clips in the order you want them; that order is
authoritative. Supply `timeline_start` only to place a clip deliberately, such as an
insert that must land on a beat.

This exists because cumulative float arithmetic across forty clips is exactly the
kind of thing that produces a one-frame gap nobody notices until the render.

### 3. `reason` is mandatory, and it is not decoration.

Every clip states why it is there, in one sentence. It is how a human reviews the
director's judgement, and how the director re-reads its own decisions on a second
pass. A clip with no stated reason is unreviewable, so the model rejects it.

## Structure

```
EditPlan
├── schema_version   "1.0"          pinned; an old build refuses a newer plan
├── project_id       str            [A-Za-z0-9._-]+
├── created_by       str            "claude-code" | "heuristic" | "human"
├── created_at       datetime?
├── output           OutputSpec     aspect_ratio, width, height, fps
├── narration        NarrationTrack?    null for a music-only montage
├── clips            TimelineClip[] at least one; list order is editorial order
├── subtitles        SubtitleCue[]  timeline time
├── music            MusicCue[]
└── notes            str?           overall editorial rationale, for humans
```

Only three fields are required: `project_id`, `created_by`, `clips`. Everything else
has a sensible default. That is deliberate — the smaller the required set, the more
likely a first draft validates.

### TimelineClip

| Field | Type | Notes |
|---|---|---|
| `id` | str | unique in the plan; `[A-Za-z0-9._#-]+` |
| `source` | MediaRef | **relative** to the project root, e.g. `raw/001.mp4` |
| `source_range` | TimeRange | the slice to take, in source time |
| `timeline_start` | float? | omit to pack sequentially |
| `speed` | float | default 1.0; 2.0 halves the timeline duration |
| `transition_in` / `transition_out` | Transition? | null means a hard cut |
| `framing` | FramingSpec? | zoom/crop; reserved, no Phase 7 renderer reads it |
| `mute_source_audio` | bool | **default true** — B-roll audio usually fights the narration |
| `reason` | str | **required**; why this clip, in one sentence |
| `confidence` | float | 0..1 |
| `scene_key` | str? | provenance, e.g. `001#3`, linking back to analysis |
| `beat_index` | int? | which narration beat this illustrates |

### Transition

`kind` is one of `cut`, `fade`, `dissolve`, `crossfade`, `slide_left/right/up/down`,
`zoom_in`, `zoom_out`. `duration` must be `0` for `cut` and greater than `0` for
everything else — the model enforces both directions. Maximum 10 s.

`cut` is a named member rather than `null` so a director can state a hard cut
deliberately instead of by omission.

### MusicCue and ducking

Ducking is modelled as a sidechain compressor (`gain_db`, `threshold_db`, `attack`,
`release`) rather than volume keyframes, because that is what both FFmpeg's
`sidechaincompress` and CapCut's audio engine express natively. Keyframes would have
to be recomputed per exporter and would drift.

`ducking: null` disables it — correct only where no narration plays. `source_offset`
skips a track's ambient intro so the bed lands with the cut instead of fading up over
it. Fades may not exceed the cue duration.

### MediaRef

Always a path **relative to the project root**, serialised POSIX-style so a plan
written on Windows opens on macOS. Absolute paths and `..` traversal are rejected —
this is a security boundary, not a style preference, because the plan is generated
text.

## Worked example

```json
{
  "schema_version": "1.0",
  "project_id": "garden-vlog",
  "created_by": "claude-code",
  "output": { "aspect_ratio": "16:9", "width": 1920, "height": 1080, "fps": 30.0 },

  "narration": {
    "source": { "path": "narration.wav" },
    "kept_ranges": [
      { "start": 0.0, "end": 4.2 },
      { "start": 5.1, "end": 11.6 }
    ],
    "gain_db": 0.0
  },

  "clips": [
    {
      "id": "c001",
      "source": { "path": "raw/001.mp4" },
      "source_range": { "start": 12.5, "end": 18.3 },
      "reason": "Wide shot of the bed being dug matches the opening line about preparing soil.",
      "confidence": 0.9,
      "scene_key": "001#3",
      "beat_index": 0
    },
    {
      "id": "c002",
      "source": { "path": "raw/002.mp4" },
      "source_range": { "start": 3.0, "end": 8.4 },
      "transition_in": { "kind": "dissolve", "duration": 0.4 },
      "reason": "Close-up of watering follows planting, so a dissolve reads as time passing.",
      "confidence": 0.85,
      "scene_key": "002#1",
      "beat_index": 1
    }
  ],

  "subtitles": [
    { "range": { "start": 0.0, "end": 4.2 }, "text": "First, prepare the soil." },
    { "range": { "start": 4.2, "end": 10.7 }, "text": "Then water it well." }
  ],

  "music": [
    {
      "track": { "path": "music/calm_forest.mp3" },
      "timeline_range": { "start": 0.0, "end": 10.7 },
      "source_offset": 8.0,
      "gain_db": -3.0,
      "fade_in": 1.5,
      "fade_out": 2.0,
      "ducking": {
        "gain_db": -12.0, "threshold_db": -30.0, "attack": 0.15, "release": 0.6
      }
    }
  ],

  "notes": "Chronological build: prepare, plant, water. Dissolves between stages, hard cuts within a stage."
}
```

Note what this example demonstrates beyond the syntax. Neither clip sets
`timeline_start`, so the Rule Engine places them. The subtitle for the second beat
starts at **4.2 s** on the timeline even though the narration for it begins at 5.1 s in
the source — the 0.9 s gap was removed. `source_offset: 8.0` skips the music track's
ambient intro.

And the coherence check worth doing by eye: the clips run 5.8 s + 5.4 s minus a 0.4 s
dissolve, so **10.8 s of video** against **10.7 s of cleaned narration**. Picture
outlasting narration by a tenth of a second is fine; the other way round would leave
the last word over black.

## Validating a plan

```powershell
aive rules validate edit_plan.json     # Phase 4 — checks without changing
aive rules normalize edit_plan.json    # Phase 4 — places clips, clamps transitions
```

Validation happens in two independent halves, and it is worth knowing which one
rejected you.

**Pydantic — is it well-formed?** Unique clip ids, `end > start`, a cut with zero
duration, no unknown keys, the right `schema_version`. Fails at parse time with a
`ValidationError`.

**Rule Engine — is it admissible?** Needs config thresholds and real probe data:

| Check | Typical finding |
|---|---|
| clip length | `clip.too_short` — below `rules.min_clip_duration` |
| source bounds | `source.out_of_bounds` — cuts past the file's real duration |
| overlap | `clip.overlap` — two clips claim the same timeline span |
| transition fit | `transition.too_long` — exceeds `max_transition_ratio` of the shorter clip |
| duplicates | `scene.duplicate` — two clips are the same take |
| coverage | `narration.uncovered` — a stretch of narration has no picture |

Findings are `Issue` objects with a stable `code`, a `message`, a `hint` and a
`location` such as `clips[3]`. Errors block a render; warnings do not.

Normalisation reports every change it makes at `INFO` severity. A silent fix-up is
indistinguishable from a bug.

## Reviewing a plan

Admissible is not the same as good, and the two failure modes look nothing alike in a
JSON file.

```powershell
aive plan subtitles edit_plan.json --in-place  # cues into plan.subtitles
aive plan show edit_plan.json                  # pacing, framing, coverage, every reason
aive plan diff before.json edit_plan.json      # what a second pass actually changed
```

`plan show` prints each clip's `reason` and appends `review.*` notes — observations, never
errors, each calibrated against a specific way an automated edit gives itself away
(identical clip lengths, unvarying framing, footage never looked at). See
[AGENT_TOOLS.md](AGENT_TOOLS.md) for the full list.

`plan diff` matches clips **by `id`, not position**, so inserting one clip does not report
every clip after it as modified.

## Ducking

`DuckingSpec` states a **target**: drop the bed by this many dB while narration plays, over
these ramp times. Not compressor settings — and that distinction was earned rather than
assumed.

Phase 7 first rendered ducking with FFmpeg's `sidechaincompress`, on the reasoning that a
compressor is what an audio engine offers natively. Measurement killed it:

| `ratio` | attenuation delivered |
|---|---|
| 3 | 6.96 dB |
| 7 | 8.94 dB |
| 12 | 9.57 dB |
| 20 | 9.91 dB |

It **saturates near 10 dB whatever ratio it is given**, so a plan asking for -12 and a plan
asking for -20 produce the same output and the field is decorative.

A consumer does not need a compressor, because it does not need to *detect* speech.
`narration.kept_ranges` are laid end to end on the timeline, so narration is one contiguous
block from 0 to its `timeline_duration` — the plan already states exactly when someone is
talking. Both the renderer and the exporter derive their envelope from that one field, so
they cannot drift apart, which is the property the compressor framing was meant to protect.

The rendered attenuation is now exactly the requested figure. `threshold_db` is retained for
exporters whose audio engine really is a compressor; the FFmpeg renderer ignores it.

## Versioning

`schema_version` is a `Literal["1.0"]`. A build that predates a new version will
**refuse** the plan rather than misread fields it does not understand — the failure
mode you want, since the alternative is a subtly wrong video.

The other half of that is migration, in `app/models/migrations.py`. Every command reads a
plan through one loader, which brings an older document to the current version *before*
validating it and reports on stderr that it did. Migrations run on raw dictionaries — they
have to, since the models describe the current shape and cannot represent the old one.

To bump the version, add the step and nothing else; the chain is found automatically:

```python
@register("1.0", "1.1")
def _add_colour_grade(document: dict[str, Any]) -> dict[str, Any]:
    for clip in document.get("clips", []):
        clip.setdefault("colour_grade", None)
    return document
```

No migrations are registered today, because only one version exists. The mechanism was
built at the point where there was nothing to migrate on purpose: retrofitting a migration
path after plans exist in the wild means guessing what those plans contain.
