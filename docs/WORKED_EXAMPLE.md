# A worked example, end to end

One project, from raw files to a validated Edit Plan, with the actual output at each step.
Everything below was run against real media.

The project: a short garden clip. `narration.wav` contains a filler ("um"), a retake, and
several pauses. `raw/` holds three clips, one of which is a repeat of another.

```
project/
    narration.wav
    raw/001.mp4  raw/002.mp4  raw/004.mp4
```

## 1. Is the environment usable?

```powershell
aive doctor
```

Run it first. It resolves FFmpeg (a static build ships in the wheel, so this works on a
clean machine) and reports which optional extras are installed. Exit `5` means stop — a
dependency is missing and nothing downstream will work.

## 2. What did the user give us?

```powershell
aive project scan ./project
```

```
# manifest=.../manifest.json editable=true clips=3 music=0 templates=0
narration   narration.wav                                  0.5MB
raw_video   raw/001.mp4                                    0.0MB
raw_video   raw/002.mp4                                    0.2MB
raw_video   raw/004.mp4                                    0.2MB
```

`editable=true` means there is narration *and* footage. `false` means stop and say so.

## 3. What is being said?

```powershell
aive analyze audio ./project
```

```
# narration=narration.wav lang=en model=faster-whisper/tiny dur=11.57 kept=5.68 beats=4 surviving=3 wordtimings=true
b000 src=0.00-3.02 tl=0.00-1.90 kw=first,prepare,soil | First, um, prepare the soil.
b001 src=4.06-5.10 tl=CUT       kw=water              | Then water it?
b002 src=5.82-7.22 tl=2.03-3.16 kw=water,well         | Then water it well.
b003 src=8.08-10.76 tl=3.31-5.66 kw=finally,add,mulch,around,base | Finally, add some mulch around the base.
# cuts: silence:0.66-1.17 ... filler:1.28=um retake:4.06-5.10
```

Read this carefully — it is the spine of the edit.

* 11.57 s of recording became **5.68 s** of narration. Silence, a filler and a retake went.
* **`b001` is `tl=CUT`.** It was the first, abandoned attempt at "then water it"; the
  corrected take is `b002`. It needs no footage. If the user asks why that line is missing,
  this is the answer.
* `src=` is where a beat sits in the recording; **`tl=` is where it lands in the finished
  video.** Place footage against `tl=`.

## 4. What footage is usable?

```powershell
aive analyze video ./project
```

```
# footage=.../footage.json clips=3 scenes=5 dup_groups=0 suppressed=0 failed=0 quality_floor=0.30
@ raw/001.mp4 640x360 25.00fps 6.00s scenes=3
001#0 0.00-2.00 d=2.0 q=0.52 blur=0.00 br=0.60 ex=1.00 st=1.00 mot=static cam=unknown shot=unknown ppl=0
001#2 4.00-6.00 d=2.0 q=0.96 blur=1.00 br=0.98 ex=0.78 st=1.00 mot=static cam=static  shot=unknown ppl=0
@ raw/004.mp4 640x360 25.00fps 3.00s scenes=1
004#0 0.00-3.00 d=3.0 q=0.97 blur=1.00 br=0.98 ex=1.00 st=0.89 mot=high  cam=pan     shot=unknown ppl=0
```

The slowest command — it decodes every clip — and cached per clip, so adding one clip later
re-analyses only that clip.

Two honest limits show plainly here. `shot=unknown` and `ppl=0` throughout, because the
classical-CV provider reads faces and nothing else; that is normal for B-roll, not a
failure. And `blur=0.00` on the flat-colour scenes is real: blur is a normalised Laplacian
variance, so a shot with no detail scores low whether or not it is in focus.

## 5. Which scenes may I actually use?

```powershell
aive rules scenes ./project
```

```
# scenes=5 eligible=5 rejected=0 eligible_duration=13.0
+ 001#0 0.00-2.00 d=2.0 q=0.52 mot=static shot=unknown
+ 004#0 0.00-3.00 d=3.0 q=0.97 mot=high   shot=unknown
```

`+` usable, `-` not, with the reason codes. A rejected scene is **not deleted** — coverage
beats perfection, so reach for one if nothing else covers a beat.

## 6. Everything needed to plan, in one document

```powershell
aive plan brief ./project
```

```
# brief=.../brief.json project=demo feasible=true narration=5.7s beats=4 need_footage=3
#        footage=13.0s ratio=2.3x scenes=5
# constraints clip=0.8-8.0s transition=dissolve@0.40s max_transition_ratio=0.25
#             output=1920x1080@30fps aspect=16:9 quality_floor=0.30
B000 tl=0.00-1.90 want=1.9 kw=first,prepare,soil cands=5 | First, um, prepare the soil.
    004#0 raw/004.mp4 0.00-3.00 d=3.0 score=0.57 q=0.97 shot=unknown mot=high | no semantic tags to match against; 3.0s covers the 1.9s beat; good quality (0.97)
    001#2 raw/001.mp4 4.00-6.00 d=2.0 score=0.57 q=0.96 shot=unknown mot=static | ...
B001 tl=CUT want=0.0 kw=water cands=0 | Then water it?
B002 tl=2.03-3.16 want=1.1 kw=water,well cands=5 | Then water it well.
    001#2 raw/001.mp4 4.00-6.00 d=2.0 score=0.57 q=0.96 ...
    004#0 raw/004.mp4 0.00-3.00 d=3.0 score=0.28 ... | already offered for an earlier beat
```

This replaces cross-referencing three digests. Note three things:

* **`feasible=true` and `ratio=2.3x`.** There is 2.3× more usable footage than narration, so
  there is room to choose. Below `1.0x` the project cannot be covered at all, and the brief
  exits `4` rather than letting you plan something impossible.
* **The constraints sit beside the choices.** A plan that violates a threshold it was never
  shown would be the tool's failure.
* **Every score is explained**, including `no semantic tags to match against` and
  `already offered for an earlier beat`. The ranking answers *"which shot is technically
  suitable and not yet used"* — it has no idea whether the picture illustrates the words.
  **That judgement is yours.** It is the one part of this pipeline that needs a director.

## 7. Write the plan

This is the only step requiring judgement. Work beat by beat, and for each ask what a
viewer needs to *see* while hearing it. `aive plan draft` gives a structurally valid
baseline to revise if you want a starting point — but it is stamped
`created_by: "heuristic"` and its `reason` fields say so, because it chose on duration and
quality alone.

Omit `timeline_start`: list clips in order and let normalisation place them.

```json
{
  "project_id": "demo",
  "created_by": "claude-code",
  "clips": [
    { "id": "c1", "source": { "path": "raw/004.mp4" },
      "source_range": { "start": 0.2, "end": 2.6 },
      "reason": "Panning shot over the bed establishes the space while the soil is introduced.",
      "scene_key": "004#0", "beat_index": 0 },
    { "id": "c2", "source": { "path": "raw/001.mp4" },
      "source_range": { "start": 4.2, "end": 5.9 },
      "transition_in": { "kind": "dissolve", "duration": 0.4 },
      "reason": "Dissolve marks the move from preparing to watering — a change of stage, not a continuous action.",
      "scene_key": "001#2", "beat_index": 2 }
  ],
  "notes": "Chronological build: prepare, then water, then mulch."
}
```

## 8. Place it, then check it

```powershell
aive rules normalize edit_plan.json --in-place
aive rules validate edit_plan.json
```

```
i normalize.placed_clip clips[0] c1: placed at 0.000s
i normalize.placed_clip clips[1] c2: placed at 2.000s
# plan=edit_plan.json ok=true errors=0 warnings=0
```

**Normalise before validating.** Normalisation fixes most of what validation would reject —
placing clips, clamping an over-long range or transition — and reports every change. What
survives is genuinely yours to decide: a clip too short to read, a duplicate you chose,
narration with no picture over it. Normalisation deliberately never invents editorial
intent.

If a source range ran past the end of a file you would see it clamped here:

```
i normalize.clamped_source clips[1] c2: source range shortened from 12.00s to 4.00s, the real length of raw/002.mp4
```

## 9. Subtitles

```powershell
aive plan subtitles edit_plan.json --in-place
```

```
# plan=edit_plan.json cues=3 duration=5.68
c000 0.00-1.90 lines=1 words=0 | First, prepare the soil.
c001 2.03-3.27 lines=1 words=0 | Then water it well.
c002 3.31-5.66 lines=1 words=0 | Finally, add some mulch around the base.
```

Note what is *absent*: the `um` and the abandoned retake. They were cut from the audio, so
they are cut from the text — and the cues are in timeline time, which is the single easiest
thing to get wrong by hand.

This writes the cues into `plan.subtitles`, which is what the renderer and the CapCut
exporter read. `aive subtitle build ./project` writes `output/subtitle.srt` and
`output/subtitle.ass` instead — useful as sidecars, but a plan without cues in it renders
without subtitles however many `.srt` files sit beside it.

Without `--in-place` nothing is written; the digest above is all you get.

## 10. Read the edit back

```powershell
aive plan show edit_plan.json
```

```
# plan=edit_plan.json project=p5final by=heuristic placed=true clips=3 duration=5.68 cuts_per_min=31.7 sources=3 scenes=3 subtitles=3
# pacing median=2.03 range=1.68-2.77 transitions=cut:1,dissolve:2 shots=unknown:3 longest_same_shot_run=0
# coverage narration=5.68 picture=5.68 delta=+0.00
000 c000 raw/004.mp4 src=0.49-2.51 tl=0.00-2.03 d=2.03 in=cut shot=unknown scene=004#0 beat=0 | Heuristic pick: highest-ranked candidate by duration, quality and variety. No semantic matching - revise this.
001 c001 raw/001.mp4 src=4.16-5.84 tl=1.63-3.31 d=1.68 in=dissolve@0.40 shot=unknown scene=001#2 beat=2 | Heuristic pick: highest-ranked candidate by duration, quality and variety. No semantic matching - revise this.
002 c002 raw/002.mp4 src=0.61-3.39 tl=2.91-5.68 d=2.77 in=dissolve@0.40 shot=unknown scene=002#0 beat=3 | Heuristic pick: highest-ranked candidate by duration, quality and variety. No semantic matching - revise this.
W review.heuristic_plan this plan was generated heuristically, not edited | clips were chosen by duration and quality; nothing reflects what the footage shows. Revise the selections and the reason fields.
```

`rules validate` said this plan was admissible. `plan show` says what it *is* — and the one
warning is the honest verdict: nobody edited this. Every `reason` admits it in the same
words, which is exactly the tell.

Two things worth reading in that output. `shots=unknown:3` is not a bug — the face-only
classical provider reports `unknown` on B-roll, and because it does, the framing note is
correctly suppressed rather than firing on a run of three unknowns it knows nothing about.
And `range=1.68-2.77` around a mean of 2.16 is a 50% spread, so `review.mechanical_rhythm`
stays quiet; a set of three 2.0s clips would have tripped it.

Now revise `c001` — hold on the hands in the soil rather than cutting away, and drop the
dissolve since the action is continuous — and check that you changed what you meant to:

```powershell
aive plan diff edit_plan.json revised.json
```

```
# diff before=edit_plan.json after=revised.json identical=false added=0 removed=0 changed=1 reordered=false duration=5.68->6.64(+0.96)
~ c001 [source_range,transition_in,reason] src=4.16-5.84 t_in=dissolve@0.40s reason="Heuristic pick: highest-ranked candidate…" -> src=4.16-6.40 t_in=none reason="Holds on the hands in the soil while the…"
```

One clip, three fields, and the timeline grew by the 0.96s you added. Nothing else moved.
Compare that with the diff across step 9, where attaching three subtitle cues touched no
clip at all:

```
# diff before=before.json after=edit_plan.json identical=false added=0 removed=0 changed=0 reordered=false duration=5.68->5.68(+0.00)
# plan_level_changes: subtitles
```

That precision is why clips are matched by `id` rather than position. Insert one clip at the
front with positional diffing and every clip after it reads as modified, burying the change
you actually care about.

## 11. Render

```powershell
aive render edit_plan.json --draft --subtitles srt --subtitles ass
```

```
  100.0%  5.6s of 5.7s
# render plan=edit_plan.json video=.../output/draft.mp4 draft=true duration=5.68 elapsed=0.5 subtitles=2
sub .../output/draft.srt
sub .../output/draft.ass
```

The result is 960x540 (half of the plan's 1920x1080), 30 fps, yuv420p, with a 48 kHz AAC
track — and 5.700s against a planned 5.684s, a difference of half a frame that is AAC frame
granularity rather than drift.

Judge the cut on that. When it is right, drop `--draft` for the deliverable. If you only want
to know whether the plan is *renderable*, `--check` runs preflight and encodes nothing:

```
# preflight plan=edit_plan.json ok=true clips=3 duration=5.68 destination=.../output/final.mp4
```

Add music by putting a cue in the plan. `analyze music` told us this bed's intro ends at 6.0s
and it sits at -31.9 LUFS, so `source_offset` skips the intro and `gain_db` places it:

```json
"music": [{
  "track": "music/beat-120-uplifting.wav",
  "timeline_range": { "start": 0.0, "end": 5.68 },
  "source_offset": 0.5, "gain_db": -6.0, "fade_in": 0.8, "fade_out": 1.5,
  "ducking": { "gain_db": -12.0, "threshold_db": -30.0, "attack": 0.15, "release": 0.6 }
}]
```

`--dry-run` shows what that becomes, and the ducking is worth reading:

```
volume='if(lt(t,0.000000),1,if(lt(t,0.150000),1+(0.251189-1)*(t-0.000000)/0.150000,
        if(lt(t,5.683900),0.251189,...)))':eval=frame
```

`0.251189` is exactly -12 dB. The bed ramps down over the 0.15s attack, holds while the
narration runs, and recovers over the 0.6s release. Measured on the rendered file, the
attenuation is 12.000 dB — see [EDIT_PLAN.md](EDIT_PLAN.md#ducking) for why that is a volume
envelope rather than a compressor.

## 12. Export

Phase 8. `aive export capcut` is documented in [AGENT_TOOLS.md](AGENT_TOOLS.md) but not yet
implemented.

---

## The shape of it

```
doctor → scan → analyze audio ─┐
                               ├→ plan brief → [YOU DECIDE] ─┐
              analyze video ───┘                             │
                                                             ▼
                        rules normalize → rules validate → plan subtitles → plan show
                                                                                │
                                            ┌───────────────────────────────────┤
                                            ▼                                   ▼
                                    plan diff (revise)                    render / export capcut
```

Every step except the bracketed one is deterministic and repeatable. That is the whole
design: the reasoning lives outside the tool so it can improve without touching the
tool, and the execution lives inside it so it can be trusted.

`plan show` is the step most easily skipped and the one that most changes the result. It is
also the only feedback loop in the pipeline: what it flags is what you fix, and `plan diff`
confirms the fix was the one you intended.
