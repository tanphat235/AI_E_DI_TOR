# scripts/shorts — building vertical shorts from a long talk

Order matters. Each step writes into `<project>/.aive/` and the next one reads it.

```powershell
# 1. transcript (per-project wrapper sets the source and model)
.\.venv\Scripts\python.exe projects\<job>\transcribe.py

# 2. crop — ALWAYS do this before rendering. See "Crop the furniture out" below.
.\.venv\Scripts\python.exe scripts\shorts\find_crop.py `
    --source <video> --aspect 1080:960 --out-dir <scratch>

# 3. optional: locate a channel's transition sting, if the upload has one
.\.venv\Scripts\python.exe scripts\shorts\find_stings.py `
    --work-dir projects\<job> --template "<start>,<end>"

# 4. cut points
.\.venv\Scripts\python.exe scripts\shorts\segment_qa.py --work-dir projects\<job>

# 5. render
.\.venv\Scripts\python.exe projects\<job>\build_shorts.py
```

## Crop the furniture out — a standing rule

Every re-upload carries channel furniture baked into the picture: a watermark in
a corner, a decorative pillarbox, a scrolling promo banner, a title card. **None
of it belongs in a short.** Set `--src-crop w:h:x:y` on every job so the top half
is the live picture only.

Pick the crop at the target aspect so the top half is a pure resize and never a
second crop: for a 1080x1920 output the top half is 1080x960, i.e. **1.125:1**.
`810:720`, `698:620` and `788:668` are all that ratio.

What the three finished jobs use, and what each removes:

| job | crop | keeps | removes |
|---|---|---|---|
| `giesu-ducphat` | `788:668:242:0` | 57% | purple "PHÁP THOẠI VẤN ĐÁP" pillarbox both sides, scrolling promo banner below y=672 |
| `kheo-them-kheo-bot` A | `698:620:341:0` | 47% | "Vấn Đáp / THẦY THÍCH PHÁP HOÀ" overlay at top-left, "phapthoaithichphaphoa" watermark below y=620 |
| `kheo-them-kheo-bot` B | `698:620:366:0` | 47% | same overlay; different centring because the second talk frames the speaker further right |
| `song-tot` | `810:720:150:0` | 63% | "Vấn Đáp / THẦY THÍCH PHÁP HOÀ" watermark from x=960 |

Also check the speaker stays inside the crop for the **whole** span, not just one
frame — cameras zoom and pan. Preview the crop at several timestamps before
committing: `song-tot` was checked at 9 points across the talk, each
`kheo-them-kheo-bot` scene at 5. On the earlier `thienchua-quyy` job the head
moved from x≈900 to x≈1030 between two sampled minutes, which is why that crop
is anchored to the right edge rather than centred. `kheo-them-kheo-bot` sits at
x≈690 in one talk and x≈715 in the other — two crops, two render passes.

## find_crop.py reports, it does not decide

It measures temporal variance and edge energy per block and flags blocks that
are unusually still **and** sharp — the signature of baked-on text. It prints a
suggested crop and writes a four-panel montage (frame, variance, edges, mask).

**Read the montage; do not take the suggestion on faith.** Measured on the two
sources still to hand:

- `song-tot`: 12 of 13 flagged blocks landed exactly on the real watermark
  (x 960-1200, y 40-120). Good detection. But one stray false positive in the
  top band blocked every larger window, so the suggestion kept 50% where the
  right answer keeps 63%.
- `kheo-them-kheo-bot`: 42 and 32 flagged blocks, mostly scattered false
  positives, and the suggestions kept 19% and 24% against a right answer of
  47%. This recording is softer and lower contrast (median block variance 6.5
  and 4.2 against song-tot's 9.1), so ordinary scene detail falls under the
  threshold.

Two earlier signal designs failed outright and are documented in the module
docstring: absolute near-zero variance finds only fully opaque overlays (the
song-tot watermark is semi-transparent and measured std 3.22 against a plain
wall at 10.85), and a local variance ratio missed a second watermark on the same
frame entirely (1.33x the median, i.e. above it).

The honest summary: a fixed-camera talk is full of still, sharp, legitimate
scene content that looks exactly like an overlay to any cheap test. Use the tool
to find candidates fast, then set the crop yourself from the montage.

## The other scripts

- `transcribe.py` — faster-whisper to `.aive/transcript.json`. `--model medium`
  is worth it when cut points depend on reading the text: it put punctuation on
  85% of segments against `small`'s 21%.
- `find_stings.py` — finds a channel transition sting by waveform
  cross-correlation, writes `.aive/stings.json`. Requires ≥2 matches, because a
  template matching only itself is a one-off sound, not an insert.
- `segment_qa.py` — cut points from acoustic pauses, never a fixed offset.
  `--pauses vad` when an energy gate cannot be tuned for the recording.
- `build_shorts.py` — the renderer. Splits, crops, flips, stacks B-roll below,
  lays a music bed. Gain for that bed is **per recording, not per track**: it
  follows the talk's own level and noise floor.

## The centre layout — the standard for new projects

The layout the four finished projects use ("half": talk on top, scenes below)
is still the default, and nothing about it has changed. New projects should use
`--layout center` instead, which reproduces the approved reference
`projects/tui-tu-tui-nhan/shorts/clip_001.mp4`:

```
+----------------------------+
|        scene video         |
|   TITLE, TWO-COLOUR CAPS   |   yellow line 1, orange line 2
+----------------------------+
|                            |
|      the original talk     |   full width, its own aspect ratio
|                            |
+----------------------------+
|   caption of the speech    |   yellow, from the transcript
|        scene video         |
+----------------------------+
```

The recipe, with the numbers that were measured rather than chosen:

```powershell
.\.venv\Scripts\python.exe scripts\shorts\build_shorts.py `
  --source <talk.mp4> --work-dir projects\<job> --broll-dir projects\stock-pexels\raw `
  --segments-file projects\<job>\.aive\qa_segments.json `
  --src-crop <w:h:x:y> `           # measure it first; see the standing rule above
  --layout center `
  --flip top `                     # mirrors the talk only; text is drawn after
  --voice-clarity --voice-pitch 0.94 `
  --music projects\music-background\"phat phap cung tieng mo.mp4" `
  --music-duck off --music-compress --music-dip-db -5 --music-fade 1.0 `
  --music-window flattest --music-under-db 14 `
  --broll-bed --broll-seed 11 --broll-chunk 12 --broll-spread golden
```

**The title comes from `thumb_text` on each segment**, the same clickbait line
the thumbnail uses, so write those before rendering (see the `set_headlines`
pattern in the project folders). Without it the title falls back to the
transcript's first sentence, which rarely reads as a hook.

**Captions need `.aive/transcript.json`.** Pass `--no-captions` for a clip with
no spoken caption.

### Why each audio flag is there

`--music-under-db 14` rather than `--music-db`. A raw gain is not portable
between recordings: the same `-20` sat 16 dB under the voice on song-tot and
20 dB under it on this source, because the two voices are at different levels.
`--music-under-db` measures the treated speech and the shaped bed and solves
for the gain, and it prints both numbers so the result is checkable.

`--music-window flattest` picks the steadiest stretch of the track for each
clip's own length. Left on the spread offsets, the bed's level wandered 7.6 dB
inside one 26 s clip; the flattest window brought that to 2.8 dB.

`--music-compress --music-dip-db -5` are the two halves of "sits under the
voice without muddying it": compression holds one level (the track has struck
bells in it and swings on its own), and the dip at 450 Hz moves the bed out of
the band the voice and the room tone share, which cut its 200-800 Hz share
from 47.8% to 30.8%.

`--voice-clarity` is a tone change, not a level change. Measured net of the
chain's own gain: -7.0 dB below 80 Hz, -0.8 dB at 250-500, +3.7 dB at 2-4 kHz,
+2.2 dB at 5-8 kHz. The last one is not asked for and comes from the
compressor lifting quiet air; it has not been a problem but it is there.

`--voice-pitch 0.94` lowers the voice about six percent, tempo restored, so
the duration is unchanged -- measured drift is under 5 ms on a 90 s sample.

This number was settled by the user's ear over three passes, and the history is
the point: **0.95** (-0.80 semitones) was asked for first and judged not deep
enough; **0.92** (-1.38) was the answer to "deeper" and came back as *trầm
quá*; **0.94** (-1.06) sits between them and is what new clips use. Measured on
90 s of the song-tot talk, F0 189.7 Hz untreated. Do not "restore" 0.95 or
0.92 -- both have already been rejected.

Going much below 0.90 is a separate problem rather than a matter of taste:
`asetrate` shifts the formants along with the pitch, so past roughly ten
percent the voice stops sounding deeper and starts sounding slowed.

### Joining spans with a transition

`--part-transition dissolve --part-transition-sec 0.5` cross-fades between the
spans of a multi-part clip instead of hard-cutting. Use it when the spans come
from different places in the talk; keep `cut` inside one continuous answer,
where a dissolve would say "time passed" about a sentence that never stopped.

**Each junction shortens the clip by the transition length**, because xfade and
acrossfade consume it from both sides. Three spans totalling 155.45 s with two
0.5 s dissolves finish at 154.45 s. Every other clock is told through
`overlap_for()`: the caption times, the music fade-out, and the `-t` on the
B-roll. Without that correction the captions ran 1.00 s past the end of this
clip -- half a second per junction, and it compounds.

**`fps` must be the last filter before xfade.** With `setpts=PTS-STARTPTS`
after it, xfade refuses the input with "the inputs needs to be a constant frame
rate; current rate of 1/0 is invalid" -- setpts clears the frame-rate metadata
that fps had just set, and the whole render dies with a bare `-22 Invalid
argument` from the encoder threads that says nothing about the cause.

### Three traps this cost, worth knowing before touching the audio chain

1. **`acompressor`'s `makeup` is a linear factor, not dB.** `makeup=2` is
   +6 dB and drove the finished mix to -1.1 dBFS. 1.6 keeps the speech within
   0.7 dB of where it started with 3 dB of headroom left.
2. **`asetrate` needs the real sample rate.** `asetrate=48000*0.95` on a
   44.1 kHz source raises the rate, and `atempo` then removes more: a 26.000 s
   clip came out 23.893 s. Put `aresample` in front of it.
3. **`aformat=channel_layouts=stereo` on a mono input costs exactly 3.01 dB**
   -- swresample rematrixes to preserve total energy. Use
   `pan=stereo|c0=c0|c1=c0` when the level matters.

### Text style

All of it is measured off the reference's pixels and lives in
`layout_center.py`'s `CentreStyle`, with each measurement recorded beside the
value. Two things there are not preferences:

* **The font must be Segoe UI Black** (`seguibl.ttf`). It is the only heavy
  face on this machine that draws Vietnamese -- Arial Black, Montserrat and
  Oswald all render tofu boxes for the u-horn and the circumflex-tilde stack.
* **The title's break and size are measured, not estimated.** `fit_title`
  draws each word once through ffmpeg and sums the real widths, then picks the
  break. The em estimate that `wrap` uses for captions is 0.58 em/char on one
  string and 0.67 on another -- a 10% spread, enough to decide whether a line
  overflows. Two rules decide the break, in order: no line may begin with a
  one- or two-character word, because Vietnamese two-syllable compounds get
  split otherwise ("CÁCH GIÚP NGƯỜI ĐỔ / VỠ TRONG HÔN NHÂN" fits the frame and
  is still wrong); then the most even split. If nothing clean fits at
  `--title-size`, the size steps down in fours -- that title landed at 74 px as
  "CÁCH GIÚP NGƯỜI ĐỔ VỠ / TRONG HÔN NHÂN". As a check, the approved
  reference's own title reproduces exactly at 78 px:
  "TUI TU, TUI NHẪN / CHỨ TUI ĐÂU CÓ NGU !".
* **Lines are positioned individually, not with `line_spacing`.** This face
  declares a 2.67 em default line height, so two 54 px lines came out 144 px
  apart, and `line_spacing` is applied at double weight on top of that. The
  scrim is a separate `drawbox` for the same reason: `drawtext`'s own `box=1`
  is sized from that 2.67 em, which made a two-line caption plate half again
  too tall.
