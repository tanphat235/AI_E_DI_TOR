# Changelog

## 1.0.0

The whole pipeline: raw media in, a finished video or an editable CapCut draft out, driven
either from a terminal or from a desktop window.

Built in ten phases. Rather than list every module, this records the decisions that were
**changed by evidence** — the places where the first design was wrong and measurement said
so — because those are the ones a reader will otherwise re-make.

### Corrected by measurement

**Ducking is a volume envelope, not a compressor.** `DuckingSpec` states a target
attenuation in dB. The renderer originally used FFmpeg's `sidechaincompress`, on the
reasoning that a compressor is what an audio engine offers natively. Measured, it saturates
near 10 dB whatever ratio it is given (6.96 dB at 3:1, 9.91 dB at 20:1), so a plan asking
for −12 and a plan asking for −20 produced the same output and the field was decorative. It
is now a `volume` envelope derived from `narration.kept_ranges`, which delivers exactly the
requested figure — verified at 12.000 dB. See `docs/EDIT_PLAN.md`.

**Tempo detection declines to answer.** The onset envelope divided by a tiny epsilon before
taking a log, which amplified FFT leakage into a signal; a pure sine wave was reported at
86 BPM with 0.96 confidence. The spectrum is now floored 80 dB below its own peak, and a
crest-factor gate rejects material with no beat — sustained tones measure 2–3, percussion
20–30. A wrong tempo is worse than none, because cuts get paced to it.

**The CapCut exporter lays segments contiguously.** The Edit Plan applies transition overlap
itself, so its clips overlap on the timeline; CapCut stores contiguous segments and applies
the overlap when a transition is marked `is_overlap`. Exporting plan positions directly put
two segments at the same instant on one track, which a track cannot represent.

**The shipped wheel was missing the CapCut exporter.** `.gitignore` carried `**/capcut/*` to
keep users' template folders out of the repo; it also matched `app/exporters/capcut/`. The
package was untracked by git and absent from the wheel while the entire test suite stayed
green against the working tree. `tests/unit/test_packaging.py` now checks the manifest.

### Also of note

- **No librosa.** Music analysis uses numpy and PyAV; loudness comes from FFmpeg's EBU R128
  meter, which is the reference implementation rather than an approximation of one. This
  avoids ~200 MB of numba/scipy/scikit-learn for a beat tracker and a resampler that take
  about fifty lines.
- **Four dependencies removed** that were declared and never imported: `ffmpeg-python`,
  `pysubs2`, `tomli-w`, `watchdog`.
- **`MediaRef` accepts a bare string** as well as `{"path": ...}`. The plan is authored by a
  language model and mentions a media reference eighty times in a forty-clip plan; the
  wrapper bought no safety, since every path constraint still applies.
- **The exporter takes an injected prober.** It needs a file's duration and geometry, but an
  exporter that imports an analyser cannot be swapped out — a rule
  `tests/unit/test_architecture.py` enforces, and which this briefly broke.

### Known limits

- The CapCut draft format is **reverse-engineered and version-specific**. It targets CapCut
  8.6.0.3667 and was verified for internal consistency, not by opening a draft — no
  reference draft existed on the development machine. Export with `--template` pointing at
  one of your own drafts if it does not open.
- **Transitions export as labelled placeholders.** CapCut resolves them by effect id from
  its own downloadable library, which AIVE cannot know, so the cut is hard until the named
  transition is picked again in the UI.
- **`FramingSpec` is not rendered.** Zoom, crop and Ken Burns are in the schema and reported
  as an unapplied warning rather than silently dropped.
- **Video analysis is sequential.** Forty 4K clips takes minutes. The seam for a process
  pool is deliberately obvious and unimplemented.
- **`xfade` has no `zoomout`.** A plan asking for it gets the configured substitute and a
  warning naming the clip.
