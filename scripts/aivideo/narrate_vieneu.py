"""Narrate a script with VieNeu-TTS, one sentence at a time.

Microsoft edge-tts is not used here. Each written line is synthesized on its
own and joined by a short gap, so the caption times are the synthesis times
and the pauses stay short.

Writes:
  .aive/audio/<id>.mp3
  .aive/audio/<id>.json     per-line start/end in seconds
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
os.environ.setdefault("HF_HOME", str(REPO / ".hf-cache"))

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Short enough that the edit does not sound like the old neural voice, long
# enough that two sentences do not collide.
GAP_S = 0.20
SR = 48000
FFMPEG = (
    REPO / ".venv-sdxl" / "Lib" / "site-packages" / "imageio_ffmpeg"
    / "binaries" / "ffmpeg-win-x86_64-v7.1.exe"
)


def work_dir(script_path: str) -> Path:
    """The project that holds the script. Artefacts are written beside it."""
    return Path(script_path).resolve().parents[1]


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: narrate_vieneu.py <script.json>")
        return 2
    doc = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    lines = [x["text"] for x in doc["lines"]]
    voice = doc["voice"]

    import numpy as np
    import soundfile as sf
    from vieneu import Vieneu

    # ONNX stays on the CPU. The GPU is reserved for the image pass, and this
    # environment has no torch.
    tts = Vieneu(backend="onnx")
    presets = tts.list_preset_voices()
    chosen = next((label for label, voice_id in presets if voice_id == voice), None)
    if chosen is None:
        print(f"voice {voice!r} is not a VieNeu preset")
        for label, voice_id in presets:
            print(f"  {voice_id}  —  {label}")
        return 2
    print(f"VieNeu ONNX  {chosen}  {len(lines)} lines", flush=True)

    out_dir = work_dir(sys.argv[1]) / ".aive" / "audio"
    part_dir = out_dir / f"{doc['id']}-vieneu"
    out_dir.mkdir(parents=True, exist_ok=True)
    part_dir.mkdir(parents=True, exist_ok=True)

    pieces: list[np.ndarray] = []
    timed: list[dict] = []
    cursor = 0.0
    gap = np.zeros(int(SR * GAP_S), dtype=np.float32)
    for i, text in enumerate(lines):
        part = part_dir / f"{i:03d}.wav"
        note = part_dir / f"{i:03d}.json"
        cached = note.is_file() and json.loads(note.read_text(encoding="utf-8")).get("text") == text
        if cached and part.is_file():
            audio, sr = sf.read(part, dtype="float32")
            if sr != SR:
                raise SystemExit(f"{part.name} is {sr} Hz, expected {SR}")
        else:
            raw = np.asarray(tts.infer(text, voice=voice))
            audio = raw.astype(np.float32, copy=False)
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            peak = float(np.max(np.abs(audio))) if audio.size else 0.0
            if peak > 2.0:
                audio = audio / 32768.0
            sf.write(part, audio, SR)
            note.write_text(json.dumps({"text": text}, ensure_ascii=False), encoding="utf-8")
        duration = len(audio) / SR
        timed.append({
            "text": text,
            "start": round(cursor, 3),
            "end": round(cursor + duration, 3),
            "matched": True,
        })
        pieces.append(audio)
        cursor += duration
        if i < len(lines) - 1:
            pieces.append(gap)
            cursor += GAP_S
        print(f"  {i + 1:3d}/{len(lines)}  {duration:5.2f}s  {text[:56]}", flush=True)

    wav = out_dir / f"{doc['id']}.wav"
    mp3 = out_dir / f"{doc['id']}.mp3"
    sf.write(wav, np.concatenate(pieces), SR)
    if not FFMPEG.is_file():
        raise SystemExit(f"ffmpeg not found at {FFMPEG}")
    subprocess.run(
        [str(FFMPEG), "-y", "-loglevel", "error", "-i", str(wav),
         "-codec:a", "libmp3lame", "-q:a", "2", str(mp3)],
        check=True,
    )
    (out_dir / f"{doc['id']}.json").write_text(
        json.dumps({"lines": timed, "voice": voice, "engine": "vieneu"}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\naudio {mp3}  {mp3.stat().st_size / 1024:.0f} KB")
    print(f"narration ends at {timed[-1]['end']:.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
