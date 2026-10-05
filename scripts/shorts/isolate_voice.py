"""Take the music out of a talk that already has a bed, keeping only the voice.

For re-cutting a short that was published with music under it: laying the
channel's own bed on top of that would play two tracks at once. Demucs
(htdemucs, MIT) splits the soundtrack into stems; the ``vocals`` stem is the
speaker, and it replaces the soundtrack of an otherwise untouched copy of the
video. That copy is then an ordinary ``--source`` for build_shorts.py, which can
lay the bed and pitch the voice as it does for any talk.

Run with .venv-sdxl (CUDA torch). Audio is read and written through ffmpeg and
numpy, not torchaudio: torchaudio 2.9+ moved its file I/O out to torchcodec,
and nothing here needs it.

Example:
  .\\.venv-sdxl\\Scripts\\python.exe scripts\\shorts\\isolate_voice.py ^
    --source old_short.mp4 --out projects\\job\\source\\voice_only.mp4
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
import wave
from pathlib import Path

import imageio_ffmpeg
import numpy as np

SR = 44100  # htdemucs' own rate


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Strip the music bed, keep the voice.")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="Video with the voice-only track.")
    parser.add_argument("--model", default="htdemucs")
    args = parser.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    import torch
    from demucs.apply import apply_model
    from demucs.pretrained import get_model

    ff = imageio_ffmpeg.get_ffmpeg_exe()
    raw = subprocess.run(
        [ff, "-loglevel", "error", "-i", str(args.source), "-vn", "-ac", "2", "-ar", str(SR),
         "-f", "f32le", "-"],
        capture_output=True, check=True,
    ).stdout
    audio = np.frombuffer(raw, dtype=np.float32).reshape(-1, 2).T.copy()
    print(f"{audio.shape[1] / SR:.1f}s of audio")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = get_model(args.model)
    model.to(device).eval()
    wav = torch.from_numpy(audio)
    # Demucs expects the mix normalised; undo it on the way out so the voice
    # keeps the level it had in the mix.
    ref = wav.mean(0)
    mean, std = ref.mean(), ref.std() + 1e-8
    t0 = time.time()
    with torch.no_grad():
        stems = apply_model(model, ((wav - mean) / std)[None].to(device), device=device,
                            split=True, overlap=0.25, progress=False)[0]
    vocals = (stems[model.sources.index("vocals")].cpu() * std + mean).numpy()
    print(f"separated with {args.model} on {device} in {time.time() - t0:.1f}s")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_suffix(".vocals.wav")
    pcm = (np.clip(vocals.T, -1.0, 1.0) * 32767).astype(np.int16)
    with wave.open(str(tmp), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())
    # Picture copied untouched; only the soundtrack is replaced.
    subprocess.run(
        [ff, "-y", "-loglevel", "error", "-i", str(args.source), "-i", str(tmp),
         "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
         "-shortest", str(args.out)],
        check=True,
    )
    tmp.unlink()
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
