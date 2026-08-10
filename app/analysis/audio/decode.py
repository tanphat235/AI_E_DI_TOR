"""Decoding audio to a mono numpy array.

PyAV rather than librosa or pydub, for three reasons that all point the same way: PyAV is
already a dependency (Phase 3 probes video with it), it decodes anything FFmpeg can, and
it needs no subprocess. librosa would pull in numba, scipy and scikit-learn to give us a
beat tracker and a resampler we can write in fifty lines.

The trade-off is real and worth stating: our tempo estimate (:mod:`.features`) is a plain
spectral-flux autocorrelation. On four-on-the-floor music it is as good as anything; on
rubato piano or dense orchestral material it will decline to answer rather than guess,
which is why :attr:`~app.models.audio.MusicTrack.bpm` is nullable.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from app.analysis.audio.base import AudioDecodeError, AudioDependencyMissingError
from app.utils.logging import get_logger

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

logger = get_logger(__name__)

DECODER_VERSION = "pyav-audio/1"


def decode_mono(path: Path, *, sample_rate: int) -> NDArray[np.float32]:
    """Decode ``path`` to a mono float32 array at ``sample_rate``.

    Samples are in the usual -1.0 to 1.0 range. Downmixing to mono is deliberate: every
    feature we extract is about the track's overall character, and a stereo-aware version
    would double the work to produce the same answer.

    Raises:
        AudioDependencyMissingError: PyAV or numpy is not installed.
        AudioDecodeError: the file has no audio, or cannot be read.
    """
    try:
        import av
        import numpy as np
    except ImportError as exc:  # pragma: no cover - exercised by a monkeypatched import
        msg = 'audio analysis needs PyAV and numpy. Install them with: pip install -e ".[audio]"'
        raise AudioDependencyMissingError(msg) from exc

    chunks: list[NDArray[np.float32]] = []
    try:
        with av.open(str(path)) as container:
            if not container.streams.audio:
                msg = f"{path.name} contains no audio stream"
                raise AudioDecodeError(msg)
            stream = container.streams.audio[0]

            # fltp gives planar float; mono collapses the layout for us so no manual
            # channel averaging is needed.
            resampler = av.AudioResampler(format="fltp", layout="mono", rate=sample_rate)
            for frame in container.decode(stream):
                chunks.extend(_to_arrays(resampler.resample(frame)))
            # The resampler buffers; without the flush the tail of the file is missing,
            # which silently shortens every duration-derived feature.
            chunks.extend(_to_arrays(resampler.resample(None)))
    except AudioDecodeError:
        raise
    except Exception as exc:  # av raises a wide family of its own error types
        msg = f"could not decode {path.name}: {type(exc).__name__}: {exc}"
        raise AudioDecodeError(msg) from exc

    if not chunks:
        msg = f"{path.name} decoded to zero samples"
        raise AudioDecodeError(msg)

    samples = np.concatenate(chunks)
    logger.debug("Decoded %s: %d samples at %d Hz", path.name, samples.size, sample_rate)
    return samples


def _to_arrays(frames: Any) -> list[NDArray[np.float32]]:
    """Flatten resampled frames into 1-D arrays.

    ``AudioResampler.resample`` returns a list in PyAV 9+ and a single frame in older
    releases; both shapes are accepted so a version bump does not become a decode failure.

    Typed ``Any`` rather than a PyAV frame type: PyAV ships no stubs for the resampler's
    return value, and importing it at module scope purely to annotate this would make an
    optional dependency mandatory.
    """
    import numpy as np

    if frames is None:
        return []
    batch: list[Any] = frames if isinstance(frames, list) else [frames]
    return [
        np.asarray(frame.to_ndarray(), dtype=np.float32).reshape(-1)
        for frame in batch
        if frame is not None
    ]


__all__ = ["DECODER_VERSION", "decode_mono"]
