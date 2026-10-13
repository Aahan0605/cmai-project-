"""Frame sampling, frame reading and audio extraction (TRD 3.2 steps 3-4).

Video decoding needs OpenCV (`pip install -e .[extract]`); audio extraction needs the
`ffmpeg` binary on PATH. Both are imported lazily so tests and CI run without them.
"""

from __future__ import annotations

import io
import subprocess
import wave
from pathlib import Path

import numpy as np


def sample_frame_indices(n_frames: int, k: int) -> list[int]:
    """K frames uniform in time: the centre of each of K equal segments (all if n < K)."""
    if n_frames <= k:
        return list(range(max(n_frames, 0)))
    return [int((i + 0.5) * n_frames / k) for i in range(k)]


def read_sampled_frames(path: Path, k: int) -> tuple[list[int], list[np.ndarray]]:
    """Decode sequentially (seeking is frame-inaccurate for many codecs); RGB uint8 frames."""
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise OSError(f"cannot open video {path}")
    wanted = set(sample_frame_indices(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), k))
    indices, frames = [], []
    i = 0
    while wanted and i <= max(wanted):
        ok, frame = cap.read()
        if not ok:
            break
        if i in wanted:
            indices.append(i)
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        i += 1
    cap.release()
    return indices, frames


def load_audio(path: Path, sample_rate: int) -> np.ndarray:
    """Mono float32 waveform in [-1, 1] at `sample_rate`.

    A sibling `<video>.wav` (as written for OptLab-TIMIT) is read directly and resampled;
    otherwise ffmpeg decodes the video's audio track.
    """
    wav_path = Path(path).with_suffix(".wav")
    if wav_path.exists():
        from math import gcd

        from scipy.signal import resample_poly

        with wave.open(str(wav_path)) as w:
            if w.getsampwidth() != 2:
                raise ValueError(f"{wav_path}: expected 16-bit PCM")
            pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
            pcm = pcm.reshape(-1, w.getnchannels()).mean(axis=1)
            rate = w.getframerate()
        g = gcd(rate, sample_rate)
        return resample_poly(pcm / 32768.0, sample_rate // g, rate // g).astype(np.float32)
    out = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(path),
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(sample_rate),
            "-f",
            "wav",
            "-acodec",
            "pcm_s16le",
            "-",
        ],
        capture_output=True,
        check=True,
    ).stdout
    with wave.open(io.BytesIO(out)) as w:
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    return pcm.astype(np.float32) / 32768.0
