"""Stage 7 — room tone.

Hard digital silence between words sounds artificial. Adds continuous
(not gated) shaped noise whose spectrum matches the INPUT's non-speech
frames, scaled to the chosen RMS (Subtle -68 dBFS / Natural -60 dBFS).
"""

from __future__ import annotations

import numpy as np
from scipy import signal

import config as C


def _input_noise_spectrum(x: np.ndarray, v48: np.ndarray,
                          n_fft: int = 1024, hop: int = 256) -> np.ndarray:
    _, _, X = signal.stft(x, fs=C.SAMPLE_RATE, window="hann",
                          nperseg=n_fft, noverlap=n_fft - hop,
                          boundary="zeros", padded=True)
    power = np.abs(X) ** 2
    n_frames = X.shape[1]
    idx = np.clip(np.arange(n_frames) * len(v48) // max(len(v48), 1), 0,
                  max(len(v48) - 1, 0))
    vf = v48[idx].astype(bool) if len(v48) else np.zeros(n_frames, bool)
    if not (~vf).any():  # no non-speech frames: use lowest-energy 10%
        e = power.mean(axis=0)
        k = max(1, n_frames // 10)
        sel = np.argpartition(e, k)[:k]
        lam = power[:, sel].mean(axis=1)
    else:
        lam = power[:, ~vf].mean(axis=1)
    return np.maximum(lam, 1e-12).astype(np.float32)


def add_room_tone(y: np.ndarray, x_input: np.ndarray, v48: np.ndarray,
                  level: str = C.DEFAULT_ROOM_TONE,
                  rng: np.random.Generator | None = None) -> np.ndarray:
    """Add continuous shaped room tone. level: off|subtle|natural."""
    if level == C.ROOM_TONE_OFF:
        return y
    target_db = (C.ROOM_TONE_SUBTLE_DBFS if level == "subtle"
                 else C.ROOM_TONE_NATURAL_DBFS)
    rng = rng or np.random.default_rng(0)
    n_fft, hop = 1024, 256
    lam = _input_noise_spectrum(x_input, v48, n_fft, hop)
    n_frames = 1 + max(0, (len(y) - n_fft)) // hop + 2
    phi = rng.uniform(0, 2 * np.pi, size=(len(lam), n_frames)).astype(np.float32)
    spec = (np.sqrt(lam)[:, None] * np.exp(1j * phi)).astype(np.complex64)
    _, tone = signal.istft(spec, fs=C.SAMPLE_RATE, window="hann",
                           nperseg=n_fft, noverlap=n_fft - hop,
                           input_onesided=True, boundary=True)
    tone = tone.astype(np.float32)
    if len(tone) > len(y):
        tone = tone[:len(y)]
    elif len(tone) < len(y):
        tone = np.pad(tone, (0, len(y) - len(tone)))
    rms = float(np.sqrt(np.mean(tone ** 2) + 1e-18))
    target_rms = 10.0 ** (target_db / 20.0)
    tone = tone * (target_rms / max(rms, 1e-12))
    return (y + tone).astype(np.float32)
