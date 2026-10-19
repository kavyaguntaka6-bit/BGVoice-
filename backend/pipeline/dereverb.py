"""Stage 5 — dereverb (WPE, toggle; on in Studio mode).

WPE per frequency bin:
    x~[k,l] = x[k,l] - sum_tau conj(g[k,tau]) * x[k, l-D-tau]
delay D = 3 frames, K = 12 taps, 3 iterations, weights 1/lambda smoothed
over +-1 frame (nara_wpe). Conservative by construction.
"""

from __future__ import annotations

import numpy as np
import librosa

import config as C


def _load():
    try:
        from nara_wpe.wpe import wpe
    except ImportError as e:
        raise RuntimeError(
            "The `nara_wpe` package is not installed. Install it with "
            "`pip install nara_wpe` to use dereverberation.") from e
    return wpe


def _dereverb_once(x: np.ndarray) -> np.ndarray:
    wpe = _load()
    S = librosa.stft(x, n_fft=C.WPE_NFFT, hop_length=C.WPE_HOP,
                     window="hann", center=True)
    # nara_wpe expects (frames, freq)
    Y = S.T.copy()
    X = wpe(Y, taps=C.WPE_TAPS, delay=C.WPE_DELAY,
            iterations=C.WPE_ITERATIONS).T
    y = librosa.istft(X, hop_length=C.WPE_HOP, window="hann", center=True,
                      length=len(x))
    return y.astype(np.float32)


def _crossfade(a_tail: np.ndarray, b_head: np.ndarray, n: int) -> np.ndarray:
    t = np.arange(n, dtype=np.float32) / max(n, 1)
    w = 0.5 - 0.5 * np.cos(np.pi * t)
    return a_tail * (1 - w) + b_head * w


def dereverb(x: np.ndarray, chunk_s: float = 20.0) -> np.ndarray:
    """Apply WPE dereverberation. Chunked to bound memory. Length-preserving."""
    sr = C.SAMPLE_RATE
    if len(x) <= int(chunk_s * sr):
        return _dereverb_once(x)
    chunk = int(chunk_s * sr)
    ov = int(2.0 * sr)
    stride = chunk - ov
    out = np.zeros_like(x)
    pos = 0
    prev_tail = None
    while pos < len(x):
        seg = x[pos:pos + chunk]
        y = _dereverb_once(seg)
        if prev_tail is not None:
            n_ov = min(ov, len(y))
            y[:n_ov] = _crossfade(prev_tail[-n_ov:], y[:n_ov], n_ov)
        end = min(pos + len(y), len(x))
        out[pos:end] = y[:end - pos]
        if end < len(x):
            prev_tail = y[end - pos:end - pos + ov]
        pos += stride
        if pos >= len(x):
            break
    return out
