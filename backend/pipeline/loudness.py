"""Stage 9a — loudness normalization + true-peak limiting.

ITU-R BS.1770-4 integrated loudness via pyloudnorm (gated -70 LUFS
absolute, -10 LU relative), then a true-peak limiter at -1.0 dBTP with
5 ms lookahead, 50 ms release and 4x oversampled peak detection.
"""

from __future__ import annotations

import numpy as np
import pyloudnorm
from scipy.signal import resample_poly

import config as C


def integrated_lufs(x: np.ndarray) -> float:
    meter = pyloudnorm.Meter(C.SAMPLE_RATE)
    try:
        return float(meter.integrated_loudness(x))
    except Exception:
        return float("-inf")


def normalize_loudness(x: np.ndarray,
                       target: float = C.LOUDNESS_TARGET) -> np.ndarray:
    if target not in C.LOUDNESS_OPTIONS:
        raise ValueError(f"loudness_target must be one of {C.LOUDNESS_OPTIONS}")
    lufs = integrated_lufs(x)
    if not np.isfinite(lufs):
        return x
    gain_db = target - lufs
    # sanity: never apply absurd gains
    gain_db = float(np.clip(gain_db, -40.0, 40.0))
    return (x * 10.0 ** (gain_db / 20.0)).astype(np.float32)


def true_peak_limit(x: np.ndarray,
                    ceiling_db: float = C.TRUE_PEAK_DBTP) -> np.ndarray:
    """4x oversampled true-peak limiter, length-preserving."""
    os_factor = C.LIMITER_OVERSAMPLE
    x_os = resample_poly(x, os_factor, 1).astype(np.float32)
    ceil_lin = 10.0 ** (ceiling_db / 20.0)
    look = int(C.LIMITER_LOOKAHEAD_MS / 1000 * C.SAMPLE_RATE * os_factor)
    rel_n = int(C.LIMITER_RELEASE_MS / 1000 * C.SAMPLE_RATE * os_factor)
    peak = np.abs(x_os)
    # lookahead: future peak within `look` samples
    if look > 0:
        # sliding maximum over the lookahead window
        from numpy.lib.stride_tricks import sliding_window_view
        sw = sliding_window_view(np.pad(peak, (0, look), mode="edge"), look + 1)
        peak_la = sw.max(axis=1)
    else:
        peak_la = peak
    need = ceil_lin / np.maximum(peak_la, 1e-12)
    need = np.minimum(need, 1.0)
    # release smoothing: gain may only rise with rel_n time constant
    c_rel = np.exp(-1.0 / max(rel_n, 1))
    g = np.empty_like(need)
    e = 1.0
    for n in range(len(need)):
        tgt = need[n]
        e = tgt if tgt < e else c_rel * e + (1 - c_rel) * tgt
        g[n] = e
    y_os = x_os * g
    y = resample_poly(y_os, 1, os_factor).astype(np.float32)
    if len(y) > len(x):
        y = y[:len(x)]
    elif len(y) < len(x):
        y = np.pad(y, (0, len(x) - len(y)))
    return y
