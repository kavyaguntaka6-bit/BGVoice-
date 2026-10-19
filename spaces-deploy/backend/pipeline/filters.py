"""Stage 1 — linear cleanup, zero-phase.

- High-pass 80 Hz, 4th-order Butterworth (sosfiltfilt).
- Hum: long-term spectrum peaks at 50/60 Hz and harmonics up to 500 Hz;
  a peak >12 dB above neighbouring bins gets an iirnotch (Q=30).
"""

from __future__ import annotations

import numpy as np
from scipy import signal

import config as C


def highpass(x: np.ndarray) -> np.ndarray:
    sos = signal.butter(C.HP_ORDER, C.HP_CUTOFF_HZ,
                        btype="high", fs=C.SAMPLE_RATE, output="sos")
    return signal.sosfiltfilt(sos, x).astype(np.float32)


def _long_term_spectrum(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = C.HUM_SPECTRUM_NFFT
    if len(x) < n:
        seg = np.pad(x, (0, n - len(x)))
        segs = seg[None, :]
    else:
        hop = n // 2
        n_seg = max(1, 1 + (len(x) - n) // hop)
        idx = np.arange(n)[None, :] + hop * np.arange(n_seg)[:, None]
        segs = x[idx]
    win = np.hanning(n).astype(np.float32)
    spec = np.abs(np.fft.rfft(segs * win, n=n)).mean(axis=0)
    freqs = np.fft.rfftfreq(n, 1.0 / C.SAMPLE_RATE)
    return freqs, spec


def _hum_notches(freqs: np.ndarray, spec_db: np.ndarray) -> list[tuple[float, float]]:
    """Return (freq, Q) notch list for hum peaks."""
    notches: list[tuple[float, float]] = []
    df = freqs[1] - freqs[0]
    for base in C.HUM_FREQS:
        h = 1
        while base * h <= C.HUM_MAX_HARM_HZ:
            f0 = base * h
            i0 = int(np.argmin(np.abs(freqs - f0)))
            lo = max(0, i0 - int(30 / df) - 2)
            hi = min(len(freqs), i0 + int(30 / df) + 3)
            neighbourhood = np.concatenate([spec_db[lo:i0 - 1], spec_db[i0 + 2:hi]])
            if neighbourhood.size == 0:
                h += 1
                continue
            peak_above = spec_db[i0] - float(np.median(neighbourhood))
            if peak_above > C.HUM_PEAK_ABOVE_NEIGHBOUR_DB:
                notches.append((f0, C.HUM_NOTCH_Q))
            h += 1
    return notches


def remove_hum(x: np.ndarray) -> tuple[np.ndarray, list[float]]:
    freqs, spec = _long_term_spectrum(x)
    spec_db = 20.0 * np.log10(spec + 1e-12)
    notches = _hum_notches(freqs, spec_db)
    y = x
    for f0, q in notches:
        b, a = signal.iirnotch(f0, q, fs=C.SAMPLE_RATE)
        sos = signal.tf2sos(b, a)
        y = signal.sosfiltfilt(sos, y)
    return y.astype(np.float32), [f for f, _ in notches]


def linear_cleanup(x: np.ndarray) -> tuple[np.ndarray, dict]:
    """Returns (cleaned, info) with info['hum_notches_hz']."""
    y = highpass(x)
    y, notches = remove_hum(y)
    return y, {"hum_notches_hz": notches}
