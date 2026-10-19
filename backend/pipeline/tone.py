"""Stage 8 — tone & dynamics (Studio mode only; each toggleable).

Light and linear: presence EQ (RBJ peaking), de-esser (dynamic band cut),
soft-knee compressor (Giannoulis et al.) with auto make-up.
All filters zero-phase (sosfiltfilt) to keep transients clean.
"""

from __future__ import annotations

import numpy as np
from scipy import signal

import config as C


def _rbj_peaking(f0: float, gain_db: float, q: float,
                 fs: float) -> np.ndarray:
    a = 10 ** (gain_db / 40.0)
    w0 = 2 * np.pi * f0 / fs
    alpha = np.sin(w0) / (2 * q)
    b = [1 + alpha * a, -2 * np.cos(w0), 1 - alpha * a]
    a_ = [1 + alpha / a, -2 * np.cos(w0), 1 - alpha / a]
    return signal.tf2sos(b, a_)


def presence_eq(x: np.ndarray) -> np.ndarray:
    sos = _rbj_peaking(C.PRESENCE_FREQ, C.PRESENCE_GAIN_DB, C.PRESENCE_Q,
                       C.SAMPLE_RATE)
    return signal.sosfiltfilt(sos, x).astype(np.float32)


def deesser(x: np.ndarray, v48: np.ndarray) -> np.ndarray:
    lo, hi = C.DEESS_BAND
    sos = signal.butter(4, [lo, hi], btype="band",
                        fs=C.SAMPLE_RATE, output="sos")
    band = signal.sosfiltfilt(sos, x)
    env_n = max(1, int(C.DEESS_ENV_MS / 1000 * C.SAMPLE_RATE))
    env = np.sqrt(np.convolve(band ** 2, np.ones(env_n) / env_n,
                              mode="same") + 1e-12)
    env_db = 20 * np.log10(env)
    speech_env = env_db[v48.astype(bool)] if v48.any() else env_db
    T = float(np.median(speech_env)) + C.DEESS_THRESH_ABOVE_MEDIAN_DB
    red_db = np.minimum(C.DEESS_MAX_DB,
                        np.maximum(0.0, (env_db - T) * (1 - 1 / 4)))
    g = 10.0 ** (-red_db / 20.0)
    # dynamic EQ: cut only the sibilant band
    return (x - band + band * g).astype(np.float32)


def compressor(x: np.ndarray) -> np.ndarray:
    T, R, W = C.COMP_THRESH_DBFS, C.COMP_RATIO, C.COMP_KNEE_DB
    x_db = 20 * np.log10(np.abs(x) + 1e-12)
    y_g = np.empty_like(x_db)
    below = 2 * (x_db - T) < -W
    above = 2 * (x_db - T) > W
    mid = ~(below | above)
    y_g[below] = x_db[below]
    y_g[above] = T + (x_db[above] - T) / R
    d = x_db[mid] - T + W / 2
    y_g[mid] = x_db[mid] + (1 / R - 1) * d ** 2 / (2 * W)
    gr_db = y_g - x_db  # <= 0
    # envelope follower on gain reduction
    att = np.exp(-1 / (C.SAMPLE_RATE * C.COMP_ATTACK_MS / 1000))
    rel = np.exp(-1 / (C.SAMPLE_RATE * C.COMP_RELEASE_MS / 1000))
    env = np.empty_like(gr_db)
    e = 0.0
    for n in range(len(gr_db)):
        tgt = gr_db[n]
        c = att if tgt < e else rel
        e = c * e + (1 - c) * tgt
        env[n] = e
    y = x * (10.0 ** (env / 20.0))
    # auto make-up: invert static gain at 0 dBFS
    makeup_db = -(T + (0.0 - T) / R)
    return (y * 10.0 ** (makeup_db / 20.0)).astype(np.float32)


def studio_tone(x: np.ndarray, v48: np.ndarray,
                presence: bool = True, deess: bool = True,
                compress: bool = True) -> np.ndarray:
    y = x
    if presence:
        y = presence_eq(y)
    if deess:
        y = deesser(y, v48)
    if compress:
        y = compressor(y)
    return y
