"""Stage 6 — classical post-filter (MMSE-LSA, Ephraim-Malah).

Polishes leftover steady noise after the neural stage; the neural stage
does the heavy work. Gains are real -> phase untouched.

Pipeline per TF bin:
  noise PSD  : lambda_d[k,l] = at*lambda_d[k,l-1] + (1-at)*|X[k,l]|^2
               at = a_d + (1-a_d)*p[k,l], a_d = 0.95
  a posteriori: gamma = |X|^2 / lambda_d
  a priori    : xi = a*|S_hat_prev|^2/lambda_d_prev + (1-a)*max(gamma-1,0),
                a = 0.98, floor -25 dB
  MMSE-LSA    : G = xi/(1+xi) * exp(0.5 * E1(v)), v = xi*gamma/(1+xi)
  floor G at -12 dB, then naturalness controls:
  attack/release smoothing, frequency smoothing + 3x3 median,
  harmonic protection (pYIN f0), sibilant protection.
Inverse via scipy istft (NOLA/COLA asserted by construction).
"""

from __future__ import annotations

import numpy as np
from scipy import signal
from scipy.special import exp1
from scipy.ndimage import median_filter
import librosa

import config as C


def _stft(x: np.ndarray):
    f, t, X = signal.stft(x, fs=C.SAMPLE_RATE, window="hann",
                          nperseg=C.PF_NFFT, noverlap=C.PF_NFFT - C.PF_HOP,
                          boundary="zeros", padded=True)
    return f, t, X


def _noise_psd_init(power: np.ndarray) -> np.ndarray:
    frame_e = power.mean(axis=0)
    k = max(1, int(0.10 * power.shape[1]))
    idx = np.argpartition(frame_e, k)[:k]
    return power[:, idx].mean(axis=1)


def _speech_presence(power: np.ndarray, lam: np.ndarray,
                     v: np.ndarray) -> np.ndarray:
    """p[k,l]: VAD v[l] combined with a per-bin SNR test."""
    gamma = power / np.maximum(lam[:, None], 1e-12)
    p = np.where(gamma > 1.5, 1.0, 0.05).astype(np.float32)
    return p * v[None, :].astype(np.float32)


def _smooth_attack_release(G: np.ndarray) -> np.ndarray:
    c_att = np.exp(-C.PF_HOP / (C.SAMPLE_RATE * C.PF_TAU_ATT_MS / 1000.0))
    c_rel = np.exp(-C.PF_HOP / (C.SAMPLE_RATE * C.PF_TAU_REL_MS / 1000.0))
    Gs = np.empty_like(G)
    prev = G[:, 0]
    Gs[:, 0] = prev
    for l in range(1, G.shape[1]):
        c = np.where(G[:, l] > prev, c_att, c_rel)
        prev = c * prev + (1 - c) * G[:, l]
        Gs[:, l] = prev
    return Gs


def _freq_smooth(G: np.ndarray) -> np.ndarray:
    k = np.asarray(C.PF_FREQ_KERNEL, dtype=np.float32)
    Gs = signal.lfilter(k, [1.0], G, axis=0)
    return median_filter(Gs, size=(C.PF_MEDIAN_SIZE, C.PF_MEDIAN_SIZE))


def _harmonic_protection(G: np.ndarray, x: np.ndarray, v: np.ndarray,
                         n_fft: int) -> np.ndarray:
    f0, _, _ = librosa.pyin(x, fmin=C.PF_F0_MIN, fmax=C.PF_F0_MAX,
                            sr=C.SAMPLE_RATE, frame_length=n_fft,
                            hop_length=C.PF_HOP, center=True)
    n_bins = G.shape[0]
    for l in range(G.shape[1]):
        if l >= len(f0) or v[l] < 0.5:
            continue
        f = f0[l]
        if not np.isfinite(f) or f <= 0:
            continue
        h = 1
        while h * f <= C.PF_HARM_MAX_HZ:
            kc = int(round(h * f * n_fft / C.SAMPLE_RATE))
            for kk in (kc - 1, kc, kc + 1):
                if 0 <= kk < n_bins:
                    G[kk, l] = max(G[kk, l], C.PF_HARM_FLOOR)
            h += 1
    return G


def _sibilant_protection(G: np.ndarray, power: np.ndarray, v: np.ndarray,
                         freqs: np.ndarray) -> np.ndarray:
    lo_f, hi_f = C.PF_SIB_BAND
    flo_f, fhi_f = C.PF_FULL_BAND[0], C.PF_FULL_BAND[1]
    sib = (freqs >= lo_f) & (freqs <= hi_f)
    full = (freqs >= flo_f) & (freqs <= fhi_f)
    e_sib = power[sib].sum(axis=0) + 1e-12
    e_full = power[full].sum(axis=0) + 1e-12
    ratio = e_sib / e_full
    # spectral flatness of 4-10 kHz: geomean / mean
    ps = power[sib] + 1e-12
    flat = np.exp(np.log(ps).mean(axis=0)) / ps.mean(axis=0)
    protect = (v >= 0.5) & (ratio > C.PF_SIB_RATIO) & (flat > C.PF_SIB_FLATNESS)
    G[sib[:, None] & protect[None, :]] = np.maximum(
        G[sib[:, None] & protect[None, :]], C.PF_SIB_FLOOR)
    return G


def mmse_lsa_filter(x: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Apply the MMSE-LSA post-filter. v: per-frame speech mask (n_frames,).
    Length-preserving."""
    freqs, _, X = _stft(x)
    n_frames = X.shape[1]
    if len(v) != n_frames:
        # resample v to frame count by nearest neighbour
        idx = np.clip((np.arange(n_frames) * len(v) // n_frames), 0, len(v) - 1)
        v = v[idx].astype(np.float32)
    power = (np.abs(X) ** 2).astype(np.float64)

    lam = _noise_psd_init(power)
    xi_floor = 10.0 ** (C.PF_XI_FLOOR_DB / 10.0)
    g_floor = 10.0 ** (C.PF_GAIN_FLOOR_DB / 20.0)
    G = np.empty_like(power, dtype=np.float32)
    s_prev = np.zeros(power.shape[0])

    for l in range(n_frames):
        p = _speech_presence(power[:, [l]], lam, v[[l]])[:, 0]
        at = C.PF_A_D + (1 - C.PF_A_D) * p
        lam = at * lam + (1 - at) * power[:, l]
        lam = np.maximum(lam, 1e-12)
        gamma = power[:, l] / lam
        xi = (C.PF_A_DD * (s_prev ** 2) / np.maximum(lam, 1e-12)
              + (1 - C.PF_A_DD) * np.maximum(gamma - 1.0, 0.0))
        xi = np.maximum(xi, xi_floor)
        vk = xi * gamma / (1 + xi)
        gain = xi / (1 + xi) * np.exp(0.5 * exp1(np.maximum(vk, 1e-12)))
        gain = np.maximum(gain, g_floor).astype(np.float32)
        G[:, l] = gain
        s_prev = gain * np.abs(X[:, l])

    G = _smooth_attack_release(G)
    G = _freq_smooth(G)
    G = _harmonic_protection(G, x, v, C.PF_NFFT)
    G = _sibilant_protection(G, power.astype(np.float32), v, freqs)

    S_hat = (G * X).astype(np.complex64)
    _, y = signal.istft(S_hat, fs=C.SAMPLE_RATE, window="hann",
                        nperseg=C.PF_NFFT, noverlap=C.PF_NFFT - C.PF_HOP,
                        input_onesided=True, boundary=True)
    y = y.astype(np.float32)
    if len(y) > len(x):
        y = y[:len(x)]
    elif len(y) < len(x):
        y = np.pad(y, (0, len(x) - len(y)))
    return y
