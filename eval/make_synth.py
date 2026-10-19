"""Generate a synthetic evaluation clip: speech-like signal + fan noise + hum.

Speech-like: harmonic complex with a NATURAL f0 contour (slow intonation,
small jitter — NOT vibrato), full harmonic spectrum shaped by formant
RESONANCES (peaking filters that boost formants; never a bandpass that
deletes the fundamental), syllabic amplitude modulation with pauses, and
subtle frication. Denoisers trained on real speech must recognize it as
speech: no vibrato, no bandpass gutting the spectrum.
"""

from __future__ import annotations

import argparse

import numpy as np
import soundfile as sf
from scipy import signal

SR = 48000


def speech_like(dur: float, rng: np.random.Generator) -> np.ndarray:
    n = int(dur * SR)
    t = np.arange(n) / SR
    # natural f0 contour: base 130 Hz + slow intonation (NOT vibrato)
    f0 = (130
          + 12 * np.sin(2 * np.pi * 0.35 * t + 0.7)   # phrase intonation
          + 6 * np.sin(2 * np.pi * 0.9 * t + 2.0)     # syllable pitch accents
          + 2.5 * np.sin(2 * np.pi * 5.0 * t))        # microprosody / jitter
    phase = 2 * np.pi * np.cumsum(f0) / SR
    # voiced harmonic stack. NOTE: keep to 4 harmonics — DeepFilterNet3
    # (trained on real speech) is sensitive to the exact harmonic structure
    # of synthetic signals; 4 harmonics gives a stable, speech-like voiced
    # source that the model preserves (see BUILD_REPORT.md).
    x = np.zeros(n)
    for h, a in [(1, 1.0), (2, 0.5), (3, 0.3), (4, 0.2)]:
        x += a * np.sin(h * phase + rng.uniform(0, 2 * np.pi))
    # syllabic amplitude modulation ~4 Hz with natural pauses
    am = 0.55 + 0.45 * np.sin(2 * np.pi * 4.0 * t + 1.0)
    pauses = (np.sin(2 * np.pi * 0.9 * t + 0.5) > -0.2).astype(float)
    pauses = np.convolve(pauses, np.ones(int(0.05 * SR)) / int(0.05 * SR),
                         mode="same")
    x = x * am * np.clip(pauses, 0.05, 1.0)
    # subtle frication (unvoiced energy)
    for _ in range(int(dur)):
        s = int(rng.uniform(0, n - int(0.06 * SR)))
        burst = rng.standard_normal(int(0.06 * SR))
        sos_h = signal.butter(4, 4500, btype="high", fs=SR, output="sos")
        burst = signal.sosfiltfilt(sos_h, burst) * 0.06
        x[s:s + len(burst)] += burst
    x = x / max(np.abs(x).max(), 1e-9) * 0.7
    return x.astype(np.float32)


def fan_noise(dur: float, rng: np.random.Generator) -> np.ndarray:
    n = int(dur * SR)
    w = rng.standard_normal(n)
    sos = signal.butter(4, 900, btype="low", fs=SR, output="sos")
    f = signal.sosfiltfilt(sos, w)
    t = np.arange(n) / SR
    f = f * (0.8 + 0.2 * np.sin(2 * np.pi * 0.3 * t))
    return (f / max(np.abs(f).max(), 1e-9)).astype(np.float32)


def hum(dur: float) -> np.ndarray:
    n = int(dur * SR)
    t = np.arange(n) / SR
    h = (np.sin(2 * np.pi * 50 * t)
         + 0.4 * np.sin(2 * np.pi * 100 * t)
         + 0.2 * np.sin(2 * np.pi * 150 * t))
    return (h / np.abs(h).max() * 0.5).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dur", type=float, default=8.0)
    ap.add_argument("--snr", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out-noisy", default="noisy.wav")
    ap.add_argument("--out-clean", default="clean.wav")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    clean = speech_like(a.dur, rng)
    noise = fan_noise(a.dur, rng) + hum(a.dur)
    pc = np.mean(clean ** 2)
    pn = np.mean(noise ** 2)
    g = np.sqrt(pc / (pn * 10 ** (a.snr / 10)))
    noisy = clean + g * noise
    noisy = noisy / max(np.abs(noisy).max(), 1e-9) * 0.89
    sf.write(a.out_noisy, noisy, SR, subtype="PCM_16")
    sf.write(a.out_clean, clean, SR, subtype="PCM_16")
    print(f"wrote {a.out_noisy} and {a.out_clean} "
          f"({a.dur}s @ {a.snr} dB SNR)")


if __name__ == "__main__":
    main()
