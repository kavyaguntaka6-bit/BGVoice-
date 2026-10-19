#!/usr/bin/env python3
"""VoiceClean v2 — objective evaluation suite (spec section 7).

Builds synthetic test mixes and measures the `standard` pipeline mode:

    noisy = clean + g * noise,   g = sqrt(P_clean / (P_noise * 10^(SNR/10)))

SNR grid: -5, 0, 5, 10, 15 dB. Noise types (all synthesized at 48 kHz with
numpy/scipy — no external data, no network):

  fan       fan / AC — lowpassed white noise with slow AM
  traffic   road traffic — lowpassed noise with slow "pass-by" AM swells
  cafe      cafe babble — sum of 6 detuned harmonic "voices" at low level
  speaker   interfering speaker — second harmonic complex, different f0
  tvmusic   TV / music — harmonic chord + percussive-ish filtered bursts
  keyboard  keyboard clicks — short filtered noise bursts
  hum       50 Hz mains hum (+ harmonics 100/150/200 Hz)
  reverb    room reverb + fan — clean convolved with a synthetic RIR
            (exponentially-decaying noise burst, implemented here with
            numpy; pyroomacoustics is *optional* and NOT required), then
            fan noise added at the requested SNR. The reverberant signal
            is the metric reference for this condition.

Clean source: ``speech_like()`` from eval/make_synth.py (imported, not copied).

Metrics:
  * SI-SDR improvement over the input mix.
    SI-SDR = 10*log10(|a*s|^2 / |a*s - s_hat|^2), a = <s_hat,s>/<s,s>.
  * STOI via the `pystoi` library (present in the project venv).
  * PESQ: skipped — the `pesq` package does not build in this environment.
    It is reported as null with a reason, never faked.
  * DNSMOS: skipped — not available offline. Reported as null with a reason.

Pass rules (spec section 7):
  * STOI(out) >= STOI(in) - 0.01            for SNR >= 5 dB
  * SI-SDR improvement > 0                  for SNR >= 0 dB
  (SNR = -5 dB conditions are reported with pass=null — informational.)

Only `standard` mode is evaluated: `music` and `isolate` modes need
multi-GB models (Demucs htdemucs, SpeechBrain ECAPA) that are not part of
the CPU test setup.

Output: per-condition results as JSON (default eval/results.json) plus a
Markdown table printed to stdout. Numbers are only ever measured.

Example:
    cd eval && ../.venv/bin/python run_eval.py --snr 0 5 --noise fan cafe
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time

import numpy as np
import soundfile as sf
from scipy import signal as sig

# ----------------------------------------------------------------------------
# Imports from the repo (read-only use; backend/ is never modified here)
# ----------------------------------------------------------------------------
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "eval"))      # make_synth
sys.path.insert(0, os.path.join(REPO_ROOT, "backend"))   # config, pipeline

from make_synth import speech_like  # noqa: E402
from pipeline.pipeline import run_pipeline  # noqa: E402
from pystoi import stoi  # noqa: E402

SR = 48000
CLIP_S = 6.0
SNR_GRID = (-5, 0, 5, 10, 15)
NOISE_TYPES = ("fan", "traffic", "cafe", "speaker", "tvmusic",
               "keyboard", "hum", "reverb")
PEAK_TARGET = 0.89  # peak-normalize mixes to this, exactly like make_synth


# ----------------------------------------------------------------------------
# Noise synthesis (48 kHz, numpy/scipy only)
# ----------------------------------------------------------------------------
def _norm(x: np.ndarray) -> np.ndarray:
    peak = float(np.max(np.abs(x)))
    return (x / max(peak, 1e-9)).astype(np.float32)


def noise_fan(dur: float, rng: np.random.Generator) -> np.ndarray:
    n = int(dur * SR)
    w = rng.standard_normal(n)
    sos = sig.butter(4, 800, btype="low", fs=SR, output="sos")
    f = sig.sosfiltfilt(sos, w)
    t = np.arange(n) / SR
    f = f * (0.8 + 0.2 * np.sin(2 * np.pi * 0.25 * t))
    return _norm(f)


def noise_traffic(dur: float, rng: np.random.Generator) -> np.ndarray:
    n = int(dur * SR)
    w = rng.standard_normal(n)
    sos = sig.butter(4, 350, btype="low", fs=SR, output="sos")
    f = sig.sosfiltfilt(sos, w)
    t = np.arange(n) / SR
    # slow "pass-by" swells
    am = 0.65 + 0.35 * np.sin(2 * np.pi * 0.13 * t + rng.uniform(0, 2 * np.pi))
    return _norm(f * am)


def noise_cafe(dur: float, rng: np.random.Generator) -> np.ndarray:
    """Sum of 6 detuned harmonic 'voices' at low level (babble)."""
    n = int(dur * SR)
    t = np.arange(n) / SR
    x = np.zeros(n)
    for _ in range(6):
        f0 = float(rng.uniform(85, 230))
        ph = 2 * np.pi * np.cumsum(np.full(n, f0)) / SR
        voice = np.zeros(n)
        for h, a in [(1, 1.0), (2, 0.5), (3, 0.28), (4, 0.15), (5, 0.08)]:
            voice += a * np.sin(h * ph + rng.uniform(0, 2 * np.pi))
        # syllabic-ish AM, random rate/phase per voice
        rate = float(rng.uniform(3.0, 5.0))
        am = 0.5 + 0.5 * np.sin(2 * np.pi * rate * t + rng.uniform(0, 2 * np.pi))
        x += voice * am * float(rng.uniform(0.25, 0.5))
    return _norm(x)


def noise_speaker(dur: float, rng: np.random.Generator) -> np.ndarray:
    """Interfering speaker: a second harmonic complex at a different f0."""
    n = int(dur * SR)
    t = np.arange(n) / SR
    f0 = 195 + 8 * np.sin(2 * np.pi * 0.5 * t + 1.2)
    ph = 2 * np.pi * np.cumsum(f0) / SR
    x = np.zeros(n)
    for h, a in [(1, 1.0), (2, 0.55), (3, 0.3), (4, 0.18), (5, 0.1)]:
        x += a * np.sin(h * ph + rng.uniform(0, 2 * np.pi))
    am = 0.55 + 0.45 * np.sin(2 * np.pi * 3.2 * t + 0.3)
    pauses = (np.sin(2 * np.pi * 0.7 * t + 2.1) > -0.3).astype(float)
    x = x * am * np.clip(pauses, 0.1, 1.0)
    return _norm(x)


def noise_tvmusic(dur: float, rng: np.random.Generator) -> np.ndarray:
    """Harmonic chord + percussive-ish filtered bursts (TV/music bed)."""
    n = int(dur * SR)
    t = np.arange(n) / SR
    x = np.zeros(n)
    for f0 in (110.0, 138.59, 164.81):  # A2, C#3, E3
        ph = 2 * np.pi * np.cumsum(np.full(n, f0)) / SR
        for h, a in [(1, 1.0), (2, 0.6), (3, 0.35), (4, 0.2)]:
            x += 0.4 * a * np.sin(h * ph + rng.uniform(0, 2 * np.pi))
    # percussive hits: bandpassed noise bursts, exponential decay
    sos_bp = sig.butter(2, (200, 4000), btype="band", fs=SR, output="sos")
    n_hits = int(dur * 1.5)
    for _ in range(n_hits):
        s = int(rng.uniform(0, n - int(0.09 * SR)))
        m = int(rng.uniform(0.03, 0.08) * SR)
        burst = rng.standard_normal(m) * np.exp(-np.arange(m) / (0.02 * SR))
        burst = sig.sosfilt(bp := sos_bp, burst)
        x[s:s + m] += burst * 0.5
    return _norm(x)


def noise_keyboard(dur: float, rng: np.random.Generator) -> np.ndarray:
    """Short filtered bursts (key clicks), random timing."""
    n = int(dur * SR)
    x = np.zeros(n)
    sos_hp = sig.butter(4, 3000, btype="high", fs=SR, output="sos")
    n_clicks = int(dur * float(rng.uniform(1.5, 3.0)))
    for _ in range(n_clicks):
        s = int(rng.uniform(0, n - int(0.04 * SR)))
        m = int(rng.uniform(0.015, 0.04) * SR)
        burst = rng.standard_normal(m) * np.exp(-np.arange(m) / (0.008 * SR))
        burst = sig.sosfilt(sos_hp, burst)
        x[s:s + m] += burst * float(rng.uniform(0.5, 1.0))
    return _norm(x)


def noise_hum(dur: float, rng: np.random.Generator) -> np.ndarray:
    n = int(dur * SR)
    t = np.arange(n) / SR
    h = (np.sin(2 * np.pi * 50 * t)
         + 0.5 * np.sin(2 * np.pi * 100 * t)
         + 0.25 * np.sin(2 * np.pi * 150 * t)
         + 0.12 * np.sin(2 * np.pi * 200 * t))
    return _norm(h)


def synthetic_rir(dur_s: float = 0.6, rt60_s: float = 0.45,
                  rng: np.random.Generator | None = None) -> np.ndarray:
    """Exponentially-decaying noise burst RIR (numpy fallback).

    pyroomacoustics is *optional* for this suite — this plain-numpy
    implementation is used instead, so the suite has no extra dependency.
    """
    rng = rng or np.random.default_rng()
    L = int(dur_s * SR)
    tau = rt60_s / (3.0 * np.log(10.0))  # e^(-RT60/tau) = 1e-3  (-60 dB)
    t = np.arange(L) / SR
    rir = rng.standard_normal(L) * np.exp(-t / tau)
    rir[0] += 1.0  # direct path
    return (rir / max(np.abs(rir).max(), 1e-9)).astype(np.float32)


NOISE_FNS = {
    "fan": noise_fan,
    "traffic": noise_traffic,
    "cafe": noise_cafe,
    "speaker": noise_speaker,
    "tvmusic": noise_tvmusic,
    "keyboard": noise_keyboard,
    "hum": noise_hum,
}


# ----------------------------------------------------------------------------
# Mix construction
# ----------------------------------------------------------------------------
def build_mix(noise_type: str, snr_db: float, seed: int):
    """Return (reference_clean, noisy_mix), both float32 at 48 kHz.

    Mixing law (spec section 7):
        noisy = clean + g * noise
        g = sqrt(P_clean / (P_noise * 10^(SNR/10)))
    The mix is peak-normalized to PEAK_TARGET afterwards (scale-preserving
    for SNR; SI-SDR and STOI are both gain-tolerant).
    """
    rng = np.random.default_rng(seed)
    clean = speech_like(CLIP_S, rng).astype(np.float32)

    if noise_type == "reverb":
        # reference = reverberated clean; "noise" = fan at the target SNR
        rir = synthetic_rir(rng=rng)
        ref = sig.fftconvolve(clean, rir, mode="full")[:len(clean)]
        ref = ref / max(np.sqrt(np.mean(ref ** 2)), 1e-9)
        ref = (ref * np.sqrt(np.mean(clean.astype(np.float64) ** 2))
               ).astype(np.float32)
        noise = noise_fan(CLIP_S, rng)
    else:
        ref = clean
        noise = NOISE_FNS[noise_type](CLIP_S, rng)

    pc = float(np.mean(ref.astype(np.float64) ** 2))
    pn = float(np.mean(noise.astype(np.float64) ** 2))
    g = float(np.sqrt(pc / (pn * 10.0 ** (snr_db / 10.0))))
    noisy = ref + g * noise
    noisy = (noisy / max(float(np.max(np.abs(noisy))), 1e-9)
             * PEAK_TARGET).astype(np.float32)
    return ref, noisy


# ----------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------
def si_sdr(ref: np.ndarray, est: np.ndarray) -> float:
    """Scale-invariant SDR, exactly per spec section 7.

        SI-SDR = 10*log10(|a*s|^2 / |a*s - s_hat|^2),  a = <s_hat,s> / |s|^2
    """
    s = ref.astype(np.float64).ravel()
    sh = est.astype(np.float64).ravel()
    n = min(len(s), len(sh))
    s, sh = s[:n], sh[:n]
    a = float(np.dot(sh, s) / max(np.dot(s, s), 1e-12))
    proj = a * s
    num = float(np.dot(proj, proj))
    den = float(np.dot(proj - sh, proj - sh))
    return float(10.0 * np.log10(max(num, 1e-12) / max(den, 1e-12)))


def stoi_score(ref: np.ndarray, est: np.ndarray) -> float:
    """STOI via the pystoi library (48 kHz -> 16 kHz resample first)."""
    n = min(len(ref), len(est))
    r16 = sig.resample_poly(ref[:n].astype(np.float64), 1, 3)
    e16 = sig.resample_poly(est[:n].astype(np.float64), 1, 3)
    return float(stoi(r16, e16, 16000, extended=False))


# ----------------------------------------------------------------------------
# One condition
# ----------------------------------------------------------------------------
def run_condition(noise_type: str, snr_db: float, seed: int,
                  workdir: str, keep_wavs: str | None) -> dict:
    ref, noisy = build_mix(noise_type, snr_db, seed)

    in_path = os.path.join(workdir, f"in_{noise_type}_{snr_db}.wav")
    out_path = os.path.join(workdir, f"out_{noise_type}_{snr_db}.wav")
    sf.write(in_path, noisy, SR, subtype="PCM_16")

    t0 = time.time()
    report = run_pipeline(in_path, out_path, mode="standard")
    elapsed = time.time() - t0

    out, sr = sf.read(out_path, dtype="float32")
    assert sr == SR
    n = min(len(ref), len(out))
    ref, out = ref[:n], out[:n]
    noisy = noisy[:n]

    sdr_in = si_sdr(ref, noisy)
    sdr_out = si_sdr(ref, out)
    sdr_imp = sdr_out - sdr_in
    st_in = stoi_score(ref, noisy)
    st_out = stoi_score(ref, out)

    if snr_db >= 0:
        passed = bool(sdr_imp > 0)
        if snr_db >= 5:
            passed = bool(passed and (st_out >= st_in - 0.01))
    else:
        passed = None  # no pass rule below 0 dB SNR — informational only

    if keep_wavs:
        os.makedirs(keep_wavs, exist_ok=True)
        sf.write(os.path.join(keep_wavs, f"ref_{noise_type}_{snr_db}.wav"),
                 ref, SR, subtype="PCM_16")
        sf.write(os.path.join(keep_wavs, f"noisy_{noise_type}_{snr_db}.wav"),
                 noisy, SR, subtype="PCM_16")
        sf.write(os.path.join(keep_wavs, f"out_{noise_type}_{snr_db}.wav"),
                 out, SR, subtype="PCM_16")

    return {
        "noise": noise_type,
        "snr_db": snr_db,
        "seed": seed,
        "mode": "standard",
        "si_sdr_in_db": round(sdr_in, 2),
        "si_sdr_out_db": round(sdr_out, 2),
        "si_sdr_improvement_db": round(sdr_imp, 2),
        "stoi_in": round(st_in, 4),
        "stoi_out": round(st_out, 4),
        "pesq_in": None,
        "pesq_out": None,
        "dnsmos": None,
        "pass": passed,
        "processing_time_s": round(elapsed, 1),
        "output_lufs": round(float(report.get("output_lufs", 0.0)), 2),
        "notes": ("PESQ skipped: the `pesq` package does not build in this "
                  "environment. DNSMOS skipped: not available offline."),
    }


# ----------------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------------
def markdown_table(results: list[dict]) -> str:
    lines = [
        "| Noise | SNR (dB) | SI-SDR in (dB) | SI-SDR out (dB) | "
        "Δ SI-SDR (dB) | STOI in | STOI out | Pass |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        p = {True: "PASS", False: "FAIL", None: "n/a"}[r["pass"]]
        lines.append(
            f"| {r['noise']} | {r['snr_db']} | {r['si_sdr_in_db']:.2f} | "
            f"{r['si_sdr_out_db']:.2f} | {r['si_sdr_improvement_db']:+.2f} | "
            f"{r['stoi_in']:.4f} | {r['stoi_out']:.4f} | {p} |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="VoiceClean v2 evaluation suite (spec section 7).")
    ap.add_argument("--snr", type=float, nargs="+", default=list(SNR_GRID),
                    help="SNR conditions in dB (default: -5 0 5 10 15)")
    ap.add_argument("--noise", nargs="+", default=list(NOISE_TYPES),
                    choices=NOISE_TYPES,
                    help="noise types to evaluate (default: all)")
    ap.add_argument("--seed", type=int, default=7,
                    help="base RNG seed; each condition uses seed+index")
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, "eval",
                                                  "results.json"),
                    help="JSON results path")
    ap.add_argument("--keep-wavs", default=None, metavar="DIR",
                    help="also save ref/noisy/out wavs to DIR")
    a = ap.parse_args()

    results: list[dict] = []
    with tempfile.TemporaryDirectory(prefix="voiceclean_eval_") as work:
        idx = 0
        for noise_type in a.noise:
            for snr_db in a.snr:
                seed = a.seed + idx
                idx += 1
                tag = f"[{noise_type} @ {snr_db:g} dB]"
                print(f"running {tag} ...", flush=True)
                try:
                    r = run_condition(noise_type, float(snr_db), seed,
                                      work, a.keep_wavs)
                except Exception as e:  # never let one condition kill the grid
                    r = {"noise": noise_type, "snr_db": snr_db, "seed": seed,
                         "mode": "standard", "pass": False,
                         "error": f"{type(e).__name__}: {e}"}
                    print(f"  ERROR: {e}", flush=True)
                results.append(r)
                print(f"  ΔSI-SDR={r.get('si_sdr_improvement_db', '?')} dB, "
                      f"STOI {r.get('stoi_in', '?')}->{r.get('stoi_out', '?')}, "
                      f"pass={r['pass']}", flush=True)
                # incremental write: a crash mid-grid keeps finished rows
                with open(a.out, "w") as f:
                    json.dump({"generated_by": "eval/run_eval.py",
                               "conditions": results}, f, indent=2)

    n_pass = sum(1 for r in results if r["pass"] is True)
    n_fail = sum(1 for r in results if r["pass"] is False)
    n_na = sum(1 for r in results if r["pass"] is None)
    print("\n" + markdown_table(results))
    print(f"\n{a.out} written. PASS={n_pass} FAIL={n_fail} n/a={n_na} "
          f"of {len(results)} conditions.")


if __name__ == "__main__":
    main()
