"""Smoke tests for the VoiceClean v2 pipeline.

Run: cd backend && PYTHONPATH=. ../.venv/bin/python -m pytest tests/ -x -q
(Requires the models to be downloadable; DeepFilterNet3 + Silero VAD.)
"""

import os
import sys

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.pipeline import run_pipeline
import config as C


def _make_noisy(path="/tmp/vc_pytest_noisy.wav", dur=4.0, seed=3):
    sr = C.SAMPLE_RATE
    n = int(sr * dur)
    t = np.arange(n) / sr
    rng = np.random.default_rng(seed)
    f0 = 130 + 10 * np.sin(2 * np.pi * 0.4 * t)
    ph = 2 * np.pi * np.cumsum(f0) / sr
    clean = sum(a * np.sin(h * ph) for h, a in
                [(1, 1.0), (2, 0.5), (3, 0.3), (4, 0.2)])
    clean = (clean / np.abs(clean).max() * 0.6).astype(np.float32)
    noise = rng.standard_normal(n).astype(np.float32) * 0.1
    g = np.sqrt(np.mean(clean ** 2) / (np.mean(noise ** 2) * 10 ** (5 / 10)))
    noisy = clean + g * noise
    noisy = (noisy / np.abs(noisy).max() * 0.89).astype(np.float32)
    sf.write(path, noisy, sr, subtype="PCM_16")
    return path, clean


def test_standard_mode_end_to_end(tmp_path):
    noisy_path, clean = _make_noisy()
    out = str(tmp_path / "out.wav")
    report = run_pipeline(noisy_path, out, mode="standard")
    assert os.path.isfile(out)
    y, sr = sf.read(out, dtype="float32")
    assert sr == C.SAMPLE_RATE
    assert len(y) == len(clean), "length must be preserved"
    assert np.isfinite(y).all()
    assert np.abs(y).max() > 0.01, "output is not silent"
    assert np.abs(y).max() <= 1.0, "output clips"
    assert report["output_lufs"] > -17.5 and report["output_lufs"] < -14.5
    assert (report["noise_floor_after_db"]
            < report["noise_floor_before_db"])


def test_invalid_mode():
    try:
        run_pipeline("/tmp/vc_pytest_noisy.wav", "/tmp/x.wav", mode="nope")
    except ValueError:
        return
    raise AssertionError("expected ValueError for bad mode")


def test_missing_input():
    try:
        run_pipeline("/tmp/does_not_exist_xyz.wav", "/tmp/x.wav")
    except Exception:
        return
    raise AssertionError("expected an error for missing input")
