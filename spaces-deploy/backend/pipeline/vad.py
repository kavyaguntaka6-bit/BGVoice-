"""Stage 2 — VAD (Silero, lazy-loaded).

Produces a sample-level boolean speech mask at 48 kHz plus a helper to map
it onto arbitrary STFT framings. The model downloads (~2 MB) on first use;
importing this module downloads nothing.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.signal import resample_poly

import config as C

_VAD_MODEL = None
_VAD_UTILS = None


def _load():
    global _VAD_MODEL, _VAD_UTILS
    if _VAD_MODEL is None:
        try:
            _VAD_MODEL, _VAD_UTILS = torch.hub.load(
                "snakers4/silero-vad", "silero_vad", trust_repo=True)
        except Exception as e:
            raise RuntimeError(
                f"Could not load Silero VAD model: {e}. "
                "Check network access to github.com / torch.hub cache.") from e
        _VAD_MODEL.eval()
    return _VAD_MODEL, _VAD_UTILS


def _apply_hangover(prob: np.ndarray, sr: int) -> np.ndarray:
    """Threshold + min-speech / min-silence / padding in the time domain."""
    speech = prob >= C.VAD_THRESHOLD
    min_speech = int(C.VAD_MIN_SPEECH_MS / 1000 * sr)
    min_sil = int(C.VAD_MIN_SILENCE_MS / 1000 * sr)
    pad = int(C.VAD_PAD_MS / 1000 * sr)

    # remove short speech bursts
    out = speech.copy()
    i = 0
    n = len(out)
    while i < n:
        if out[i]:
            j = i
            while j < n and out[j]:
                j += 1
            if j - i < min_speech:
                out[i:j] = False
            i = j
        else:
            i += 1
    # fill short silences
    i = 0
    while i < n:
        if not out[i]:
            j = i
            while j < n and not out[j]:
                j += 1
            if j - i < min_sil and i > 0 and j < n:
                out[i:j] = True
            i = j
        else:
            i += 1
    if pad:
        idx = np.nonzero(out)[0]
        if idx.size:
            lo = max(0, idx.min() - pad)
            hi = min(n, idx.max() + pad + 1)
            out[lo:hi] = True
    return out


def compute_vad_mask(x48: np.ndarray) -> np.ndarray:
    """Boolean speech mask at 48 kHz, same length as x48."""
    model, utils = _load()
    (get_speech_timestamps, _, _, _, _) = utils
    x16 = resample_poly(x48, 16000, C.SAMPLE_RATE).astype(np.float32)
    chunk = C.VAD_CHUNK_SAMPLES
    probs = []
    with torch.no_grad():
        for i in range(0, len(x16), chunk):
            seg = x16[i:i + chunk]
            if len(seg) < chunk:
                seg = np.pad(seg, (0, chunk - len(seg)))
            t = torch.from_numpy(seg)
            # silero_vad returns per-chunk speech probability
            p = model(t, C.VAD_INTERNAL_SR).item()
            probs.extend([p] * min(chunk, len(x16) - i))
    probs = np.asarray(probs, dtype=np.float32)
    mask16 = _apply_hangover(probs, C.VAD_INTERNAL_SR)
    # upsample mask to 48 kHz (nearest neighbour)
    idx = (np.arange(len(x48)) * len(mask16) // len(x48)).astype(int)
    idx = np.clip(idx, 0, len(mask16) - 1)
    return mask16[idx]


def vad_frame_mask(mask48: np.ndarray, hop: int, n_frames: int,
                   n_fft: int | None = None) -> np.ndarray:
    """Map the sample mask onto STFT frames: majority vote over each frame's
    hop window (centred like librosa/scipy framing). Returns float {0,1}."""
    wins = []
    for l in range(n_frames):
        c = l * hop
        lo = max(0, c - hop // 2)
        hi = min(len(mask48), c + hop // 2)
        wins.append(mask48[lo:hi].mean() >= 0.5 if hi > lo else 0.0)
    return np.asarray(wins, dtype=np.float32)
