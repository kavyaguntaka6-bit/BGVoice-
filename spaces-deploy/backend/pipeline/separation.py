"""Stage 3 — pre-separation by mode.

- music mode: Demucs htdemucs two-stem, vocals stem feeds stage 4.
- isolate mode: target-speaker gain from ECAPA-TDNN embeddings, wrapped in
  TargetSpeakerExtractor(audio, e_ref) -> audio so a neural TSE model can
  replace the gain-based implementation later.

Heavy models are lazy-loaded; importing this module downloads nothing.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.signal import resample_poly

import config as C

_DEMUCS = None
_ECAPA = None


# ----------------------------------------------------------------------------
# Demucs (music / background-track removal)
# ----------------------------------------------------------------------------
def _load_demucs():
    global _DEMUCS
    if _DEMUCS is None:
        try:
            from demucs.pretrained import get_model
            from demucs.apply import apply_model
            from demucs.audio import convert_audio
        except ImportError as e:
            raise RuntimeError(
                "Demucs is not installed. Install it with "
                "`pip install demucs` to use mode='music'.") from e
        _DEMUCS = (get_model, apply_model, convert_audio)
    return _DEMUCS


def separate_vocals(x48: np.ndarray, device: str = "cpu") -> np.ndarray:
    """Return the vocals stem (48 kHz float32 mono) of the mixture."""
    get_model, apply_model, convert_audio = _load_demucs()
    model = get_model(C.DEMUCS_MODEL)
    model.to(device).eval()
    wav = torch.from_numpy(x48).unsqueeze(0)  # (1, T)
    wav = convert_audio(wav, C.SAMPLE_RATE, model.samplerate, model.audio_channels)
    with torch.no_grad():
        sources = apply_model(model, wav.to(device), device=device)[0]
    # sources: (4, channels, T) in demucs source order
    vocals_idx = model.sources.index("vocals")
    vocals = sources[vocals_idx].mean(dim=0).cpu().numpy().astype(np.float32)
    if model.samplerate != C.SAMPLE_RATE:
        vocals = resample_poly(vocals, C.SAMPLE_RATE, model.samplerate).astype(np.float32)
    # length-align
    if len(vocals) > len(x48):
        vocals = vocals[:len(x48)]
    elif len(vocals) < len(x48):
        vocals = np.pad(vocals, (0, len(x48) - len(vocals)))
    return vocals


# ----------------------------------------------------------------------------
# Target-speaker isolation (ECAPA-TDNN embeddings)
# ----------------------------------------------------------------------------
def _load_ecapa():
    global _ECAPA
    if _ECAPA is None:
        try:
            from speechbrain.inference.speaker import EncoderClassifier
        except ImportError as e:
            raise RuntimeError(
                "SpeechBrain is not installed. Install it with "
                "`pip install speechbrain` to use mode='isolate'.") from e
        try:
            _ECAPA = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb", run_opts={"device": "cpu"})
        except Exception as e:
            raise RuntimeError(
                f"Could not load ECAPA-TDNN model: {e}.") from e
    return _ECAPA


def _embed_windows(encoder, x48: np.ndarray, win_s: float, hop_s: float) -> np.ndarray:
    """L2-normalized embeddings for sliding windows. Returns (n_win, dim)."""
    win = int(win_s * C.SAMPLE_RATE)
    hop = int(hop_s * C.SAMPLE_RATE)
    embs = []
    for start in range(0, max(1, len(x48) - win + 1), hop):
        seg = x48[start:start + win]
        if len(seg) < win:
            seg = np.pad(seg, (0, win - len(seg)))
        t = torch.from_numpy(seg).unsqueeze(0)
        with torch.no_grad():
            e = encoder.encode_batch(t).squeeze().cpu().numpy()
        e = e / (np.linalg.norm(e) + 1e-12)
        embs.append(e)
    return np.asarray(embs, dtype=np.float32)


def reference_embedding(x48: np.ndarray, vad_mask: np.ndarray | None,
                        enroll48: np.ndarray | None = None) -> np.ndarray:
    """e_ref: mean of L2-normalized ECAPA embeddings, re-normalized."""
    encoder = _load_ecapa()
    if enroll48 is not None:
        src, span = enroll48, None
    else:
        # auto-pick highest-SNR 10 s span where v=1
        span_n = int(C.SPK_AUTOREF_SPAN_S * C.SAMPLE_RATE)
        if vad_mask is None:
            src = x48[:span_n]
        else:
            best, best_e = 0, -np.inf
            for s in range(0, max(1, len(x48) - span_n + 1), span_n // 2):
                seg = x48[s:s + span_n]
                m = vad_mask[s:s + span_n]
                speech_e = float(np.mean(seg[m] ** 2) + 1e-12) if m.any() else 1e-12
                noise_e = float(np.mean(seg[~m] ** 2) + 1e-12) if (~m).any() else 1e-12
                snr = speech_e / noise_e
                if snr > best_e:
                    best_e, best = snr, s
            src = x48[best:best + span_n]
    embs = _embed_windows(encoder, src, C.SPK_EMBED_WINDOW_S, C.SPK_EMBED_WINDOW_S / 2)
    e_ref = embs.mean(axis=0)
    return e_ref / (np.linalg.norm(e_ref) + 1e-12)


def _smooth_ramps(g: np.ndarray, ramp_n: int) -> np.ndarray:
    """Enforce minimum ramp length on gain transitions (no clicks)."""
    if ramp_n < 2:
        return g
    # smooth with a moving-average of length ramp_n (raises short dips)
    kernel = np.ones(ramp_n, dtype=np.float32) / ramp_n
    return np.convolve(g, kernel, mode="same").astype(np.float32)


class TargetSpeakerExtractor:
    """Gain-based target-speaker extraction, swappable for a neural TSE.

    Interface: TargetSpeakerExtractor(audio, e_ref) -> audio.
    A neural target-speaker-extraction model (VoiceFilter / SpeakerBeam /
    WeSep style, conditioned on e_ref) can replace the internals without
    changing callers.
    """

    def __init__(self):
        self.encoder = _load_ecapa()

    def __call__(self, x48: np.ndarray, e_ref: np.ndarray) -> np.ndarray:
        wins = _embed_windows(self.encoder, x48, C.SPK_WINDOW_S, C.SPK_HOP_S)
        sims = wins @ e_ref  # cosine similarity (both L2-normalized)
        p = 1.0 / (1.0 + np.exp(-(sims - C.SPK_THETA) / C.SPK_TAU))
        # median filter over 5 windows
        k = C.SPK_MEDIAN_WIN
        pad = k // 2
        pp = np.pad(p, (pad, pad), mode="edge")
        p_med = np.median(
            np.stack([pp[i:i + len(p)] for i in range(k)], axis=1), axis=1)
        # interpolate to sample rate
        hop = int(C.SPK_HOP_S * C.SAMPLE_RATE)
        centers = (np.arange(len(p_med)) * hop + hop // 2).astype(int)
        grid = np.clip(centers, 0, len(x48) - 1)
        p_up = np.interp(np.arange(len(x48)), grid, p_med).astype(np.float32)
        floor = 10.0 ** (C.SPK_GAIN_FLOOR_DB / 20.0)
        g = np.maximum(p_up, floor)
        g = _smooth_ramps(g, int(C.SPK_RAMP_MS / 1000 * C.SAMPLE_RATE))
        return (x48 * g).astype(np.float32)
