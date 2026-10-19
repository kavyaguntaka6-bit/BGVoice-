"""Stage 4 — neural denoise (DeepFilterNet3, the main engine).

Genuine pretrained DeepFilterNet3 via the `deepfilternet` pip package.
STFT at 48 kHz, FFT 960, hop 480 (handled inside deepfilterlib).
Strength control uses the native atten_lim_db blend:
    out = lim * x_noisy + (1 - lim) * y_enhanced,  lim = 10^(-A/20),
    A(s) = 10 + 50*s dB.

The model downloads on first use (~100 MB); importing this module
downloads nothing. Long files (>10 min) are processed in 60 s chunks
with 2 s overlap, 1 s warm-up discard and raised-cosine crossfade.
"""

from __future__ import annotations

import sys
import types

import numpy as np
import torch

import config as C

_DF = None  # (model, df_state)


def _shim_torchaudio_backend():
    """deepfilternet 0.5.6 imports torchaudio.backend.common.AudioMetaData,
    removed in torchaudio>=2.4. We never use df.io.load_audio, so a minimal
    shim is sufficient and honest."""
    try:
        import torchaudio.backend.common  # noqa: F401
        return
    except ImportError:
        pass
    import torchaudio
    from collections import namedtuple
    AudioMetaData = namedtuple(
        "AudioMetaData", ["sample_rate", "num_frames", "num_channels",
                          "bits_per_sample", "encoding"])
    common = types.ModuleType("torchaudio.backend.common")
    common.AudioMetaData = AudioMetaData
    backend = types.ModuleType("torchaudio.backend")
    backend.common = common
    sys.modules["torchaudio.backend"] = backend
    sys.modules["torchaudio.backend.common"] = common
    torchaudio.backend = backend


def _load():
    global _DF
    if _DF is None:
        _shim_torchaudio_backend()
        try:
            from df.enhance import enhance, init_df
        except ImportError as e:
            raise RuntimeError(
                "The `deepfilternet` package is not installed. Install it with "
                "`pip install deepfilternet deepfilterlib`.") from e
        try:
            model, df_state, _ = init_df(log_level="WARNING", log_file=None)
        except Exception as e:
            raise RuntimeError(
                f"Could not load the DeepFilterNet3 model: {e}. "
                "The pretrained weights download on first use; "
                "check network access.") from e
        sr = int(model.sr) if hasattr(model, "sr") else C.SAMPLE_RATE
        if sr != C.SAMPLE_RATE:
            raise RuntimeError(
                f"DeepFilterNet model sample rate is {sr}, expected {C.SAMPLE_RATE}.")
        _DF = (model, df_state)
    return _DF


def _enhance_once(x: np.ndarray, atten_db: float) -> np.ndarray:
    model, df_state = _load()
    from df.enhance import enhance
    with torch.no_grad():
        t = torch.from_numpy(x.astype(np.float32)).unsqueeze(0)  # [1, T]
        y = enhance(model, df_state, t, pad=True, atten_lim_db=atten_db)
    y = y.squeeze(0).cpu().numpy().astype(np.float32)
    if len(y) > len(x):
        y = y[:len(x)]
    elif len(y) < len(x):
        y = np.pad(y, (0, len(x) - len(y)))
    return y


def _crossfade(a_tail: np.ndarray, b_head: np.ndarray, n: int) -> np.ndarray:
    t = np.arange(n, dtype=np.float32) / max(n, 1)
    w = 0.5 - 0.5 * np.cos(np.pi * t)  # raised cosine 0 -> 1
    return a_tail * (1 - w) + b_head * w


def enhance_audio(x: np.ndarray, strength: float = C.DEFAULT_STRENGTH,
                  progress=None) -> np.ndarray:
    """Run DeepFilterNet3 with attenuation cap A(strength). Length-preserving."""
    atten_db = C.atten_cap_db(strength)
    sr = C.SAMPLE_RATE
    total = len(x)
    if total <= C.LONG_FILE_MIN_S * sr:
        if progress:
            progress(0.5)
        y = _enhance_once(x, atten_db)
        if progress:
            progress(1.0)
        return y

    # Chunked inference for long files
    chunk = int(C.DF_CHUNK_S * sr)
    stride = chunk - int(C.DF_OVERLAP_S * sr)
    warm = int(C.DF_WARMUP_S * sr)
    ov = int(C.DF_OVERLAP_S * sr)
    out = np.zeros_like(x)
    # first chunk
    n_chunks = max(1, 1 + (total - chunk + ov - 1) // max(stride, 1))
    prev_tail = None
    for i in range(n_chunks):
        start = i * stride
        seg_start = max(0, start - (warm if i > 0 else 0))
        seg_end = min(total, start + chunk)
        seg = x[seg_start:seg_end]
        y = _enhance_once(seg, atten_db)
        drop = (start - seg_start)  # warm-up to discard
        y = y[drop:]
        cur_start = start
        cur_end = min(total, start + chunk - drop)
        if prev_tail is not None:
            n_ov = min(ov, len(y), cur_end - cur_start)
            if n_ov > 0:
                y[:n_ov] = _crossfade(prev_tail[-n_ov:], y[:n_ov], n_ov)
        copy_end = min(cur_end, total)
        out[cur_start:copy_end] = y[:copy_end - cur_start]
        if cur_end < total:
            prev_tail = y[copy_end - cur_start:copy_end - cur_start + ov]
        else:
            prev_tail = None
        if progress:
            progress((i + 1) / n_chunks)
        if cur_end >= total:
            break
    return out
