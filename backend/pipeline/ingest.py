"""Stage 0 — ingest.

Single decode via ffmpeg to 48 kHz mono float32. Records input LUFS and the
noise floor (10th-percentile frame RMS, dBFS) for the report.
"""

from __future__ import annotations

import os
import subprocess
import tempfile

import numpy as np
import pyloudnorm
import soundfile as sf

import config as C


class IngestError(ValueError):
    """Raised for corrupt / empty / unsupported / oversized inputs."""


def _run(cmd: list[str]) -> None:
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise IngestError(f"ffmpeg failed: {p.stderr.strip()[-500:]}")


def _probe_duration(path: str) -> float:
    p = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", path],
        capture_output=True, text=True,
    )
    try:
        return float(p.stdout.strip())
    except ValueError:
        return 0.0


def _has_video_stream(path: str) -> bool:
    p = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=index", "-of", "csv=p=0", path],
        capture_output=True, text=True,
    )
    return bool(p.stdout.strip())


def _decode(path: str, tmp_wav: str, channels: int) -> None:
    _run([
        "ffmpeg", "-v", "error", "-y", "-i", path,
        "-vn", "-ac", str(channels), "-ar", str(C.SAMPLE_RATE),
        "-c:a", C.INGEST_CODEC, tmp_wav,
    ])


def _noise_floor_db(x: np.ndarray) -> float:
    n, h = C.NOISE_FLOOR_FRAME, C.NOISE_FLOOR_HOP
    if len(x) < n:
        frames = x.reshape(1, -1)
    else:
        n_frames = 1 + (len(x) - n) // h
        idx = np.arange(n)[None, :] + h * np.arange(n_frames)[:, None]
        frames = x[idx]
    rms = np.sqrt(np.mean(frames ** 2, axis=1) + 1e-12)
    pct = np.percentile(rms, C.NOISE_FLOOR_PERCENTILE)
    return float(20.0 * np.log10(max(pct, 1e-9)))


def ingest(path: str, work_dir: str) -> dict:
    """Decode `path` once. Returns dict with audio + measurements.

    Keys: audio (float32 mono @48k), duration_s, input_lufs, noise_floor_db,
          is_video, src_path, n_in_channels.
    """
    if not os.path.isfile(path):
        raise IngestError(f"input not found: {path}")
    size_mb = os.path.getsize(path) / 1e6
    if size_mb > C.MAX_FILE_MB:
        raise IngestError(f"file too large: {size_mb:.0f} MB > {C.MAX_FILE_MB} MB")

    is_video = _has_video_stream(path)

    tmp_stereo = os.path.join(work_dir, "ingest_stereo.wav")
    _decode(path, tmp_stereo, channels=2)
    data, sr = sf.read(tmp_stereo, dtype="float32", always_2d=True)
    os.remove(tmp_stereo)
    if data.size == 0:
        raise IngestError("empty audio stream")

    n_in_channels = data.shape[1]
    if n_in_channels == 2:
        rms0 = float(np.sqrt(np.mean(data[:, 0] ** 2) + 1e-12))
        rms1 = float(np.sqrt(np.mean(data[:, 1] ** 2) + 1e-12))
        imbalance = 20.0 * np.log10(max(rms0, rms1) / max(min(rms0, rms1), 1e-9))
        if imbalance > C.STEREO_IMBALANCE_DB:
            mono = data[:, 0] if rms0 >= rms1 else data[:, 1]
        else:
            mono = data.mean(axis=1)
    else:
        mono = data[:, 0]

    x = np.asarray(mono, dtype=np.float32)
    x = x - float(np.mean(x))  # remove DC

    duration_s = len(x) / C.SAMPLE_RATE
    if duration_s > C.MAX_DURATION_MIN * 60:
        raise IngestError(
            f"audio too long: {duration_s/60:.1f} min > {C.MAX_DURATION_MIN} min")

    meter = pyloudnorm.Meter(C.SAMPLE_RATE)
    try:
        input_lufs = float(meter.integrated_loudness(x))
    except Exception:
        input_lufs = float("-inf")

    return {
        "audio": x,
        "duration_s": duration_s,
        "input_lufs": input_lufs,
        "noise_floor_db": _noise_floor_db(x),
        "is_video": is_video,
        "src_path": path,
        "n_in_channels": n_in_channels,
    }
