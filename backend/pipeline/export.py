"""Stage 9b — export.

WAV 48 kHz 24-bit; MP3 320 kbps; MP4 (video stream copied untouched,
audio replaced with AAC 256k, faststart, -shortest). Length-preserving:
assert len(out) == len(in) so A/V stays in sync. 16-bit export uses
TPDF dither.
"""

from __future__ import annotations

import os
import subprocess

import numpy as np
import soundfile as sf

import config as C


def _run(cmd: list[str]) -> None:
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"ffmpeg export failed: {p.stderr.strip()[-500:]}")


def tpdf_dither_to_16bit(x: np.ndarray) -> np.ndarray:
    r1 = np.random.uniform(-0.5, 0.5, size=x.shape)
    r2 = np.random.uniform(-0.5, 0.5, size=x.shape)
    dither = (r1 - r2) / 32768.0
    y = np.clip(x + dither, -1.0, 1.0)
    return (y * 32767.0).round().astype(np.int16)


def write_wav24(x: np.ndarray, path: str) -> None:
    x = np.clip(x, -1.0, 1.0).astype(np.float32)
    sf.write(path, x, C.EXPORT_WAV_SR, subtype="PCM_24")


def write_mp3(wav_path: str, mp3_path: str) -> None:
    _run(["ffmpeg", "-v", "error", "-y", "-i", wav_path,
          "-c:a", "libmp3lame", "-b:a", C.MP3_BITRATE, mp3_path])


def write_mp4(src_video: str, clean_wav: str, out_mp4: str,
              in_len: int) -> None:
    _run(["ffmpeg", "-v", "error", "-y", "-i", src_video, "-i", clean_wav,
          "-map", "0:v:0", "-map", "1:a:0",
          "-c:v", "copy", "-c:a", "aac", "-b:a", C.AAC_BITRATE,
          "-movflags", "+faststart", "-shortest", out_mp4])
    if not os.path.isfile(out_mp4) or os.path.getsize(out_mp4) == 0:
        raise RuntimeError("mp4 export produced no output")


def assert_length_parity(out: np.ndarray, in_len: int) -> np.ndarray:
    if len(out) != in_len:
        raise RuntimeError(
            f"pipeline changed length: {len(out)} != {in_len}")
    return out
