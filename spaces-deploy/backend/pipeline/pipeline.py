"""VoiceClean v2 pipeline orchestrator — exact stage order per spec section 3.

Modes (spec section 4):
  standard: 0,1,2,4,6,7,9
  music:    0,1,3(demucs),4,6,7,9
  isolate:  0,1,2,3(speaker),4,6,7,9
  studio:   0,1,2,4,5,6,7,8,9   (dereverb on, tone/dynamics on, s=0.8)
"""

from __future__ import annotations

import os
import tempfile
import time

import numpy as np

import config as C
from .ingest import ingest
from .filters import linear_cleanup
from .vad import compute_vad_mask, vad_frame_mask
from .separation import separate_vocals, reference_embedding, TargetSpeakerExtractor
from .denoise import enhance_audio
from .dereverb import dereverb
from .postfilter import mmse_lsa_filter
from .roomtone import add_room_tone
from .tone import studio_tone
from .loudness import normalize_loudness, true_peak_limit, integrated_lufs
from .export import write_wav24, assert_length_parity


def _stages_for(mode: str, dereverb_on: bool) -> list[str]:
    if mode == "standard":
        return ["ingest", "linear", "vad", "denoise", "postfilter",
                "roomtone", "loudness", "export"]
    if mode == "music":
        return ["ingest", "linear", "separate_music", "vad", "denoise",
                "postfilter", "roomtone", "loudness", "export"]
    if mode == "isolate":
        return ["ingest", "linear", "vad", "separate_speaker", "denoise",
                "postfilter", "roomtone", "loudness", "export"]
    if mode == "studio":
        stages = ["ingest", "linear", "vad", "denoise"]
        if dereverb_on:
            stages.append("dereverb")
        stages += ["postfilter", "roomtone", "tone", "loudness", "export"]
        return stages
    raise ValueError(f"unknown mode: {mode!r} (choose from {C.MODES})")


def run_pipeline(input_path: str, output_wav: str, *,
                 mode: str = "standard",
                 strength: float = C.DEFAULT_STRENGTH,
                 loudness_target: float = C.LOUDNESS_TARGET,
                 room_tone: str = C.DEFAULT_ROOM_TONE,
                 dereverb_on: bool | None = None,
                 enroll_path: str | None = None,
                 progress=None) -> dict:
    """Run the full pipeline. Returns a report dict with measured values."""
    if mode not in C.MODES:
        raise ValueError(f"unknown mode: {mode!r}")
    if room_tone not in ("off", "subtle", "natural"):
        raise ValueError("room_tone must be off|subtle|natural")
    if dereverb_on is None:
        dereverb_on = (mode == "studio")
    if mode == "studio":
        strength = C.STUDIO_STRENGTH

    stages = _stages_for(mode, dereverb_on)
    t_start = time.time()

    def emit(name: str, frac: float):
        if progress:
            progress(name, float(frac))

    report: dict = {"mode": mode, "strength": strength,
                    "loudness_target": loudness_target,
                    "room_tone": room_tone, "dereverb": dereverb_on,
                    "stages": stages}
    n = len(stages)

    with tempfile.TemporaryDirectory(prefix="voiceclean_") as work:
        for i, stage in enumerate(stages):
            base = i / n
            emit(stage, base)

            if stage == "ingest":
                ing = ingest(input_path, work)
                x = ing["audio"]
                in_len = len(x)
                x_input = x.copy()
                report.update(duration_s=ing["duration_s"],
                              input_lufs=ing["input_lufs"],
                              noise_floor_before_db=ing["noise_floor_db"])
            elif stage == "linear":
                x, info = linear_cleanup(x)
                report["hum_notches_hz"] = info["hum_notches_hz"]
            elif stage == "separate_music":
                x = separate_vocals(x)
            elif stage == "vad":
                v48 = compute_vad_mask(x)
            elif stage == "separate_speaker":
                enroll48 = None
                if enroll_path:
                    en = ingest(enroll_path, work)
                    enroll48 = en["audio"]
                e_ref = reference_embedding(x, v48, enroll48)
                x = TargetSpeakerExtractor()(x, e_ref)
                v48 = compute_vad_mask(x)  # re-run VAD on isolated voice
            elif stage == "denoise":
                x = enhance_audio(
                    x, strength,
                    progress=lambda f: emit("denoise", base + f / n))
            elif stage == "dereverb":
                x = dereverb(x)
            elif stage == "postfilter":
                vf = vad_frame_mask(v48, C.PF_HOP,
                                    1 + len(x) // C.PF_HOP + 1)
                x = mmse_lsa_filter(x, vf)
            elif stage == "roomtone":
                x = add_room_tone(x, x_input, v48, level=room_tone)
            elif stage == "tone":
                x = studio_tone(x, v48)
            elif stage == "loudness":
                x = normalize_loudness(x, loudness_target)
                x = true_peak_limit(x)
            elif stage == "export":
                x = assert_length_parity(x, in_len)
                os.makedirs(os.path.dirname(os.path.abspath(output_wav)),
                            exist_ok=True)
                write_wav24(x, output_wav)
                report["output_lufs"] = integrated_lufs(x)
                # noise floor after: 10th-pct frame RMS on the OUTPUT
                from .ingest import _noise_floor_db
                report["noise_floor_after_db"] = _noise_floor_db(x)
            emit(stage, (i + 1) / n)

    report["processing_time_s"] = time.time() - t_start
    report["output_path"] = output_wav
    return report
