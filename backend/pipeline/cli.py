"""CLI: python -m pipeline.cli in.wav out.wav --mode standard

Run from the backend/ directory (or with backend/ on PYTHONPATH).
"""

from __future__ import annotations

import argparse
import json
import sys

import config as C
from .pipeline import run_pipeline


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="VoiceClean v2 — remove background sound, keep the voice.")
    ap.add_argument("input", help="input audio/video file")
    ap.add_argument("output", help="output WAV path (48 kHz 24-bit)")
    ap.add_argument("--mode", default="standard", choices=C.MODES)
    ap.add_argument("--strength", type=float, default=C.DEFAULT_STRENGTH,
                    help="0..1, default 0.6 (attenuation cap 10+50*s dB)")
    ap.add_argument("--loudness-target", type=float, default=C.LOUDNESS_TARGET,
                    choices=C.LOUDNESS_OPTIONS)
    ap.add_argument("--room-tone", default=C.DEFAULT_ROOM_TONE,
                    choices=("off", "subtle", "natural"))
    ap.add_argument("--dereverb", dest="dereverb", action="store_true",
                    default=None, help="force dereverb on")
    ap.add_argument("--no-dereverb", dest="dereverb", action="store_false",
                    help="force dereverb off")
    ap.add_argument("--enroll", default=None,
                    help="voice sample for isolate mode")
    args = ap.parse_args(argv)

    def progress(stage: str, frac: float):
        print(f"[{frac*100:5.1f}%] {stage}", flush=True)

    try:
        report = run_pipeline(
            args.input, args.output, mode=args.mode,
            strength=args.strength, loudness_target=args.loudness_target,
            room_tone=args.room_tone, dereverb_on=args.dereverb,
            enroll_path=args.enroll, progress=progress)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
