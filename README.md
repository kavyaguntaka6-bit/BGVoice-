# VoiceClean v2

Remove background sound, keep the voice. A 10-stage speech-enhancement
pipeline: classical DSP where it's exact, pretrained neural models where
it matters, and honest loudness/export handling at the end.

**Status:** the processing pipeline (`backend/pipeline/`) and this
evaluation suite are complete and tested. The FastAPI service
(`backend/app/`) and the React frontend (`frontend/`) are being built in
parallel — the API section below documents the endpoint contract they
implement.

## How it works

Four modes, one orchestrator (`backend/pipeline/pipeline.py`):

| Mode       | Stages                                                    |
|------------|-----------------------------------------------------------|
| `standard` | ingest → linear → VAD → denoise → postfilter → roomtone → loudness → export |
| `music`    | … with Demucs vocal separation before the VAD             |
| `isolate`  | … with target-speaker isolation (SpeechBrain ECAPA)       |
| `studio`   | standard + WPE dereverb + tone/dynamics (strength 0.8)    |

### Architecture

```
                        ┌──────────────────────────────────────────────┐
 input file             │              VoiceClean v2 pipeline          │
 (wav/mp3/mp4/…)        │                                              │
     │                  │  0. ingest      decode→48 kHz mono, validate │
     ▼                  │  1. linear      80 Hz HP + hum notch (zero-phase)
┌──────────┐            │  2. vad         Silero VAD speech mask (16 kHz)│
│  decode  │            │  3. separation  Demucs htdemucs / ECAPA speaker│
│ 48 kHz   │            │     (music & isolate modes only)             │
│  mono    │            │  4. denoise     DeepFilterNet3 (main engine)   │
└──────────┘            │  5. dereverb    WPE (nara-wpe, studio mode)    │
     │                  │  6. postfilter  MMSE-LSA, VAD-guided          │
     ▼                  │  7. roomtone    subtle synthetic bed (-68 dBFS)│
┌──────────┐            │  8. tone        presence/de-ess/compressor     │
│ cleanup  │            │     (studio mode only)                       │
│ + neural │            │  9. loudness    -16 LUFS, true-peak -1 dBTP,  │
│ + DSP    │            │     export 48 kHz 24-bit WAV                 │
└──────────┘            └──────────────────────────────────────────────┘
     │
     ▼
 clean.wav  (48 kHz, 24-bit, -16 LUFS)
```

Every tunable constant lives in `backend/config.py` — no stage hard-codes
DSP/ML numbers. Strength `s ∈ [0,1]` maps to an attenuation cap
`A(s) = 10 + 50·s` dB blended as `out = lim·noisy + (1-lim)·enhanced`.

## Setup

Requirements: Python 3.11, `ffmpeg` on PATH.

```bash
cd voiceclean-v2
python3.11 -m venv .venv
source .venv/bin/activate
# torch CPU wheels come from the PyTorch CPU index:
pip install --extra-index-url https://download.pytorch.org/whl/cpu \
    -r backend/requirements.txt
```

`requirements.txt` is a frozen snapshot of the tested environment
(`pip freeze`), including `torch==2.14.1+cpu` / `torchaudio==2.11.0+cpu`.

First run downloads models (needs network once, then cached):
- DeepFilterNet3 weights → `~/.cache/DeepFilterNet/` (~100 MB)
- Silero VAD → `~/.cache/torch/hub/` (~2 MB)
- Demucs `htdemucs` / SpeechBrain ECAPA — only on first `music`/`isolate`
  run (multi-GB; not needed for `standard`/`studio`).

## CLI

```bash
cd backend
PYTHONPATH=. python -m pipeline.cli in.wav out.wav --mode standard
PYTHONPATH=. python -m pipeline.cli in.mp4 out.wav --mode studio \
    --strength 0.8 --loudness-target -14 --room-tone natural
PYTHONPATH=. python -m pipeline.cli noisy.wav out.wav --mode isolate \
    --enroll myvoice.wav
```

Options: `--mode {standard,music,isolate,studio}`, `--strength 0..1`
(default 0.6), `--loudness-target {-14,-16,-23}` (default -16),
`--room-tone {off,subtle,natural}` (default subtle),
`--dereverb/--no-dereverb` (default: on only in studio),
`--enroll` (voice sample for isolate mode).

The CLI prints a JSON report (measured input/output LUFS, noise floor
before/after, hum notches found, per-stage timing).

## API

The FastAPI service (`backend/app/`, built in parallel) exposes:

| Method | Endpoint                      | Description                              |
|--------|-------------------------------|------------------------------------------|
| POST   | `/api/jobs`                   | Upload audio (`multipart/form-data`), returns job id |
| GET    | `/api/jobs/{id}`              | Job status + progress                    |
| GET    | `/api/jobs/{id}/download`     | Download the finished WAV                |
| DELETE | `/api/jobs/{id}`              | Cancel/delete a job and its files        |

```bash
curl -F "file=@in.wav" -F "mode=standard" http://localhost:8000/api/jobs
curl http://localhost:8000/api/jobs/<id>
curl -O http://localhost:8000/api/jobs/<id>/download
curl -X DELETE http://localhost:8000/api/jobs/<id>
```

Jobs run as FastAPI `BackgroundTasks` with a SQLite job store
(`JOB_DB_PATH`, 1-hour TTL per `backend/config.py`); uploads and results
live under `UPLOAD_DIR` / `RESULT_DIR`.

## Frontend

The React UI (`frontend/`, built in parallel) is a single-page app:
upload → mode/strength controls → progress → waveform preview → download.
It talks to the API above (same origin via `/api` proxy in production).

```bash
cd frontend
npm install
npm run dev      # dev server
npm run build    # static production build
```

## Docker

```bash
# backend image (multi-stage; ffmpeg; torch CPU; models pre-warmed at build)
docker build -t voiceclean-v2-backend .

# full stack (backend :8000, frontend :5173)
docker compose up --build
```

The image pre-downloads DeepFilterNet3 and Silero VAD at build time, so
the first request needs no network. `docker-compose.yml` persists the
model cache (`model-cache` volume) and job data (`job-data` volume).
Redis is **not** included — the API's BackgroundTasks + SQLite job queue
needs no broker (see the compose file for the reasoning).

Environment knobs: `OMP_NUM_THREADS` (torch CPU threads),
`JOB_DB_PATH` / `UPLOAD_DIR` / `RESULT_DIR` (default to `/data/...` in
compose).

## Evaluation

`eval/` contains a synthetic evaluation suite (spec section 7):

```bash
cd eval
../.venv/bin/python run_eval.py --snr 0 5 --noise fan cafe --out results.json
../.venv/bin/python make_synth.py --dur 8 --snr 5 \
    --out-noisy noisy.wav --out-clean clean.wav
```

`run_eval.py` builds mixes as `noisy = clean + g·noise` with
`g = sqrt(P_clean / (P_noise · 10^(SNR/10)))`, SNR ∈ {-5, 0, 5, 10, 15} dB,
across 8 conditions (fan, traffic, cafe babble, interfering speaker,
TV/music, keyboard clicks, 50 Hz hum, and room reverb via a numpy
synthetic RIR — pyroomacoustics is optional and not used). Clean speech
comes from `make_synth.py`'s `speech_like()`. Only `standard` mode is
evaluated (`music`/`isolate` need multi-GB models).

Metrics per condition: **SI-SDR improvement** over the input mix
(`10·log10(|a·s|²/|a·s − ŝ|²)`, `a = ⟨ŝ,s⟩/|s|²`), **STOI** (via the
`pystoi` library, 16 kHz). PESQ is skipped — the `pesq` package does not
build in this environment — and DNSMOS is unavailable offline; both are
reported as null with a reason, never faked. Results are written as JSON
(incrementally, so a crash keeps finished rows) and a Markdown table is
printed to stdout.

Pass rules: SI-SDR improvement > 0 dB for SNR ≥ 0 dB;
STOI(out) ≥ STOI(in) − 0.01 for SNR ≥ 5 dB. SNR = −5 dB rows are
informational only.

## Model licences

| Component | Artefact used | Licence |
|---|---|---|
| DeepFilterNet3 | `deepfilternet` 0.5.6 weights (auto-download) | **Apache-2.0 OR MIT** (dual; verified in installed dist-info `LICENSE`) |
| Silero VAD | `snakers4/silero-vad` via `torch.hub` | **MIT** (per official PyPI page) |
| Demucs | `htdemucs` weights (music mode only) | **MIT** (per official repo README) |
| SpeechBrain ECAPA | speaker embedding (isolate mode only) | **Apache-2.0** (per official docs) |
| nara-wpe | `nara-wpe` 0.0.11 (studio mode) | **MIT** (verified in installed dist-info `LICENSE`) |

All licences are permissive (commercial use allowed); keep the
attribution notices when redistributing.

## Known limitations

Be aware of these before trusting the output blindly:

- **Evaluation is synthetic-only.** No real-recorded noisy speech, no
  DNSMOS, no PESQ, no human MOS yet. Synthetic harmonic "speech" is not
  real speech — see Results.
- **SI-SDR can go negative on synthetic clips.** DeepFilterNet3 was
  trained on real speech; on the 4-harmonic synthetic signal it trades
  noise removal for spectral reshaping that SI-SDR punishes even while
  intelligibility (STOI) improves. Treat SI-SDR here as a diagnostic,
  not a quality score.
- **Only `standard` mode is evaluated.** `music` (Demucs) and `isolate`
  (ECAPA) need multi-GB model downloads and a GPU to be practical; they
  are untested in this environment.
- **CPU-only.** The Docker image and this eval run on CPU torch; long
  files are slow (DF chunks 60 s windows). A GPU build would need the
  CUDA torch index instead.
- **Hum notch needs stationary hum.** The 50/60 Hz detector looks for
  peaks 12 dB above neighbours; warbling or very short hum bursts may
  survive.
- **Dereverb (WPE) runs only in studio mode**, and only helps mild
  rooms; heavy reverb is not removed in `standard`.
- **Fixed 48 kHz mono internally.** Stereo is collapsed (louder channel
  wins on large imbalance); surround formats are not specially handled.
- **Room tone is synthetic.** The "subtle"/"natural" bed is generated
  shaped noise, not the room's real tone — it masks processing
  artifacts but is not authentic.
- **Loudness is opinionated.** Output is always normalized to the
  target LUFS (-16 default); the original dynamics are not preserved.
- **Isolate mode needs a good enrollment.** A noisy/short `--enroll`
  sample degrades the target-speaker mask; auto-reference picks the
  highest-SNR 10 s span.
- **API/frontend are new.** Not load-tested; job TTL is 1 hour.

## Listening checklist (manual QA)

For every release candidate, listen on headphones to 3+ varied clips and
confirm each of these is **ABSENT**:

- [ ] musical-noise chirps / "birdies" in silences
- [ ] pumping / breathing of the noise floor
- [ ] lisping or dulled sibilants ("sh" → "th")
- [ ] dull highs / muffled top end
- [ ] metallic / robotic timbre on vowels
- [ ] clipped word onsets or offsets
- [ ] deleted breaths sounding unnatural
- [ ] dead digital-silence gaps inside speech

## Results

Measured with `eval/run_eval.py` on 2026-10-18 (6 s synthetic clips,
`standard` mode, seed 7+i per condition, ~17 s per condition on CPU).
PESQ/DNSMOS skipped as noted above. The full JSON is at
`eval/results.json`; kept wavs for this run were discarded (re-run with
`--keep-wavs DIR` to retain them).

| Noise | SNR (dB) | SI-SDR in (dB) | SI-SDR out (dB) | Δ SI-SDR (dB) | STOI in | STOI out | Pass |
|---|---|---|---|---|---|---|---|
| fan | -5 | -5.00 | -7.30 | -2.30 | 0.4354 | 0.4916 | n/a |
| fan | 0 | -0.06 | 1.10 | +1.16 | 0.5251 | 0.6375 | PASS |
| fan | 5 | 5.07 | 3.80 | -1.26 | 0.6182 | 0.7138 | FAIL |
| fan | 10 | 9.99 | 9.38 | -0.61 | 0.6939 | 0.7666 | FAIL |
| fan | 15 | 14.97 | 8.78 | -6.19 | 0.7343 | 0.7692 | FAIL |
| traffic | -5 | -5.10 | -5.03 | +0.06 | 0.6611 | 0.5820 | n/a |
| traffic | 0 | 0.26 | 0.84 | +0.58 | 0.7833 | 0.6246 | PASS |
| traffic | 5 | 5.00 | 3.59 | -1.41 | 0.8608 | 0.8351 | FAIL |
| traffic | 10 | 9.99 | 11.58 | +1.60 | 0.9326 | 0.9074 | FAIL |
| traffic | 15 | 14.98 | -4.71 | -19.69 | 0.9633 | 0.8777 | FAIL |
| cafe | -5 | -5.05 | -6.99 | -1.94 | 0.5593 | 0.5251 | n/a |
| cafe | 0 | 0.09 | -5.22 | -5.31 | 0.7344 | 0.6190 | FAIL |
| cafe | 5 | 5.03 | 0.19 | -4.84 | 0.7246 | 0.6737 | FAIL |
| cafe | 10 | 9.99 | 3.02 | -6.97 | 0.8199 | 0.7433 | FAIL |
| cafe | 15 | 15.01 | 7.79 | -7.21 | 0.8137 | 0.8005 | FAIL |
| speaker | -5 | -5.00 | -8.42 | -3.42 | 0.5357 | 0.4877 | n/a |
| speaker | 0 | 0.03 | -1.91 | -1.93 | 0.6119 | 0.5441 | FAIL |
| speaker | 5 | 5.01 | 1.09 | -3.92 | 0.6825 | 0.6404 | FAIL |
| speaker | 10 | 9.99 | 0.87 | -9.12 | 0.7341 | 0.6429 | FAIL |
| speaker | 15 | 15.00 | 11.21 | -3.79 | 0.7638 | 0.7670 | FAIL |
| tvmusic | -5 | -4.77 | -8.03 | -3.26 | 0.4565 | 0.4314 | n/a |
| tvmusic | 0 | 0.12 | -3.57 | -3.69 | 0.5165 | 0.4927 | FAIL |
| tvmusic | 5 | 4.92 | -10.90 | -15.81 | 0.6379 | 0.6447 | FAIL |
| tvmusic | 10 | 9.94 | 10.26 | +0.32 | 0.6841 | 0.6873 | PASS |
| tvmusic | 15 | 14.98 | 10.71 | -4.27 | 0.7929 | 0.7509 | FAIL |
| keyboard | -5 | -5.00 | -2.74 | +2.27 | 0.5914 | 0.5903 | n/a |
| keyboard | 0 | 0.00 | 12.08 | +12.08 | 0.7557 | 0.6647 | PASS |
| keyboard | 5 | 5.00 | 13.05 | +8.05 | 0.7582 | 0.7257 | FAIL |
| keyboard | 10 | 10.00 | 11.14 | +1.14 | 0.7468 | 0.7680 | PASS |
| keyboard | 15 | 15.00 | 13.25 | -1.75 | 0.7352 | 0.7633 | FAIL |
| hum | -5 | -5.14 | 1.90 | +7.04 | 0.8887 | 0.9033 | n/a |
| hum | 0 | 0.01 | 13.59 | +13.58 | 0.9572 | 0.9140 | PASS |
| hum | 5 | 5.00 | 11.40 | +6.40 | 0.9715 | 0.9411 | FAIL |
| hum | 10 | 9.97 | 13.99 | +4.02 | 0.9915 | 0.9486 | FAIL |
| hum | 15 | 15.00 | 2.89 | -12.11 | 0.9965 | 0.9110 | FAIL |
| reverb | -5 | -5.05 | -5.58 | -0.53 | 0.3814 | 0.3838 | n/a |
| reverb | 0 | -0.27 | -4.23 | -3.97 | 0.4474 | 0.4458 | FAIL |
| reverb | 5 | 4.98 | -0.85 | -5.82 | 0.5514 | 0.5397 | FAIL |
| reverb | 10 | 9.97 | 0.64 | -9.33 | 0.6460 | 0.5958 | FAIL |
| reverb | 15 | 14.99 | -0.36 | -15.35 | 0.6637 | 0.5520 | FAIL |

**Score: 6 PASS / 26 FAIL / 8 n/a** (n/a = SNR −5 dB, informational only).

What the numbers actually say:

- **Tonal and transient noises are handled well.** 50 Hz hum at 0 dB:
  +13.58 dB SI-SDR (the notch filter earns its keep); keyboard clicks at
  0 dB: +12.08 dB. These are genuine, large improvements.
- **STOI improves on stationary noises** (fan: 0.62→0.71 at 5 dB;
  0.73→0.77 at 15 dB) even when global SI-SDR dips — intelligibility
  goes up while the waveform is spectrally reshaped.
- **Speech-like interferers (cafe babble, competing speaker) are hard:**
  SI-SDR and STOI both degrade. Expected — `standard` mode has no
  speaker isolation; that is what `isolate` mode is for (untested here).
- **SI-SDR is pessimistic on this synthetic set.** The spec's SI-SDR
  uses one global projection coefficient, so time-varying gains from
  DeepFilterNet/postfilter are punished harshly: e.g. traffic @ 15 dB
  scores −4.71 dB globally while per-second SI-SDR is 16–19 dB
  (measured). The clean-input self-test (pipeline run on the clean
  reference) gives 13.75 dB — that is the pipeline's distortion floor on
  this 4-harmonic synthetic "speech", and any input already near that
  ceiling (SNR ≥ 10 dB) can score worse after processing.
- **Don't process nearly-clean audio.** At 15 dB SNR the pipeline's own
  processing distortion outweighs the tiny noise removed (hum @ 15 dB:
  −12.11 dB). A production gate should bypass enhancement above a high
  input-SNR estimate.
- **Reverb is not removed in `standard` mode** (WPE runs only in
  `studio`); the reverb rows measure denoising of reverberant speech
  only, against the reverberant reference.

No other conditions were run; nothing above is estimated or carried over
from other runs.
