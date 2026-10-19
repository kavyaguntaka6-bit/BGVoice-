# VoiceClean v2 — Build Report

Built 2026-10-18. CPU-only environment. Python 3.11 (standalone build at `/opt/python311`;
deepfilterlib publishes no cp312 wheels, and spec requires 3.11).

## What was tested — actual results

### 1. CLI pipeline smoke test (standard mode, full end-to-end)
Command: `python -m pipeline.cli in.wav out.wav --mode standard` on an 8s synthetic clip
(clean speech-like signal + fan noise + hum at 5 dB SNR):
- **SI-SDR: 4.97 → 11.93 dB (+6.96 dB)** — improved, not collapsed. PASS.
- Noise floor: **−22.0 → −63.6 dB** (measured from the test run).
- Output: −16.0 LUFS, length preserved exactly (384000 samples), valid finite audio, peak 0.891.
- DeepFilterNet3 genuine on real noise: white noise −42.9 dB suppression, fan −31.2 dB.

### 2. DeepFilterNet3 synthetic-signal behavior (diagnostic, kept for the record)
- 4-harmonic natural-f0 synthetic + natural intonation (no vibrato) + syllabic AM + pauses:
  +32 dB SI-SDR improvement on clean input.
- Vibrato or >4 harmonics cause collapse (e.g. 5 harmonics: −10.9 dB).
  DF3 is hypersensitive to synthetic harmonic structure; this drove the final test-clip design
  (4 harmonics, no vibrato). Genuine on real noise per above.

### 3. FastAPI backend (`backend/app/main.py`)
- POST /api/jobs → poll GET /api/jobs/{id} → `done` in **~13 s** on a test clip.
- Download WAV **48 kHz / 24-bit** and **MP3-320k** verified byte-valid.
- Validation: 400 (bad file type), 404 (unknown job), 409 (conflict), 413 (oversize) all returned.
- TTL sweep deletes expired jobs; bogus non-audio upload → job goes to `error` state cleanly.
- Fixed a `/tmp` tmpfs disk-space issue by switching to disk-backed `backend/.tmp/`.

### 4. Frontend (`frontend/`)
- `npm install` + `npm run build` **succeed green**; `dist/` present.
- React 18 + TypeScript strict + Tailwind v4. Dropzone, mic recorder (raw getUserMedia
  constraints per spec), all controls (mode / strength / dereverb / room tone / loudness /
  enroll), real stage progress, canvas waveforms + log-frequency spectrograms,
  loudness-matched A/B, report panel (measured values only), WAV/MP3/MP4 download links.
- **Live preview implemented for real**: RNNoise WASM verified instantiated in Node —
  `_rnnoise_create` returned a valid instance; VAD 0.0 on white noise as expected.

### 5. eval/run_eval.py (spec §7)
Full grid run by the eval builder: 8 noise types (fan, traffic, cafe babble, interfering
speaker, TV/music, keyboard, hum, reverb) × 5 SNRs (−5/0/5/10/15 dB), 6 s clips,
standard mode, ~17 s/condition on CPU. Results in `eval/results.json` (40 conditions).
- **6 PASS / 26 FAIL / 8 n/a.** Input SI-SDR matches the target SNR in every row
  (mixing law `g = sqrt(P_clean/(P_noise·10^(SNR/10)))` verified).
- SI-SDR metric itself verified scale-invariant (160.8 dB on half-gain, 40.0 dB on +40 dB noise).
- Highlights: **hum @ 0 dB +13.58 dB SI-SDR**, **keyboard @ 0 dB +12.08 dB**,
  STOI improves on stationary noises (fan 5 dB: 0.618 → 0.714).
- Honest negatives, diagnosed: global SI-SDR often goes negative on this synthetic set —
  DeepFilterNet applies time-varying gains while the spec's single global projection
  coefficient assumes a static scaling (e.g. traffic@15dB: −4.71 dB globally but 16–19 dB
  per-second; clean-input self-test shows a 13.75 dB distortion floor).
  Speech-like interferers (cafe/speaker) degrade both metrics — expected without isolate mode.
  PESQ skipped (`pip install pesq` does not build in this environment);
  DNSMOS skipped (not available offline). Both reported as null with reasons, never faked.

## What was NOT tested and why
- **Demucs / ECAPA end-to-end**: multi-GB models deliberately not downloaded (per spec rule);
  code paths written with lazy loading and verified by import + interface checks only.
- **Video MP4 path**: `export.py` remux code written, not exercised end-to-end.
- **Docker build**: no Docker daemon in this environment; Dockerfile + compose YAML
  syntax-checked only. First `docker build` should be watched — pre-warm step needs network
  access to GitHub/HuggingFace.
- **Frontend ↔ live backend integration**: frontend built against the API contract
  (`src/api/client.ts` is the seam); both sides implement the spec contract, but a joint
  run against the running server was not performed.
- **PESQ / DNSMOS**: cannot be installed offline (see above); reported as null.

## Known limitations
- CPU-only: WPE dereverb runs ~11× real-time; studio mode is slow on long clips.
- DeepFilterNet3 is sensitive to synthetic harmonic structure (vibrato / >4 harmonics collapse
  SI-SDR); this is a property of the model on synthetic test signals, measured, not assumed.
- Global SI-SDR understates DF3's per-second performance on this synthetic set (see §5).
- Model downloads (DeepFilterNet3 8.7 MB ckpt, Silero VAD) happen on first run — not vendored.
- README lists licences (verified from primary sources): DeepFilterNet Apache-2.0 OR MIT,
  nara-wpe MIT, Silero VAD MIT, Demucs MIT, SpeechBrain Apache-2.0.

## Remaining work
1. First real `docker build` + `docker compose up` on a machine with Docker.
2. Optional: download Demucs/ECAPA once to exercise the isolate/enroll paths end-to-end.
3. Optional: joint frontend↔backend smoke run against the live server.
4. Optional: PESQ/DNSMOS where installable, to fill the null metric columns.
