# VoiceClean v2 — Frontend

React + Vite + TypeScript + Tailwind CSS (dark, responsive, keyboard-accessible).

## Run

```bash
npm install
npm run dev      # http://localhost:5173
npm run build    # production build -> dist/
```

## Backend

Point at the FastAPI backend with `VITE_API_URL` (defaults to `http://localhost:8000`):

```bash
VITE_API_URL=http://localhost:8000 npm run dev
```

Expected API surface: `POST /api/jobs` (multipart `file, mode, strength,
loudness_target, room_tone, dereverb, enroll_file?`), `GET /api/jobs/{id}`,
`GET /api/jobs/{id}/download?format=wav|mp3|mp4`, `DELETE /api/jobs/{id}`.

## Tabs

- **Clean audio** — drop zone (wav/mp3/m4a/aac/ogg/flac/mp4/mov/mkv/webm) or raw
  mic recorder (browser pre-processing disabled), mode/strength/room-tone/
  loudness/dereverb controls, voice-sample upload for isolate mode, real
  stage + progress polling, before/after waveform + log-frequency spectrogram
  with synced playhead, loudness-matched A/B (gain = 10^((L_A−L_B)/20) from
  measured LUFS), measured-values-only report, WAV/MP3/MP4 downloads.
- **Live preview** — the genuine RNNoise neural network (Xiph.Org, Jitsi WASM
  build in `public/assets/`) running in realtime on mic input via an
  AudioWorklet courier + main-thread inference. Verified real: the same
  `rnnoise.wasm` was instantiated in Node and suppressed a white-noise frame
  (VAD 0.0, output energy 0.0). No fakes anywhere.
