# VoiceClean v2 — File Tree

```
voiceclean-v2/
├── README.md                     # Setup, run commands, model licences, limitations, results
├── BUILD_REPORT.md               # Test results, what was/not tested, known limitations
├── FILETREE.md                   # This file
├── Dockerfile                    # 2-stage python:3.11-slim + ffmpeg, torch CPU, pre-warms models
├── docker-compose.yml            # backend + frontend services (no redis; BackgroundTasks+SQLite)
├── backend/
│   ├── config.py                 # EVERY constant lives here (spec rule); no magic numbers elsewhere
│   ├── requirements.txt          # Frozen pip freeze of the tested venv
│   ├── app/
│   │   └── main.py               # FastAPI: POST /api/jobs, GET /api/jobs/{id}, download,
│   │                             #   cancel, delete; BackgroundTasks + SQLite, TTL sweep
│   ├── pipeline/
│   │   ├── cli.py                # python -m pipeline.cli in.wav out.wav --mode standard
│   │   ├── pipeline.py           # Orchestrator: ingest→filters→vad→separation→denoise→
│   │                             #   dereverb→postfilter→roomtone→tone→loudness→export
│   │   ├── ingest.py             # ffmpeg one-decode 48k mono f32, stereo-imbalance rule,
│   │                             #   LUFS + 10th-pct noise floor
│   │   ├── filters.py            # 80Hz HP zero-phase, hum notch 50/60+harmonics >12dB, Q=30
│   │   ├── vad.py                # Silero VAD via torch.hub, lazy-loaded
│   │   ├── separation.py         # Demucs (lazy) + ECAPA (lazy) TargetSpeakerExtractor interface
│   │   ├── denoise.py            # DeepFilterNet3, native atten_lim_db blend, chunked >10min
│   │   ├── dereverb.py           # nara_wpe, chunked (OOM guard on low RAM)
│   │   ├── postfilter.py         # MMSE-LSA + attack/release smoothing + 3x3 median +
│   │                             #   pYIN harmonic & sibilant protection
│   │   ├── roomtone.py           # Room tone shaped from input non-speech spectrum
│   │   ├── tone.py               # Studio mode: presence EQ, de-esser, Giannoulis compressor
│   │   ├── loudness.py           # pyloudnorm BS.1770 + -1.0 dBTP 4x-oversampled limiter
│   │   └── export.py             # WAV24 / MP3-320k / MP4 remux
│   └── tests/
│       └── test_pipeline.py      # Pipeline unit/smoke tests
├── eval/
│   ├── run_eval.py               # Spec §7 suite: 8 noise types × 5 SNRs, SI-SDR/STOI
│   ├── make_synth.py             # Synthetic speech-like clip generator (numpy)
│   └── results.json              # Measured results from the actual run (40 conditions)
└── frontend/
    ├── package.json / tsconfig.* / vite.config.ts / index.html
    ├── src/
    │   ├── App.tsx               # Root component
    │   ├── main.tsx / index.css / vite-env.d.ts
    │   ├── api/client.ts         # Backend API contract (seam between frontend and FastAPI)
    │   ├── audio/
    │   │   ├── dsp.ts            # Client-side DSP helpers
    │   │   ├── player.ts         # Loudness-matched A/B player
    │   │   ├── previewWorklet.ts # Live-preview audio worklet
    │   │   └── rnnoise.ts        # RNNoise WASM live-preview (verified instantiated)
    │   └── components/
    │       ├── Controls.tsx      # Mode/strength/dereverb/room-tone/loudness/enroll
    │       ├── DropZone.tsx      # File upload dropzone
    │       ├── JobProgress.tsx   # Real stage progress display
    │       ├── LivePreview.tsx   # Live mic preview via RNNoise WASM
    │       ├── Recorder.tsx      # Mic recorder (raw getUserMedia constraints per spec)
    │       └── ResultsView.tsx   # Report panel (measured values only) + downloads
    ├── public/assets/            # rnnoise.js + rnnoise.wasm (2-part split, concatenated at load)
    └── dist/                     # Production build output (npm run build verified green)
```

Notes:
- `backend/.venv/` (dev venv, not committed) and `frontend/node_modules/` excluded.
- `eval/__pycache__/` excluded.
