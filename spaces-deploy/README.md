---
title: VoiceClean v2 Backend
emoji: 🎙️
colorFrom: indigo
colorTo: blue
sdk: docker
pinned: false
---

# VoiceClean v2 — Backend API

FastAPI + PyTorch (CPU) voice-cleaning pipeline: FFmpeg ingest → high-pass + hum notch → Silero VAD → DeepFilterNet3 neural denoise → WPE dereverb → MMSE-LSA post-filter → room tone → studio mastering → loudness-normalized export.

Endpoints: `POST /api/jobs`, `GET /api/jobs/{id}`, `GET /api/jobs/{id}/download`, `DELETE /api/jobs/{id}`.
