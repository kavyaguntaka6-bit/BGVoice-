# VoiceClean v2 — backend image (CPU).
#
# Two stages:
#   builder  — pip-installs backend/requirements.txt (torch/torchaudio from
#              the PyTorch CPU wheel index) into /install
#   runtime  — slim image + ffmpeg, copies the installed packages, pre-warms
#              the DeepFilterNet3 and Silero VAD model caches so the first
#              API request never waits on a download.
#
# Build:  docker build -t voiceclean-v2-backend .
# Run:    docker run -p 8000:8000 voiceclean-v2-backend
#
# The FastAPI app is assumed to expose `app` in backend/app/main.py
# (POST /api/jobs, GET /api/jobs/{id}, download, DELETE).

# ---------------------------------------------------------------- builder ---
FROM python:3.11-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

COPY backend/requirements.txt /tmp/requirements.txt

# torch/torchaudio "+cpu" builds only exist on the PyTorch CPU index.
RUN pip install --no-cache-dir --prefix=/install \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        -r /tmp/requirements.txt

# ---------------------------------------------------------------- runtime ---
FROM python:3.11-slim AS runtime

# ffmpeg: audio decode/encode for the ingest stage.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /install /usr/local

COPY backend /app/backend
WORKDIR /app/backend

# Model caches live inside the image (pre-warmed below) and can be
# overridden with a volume at runtime for persistence across restarts.
ENV TORCH_HOME=/app/.cache/torch \
    HF_HUB_CACHE=/app/.cache/hf \
    XDG_CACHE_HOME=/app/.cache \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=4

# Pre-warm: download DeepFilterNet3 (~100 MB) and Silero VAD (~2 MB) now,
# at build time, so container startup and first requests need no network.
RUN mkdir -p /app/.cache && python - <<'EOF'
from df.enhance import init_df
init_df(log_level="WARNING", log_file=None)   # DeepFilterNet3 weights
print("DeepFilterNet3 cached")
import torch
torch.hub.load("snakers4/silero-vad", "silero_vad", trust_repo=True)
print("Silero VAD cached")
EOF

EXPOSE 7860

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
