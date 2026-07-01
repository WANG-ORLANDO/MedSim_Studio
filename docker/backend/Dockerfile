# ============================================================
# CT Simulator - Backend Dockerfile
# ============================================================
# Multi-stage build: Python deps → runtime
# ============================================================

# ---- Stage 1: Dependencies ----
FROM python:3.11-slim AS builder

ARG HTTP_PROXY=""
ARG HTTPS_PROXY=""
ARG NO_PROXY="*"
ENV HTTP_PROXY=${HTTP_PROXY} HTTPS_PROXY=${HTTPS_PROXY} NO_PROXY=${NO_PROXY}

WORKDIR /app

# System dependencies for medical imaging libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    wget \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt .

# Install PyTorch CPU version (ARM64 macOS doesn't support CUDA)
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install \
    --timeout 600 \
    torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install rest of Python dependencies
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install \
    --timeout 600 \
    -r requirements.txt
# ---- Stage 2: Runtime ----
FROM python:3.11-slim AS runner

ARG HTTP_PROXY=""
ARG HTTPS_PROXY=""
ARG NO_PROXY="*"
ENV HTTP_PROXY=${HTTP_PROXY} HTTPS_PROXY=${HTTPS_PROXY} NO_PROXY=${NO_PROXY}

WORKDIR /app

# Runtime system dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY backend/ .

# Create necessary directories
RUN mkdir -p /tmp/uploads /app/ai-models /app/static

# Copy custom-trained MONAI UNet weights to a path NOT overridden by volumes
# (/app/ai-models is mounted as a named volume in docker-compose, which
#  would hide files COPY'd there at build time)
# Optional: skip if unet.pth not present
RUN mkdir -p /app/custom-models

# TotalSegmentator pretrained weights dir (~2GB, downloaded on first use)
# Cache them so they persist across container restarts
ENV TOTALSEGMENTATOR_DIR=/app/ai-models/totalsegmentator
RUN mkdir -p ${TOTALSEGMENTATOR_DIR}

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

EXPOSE 8000

# Run with uvicorn (2 workers — one for background tasks, one to stay responsive)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
