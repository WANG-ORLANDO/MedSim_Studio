#!/bin/bash
# ============================================================
# CT Simulator - Pre-download Python Wheels for Docker Build
# ============================================================
# Pre-downloads all Python dependencies as Linux wheels
# into pip_cache/, so Docker builds can install offline.
#
# Usage:
#   ./scripts/preload-pip.sh
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CACHE_DIR="$SCRIPT_DIR/pip_cache"
REQUIREMENTS="$SCRIPT_DIR/backend/requirements.txt"

# Convert path for Docker volume mount (handles Git Bash on Windows)
to_docker_path() {
  local path="$1"
  if command -v cygpath &>/dev/null; then
    cygpath -w "$path"
  else
    echo "$path"
  fi
}

DOCKER_CACHE="$(to_docker_path "$CACHE_DIR")"

echo "============================================"
echo " Pre-downloading Python wheels for Docker"
echo "============================================"

# Ensure pip_cache directory exists
mkdir -p "$CACHE_DIR"

# Copy requirements.txt into cache dir so we only mount one volume
cp "$REQUIREMENTS" "$CACHE_DIR/requirements.txt"

echo ""
echo "Downloading all dependencies (this may take a while)..."
echo ""

# Use the same base image as the Dockerfile to ensure correct platform
docker run --rm \
  -v "$DOCKER_CACHE:/pip_cache" \
  python:3.11-slim \
  sh -c "
    pip install --upgrade pip setuptools wheel -q && \
    pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple && \
    echo 'Downloading wheels...' && \
    pip download -r /pip_cache/requirements.txt -d /pip_cache --timeout 600 && \
    rm -f /pip_cache/requirements.txt && \
    echo 'Done! Wheels saved to /pip_cache' && \
    echo 'Total files:' && ls -1 /pip_cache | wc -l
  "

echo ""
echo "============================================"
echo " Pre-download complete!"
echo "============================================"
echo "Wheels cached at: $CACHE_DIR"
echo "($(ls -1 "$CACHE_DIR" 2>/dev/null | wc -l) files, $(du -sh "$CACHE_DIR" 2>/dev/null | cut -f1))"
echo ""
echo "Now run Docker build:"
echo "  docker-compose build backend"
echo "============================================"
