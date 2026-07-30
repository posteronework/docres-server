#!/usr/bin/env bash
# One-shot setup + run for RunPod (PyTorch base image).
# Usage on pod:  DOCRES_API_KEY=yourkey bash start.sh
set -e

PORT="${PORT:-8000}"

echo "[start] git-lfs..."
if ! command -v git-lfs >/dev/null 2>&1; then
  apt-get update && apt-get install -y git-lfs
fi
git lfs install

# Must be run from inside the cloned repo (where server.py lives).
if [ ! -f server.py ]; then
  echo "[start] ERROR: run this from the docres-server repo dir (server.py not found)."
  echo "        git clone https://github.com/posteronework/docres-server.git && cd docres-server"
  exit 1
fi

echo "[start] pulling LFS weights..."
git lfs pull
ls -lh checkpoints/

echo "[start] installing deps (torch/torchvision come from base image)..."
pip install --no-cache-dir -r requirements.txt

echo "[start] launching server on port ${PORT} (auto-restart on crash) ..."
while true; do
  uvicorn server:app --host 0.0.0.0 --port "${PORT}" --workers 1 || true
  echo "[start] server exited (code $?), restarting in 3s..."
  sleep 3
done
