#!/usr/bin/env bash
set -euo pipefail

# Starts a vLLM server on the RunPod pod and tunnels its port to localhost.
# Keep this running in a terminal; Ctrl+C stops both the tunnel and the server.

if [ -f .env ]; then
  set -a
  source .env
  set +a
fi

: "${RUNPOD_HOST:?RUNPOD_HOST not set (see .env.example)}"
: "${RUNPOD_PORT:=22}"
: "${RUNPOD_SSH_KEY:?RUNPOD_SSH_KEY not set (see .env.example)}"
: "${MODEL_NAME:=gpt2}"
: "${VLLM_PORT:=8000}"

# VLLM_USE_FLASHINFER_SAMPLER=0: on Blackwell GPUs (SM 12.x, e.g. RTX 50 series)
# FlashInfer's sampling kernel doesn't recognize the architecture and throws
# "FlashInfer requires GPUs with sm75 or higher" even though CUDA is new enough.
# This skips FlashInfer's sampler and falls back to vLLM's native PyTorch one.
ssh -o StrictHostKeyChecking=accept-new -i "$RUNPOD_SSH_KEY" -p "$RUNPOD_PORT" -L "$VLLM_PORT:localhost:$VLLM_PORT" "root@$RUNPOD_HOST" \
  "pip install -q --break-system-packages vllm && VLLM_USE_FLASHINFER_SAMPLER=0 vllm serve $MODEL_NAME --port $VLLM_PORT"
