#!/usr/bin/env bash
set -euo pipefail

# Runs the isolated prefill/decode profiling script on the RunPod pod (it needs
# a real GPU) and pulls the resulting CSV back to results/roofline_profile.csv.
# Does not require serve.sh to be running -- this loads the model directly,
# separately from the vLLM server.

if [ -f .env ]; then
  set -a
  source .env
  set +a
fi

: "${RUNPOD_HOST:?RUNPOD_HOST not set (see .env.example)}"
: "${RUNPOD_PORT:=22}"
: "${RUNPOD_SSH_KEY:?RUNPOD_SSH_KEY not set (see .env.example)}"
: "${MODEL_NAME:=gpt2}"

REMOTE_DIR="~/inferbench-profile"

ssh -o StrictHostKeyChecking=accept-new -i "$RUNPOD_SSH_KEY" -p "$RUNPOD_PORT" "root@$RUNPOD_HOST" "mkdir -p $REMOTE_DIR"

scp -o StrictHostKeyChecking=accept-new -i "$RUNPOD_SSH_KEY" -P "$RUNPOD_PORT" \
  profile_model.py hw_specs.py "root@$RUNPOD_HOST:$REMOTE_DIR/"

# HF_HUB_ENABLE_HF_TRANSFER=1 uses hf_transfer's parallel-connection downloader
# instead of a single HTTP stream -- default single-connection downloads from
# HF Hub can be extremely slow (sub-1MB/s observed) regardless of auth status.
ssh -o StrictHostKeyChecking=accept-new -i "$RUNPOD_SSH_KEY" -p "$RUNPOD_PORT" "root@$RUNPOD_HOST" \
  "pip install -q --break-system-packages transformers accelerate hf_transfer && cd $REMOTE_DIR && HF_HUB_ENABLE_HF_TRANSFER=1 python3 profile_model.py --model '$MODEL_NAME' --output roofline_profile.csv --ncu-status-output ncu_status.txt"

mkdir -p results
scp -o StrictHostKeyChecking=accept-new -i "$RUNPOD_SSH_KEY" -P "$RUNPOD_PORT" \
  "root@$RUNPOD_HOST:$REMOTE_DIR/roofline_profile.csv" "root@$RUNPOD_HOST:$REMOTE_DIR/ncu_status.txt" \
  results/

echo "wrote results/roofline_profile.csv and results/ncu_status.txt"
