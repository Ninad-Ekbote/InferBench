# InferBench

Run a persistent vLLM server on a RunPod GPU pod, send it requests from your machine, and benchmark it under concurrent load.

## Setup

1. Deploy a GPU pod on RunPod with SSH enabled (1 GPU, 24GB+ VRAM recommended for a 7B model).
2. Copy `.env.example` to `.env` and fill in your pod's SSH host, port, key path, and the model you want to serve.
3. `./serve.sh` — installs vLLM on the pod, starts it, and tunnels its port to `localhost`. Leave this running.

## Usage

In a second terminal (with `serve.sh` still running):

- **Single request:** `python client.py "your prompt here"`
- **Concurrent load test:** `python load_test.py --users 20`

`load_test.py` fires N requests at the server at the same time (via a thread pool) and prints total time, average latency, and throughput. By default it cycles through the sample prompts in `prompts.json`; pass a prompt explicitly (`python load_test.py "..." --users 20`) to send that one prompt to every request instead.

Every parameter is overridable via CLI flags (`--users`, `--max-tokens`, `--model`, `--host`, `--port`) and otherwise falls back to `.env`.

## Results

Each `load_test.py` run appends a row (timestamp, model, users, max_tokens, total_s, avg_latency_s, throughput_req_s) to `results/load_test.csv`, so you can compare runs across models, GPU types, or user counts. This file isn't committed to git.

## Files

- `serve.sh` — starts the vLLM server on the pod over SSH and tunnels its port locally.
- `client.py` — sends a single prompt to the running server.
- `load_test.py` — sends concurrent requests to simulate multiple users and logs results.
- `env.py` — tiny shared helper both Python scripts use to read `.env`.
- `prompts.json` — sample prompts used when no prompt is passed to `load_test.py`.

## Notes

- On very new GPUs (e.g. RTX 50 series / Blackwell, SM 12.x), `serve.sh` sets `VLLM_USE_FLASHINFER_SAMPLER=0` — FlashInfer's sampling kernel doesn't yet support this architecture and crashes on startup otherwise.
