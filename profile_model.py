"""Runs ON the GPU pod. Loads the same model the benchmark serves and profiles
isolated prefill / decode forward passes with torch.profiler.

Deliberately does NOT profile the live vLLM HTTP server: torch.profiler can't
usefully attach to a running async server handling concurrent requests, and
vLLM's CUDA-graph capture would break op-level attribution anyway. Instead this
loads the model directly via transformers (attn_implementation="eager", so
attention decomposes into individual matmuls torch.profiler can attribute
FLOPs to -- the fused SDPA kernel vLLM actually uses often doesn't report
FLOPs). This means absolute wall-clock numbers here will differ from vLLM's
real serving speed; what's representative is the workload's FLOPs/byte shape
(arithmetic intensity), which is a property of the architecture and the
prefill/decode shapes, not the specific kernel implementation.

Representative shapes (documented, not tuned per-run):
  Prefill: sequence_length=128 tokens, batch_size in {1, 20}
           (20 matches load_test.py's default benchmark concurrency)
  Decode:  context_length=128 tokens already in KV cache, one new token,
           batch_size in {1, 20}
"""

import argparse
import csv
import subprocess
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from hw_specs import lookup_gpu_specs

PREFILL_SEQ_LEN = 128
PREFILL_BATCH_SIZES = [1, 20]
DECODE_CONTEXT_LEN = 128
DECODE_BATCH_SIZES = [1, 20]
WARMUP_ITERS = 3
TIMED_ITERS = 10


def detect_gpu_name():
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True
    )
    return out.strip().splitlines()[0].strip()


def check_ncu():
    """Best-effort: is Nsight Compute present and permitted? Never fatal."""
    try:
        subprocess.run(["ncu", "--version"], capture_output=True, text=True, timeout=10)
    except FileNotFoundError:
        return "ncu not installed on this pod"
    except Exception as e:
        return f"ncu present but version check failed: {e}"

    probe = subprocess.run(
        ["ncu", "--target-processes", "all", "python3", "-c", "import torch;a=torch.randn(64,64,device='cuda')@torch.randn(64,64,device='cuda');torch.cuda.synchronize()"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    combined = (probe.stdout or "") + (probe.stderr or "")
    if probe.returncode == 0 and "ERR_NVGPUCTRPERM" not in combined:
        return "ncu available and permitted (not used for this run -- see analysis doc)"
    if "ERR_NVGPUCTRPERM" in combined:
        return "ncu installed but blocked by container permissions (ERR_NVGPUCTRPERM) -- falling back to torch.profiler + analytical memory traffic"
    return f"ncu probe failed (exit {probe.returncode}) -- falling back to torch.profiler + analytical memory traffic"


def load_model(model_name):
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, attn_implementation="eager"
    ).to("cuda")
    model.eval()
    return tok, model


def model_weight_bytes(model):
    return sum(p.numel() for p in model.parameters()) * 2  # bf16 = 2 bytes


def kv_bytes_per_token(config):
    num_layers = config.num_hidden_layers
    num_kv_heads = getattr(config, "num_key_value_heads", None) or config.num_attention_heads
    head_dim = config.hidden_size // config.num_attention_heads
    dtype_bytes = 2
    return 2 * num_layers * num_kv_heads * head_dim * dtype_bytes  # 2 = K and V


def profile_flops(fn):
    """Runs fn() once under torch.profiler(with_flops=True) and sums attributed
    FLOPs. Returns (total_flops, sorted list of op names with time but no FLOPs
    attributed, for documentation of coverage gaps)."""
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        with_flops=True,
    ) as prof:
        fn()
        torch.cuda.synchronize()

    total_flops = 0
    uncounted = set()
    for evt in prof.key_averages():
        flops = getattr(evt, "flops", 0) or 0
        if flops:
            total_flops += flops
        elif evt.count > 0 and not evt.key.startswith("cuda") and not evt.key.startswith("Memcpy"):
            uncounted.add(evt.key)
    return total_flops, sorted(uncounted)


def time_fn(fn, warmup=WARMUP_ITERS, iters=TIMED_ITERS):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / iters


def make_prefill_fn(tok, model, batch_size, seq_len):
    input_ids = torch.randint(0, tok.vocab_size, (batch_size, seq_len), device="cuda")

    def fn():
        with torch.no_grad():
            model(input_ids, use_cache=True)

    return fn


def make_decode_fn(tok, model, batch_size, context_len):
    context_ids = torch.randint(0, tok.vocab_size, (batch_size, context_len), device="cuda")
    with torch.no_grad():
        out = model(context_ids, use_cache=True)
    past = out.past_key_values
    next_token = torch.randint(0, tok.vocab_size, (batch_size, 1), device="cuda")

    def fn():
        with torch.no_grad():
            model(next_token, past_key_values=past, use_cache=True)

    return fn


def build_row(stage, batch_size, seq_len, flops, exec_time, mem_bytes, gpu_name, model_name, uncounted, tokens_produced):
    achieved_flops_per_sec = flops / exec_time if exec_time > 0 else 0.0
    arithmetic_intensity = flops / mem_bytes if mem_bytes > 0 else 0.0
    tokens_per_sec = tokens_produced / exec_time if exec_time > 0 else 0.0
    return {
        "gpu": gpu_name,
        "model": model_name,
        "stage": stage,
        "batch_size": batch_size,
        "sequence_length": seq_len,
        "flops": f"{flops:.0f}",
        "flops_source": "torch.profiler(with_flops=True)",
        "execution_time_s": f"{exec_time:.6f}",
        "achieved_flops_per_sec": f"{achieved_flops_per_sec:.0f}",
        "estimated_memory_bytes": f"{mem_bytes:.0f}",
        "memory_bytes_source": "analytical (weights + KV cache, see roofline_analysis.md)",
        "arithmetic_intensity": f"{arithmetic_intensity:.4f}",
        "tokens_per_sec": f"{tokens_per_sec:.2f}",
        "uncounted_flops_ops": ";".join(uncounted),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", default="roofline_profile.csv")
    parser.add_argument("--ncu-status-output", default="ncu_status.txt")
    args = parser.parse_args()

    gpu_name = detect_gpu_name()
    specs = lookup_gpu_specs(gpu_name)  # fail-fast if unsupported, per requirements
    print(f"Detected GPU: {gpu_name}", file=sys.stderr)
    print(f"Theoretical specs: {specs}", file=sys.stderr)

    ncu_status = check_ncu()
    print(f"ncu status: {ncu_status}", file=sys.stderr)
    with open(args.ncu_status_output, "w") as f:
        f.write(ncu_status + "\n")

    tok, model = load_model(args.model)
    weight_bytes = model_weight_bytes(model)
    kv_per_tok = kv_bytes_per_token(model.config)
    print(f"weight_bytes={weight_bytes} kv_bytes_per_token={kv_per_tok}", file=sys.stderr)

    rows = []

    for batch in PREFILL_BATCH_SIZES:
        fn = make_prefill_fn(tok, model, batch, PREFILL_SEQ_LEN)
        flops, uncounted = profile_flops(fn)
        exec_time = time_fn(fn)
        # Prefill: weights read once (amortized across batch), KV cache written
        # for every prompt token. Activation traffic is deliberately excluded
        # (see limitations in roofline_analysis.md).
        mem_bytes = weight_bytes + batch * PREFILL_SEQ_LEN * kv_per_tok
        tokens_produced = batch * PREFILL_SEQ_LEN
        rows.append(
            build_row("prefill", batch, PREFILL_SEQ_LEN, flops, exec_time, mem_bytes, gpu_name, args.model, uncounted, tokens_produced)
        )
        print(f"prefill batch={batch}: flops={flops:.3e} time={exec_time*1000:.2f}ms uncounted={uncounted}", file=sys.stderr)

    for batch in DECODE_BATCH_SIZES:
        fn = make_decode_fn(tok, model, batch, DECODE_CONTEXT_LEN)
        flops, uncounted = profile_flops(fn)
        exec_time = time_fn(fn)
        # Decode: weights read once per step (amortized across batch), KV cache
        # read for existing context (per sequence) + written for the new token.
        mem_bytes = weight_bytes + batch * DECODE_CONTEXT_LEN * kv_per_tok + batch * kv_per_tok
        tokens_produced = batch  # one new token per sequence in the batch
        rows.append(
            build_row("decode", batch, DECODE_CONTEXT_LEN, flops, exec_time, mem_bytes, gpu_name, args.model, uncounted, tokens_produced)
        )
        print(f"decode batch={batch}: flops={flops:.3e} time={exec_time*1000:.2f}ms uncounted={uncounted}", file=sys.stderr)

    fieldnames = list(rows[0].keys())
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
