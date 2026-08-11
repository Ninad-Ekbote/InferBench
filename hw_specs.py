"""Published THEORETICAL PEAK hardware specs, keyed by substring match against
the name nvidia-smi reports. Dense (non-sparse) BF16/FP16 tensor throughput
WITH FP32 ACCUMULATE (the mode PyTorch/vLLM actually use for matmuls -- NOT
the higher FP16-accumulate figure some spec sheets also list) and memory
bandwidth, from public manufacturer spec sheets. Not measured on this
hardware. Ordered most-specific-first so e.g. "A100" is checked before "A10"
(which is a substring of "A100").

RTX 5090 verified against forum/spec discussion of NVIDIA's Blackwell
whitepaper (2026-08): the commonly-quoted "104.8 TFLOPS" figure was WRONG --
that's roughly half the real dense BF16/FP32-accumulate figure. Confirmed via
this project's own profiling: an actual measured (torch.profiler) prefill
FLOPs/sec exceeded 104.8 TFLOPS, which is physically impossible for a true
peak, and prompted this correction to 209.5 TFLOPS. The other entries below
have NOT been re-verified with this same scrutiny -- treat them with
appropriate skepticism until similarly cross-checked.
"""

GPU_SPECS = [
    ("RTX 5090", {"memory_bandwidth_gb_s": 1792, "peak_bf16_tflops": 209.5}),
    ("RTX 4090", {"memory_bandwidth_gb_s": 1008, "peak_bf16_tflops": 82.6}),
    ("RTX 3090", {"memory_bandwidth_gb_s": 936, "peak_bf16_tflops": 35.6}),
    ("H200", {"memory_bandwidth_gb_s": 4800, "peak_bf16_tflops": 989}),
    ("H100", {"memory_bandwidth_gb_s": 3350, "peak_bf16_tflops": 989}),
    ("A100-SXM4-80GB", {"memory_bandwidth_gb_s": 2039, "peak_bf16_tflops": 312}),
    ("A100", {"memory_bandwidth_gb_s": 1555, "peak_bf16_tflops": 312}),
    ("A10G", {"memory_bandwidth_gb_s": 600, "peak_bf16_tflops": 125}),
    ("A10", {"memory_bandwidth_gb_s": 600, "peak_bf16_tflops": 125}),
]


def lookup_gpu_specs(gpu_name):
    for key, specs in GPU_SPECS:
        if key.lower() in gpu_name.lower():
            return dict(specs, matched_key=key, gpu_name=gpu_name)
    known = ", ".join(k for k, _ in GPU_SPECS)
    raise SystemExit(
        f"No published hardware specs for detected GPU '{gpu_name}'. "
        f"Known GPUs: {known}. Add it to hw_specs.py rather than guessing."
    )
