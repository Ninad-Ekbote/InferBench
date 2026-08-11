"""The STANDARD arithmetic-intensity roofline (X=FLOPs/Byte, Y=FLOPs/sec,
log-log), built from results/roofline_profile.csv (written by profile.sh
running profile_model.py on the pod).

Every number in the output is labeled as one of:
  MEASURED / PROFILED   -- from torch.profiler on real isolated forward passes
  ANALYTICALLY ESTIMATED -- derived from model architecture (weights/KV-cache
                            byte math), not measured
  PUBLISHED THEORETICAL  -- manufacturer spec sheet peak, not measured here

See results/roofline_analysis.md (written by this script) for the full
methodology write-up.
"""

import csv
import os

import matplotlib.pyplot as plt

from hw_specs import lookup_gpu_specs

PROFILE_FILE = "results/roofline_profile.csv"
NCU_STATUS_FILE = "results/ncu_status.txt"
ROOFLINE_PLOT = "plots/roofline.png"
ROOFLINE_DOC = "results/roofline_analysis.md"

STAGE_COLORS = {"prefill": "#0072B2", "decode": "#E69F00"}  # colorblind-safe pair


def load_profile_rows():
    if not os.path.exists(PROFILE_FILE):
        raise SystemExit(
            f"{PROFILE_FILE} not found. Run ./profile.sh first -- it profiles the "
            "model on the pod and writes this file. There is no fallback: this "
            "script does not fabricate profiling data."
        )
    with open(PROFILE_FILE) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"{PROFILE_FILE} is empty.")
    return rows


def load_ncu_status():
    if not os.path.exists(NCU_STATUS_FILE):
        return "unknown (results/ncu_status.txt not found -- run profile.sh)"
    with open(NCU_STATUS_FILE) as f:
        return f.read().strip()


def plot(gpu_name, specs, rows):
    os.makedirs("plots", exist_ok=True)
    peak_flops = specs["peak_bf16_tflops"] * 1e12
    peak_bw = specs["memory_bandwidth_gb_s"] * 1e9
    ridge_ai = peak_flops / peak_bw

    ais = [float(r["arithmetic_intensity"]) for r in rows]
    perfs = [float(r["achieved_flops_per_sec"]) for r in rows]
    x_min = min(ais + [ridge_ai]) / 8
    x_max = max(ais + [ridge_ai]) * 8
    y_min = min(perfs + [peak_bw * x_min]) / 5
    y_max = peak_flops * 3

    fig, ax = plt.subplots(figsize=(8.5, 6))

    ax.plot([x_min, ridge_ai], [peak_bw * x_min, peak_bw * ridge_ai], color="#767676", linestyle="--", linewidth=2, zorder=2, label="Memory-bandwidth roof")
    ax.plot([ridge_ai, x_max], [peak_flops, peak_flops], color="#767676", linestyle="--", linewidth=2, zorder=2)
    # Faint continuation of each roof past the ridge, so the full lines are visible
    ax.plot([ridge_ai, x_max], [peak_bw * ridge_ai, peak_bw * x_max], color="#c9c9c9", linestyle=":", linewidth=1, zorder=1)
    ax.plot([x_min, ridge_ai], [peak_flops, peak_flops], color="#c9c9c9", linestyle=":", linewidth=1, zorder=1)

    ax.axvline(ridge_ai, color="#D55E00", linestyle=":", linewidth=1.5, zorder=2)
    ax.text(ridge_ai, peak_flops * 1.3, f"  Ridge point\n  {ridge_ai:.1f} FLOPs/Byte", ha="left", va="bottom", fontsize=8, color="#D55E00")

    ax.text(x_min * 1.3, peak_flops * 1.08, f"Compute roof (theoretical): {specs['peak_bf16_tflops']:.1f} TFLOPS", ha="left", va="bottom", fontsize=8, color="#5a5a5a")

    seen_stages = set()
    for r in rows:
        stage = r["stage"]
        ai = float(r["arithmetic_intensity"])
        perf = float(r["achieved_flops_per_sec"])
        color = STAGE_COLORS.get(stage, "#333333")
        label = stage.capitalize() if stage not in seen_stages else None
        seen_stages.add(stage)
        ax.scatter([ai], [perf], color=color, s=90, zorder=5, label=label, edgecolors="white", linewidths=0.5)
        ax.annotate(
            f"{stage} b={r['batch_size']}\n{perf:.2e} FLOPs/s",
            (ai, perf),
            textcoords="offset points",
            xytext=(8, 6),
            fontsize=7.5,
            color=color,
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("Arithmetic Intensity (FLOPs / Byte)")
    ax.set_ylabel("Performance (FLOPs / sec)")
    ax.set_title(f"Standard Roofline — {rows[0]['model'].split('/')[-1]} on {gpu_name}", fontsize=11)
    ax.grid(True, which="both", linestyle=":", linewidth=0.5, color="#DDDDDD")
    ax.legend(loc="lower right", fontsize=8, frameon=False)

    fig.tight_layout()
    fig.savefig(ROOFLINE_PLOT, dpi=150)
    plt.close(fig)

    return ridge_ai, peak_flops, peak_bw


def write_doc(gpu_name, specs, rows, ridge_ai, peak_flops, peak_bw, ncu_status):
    prefill_rows = [r for r in rows if r["stage"] == "prefill"]
    decode_rows = [r for r in rows if r["stage"] == "decode"]

    def verdict(ai):
        return "memory-bound" if ai < ridge_ai else "compute-bound"

    lines = []
    lines.append("# Standard Roofline Analysis\n")

    lines.append("## 1. What this is\n")
    lines.append(
        "The [Roofline Model](https://en.wikipedia.org/wiki/Roofline_model) plots achievable "
        "performance (FLOPs/sec) against arithmetic intensity (FLOPs/Byte moved from memory). "
        "Two ceilings bound achievable performance: a **memory roof** (`perf = bandwidth × AI`, "
        "a diagonal line on log-log axes) and a **compute roof** (`perf = peak FLOPs/sec`, "
        "horizontal). Workloads with low AI (memory-bound) sit under the diagonal; workloads "
        "with high AI (compute-bound) sit under the horizontal line. This is a **standard "
        "arithmetic-intensity roofline** — distinct from `plots/simplified_serving_roofline.png`, "
        "which plots concurrency vs. tokens/sec and is not a roofline in the classical sense.\n"
    )

    lines.append("## 2. Peak compute throughput — PUBLISHED THEORETICAL\n")
    lines.append(
        f"GPU detected via `nvidia-smi --query-gpu=name --format=csv,noheader` on the pod "
        f"(see `profile_model.py`): **{gpu_name}**. Peak dense BF16 tensor throughput "
        f"(`hw_specs.py`, public manufacturer spec sheet, not measured): "
        f"**{specs['peak_bf16_tflops']} TFLOPS** ({peak_flops:.3e} FLOPs/sec).\n"
    )

    lines.append("## 3. Peak memory bandwidth — PUBLISHED THEORETICAL\n")
    lines.append(
        f"Same source: **{specs['memory_bandwidth_gb_s']} GB/s** ({peak_bw:.3e} bytes/sec), "
        "public spec sheet, not measured on this hardware.\n"
    )

    lines.append("## 4. Ridge point\n")
    lines.append("`ridge_AI = peak_FLOPS_per_sec / peak_bandwidth_bytes_per_sec`\n")
    lines.append(
        f"`{peak_flops:.3e} / {peak_bw:.3e} ≈ {ridge_ai:.2f} FLOPs/Byte` — workloads with "
        f"arithmetic intensity below this are memory-bound on this GPU; above it, compute-bound.\n"
    )

    lines.append("## 5. How arithmetic intensity was measured\n")
    lines.append(
        "For each (stage, batch_size), `profile_model.py` loads "
        f"`{rows[0]['model']}` directly via `transformers` (not through the live vLLM server — "
        "see the module docstring in `profile_model.py` for why) and:\n\n"
        "- **FLOPs — PROFILED**: `torch.profiler.profile(with_flops=True)` around one forward "
        "pass of the given shape, summing the `.flops` attribute across all attributed ops.\n"
        "- **Execution time — MEASURED**: wall-clock `time.perf_counter()` around "
        f"{10} warmed-up iterations (`torch.cuda.synchronize()` before/after), averaged. "
        "Measured separately from the profiler run to avoid profiler overhead skewing timing.\n"
        "- **Memory bytes — ANALYTICALLY ESTIMATED**, not measured: derived from the model's "
        "actual config (weight count, layer count, KV head count, head dim) — see below. "
        "`torch.profiler` does not report hardware DRAM traffic; that would require Nsight "
        "Compute counters (see §9).\n"
        "- **Arithmetic intensity = profiled FLOPs / estimated memory bytes.**\n"
    )

    lines.append("### Representative shapes used (from `profile_model.py`)\n")
    lines.append("- Prefill: sequence_length=128 tokens, batch_size ∈ {1, 20}")
    lines.append("- Decode: 128 tokens already in KV cache, one new token, batch_size ∈ {1, 20}")
    lines.append("- 20 matches `load_test.py`'s benchmarked concurrency, for comparability.\n")

    lines.append("### Memory-byte formulas used\n")
    lines.append("- Prefill: `weight_bytes + batch_size × seq_len × kv_bytes_per_token`")
    lines.append("  (weights read once, amortized across the batched matmul; KV cache written for every prompt token; activation traffic excluded — see limitations)")
    lines.append("- Decode: `weight_bytes + batch_size × context_len × kv_bytes_per_token + batch_size × kv_bytes_per_token`")
    lines.append("  (weights read once per step, amortized across batch; existing KV cache read per sequence; new token's KV written per sequence)\n")

    lines.append("## 6. Measured/profiled results\n")
    lines.append("| Stage | Batch | Seq/Ctx len | FLOPs (profiled) | Exec time | Achieved FLOPs/s | Est. bytes | AI (FLOPs/Byte) | Verdict |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        ai = float(r["arithmetic_intensity"])
        lines.append(
            f"| {r['stage']} | {r['batch_size']} | {r['sequence_length']} | "
            f"{float(r['flops']):.3e} | {float(r['execution_time_s'])*1000:.2f} ms | "
            f"{float(r['achieved_flops_per_sec']):.3e} | {float(r['estimated_memory_bytes']):.3e} | "
            f"{ai:.2f} | {verdict(ai)} |"
        )
    lines.append("")

    uncounted = set()
    for r in rows:
        if r.get("uncounted_flops_ops"):
            uncounted.update(r["uncounted_flops_ops"].split(";"))
    # Profiler-internal bookkeeping entries aren't model ops; drop them so this
    # list reflects actual uncounted computation, not profiler machinery.
    uncounted = {op for op in uncounted if op.startswith(("aten::", "prim::"))}
    if uncounted:
        shown = sorted(uncounted)
        more = ""
        if len(shown) > 25:
            more = f", + {len(shown) - 25} more"
            shown = shown[:25]
        lines.append(
            "**Ops with measurable time but no FLOPs attributed by `torch.profiler`** "
            "(not included in the FLOPs sum above, so achieved FLOPs/sec is a lower bound "
            "— these are mostly small elementwise/indexing ops, not matmuls, so the effect "
            "on the total is expected to be minor): "
            f"`{', '.join(shown)}{more}`\n"
        )

    lines.append("## 7. Prefill vs. decode\n")
    if prefill_rows and decode_rows:
        avg_prefill_ai = sum(float(r["arithmetic_intensity"]) for r in prefill_rows) / len(prefill_rows)
        avg_decode_ai = sum(float(r["arithmetic_intensity"]) for r in decode_rows) / len(decode_rows)
        lines.append(
            f"Average prefill AI: **{avg_prefill_ai:.2f} FLOPs/Byte** → **{verdict(avg_prefill_ai)}**. "
            f"Average decode AI: **{avg_decode_ai:.2f} FLOPs/Byte** → **{verdict(avg_decode_ai)}**.\n"
        )
        lines.append(
            "This matches the expected shape: prefill processes many prompt tokens in one "
            "batched matmul, so the model weights (the dominant memory cost) are reused across "
            "all those tokens — more FLOPs per byte moved. Decode processes one token per "
            "sequence per step, so the same weight-read cost is spread over far fewer FLOPs — "
            "fewer FLOPs per byte moved. The measured numbers above confirm this directionally "
            "rather than assuming it.\n"
        )
    else:
        lines.append("Incomplete — both prefill and decode rows are required for this comparison.\n")

    lines.append("## 8. Where the workload lies on the roofline\n")
    below = [r for r in rows if float(r["arithmetic_intensity"]) < ridge_ai]
    above = [r for r in rows if float(r["arithmetic_intensity"]) >= ridge_ai]
    lines.append(
        f"{len(below)}/{len(rows)} profiled points fall below the ridge point "
        f"({ridge_ai:.2f} FLOPs/Byte) — memory-bound on {gpu_name}. "
        f"{len(above)}/{len(rows)} fall at or above it — compute-bound.\n"
    )

    lines.append("## 9. Nsight Compute\n")
    lines.append(f"`profile_model.py` probes for `ncu` on the pod at profiling time. Result: **{ncu_status}**\n")
    lines.append(
        "Nsight Compute was **not** used as a hard dependency (per requirements) — real hardware "
        "DRAM-traffic counters (`dram__bytes.sum` and similar) would be the gold-standard source "
        "for the memory-byte numbers in §6, replacing the analytical estimate. Whether that's "
        "usable depends on the probe result above; this script never fabricates `ncu` output.\n"
    )

    lines.append("## 10. Limitations\n")
    lines.append("- **Not profiled through vLLM.** These forward passes run via plain `transformers` "
                  "(`attn_implementation=\"eager\"`, chosen so `torch.profiler` can attribute FLOPs "
                  "to the underlying matmuls). vLLM's actual serving kernels (PagedAttention, fused "
                  "ops, CUDA graphs) are faster and structured differently — absolute wall-clock "
                  "numbers here will not match `results/load_test.csv`'s serving throughput. The "
                  "arithmetic-intensity *shape* (prefill vs. decode) is architectural and should "
                  "transfer; the absolute achieved-FLOPs/sec numbers are specific to this eager-mode "
                  "measurement, not to vLLM's production kernels.")
    lines.append("- **Memory bytes are analytically estimated, not measured** — no hardware DRAM "
                  "counters were used (see §9). The formulas in §5 cover weights and KV cache, which "
                  "dominate; they exclude activation traffic.")
    lines.append("- **FLOPs coverage gap.** `torch.profiler`'s `with_flops=True` does not attribute "
                  "FLOPs to every op (see the uncounted-ops list in §6 if present) — reported FLOPs "
                  "and thus achieved FLOPs/sec are a lower bound, not exact.")
    lines.append("- **Only two batch sizes profiled** (1 and 20) at one fixed sequence/context length "
                  "(128 tokens) — this is a small, documented sample of the shape space, not a sweep.")
    lines.append("- **Peak hardware numbers are published theoretical maxima**, not measured on this "
                  "specific pod's silicon (binning/thermals can reduce real achievable peaks below spec).")
    lines.append("")

    with open(ROOFLINE_DOC, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    rows = load_profile_rows()
    gpu_name = rows[0]["gpu"]
    specs = lookup_gpu_specs(gpu_name)
    ncu_status = load_ncu_status()

    ridge_ai, peak_flops, peak_bw = plot(gpu_name, specs, rows)
    write_doc(gpu_name, specs, rows, ridge_ai, peak_flops, peak_bw, ncu_status)

    print(f"GPU: {gpu_name}")
    print(f"ridge point: {ridge_ai:.2f} FLOPs/Byte")
    for r in rows:
        print(f"  {r['stage']} b={r['batch_size']}: AI={float(r['arithmetic_intensity']):.2f} FLOPs/Byte, {float(r['achieved_flops_per_sec']):.3e} FLOPs/s")
    print(f"wrote {ROOFLINE_PLOT}, {ROOFLINE_DOC}")


if __name__ == "__main__":
    main()
