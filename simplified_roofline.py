"""The ORIGINAL simplified serving/batch roofline (X=concurrency, Y=tokens/sec),
kept only for before/after comparison against the standard arithmetic-intensity
roofline in roofline.py. This is NOT a standard roofline -- see plot title.

GPU is read from results/roofline_profile.csv (written by profile.sh), not
hardcoded, so run profile.sh at least once before this script.
"""

import csv
import os

import matplotlib.pyplot as plt

from hw_specs import lookup_gpu_specs

RESULTS_FILE = "results/load_test.csv"
PROFILE_FILE = "results/roofline_profile.csv"
ROOFLINE_CSV = "results/simplified_roofline_data.csv"
ROOFLINE_PLOT = "plots/simplified_serving_roofline.png"
ROOFLINE_DOC = "results/simplified_roofline_analysis.md"

# --- THEORETICAL model specs (Mistral-7B-Instruct-v0.2, BF16) ---
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
MODEL_PARAMS_B = 7.24  # billion parameters
MODEL_SIZE_GB = 14.5  # vLLM reported 13.5 GiB at load time (see EXPERIMENTS.md log)
FLOPS_PER_TOKEN_G = 2 * MODEL_PARAMS_B  # 2*N approximation, GFLOPs/token


def detect_gpu():
    if not os.path.exists(PROFILE_FILE):
        raise SystemExit(
            f"{PROFILE_FILE} not found. Run ./profile.sh first -- it detects the GPU "
            "via nvidia-smi on the pod and this script reads that instead of guessing."
        )
    with open(PROFILE_FILE) as f:
        row = next(csv.DictReader(f))
    return row["gpu"]


def theoretical_ceilings(specs):
    memory_ceiling = specs["memory_bandwidth_gb_s"] / MODEL_SIZE_GB
    compute_ceiling = (specs["peak_bf16_tflops"] * 1000) / FLOPS_PER_TOKEN_G
    ridge_point = compute_ceiling / memory_ceiling
    return memory_ceiling, compute_ceiling, ridge_point


def load_measured_points():
    if not os.path.exists(RESULTS_FILE):
        return []
    points = []
    with open(RESULTS_FILE) as f:
        for row in csv.DictReader(f):
            if row["model"] != MODEL_NAME:
                continue
            points.append(
                {"concurrency": int(row["users"]), "tokens_per_sec": float(row["actual_tokens_per_sec"])}
            )
    return points


def write_csv(memory_ceiling, compute_ceiling, ridge_point, measured):
    os.makedirs("results", exist_ok=True)
    with open(ROOFLINE_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["concurrency", "measured_tokens_per_sec", "throughput_per_gpu", "memory_ceiling", "compute_ceiling", "ridge_point"]
        )
        if measured:
            for p in measured:
                writer.writerow(
                    [p["concurrency"], f"{p['tokens_per_sec']:.2f}", f"{p['tokens_per_sec']:.2f}", f"{memory_ceiling:.2f}", f"{compute_ceiling:.2f}", f"{ridge_point:.2f}"]
                )
        else:
            writer.writerow(["PENDING", "", "", f"{memory_ceiling:.2f}", f"{compute_ceiling:.2f}", f"{ridge_point:.2f}"])


def plot(gpu_name, memory_ceiling, compute_ceiling, ridge_point, measured):
    os.makedirs("plots", exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5.5))

    max_measured_x = max((p["concurrency"] for p in measured), default=1)
    x_max = max(ridge_point * 2, max_measured_x * 1.5, 10)

    ax.axhline(memory_ceiling, color="#767676", linestyle="--", linewidth=2, zorder=2)
    ax.text(x_max * 0.98, memory_ceiling * 1.1, f"Memory-bandwidth ceiling (theoretical)\n{memory_ceiling:.0f} tok/s", ha="right", va="bottom", fontsize=8, color="#5a5a5a")

    ax.axhline(compute_ceiling, color="#767676", linestyle="--", linewidth=2, zorder=2)
    ax.text(x_max * 0.98, compute_ceiling * 0.9, f"Compute ceiling (theoretical)\n{compute_ceiling:.0f} tok/s", ha="right", va="top", fontsize=8, color="#5a5a5a")

    ax.axvline(ridge_point, color="#D55E00", linestyle=":", linewidth=1.5, zorder=2)
    mid_y = (memory_ceiling * compute_ceiling) ** 0.5
    ax.text(ridge_point, mid_y, f"  Ridge point (theoretical)\n  ~{ridge_point:.0f} concurrent reqs", ha="left", va="center", fontsize=8, color="#D55E00")

    if measured:
        xs = [p["concurrency"] for p in measured]
        ys = [p["tokens_per_sec"] for p in measured]
        ax.scatter(xs, ys, color="#0072B2", s=70, zorder=5)
        for x, y in zip(xs, ys):
            ax.annotate(f"Measured: {y:.0f} tok/s", (x, y), textcoords="offset points", xytext=(0, -14), ha="center", fontsize=8, color="#0072B2")
    else:
        ax.text(x_max * 0.5, memory_ceiling * 3, "No measured data yet — run load_test.py first", ha="center", fontsize=9, color="#a00000", style="italic")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1, x_max)
    ax.set_ylim(memory_ceiling * 0.5, compute_ceiling * 2)
    ax.set_xlabel("Concurrent users / approximate batch size")
    ax.set_ylabel("Throughput (tokens/sec/GPU)")
    ax.set_title(
        f"Simplified Serving Roofline — NOT a standard arithmetic-intensity roofline\n"
        f"{MODEL_NAME.split('/')[-1]} on {gpu_name}",
        fontsize=10,
    )
    ax.grid(True, which="both", linestyle=":", linewidth=0.5, color="#DDDDDD")

    fig.tight_layout()
    fig.savefig(ROOFLINE_PLOT, dpi=150)
    plt.close(fig)


def write_doc(gpu_name, specs, memory_ceiling, compute_ceiling, ridge_point, measured):
    lines = []
    lines.append("# Simplified Serving Roofline (superseded — see roofline_analysis.md)\n")
    lines.append(
        "This is the **original simplified serving/batch roofline**, kept only for "
        "before/after comparison. It is **not** a standard academic arithmetic-intensity "
        "roofline — see `results/roofline_analysis.md` for the real one (AI vs FLOPs/sec, "
        "measured via `torch.profiler`).\n"
    )

    lines.append("## Theoretical inputs\n")
    lines.append(f"- GPU: {gpu_name} (detected via nvidia-smi in profile.sh, not hardcoded)")
    lines.append(f"- Memory bandwidth: {specs['memory_bandwidth_gb_s']} GB/s (public spec)")
    lines.append(f"- Peak BF16 tensor compute: {specs['peak_bf16_tflops']} TFLOPS (dense, public spec)")
    lines.append(f"- Model: {MODEL_NAME}")
    lines.append(f"- Model size in BF16: ~{MODEL_SIZE_GB} GB")
    lines.append(f"- Estimated FLOPs/token: ~{FLOPS_PER_TOKEN_G:.1f} GFLOPs (2 × {MODEL_PARAMS_B}B params)\n")

    lines.append("## Ceilings\n")
    lines.append(f"- Memory ceiling: {specs['memory_bandwidth_gb_s']} / {MODEL_SIZE_GB} ≈ {memory_ceiling:.1f} tokens/s")
    lines.append(f"- Compute ceiling: {specs['peak_bf16_tflops']*1000:.0f} / {FLOPS_PER_TOKEN_G:.1f} ≈ {compute_ceiling:.0f} tokens/s")
    lines.append(f"- Ridge point: {compute_ceiling:.0f} / {memory_ceiling:.1f} ≈ {ridge_point:.0f} concurrent requests\n")

    lines.append("## Measured data\n")
    if measured:
        lines.append("| Concurrency | Measured tokens/sec/GPU (actual, from load_test.py) |")
        lines.append("|---|---|")
        for p in measured:
            lines.append(f"| {p['concurrency']} | {p['tokens_per_sec']:.1f} |")
        lines.append("")
        lines.append(
            "This now uses `actual_tokens_per_sec` directly from `results/load_test.csv` "
            "(real completion-token counts from the API, no longer approximated).\n"
        )
    else:
        lines.append("No measured data yet — run `python load_test.py` first.\n")

    with open(ROOFLINE_DOC, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    gpu_name = detect_gpu()
    specs = lookup_gpu_specs(gpu_name)
    memory_ceiling, compute_ceiling, ridge_point = theoretical_ceilings(specs)
    measured = load_measured_points()

    write_csv(memory_ceiling, compute_ceiling, ridge_point, measured)
    plot(gpu_name, memory_ceiling, compute_ceiling, ridge_point, measured)
    write_doc(gpu_name, specs, memory_ceiling, compute_ceiling, ridge_point, measured)

    print(f"GPU (detected): {gpu_name}")
    print(f"memory ceiling:  {memory_ceiling:.1f} tokens/s")
    print(f"compute ceiling: {compute_ceiling:.1f} tokens/s")
    print(f"ridge point:     ~{ridge_point:.0f} concurrent requests")
    print(f"measured points: {len(measured)}")
    print(f"wrote {ROOFLINE_CSV}, {ROOFLINE_PLOT}, {ROOFLINE_DOC}")


if __name__ == "__main__":
    main()
