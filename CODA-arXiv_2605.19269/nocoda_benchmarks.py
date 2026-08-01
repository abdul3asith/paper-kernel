"""Plot and explain the no-CODA SGEMM data in Results.csv.

Run: python nocoda_benchmarks.py
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
RESULTS_CSV = ROOT / "Results.csv"
PLOT_FILE = ROOT / "nocoda_benchmarks.png"
REPORT_FILE = ROOT / "nocoda_benchmark_report.txt"
DISPLAY_NAMES = {
    "pytorch-cuda": "PyTorch / cuBLAS",
    "cuda-naive": "Naive CUDA",
    "triton-naive": "Naive Triton",
}
COLORS = {
    "pytorch-cuda": "#2f6fed",
    "cuda-naive": "#f28e2b",
    "triton-naive": "#e15759",
}


def load_latest_results() -> list[dict[str, str]]:
    """Load the latest row for each implementation and validate fairness."""
    with RESULTS_CSV.open(newline="", encoding="utf-8") as csv_file:
        rows = list(csv.DictReader(csv_file))

    latest = {row["implementation"]: row for row in rows}
    missing = DISPLAY_NAMES.keys() - latest.keys()
    if missing:
        raise RuntimeError(f"Results.csv is missing: {', '.join(sorted(missing))}")

    selected = [latest[name] for name in DISPLAY_NAMES]
    shapes = {(row["m"], row["k"], row["n"]) for row in selected}
    devices = {row["device"] for row in selected}
    if len(shapes) != 1 or len(devices) != 1:
        raise RuntimeError("All compared rows must use the same shape and GPU.")
    return selected


def build_report(rows: list[dict[str, str]]) -> str:
    by_name = {row["implementation"]: row for row in rows}
    pytorch_ms = float(by_name["pytorch-cuda"]["average_ms"])
    cuda_ms = float(by_name["cuda-naive"]["average_ms"])
    triton_ms = float(by_name["triton-naive"]["average_ms"])
    m, k, n = (by_name["pytorch-cuda"][key] for key in ("m", "k", "n"))
    device = by_name["pytorch-cuda"]["device"]

    return f"""No-CODA SGEMM benchmark interpretation
======================================
Workload: C[M,N] = A[M,K] @ B[K,N], M={m}, K={k}, N={n}, FP32
Device: {device}; 30 timed repetitions after warm-up

Measured comparison
-------------------
PyTorch/cuBLAS: {pytorch_ms:.3f} ms (baseline)
Naive CUDA:     {cuda_ms:.3f} ms ({cuda_ms / pytorch_ms:.1f}x slower than cuBLAS)
Naive Triton:   {triton_ms:.3f} ms ({triton_ms / pytorch_ms:.1f}x slower than cuBLAS,
                {triton_ms / cuda_ms:.1f}x slower than naive CUDA)

Why PyTorch/cuBLAS wins
-----------------------
PyTorch dispatches matrix multiplication to cuBLAS, NVIDIA's tuned GEMM library.
cuBLAS uses tiled matrix multiplication: a thread block cooperatively loads tiles
of A and B into fast on-chip storage, then reuses each value for many fused
multiply-adds. This raises arithmetic intensity (FLOPs per byte moved), reduces
global-memory traffic, and keeps the GPU's execution units busy. Its kernels are
selected and tuned for the matrix shape, datatype, and GPU architecture.

Why the naive CUDA kernel is slower
-----------------------------------
Each CUDA thread computes one output and independently walks through K. Neighboring
threads obtain reasonably coalesced B accesses, and repeated A values can benefit
from caches, but the block never explicitly stages reusable A/B tiles in shared
memory. Operands are fetched much more often than in a tiled GEMM. The result has
lower arithmetic intensity and less instruction-level parallelism for hiding
memory latency.

Why this naive Triton kernel is slowest
---------------------------------------
The launch creates one Triton program per output element: M*N programs. Each
program vectorizes only one dot product in K chunks. B is read with a stride of N,
and no cooperative 2D output tile reuses A and B across outputs. Triton is a kernel
language and compiler, not an automatic guarantee of speed. A normal fast Triton
GEMM assigns each program a BLOCK_M x BLOCK_N output tile, loops over BLOCK_K,
groups program IDs for cache locality, and uses tl.dot so the compiler can target
efficient matrix-multiply hardware.

Jargon decoder
--------------
Tiling: split matrices into blocks so operands can be reused on-chip.
Coalescing: adjacent GPU lanes access adjacent addresses in fewer transactions.
Arithmetic intensity: computation divided by memory traffic; GEMM wants it high.
Occupancy: resident warps per SM, which helps the GPU hide latency.
Warp: 32 CUDA threads issued together on an NVIDIA GPU.
SM: streaming multiprocessor, the GPU unit that schedules and executes warps.
Shared memory: fast, programmer-managed memory shared by a thread block.
FMA: fused multiply-add, GEMM's core operation; conventionally counted as 2 FLOPs.
TFLOP/s: trillions of floating-point operations completed per second.

Important caveats
-----------------
C[0,0] differs because each runner initializes A and B differently. Those values
are per-run correctness checks, not outputs that can be compared across runners.
The measurements characterize this GPU, shape, stack, and these naive designs.
Multiple trials and matrix sizes would make a stronger performance study.
"""


def plot_results(rows: list[dict[str, str]]) -> None:
    implementations = [row["implementation"] for row in rows]
    labels = [DISPLAY_NAMES[name] for name in implementations]
    colors = [COLORS[name] for name in implementations]
    latency = [float(row["average_ms"]) for row in rows]
    throughput = [float(row["tflops"]) for row in rows]

    fig, (latency_ax, throughput_ax) = plt.subplots(1, 2, figsize=(13, 6))
    fig.suptitle("No-CODA FP32 SGEMM — Tesla T4, 1024³", fontsize=16, weight="bold")

    latency_bars = latency_ax.bar(labels, latency, color=colors)
    latency_ax.set_title("Average latency (lower is better)")
    latency_ax.set_ylabel("Milliseconds — logarithmic scale")
    latency_ax.set_yscale("log")
    latency_ax.bar_label(
        latency_bars,
        labels=[f"{value:.3f} ms" for value in latency],
        padding=4,
    )
    latency_ax.grid(axis="y", alpha=0.25, which="both")

    throughput_bars = throughput_ax.bar(labels, throughput, color=colors)
    throughput_ax.set_title("Throughput (higher is better)")
    throughput_ax.set_ylabel("TFLOP/s")
    throughput_ax.bar_label(
        throughput_bars,
        labels=[f"{value:.3f}" for value in throughput],
        padding=4,
    )
    throughput_ax.grid(axis="y", alpha=0.25)

    baseline = latency[0]
    for bar, value in zip(latency_bars, latency):
        latency_ax.text(
            bar.get_x() + bar.get_width() / 2,
            value / 1.6,
            f"{value / baseline:.1f}x",
            ha="center",
            va="top",
            color="white",
            weight="bold",
        )

    fig.text(
        0.5,
        0.02,
        "Same 1024³ workload and 30 timed repetitions; latency ratios use cuBLAS as 1.0x.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.93))
    fig.savefig(PLOT_FILE, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    rows = load_latest_results()
    report = build_report(rows)
    plot_results(rows)
    REPORT_FILE.write_text(report, encoding="utf-8")
    print(report)
    print(f"Chart saved to: {PLOT_FILE}")
    print(f"Report saved to: {REPORT_FILE}")


if __name__ == "__main__":
    main()
