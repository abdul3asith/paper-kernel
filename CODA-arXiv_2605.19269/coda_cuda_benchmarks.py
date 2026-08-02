"""Generate plots and a report for the CODA vs no-CODA CUDA benchmark."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
RESULTS_FILE = ROOT / "coda_cuda_results.csv"
TIMELINE_FILE = ROOT / "coda_cuda_timeline.csv"
PLOT_FILE = ROOT / "coda_cuda_benchmarks.png"
REPORT_FILE = ROOT / "coda_cuda_benchmark_report.txt"

ORDER = ["coda-fused", "no-coda-separate"]
LABELS = {
    "coda-fused": "CODA\n1 fused kernel",
    "no-coda-separate": "No-CODA\n3 separate kernels",
}
COLORS = {"coda-fused": "#2f6fed", "no-coda-separate": "#f28e2b"}


def load_results() -> list[dict[str, str]]:
    with RESULTS_FILE.open(newline="", encoding="utf-8") as csv_file:
        rows = list(csv.DictReader(csv_file))
    by_name = {row["implementation"]: row for row in rows}
    missing = set(ORDER) - by_name.keys()
    if missing:
        raise RuntimeError(f"Missing benchmark rows: {', '.join(sorted(missing))}")

    selected = [by_name[name] for name in ORDER]
    comparison_keys = ("gpu", "m", "k", "n", "repeats")
    if any(len({row[key] for row in selected}) != 1 for key in comparison_keys):
        raise RuntimeError("Both rows must use the same GPU, shape, and repetitions.")
    return selected


def load_timeline() -> list[dict[str, str]]:
    with TIMELINE_FILE.open(newline="", encoding="utf-8") as csv_file:
        return list(csv.DictReader(csv_file))


def build_report(rows: list[dict[str, str]]) -> str:
    coda, no_coda = rows
    coda_ms = float(coda["average_ms"])
    no_coda_ms = float(no_coda["average_ms"])
    speedup = no_coda_ms / coda_ms
    latency_reduction = (no_coda_ms - coda_ms) / no_coda_ms * 100
    saved_ms = no_coda_ms - coda_ms
    output_bytes = int(coda["m"]) * int(coda["n"]) * 4
    avoided_bytes = 4 * output_bytes

    return f"""CODA vs no-CODA CUDA benchmark report
=======================================
Operation: ReLU(A x B + bias), FP32
Shape: M={coda['m']}, K={coda['k']}, N={coda['n']}
GPU: {coda['gpu']}; repetitions: {coda['repeats']} after warm-up

Measured results
----------------
CODA fused epilogue: {coda_ms:.3f} ms, {float(coda['tflops']):.3f} effective TFLOP/s
No-CODA separate:    {no_coda_ms:.3f} ms, {float(no_coda['tflops']):.3f} effective TFLOP/s
CODA saves {saved_ms:.3f} ms per layer, is {speedup:.2f}x faster, and reduces
end-to-end layer latency by {latency_reduction:.1f}% in this run.

Correctness signal
------------------
Both programs produced C[0,0] = {float(coda['c00']):.6f}. They use the same
deterministic input and bias initialization. Checking the full output against a
reference would be stronger than this single-element smoke test.

What changed
------------
Both versions use the same 16x16 shared-memory tiled FP32 GEMM main loop. The
difference is the epilogue—the work performed after the dot-product accumulator
is ready:

CODA keeps the accumulator in a register, adds bias, applies ReLU, and performs
one final global-memory store, all inside one kernel launch.

No-CODA launches three kernels: GEMM writes raw C to global memory, the bias
kernel reads and rewrites C, and the ReLU kernel reads and rewrites C again.

Why fusion helps
----------------
1. Global-memory traffic: for a 1024x1024 FP32 output, C occupies 4 MiB. The
   separate path performs one GEMM write plus two read-modify-write passes: about
   20 MiB of C traffic. The fused path writes C once: about 4 MiB. Fusion avoids
   roughly {avoided_bytes / 1024**2:.0f} MiB of intermediate C traffic per layer
   invocation, ignoring cache effects.
2. Kernel-launch overhead: the fused path launches one timed kernel instead of
   three. It avoids two launches and the scheduling boundaries between stages.
3. Locality: bias and ReLU consume the accumulator while it is still in a fast
   per-thread register. The intermediate value never needs a VRAM round trip.
4. Arithmetic intensity: fusion performs more useful work for each byte moved
   through global memory, improving the compute-to-memory-traffic ratio.

Jargon decoder
--------------
Epilogue: operations applied to GEMM accumulators before storing output.
Fusion: combine operations that were separate kernels into one kernel.
Register: very fast thread-private storage located on the GPU execution unit.
Global memory/VRAM: large off-chip GPU memory with much higher access latency.
Shared memory: fast on-chip storage shared by threads in one CUDA block.
Tiling: cooperatively load small A/B blocks and reuse them for many FMAs.
FMA: fused multiply-add; conventionally counted as two floating-point operations.
Arithmetic intensity: FLOPs per byte transferred; higher can reduce memory limits.
Kernel launch: CPU/runtime command that schedules a GPU kernel.

Interpretation limits
---------------------
The reported TFLOP/s counts GEMM FLOPs but divides by full-layer time, so it is
an effective comparison metric—not a measurement of every bias/ReLU operation.
This is one run, one shape, one tile size, and one Tesla T4. Repeat trials and
multiple shapes are needed for confidence intervals and broader conclusions.

Timeline measurement
--------------------
The graph shows every timed invocation after a 100-invocation untimed warm-up
buffer. CUDA events measure elapsed GPU time at microsecond-scale resolution;
nanosecond sampling would imply more precision than this measurement provides.
"""


def make_plot(rows: list[dict[str, str]], timeline: list[dict[str, str]]) -> None:
    iterations = [int(row["iteration"]) for row in timeline]
    coda_us = [float(row["coda_fused_ms"]) * 1_000 for row in timeline]
    no_coda_us = [float(row["no_coda_separate_ms"]) * 1_000 for row in timeline]
    coda_mean_us = float(rows[0]["average_ms"]) * 1_000
    no_coda_mean_us = float(rows[1]["average_ms"]) * 1_000

    fig, axis = plt.subplots(figsize=(13, 6.5))
    axis.plot(
        iterations,
        coda_us,
        color=COLORS["coda-fused"],
        linewidth=2,
        label=f"CODA fused — mean {coda_mean_us:.0f} µs",
    )
    axis.plot(
        iterations,
        no_coda_us,
        color=COLORS["no-coda-separate"],
        linewidth=2,
        label=f"No-CODA separate — mean {no_coda_mean_us:.0f} µs",
    )
    axis.fill_between(
        iterations,
        coda_us,
        no_coda_us,
        color=COLORS["coda-fused"],
        alpha=0.10,
        label="Per-iteration latency gap",
    )
    axis.axhline(coda_mean_us, color=COLORS["coda-fused"], alpha=0.55, linestyle="--")
    axis.axhline(
        no_coda_mean_us,
        color=COLORS["no-coda-separate"],
        alpha=0.55,
        linestyle="--",
    )
    axis.scatter(
        [iterations[0], iterations[-1]],
        [coda_us[0], coda_us[-1]],
        color=COLORS["coda-fused"],
        s=35,
        zorder=3,
    )
    axis.scatter(
        [iterations[0], iterations[-1]],
        [no_coda_us[0], no_coda_us[-1]],
        color=COLORS["no-coda-separate"],
        s=35,
        zorder=3,
    )
    axis.set_title("CODA vs no-CODA: every timed CUDA layer invocation")
    axis.set_xlabel("Timed iteration (after 100 untimed warm-up iterations)")
    axis.set_ylabel("Full-layer GPU time (microseconds)")
    axis.set_xlim(iterations[0], iterations[-1])
    axis.grid(alpha=0.25)
    axis.legend(loc="best")

    fig.suptitle(
        "ReLU(A×B+bias) — FP32, Tesla T4, 1024³",
        fontsize=15,
        weight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(PLOT_FILE, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    rows = load_results()
    timeline = load_timeline()
    report = build_report(rows)
    make_plot(rows, timeline)
    REPORT_FILE.write_text(report, encoding="utf-8")
    print(report)
    print(f"Chart saved to: {PLOT_FILE}")
    print(f"Report saved to: {REPORT_FILE}")


if __name__ == "__main__":
    main()
