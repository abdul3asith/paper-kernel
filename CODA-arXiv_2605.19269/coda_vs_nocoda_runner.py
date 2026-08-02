"""Compile and benchmark fused CODA and non-fused CUDA layers on Modal."""

import csv
from pathlib import Path
import re
import subprocess

import modal


ROOT = Path(__file__).resolve().parent
CODA_SOURCE = ROOT / "coda" / "cuda" / "coda-sgemm.cu"
NO_CODA_SOURCE = ROOT / "no-coda" / "gemm" / "cuda" / "tiled-sgemm.cu"
RESULTS_FILE = ROOT / "coda_cuda_results.csv"
TIMELINE_FILE = ROOT / "coda_cuda_timeline.csv"

app = modal.App("coda-vs-nocoda-cuda")
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-devel-ubuntu22.04",
        add_python="3.11",
    )
    .add_local_file(CODA_SOURCE, "/root/coda-sgemm.cu", copy=True)
    .add_local_file(NO_CODA_SOURCE, "/root/tiled-sgemm.cu", copy=True)
)


def parse_output(implementation: str, output: str) -> dict:
    def capture(pattern: str) -> str:
        match = re.search(pattern, output)
        if match is None:
            raise RuntimeError(f"Could not parse {implementation} output: {pattern}")
        return match.group(1)

    result = {
        "implementation": implementation,
        "gpu": capture(r"GPU: (.+)"),
        "average_ms": float(capture(r"Average full-layer time: ([0-9.eE+-]+) ms")),
        "tflops": float(capture(r"GEMM throughput: ([0-9.eE+-]+) TFLOP/s")),
        "c00": float(capture(r"C\[0, 0\]: ([0-9.eE+-]+)")),
    }
    result["iteration_times_ms"] = [
        float(value)
        for value in capture(r"Iteration times \(ms\): ([0-9.,eE+ -]+)").split(",")
    ]
    return result


@app.function(image=image, gpu="T4", timeout=600)
def run_benchmarks(m: int, k: int, n: int, repeats: int) -> list[dict]:
    programs = [
        ("coda-fused", "/root/coda-sgemm.cu", "/tmp/coda-sgemm"),
        ("no-coda-separate", "/root/tiled-sgemm.cu", "/tmp/no-coda-sgemm"),
    ]
    results = []
    for implementation, source, executable in programs:
        subprocess.run(
            ["nvcc", "-O3", "-std=c++17", source, "-o", executable],
            check=True,
        )
        completed = subprocess.run(
            [executable, str(m), str(k), str(n), str(repeats)],
            check=True,
            text=True,
            capture_output=True,
        )
        print(f"\n--- {implementation} ---\n{completed.stdout}", end="")
        results.append(parse_output(implementation, completed.stdout))
    return results


@app.local_entrypoint()
def main(m: int = 1024, k: int = 1024, n: int = 1024, repeats: int = 30) -> None:
    results = run_benchmarks.remote(m, k, n, repeats)
    fieldnames = [
        "implementation", "gpu", "m", "k", "n", "repeats",
        "average_ms", "tflops", "c00",
    ]
    with RESULTS_FILE.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            summary = {key: value for key, value in result.items() if key != "iteration_times_ms"}
            writer.writerow({**summary, "m": m, "k": k, "n": n, "repeats": repeats})

    with TIMELINE_FILE.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=["iteration", "coda_fused_ms", "no_coda_separate_ms"],
        )
        writer.writeheader()
        for index, (coda_ms, no_coda_ms) in enumerate(
            zip(results[0]["iteration_times_ms"], results[1]["iteration_times_ms"]),
            start=1,
        ):
            writer.writerow({
                "iteration": index,
                "coda_fused_ms": coda_ms,
                "no_coda_separate_ms": no_coda_ms,
            })
    print(f"Results saved to: {RESULTS_FILE}")
    print(f"Timeline saved to: {TIMELINE_FILE}")
