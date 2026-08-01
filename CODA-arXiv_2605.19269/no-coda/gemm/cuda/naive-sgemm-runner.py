import modal
from pathlib import Path
import re
import sys


CUDA_SOURCE = Path(__file__).resolve().with_name("naive-sgemm.cu")

app = modal.App("cuda-naive-sgemm")

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-devel-ubuntu22.04",
        add_python="3.11",
    )
    .apt_install("build-essential")
    .add_local_file(
        CUDA_SOURCE,
        "/root/project/naive_sgemm.cu",
        copy=True,
    )
)


@app.function(image=image, gpu="T4", timeout=600)
def run_sgemm(m: int, k: int, n: int, repeats: int) -> str:
    import subprocess

    subprocess.run(
        [
            "nvcc",
            "-O3",
            "-std=c++17",
            "-o",
            "/tmp/naive_sgemm",
            "/root/project/naive_sgemm.cu",
        ],
        check=True,
    )

    result = subprocess.run(
        ["/tmp/naive_sgemm", str(m), str(k), str(n), str(repeats)],
        check=True,
        text=True,
        capture_output=True,
    )

    return result.stdout


@app.local_entrypoint()
def main(
    m: int = 1024,
    k: int = 1024,
    n: int = 1024,
    repeats: int = 30,
) -> None:
    project_root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(project_root))
    from benchmark_results import append_result

    output = run_sgemm.remote(m, k, n, repeats)
    print(output, end="")

    def capture(pattern: str) -> str:
        match = re.search(pattern, output)
        if match is None:
            raise RuntimeError(f"Could not parse CUDA benchmark output: {pattern}")
        return match.group(1)

    results_path = append_result(
        implementation="cuda-naive",
        device=capture(r"GPU: (.+)"),
        m=m,
        k=k,
        n=n,
        repeats=repeats,
        average_ms=float(capture(r"Average naive CUDA SGEMM time: ([0-9.eE+-]+) ms")),
        tflops=float(capture(r"Throughput: ([0-9.eE+-]+) TFLOP/s")),
        c00=float(capture(r"C\[0, 0\]: ([0-9.eE+-]+)")),
    )
    print(f"Result saved to: {results_path}")
