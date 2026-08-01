import modal
from pathlib import Path
import sys

app = modal.App("triton-sgemm-naive")
image = modal.Image.debian_slim(python_version="3.11").pip_install("torch", "triton")


@app.function(image=image, gpu="T4", timeout=600)
def run_sgemm(m: int, k: int, n: int, repeats: int) -> dict[str, float | int]:
    import torch
    import triton
    import triton.language as tl

    # A program processes BLOCK_K values of a row/column dot product at a time.
    # It still writes only one scalar C[row, col], which makes this a naive GEMM.
    BLOCK_K = 256

    @triton.jit
    def naive_sgemm_kernel(
        a_ptr,
        b_ptr,
        c_ptr,
        M: tl.constexpr,
        K: tl.constexpr,
        N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        # pid runs from 0 to M*N - 1. Map it to one output cell.
        pid = tl.program_id(axis=0)
        row = pid // N
        col = pid % N

        accumulator = 0.0
        for k_start in range(0, K, BLOCK_K):
            k_offsets = k_start + tl.arange(0, BLOCK_K)

            # A is row-major [M, K], B is row-major [K, N].
            a = tl.load(a_ptr + row * K + k_offsets, mask=k_offsets < K, other=0.0)
            b = tl.load(b_ptr + k_offsets * N + col, mask=k_offsets < K, other=0.0)
            accumulator += tl.sum(a * b, axis=0)

        tl.store(c_ptr + row * N + col, accumulator)

    if min(m, k, n, repeats) <= 0:
        raise ValueError("m, k, n, and repeats must all be positive")

    torch.manual_seed(7)
    a = torch.randn((m, k), device="cuda", dtype=torch.float32)
    b = torch.randn((k, n), device="cuda", dtype=torch.float32)
    c = torch.empty((m, n), device="cuda", dtype=torch.float32)

    grid = (m * n,)
    launch_args = (a, b, c, m, k, n, BLOCK_K)

    # Check against PyTorch/cuBLAS on the same input matrices.
    naive_sgemm_kernel[grid](*launch_args)
    torch.cuda.synchronize()
    reference = a @ b
    torch.testing.assert_close(c, reference, rtol=1e-3, atol=1e-3)

    # Warm-up avoids timing first-use compilation and initialization.
    for _ in range(10):
        naive_sgemm_kernel[grid](*launch_args)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        naive_sgemm_kernel[grid](*launch_args)
    end.record()
    end.synchronize()

    average_ms = start.elapsed_time(end) / repeats
    tflops = (2 * m * k * n) / (average_ms * 1e-3) / 1e12
    gpu_name = torch.cuda.get_device_name(0)

    print(f"GPU: {gpu_name}")
    print(f"Triton: {triton.__version__}")
    print(f"A: ({m}, {k}), B: ({k}, {n}), C: ({m}, {n})")
    print(f"C[0, 0]: {c[0, 0].item():.6f} (correctness check passed)")
    print(f"Average naive Triton SGEMM time: {average_ms:.3f} ms")
    print(f"Throughput: {tflops:.3f} TFLOP/s")

    return {
        "gpu": gpu_name,
        "c00": float(c[0, 0].item()),
        "average_ms": average_ms,
        "tflops": tflops,
    }


@app.local_entrypoint()
def main(m: int = 1024, k: int = 1024, n: int = 1024, repeats: int = 30):
    project_root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(project_root))
    from benchmark_results import append_result

    result = run_sgemm.remote(m, k, n, repeats)
    results_path = append_result(
        implementation="triton-naive",
        device=result["gpu"],
        m=m,
        k=k,
        n=n,
        repeats=repeats,
        average_ms=result["average_ms"],
        tflops=result["tflops"],
        c00=result["c00"],
    )
    print(f"Result saved to: {results_path}")
