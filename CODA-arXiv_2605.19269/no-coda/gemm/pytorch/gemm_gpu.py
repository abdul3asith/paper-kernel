"""Plain FP32 PyTorch GEMM executed on a Modal NVIDIA GPU.

This is the baseline for later Triton and custom CUDA GEMM kernels.
It performs only C = A @ B: no bias, activation, residual add, or CODA fusion.

Run from your local terminal:
    modal run modal_pytorch_gemm.py

Optionally choose dimensions and number of timed iterations:
    modal run modal_pytorch_gemm.py --m 2048 --k 2048 --n 2048 --repeats 50
"""

import modal

app = modal.App("pytorch-gemm-baseline")

# PyTorch is installed inside Modal's remote container, not on your Mac.
image = modal.Image.debian_slim(python_version="3.11").pip_install("torch")


@app.function(image=image, gpu="T4", timeout=300)
def run_gemm(m: int, k: int, n: int, repeats: int) -> dict[str, float | str]:
    """Create matrices and benchmark C = A @ B entirely on the remote GPU."""
    from time import perf_counter

    import torch

    if min(m, k, n, repeats) <= 0:
        raise ValueError("m, k, n, and repeats must all be positive.")
    if not torch.cuda.is_available():
        raise RuntimeError("Modal did not attach a CUDA GPU to this function.")

    device = torch.device("cuda")
    torch.manual_seed(0)

    # FP32 inputs: this is SGEMM (single-precision GEMM).
    # A is [M, K], B is [K, N], therefore C is [M, N].
    a = torch.randn(m, k, device=device, dtype=torch.float32)
    b = torch.randn(k, n, device=device, dtype=torch.float32)

    # Plain PyTorch GEMM. On CUDA, PyTorch uses its optimized GEMM backend.
    c = a @ b

    # Validate one result against the mathematical definition of matrix multiply.
    expected_c00 = (a[0, :] * b[:, 0]).sum()
    torch.testing.assert_close(c[0, 0], expected_c00, rtol=1e-4, atol=1e-4)

    # CUDA launches are asynchronous, so synchronize around the timed region.
    for _ in range(10):
        _ = a @ b
    torch.cuda.synchronize()

    start = perf_counter()
    for _ in range(repeats):
        c = a @ b
    torch.cuda.synchronize()
    average_ms = (perf_counter() - start) * 1_000 / repeats

    # Each output cell performs K multiply-adds: about 2 * M * K * N FLOPs.
    tflops = (2 * m * k * n) / (average_ms * 1e-3) / 1e12

    return {
        "gpu": torch.cuda.get_device_name(0),
        "shape": f"A=({m}, {k}), B=({k}, {n}), C=({m}, {n})",
        "c00": float(c[0, 0].item()),
        "average_ms": average_ms,
        "tflops": tflops,
    }


@app.local_entrypoint()
def main(m: int = 1024, k: int = 2048, n: int = 512, repeats: int = 100) -> None:
    """Runs locally only as an entrypoint; the GEMM itself runs on Modal."""
    result = run_gemm.remote(m, k, n, repeats)
    print(f"GPU: {result['gpu']}")
    print(f"Shapes: {result['shape']}")
    print(f"C[0, 0]: {result['c00']:.6f} (correctness check passed)")
    print(f"Average GEMM time: {result['average_ms']:.3f} ms")
    print(f"Throughput: {result['tflops']:.2f} TFLOP/s")
