# plain single precison fp32 gemm reference - pytorch

from time import perf_counter

import torch


def synchronize(device: torch.device) -> None:
    """Wait for queued GPU work so timing measures the GEMM itself."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main() -> None:
    # GEMM dimensions: A is [M, K], B is [K, N], so C is [M, N].
    M, K, N = 1024, 2048, 512
    repeats = 100
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)

    # FP32 inputs: this is SGEMM (single-precision GEMM).
    A = torch.randn(M, K, device=device, dtype=torch.float32)
    B = torch.randn(K, N, device=device, dtype=torch.float32)

    # Plain GEMM. On a CUDA device, PyTorch dispatches to its optimized backend.
    C = torch.matmul(A, B)

    # Check one output value against its mathematical definition.
    expected_c00 = (A[0, :] * B[:, 0]).sum()
    torch.testing.assert_close(C[0, 0], expected_c00, rtol=1e-4, atol=1e-4)

    # Warm up before measuring, then report average GEMM time and throughput.
    for _ in range(10):
        torch.matmul(A, B)
    synchronize(device)

    start = perf_counter()
    for _ in range(repeats):
        C = torch.matmul(A, B)
    synchronize(device)
    average_ms = (perf_counter() - start) * 1_000 / repeats

    # One MxK times KxN GEMM performs roughly 2*M*K*N floating-point operations.
    tflops = (2 * M * K * N) / (average_ms * 1e-3) / 1e12

    print(f"device: {device}")
    print(f"A: {tuple(A.shape)}, B: {tuple(B.shape)}, C: {tuple(C.shape)}")
    print(f"C[0, 0]: {C[0, 0].item():.6f} (correctness check passed)")
    print(f"average GEMM time: {average_ms:.3f} ms")
    print(f"throughput: {tflops:.2f} TFLOP/s")


if __name__ == "__main__":
    main()
