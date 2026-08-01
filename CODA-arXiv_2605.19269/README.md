# Benchmarks

## Pytorch Reference Outputs

device: cpu
A: (1024, 2048), B: (2048, 512), C: (1024, 512)
C[0, 0]: -44.824772 (correctness check passed)
average GEMM time: 1.286 ms
throughput: 1.67 TFLOP/s

GPU: Tesla T4
Shapes: A=(1024, 2048), B=(2048, 512), C=(1024, 512)
C[0, 0]: -8.299975 (correctness check passed)
Average GEMM time: 0.789 ms
Throughput: 2.72 TFLOP/s

T4 GPU is about 1.63× faster than CPU