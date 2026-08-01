# Benchmarks

Each PyTorch, Triton, and CUDA SGEMM runner appends its measurements to
`Results.csv` in this directory. The CSV uses one common schema so results from
all implementations can be compared or loaded into a dataframe directly.

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


## CUDA GPU Outputs

GPU: Tesla T4
A: (1024, 1024), B: (1024, 1024), C: (1024, 1024)
Block: (16, 16), Grid: (64, 64)
C[0, 0]: 64.875000
Average naive CUDA SGEMM time: 6.654 ms
Throughput: 0.323 TFLOP/s


## Triton GPU Outputs

GPU: Tesla T4
Triton: 3.7.1
A: (1024, 1024), B: (1024, 1024), C: (1024, 1024)
C[0, 0]: 96.995247 (correctness check passed)
Average naive Triton SGEMM time: 52.563 ms
Throughput: 0.041 TFLOP/s
