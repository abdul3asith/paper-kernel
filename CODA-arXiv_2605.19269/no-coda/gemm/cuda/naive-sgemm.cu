

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t error = (call);                                        \
        if (error != cudaSuccess) {                                        \
            std::fprintf(stderr, "CUDA error: %s\n",                       \
                         cudaGetErrorString(error));                       \
            std::exit(EXIT_FAILURE);                                       \
        }                                                                  \
    } while (0)

__global__ void initialize_matrix(float* matrix, int size) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < size) {
        // Deterministic non-zero FP32 values.
        matrix[index] = static_cast<float>((index % 17) - 8) / 8.0f;
    }
}

__global__ void naive_sgemm_kernel(
    const float* A,
    const float* B,
    float* C,
    int M,
    int K,
    int N
) {
    // This thread owns one output position C[row, col].
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;

        // A[row, :] dot B[:, col]
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }

        C[row * N + col] = sum;
    }
}

int main(int argc, char** argv) {
    int M = argc > 1 ? std::atoi(argv[1]) : 1024;
    int K = argc > 2 ? std::atoi(argv[2]) : 1024;
    int N = argc > 3 ? std::atoi(argv[3]) : 1024;
    int repeats = argc > 4 ? std::atoi(argv[4]) : 30;

    if (M <= 0 || K <= 0 || N <= 0 || repeats <= 0) {
        std::fprintf(stderr, "Usage: ./naive_sgemm [M K N repeats]\n");
        return EXIT_FAILURE;
    }

    size_t bytes_A = static_cast<size_t>(M) * K * sizeof(float);
    size_t bytes_B = static_cast<size_t>(K) * N * sizeof(float);
    size_t bytes_C = static_cast<size_t>(M) * N * sizeof(float);

    float* d_A;
    float* d_B;
    float* d_C;

    CUDA_CHECK(cudaMalloc(&d_A, bytes_A));
    CUDA_CHECK(cudaMalloc(&d_B, bytes_B));
    CUDA_CHECK(cudaMalloc(&d_C, bytes_C));

    // Initialize A and B entirely on the GPU.
    int init_threads = 256;

    int init_blocks_A = (M * K + init_threads - 1) / init_threads;
    int init_blocks_B = (K * N + init_threads - 1) / init_threads;

    initialize_matrix<<<init_blocks_A, init_threads>>>(d_A, M * K);
    initialize_matrix<<<init_blocks_B, init_threads>>>(d_B, K * N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // 256 threads per block.
    dim3 block(16, 16);

    // One thread per C[row, col].
    dim3 grid(
        (N + block.x - 1) / block.x,
        (M + block.y - 1) / block.y
    );

    // Warm up GPU/kernel.
    for (int i = 0; i < 10; ++i) {
        naive_sgemm_kernel<<<grid, block>>>(d_A, d_B, d_C, M, K, N);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start;
    cudaEvent_t stop;

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));

    for (int i = 0; i < repeats; ++i) {
        naive_sgemm_kernel<<<grid, block>>>(d_A, d_B, d_C, M, K, N);
    }

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));

    float average_ms = total_ms / repeats;

    // Each output value performs K multiplications + K additions.
    double tflops =
        (2.0 * M * K * N) /
        (average_ms * 1e-3) /
        1e12;

    // Copy just one result back so we can display it.
    float first_output;
    CUDA_CHECK(cudaMemcpy(
        &first_output,
        d_C,
        sizeof(float),
        cudaMemcpyDeviceToHost
    ));

    cudaDeviceProp device;
    CUDA_CHECK(cudaGetDeviceProperties(&device, 0));

    std::printf("GPU: %s\n", device.name);
    std::printf("A: (%d, %d), B: (%d, %d), C: (%d, %d)\n",
                M, K, K, N, M, N);
    std::printf("Block: (%u, %u), Grid: (%u, %u)\n",
                block.x, block.y, grid.x, grid.y);
    std::printf("C[0, 0]: %.6f\n", first_output);
    std::printf("Average naive CUDA SGEMM time: %.3f ms\n", average_ms);
    std::printf("Throughput: %.3f TFLOP/s\n", tflops);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    return EXIT_SUCCESS;
}