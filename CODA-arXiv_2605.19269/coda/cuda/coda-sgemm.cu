// coda_style_sgemm.cu
// FP32 SGEMM with a fused epilogue:
//
// C = ReLU(A * B + bias)
//
// A:    M x K
// B:    K x N
// bias: N       (one bias value per output column)
// C:    M x N

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <vector>

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t error = (call);                                          \
        if (error != cudaSuccess) {                                          \
            std::fprintf(stderr, "CUDA error: %s\n",                         \
                         cudaGetErrorString(error));                         \
            std::exit(EXIT_FAILURE);                                         \
        }                                                                    \
    } while (0)

constexpr int TILE = 16;

// Runs once to create deterministic input values directly on the GPU.
__global__ void initialize_matrix(float* matrix, int size) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < size) {
        matrix[index] = static_cast<float>(
            static_cast<int>(index % 17) - 8
        ) / 8.0f;
    }
}

__global__ void initialize_bias(float* bias, int n) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < n) {
        bias[col] = 0.1f * static_cast<float>(
            static_cast<int>(col % 5) - 2
        );
    }
}

// CODA-style idea:
// 1. GEMM mainloop accumulates an output value in a register.
// 2. Epilogue applies bias + ReLU before global-memory store.
__global__ void coda_style_sgemm_kernel(
    const float* A,
    const float* B,
    const float* bias,
    float* C,
    int M,
    int K,
    int N
) {
    // Shared-memory tiles loaded cooperatively by this thread block.
    __shared__ float tile_A[TILE][TILE];
    __shared__ float tile_B[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    // This stays in the thread's register during the whole GEMM.
    float accumulator = 0.0f;

    int num_k_tiles = (K + TILE - 1) / TILE;

    for (int tile = 0; tile < num_k_tiles; ++tile) {
        int a_col = tile * TILE + threadIdx.x;
        int b_row = tile * TILE + threadIdx.y;

        // Cooperatively load one A tile and one B tile from VRAM to shared memory.
        tile_A[threadIdx.y][threadIdx.x] =
            (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;

        tile_B[threadIdx.y][threadIdx.x] =
            (b_row < K && col < N) ? B[b_row * N + col] : 0.0f;

        __syncthreads();

        // GEMM mainloop: reuse shared-memory values.
        for (int k = 0; k < TILE; ++k) {
            accumulator +=
                tile_A[threadIdx.y][k] *
                tile_B[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        // ----- Programmable GEMM epilogue -----
        // Still in a register: no intermediate C = A*B write.
        float output = accumulator + bias[col];
        output = fmaxf(output, 0.0f);  // ReLU

        // One final global-memory write.
        C[row * N + col] = output;
    }
}

int main(int argc, char** argv) {
    int M = argc > 1 ? std::atoi(argv[1]) : 1024;
    int K = argc > 2 ? std::atoi(argv[2]) : 1024;
    int N = argc > 3 ? std::atoi(argv[3]) : 1024;
    int repeats = argc > 4 ? std::atoi(argv[4]) : 30;

    if (M <= 0 || K <= 0 || N <= 0 || repeats <= 0) {
        std::fprintf(stderr, "Usage: ./coda_sgemm [M K N repeats]\n");
        return EXIT_FAILURE;
    }

    float *d_A, *d_B, *d_bias, *d_C;

    CUDA_CHECK(cudaMalloc(&d_A, M * K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B, K * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_bias, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C, M * N * sizeof(float)));

    int threads = 256;

    initialize_matrix<<<(M * K + threads - 1) / threads, threads>>>(
        d_A, M * K
    );
    initialize_matrix<<<(K * N + threads - 1) / threads, threads>>>(
        d_B, K * N
    );
    initialize_bias<<<(N + threads - 1) / threads, threads>>>(d_bias, N);

    dim3 block(TILE, TILE);
    dim3 grid(
        (N + TILE - 1) / TILE,
        (M + TILE - 1) / TILE
    );

    // A longer untimed buffer lets GPU clocks and caches stabilize first.
    for (int i = 0; i < 100; ++i) {
        coda_style_sgemm_kernel<<<grid, block>>>(
            d_A, d_B, d_bias, d_C, M, K, N
        );
    }

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<cudaEvent_t> iteration_events(repeats + 1);
    for (cudaEvent_t& event : iteration_events) {
        CUDA_CHECK(cudaEventCreate(&event));
    }
    CUDA_CHECK(cudaEventRecord(iteration_events[0]));

    for (int i = 0; i < repeats; ++i) {
        coda_style_sgemm_kernel<<<grid, block>>>(
            d_A, d_B, d_bias, d_C, M, K, N
        );
        CUDA_CHECK(cudaEventRecord(iteration_events[i + 1]));
    }

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventSynchronize(iteration_events.back()));

    float total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(
        &total_ms, iteration_events.front(), iteration_events.back()
    ));
    float average_ms = total_ms / repeats;
    double tflops =
        (2.0 * static_cast<double>(M) * K * N) /
        (average_ms * 1e-3) /
        1e12;

    float result;
    CUDA_CHECK(cudaMemcpy(
        &result, d_C, sizeof(float), cudaMemcpyDeviceToHost
    ));

    cudaDeviceProp device;
    CUDA_CHECK(cudaGetDeviceProperties(&device, 0));

    std::printf("GPU: %s\n", device.name);
    std::printf("Operation: ReLU(A x B + bias), fused epilogue\n");
    std::printf("A: (%d, %d), B: (%d, %d), C: (%d, %d)\n",
                M, K, K, N, M, N);
    std::printf("Average full-layer time: %.3f ms\n", average_ms);
    std::printf("GEMM throughput: %.3f TFLOP/s\n", tflops);
    std::printf("C[0, 0]: %.6f\n", result);
    std::printf("Iteration times (ms):");
    for (int i = 0; i < repeats; ++i) {
        float iteration_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(
            &iteration_ms, iteration_events[i], iteration_events[i + 1]
        ));
        std::printf("%s%.6f", i == 0 ? " " : ",", iteration_ms);
    }
    std::printf("\n");

    for (cudaEvent_t event : iteration_events) {
        CUDA_CHECK(cudaEventDestroy(event));
    }
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_bias));
    CUDA_CHECK(cudaFree(d_C));

    return 0;
}
