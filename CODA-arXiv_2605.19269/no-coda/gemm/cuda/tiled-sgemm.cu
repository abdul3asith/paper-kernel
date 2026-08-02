// non_coda_sgemm_bias_relu.cu
//
// Standard non-fused neural-network layer:
//
// 1. C = A × B          (tiled SGEMM)
// 2. C = C + bias       (separate kernel)
// 3. C = ReLU(C)        (separate kernel)
//
// This intentionally writes C to global memory after GEMM, then reads/writes it
// again for bias and ReLU. That is the non-CODA baseline.

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <vector>

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t error = (call);                                             \
        if (error != cudaSuccess) {                                             \
            std::fprintf(stderr, "CUDA error at %s:%d: %s\n",                  \
                         __FILE__, __LINE__, cudaGetErrorString(error));        \
            std::exit(EXIT_FAILURE);                                            \
        }                                                                       \
    } while (0)

constexpr int TILE = 16;

// Creates deterministic matrix values directly on the GPU.
__global__ void initialize_matrix(float* matrix, int size) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < size) {
        matrix[index] =
            static_cast<float>(static_cast<int>(index % 17) - 8) / 8.0f;
    }
}

// Creates one bias value for every output column.
__global__ void initialize_bias(float* bias, int N) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < N) {
        bias[col] =
            0.1f * static_cast<float>(static_cast<int>(col % 5) - 2);
    }
}

// Kernel 1: tiled matrix multiplication.
// C = A × B
//
// A: M × K
// B: K × N
// C: M × N
__global__ void tiled_sgemm_kernel(
    const float* A,
    const float* B,
    float* C,
    int M,
    int K,
    int N
) {
    __shared__ float tile_A[TILE][TILE];
    __shared__ float tile_B[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    float accumulator = 0.0f;

    int num_k_tiles = (K + TILE - 1) / TILE;

    for (int tile = 0; tile < num_k_tiles; ++tile) {
        int a_col = tile * TILE + threadIdx.x;
        int b_row = tile * TILE + threadIdx.y;

        // Load A tile from global memory to shared memory.
        tile_A[threadIdx.y][threadIdx.x] =
            (row < M && a_col < K)
                ? A[row * K + a_col]
                : 0.0f;

        // Load B tile from global memory to shared memory.
        tile_B[threadIdx.y][threadIdx.x] =
            (b_row < K && col < N)
                ? B[b_row * N + col]
                : 0.0f;

        __syncthreads();

        // Multiply shared-memory tiles.
        for (int k = 0; k < TILE; ++k) {
            accumulator +=
                tile_A[threadIdx.y][k] *
                tile_B[k][threadIdx.x];
        }

        __syncthreads();
    }

    // First global-memory write: raw GEMM result.
    if (row < M && col < N) {
        C[row * N + col] = accumulator;
    }
}

// Kernel 2: add one bias value per output column.
// C[row, col] = C[row, col] + bias[col]
__global__ void add_bias_kernel(
    float* C,
    const float* bias,
    int M,
    int N
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        C[row * N + col] += bias[col];
    }
}

// Kernel 3: ReLU.
// ReLU(x) = max(0, x)
__global__ void relu_kernel(float* C, int size) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < size) {
        C[index] = fmaxf(C[index], 0.0f);
    }
}

int main(int argc, char** argv) {
    int M = argc > 1 ? std::atoi(argv[1]) : 1024;
    int K = argc > 2 ? std::atoi(argv[2]) : 1024;
    int N = argc > 3 ? std::atoi(argv[3]) : 1024;
    int repeats = argc > 4 ? std::atoi(argv[4]) : 30;

    if (M <= 0 || K <= 0 || N <= 0 || repeats <= 0) {
        std::fprintf(
            stderr,
            "Usage: ./non_coda_sgemm_bias_relu [M K N repeats]\n"
        );
        return EXIT_FAILURE;
    }

    size_t elements_A = static_cast<size_t>(M) * K;
    size_t elements_B = static_cast<size_t>(K) * N;
    size_t elements_C = static_cast<size_t>(M) * N;

    size_t bytes_A = elements_A * sizeof(float);
    size_t bytes_B = elements_B * sizeof(float);
    size_t bytes_C = elements_C * sizeof(float);
    size_t bytes_bias = static_cast<size_t>(N) * sizeof(float);

    float* d_A = nullptr;
    float* d_B = nullptr;
    float* d_C = nullptr;
    float* d_bias = nullptr;

    CUDA_CHECK(cudaMalloc(&d_A, bytes_A));
    CUDA_CHECK(cudaMalloc(&d_B, bytes_B));
    CUDA_CHECK(cudaMalloc(&d_C, bytes_C));
    CUDA_CHECK(cudaMalloc(&d_bias, bytes_bias));

    int init_threads = 256;

    initialize_matrix<<<
        (elements_A + init_threads - 1) / init_threads,
        init_threads
    >>>(d_A, static_cast<int>(elements_A));

    initialize_matrix<<<
        (elements_B + init_threads - 1) / init_threads,
        init_threads
    >>>(d_B, static_cast<int>(elements_B));

    initialize_bias<<<
        (N + init_threads - 1) / init_threads,
        init_threads
    >>>(d_bias, N);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    dim3 gemm_block(TILE, TILE);
    dim3 gemm_grid(
        (N + TILE - 1) / TILE,
        (M + TILE - 1) / TILE
    );

    dim3 bias_block(16, 16);
    dim3 bias_grid(
        (N + bias_block.x - 1) / bias_block.x,
        (M + bias_block.y - 1) / bias_block.y
    );

    int relu_threads = 256;
    int relu_blocks =
        static_cast<int>((elements_C + relu_threads - 1) / relu_threads);

    // Warm-up launches.
    // A longer untimed buffer lets GPU clocks and caches stabilize first.
    for (int i = 0; i < 100; ++i) {
        tiled_sgemm_kernel<<<gemm_grid, gemm_block>>>(
            d_A, d_B, d_C, M, K, N
        );

        add_bias_kernel<<<bias_grid, bias_block>>>(
            d_C, d_bias, M, N
        );

        relu_kernel<<<relu_blocks, relu_threads>>>(
            d_C, static_cast<int>(elements_C)
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
        // 1. C = A × B
        tiled_sgemm_kernel<<<gemm_grid, gemm_block>>>(
            d_A, d_B, d_C, M, K, N
        );

        // 2. C = C + bias
        add_bias_kernel<<<bias_grid, bias_block>>>(
            d_C, d_bias, M, N
        );

        // 3. C = ReLU(C)
        relu_kernel<<<relu_blocks, relu_threads>>>(
            d_C, static_cast<int>(elements_C)
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

    // GEMM FLOPs only: M × N outputs, each with K multiply-add operations.
    double tflops =
        (2.0 * static_cast<double>(M) * K * N) /
        (average_ms * 1e-3) /
        1e12;

    float first_output = 0.0f;
    CUDA_CHECK(cudaMemcpy(
        &first_output,
        d_C,
        sizeof(float),
        cudaMemcpyDeviceToHost
    ));

    cudaDeviceProp device;
    CUDA_CHECK(cudaGetDeviceProperties(&device, 0));

    std::printf("GPU: %s\n", device.name);
    std::printf("Operation: ReLU(A x B + bias), non-fused\n");
    std::printf("A: (%d, %d), B: (%d, %d), C: (%d, %d)\n",
                M, K, K, N, M, N);
    std::printf("Average full-layer time: %.3f ms\n", average_ms);
    std::printf("GEMM throughput: %.3f TFLOP/s\n", tflops);
    std::printf("C[0, 0]: %.6f\n", first_output);
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
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaFree(d_bias));

    return EXIT_SUCCESS;
}
