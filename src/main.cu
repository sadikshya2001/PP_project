#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define TILE_SIZE 16

void cpuMatMul(float* A, float* B, float* C, int N) {
    for (int row = 0; row < N; row++) {
        for (int col = 0; col < N; col++) {
            float sum = 0.0f;
            for (int k = 0; k < N; k++) {
                sum += A[row * N + k] * B[k * N + col];
            }
            C[row * N + col] = sum;
        }
    }
}

__global__ void gpuNaiveMatMul(float* A, float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

__global__ void gpuTiledMatMul(float* A, float* B, float* C, int N) {
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;

    for (int tile = 0; tile < (N + TILE_SIZE - 1) / TILE_SIZE; tile++) {
        int tiledCol = tile * TILE_SIZE + threadIdx.x;
        int tiledRow = tile * TILE_SIZE + threadIdx.y;

        tileA[threadIdx.y][threadIdx.x] =
            (row < N && tiledCol < N) ? A[row * N + tiledCol] : 0.0f;

        tileB[threadIdx.y][threadIdx.x] =
            (tiledRow < N && col < N) ? B[tiledRow * N + col] : 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE_SIZE; k++) {
            sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}

bool verify(float* ref, float* test, int N) {
    float eps = 1e-2f;

    for (int i = 0; i < N * N; i++) {
        if (fabs(ref[i] - test[i]) > eps) {
            printf("Mismatch at index %d: CPU = %f, Test = %f\n", i, ref[i], test[i]);
            return false;
        }
    }

    return true;
}

int main() {
    int sizes[] = {128, 256, 512, 1024, 2048};
    int numSizes = sizeof(sizes) / sizeof(sizes[0]);

    FILE* file = fopen("results/benchmark_results.csv", "w");

    fprintf(file,
            "N,CPU_ms,GPU_Naive_ms,GPU_Tiled_ms,cuBLAS_ms,"
            "Naive_Speedup,Tiled_Speedup,cuBLAS_Speedup,"
            "Naive_Correct,Tiled_Correct,cuBLAS_Correct\n");

    for (int s = 0; s < numSizes; s++) {
        int N = sizes[s];
        size_t size = N * N * sizeof(float);

        printf("\nRunning N = %d\n", N);

        float* A = (float*)malloc(size);
        float* B = (float*)malloc(size);
        float* C_cpu = (float*)malloc(size);
        float* C_gpu_naive = (float*)malloc(size);
        float* C_gpu_tiled = (float*)malloc(size);
        float* C_cublas = (float*)malloc(size);

        for (int i = 0; i < N * N; i++) {
            A[i] = 1.0f;
            B[i] = 1.0f;
        }

        clock_t cpuStart = clock();
        cpuMatMul(A, B, C_cpu, N);
        clock_t cpuEnd = clock();

        float cpuTime = 1000.0f * (cpuEnd - cpuStart) / CLOCKS_PER_SEC;

        float *d_A, *d_B, *d_C_naive, *d_C_tiled, *d_C_cublas;

        cudaMalloc((void**)&d_A, size);
        cudaMalloc((void**)&d_B, size);
        cudaMalloc((void**)&d_C_naive, size);
        cudaMalloc((void**)&d_C_tiled, size);
        cudaMalloc((void**)&d_C_cublas, size);

        cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

        dim3 block(TILE_SIZE, TILE_SIZE);
        dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE,
                  (N + TILE_SIZE - 1) / TILE_SIZE);

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start);
        gpuNaiveMatMul<<<grid, block>>>(d_A, d_B, d_C_naive, N);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float naiveTime;
        cudaEventElapsedTime(&naiveTime, start, stop);
        cudaMemcpy(C_gpu_naive, d_C_naive, size, cudaMemcpyDeviceToHost);

        cudaEventRecord(start);
        gpuTiledMatMul<<<grid, block>>>(d_A, d_B, d_C_tiled, N);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float tiledTime;
        cudaEventElapsedTime(&tiledTime, start, stop);
        cudaMemcpy(C_gpu_tiled, d_C_tiled, size, cudaMemcpyDeviceToHost);

        cublasHandle_t handle;
        cublasCreate(&handle);

        float alpha = 1.0f;
        float beta = 0.0f;

        cudaEventRecord(start);

        cublasSgemm(handle,
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    N, N, N,
                    &alpha,
                    d_B, N,
                    d_A, N,
                    &beta,
                    d_C_cublas, N);

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float cublasTime;
        cudaEventElapsedTime(&cublasTime, start, stop);
        cudaMemcpy(C_cublas, d_C_cublas, size, cudaMemcpyDeviceToHost);

        bool naiveCorrect = verify(C_cpu, C_gpu_naive, N);
        bool tiledCorrect = verify(C_cpu, C_gpu_tiled, N);
        bool cublasCorrect = verify(C_cpu, C_cublas, N);

        float naiveSpeedup = cpuTime / naiveTime;
        float tiledSpeedup = cpuTime / tiledTime;
        float cublasSpeedup = cpuTime / cublasTime;

        printf("CPU time:        %.3f ms\n", cpuTime);
        printf("GPU naive time:  %.3f ms\n", naiveTime);
        printf("GPU tiled time:  %.3f ms\n", tiledTime);
        printf("cuBLAS time:     %.3f ms\n", cublasTime);

        printf("Naive speedup:   %.2fx\n", naiveSpeedup);
        printf("Tiled speedup:   %.2fx\n", tiledSpeedup);
        printf("cuBLAS speedup:  %.2fx\n", cublasSpeedup);

        printf("Naive correct:   %s\n", naiveCorrect ? "YES" : "NO");
        printf("Tiled correct:   %s\n", tiledCorrect ? "YES" : "NO");
        printf("cuBLAS correct:  %s\n", cublasCorrect ? "YES" : "NO");

        fprintf(file,
                "%d,%.3f,%.3f,%.3f,%.3f,%.2f,%.2f,%.2f,%s,%s,%s\n",
                N,
                cpuTime,
                naiveTime,
                tiledTime,
                cublasTime,
                naiveSpeedup,
                tiledSpeedup,
                cublasSpeedup,
                naiveCorrect ? "YES" : "NO",
                tiledCorrect ? "YES" : "NO",
                cublasCorrect ? "YES" : "NO");

        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C_naive);
        cudaFree(d_C_tiled);
        cudaFree(d_C_cublas);

        free(A);
        free(B);
        free(C_cpu);
        free(C_gpu_naive);
        free(C_gpu_tiled);
        free(C_cublas);

        cublasDestroy(handle);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    fclose(file);

    printf("\nBenchmark complete. Results saved to results/benchmark_results.csv\n");

    return 0;
}