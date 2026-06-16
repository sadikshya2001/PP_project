#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>
#define TILE_SIZE 16
// ---------------- CPU NAIVE MATRIX MULTIPLICATION ----------------
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



// ---------------- GPU NAIVE MATRIX MULTIPLICATION ----------------

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



// ---------------- GPU TILED MATRIX MULTIPLICATION ----------------

__global__ void gpuTiledMatMul(float* A, float* B, float* C, int N) {

    __shared__ float tileA[TILE_SIZE][TILE_SIZE];

    __shared__ float tileB[TILE_SIZE][TILE_SIZE];



    int row = blockIdx.y * TILE_SIZE + threadIdx.y;

    int col = blockIdx.x * TILE_SIZE + threadIdx.x;



    float sum = 0.0f;



    for (int tile = 0; tile < (N + TILE_SIZE - 1) / TILE_SIZE; tile++) {

        int tiledCol = tile * TILE_SIZE + threadIdx.x;

        int tiledRow = tile * TILE_SIZE + threadIdx.y;



        if (row < N && tiledCol < N)

            tileA[threadIdx.y][threadIdx.x] = A[row * N + tiledCol];

        else

            tileA[threadIdx.y][threadIdx.x] = 0.0f;



        if (tiledRow < N && col < N)

            tileB[threadIdx.y][threadIdx.x] = B[tiledRow * N + col];

        else

            tileB[threadIdx.y][threadIdx.x] = 0.0f;



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



// ---------------- VERIFY RESULTS ----------------

bool verify(float* ref, float* test, int N) {

    float eps = 1e-2f;



    for (int i = 0; i < N * N; i++) {

        if (fabs(ref[i] - test[i]) > eps) {

            printf("Mismatch at index %d: CPU = %f, GPU = %f\n", i, ref[i], test[i]);

            return false;

        }

    }



    return true;

}



// ---------------- MAIN BENCHMARK ----------------

int main() {

    int sizes[] = {128, 256, 512, 1024};

    int numSizes = 4;



    FILE* file = fopen("results/benchmark_results.csv", "w");

    fprintf(file, "N,CPU_ms,GPU_Naive_ms,GPU_Tiled_ms,Naive_Speedup,Tiled_Speedup,Naive_Correct,Tiled_Correct\n");



    for (int s = 0; s < numSizes; s++) {

        int N = sizes[s];

        size_t size = N * N * sizeof(float);



        printf("\nRunning N = %d\n", N);



        float* A = (float*)malloc(size);

        float* B = (float*)malloc(size);

        float* C_cpu = (float*)malloc(size);

        float* C_gpu_naive = (float*)malloc(size);

        float* C_gpu_tiled = (float*)malloc(size);



        for (int i = 0; i < N * N; i++) {

            A[i] = 1.0f;

            B[i] = 1.0f;

        }



        // CPU timing

        clock_t cpuStart = clock();

        cpuMatMul(A, B, C_cpu, N);

        clock_t cpuEnd = clock();



        float cpuTime = 1000.0f * (cpuEnd - cpuStart) / CLOCKS_PER_SEC;



        // GPU memory allocation

        float *d_A, *d_B, *d_C_naive, *d_C_tiled;

        cudaMalloc((void**)&d_A, size);

        cudaMalloc((void**)&d_B, size);

        cudaMalloc((void**)&d_C_naive, size);

        cudaMalloc((void**)&d_C_tiled, size);



        cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);

        cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);



        dim3 block(TILE_SIZE, TILE_SIZE);

        dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);



        cudaEvent_t start, stop;

        cudaEventCreate(&start);

        cudaEventCreate(&stop);



        // Naive GPU timing

        cudaEventRecord(start);

        gpuNaiveMatMul<<<grid, block>>>(d_A, d_B, d_C_naive, N);

        cudaEventRecord(stop);

        cudaEventSynchronize(stop);



        float naiveTime;

        cudaEventElapsedTime(&naiveTime, start, stop);



        cudaMemcpy(C_gpu_naive, d_C_naive, size, cudaMemcpyDeviceToHost);



        // Tiled GPU timing

        cudaEventRecord(start);

        gpuTiledMatMul<<<grid, block>>>(d_A, d_B, d_C_tiled, N);

        cudaEventRecord(stop);

        cudaEventSynchronize(stop);



        float tiledTime;

        cudaEventElapsedTime(&tiledTime, start, stop);



        cudaMemcpy(C_gpu_tiled, d_C_tiled, size, cudaMemcpyDeviceToHost);



        bool naiveCorrect = verify(C_cpu, C_gpu_naive, N);

        bool tiledCorrect = verify(C_cpu, C_gpu_tiled, N);



        float naiveSpeedup = cpuTime / naiveTime;

        float tiledSpeedup = cpuTime / tiledTime;



        printf("CPU time:        %.3f ms\n", cpuTime);

        printf("GPU naive time:  %.3f ms\n", naiveTime);

        printf("GPU tiled time:  %.3f ms\n", tiledTime);

        printf("Naive speedup:   %.2fx\n", naiveSpeedup);

        printf("Tiled speedup:   %.2fx\n", tiledSpeedup);

        printf("Naive correct:   %s\n", naiveCorrect ? "YES" : "NO");

        printf("Tiled correct:   %s\n", tiledCorrect ? "YES" : "NO");



        fprintf(file, "%d,%.3f,%.3f,%.3f,%.2f,%.2f,%s,%s\n",

                N,

                cpuTime,

                naiveTime,

                tiledTime,

                naiveSpeedup,

                tiledSpeedup,

                naiveCorrect ? "YES" : "NO",

                tiledCorrect ? "YES" : "NO");



        cudaFree(d_A);

        cudaFree(d_B);

        cudaFree(d_C_naive);

        cudaFree(d_C_tiled);



        free(A);

        free(B);

        free(C_cpu);

        free(C_gpu_naive);

        free(C_gpu_tiled);



        cudaEventDestroy(start);

        cudaEventDestroy(stop);

    }



    fclose(file);



    printf("\nBenchmark complete. Results saved to results/benchmark_results.csv\n");



    return 0;
}