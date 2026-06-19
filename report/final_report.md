# GPU-Optimized Matrix Multiplication Using CUDA

**Authors:** Sadikshya Satyal, Ayodeji Ibrahim, Muhammad Zahid

## Abstract

This project investigates the performance benefits of GPU acceleration for matrix multiplication using CUDA. A CPU implementation, a naive CUDA implementation, a tiled CUDA implementation using shared memory, and NVIDIA's cuBLAS library were developed and evaluated. Performance was benchmarked on the CSC Mahti supercomputer using an NVIDIA A100 GPU. Results show that GPU acceleration provides substantial performance improvements, with the tiled CUDA implementation achieving a speedup of approximately 11,760× and cuBLAS achieving over 41,000× relative to the CPU baseline for the largest tested matrix size.

## 1. Introduction

Matrix multiplication is a fundamental operation in scientific computing, machine learning, computer graphics, and numerical simulations. Due to the independence of individual output elements, matrix multiplication is highly suitable for parallel execution.

The objective of this project is to investigate GPU acceleration using CUDA and evaluate how memory optimization techniques affect performance. Four implementations were compared:

1. CPU Naive Matrix Multiplication
2. Naive CUDA GPU Matrix Multiplication
3. Tiled CUDA GPU Matrix Multiplication using Shared Memory
4. NVIDIA cuBLAS Matrix Multiplication

The project focuses on performance optimization, benchmarking, correctness verification, and comparison against an industry-standard implementation.

## 2. Background

Given matrices A and B, matrix multiplication computes:

C[i][j] = Σ A[i][k] × B[k][j]

Each output element can be computed independently, making the operation naturally parallel.

The naive GPU implementation assigns one thread per output element. Although this exposes significant parallelism, repeated accesses to global memory limit performance.

The tiled implementation improves performance by loading blocks of matrix data into shared memory. Threads within a block reuse these values, reducing expensive global memory accesses and improving memory locality.

NVIDIA cuBLAS was used as a state-of-the-art reference implementation for comparison.

## 3. Implementation

### 3.1 CPU Naive Implementation

The CPU implementation uses three nested loops to compute the matrix product. This implementation serves as the performance baseline.

### 3.2 Naive CUDA Implementation

The naive CUDA kernel assigns one thread to compute each output element. Each thread computes one row-column dot product independently.

### 3.3 Tiled CUDA Implementation

The tiled CUDA implementation utilizes shared memory.

Each block loads tiles of matrices A and B into shared memory and reuses them during computation. This reduces global memory traffic and improves overall efficiency.

A tile size of 16 × 16 was used.

### 3.4 NVIDIA cuBLAS Implementation

To compare against an optimized production-quality solution, NVIDIA's cuBLAS library was used. Matrix multiplication was performed using the `cublasSgemm()` routine.

cuBLAS serves as an industry-standard baseline and demonstrates the performance achievable through highly optimized GPU libraries.

## 4. Experimental Setup

Experiments were executed on the CSC Mahti supercomputer.

### Hardware and Software

* Platform: CSC Mahti
* GPU: NVIDIA A100
* CUDA Version: 11.5
* Programming Language: CUDA C/C++
* Scheduler: Slurm
* Partition: gputest

### Matrix Sizes

* 128 × 128
* 256 × 256
* 512 × 512
* 1024 × 1024
* 2048 × 2048

The following metrics were recorded:

* CPU execution time
* Naive GPU execution time
* Tiled GPU execution time
* cuBLAS execution time
* Speedup relative to CPU
* Correctness verification

## 5. Results

| Matrix Size N | CPU Time (ms) | GPU Naive (ms) | GPU Tiled (ms) | cuBLAS (ms) |
| ------------: | ------------: | -------------: | -------------: | ----------: |
|           128 |         9.146 |          0.062 |          0.015 |     168.628 |
|           256 |        73.919 |          0.035 |          0.027 |      21.019 |
|           512 |       609.954 |          0.143 |          0.093 |       0.105 |
|          1024 |      4941.685 |          0.940 |          0.605 |       0.277 |
|          2048 |     55221.766 |          7.022 |          4.696 |       1.318 |

All implementations produced correct results when compared against the CPU reference implementation.

### Runtime Comparison

Picture is inside result/runtime_comparison.png

### Speedup Comparison

Picture is inside result/spedup_comparison.png

## 6. Discussion

The results demonstrate the substantial benefits of GPU acceleration for matrix multiplication.

Even the naive CUDA implementation significantly outperformed the CPU baseline. The tiled implementation further improved performance through effective use of shared memory and reduced global memory accesses.

For the largest matrix size (2048 × 2048), the tiled CUDA implementation achieved a speedup of approximately 11,760× relative to the CPU implementation.

The cuBLAS implementation achieved the highest performance, completing the largest benchmark in only 1.318 ms and reaching a speedup of approximately 41,883×.

These results confirm the importance of memory hierarchy optimization and demonstrate how shared memory can substantially improve CUDA kernel performance.

## 7. Conclusion

This project successfully demonstrated the effectiveness of GPU acceleration for matrix multiplication.

A CPU baseline, a naive CUDA implementation, a tiled CUDA implementation, and a cuBLAS implementation were evaluated. The tiled implementation achieved a speedup of approximately 11,760×, while cuBLAS achieved over 41,000× speedup for the largest tested matrix.

The results show that matrix multiplication is highly suitable for GPU execution and that shared-memory optimization is a key factor in achieving high performance.

## 8. Future Work

Future work may include:

* Testing larger matrix sizes
* Implementing OpenMP-based CPU parallelization
* Measuring end-to-end execution times including memory transfers
* Comparing against additional libraries such as CUTLASS and oneMKL
* Exploring different tile sizes
* Profiling using NVIDIA Nsight Compute
