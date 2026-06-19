# GPU-Optimized Matrix Multiplication Using CUDA

**Authors:** Sadikshya Satyal, Ayodeji Ibrahim, Muhammad Zahid

---

# 1. Introduction

Matrix multiplication is one of the most important operations in scientific computing, machine learning, computer graphics, and numerical simulations. Because each output element can be computed independently, matrix multiplication is highly suitable for parallel execution on modern GPUs.

The objective of this project is to investigate the performance benefits of GPU acceleration using CUDA. Four implementations were developed and compared:

1. CPU Naive Matrix Multiplication
2. Naive CUDA GPU Matrix Multiplication
3. Tiled CUDA GPU Matrix Multiplication using Shared Memory
4. NVIDIA cuBLAS Matrix Multiplication

The project focuses on CUDA programming, performance optimization, benchmarking, correctness verification, and comparison against a state-of-the-art GPU library.

---

# 2. Background

Given two matrices A and B, the matrix product C is computed as:

C[i][j] = Σ A[i][k] × B[k][j]

Each element C[i][j] can be calculated independently. This independence makes matrix multiplication highly suitable for GPU execution, where thousands of threads can run simultaneously.

In the naive GPU implementation, each CUDA thread computes one output element of matrix C. While this provides significant parallelism, it repeatedly accesses global memory, which is relatively slow.

To improve performance, a tiled implementation using shared memory was developed. In this approach, blocks of matrix data are loaded into shared memory and reused by multiple threads. This reduces global memory traffic and improves memory locality.

To further evaluate performance, the implementation was compared against NVIDIA cuBLAS, a highly optimized GPU linear algebra library widely used in scientific computing and deep learning applications.

---

# 3. Implementation

## 3.1 CPU Naive Implementation

The CPU implementation uses three nested loops. For each output element, the program computes the dot product of one row of matrix A and one column of matrix B.

This implementation serves as the baseline for all performance comparisons.

## 3.2 Naive CUDA Implementation

The naive CUDA implementation assigns one GPU thread to each output element.

Each thread:

* Computes its row index
* Computes its column index
* Iterates over all values of k
* Produces one element of the output matrix

This implementation exposes the natural parallelism of matrix multiplication.

## 3.3 Tiled CUDA Implementation

The tiled CUDA implementation uses shared memory to reduce global memory accesses.

Each CUDA block loads tiles of matrices A and B into shared memory. Threads reuse these values to perform multiplication and accumulation operations.

Benefits of this approach include:

* Reduced global memory access
* Improved memory locality
* Better GPU utilization
* Higher overall performance

A tile size of 16 × 16 was used throughout the experiments.

## 3.4 NVIDIA cuBLAS Implementation

To compare against a state-of-the-art solution, NVIDIA's cuBLAS library was used.

Matrix multiplication was performed using the `cublasSgemm()` routine. cuBLAS is highly optimized for NVIDIA GPUs and represents an industry-standard implementation for dense matrix multiplication.

This comparison allows evaluation of how closely the custom CUDA implementation approaches production-quality GPU performance.

---

# 4. Experimental Setup

Experiments were performed on the CSC Mahti supercomputer.

## Hardware and Software

* Platform: CSC Mahti
* GPU: NVIDIA A100
* CUDA Version: 11.5
* Programming Language: CUDA C/C++
* Scheduler: Slurm
* GPU Partition: gputest

## Matrix Sizes

The following square matrix sizes were tested:

* 128 × 128
* 256 × 256
* 512 × 512
* 1024 × 1024
* 2048 × 2048

For each matrix size, the following metrics were measured:

* CPU execution time
* Naive GPU execution time
* Tiled GPU execution time
* cuBLAS execution time
* Speedup relative to CPU execution
* Correctness verification

---

# 5. Results

The benchmark results are shown below.

| Matrix Size N | CPU Time (ms) | GPU Naive (ms) | GPU Tiled (ms) | cuBLAS (ms) |
| ------------: | ------------: | -------------: | -------------: | ----------: |
|           128 |         9.146 |          0.062 |          0.015 |     168.628 |
|           256 |        73.919 |          0.035 |          0.027 |      21.019 |
|           512 |       609.954 |          0.143 |          0.093 |       0.105 |
|          1024 |      4941.685 |          0.940 |          0.605 |       0.277 |
|          2048 |     55221.766 |          7.022 |          4.696 |       1.318 |

All implementations produced correct results when compared against the CPU reference implementation.

Generated graphs:

* Runtime comparison graph
* Speedup comparison graph

These graphs are available in the repository under:

`results/graphs/`

---

# 6. Discussion

The benchmark results demonstrate the significant benefits of GPU acceleration for matrix multiplication.

Even the naive CUDA implementation substantially outperformed the CPU baseline because the workload can be distributed across thousands of GPU threads.

The tiled CUDA implementation further improved performance by utilizing shared memory. By reducing global memory accesses and increasing data reuse, the tiled implementation achieved significantly better performance than the naive GPU version.

For the largest matrix size (2048 × 2048), the CPU implementation required 55221.766 ms, while the tiled CUDA implementation required only 4.696 ms. This corresponds to a speedup of approximately 11759.88×.

The results also demonstrate that GPU acceleration becomes increasingly beneficial as matrix size grows. Larger matrices provide sufficient computational work to fully utilize the GPU's parallel processing capabilities.

The cuBLAS implementation achieved the highest performance for large matrix sizes. For the largest benchmark, cuBLAS completed the computation in only 1.318 ms, corresponding to a speedup of approximately 41883.41× relative to the CPU baseline.

Although the custom tiled CUDA implementation did not surpass cuBLAS, it demonstrated many of the same optimization principles employed by highly optimized GPU libraries, particularly the effective use of shared memory and reduction of global memory traffic.

One limitation of this study is that only kernel execution time was measured. Future work could include full end-to-end performance measurements that incorporate memory transfer overheads between host and device memory.

---

# 7. Conclusion

This project demonstrated the effectiveness of GPU acceleration for matrix multiplication.

Four implementations were evaluated: a CPU baseline, a naive CUDA implementation, an optimized tiled CUDA implementation using shared memory, and NVIDIA's cuBLAS implementation.

The tiled CUDA implementation achieved a speedup of approximately 11759.88× compared to the CPU baseline, while cuBLAS achieved the highest overall performance with a speedup of approximately 41883.41×.

The results confirm that matrix multiplication is highly suitable for GPU parallelization and that memory optimization through shared memory plays a critical role in achieving high performance.

---

# 8. Future Work

Possible future improvements include:

* Testing larger matrix sizes using longer GPU job allocations
* Adding OpenMP-based CPU parallel implementations
* Measuring complete end-to-end execution times including memory transfers
* Comparing against additional high-performance libraries such as CUTLASS and oneMKL
* Investigating different tile sizes and kernel configurations
* Profiling the implementation using NVIDIA Nsight Compute
