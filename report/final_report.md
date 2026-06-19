# GPU-Optimized Matrix Multiplication Using CUDA

**Authors:** Sadikshya Satyal, Ayodeji Ibrahim, Muhammad Zahid

## 1. Introduction

Matrix multiplication is one of the most important operations in scientific computing, computer graphics, numerical simulations, and machine learning. Many modern workloads rely heavily on multiplying large matrices, especially in deep learning and linear algebra applications.

However, matrix multiplication is computationally expensive. For two square matrices of size N x N, the standard matrix multiplication algorithm has a time complexity of O(N³). This makes it a good candidate for parallel computing, since each element of the output matrix can be computed independently.

The goal of this project is to investigate how GPU parallelization and memory optimization affect the performance of matrix multiplication. We implemented and compared three versions:

1. CPU naive matrix multiplication
2. Naive CUDA GPU matrix multiplication
3. Optimized CUDA tiled matrix multiplication using shared memory

The project focuses on CUDA programming, benchmarking, correctness checking, and performance analysis.

---

## 2. Background

Given two matrices A and B, the matrix product C is computed as:

C[i][j] = sum of A[i][k] * B[k][j] for all k

Each element C[i][j] can be calculated independently. This independence makes matrix multiplication highly suitable for GPU execution, where thousands of threads can run in parallel.

In the naive GPU version, each CUDA thread computes one output element of matrix C. This already provides significant parallelism. However, the naive approach repeatedly accesses global memory, which is slower than shared memory.

To improve performance, we implemented a tiled version. In this version, each thread block loads a small tile of matrix A and matrix B into shared memory. Threads then reuse these cached values to compute parts of the output matrix. This reduces repeated global memory access and improves performance.

---

## 3. Implementation

### 3.1 CPU Naive Implementation

The CPU implementation uses three nested loops. For each output element, the program computes the dot product of one row of A and one column of B.

This version is simple and acts as the baseline for performance comparison.

### 3.2 Naive CUDA Implementation

The naive CUDA implementation assigns one GPU thread to each output element of the result matrix.

Each thread:
- Computes its row and column index
- Iterates over k
- Computes one value of C[row][col]

This version exposes the parallel nature of matrix multiplication.

### 3.3 Tiled CUDA Implementation

The tiled CUDA implementation uses shared memory. Each CUDA block loads small tiles of matrix A and matrix B into shared memory. Threads inside the block then reuse these values.

This reduces expensive global memory accesses and improves memory locality.

The tile size used in this project was 16 x 16.

---

## 4. Experimental Setup

The experiments were run on the CSC Mahti supercomputer using a GPU node.

Hardware and software setup:

- Platform: CSC Mahti
- GPU: NVIDIA A100
- CUDA version: 11.5
- Programming language: CUDA C/C++
- Job scheduler: Slurm
- GPU partition: gputest

The tested matrix sizes were:

- 128 x 128
- 256 x 256
- 512 x 512
- 1024 x 1024
- 2048 x 2048

For each matrix size, we measured:
- CPU execution time
- Naive GPU kernel execution time
- Tiled GPU kernel execution time
- Speedup relative to CPU
- Correctness compared to CPU result

---

## 5. Results

The benchmark results are shown below.

| Matrix Size N | CPU Time (ms) | GPU Naive Time (ms) | GPU Tiled Time (ms) | Naive Speedup | Tiled Speedup |
|---:|---:|---:|---:|---:|---:|
| 128 | 9.057 | 0.073 | 0.015 | 124.57x | 609.98x |
| 256 | 72.785 | 0.032 | 0.024 | 2302.16x | 3048.97x |
| 512 | 594.912 | 0.139 | 0.091 | 4283.64x | 6532.33x |
| 1024 | 4965.791 | 0.899 | 0.604 | 5525.60x | 8218.46x |
| 2048 | 42802.781 | 7.014 | 4.697 | 6102.33x | 9113.49x |

All GPU results were verified against the CPU implementation, and both GPU versions produced correct outputs for all tested matrix sizes.

The generated graphs are available in:

- `results/graphs/runtime_comparison.png`
- `results/graphs/speedup_comparison.png`

---

## 6. Discussion

The benchmark results show a large performance improvement when using GPU acceleration. Even the naive CUDA implementation significantly outperformed the CPU implementation because the matrix multiplication workload can be divided across many GPU threads.

The tiled GPU implementation achieved the best performance overall. This is because it uses shared memory to reduce repeated global memory accesses. Global memory access is relatively slow, so reusing values from shared memory improves efficiency.

For the largest matrix size, N = 2048, the CPU implementation required 42802.781 ms, while the tiled GPU implementation required only 4.697 ms. This corresponds to a speedup of approximately 9113.49x.

The results also show that the benefit of GPU acceleration becomes more visible as the matrix size increases. Larger matrices provide more work for the GPU, allowing better utilization of parallel hardware.

One limitation of this benchmark is that the GPU timing only measures kernel execution time and does not include all data transfer overheads between CPU and GPU memory. This is acceptable for analyzing kernel performance, but future work could include end-to-end timing for a more complete comparison.

---

## 7. Conclusion

This project demonstrated the performance benefits of GPU acceleration for matrix multiplication.

We implemented a CPU baseline, a naive CUDA implementation, and an optimized tiled CUDA implementation using shared memory. The tiled implementation achieved the best performance and reached a speedup of approximately 9113.49x for a 2048 x 2048 matrix.

The results confirm that matrix multiplication is highly suitable for GPU parallelization and that memory optimization using shared memory can significantly improve performance.

---

## 8. Future Work

Possible future improvements include:

- Testing larger matrix sizes using longer GPU job limits
- Including CPU parallel implementations using OpenMP
- Measuring end-to-end runtime including memory transfers
- Comparing the implementation with cuBLAS
- Experimenting with different tile sizes
- Using profiling tools such as NVIDIA Nsight Compute