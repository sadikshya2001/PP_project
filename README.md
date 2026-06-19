# GPU-Optimized Matrix Multiplication Using CUDA

## Authors

* Sadikshya Satyal
* Ayodeji Ibrahim
* Muhammad Zahid

## Project Overview

This project investigates the performance benefits of GPU acceleration for matrix multiplication using CUDA. We compare four implementations:

1. CPU Naive Matrix Multiplication
2. Naive CUDA GPU Matrix Multiplication
3. Tiled CUDA GPU Matrix Multiplication using Shared Memory
4. NVIDIA cuBLAS Matrix Multiplication

The goal is to evaluate how GPU parallelism and memory optimization improve execution performance compared to a traditional CPU implementation and compare our custom CUDA kernels against NVIDIA's highly optimized cuBLAS library.

## Project Objectives

* Implement matrix multiplication on CPU and GPU
* Explore CUDA thread-based parallelism
* Optimize GPU performance using shared memory
* Compare custom CUDA kernels with NVIDIA cuBLAS
* Benchmark different implementations
* Analyze performance improvements and scalability

## Technologies Used

* CUDA C/C++
* NVIDIA cuBLAS
* NVIDIA A100 GPU
* CSC Mahti Supercomputer
* CUDA 11.5
* Slurm Job Scheduler
* Python
* Pandas
* Matplotlib

## Project Structure

```text
gpu-matrix-multiplication/
│
├── src/
│   └── main.cu
│
├── results/
│   ├── benchmark_results.csv
│   ├── matmul_6864059.out
│   └── graphs/
│       ├── runtime_comparison.png
│       └── speedup_comparison.png
│
├── report/
│   └── final_report.md
│
├── scripts/
│   └── generate_graphs.py
│
├── run_gpu.sh
│
└── README.md
```

## Experimental Setup

The benchmarks were executed on CSC Mahti using an NVIDIA A100 GPU.

### Tested Matrix Sizes

* 128 × 128
* 256 × 256
* 512 × 512
* 1024 × 1024
* 2048 × 2048

### Implementations Compared

* CPU Naive
* CUDA Naive Kernel
* CUDA Tiled Kernel (Shared Memory)
* NVIDIA cuBLAS SGEMM

## Performance Results

| Matrix Size |  CPU (ms) | GPU Naive (ms) | GPU Tiled (ms) | cuBLAS (ms) |
| ----------- | --------: | -------------: | -------------: | ----------: |
| 128         |     9.146 |          0.062 |          0.015 |     168.628 |
| 256         |    73.919 |          0.035 |          0.027 |      21.019 |
| 512         |   609.954 |          0.143 |          0.093 |       0.105 |
| 1024        |  4941.685 |          0.940 |          0.605 |       0.277 |
| 2048        | 55221.766 |          7.022 |          4.696 |       1.318 |

### Maximum Observed Speedups

| Implementation | Maximum Speedup |
| -------------- | --------------: |
| GPU Naive      |        7864.56× |
| GPU Tiled      |       11759.88× |
| cuBLAS         |       41883.41× |

## Running the Project on Mahti

Load CUDA:

```bash
module load cuda/11.5.0
```

Compile:

```bash
nvcc src/main.cu -o matmul -lcublas
```

Run using Slurm:

```bash
sbatch run_gpu.sh
```

View benchmark results:

```bash
cat results/benchmark_results.csv
```

## Key Findings

* GPU acceleration dramatically outperforms CPU execution.
* Shared-memory tiling significantly improves performance compared to the naive CUDA implementation.
* NVIDIA cuBLAS achieves the highest overall performance.
* The custom tiled implementation demonstrates many of the same optimization principles used in high-performance GPU libraries.
* All implementations were verified against the CPU reference implementation and produced correct results.

## Generated Results

The project includes:

* Benchmark results (`benchmark_results.csv`)
* Runtime comparison graph
* Speedup comparison graph
* Full implementation source code
* Technical report

## Conclusion

This project demonstrates how GPU parallelism and memory optimization can dramatically accelerate matrix multiplication. The custom tiled CUDA implementation achieved over 11,000× speedup compared to the CPU baseline, while NVIDIA cuBLAS achieved over 41,000× speedup for the largest tested matrix size.

The results highlight the importance of shared memory optimization and provide insight into how modern high-performance GPU libraries achieve their efficiency.

## Repository

This repository contains all source code, benchmark data, graphs, report files, and scripts necessary to reproduce the results presented in the project.
