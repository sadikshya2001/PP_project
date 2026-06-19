# GPU-Optimized Matrix Multiplication Using CUDA

## Authors

* Sadikshya Satyal
* Ayodeji Ibrahim
* Muhammad Zahid

## Project Overview

This project investigates the performance benefits of GPU acceleration for matrix multiplication using CUDA. We compare three implementations:

1. CPU Naive Matrix Multiplication
2. Naive CUDA GPU Matrix Multiplication
3. Tiled CUDA GPU Matrix Multiplication using Shared Memory

The goal is to evaluate how GPU parallelism and memory optimization improve execution performance compared to a traditional CPU implementation.

## Project Objectives

* Implement matrix multiplication on CPU and GPU
* Explore CUDA thread-based parallelism
* Optimize GPU performance using shared memory
* Benchmark different implementations
* Analyze performance improvements and scalability

## Technologies Used

* CUDA C/C++
* NVIDIA A100 GPU
* CSC Mahti Supercomputer
* CUDA 11.5
* Slurm Job Scheduler
* Python (for graph generation)
* Matplotlib
* Pandas

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
└── README.md
```

## Experimental Setup

The benchmarks were executed on CSC Mahti using an NVIDIA A100 GPU.

Tested matrix sizes:

* 128 × 128
* 256 × 256
* 512 × 512
* 1024 × 1024
* 2048 × 2048

## Performance Results

| Matrix Size |  CPU (ms) | GPU Naive (ms) | GPU Tiled (ms) |
| ----------- | --------: | -------------: | -------------: |
| 128         |     9.057 |          0.073 |          0.015 |
| 256         |    72.785 |          0.032 |          0.024 |
| 512         |   594.912 |          0.139 |          0.091 |
| 1024        |  4965.791 |          0.899 |          0.604 |
| 2048        | 42802.781 |          7.014 |          4.697 |

Maximum observed tiled GPU speedup:

**9113.49× faster than the CPU implementation**

## Running the Project on Mahti

Load CUDA:

```bash
module load cuda/11.5.0
```

Compile:

```bash
nvcc src/main.cu -o matmul
```

Run with Slurm:

```bash
sbatch run_gpu.sh
```

View results:

```bash
cat results/benchmark_results.csv
```

## Key Findings

* GPU acceleration significantly outperforms CPU execution.
* Shared-memory tiling provides additional performance improvements over the naive GPU approach.
* The tiled CUDA implementation achieved the best overall performance.
* All GPU results were verified against the CPU implementation and produced correct outputs.

## Repository

This repository contains the source code, benchmark data, generated graphs, report, and supporting files required to reproduce the project results.
