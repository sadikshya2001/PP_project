# GPU-Optimized Matrix Multiplication

## Overview

This project implements and benchmarks matrix multiplication using CPU and CUDA GPU programming.

We compare three versions:

1. CPU naive matrix multiplication
2. Naive CUDA GPU matrix multiplication
3. Optimized CUDA tiled matrix multiplication using shared memory

## Goal

The goal is to study how GPU parallelism and memory optimization improve performance for matrix multiplication.

## Platform

Benchmarks were executed on CSC Mahti using an NVIDIA A100 GPU.

## Results Summary

The tiled GPU implementation achieved the best performance. For a 1024 x 1024 matrix, it achieved a speedup of approximately 8039x compared to the CPU naive implementation.

## Files

- `src/main.cu` — main CUDA implementation
- `results/benchmark_results.csv` — benchmark results
- `run_gpu.sh` — Mahti Slurm GPU job script
- `report/final_report.md` — final report

## Authors

- Sadikshya Satyal
- Muhammad Zahid
- Ayodeji Ibrahim

## How to Run on Mahti

```bash
module load cuda/11.5.0
sbatch run_gpu.sh

