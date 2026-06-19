#!/bin/bash
#SBATCH --job-name=matmul_gpu
#SBATCH --account=project_2019091
#SBATCH --partition=gputest
#SBATCH --time=00:10:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:a100:1
#SBATCH --output=results/matmul_%j.out

module load cuda/11.5.0

nvcc src/main.cu -o matmul -lcublas

./matmul