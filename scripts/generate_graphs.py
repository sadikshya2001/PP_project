import pandas as pd
import matplotlib.pyplot as plt
import os

df = pd.read_csv("results/benchmark_results.csv")

os.makedirs("results/graphs", exist_ok=True)

# Runtime graph
plt.figure()
plt.plot(df["N"], df["CPU_ms"], marker="o", label="CPU Naive")
plt.plot(df["N"], df["GPU_Naive_ms"], marker="o", label="GPU Naive")
plt.plot(df["N"], df["GPU_Tiled_ms"], marker="o", label="GPU Tiled")
plt.xlabel("Matrix Size N")
plt.ylabel("Runtime (ms)")
plt.title("Matrix Multiplication Runtime Comparison")
plt.yscale("log")
plt.legend()
plt.grid(True)
plt.savefig("results/graphs/runtime_comparison.png", dpi=300)
plt.close()

# Speedup graph
plt.figure()
plt.plot(df["N"], df["Naive_Speedup"], marker="o", label="GPU Naive Speedup")
plt.plot(df["N"], df["Tiled_Speedup"], marker="o", label="GPU Tiled Speedup")
plt.xlabel("Matrix Size N")
plt.ylabel("Speedup over CPU")
plt.title("GPU Speedup Compared to CPU")
plt.legend()
plt.grid(True)
plt.savefig("results/graphs/speedup_comparison.png", dpi=300)
plt.close()

print("Graphs saved in results/graphs/")