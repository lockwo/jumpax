import csv
import os
import subprocess
import time


results_with_compile = []
results_no_compile = []

for n_traj in range(1, 102, 10):
    print(f"Running n={n_traj}...")

    start_time = time.time()
    result = subprocess.run(
        ["julia", "julia_benchmark.jl", str(n_traj)], capture_output=True, text=True
    )
    elapsed_with_compile = time.time() - start_time

    times = result.stdout.strip().split(",")
    elapsed_no_compile = float(times[1])

    results_with_compile.append([n_traj, elapsed_with_compile])
    results_no_compile.append([n_traj, elapsed_no_compile])

    print(
        f"n={n_traj}, with compile: {elapsed_with_compile:.4f}s, no compile: {elapsed_no_compile:.4f}s"
    )

os.makedirs("results", exist_ok=True)

with open("results/julia_vilar_with_compile.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["n_trajectories", "time_seconds"])
    writer.writerows(results_with_compile)

with open("results/julia_vilar_no_compile.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["n_trajectories", "time_seconds"])
    writer.writerows(results_no_compile)
