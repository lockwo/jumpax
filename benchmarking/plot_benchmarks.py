import os

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from matplotlib.lines import Line2D


plt.figure(figsize=(12, 7))

files = [
    ("results/rebop_vilar.csv", "rebop", "C0"),
    ("results/julia_vilar_with_compile.csv", "Julia (with compile)", "C1"),
    ("results/julia_vilar_no_compile.csv", "Julia (no compile)", "C4"),
    ("results/jumpax_vilar_with_compile.csv", "jumpax (with compile)", "C2"),
    ("results/jumpax_vilar_no_compile.csv", "jumpax (no compile)", "C3"),
    ("results/jumpax_vilar_gpu_no_compile.csv", "jumpax GPU (no compile)", "C5"),
]

legend_elements = []

for csv_file, label, color in files:
    if os.path.exists(csv_file):
        data = np.loadtxt(csv_file, delimiter=",", skiprows=1)
        ns = data[:, 0]
        times = data[:, 1]

        res = stats.linregress(ns, times)

        plt.plot(ns, times, "o", color=color)
        plt.plot(
            ns,
            res.intercept + res.slope * ns,
            "--",
            color=color,
        )

        legend_elements.append(
            Line2D(
                [0],
                [0],
                color=color,
                marker="o",
                linestyle="--",
                label=f"{label}: {res.slope:.6f} * N",
            )
        )
    else:
        print(f"Warning: {csv_file} not found")

plt.xlabel("Number of trajectories")
plt.ylabel("Time [s]")
plt.title("Vilar oscillator benchmark comparison")
plt.yscale("log")
plt.grid(True, alpha=0.3)
plt.legend(handles=legend_elements)
os.makedirs("results", exist_ok=True)
plt.savefig("results/vilar_benchmark.png", bbox_inches="tight", dpi=300)
