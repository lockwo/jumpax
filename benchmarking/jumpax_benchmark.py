import csv
import os
import time

import jax
import jax.numpy as jnp
import jumpax as jx


try:
    jax.devices("gpu")
    has_gpu = True
except RuntimeError:
    has_gpu = False

alphaA = 50
alphapA = 500
alphaR = 0.01
alphapR = 50
betaA = 50
betaR = 5
deltaMA = 10
deltaMR = 0.5
deltaA = 1
deltaR = 0.2
gammaA = 1
gammaR = 1
gammaC = 2
thetaA = 50
thetaR = 100

reactant_stoich = jnp.array(
    [
        [1, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0],
    ]
)

net_stoich = jnp.array(
    [
        [-1, 0, 1, 0, 0, 0, -1, 0, 0],
        [1, 0, -1, 0, 0, 0, 1, 0, 0],
        [0, -1, 0, 1, 0, 0, -1, 0, 0],
        [0, 1, 0, -1, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, -1, -1, 1],
        [0, 0, 0, 0, 0, 0, 0, 1, -1],
        [0, 0, 0, 0, -1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, -1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, -1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, -1, 0],
    ]
)

rates = jnp.array(
    [
        gammaA,
        thetaA,
        gammaR,
        thetaR,
        alphaA,
        alphaR,
        alphapA,
        alphapR,
        betaA,
        betaR,
        gammaC,
        deltaA,
        deltaMA,
        deltaMR,
        deltaA,
        deltaR,
    ]
)

maj = jx.MassActionJump(reactant_stoich, net_stoich, rates=rates)

u0 = jnp.array([1, 1, 0, 0, 0, 0, 0, 0, 0])
t0, t1 = 0.0, 200.0

solver = jx.SSA()
save = jx.Save(states=False, reaction_counts=False)

results_with_compile = []
results_no_compile = []
results_gpu_no_compile = []

for n_traj in range(1, 102, 10):
    key = jax.random.key(n_traj * 42)
    keys = jax.random.split(key, n_traj)
    solve_fn = jax.jit(
        jax.vmap(
            lambda k: jx.solve(maj, solver, save, u0, t0=t0, t1=t1, args=None, key=k)
        ),
        device=jax.devices("cpu")[0]
    )

    start_time = time.time()
    _ = jax.block_until_ready(solve_fn(keys))
    elapsed_with_compile = time.time() - start_time

    start_time = time.time()
    _ = jax.block_until_ready(solve_fn(keys))
    elapsed_no_compile = time.time() - start_time

    results_with_compile.append([n_traj, elapsed_with_compile])
    results_no_compile.append([n_traj, elapsed_no_compile])

    if has_gpu:
        gpu_keys = jax.device_put(keys, jax.devices("gpu")[0])
        gpu_solve_fn = jax.jit(
            jax.vmap(
                lambda k: jx.solve(
                    maj, solver, save, u0, t0=t0, t1=t1, args=None, key=k
                )
            ),
            device=jax.devices("gpu")[0],
        )
        _ = jax.block_until_ready(gpu_solve_fn(gpu_keys))

        start_time = time.time()
        _ = jax.block_until_ready(gpu_solve_fn(gpu_keys))
        elapsed_gpu_no_compile = time.time() - start_time
        results_gpu_no_compile.append([n_traj, elapsed_gpu_no_compile])
        print(
            f"n={n_traj}, with compile: {elapsed_with_compile:.4f}s, no compile: {elapsed_no_compile:.4f}s, gpu no compile: {elapsed_gpu_no_compile:.4f}s"
        )
    else:
        print(
            f"n={n_traj}, with compile: {elapsed_with_compile:.4f}s, no compile: {elapsed_no_compile:.4f}s"
        )

os.makedirs("results", exist_ok=True)

with open("results/jumpax_vilar_with_compile.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["n_trajectories", "time_seconds"])
    writer.writerows(results_with_compile)

with open("results/jumpax_vilar_no_compile.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["n_trajectories", "time_seconds"])
    writer.writerows(results_no_compile)

if has_gpu:
    with open("results/jumpax_vilar_gpu_no_compile.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["n_trajectories", "time_seconds"])
        writer.writerows(results_gpu_no_compile)
