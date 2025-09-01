# Jumpax

Jumpax is a [JAX](https://github.com/google/jax)-based library providing numerical jump process solvers.

Heavily adapted/inspired by [JumpProcesses.jl](https://github.com/SciML/JumpProcesses.jl) and [diffrax](https://github.com/patrick-kidger/diffrax).

## Installation

```
git clone https://github.com/lockwo/jumpax.git
cd jumpax
pip install -e .
```

Requires Python >= 3.10

## Quick example

Simulate a simple birth-death process:

```python
import jax.numpy as jnp
import jumpax as jpx

# Reactant stoichiometry: birth needs 0, death needs 1
reactants = jnp.array([[0], [1]])
# Net stoichiometry: birth adds 1, death removes 1
net_stoich = jnp.array([[1], [-1]])
# Rate constants
rates = jnp.array([10.0, 0.1])

jumps = jpx.MassActionJump(reactants, net_stoich, rates=rates)
solver = jpx.SSA()
save = jpx.Save(states=True)

u0 = jnp.array([50.0]) # initial population
key = jax.random.key(0)

sol = jpx.solve(jumps, solver, save, u0, t0=0.0, t1=1.0, key=key)
mask = jnp.isfinite(sol.ts)
ts, us = sol.ts[mask], sol.us[mask]

print(f"Final population: {us[-1]}")
```


## Citation

If you found this library useful in academic research, please cite: 

<!-- ```bibtex
@software{lockwood2025jumpax,
  title = {jumpax: Jump Processes in JAX},
  author = {Owen Lockwood},
  url = {https://github.com/lockwo/jumpax},
  doi = {},
}
``` -->

(Also consider starring the project on GitHub.)

## See also: other libraries in the JAX ecosystem

[Awesome JAX](https://github.com/lockwo/awesome-jax): a longer list of other JAX projects.  
