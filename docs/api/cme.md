# Chemical Master Equation

Exact Chemical Master Equation solver via matrix exponentiation.

For time-independent reaction systems with finite (or finitely truncated) state spaces,
the probability distribution $P(t)$ evolves according to:

$$\frac{dP}{dt} = Q P$$

where $Q$ is the generator (rate) matrix. The exact solution is:

$$P(t) = e^{Qt} P(0)$$

This module provides utilities to:

1. Build the generator matrix $Q$ from a `MassActionJump` problem on a truncated state space
2. Solve the CME exactly using matrix exponentiation

## Functions

::: jumpax.build_generator

---

::: jumpax.solve_cme

---

::: jumpax.marginal_distribution

## Types

::: jumpax.CMEState
    options:
        members: false
