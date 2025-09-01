import jax
import jax.numpy as jnp
from jaxtyping import Array, Key

from .._custom_types import Args, Info, JumpState, RealScalarLike, U
from .base import AbstractAggregator


class SimpleTauLeaping(AbstractAggregator[None]):
    r"""Simple $\tau$-leaping aggregator with fixed step size.

    At each step, draws reaction counts from Poisson distributions:

    $$K_j \sim \text{Poisson}(a_j(u) \cdot \tau)$$

    where $a_j(u)$ is the propensity of reaction $j$ and $\tau$ is the step size.
    The state is then updated by applying all reactions simultaneously.

    This is an approximate method that trades exactness for computational efficiency,
    particularly useful for systems with many reactions per unit time.

    ??? cite "Reference"

        ```bibtex
        @article{gillespie2001approximate,
            title={Approximate accelerated stochastic simulation of chemically
                   reacting systems},
            author={Gillespie, Daniel T},
            journal={The Journal of Chemical Physics},
            volume={115},
            number={4},
            pages={1716--1733},
            year={2001},
            publisher={AIP Publishing}
        }
        ```
    """

    dt: RealScalarLike

    def init(self, jumps, u0: U, args: Args, key: Key[Array, ""]) -> None:
        """No solver state needed for tau-leaping."""
        return None

    def step(
        self,
        jumps,
        t: RealScalarLike,
        u: U,
        args: Args,
        key: Key[Array, ""],
        t1: RealScalarLike,
        jump_state: JumpState,
        solver_state: None,
    ) -> tuple[U, jnp.ndarray, jnp.ndarray, jnp.ndarray, JumpState, None, Info]:
        """Draw Poisson counts and apply the leap update."""
        t = jnp.asarray(t)
        t1 = jnp.asarray(t1, dtype=t.dtype)
        dt_step = jnp.minimum(jnp.asarray(self.dt, dtype=t.dtype), t1 - t)
        rates = jumps.rate(t, u, args)
        # Poisson draws per reaction channel for this leap
        counts = jax.random.poisson(key, rates * dt_step, shape=rates.shape)
        # Linear update using leap delta
        delta = jumps.leap_delta(t, u, args)
        u_arr = u
        u_new = u_arr + counts.astype(u_arr.dtype) @ delta
        tnext = t + dt_step
        made = jnp.array(True)
        # Return per-reaction counts so the solver can accumulate reaction_counts
        return u_new, tnext, made, counts, jump_state, solver_state, None
