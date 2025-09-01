import jax
import jax.numpy as jnp
from jaxtyping import Array, Key

from .._custom_types import Args, Info, JumpState, RealScalarLike, U
from .base import AbstractAggregator


class SSA(AbstractAggregator[None]):
    r"""Stochastic Simulation Algorithm (Gillespie's direct method).

    Samples the next reaction time $\tau$ from an exponential distribution with
    rate $a_0 = \sum_j a_j(u)$, where $a_j(u)$ is the propensity of reaction $j$.
    The reaction channel is selected proportionally to its rate.

    This is an exact method for simulating continuous-time Markov chains and
    chemical reaction networks.

    ??? cite "Reference"

        ```bibtex
        @article{gillespie1977exact,
            title={Exact stochastic simulation of coupled chemical reactions},
            author={Gillespie, Daniel T},
            journal={The Journal of Physical Chemistry},
            volume={81},
            number={25},
            pages={2340--2361},
            year={1977},
            publisher={ACS Publications}
        }
        ```
    """

    def init(self, jumps, u0: U, args: Args, key: Key[Array, ""]) -> None:
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
        """Sample next reaction time and channel using the direct method."""
        rates = jumps.rate(t, u, args)
        total = jnp.sum(rates)

        def _no_jump(_):
            return (
                u,
                t1,
                jnp.array(False),
                jnp.array(-1),
                jump_state,
                solver_state,
                None,
            )

        def _do_jump(_):
            key1, key2 = jax.random.split(key, 2)
            dt = jax.random.exponential(key1, dtype=total.dtype) / total
            t_prop = t + dt
            tnext = jnp.minimum(t_prop, t1)

            cum = jnp.cumsum(rates)
            r = jax.random.uniform(key2, dtype=total.dtype) * total
            idx = jnp.searchsorted(cum, r, method="scan_unrolled")
            idx = jnp.minimum(idx, cum.shape[0] - 1)

            made = t_prop <= t1

            def _apply():
                affects, jump_new = jumps.affect(tnext, u, args, jump_state)
                u_new = jax.tree.map(lambda x: x[idx], affects)
                return u_new, jump_new

            def _keep():
                return u, jump_state

            u_new, jump_next = jax.lax.cond(made, _apply, _keep)
            jidx = jnp.where(made, idx, -1)
            return u_new, tnext, made, jidx, jump_next, solver_state, None

        return jax.lax.cond(total <= 0, _no_jump, _do_jump, operand=None)
