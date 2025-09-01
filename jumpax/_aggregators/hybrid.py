import diffrax as dfx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Key

from .._custom_types import Args, Info, JumpState, RealScalarLike, U
from .base import AbstractHybridAggregator


class HybridSSA(AbstractHybridAggregator[None]):
    """
    Hybrid SSA + diffrax ODE integration for CRJ/MAJ with piecewise-constant rates.

    Assumptions:

    - Jump rates do not depend on ODE-evolved components (constant between jumps).
    - `ode_fn(t, y, args)` returns dy with same shape as y.
    - Only ODE is integrated between jumps; at a jump time we apply the discrete affect.
    """

    term: dfx.ODETerm
    solver: dfx.AbstractSolver
    dt0: RealScalarLike | None
    controller: dfx.AbstractStepSizeController
    ode_args: Args
    save_steps: bool
    max_steps: int | None

    def __init__(
        self,
        ode_fn,
        *,
        solver: dfx.AbstractSolver,
        dt0: RealScalarLike | None = None,
        stepsize_controller: dfx.AbstractStepSizeController | None = None,
        ode_args: Args | None = None,
        max_steps: int | None = None,
    ):
        """
        **Arguments:**

        - `ode_fn`: ODE drift function `f(t, u, args) -> du/dt`
        - `solver`: diffrax solver (e.g., `diffrax.Tsit5()`)
        - `dt0`: initial step size for diffrax
        - `stepsize_controller`: diffrax step size controller
        - `ode_args`: separate args for the ODE
        - `max_steps`: maximum diffrax steps per integration interval
        """
        self.term = dfx.ODETerm(ode_fn)
        self.solver = solver
        self.dt0 = dt0
        self.controller = (
            stepsize_controller
            if stepsize_controller is not None
            else dfx.ConstantStepSize()
        )
        self.ode_args = ode_args
        self.save_steps = False
        if max_steps is not None:
            if max_steps <= 0:
                raise ValueError("max_steps must be positive when provided.")
            self.max_steps = int(max_steps)
        else:
            self.max_steps = None

    def _integrate(self, t0, t1, y0, args):
        """Integrate the ODE from t0 to t1."""
        max_steps = self.max_steps
        saveat = dfx.SaveAt(t1=True, steps=self.save_steps)
        sol = dfx.diffeqsolve(
            self.term,
            self.solver,
            t0=t0,
            t1=t1,
            dt0=self.dt0,
            y0=y0,
            args=self.ode_args if self.ode_args is not None else args,
            saveat=saveat,
            stepsize_controller=self.controller,
            max_steps=max_steps,
        )
        # Pick the last finite step
        assert sol.ts is not None and sol.ys is not None
        if self.save_steps:
            valid = jnp.isfinite(sol.ts)
            last = jnp.maximum(jnp.sum(valid) - 1, 0)
            y_end = sol.ys[last]
        else:
            y_end = jnp.squeeze(sol.ys, axis=0)
        info: Info | None = None
        if self.save_steps:
            info = {"ts": sol.ts, "ys": sol.ys}
        return y_end, info

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
        """Sample next reaction, integrate ODE to that time, then apply the jump."""
        rates = jumps.rate(t, u, args)
        total = jnp.sum(rates)

        def _no_jump(_):
            u_flow, info = self._integrate(t, t1, u, args)
            return (
                u_flow,
                t1,
                jnp.array(False),
                jnp.array(-1),
                jump_state,
                solver_state,
                info,
            )

        def _do_jump(_):
            key1, key2 = jax.random.split(key, 2)
            dt = jax.random.exponential(key1, dtype=total.dtype) / total
            t_prop = t + dt
            tnext = jnp.minimum(t_prop, t1)

            u_flow, info = self._integrate(t, tnext, u, args)

            cum = jnp.cumsum(rates)
            r = jax.random.uniform(key2, dtype=total.dtype) * total
            idx = jnp.searchsorted(cum, r, method="scan_unrolled")
            idx = jnp.minimum(idx, cum.shape[0] - 1)

            made = t_prop <= t1

            def _apply():
                affects, jump_new = jumps.affect(tnext, u_flow, args, jump_state)
                u_new = jax.tree.map(lambda x: x[idx], affects)
                return u_new, jump_new

            def _keep():
                return u_flow, jump_state

            u_new, jump_next = jax.lax.cond(made, _apply, _keep)
            jidx = jnp.where(made, idx, -1)
            return u_new, tnext, made, jidx, jump_next, solver_state, info

        return jax.lax.cond(total <= 0, _no_jump, _do_jump, operand=None)
