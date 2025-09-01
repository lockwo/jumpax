from collections.abc import Callable

import diffrax as dfx
import jax
import jax.numpy as jnp
import optimistix as optx
from jaxtyping import Array, Float, Key

from .._custom_types import Args, Info, JumpState, RealScalarLike, U
from .base import AbstractHybridAggregator


class HazardSSA(AbstractHybridAggregator[Float[Array, " R"]]):
    r"""
    Hazard (time-change) SSA that supports ODE and SDEs.

    Uses the random time change representation: for each reaction channel $j$,
    maintain an integrated hazard $A_j(t) = \int_0^t \lambda_j(s, u(s)) ds$ and
    fire when $A_j$ exceeds an exponential threshold $z_j$.

    This allows coupling jump processes with continuous dynamics via diffrax.
    """

    ode_fn: Callable | None
    diffusion_fn: Callable | None
    solver: dfx.AbstractSolver
    dt0: RealScalarLike | None
    controller: dfx.AbstractStepSizeController
    ode_args: Args
    brownian_tol: RealScalarLike
    save_steps: bool
    max_steps: int | None

    def __init__(
        self,
        drift_fn: Callable | None,
        *,
        solver: dfx.AbstractSolver,
        dt0: RealScalarLike | None = None,
        stepsize_controller: dfx.AbstractStepSizeController | None = None,
        diffusion_fn: Callable | None = None,
        brownian_tol: RealScalarLike = 1e-3,
        ode_args: Args | None = None,
        max_steps: int | None = None,
    ):
        """
        **Arguments:**

        - `drift_fn`: drift function `f(t, u, args) -> du/dt`
        - `solver`: diffrax solver (e.g., `diffrax.Tsit5()`)
        - `dt0`: initial step size for diffrax
        - `stepsize_controller`: diffrax step size controller
        - `diffusion_fn`: SDE diffusion function `g(t, u, args)`, or None for ODE
        - `brownian_tol`: tolerance for virtual Brownian tree (SDE only)
        - `ode_args`: separate args for the ODE/SDE (defaults to jump args)
        - `max_steps`: maximum diffrax steps per integration interval
        """
        self.ode_fn = drift_fn
        self.diffusion_fn = diffusion_fn
        self.solver = solver
        self.dt0 = dt0
        self.controller = (
            stepsize_controller
            if stepsize_controller is not None
            else dfx.ConstantStepSize()
        )
        self.ode_args = ode_args
        self.brownian_tol = brownian_tol
        self.save_steps = False
        if max_steps is not None:
            if max_steps <= 0:
                raise ValueError("max_steps must be positive when provided.")
            self.max_steps = max_steps
        else:
            self.max_steps = None

    def init(self, jumps, u0: U, args: Args, key: Key[Array, ""]) -> Float[Array, " R"]:
        """Initialize per-reaction exponential thresholds (residual clocks)."""
        t_probe = jnp.asarray(0.0, dtype=jnp.result_type(u0, 1.0))
        rates0 = jumps.rate(t_probe, u0, args)
        R = rates0.shape[0]
        z0 = jax.random.exponential(key, shape=(R,), dtype=rates0.dtype)
        return z0

    def step(
        self,
        jumps,
        t: RealScalarLike,
        u: U,
        args: Args,
        key: Key[Array, ""],
        t1: RealScalarLike,
        jump_state: JumpState,
        solver_state: Float[Array, " R"],
    ) -> tuple[
        U, jnp.ndarray, jnp.ndarray, jnp.ndarray, JumpState, Float[Array, " R"], Info
    ]:
        """Integrate the augmented system until a hazard threshold is crossed."""
        t = jnp.asarray(t)
        t1 = jnp.asarray(t1, dtype=t.dtype)

        rates0 = jumps.rate(t, u, args)
        R = rates0.shape[0]
        u_arr = jnp.asarray(u)
        S = u_arr.shape[0]

        # Carry per-reaction residual thresholds between steps.
        # Solver state contains the hazard thresholds z
        z_prev = solver_state
        key_flow, key_z, key_new = jax.random.split(key, 3)
        z = jax.lax.cond(
            z_prev is None,
            lambda _: jax.random.exponential(key_z, shape=(R,), dtype=rates0.dtype),
            lambda _: jnp.asarray(z_prev, dtype=rates0.dtype),
            operand=None,
        )

        # Build augmented vector field for [u, A]; dA/dt = rates(t, u)
        def drift(ti, y, vf_args):
            u_part = y[:S]
            if self.ode_fn is None:
                du = jnp.zeros_like(u_part)
            else:
                du = self.ode_fn(ti, u_part, vf_args)
            rates = jumps.rate(ti, u_part, args)
            return jnp.concatenate([du, rates])

        diffusion_fn = self.diffusion_fn
        if diffusion_fn is None:
            terms = dfx.ODETerm(drift)
            brownian = None
        else:

            def diffusion(ti, y, vf_args):
                u_part = y[:S]
                g = jnp.asarray(diffusion_fn(ti, u_part, vf_args))
                if g.ndim == 0:
                    g = jnp.expand_dims(g, (0, 1))
                elif g.ndim == 1:
                    g = jnp.expand_dims(g, axis=1)
                zeros = jnp.zeros((R, g.shape[1]), dtype=g.dtype)
                return jnp.concatenate([g, zeros], axis=0)

            g_shape = jax.eval_shape(diffusion_fn, t, u, args)
            if g_shape.ndim == 0:
                noise_dim = 1
            elif g_shape.ndim == 1:
                noise_dim = 1
            else:
                noise_dim = g_shape.shape[1]

            brownian = dfx.VirtualBrownianTree(
                t0=t,
                t1=t1,
                tol=self.brownian_tol,
                shape=(noise_dim,),
                key=key_flow,
            )
            terms = dfx.MultiTerm(
                dfx.ODETerm(drift),
                dfx.ControlTerm(diffusion, brownian),
            )

        # Event: trigger when any channel reaches its threshold.
        # Use max(A - z): starts negative (all below threshold) and hits 0 when
        # the earliest channel satisfies A_j - z_j = 0.
        def cond_fn(t, y, args, **kwargs):
            A = y[S:]
            return jnp.max(A - z)

        # todo: expose root find to user
        event = dfx.Event(cond_fn, root_finder=optx.Newton(1e-4, 1e-4, optx.rms_norm))

        y0 = jnp.concatenate([u_arr, jnp.zeros((R,), dtype=u_arr.dtype)])
        max_steps = self.max_steps

        sol = dfx.diffeqsolve(
            terms,
            self.solver,
            t0=t,
            t1=t1,
            dt0=self.dt0,
            y0=y0,
            args=self.ode_args if self.ode_args is not None else args,
            saveat=dfx.SaveAt(t1=True, steps=self.save_steps),
            stepsize_controller=self.controller,
            event=event,
            max_steps=max_steps,
        )

        assert sol.ts is not None and sol.ys is not None
        if self.save_steps:
            valid = jnp.isfinite(sol.ts)
            last = jnp.maximum(jnp.sum(valid) - 1, 0)
            y_end = sol.ys[last]
            t_end = sol.ts[last]
        else:
            y_end = sol.ys[0]
            t_end = sol.ts[0]
        u_flow = y_end[:S]
        A_end = y_end[S:]

        # Determine if event happened before t1
        made = t_end < t1

        # Residual thresholds after integrating hazards
        z_after = z - A_end
        # Identify fired channel as the one whose residual hit zero first
        idx = jnp.argmin(z_after)
        # Sample fresh threshold values; we will only use the one at `idx`.
        new_exp_all = jax.random.exponential(key_new, shape=(R,), dtype=rates0.dtype)

        def _apply_jump(_):
            affects, jump_new = jumps.affect(t_end, u_flow, args, jump_state)
            u_new = jax.tree.map(lambda x: x[idx], affects)
            z_next = z_after.at[idx].set(new_exp_all[idx])
            return u_new, jnp.array(idx, dtype=jnp.int32), jump_new, z_next

        def _no_jump(_):
            return u_flow, jnp.array(-1, dtype=jnp.int32), jump_state, z_after

        u_new, jidx, jump_next, solver_next = jax.lax.cond(
            made, _apply_jump, _no_jump, operand=None
        )
        info: Info | None = None
        if self.save_steps:
            info = {"ts": sol.ts, "ys": sol.ys[:, :S]}
        return u_new, t_end, made, jidx, jump_next, solver_next, info
