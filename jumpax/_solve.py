import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int, Key

from ._aggregators import AbstractAggregator, AbstractHybridAggregator
from ._aggregators.ssa import SSA
from ._custom_types import Args, JumpState, RealScalarLike, SolverState, U
from ._jumps import AbstractJumpProblem, VariableRateJump
from ._save import Save
from ._solution import Solution


class State(eqx.Module):
    """Internal state for the solve loop."""

    # current time and state
    t: Float[Array, ""]
    u: U

    # RNG key for solver stepping
    key: Key[Array, ""]

    # loop index and termination flag
    i: Int[Array, ""]
    done: Bool[Array, ""]

    # saved times and states (capacity = max_steps + 2)
    ts: Float[Array, " t_cap"]
    us: Float[Array, " t_cap *ushape"]

    # reaction counts (one per reaction channel)
    counts: Int[Array, " num_reactions"]

    # jump-specific state
    jump_state: JumpState

    # solver-specific state
    solver_state: SolverState

    # dense output: integration steps between jumps
    dense_ts: Float[Array, "max_jumps max_integration_steps"] | None
    dense_us: Float[Array, "max_jumps max_integration_steps *ushape"] | None


@eqx.filter_jit
def solve(
    jumps: AbstractJumpProblem,
    solver: AbstractAggregator,
    save: Save,
    u0: U,
    *,
    t0: RealScalarLike,
    t1: RealScalarLike,
    key: Key[Array, ""],
    args: Args = None,
    max_steps: int = 4096,
) -> Solution:
    """
    Simulate a jump process from `t0` to `t1`.

    **Arguments:**

    - `jumps`: the jump problem to solve
    - `solver`: the aggregator/solver to use (e.g., [`jumpax.SSA`][])
    - `save`: controls what to save (states, counts, dense output)
    - `u0`: initial state
    - `t0`: start time
    - `t1`: end time
    - `key`: JAX random key
    - `args`: static arguments passed to rate and affect functions
    - `max_steps`: maximum number of solver steps before termination

    **Returns:**

    A [`jumpax.Solution`][] containing saved times, states, and statistics.
    """
    if isinstance(jumps, VariableRateJump) and isinstance(solver, SSA):
        raise ValueError(
            "SSA cannot be used with VariableRateJump. "
            "Use HazardSSA for time-dependent or multi-reaction rates."
        )

    if save.dense:
        solver = eqx.tree_at(lambda s: s.save_steps, solver, True)

    t0 = jnp.asarray(t0, dtype=jnp.result_type(t0, t1))
    t1 = jnp.asarray(t1, dtype=jnp.result_type(t0, t1))

    out_cap = max_steps + 2  # t0 + up to max_steps jumps + t1
    if save.states:
        ts = jnp.full((out_cap,), jnp.inf, dtype=t0.dtype)
        us = jnp.full((out_cap,) + u0.shape, jnp.inf, dtype=u0.dtype)
    else:
        ts = jnp.empty((0,), dtype=t0.dtype)
        us = jnp.empty((0,), dtype=u0.dtype)

    if save.reaction_counts:
        rates_shape = eqx.filter_eval_shape(jumps.rate, t0, u0, args)
        counts = jnp.zeros(rates_shape.shape[0], dtype=jnp.int32)
    else:
        counts = jnp.empty((0,), dtype=jnp.int32)

    if save.dense:
        if not isinstance(solver, AbstractHybridAggregator):
            raise ValueError(
                "Cannot save dense diffrax solution for a non-diffrax aggregator!"
            )
        max_integration_steps = solver.max_steps
        if max_integration_steps is None:
            raise ValueError(
                "Cannot keep track of the solver steps for a "
                "unbounded diffrax integration!"
            )
        dense_ts = jnp.full((out_cap, max_integration_steps), jnp.inf, dtype=t0.dtype)
        dense_us = jnp.full(
            (out_cap, max_integration_steps) + u0.shape, jnp.inf, dtype=u0.dtype
        )
    else:
        dense_ts = None
        dense_us = None

    def write(idx, _t, _u, _ts, _us):
        _ts = _ts.at[idx].set(_t)
        _us = _us.at[idx].set(_u)
        return _ts, _us

    if save.states:
        ts, us = write(0, t0, u0, ts, us)

    def cond_fun(state: State):
        return jnp.logical_and(state.i < out_cap - 1, jnp.logical_not(state.done))

    def body_fun(state: State):
        key, subkey = jax.random.split(state.key)
        u_new, tnext, made, jidx, jump_state_next, solver_state_next, info = (
            solver.step(
                jumps,
                state.t,
                state.u,
                args,
                subkey,
                t1,
                state.jump_state,
                state.solver_state,
            )
        )

        if save.reaction_counts:
            if jnp.ndim(jidx) == 0:

                def _inc(c):
                    return c.at[jidx].add(1)

                do_inc = jnp.logical_and(made, jidx >= 0)
                counts = jax.lax.cond(do_inc, _inc, lambda c: c, state.counts)
            else:
                counts = state.counts + jidx.astype(state.counts.dtype)
        else:
            counts = state.counts

        i_next = state.i + 1
        ts, us = (state.ts, state.us)
        if save.states:
            ts, us = write(i_next, tnext, u_new, ts, us)

        dense_ts = state.dense_ts
        dense_us = state.dense_us
        if (
            save.dense
            and info is not None
            and dense_ts is not None
            and dense_us is not None
        ):
            assert isinstance(info, dict) and "ts" in info and "ys" in info
            step_ts = info["ts"]
            step_ys = info["ys"]
            dense_ts = dense_ts.at[state.i].set(step_ts)
            dense_us = dense_us.at[state.i].set(step_ys)

        done_next = jnp.logical_or(tnext >= t1, jnp.logical_not(made))
        return State(
            t=jnp.asarray(tnext),
            u=u_new,
            key=key,
            i=i_next,
            done=done_next,
            ts=ts,
            us=us,
            counts=counts,
            jump_state=jump_state_next,
            solver_state=solver_state_next,
            dense_ts=dense_ts,
            dense_us=dense_us,
        )

    key_solver, key_affect, key_solver_init = jax.random.split(key, 3)
    init_jump_state = jumps.init(u0, args, key_affect)
    init_solver_state = solver.init(jumps, u0, args, key_solver_init)

    init_state = State(
        t=t0,
        u=u0,
        key=key_solver,
        i=jnp.asarray(0, dtype=jnp.int32),
        done=jnp.asarray(False),
        ts=ts,
        us=us,
        counts=counts,
        jump_state=init_jump_state,
        solver_state=init_solver_state,
        dense_ts=dense_ts,
        dense_us=dense_us,
    )
    final_state = jax.lax.while_loop(cond_fun, body_fun, init_state)
    i_final = final_state.i
    ts = final_state.ts
    us = final_state.us
    counts = final_state.counts

    stats = {
        "num_steps": i_final,
    }
    out_counts = counts if save.reaction_counts else None

    dense_data = None
    if save.dense and final_state.dense_ts is not None:
        dense_data = {
            "dense_ts": final_state.dense_ts,
            "dense_us": final_state.dense_us,
        }

    return Solution(ts=ts, us=us, counts=out_counts, stats=stats, dense=dense_data)
