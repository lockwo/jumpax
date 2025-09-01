from abc import abstractmethod
from typing import Generic, TypeVar

import equinox as eqx
from jaxtyping import Array, Bool, Int, Key

from .._custom_meta import AbstractStrictModule
from .._custom_types import Args, Info, JumpState, RealScalarLike, SolverState, U


_SolverState = TypeVar("_SolverState", bound=SolverState)


class AbstractAggregator(AbstractStrictModule, Generic[_SolverState]):
    """
    Abstract base class for all aggregators (solvers).

    Aggregators implement the stepping logic for jump processes, determining
    when jumps occur and which reaction(s) fire(s).
    """

    @abstractmethod
    def init(self, jumps, u0: U, args: Args, key: Key[Array, ""]) -> _SolverState:
        """
        Initialize solver-specific state.

        **Arguments:**

        - `jumps`: the jump problem
        - `u0`: initial state
        - `args`: static arguments
        - `key`: random key

        **Returns:**

        Initial solver state for this aggregator.
        """
        raise NotImplementedError

    @abstractmethod
    def step(
        self,
        jumps,
        t: RealScalarLike,
        u: U,
        args: Args,
        key: Key[Array, ""],
        t1: RealScalarLike,
        jump_state: JumpState,
        solver_state: _SolverState,
    ) -> tuple[
        U,
        RealScalarLike,
        Bool[Array, ""],
        Int[Array, ""],
        JumpState,
        _SolverState,
        Info,
    ]:
        """
        Advance one step of the simulation.

        **Arguments:**

        - `jumps`: the jump problem
        - `t`: current time
        - `u`: current state
        - `args`: static arguments
        - `key`: random key
        - `t1`: end time (step will not exceed this)
        - `jump_state`: current jump state
        - `solver_state`: current solver state

        **Returns:**

        A tuple
        `(u_new, t_next, made_jump, jump_index, jump_state_new, solver_state_new, info)`
        , if `made_jump` is False then `u_new == u` and `jump_index == -1`.
        """
        raise NotImplementedError


class AbstractHybridAggregator(AbstractAggregator[_SolverState]):
    """
    Abstract base class for hybrid aggregators that combine diffrax integration with
    jumps.
    """

    max_steps: eqx.AbstractVar[int | None]
