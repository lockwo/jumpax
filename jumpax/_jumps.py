from abc import abstractmethod
from collections.abc import Callable
from typing import Generic, TypeAlias, TypeVar

import jax
from jax import numpy as jnp
from jax.scipy.special import gammaln
from jaxtyping import Array, Float, Int, Key

from ._custom_meta import AbstractStrictModule
from ._custom_types import Args, JumpState, Rate, RealScalarLike, U


_Rate = TypeVar("_Rate", bound=Rate)
_JumpState = TypeVar("_JumpState", bound=JumpState)


class AbstractAffect(AbstractStrictModule, Generic[_JumpState]):
    """
    Abstract base class for all affects.
    """

    @abstractmethod
    def init(self, u: U, args: Args, key: Key[Array, ""]) -> _JumpState:
        """
        Initialize the state of the affect.

        **Arguments:**

        - `u`: the current state of the system
        - `args`: any static arguments as passed to [`jumpax.solve`][]
        - `key`: a random key to use

        **Returns:**

        The initialized state.
        """
        raise NotImplementedError

    @abstractmethod
    def __call__(
        self, t: RealScalarLike, u: U, args: Args, jump_state: _JumpState
    ) -> tuple[U, _JumpState]:
        """
        Compute the result of applying the affect.

        **Arguments:**

        - `t`: the time of the affect
        - `u`: the current state
        - `args`: any static arguments
        - `jump_state`: the state of the affect currently

        **Results:**

        The new state and the new affect state.
        """
        raise NotImplementedError


class StatelessAffect(AbstractAffect[None]):
    """
    A convenience wrapper for the common use case of stateless affect functions.
    """

    fn: Callable[[RealScalarLike, U, Args], U]

    def init(self, u: U, args: Args, key: Key[Array, ""]) -> None:
        """Returns None (no state needed)."""
        return None

    def __call__(
        self, t: RealScalarLike, u: U, args: Args, jump_state: None
    ) -> tuple[U, None]:
        """Apply the wrapped function."""
        return self.fn(t, u, args), jump_state


class AbstractJumpProblem(AbstractStrictModule, Generic[_Rate]):
    """
    Abstract class for general jump problems.
    """

    @abstractmethod
    def rate(self, t: RealScalarLike, u: U, args: Args) -> _Rate:
        """
        Computes the rates of each possible reaction.

        **Arguments:**

        - `t`: current time
        - `u`: current state
        - `args`: static arguments

        **Returns:**

        The rates for each possible reaction.
        """
        raise NotImplementedError

    @abstractmethod
    def affect(
        self, t: RealScalarLike, u: U, args: Args, jump_state: JumpState
    ) -> tuple[U, JumpState]:
        """
        Compute the result of applying some affect.

        **Arguments:**

        - `t`: the time of the affect
        - `u`: the current state
        - `args`: any static arguments
        - `jump_state`: the state of the affect currently

        **Results:**

        The new state and the new affect state.
        """
        raise NotImplementedError

    @abstractmethod
    def init(self, u: U, args: Args, key: Key[Array, ""]) -> JumpState:
        """
        Initialize the state of the affect.

        **Arguments:**

        - `u`: the current state of the system
        - `args`: any static arguments as passed to [`jumpax.solve`][]
        - `key`: a random key to use

        **Returns:**

        The initialized state.
        """
        raise NotImplementedError

    @abstractmethod
    def leap_delta(self, t: RealScalarLike, u: U, args: Args):
        """
        Compute the per-reaction net state change.

        **Arguments:**

        - `t`: current time
        - `u`: current state
        - `args`: static arguments

        **Returns:**

        An array of shape `(R, S)` where `R` is the number of reactions and `S` is
        the number of species, representing the net state change for each reaction.
        """
        raise NotImplementedError


_ScalarRate: TypeAlias = Float[Array, ""]


class ConstantRateJump(AbstractJumpProblem[_ScalarRate]):
    r"""
    A jump process with a rate $\lambda(u)$ that depends only on the current state.

    The rate function is evaluated only at jump times, making this suitable for
    processes where the rate does not depend on continuously evolving dynamics
    between jumps.
    """

    rate_fn: Callable[[RealScalarLike, U, Args], _ScalarRate]
    affect_fn: AbstractAffect

    def rate(self, t: RealScalarLike, u: U, args: Args) -> _ScalarRate:
        """Evaluate the rate function."""
        return self.rate_fn(t, u, args)

    def affect(
        self, t: RealScalarLike, u: U, args: Args, jump_state: JumpState
    ) -> tuple[U, JumpState]:
        """Apply the affect and validate the output structure."""
        affect, jump_state = self.affect_fn(t, u, args, jump_state)
        if jax.tree.structure(affect) != jax.tree.structure(u):
            raise ValueError("`affect` and `u` should have the same tree structure.")
        return affect, jump_state

    def init(self, u: U, args: Args, key: Key[Array, ""]) -> JumpState:
        """Initialize the affect state."""
        return self.affect_fn.init(u, args, key)

    def leap_delta(self, t: RealScalarLike, u: U, args: Args) -> Int[Array, "1 S"]:
        """Compute leap delta by differencing the affect output."""
        if isinstance(self.affect, StatelessAffect):
            affects, _ = self.affect(t, u, args, None)
            return affects - u[None, :]
        raise NotImplementedError(
            "ConstantRateJump needs a StatelessAffect to derive leap delta."
        )


def _massaction_comb(
    n: Float[Array, "*shape"],
    k: Int[Array, "*shape"],
    lgamma_k_plus_1: Float[Array, "*shape"],
) -> Float[Array, "*shape"]:
    r"""
    Compute the binomial coefficient $\binom{n}{k}$ using precomputed log-factorials.

    Returns 0 when $k < 0$ or $k > n$.
    """
    n = jnp.asarray(n)
    k = jnp.asarray(k, dtype=n.dtype)
    n_f = n.astype(jnp.float32)
    k_f = k.astype(jnp.float32)
    valid = (k_f >= 0) & (n_f >= k_f)
    logc = gammaln(n_f + 1.0) - lgamma_k_plus_1 - gammaln(n_f - k_f + 1.0)
    c = jnp.exp(logc)
    return jnp.where(valid, c, jnp.zeros_like(c))


class MassActionJump(AbstractJumpProblem[_ScalarRate]):
    r"""
    Array-based mass-action reaction system.

    The propensity for reaction $j$ is given by:

    $$a_j(u) = \kappa_j \prod_i \binom{u_i}{r_{ji}}$$

    where $\kappa_j$ is the rate constant, $u_i$ is the population of species $i$,
    and $r_{ji}$ is the stoichiometric coefficient of species $i$ in reaction $j$.
    The binomial coefficient $\binom{n}{k} = 0$ when $k > n$.
    """

    reactants: Int[Array, "R S"]
    net_stoich: Int[Array, "R S"]
    kappas: Float[Array, " R"]
    _lgamma_rp1: Float[Array, "R S"]

    def __init__(
        self,
        reactants: Int[Array, "R S"],
        net_stoich: Int[Array, "R S"],
        *,
        rates: Float[Array, " R"],
    ):
        r"""
        **Arguments:**

        - `reactants`: int array $(R, S)$ with nonnegative stoichiometry for each
            reaction $j$ and species $i$.
        - `net_stoich`: int array $(R, S)$ with net state change per reaction.
        - `rates`: float array $(R,)$ of rate constants $\kappa_j$.
        """
        if rates is None:
            raise ValueError(
                "`rates` must be provided as a (R,) array of rate constants."
            )
        reactants = jnp.asarray(reactants)
        net_stoich = jnp.asarray(net_stoich)
        if reactants.ndim != 2 or net_stoich.ndim != 2:
            raise ValueError("reactants and net_stoich must be 2D arrays (R, S).")
        if reactants.shape != net_stoich.shape:
            raise ValueError("reactants and net_stoich shapes must match.")
        self.reactants = reactants.astype(jnp.int32)
        self.net_stoich = net_stoich.astype(jnp.int32)
        self.kappas = jnp.asarray(rates)
        # Precompute log-factorials of reactant stoichiometries
        self._lgamma_rp1 = gammaln(self.reactants.astype(jnp.float32) + 1.0)

    def rate(self, t: RealScalarLike, u: U, args: Args) -> Float[Array, " R"]:
        """Compute propensities for all reactions."""
        u_arr = jnp.asarray(u)
        kappas = self.kappas.astype(jnp.result_type(u_arr, 1.0))
        u2 = jnp.broadcast_to(u_arr, self.reactants.shape)
        comb = _massaction_comb(u2, self.reactants, self._lgamma_rp1)
        per_rxn = jnp.prod(comb, axis=1)
        return kappas * per_rxn

    def affect(
        self,
        t: RealScalarLike,
        u: U,
        args: Args,
        jump_state: JumpState,
    ) -> tuple[U, JumpState]:
        """Return post-jump states for all reactions."""
        u_arr = jnp.asarray(u)
        affects = u_arr[None, :] + self.net_stoich
        return affects, jump_state

    def init(self, u: U, args: Args, key: Key[Array, ""]) -> None:
        """No state needed for mass-action jumps."""
        return None

    def leap_delta(self, t: RealScalarLike, u: U, args: Args) -> U:
        """Return the net stoichiometry matrix."""
        return self.net_stoich


class VariableRateJump(AbstractJumpProblem[_ScalarRate]):
    r"""
    A jump process with a rate $\lambda(t, u)$ that depends on time and state.

    Unlike [`jumpax.ConstantRateJump`][], the rate is evaluated continuously during
    integration, making this suitable for processes coupled to ODEs or other
    continuously evolving dynamics.
    """

    rate_fn: Callable[[RealScalarLike, U, Args], _ScalarRate]
    affect_fn: AbstractAffect
    leap_delta_fn: Callable[[RealScalarLike, U, Args], U] | None = None

    def rate(self, t: RealScalarLike, u: U, args: Args) -> _ScalarRate:
        """Evaluate the rate function."""
        return self.rate_fn(t, u, args)

    def affect(
        self, t: RealScalarLike, u: U, args: Args, jump_state: JumpState
    ) -> tuple[U, JumpState]:
        """Apply the affect and validate the output structure."""
        affect, jump_state = self.affect_fn(t, u, args, jump_state)
        if jax.tree.structure(affect) != jax.tree.structure(u):
            raise ValueError("`affect` and `u` should have the same tree structure.")
        return affect, jump_state

    def init(self, u: U, args: Args, key: Key[Array, ""]) -> JumpState:
        """Initialize the affect state."""
        return self.affect_fn.init(u, args, key)

    def leap_delta(self, t: RealScalarLike, u: U, args: Args) -> U:
        """Evaluate the leap delta function if provided."""
        if self.leap_delta_fn is not None:
            return self.leap_delta_fn(t, u, args)
        raise NotImplementedError(
            "VariableRateJump requires explicit leap_delta_fn for tau-leaping. "
            "Either provide `leap_delta_fn` or use SSA/VRJ methods instead."
        )
