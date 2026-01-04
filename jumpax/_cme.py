import equinox as eqx
import jax
import jax.numpy as jnp
from jax.scipy.linalg import expm
from jaxtyping import Array, Float, Int

from ._custom_types import Args
from ._jumps import MassActionJump


class CMEState(eqx.Module):
    """State space information for Chemical Master Equation (CME) solver.

    Attributes:
        states: array of all states in the truncated state space
        strides: array for mixed-radix encoding
        max_counts: array of maximum count per species
    """

    states: Int[Array, "N S"]
    strides: Int[Array, " S"]
    max_counts: Int[Array, " S"]


def _make_strides(max_counts: Int[Array, " S"]) -> Int[Array, " S"]:
    """Compute mixed-radix strides for state encoding (row major).

    For max_counts = [m0, m1, m2], the bases are [m0+1, m1+1, m2+1]
    and strides = [(m1+1)*(m2+1), m2+1, 1].
    """
    bases = max_counts + 1
    # strides[i] = prod_{k>i} bases[k]
    reversed_cumprod = jnp.cumprod(bases[::-1])[::-1]
    strides = jnp.concatenate([reversed_cumprod[1:], jnp.array([1], dtype=jnp.int32)])
    return strides


def _encode_states(
    states: Int[Array, "... S"], strides: Int[Array, " S"]
) -> Int[Array, "..."]:
    """Encode states as integer indices using mixed-radix representation."""
    return jnp.sum(states * strides, axis=-1)


def _enumerate_states(max_counts: Int[Array, " S"]) -> Int[Array, "N S"]:
    """Enumerate all states in the truncated rectangular lattice.

    Returns states in the order compatible with mixed-radix encoding.
    """
    axes = [jnp.arange(m + 1, dtype=jnp.int32) for m in max_counts.tolist()]
    grids = jnp.meshgrid(*axes, indexing="ij")
    S = max_counts.shape[0]
    states = jnp.stack(grids, axis=-1).reshape(-1, S)
    return states


def build_generator(
    problem: MassActionJump,
    max_counts: Int[Array, " S"],
    *,
    args: Args = None,
    t0: float = 0.0,
) -> tuple[Float[Array, "N N"], CMEState]:
    r"""Build the generator matrix Q for a [`jumpax.MassActionJump`][] on a
    truncated state space.

    The state space is the rectangular lattice $\{0, ..., \text{max\_counts}_i\}$ for
    each species $i$. Transitions leaving this box are dropped.

    **Arguments:**

    - `problem`: A [`jumpax.MassActionJump`][] defining the reaction system
    - `max_counts`: array of maximum population per species
    - `args`: Optional arguments passed to rate function
    - `t0`: Time at which to evaluate rates (only matters for time-dependent rates)

    **Returns:**

    - `Q`: generator matrix where N = prod(max_counts + 1)
    - `cme_state`: CMEState containing state space information
    """
    max_counts = jnp.asarray(max_counts, dtype=jnp.int32)
    strides = _make_strides(max_counts)
    states = _enumerate_states(max_counts)
    N = states.shape[0]

    rates = jax.vmap(lambda u: problem.rate(t0, u, args))(states)

    # Net stoichiometry gives state change per reaction
    net_stoich = problem.net_stoich

    # Compute destination states for each (state, reaction) pair
    # states: (N, S), net_stoich: (R, S) -> next_states: (N, R, S)
    next_states = states[:, None, :] + net_stoich[None, :, :]

    # Check which transitions stay in bounds
    in_bounds = jnp.all((next_states >= 0) & (next_states <= max_counts), axis=-1)

    # Encode destination states as indices
    next_ids = _encode_states(next_states, strides)

    # Identify self-transitions (no state change)
    row_ids = jnp.arange(N, dtype=jnp.int32)[:, None]
    is_self = next_ids == row_ids

    # Off-diagonal entries: rate * in_bounds * (not self-transition)
    weights = rates * in_bounds * (~is_self)

    Q = jnp.zeros((N, N), dtype=rates.dtype)

    # Q[dest, source] = rate (for CME: dP/dt = Q @ P)
    rows = next_ids.reshape(-1)  # destination states
    cols = jnp.broadcast_to(row_ids, (N, rates.shape[1])).reshape(-1)  # source states
    vals = weights.reshape(-1)

    Q = Q.at[rows, cols].add(vals)

    # Set diagonal: Q[j,j] = -sum_{i != j} Q[i,j] (negative sum of column)
    col_sums = jnp.sum(Q, axis=0)
    Q = Q.at[jnp.arange(N), jnp.arange(N)].set(-col_sums)

    cme_state = CMEState(states=states, strides=strides, max_counts=max_counts)
    return Q, cme_state


def solve_cme(
    Q: Float[Array, "N N"],
    p0: Float[Array, " N"],
    t: float,
) -> Float[Array, " N"]:
    r"""Solve the CME exactly via matrix exponentiation.

    Computes $P(t) = \exp(Q t) P(0)$.

    **Arguments:**

    - `Q`: generator matrix from [`jumpax.build_generator`][]
    - `p0`: initial probability distribution (should sum to 1)
    - `t`: Time at which to evaluate the distribution

    **Returns:**

    - `p_t`: probability distribution at time t
    """
    return expm(Q * t) @ p0


def marginal_distribution(
    p: Float[Array, " N"],
    cme_state: CMEState,
    species: int,
) -> Float[Array, " M"]:
    """Compute the marginal distribution for a single species.

    **Arguments:**

    - `p`: probability distribution over states
    - `cme_state`: State space information
    - `species`: Index of species to marginalize over

    **Returns:**

    - `marginal`: marginal distribution where M = max_counts[species] + 1
    """
    species_counts = cme_state.states[:, species]  # (N,)
    max_count = cme_state.max_counts[species]
    # Sum probabilities for each count value
    marginal = jnp.zeros(max_count + 1, dtype=p.dtype)
    marginal = marginal.at[species_counts].add(p)
    return marginal
