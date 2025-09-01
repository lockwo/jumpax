from typing import Any

import equinox as eqx
from jaxtyping import Array, Float, Int


class Solution(eqx.Module):
    """
    A module containing all possible solution information.

    - `ts`: is the times at which states where recorded (if the
        states were recorded)
    - `us`: the states logged
    - `counts`: optionally the reaction counts for each possible reaction
    - `stats`: information on the solution finding process
    - `dense`: optionally a dict with `dense_ts` and `dense_us` from the diffrax solver
    """

    ts: Float[Array, " max_steps"]
    us: Float[Array, " max_steps *ushape"]
    counts: Int[Array, " num_reactions"] | None
    stats: dict[str, Any]
    dense: dict[str, Array] | None
