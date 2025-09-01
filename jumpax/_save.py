import equinox as eqx


class Save(eqx.Module):
    """Controls what data to save during simulation."""

    states: bool = True
    reaction_counts: bool = False
    dense: bool = False


Save.__init__.__doc__ = """**Arguments:**

- `states`: Whether to save the state at each step of the solver. Defaults to `True`.
- `reaction_counts`: Whether to save an array containing the number of times each
    reaction occurred. Defaults to `False`.
- `dense`: Whether to save the intermediate diffrax solver steps in hybrid models.
    Only applicable when using [`jumpax.HybridSSA`][] or [`jumpax.HazardSSA`][].
    Defaults to `False`.
"""
