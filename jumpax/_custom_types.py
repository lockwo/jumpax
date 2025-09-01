from typing import Any, TYPE_CHECKING

import numpy as np
from jaxtyping import (
    Array,
    ArrayLike,
    PyTree,
    Real,
    Shaped,
)


if TYPE_CHECKING:
    RealScalarLike = bool | int | float | Array | np.ndarray
else:
    RealScalarLike = Real[ArrayLike, ""]


U = Shaped[Array, "?*u"]  # PyTree[Shaped[ArrayLike, "?*u"], "U"]
Rate = PyTree[Shaped[ArrayLike, "?*rate"], "R"]
Args = PyTree[Any]
JumpState = PyTree[Any]
SolverState = PyTree[Any]
DenseSolution = PyTree[Any]
Info = PyTree[Any]

del Array, ArrayLike, PyTree, Real, Shaped
