from __future__ import annotations

from typing import TypeAlias

import jax
import numpy as np
from beartype.typing import Callable
from jaxtyping import Bool, Float, Int

NpControlPoints: TypeAlias = Float[np.ndarray, "C n_ctrl 2"]
NpBasisMatrix: TypeAlias = Float[np.ndarray, "M n_ctrl"]
NpSdfGrid: TypeAlias = Float[np.ndarray, "H W"]
NpMask: TypeAlias = Bool[np.ndarray, "H W"]
NpOrigin: TypeAlias = Float[np.ndarray, "2"]
NpSamplePoints: TypeAlias = Float[np.ndarray, "Q 2"]
NpCurveSamples: TypeAlias = Float[np.ndarray, "C M 2"]
NpPolygon: TypeAlias = Float[np.ndarray, "N 2"]
NpRadii: TypeAlias = Float[np.ndarray, "C"]
PairsNp: TypeAlias = tuple[
    Int[np.ndarray, "P"],
    Int[np.ndarray, "P"],
    Int[np.ndarray, "P"],
    Int[np.ndarray, "P"],
]
JaxControlPoints: TypeAlias = Float[jax.Array, "C n_ctrl 2"]
JaxCurveSamples: TypeAlias = Float[jax.Array, "C M 2"]
JaxRadii: TypeAlias = Float[jax.Array, "C"]
PairsJax: TypeAlias = tuple[
    Int[jax.Array, "P"],
    Int[jax.Array, "P"],
    Int[jax.Array, "P"],
    Int[jax.Array, "P"],
]
JaxScalar: TypeAlias = Float[jax.Array, ""]
LossFn: TypeAlias = Callable[
    [JaxControlPoints, PairsJax, JaxRadii, JaxScalar, JaxScalar], JaxScalar
]
