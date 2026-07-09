from __future__ import annotations

from typing import TypeAlias

import numpy as np
import numpy.typing as npt
from jaxtyping import Float32

type Vector3 = Float32[np.ndarray, "3"]  # noqa: F722
type Polyline = Float32[np.ndarray, "n 3"]  # noqa: F722

type EdgeIndex = int

Edge: TypeAlias = tuple[Vector3, Vector3]
