from __future__ import annotations

import numpy as np
from jaxtyping import Float32

type Vector3 = Float32[np.ndarray, "3"]  # noqa: F722
type Polyline = Float32[np.ndarray, "n 3"]  # noqa: F722

type EdgeIndex = int
