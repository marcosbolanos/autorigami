from __future__ import annotations

import numpy as np
from beartype import beartype
from jaxtyping import jaxtyped

from .optimize_types import NpControlPoints, NpSamplePoints


@jaxtyped(typechecker=beartype)
def sample_init_control_points(
    Y: NpSamplePoints,
    C: int,
    n_ctrl: int,
    rng: np.random.Generator,
) -> NpControlPoints:
    """
    Initialize control points by picking random interior points and smoothing them.
    Returns P0: (C,n_ctrl,2)
    """
    Q = Y.shape[0]
    P0 = np.zeros((C, n_ctrl, 2), dtype=np.float32)
    for i in range(C):
        idx = rng.integers(0, Q, size=n_ctrl)
        pts = Y[idx].copy()
        for _ in range(5):
            pts[1:-1] = 0.25 * pts[:-2] + 0.5 * pts[1:-1] + 0.25 * pts[2:]
        P0[i] = pts
    return P0
