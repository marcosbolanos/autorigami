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
    init_len: float | None = None,
    init_jitter: float | None = None,
) -> NpControlPoints:
    """
    Initialize control points by picking random interior points and smoothing them.
    Returns P0: (C,n_ctrl,2)
    """
    Q = Y.shape[0]
    P0 = np.zeros((C, n_ctrl, 2), dtype=np.float32)
    if init_len is not None and init_len > 0:
        min_xy = Y.min(axis=0)
        max_xy = Y.max(axis=0)
        jitter = 0.0 if init_jitter is None else float(init_jitter)
        for i in range(C):
            anchor = Y[rng.integers(0, Q)]
            theta = float(rng.random()) * (2.0 * np.pi)
            direction = np.array([np.cos(theta), np.sin(theta)], dtype=np.float32)
            perp = np.array([-direction[1], direction[0]], dtype=np.float32)
            t = np.linspace(-0.5, 0.5, n_ctrl, dtype=np.float32)
            pts = anchor + (t[:, None] * float(init_len)) * direction
            if jitter > 0:
                jitter_vals = (
                    (rng.random(n_ctrl).astype(np.float32) - 0.5) * 2.0 * jitter
                )
                pts = pts + jitter_vals[:, None] * perp
            P0[i] = np.clip(pts, min_xy, max_xy)
        return P0
    for i in range(C):
        idx = rng.integers(0, Q, size=n_ctrl)
        pts = Y[idx].copy()
        for _ in range(5):
            pts[1:-1] = 0.25 * pts[:-2] + 0.5 * pts[1:-1] + 0.25 * pts[2:]
        P0[i] = pts
    return P0
