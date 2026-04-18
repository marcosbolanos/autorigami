from __future__ import annotations

import numpy as np

from autorigami._native import validate_piecewise_curve_curvature


def test_validate_piecewise_curve_curvature_accepts_straight_segments() -> None:
    segments = np.array(
        [
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
            ],
            [
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
            ],
        ],
        dtype=float,
    )

    assert validate_piecewise_curve_curvature(
        segments=segments,
        max_curvature=2.0,
        curvature_tolerance=0.01,
    )
