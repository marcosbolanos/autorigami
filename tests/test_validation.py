from __future__ import annotations

import numpy as np

from autorigami._native import piecewise_hermite_generator, validate_piecewise_curve_curvature
from autorigami.parametrization import PiecewiseHermite


def test_validate_piecewise_curve_curvature_accepts_straight_segments() -> None:
    piecewise_hermite = PiecewiseHermite(
        points=np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
            ],
            dtype=float,
        ),
        tangents=np.array(
            [
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
            ],
            dtype=float,
        ),
    )

    assert validate_piecewise_curve_curvature(
        piecewise_hermite=piecewise_hermite,
        max_curvature=2.0,
        curvature_tolerance=0.01,
    )


def test_piecewise_hermite_generator_returns_dataclass() -> None:
    generated = piecewise_hermite_generator()

    assert isinstance(generated, PiecewiseHermite)
    assert generated.points.shape == generated.tangents.shape
    assert generated.points.shape[0] >= 2
    assert generated.points.shape[1] == 3
    assert validate_piecewise_curve_curvature(
        piecewise_hermite=generated,
        max_curvature=100.0,
        curvature_tolerance=1e-6,
    )
