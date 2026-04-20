from __future__ import annotations

import numpy as np

from autorigami.parametrization import (
    PiecewiseBezier,
    PiecewiseHermite,
    Polyline,
    piecewise_hermite_to_bezier,
    sample_cubic_bezier_chain,
)


def _eval_cubic_hermite_segment(
    p0: np.ndarray,
    p1: np.ndarray,
    m0: np.ndarray,
    m1: np.ndarray,
    t: np.ndarray,
) -> np.ndarray:
    t_col = t[:, None]
    t2 = t_col * t_col
    t3 = t2 * t_col
    h00 = 2.0 * t3 - 3.0 * t2 + 1.0
    h10 = t3 - 2.0 * t2 + t_col
    h01 = -2.0 * t3 + 3.0 * t2
    h11 = t3 - t2
    return h00 * p0 + h10 * m0 + h01 * p1 + h11 * m1


def test_piecewise_hermite_to_bezier_is_exact() -> None:
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.2, -0.5, 0.3],
            [2.0, 1.0, -0.2],
        ],
        dtype=np.float64,
    )
    tangents = np.array(
        [
            [0.8, 0.1, -0.3],
            [0.2, 0.9, 0.4],
            [0.5, -0.7, 0.6],
        ],
        dtype=np.float64,
    )

    hermite = PiecewiseHermite(points=points, tangents=tangents)
    bezier = piecewise_hermite_to_bezier(hermite)

    ts = np.linspace(0.0, 1.0, 31, dtype=np.float64)
    for i in range(bezier.segments.shape[0]):
        b = bezier.segments[i]
        p0, p1 = points[i], points[i + 1]
        m0, m1 = tangents[i], tangents[i + 1]

        hermite_points = _eval_cubic_hermite_segment(p0, p1, m0, m1, ts)

        t_col = ts[:, None]
        omt = 1.0 - t_col
        bezier_points = (
            (omt**3) * b[0]
            + 3.0 * (omt**2) * t_col * b[1]
            + 3.0 * omt * (t_col**2) * b[2]
            + (t_col**3) * b[3]
        )

        np.testing.assert_allclose(hermite_points, bezier_points, rtol=0.0, atol=1e-12)


def test_explicit_types_and_sampling() -> None:
    bezier = PiecewiseBezier(
        segments=np.array(
            [
                [[0.0, 0.0, 0.0], [0.3, 0.0, 0.0], [0.7, 0.0, 0.0], [1.0, 0.0, 0.0]],
                [[1.0, 0.0, 0.0], [1.3, 0.5, 0.0], [1.7, -0.5, 0.0], [2.0, 0.0, 0.0]],
            ],
            dtype=np.float64,
        )
    )

    sampled = sample_cubic_bezier_chain(bezier, num_samples=25)

    assert isinstance(sampled, Polyline)
    assert sampled.points.shape == (25, 3)
    np.testing.assert_allclose(
        sampled.points[0], bezier.segments[0, 0], rtol=0.0, atol=0.0
    )
    np.testing.assert_allclose(
        sampled.points[-1], bezier.segments[-1, 3], rtol=0.0, atol=0.0
    )
