from __future__ import annotations

import numpy as np

from autorigami._native import validate_polyline_constraints


def test_validate_polyline_constraints_accepts_straight_polyline() -> None:
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
        ],
        dtype=float,
    )

    report = validate_polyline_constraints(
        points=points,
        separation=0.5,
        max_curvature=2.0,
        neighbor_exclusion=1,
    )

    assert report.separation.compliant_count == 4
    assert report.separation.total_count == 4
    assert report.separation.ratio == 1.0
    assert report.curvature.compliant_count == 4
    assert report.curvature.total_count == 4
    assert report.curvature.ratio == 1.0
