from __future__ import annotations

from autorigami.parametrization import PiecewiseHermite

def add(left: int, right: int) -> int: ...
def piecewise_hermite_generator() -> PiecewiseHermite: ...

def validate_piecewise_curve_curvature(
    piecewise_hermite: PiecewiseHermite,
    max_curvature: float,
    curvature_tolerance: float,
) -> bool: ...
