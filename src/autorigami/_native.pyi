from __future__ import annotations

from typing import TypeAlias

from autorigami.parametrization import PiecewiseHermite

GeneratorRunData: TypeAlias = dict[str, int | float]

def add(left: int, right: int) -> int: ...
def piecewise_hermite_generator() -> tuple[PiecewiseHermite, GeneratorRunData]: ...
def validate_piecewise_curve_curvature(
    piecewise_hermite: PiecewiseHermite,
    max_curvature: float,
    curvature_tolerance: float,
) -> bool: ...
