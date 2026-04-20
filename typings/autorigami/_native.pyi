from __future__ import annotations

from autorigami.types import HermiteSegments

def add(left: int, right: int) -> int: ...

def validate_piecewise_curve_curvature(
    segments: HermiteSegments,
    max_curvature: float,
    curvature_tolerance: float,
) -> bool: ...
