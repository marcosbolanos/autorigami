"""Validate the curves of a spiral together.

`autorigami.geometry.validation.validate_polyline` examines one curve. A
spiral with a Y segment can have more than one curve, and two valid curves
can still be too near to each other. `validate_curves` calls
`validate_polyline` on each curve and then measures the distance between each
pair of curves.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from autorigami._native import find_close_edge_pairs, segment_segment_distance
from autorigami.geometry.separation import DEFAULT_MIN_DISTANCE
from autorigami.geometry.validation import PolylineValidationResult, validate_polyline
from autorigami.types import Polyline


@dataclass(frozen=True)
class CrossCurveSeparation:
    """Separation between two different curves of one spiral."""

    first_curve: int
    """Index of the first curve in the validated sequence."""
    second_curve: int
    """Index of the second curve in the validated sequence."""
    violation_count: int
    """Number of edge pairs, one edge from each curve, that are too near."""
    minimum_violating_distance: float | None
    """Smallest distance of these edge pairs, in nm. None if there are none."""


@dataclass(frozen=True)
class SpiralValidationResult:
    """Curvature and separation validation of all curves of a spiral."""

    curves: tuple[PolylineValidationResult, ...]
    """The result of `validate_polyline` for each curve, in sequence."""
    cross_curve: tuple[CrossCurveSeparation, ...]
    """The separation of each pair of curves. Empty for one curve."""

    @property
    def valid(self) -> bool:
        """True if each curve is valid and no two curves are too near."""
        return all(curve.valid for curve in self.curves) and all(
            pair.violation_count == 0 for pair in self.cross_curve
        )

    def __str__(self) -> str:
        lines = ["valid" if self.valid else "NOT VALID"]
        for index, curve in enumerate(self.curves):
            lines.append(
                f"curve {index}: {curve.curvature_violation_count} curvature "
                f"violations (largest angle {np.degrees(curve.maximum_angle):.2f} "
                f"deg), {curve.separation_violation_count} separation violations"
                + _smallest(curve.minimum_violating_distance)
            )
        for pair in self.cross_curve:
            lines.append(
                f"curves {pair.first_curve} and {pair.second_curve}: "
                f"{pair.violation_count} separation violations"
                + _smallest(pair.minimum_violating_distance)
            )
        return "\n".join(lines)


def validate_curves(
    curves: Sequence[Polyline],
    *,
    minimum_distance: float = DEFAULT_MIN_DISTANCE,
) -> SpiralValidationResult:
    """Validate each curve, and the separation between each pair of curves.

    Args:
        curves: The centerlines, for example from `Spiral.curves`. Each is a
            float32 polyline with shape (n, 3), in nm.
        minimum_distance: Smallest permitted distance between two edges, in nm.

    Returns:
        The result for each curve and for each pair of curves.
    """
    return SpiralValidationResult(
        curves=tuple(
            validate_polyline(curve, minimum_distance=minimum_distance)
            for curve in curves
        ),
        cross_curve=tuple(
            cross_curve_separation(
                curves[first], curves[second], first, second, minimum_distance
            )
            for first in range(len(curves))
            for second in range(first + 1, len(curves))
        ),
    )


def cross_curve_separation(
    first: Polyline,
    second: Polyline,
    first_index: int = 0,
    second_index: int = 1,
    minimum_distance: float = DEFAULT_MIN_DISTANCE,
) -> CrossCurveSeparation:
    """Find the edge pairs across two curves that are too near.

    The two curves are joined into one polyline, so that the native functions
    of the validator can measure them. Only the pairs with one edge from each
    curve are kept. The false edge at the joint is not in any kept pair.
    """
    joined = np.concatenate((first, second), axis=0).astype(np.float32)
    joint_edge = len(first) - 1
    pairs = [
        (low, high)
        for low, high in (
            sorted(pair) for pair in find_close_edge_pairs(joined, minimum_distance, 1)
        )
        if low < joint_edge < high
    ]
    violating = [
        distance
        for distance, _, _ in segment_segment_distance(joined, pairs)
        if distance < minimum_distance
    ]
    return CrossCurveSeparation(
        first_curve=first_index,
        second_curve=second_index,
        violation_count=len(violating),
        minimum_violating_distance=min(violating) if violating else None,
    )


def _smallest(distance: float | None) -> str:
    return "" if distance is None else f" (smallest distance {distance:.3f} nm)"
