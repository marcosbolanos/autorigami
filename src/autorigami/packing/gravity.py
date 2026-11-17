"""Pack spiral centerlines along z with a PhysX gravity simulation.

The horizontal shape of the curve never changes: each vertex is a sphere on a
vertical rail. Gravity pulls the spheres down, contacts between spheres keep
the separation constraint, and a spring between neighbor vertices acts as
string tension that keeps the height profile smooth. The few sharp vertical
corners that remain are rounded upward by a linear program.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import logging
import os
from typing import Final

import numpy as np
import numpy.typing as npt
from scipy.optimize import linprog
from scipy.sparse import coo_matrix, vstack
from scipy.spatial import KDTree

from autorigami._native import AxialGravitySimulation
from autorigami.geometry.curvature import (
    DEFAULT_EDGE_LENGTH_NM,
    DEFAULT_MAX_ANGLE_RADIANS,
    get_polyline_angles,
)
from autorigami.geometry.reparametrize import reparametrize_arc_length
from autorigami.geometry.separation import DEFAULT_MIN_DISTANCE
from autorigami.geometry.validation import PolylineValidationResult
from autorigami.packing.curvature_repair import fix_curvature_violations
from autorigami.spiral_generation.validation import (
    SpiralValidationResult,
    validate_curves,
)
from autorigami.types import Polyline

logger = logging.getLogger(__name__)

# Clearance above the minimum distance. Spheres sit at the vertices, so the
# distance between two segments can be a little smaller than the distance
# between their vertices, and the contact solver permits very small overlaps.
_DISTANCE_MARGIN: Final = 0.05
# The packed curve is designed for this angle, which leaves a reserve below
# the maximum angle for the final arc-length resampling.
_ANGLE_RESERVE: Final = float(np.deg2rad(0.25))
# Largest horizontal turning angle after the horizontal repair. The remainder
# up to the design angle is available for vertical bending.
_HORIZONTAL_ANGLE: Final = float(np.deg2rad(2.5))
# Vertex pairs this near along the curve never collide (see the validator).
_IGNORED_ADJACENT_VERTICES: Final = 100

# Simulation units are arbitrary: only the ratio of tension to weight matters.
_GRAVITY: Final = 1.0
_LINEAR_DAMPING: Final = 1.0
_TENSION_DAMPING: Final = 50.0
_TIME_STEP: Final = 1.0 / 60.0
_POSITION_ITERATIONS: Final = 32
_STEPS_PER_CHECK: Final = 250
_MAXIMUM_STEPS: Final = 20_000
# The curve is at rest when no vertex moves more than this in one check.
_REST_DISPLACEMENT: Final = 1e-3
_MAXIMUM_ROUNDS: Final = 12
# Smallest distance between two vertices after the rounding of corners. It
# is between the minimum distance and the contact distance of the simulation:
# the rounding repairs real overlaps, but not the very small overlaps that
# the contact solver permits.
_ROUNDING_DISTANCE: Final = DEFAULT_MIN_DISTANCE + 0.5 * _DISTANCE_MARGIN
# The rounding examines the vertex pairs that are this much farther apart
# than the rounding distance, because it can move them nearer.
_ROUNDING_SEARCH_MARGIN: Final = 0.25
_MAXIMUM_ROUNDING_PASSES: Final = 8


@dataclass(frozen=True)
class GravityPackingResult:
    """A packed centerline and its standard validation."""

    points: Polyline
    """The packed curve, sampled at the base-pair interval."""
    validation: PolylineValidationResult
    initial_height: float
    final_height: float


@dataclass(frozen=True)
class CurvesGravityPackingResult:
    """Packed centerlines of one structure and their validation."""

    curves: list[Polyline]
    """The packed curves, sampled at the base-pair interval, in input sequence."""
    validation: SpiralValidationResult
    """Validation of each curve and of each pair of curves."""
    initial_height: float
    """Height of the structure before packing, in nm."""
    final_height: float
    """Height of the structure after packing, in nm."""


def pack_under_gravity(
    polyline: Polyline,
    *,
    bridging_length: float = 28.0,
) -> GravityPackingResult:
    """Pack a spiral along z as tightly as the hard constraints permit.

    The curve must wind upward along z, and its first turn is the support of
    the stack: that turn stays where it is. The x and y coordinates of the
    curve are preserved, except for the repair of horizontal kinks that are
    sharper than the curvature limit. The input does not have to be valid.

    Args:
        polyline: Float32 centerline with shape ``(n, 3)``, in nm.
        bridging_length: Stiffness of the height profile, in nm. A part of
            the curve that has no support below it stays almost straight
            across gaps shorter than this length. A smaller value packs
            tighter, but lets unsupported loops hang down.

    Returns:
        The packed curve sampled at 0.34 nm with its validation result.

    Raises:
        RuntimeError: If no valid curve was found. The standard validator
            ``validate_polyline`` makes this decision.
    """
    packed = pack_curves_under_gravity([polyline], bridging_length=bridging_length)
    return GravityPackingResult(
        points=packed.curves[0],
        validation=packed.validation.curves[0],
        initial_height=packed.initial_height,
        final_height=packed.final_height,
    )


def pack_curves_under_gravity(
    curves: Sequence[Polyline],
    *,
    bridging_length: float = 28.0,
) -> CurvesGravityPackingResult:
    """Pack the curves of one structure along z, all in one simulation.

    Use this for a spiral with a Y segment, which has more than one curve
    (see `Spiral.curves`). The first turn of the first curve is the support
    of the stack: that turn stays where it is. All other vertices fall, and
    the other curves come to rest on the curves below them. The x and y
    coordinates are preserved, except for the repair of horizontal kinks
    that are sharper than the curvature limit. The input does not have to be
    valid.

    The work has three steps. First, gradient descent removes the horizontal
    kinks. Second, the gravity simulation lets the curves fall until they are
    at rest. Third, a linear program rounds the sharp vertical corners that
    remain. The second and third steps repeat until the result is valid.

    Args:
        curves: Float32 centerlines with shape ``(n, 3)``, in nm. Each must
            wind upward along z.
        bridging_length: Stiffness of the height profile, in nm. A part of
            a curve that has no support below it stays almost straight
            across gaps shorter than this length. A smaller value packs
            tighter, but lets unsupported loops hang down.

    Returns:
        The packed curves sampled at 0.34 nm with their validation result.

    Raises:
        RuntimeError: If no valid result was found. The standard validator
            ``validate_polyline`` and the cross-curve check of
            ``validate_curves`` make this decision.
    """
    assert len(curves) >= 1
    assert bridging_length > 0.0
    for curve in curves:
        assert curve.dtype == np.float32
        assert curve.ndim == 2 and curve.shape[1] == 3 and len(curve) >= 3

    parts = [_repair_horizontal_kinks(curve) for curve in curves]
    curve_indices = np.concatenate(
        [np.full(len(part), index, dtype=np.uint32) for index, part in enumerate(parts)]
    )
    boundaries = np.cumsum([len(part) for part in parts])[:-1]
    anchored = np.zeros(len(curve_indices), dtype=np.bool_)
    anchored[: _first_turn_vertex_count(parts[0])] = True
    tension = _GRAVITY * (bridging_length / DEFAULT_EDGE_LENGTH_NM) ** 2
    initial_height = _height(curves)

    def result(candidates: Sequence[Polyline]) -> CurvesGravityPackingResult:
        sampled = [
            reparametrize_arc_length(candidate, DEFAULT_EDGE_LENGTH_NM)
            for candidate in candidates
        ]
        return CurvesGravityPackingResult(
            curves=sampled,
            validation=validate_curves(sampled),
            initial_height=initial_height,
            final_height=_height(sampled),
        )

    def settle(pinned: npt.NDArray[np.bool_]) -> list[Polyline]:
        settled = _settle(np.concatenate(parts, axis=0), pinned, curve_indices, tension)
        return np.split(settled, boundaries)

    pinned = anchored.copy()
    parts = settle(pinned)
    for round_index in range(_MAXIMUM_ROUNDS):
        settled = result(parts)
        if settled.validation.valid:
            return settled
        # The remaining violations of settled curves are sharp vertical
        # corners and small overlaps under a large load. Rounding removes
        # them.
        parts, raised = _round_vertical_corners(parts)
        rounded = result(parts)
        if rounded.validation.valid:
            return rounded
        logger.info(
            "round %d: not valid after rounding; pinned %d raised vertices\n%s",
            round_index,
            int(np.count_nonzero(raised)),
            rounded.validation,
        )
        # Raised vertices become supports, and the curves settle around them.
        pinned = pinned | raised
        parts = settle(pinned)
    raise RuntimeError(
        f"gravity packing found no valid curve in {_MAXIMUM_ROUNDS} rounds; "
        "try a larger bridging_length"
    )


def _height(curves: Sequence[Polyline]) -> float:
    return float(
        max(curve[:, 2].max() for curve in curves)
        - min(curve[:, 2].min() for curve in curves)
    )


def _repair_horizontal_kinks(polyline: Polyline) -> Polyline:
    """Remove kinks of the xy projection and keep the height profile."""
    flat = polyline.copy()
    flat[:, 2] = 0.0
    flat = fix_curvature_violations(
        reparametrize_arc_length(flat, DEFAULT_EDGE_LENGTH_NM),
        target_angle=_HORIZONTAL_ANGLE,
        edge_length=DEFAULT_EDGE_LENGTH_NM,
        coordinate_mask=(True, True, False),
    )
    # Arc-length sampling leaves a last edge of any length down to zero. Two
    # vertices at almost the same xy would settle at the same point.
    last_edge = float(np.linalg.norm(flat[-1, :2] - flat[-2, :2]))
    if last_edge < 0.5 * DEFAULT_EDGE_LENGTH_NM:
        flat = flat[:-1]
    old_arc = _horizontal_arc_fraction(polyline)
    flat[:, 2] = np.interp(_horizontal_arc_fraction(flat), old_arc, polyline[:, 2])
    return np.ascontiguousarray(flat, dtype=np.float32)


def _horizontal_arc_fraction(polyline: Polyline) -> npt.NDArray[np.float64]:
    lengths = np.linalg.norm(np.diff(polyline[:, :2], axis=0), axis=1)
    arc = np.concatenate(([0.0], np.cumsum(lengths, dtype=np.float64)))
    return arc / arc[-1]


def _first_turn_vertex_count(points: Polyline) -> int:
    """Count the vertices of the first complete horizontal turn."""
    edges = np.diff(points[:, :2].astype(np.float64), axis=0)
    winding = np.abs(np.unwrap(np.arctan2(edges[:, 1], edges[:, 0])))
    complete = (winding - winding[0]) >= 2.0 * np.pi
    assert np.any(complete), "the curve must make more than one turn"
    return int(np.argmax(complete)) + 1


def _settle(
    points: Polyline,
    pinned: npt.NDArray[np.bool_],
    curve_indices: npt.NDArray[np.uint32],
    tension: float,
) -> Polyline:
    """Run the gravity simulation until all curves are at rest."""
    collision_radius = 0.5 * (DEFAULT_MIN_DISTANCE + _DISTANCE_MARGIN)
    simulation = AxialGravitySimulation(
        points=points,
        pinned=pinned,
        curve_indices=curve_indices,
        collision_radius=collision_radius,
        ignored_adjacent_vertices=_IGNORED_ADJACENT_VERTICES,
        tension_stiffness=tension,
        tension_damping=_TENSION_DAMPING,
        floor_height=float(points[:, 2].min()) - collision_radius,
        gravity=_GRAVITY,
        linear_damping=_LINEAR_DAMPING,
        maximum_depenetration_velocity=1.0,
        maximum_linear_velocity=5.0,
        position_iterations=_POSITION_ITERATIONS,
        velocity_iterations=1,
        thread_count=os.cpu_count() or 1,
    )
    previous = simulation.points()
    for step in range(_STEPS_PER_CHECK, _MAXIMUM_STEPS + 1, _STEPS_PER_CHECK):
        simulation.step(_STEPS_PER_CHECK, _TIME_STEP)
        current = simulation.points()
        if not np.all(np.isfinite(current)):
            raise RuntimeError(f"the gravity simulation diverged at step {step}")
        displacement = float(np.max(np.abs(current[:, 2] - previous[:, 2])))
        previous = current
        logger.info(
            "step %d: height %.3f nm, largest displacement %.5f nm",
            step,
            float(np.ptp(current[:, 2])),
            displacement,
        )
        if displacement < _REST_DISPLACEMENT:
            break
    return previous


def _round_vertical_corners(
    parts: Sequence[Polyline],
) -> tuple[list[Polyline], npt.NDArray[np.bool_]]:
    """Raise vertices by the least total amount that removes sharp corners.

    With fixed x and y, the turning angle at a vertex is at most the root sum
    of squares of the horizontal turning angle and the change of slope. The
    change of slope is linear in z, so the least upward correction is a
    linear program.

    The correction must not move a vertex into the curve above it. For two
    vertices with fixed x and y, the separation is a lower limit on their
    height difference, which is also linear in z. Thus the same linear
    program keeps each vertex pair at the rounding distance or farther
    apart, and also moves apart the pairs that are nearer when it starts.

    The first vertex of the first curve stays where it is.

    Returns:
        The corrected curves, and a mask of the vertices that were raised.
        The mask is for the vertices of all curves, one curve after the other.
    """
    joined = np.concatenate(parts, axis=0)
    vertex_count = len(joined)
    heights = joined[:, 2].astype(np.float64)
    offsets = np.cumsum([0] + [len(part) for part in parts])
    curve_indices = np.repeat(np.arange(len(parts)), [len(part) for part in parts])
    design_angle = DEFAULT_MAX_ANGLE_RADIANS - _ANGLE_RESERVE

    blocks = []
    limits = []
    for part, offset in zip(parts, offsets[:-1], strict=True):
        horizontal_lengths = np.linalg.norm(
            np.diff(part[:, :2].astype(np.float64), axis=0), axis=1
        )
        flat = part.copy()
        flat[:, 2] = 0.0
        horizontal_angles = get_polyline_angles(flat).astype(np.float64)
        limits.append(
            np.tan(np.sqrt(np.maximum(design_angle**2 - horizontal_angles**2, 1e-8)))
        )
        inner = np.arange(1, len(part) - 1)
        before = 1.0 / horizontal_lengths[inner - 1]
        after = 1.0 / horizontal_lengths[inner]
        blocks.append(
            coo_matrix(
                (
                    np.stack((before, -before - after, after), axis=1).ravel(),
                    (
                        np.repeat(inner - 1, 3),
                        offset
                        + np.stack((inner - 1, inner, inner + 1), axis=1).ravel(),
                    ),
                ),
                shape=(len(part) - 2, vertex_count),
            )
        )
    slope_change = vstack(blocks).tocsr()
    slope_change_limit = np.concatenate(limits)
    current = slope_change @ heights

    horizontal = joined[:, :2].astype(np.float64)
    lower = np.zeros(0, dtype=np.intp)
    upper = np.zeros(0, dtype=np.intp)
    raise_amount = np.zeros(vertex_count)
    for _ in range(_MAXIMUM_ROUNDING_PASSES):
        # Find the vertex pairs that the correction can move too near.
        moved = np.column_stack((horizontal, heights + raise_amount))
        pairs = KDTree(moved).query_pairs(
            _ROUNDING_DISTANCE + _ROUNDING_SEARCH_MARGIN, output_type="ndarray"
        )
        first, second = pairs[:, 0], pairs[:, 1]
        can_collide = (curve_indices[first] != curve_indices[second]) | (
            second - first > _IGNORED_ADJACENT_VERTICES
        )
        first, second = first[can_collide], second[can_collide]
        first_is_lower = moved[first, 2] <= moved[second, 2]
        known = len(lower)
        codes = np.unique(
            np.concatenate(
                (
                    lower * vertex_count + upper,
                    np.where(first_is_lower, first, second) * vertex_count
                    + np.where(first_is_lower, second, first),
                )
            )
        )
        lower, upper = np.divmod(codes, vertex_count)
        if len(lower) == known:
            break  # No new pair came near: the correction is complete.
        horizontal_distance = np.linalg.norm(
            horizontal[upper] - horizontal[lower], axis=1
        )
        height_difference = np.sqrt(
            np.maximum(_ROUNDING_DISTANCE**2 - horizontal_distance**2, 0.0)
        )
        pair_rows = np.arange(len(lower))
        separation = coo_matrix(
            (
                np.concatenate((np.ones(len(lower)), -np.ones(len(lower)))),
                (
                    np.concatenate((pair_rows, pair_rows)),
                    np.concatenate((lower, upper)),
                ),
            ),
            shape=(len(lower), vertex_count),
        )
        solution = linprog(
            np.ones(vertex_count),
            A_ub=vstack((slope_change, -slope_change, separation)).tocsr(),
            b_ub=np.concatenate(
                (
                    slope_change_limit - current,
                    slope_change_limit + current,
                    heights[upper] - heights[lower] - height_difference,
                )
            ),
            bounds=[(0.0, 0.0)] + [(0.0, None)] * (vertex_count - 1),
            method="highs",
        )
        if solution.status != 0:
            raise RuntimeError(f"corner rounding failed: {solution.message}")
        raise_amount = solution.x
    rounded = joined.copy()
    rounded[:, 2] = (heights + raise_amount).astype(np.float32)
    return np.split(rounded, offsets[1:-1]), raise_amount > 1e-5
