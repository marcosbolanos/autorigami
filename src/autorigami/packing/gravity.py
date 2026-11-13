"""Pack a spiral centerline along z with a PhysX gravity simulation.

The horizontal shape of the curve never changes: each vertex is a sphere on a
vertical rail. Gravity pulls the spheres down, contacts between spheres keep
the separation constraint, and a spring between neighbor vertices acts as
string tension that keeps the height profile smooth. The few sharp vertical
corners that remain are rounded upward by a linear program.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import os
from typing import Final

import numpy as np
import numpy.typing as npt
from scipy.optimize import linprog
from scipy.sparse import coo_matrix, vstack

from autorigami._native import AxialGravitySimulation
from autorigami.geometry.curvature import (
    DEFAULT_EDGE_LENGTH_NM,
    DEFAULT_MAX_ANGLE_RADIANS,
    get_polyline_angles,
)
from autorigami.geometry.reparametrize import reparametrize_arc_length
from autorigami.geometry.separation import DEFAULT_MIN_DISTANCE
from autorigami.geometry.validation import (
    PolylineValidationResult,
    validate_polyline,
)
from autorigami.packing.curvature_repair import fix_curvature_violations
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


@dataclass(frozen=True)
class GravityPackingResult:
    """A packed centerline and its standard validation."""

    points: Polyline
    """The packed curve, sampled at the base-pair interval."""
    validation: PolylineValidationResult
    initial_height: float
    final_height: float


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
    assert polyline.dtype == np.float32
    assert polyline.ndim == 2 and polyline.shape[1] == 3 and len(polyline) >= 3
    assert bridging_length > 0.0

    points = _repair_horizontal_kinks(polyline)
    anchored = np.zeros(len(points), dtype=np.bool_)
    anchored[: _first_turn_vertex_count(points)] = True
    tension = _GRAVITY * (bridging_length / DEFAULT_EDGE_LENGTH_NM) ** 2
    initial_height = float(np.ptp(polyline[:, 2]))

    def result(candidate: Polyline) -> GravityPackingResult:
        sampled = reparametrize_arc_length(candidate, DEFAULT_EDGE_LENGTH_NM)
        return GravityPackingResult(
            points=sampled,
            validation=validate_polyline(sampled),
            initial_height=initial_height,
            final_height=float(np.ptp(sampled[:, 2])),
        )

    pinned = anchored.copy()
    points = _settle(points, pinned, tension)
    for round_index in range(_MAXIMUM_ROUNDS):
        settled = result(points)
        if settled.validation.valid:
            return settled
        # The remaining violations of a settled curve are sharp vertical
        # corners, which rounding removes.
        points, raised = _round_vertical_corners(points)
        rounded = result(points)
        if rounded.validation.valid:
            return rounded
        logger.info(
            "round %d: %d curvature and %d separation violations; "
            "pinned %d raised vertices",
            round_index,
            rounded.validation.curvature_violation_count,
            rounded.validation.separation_violation_count,
            int(np.count_nonzero(raised)),
        )
        # Raised vertices become supports, and the curve settles around them.
        pinned = pinned | raised
        points = _settle(points, pinned, tension)
    raise RuntimeError(
        f"gravity packing found no valid curve in {_MAXIMUM_ROUNDS} rounds; "
        "try a larger bridging_length"
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
    tension: float,
) -> Polyline:
    """Run the gravity simulation until the curve is at rest."""
    collision_radius = 0.5 * (DEFAULT_MIN_DISTANCE + _DISTANCE_MARGIN)
    simulation = AxialGravitySimulation(
        points=points,
        pinned=pinned,
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
    points: Polyline,
) -> tuple[Polyline, npt.NDArray[np.bool_]]:
    """Raise vertices by the least total amount that removes sharp corners.

    With fixed x and y, the turning angle at a vertex is at most the root sum
    of squares of the horizontal turning angle and the change of slope. The
    change of slope is linear in z, so the least upward correction is a
    linear program. Upward motion keeps the clearance to the curve below.
    The first vertex stays where it is.

    Returns:
        The corrected points and a mask of the vertices that were raised.
    """
    vertex_count = len(points)
    heights = points[:, 2].astype(np.float64)
    horizontal_lengths = np.linalg.norm(
        np.diff(points[:, :2].astype(np.float64), axis=0), axis=1
    )
    flat = points.copy()
    flat[:, 2] = 0.0
    horizontal_angles = get_polyline_angles(flat).astype(np.float64)
    design_angle = DEFAULT_MAX_ANGLE_RADIANS - _ANGLE_RESERVE
    slope_change_limit = np.tan(
        np.sqrt(np.maximum(design_angle**2 - horizontal_angles**2, 1e-8))
    )

    inner = np.arange(1, vertex_count - 1)
    before = 1.0 / horizontal_lengths[inner - 1]
    after = 1.0 / horizontal_lengths[inner]
    slope_change = coo_matrix(
        (
            np.stack((before, -before - after, after), axis=1).ravel(),
            (
                np.repeat(inner - 1, 3),
                np.stack((inner - 1, inner, inner + 1), axis=1).ravel(),
            ),
        ),
        shape=(vertex_count - 2, vertex_count),
    ).tocsr()
    current = slope_change @ heights
    solution = linprog(
        np.ones(vertex_count),
        A_ub=vstack((slope_change, -slope_change)).tocsr(),
        b_ub=np.concatenate(
            (slope_change_limit - current, slope_change_limit + current)
        ),
        bounds=[(0.0, 0.0)] + [(0.0, None)] * (vertex_count - 1),
        method="highs",
    )
    if solution.status != 0:
        raise RuntimeError(f"corner rounding failed: {solution.message}")
    rounded = points.copy()
    rounded[:, 2] = (heights + solution.x).astype(np.float32)
    return rounded, solution.x > 1e-5
