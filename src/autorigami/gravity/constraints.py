from __future__ import annotations

# pyright: reportOptionalSubscript=false

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.sparse import csc_matrix, csr_matrix, eye, vstack

from autorigami._native import segment_segment_distance_parameters
from autorigami.geometry.curvature import (
    DEFAULT_MAX_ANGLE_RADIANS,
    get_polyline_angles,
)
from autorigami.geometry.separation import (
    DEFAULT_MIN_DISTANCE,
    get_candidate_intersecting_edges,
)
from autorigami.optimization.constraints import active_constraint_jacobian
from autorigami.optimization.inextensibility import (
    edge_constraint_jacobian,
    edge_constraint_residuals,
    maximum_edge_length_error,
)

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class GravityConstraintConfiguration:
    maximum_angle: float = DEFAULT_MAX_ANGLE_RADIANS
    minimum_distance: float = DEFAULT_MIN_DISTANCE
    ignored_adjacent_edges: int = 100
    contact_safety_margin: float = 0.01
    curvature_safety_margin: float = 5e-4
    floor_safety_margin: float = 0.0

    def __post_init__(self) -> None:
        assert 0.0 < self.maximum_angle < np.pi
        assert self.minimum_distance > 0.0
        assert self.ignored_adjacent_edges >= 1
        assert self.contact_safety_margin >= 0.0
        assert 0.0 <= self.curvature_safety_margin < self.maximum_angle
        assert self.floor_safety_margin >= 0.0


@dataclass(frozen=True)
class GravityGeometricLinearization:
    matrix: csr_matrix
    slacks: FloatArray
    curvature_count: int
    contact_count: int
    floor_count: int
    contact_pairs: tuple[tuple[int, int], ...]

    def __post_init__(self) -> None:
        assert self.matrix.shape[0] == len(self.slacks)
        assert self.matrix.shape[0] == (
            self.curvature_count + self.contact_count + self.floor_count
        )


@dataclass(frozen=True)
class GravityConstraintLinearization:
    matrix: csc_matrix
    lower_bounds: FloatArray
    upper_bounds: FloatArray
    vertex_count: int
    edge_count: int
    curvature_count: int
    contact_count: int
    floor_count: int
    trust_count: int
    contact_pairs: tuple[tuple[int, int], ...]

    def __post_init__(self) -> None:
        row_count = (
            self.edge_count
            + self.curvature_count
            + self.contact_count
            + self.floor_count
            + self.trust_count
        )
        assert self.matrix.shape == (row_count, 3 * self.vertex_count)
        assert self.lower_bounds.shape == self.upper_bounds.shape == (row_count,)


@dataclass(frozen=True)
class GravityConstraintResiduals:
    maximum_edge_error: float
    maximum_angle: float
    minimum_contact_slack: float
    separation_violation_count: int
    minimum_floor_slack: float


def floor_slacks(
    points: FloatArray,
    *,
    floor_height: float,
    tube_radius: float,
) -> FloatArray:
    """Return signed centerline clearance above a horizontal thick-tube floor."""
    _validate_points(points)
    assert np.isfinite(floor_height)
    assert tube_radius > 0.0
    return np.asarray(points[:, 2] - floor_height - tube_radius, dtype=np.float64)


def floor_constraint_jacobian(vertex_count: int) -> csr_matrix:
    """Differentiate every floor slack with respect to flattened vertices."""
    assert vertex_count >= 1
    rows = np.arange(vertex_count, dtype=np.int64)
    columns = 3 * rows + 2
    return csr_matrix(
        (np.ones(vertex_count), (rows, columns)),
        shape=(vertex_count, 3 * vertex_count),
        dtype=np.float64,
    )


def contact_candidate_radius(
    configuration: GravityConstraintConfiguration,
    trust_radius: float,
) -> float:
    """Return the complete search radius for a bounded displacement step."""
    assert trust_radius > 0.0
    return (
        configuration.minimum_distance
        + configuration.contact_safety_margin
        + 2.0 * trust_radius
    )


def linearize_geometric_constraints(
    points: FloatArray,
    *,
    floor_height: float,
    tube_radius: float,
    contact_search_radius: float,
    configuration: GravityConstraintConfiguration,
    include_all_curvatures: bool = True,
) -> GravityGeometricLinearization:
    """Linearize every curvature/floor row and all nearby exact contacts."""
    _validate_points(points)
    assert contact_search_radius >= (
        configuration.minimum_distance + configuration.contact_safety_margin
    )
    geometric = active_constraint_jacobian(
        points,
        target_angle=configuration.maximum_angle,
        min_distance=configuration.minimum_distance,
        ignored_adjacent_edges=configuration.ignored_adjacent_edges,
        contact_activation_distance=(
            contact_search_radius - configuration.minimum_distance
        ),
        curvature_activation_angle=(
            configuration.maximum_angle
            if include_all_curvatures
            else configuration.curvature_safety_margin
        ),
    )
    contact_count = geometric.contact_count
    curvature_count = geometric.curvature_count
    contact_matrix = geometric.matrix[:contact_count]
    curvature_matrix = geometric.matrix[contact_count:]
    contact_slack = (
        geometric.slacks[:contact_count] - configuration.contact_safety_margin
    )
    curvature_slack = (
        geometric.slacks[contact_count:] - configuration.curvature_safety_margin
    )
    raw_floor_slack = floor_slacks(
        points,
        floor_height=floor_height,
        tube_radius=tube_radius,
    )
    floor_slack = raw_floor_slack - configuration.floor_safety_margin
    floor_matrix = floor_constraint_jacobian(len(points))
    matrix = csr_matrix(
        vstack((curvature_matrix, contact_matrix, floor_matrix), format="csr")
    )
    slacks = np.concatenate((curvature_slack, contact_slack, floor_slack))
    return GravityGeometricLinearization(
        matrix=matrix,
        slacks=np.asarray(slacks, dtype=np.float64),
        curvature_count=curvature_count,
        contact_count=contact_count,
        floor_count=len(points),
        contact_pairs=geometric.contact_pairs,
    )


def linearize_gravity_constraints(
    points: FloatArray,
    reference_lengths: FloatArray,
    *,
    floor_height: float,
    tube_radius: float,
    trust_radius: float,
    configuration: GravityConstraintConfiguration,
) -> GravityConstraintLinearization:
    """Assemble edge, geometry, floor, and trust rows for one gravity QP."""
    _validate_points(points)
    assert reference_lengths.shape == (len(points) - 1,)
    assert reference_lengths.dtype == np.float64
    assert np.all(reference_lengths > 0.0)
    assert trust_radius > 0.0
    edge_matrix = edge_constraint_jacobian(points, reference_lengths)
    edge_rhs = -edge_constraint_residuals(points, reference_lengths)
    geometric = linearize_geometric_constraints(
        points,
        floor_height=floor_height,
        tube_radius=tube_radius,
        contact_search_radius=contact_candidate_radius(configuration, trust_radius),
        configuration=configuration,
    )
    trust_matrix = eye(3 * len(points), format="csr", dtype=np.float64)
    matrix = csc_matrix(
        vstack((edge_matrix, geometric.matrix, trust_matrix), format="csc")
    )
    geometric_lower = -geometric.slacks
    trust_bound = trust_radius / np.sqrt(3.0)
    lower = np.concatenate(
        (
            edge_rhs,
            geometric_lower,
            np.full(3 * len(points), -trust_bound),
        )
    )
    upper = np.concatenate(
        (
            edge_rhs,
            np.full(len(geometric.slacks), np.inf),
            np.full(3 * len(points), trust_bound),
        )
    )
    return GravityConstraintLinearization(
        matrix=matrix,
        lower_bounds=np.asarray(lower, dtype=np.float64),
        upper_bounds=np.asarray(upper, dtype=np.float64),
        vertex_count=len(points),
        edge_count=len(reference_lengths),
        curvature_count=geometric.curvature_count,
        contact_count=geometric.contact_count,
        floor_count=geometric.floor_count,
        trust_count=3 * len(points),
        contact_pairs=geometric.contact_pairs,
    )


def evaluate_gravity_constraint_residuals(
    points: FloatArray,
    reference_lengths: FloatArray,
    *,
    floor_height: float,
    tube_radius: float,
    configuration: GravityConstraintConfiguration,
) -> GravityConstraintResiduals:
    """Evaluate exact nonlinear residuals without weakening standard distances."""
    _validate_points(points)
    search_radius = (
        configuration.minimum_distance + configuration.contact_safety_margin
    )
    points32 = np.asarray(points, dtype=np.float32)
    pairs = get_candidate_intersecting_edges(
        points32,
        min_distance=search_radius,
        n_ignored_adjacent_edges=configuration.ignored_adjacent_edges,
    )
    parameters = np.asarray(
        segment_segment_distance_parameters(points32, pairs), dtype=np.float64
    ).reshape((-1, 3))
    contact_slacks = (
        parameters[:, 0] - configuration.minimum_distance
        if len(parameters)
        else np.empty(0, dtype=np.float64)
    )
    angles = get_polyline_angles(points32)
    raw_floor_slacks = floor_slacks(
        points,
        floor_height=floor_height,
        tube_radius=tube_radius,
    )
    return GravityConstraintResiduals(
        maximum_edge_error=maximum_edge_length_error(points, reference_lengths),
        maximum_angle=float(np.max(angles, initial=np.float32(0.0))),
        minimum_contact_slack=float(np.min(contact_slacks, initial=np.inf)),
        separation_violation_count=int(np.count_nonzero(contact_slacks < 0.0)),
        minimum_floor_slack=float(np.min(raw_floor_slacks)),
    )


def _validate_points(points: FloatArray) -> None:
    assert points.dtype == np.float64
    assert points.ndim == 2 and points.shape[1] == 3
    assert len(points) >= 3
    assert np.all(np.isfinite(points))
