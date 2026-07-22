from __future__ import annotations

# pyright: reportOptionalSubscript=false

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
from scipy.sparse import csr_matrix

from autorigami._native import segment_segment_distance_parameters
from autorigami.geometry.curvature import get_polyline_angles
from autorigami.geometry.separation import get_candidate_intersecting_edges
from autorigami.types import Polyline

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class EqualityConstraintSystem:
    """Values and sparse Jacobian for constraints defined by value == 0."""

    values: FloatArray
    jacobian: csr_matrix
    vertex_count: int

    def __post_init__(self) -> None:
        assert self.values.shape == (self.jacobian.shape[0],)
        assert self.jacobian.shape[1] == 3 * self.vertex_count


@dataclass(frozen=True)
class ActiveConstraintJacobian:
    """Sparse gradients of the currently active geometric constraints."""

    matrix: csr_matrix
    slacks: FloatArray
    vertex_count: int
    contact_count: int
    curvature_count: int
    contact_pairs: tuple[tuple[int, int], ...] = ()
    contact_parameters: FloatArray = field(
        default_factory=lambda: np.empty((0, 3), dtype=np.float64)
    )
    curvature_vertices: npt.NDArray[np.int64] = field(
        default_factory=lambda: np.empty(0, dtype=np.int64)
    )

    def __post_init__(self) -> None:
        assert self.matrix.shape == (self.constraint_count, 3 * self.vertex_count)
        assert self.slacks.shape == (self.constraint_count,)

    @property
    def constraint_count(self) -> int:
        return self.contact_count + self.curvature_count

    def apply(self, displacement: FloatArray) -> FloatArray:
        """Apply the constraint differential to a vertex displacement."""
        assert displacement.shape == (self.vertex_count, 3)
        return np.asarray(self.matrix @ displacement.reshape(-1), dtype=np.float64)

    def transpose_apply(self, multipliers: FloatArray) -> FloatArray:
        """Scatter constraint multipliers back to vertex vectors."""
        assert multipliers.shape == (self.constraint_count,)
        return np.asarray(self.matrix.T @ multipliers, dtype=np.float64).reshape(
            (-1, 3)
        )

    def select(self, indices: npt.NDArray[np.int64]) -> ActiveConstraintJacobian:
        """Return constraint rows selected by their original indices."""
        assert indices.ndim == 1
        assert np.all((0 <= indices) & (indices < self.constraint_count))
        contact_count = int(np.count_nonzero(indices < self.contact_count))
        contact_indices = indices[indices < self.contact_count]
        curvature_indices = indices[indices >= self.contact_count] - self.contact_count
        return ActiveConstraintJacobian(
            matrix=self.matrix[indices],
            slacks=self.slacks[indices],
            vertex_count=self.vertex_count,
            contact_count=contact_count,
            curvature_count=len(indices) - contact_count,
            contact_pairs=(
                tuple(self.contact_pairs[index] for index in contact_indices)
                if self.contact_pairs
                else ()
            ),
            contact_parameters=(
                self.contact_parameters[contact_indices]
                if len(self.contact_parameters)
                else np.empty((0, 3), dtype=np.float64)
            ),
            curvature_vertices=(
                self.curvature_vertices[curvature_indices]
                if len(self.curvature_vertices)
                else np.empty(0, dtype=np.int64)
            ),
        )


def active_constraint_jacobian(
    polyline: Polyline | FloatArray,
    *,
    target_angle: float,
    min_distance: float,
    ignored_adjacent_edges: int,
    contact_activation_distance: float,
    curvature_activation_angle: float,
) -> ActiveConstraintJacobian:
    """Linearize contacts and turning angles close to their hard bounds.

    Contact rows differentiate segment distance using the closest-point
    parameters returned by the native geometry kernel. Curvature rows
    differentiate ``target_angle - turning_angle`` analytically.
    """
    points = np.asarray(polyline, dtype=np.float64)
    assert points.ndim == 2 and points.shape[1] == 3
    assert len(points) >= 3
    assert 0.0 < target_angle < np.pi
    assert min_distance > 0.0
    assert ignored_adjacent_edges >= 1
    assert contact_activation_distance >= 0.0
    assert curvature_activation_angle >= 0.0

    contact_limit = min_distance + contact_activation_distance
    pairs = get_candidate_intersecting_edges(
        np.asarray(points, dtype=np.float32),
        min_distance=contact_limit,
        n_ignored_adjacent_edges=ignored_adjacent_edges,
    )
    parameter_data = segment_segment_distance_parameters(
        np.asarray(points, dtype=np.float32), pairs
    )
    active_contacts = [
        (pair, parameters)
        for pair, parameters in zip(pairs, parameter_data, strict=True)
        if float(parameters[0]) <= contact_limit
    ]

    angles = get_polyline_angles(np.asarray(points, dtype=np.float32))
    active_curvatures = np.flatnonzero(
        angles >= target_angle - curvature_activation_angle
    )
    row_count = len(active_contacts) + len(active_curvatures)
    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []
    slacks = np.empty(row_count, dtype=np.float64)

    def append_vertex_gradient(
        row: int,
        vertex: int,
        gradient: FloatArray,
    ) -> None:
        for coordinate in range(3):
            rows.append(row)
            columns.append(3 * vertex + coordinate)
            values.append(float(gradient[coordinate]))

    for row, ((first, second), parameters) in enumerate(active_contacts):
        distance, first_parameter, second_parameter = map(float, parameters)
        assert distance > 0.0, "active contact distance must be positive"
        slacks[row] = distance - min_distance
        first_point = (1.0 - first_parameter) * points[
            first
        ] + first_parameter * points[first + 1]
        second_point = (1.0 - second_parameter) * points[
            second
        ] + second_parameter * points[second + 1]
        normal = (first_point - second_point) / distance
        append_vertex_gradient(row, first, (1.0 - first_parameter) * normal)
        append_vertex_gradient(row, first + 1, first_parameter * normal)
        append_vertex_gradient(row, second, -(1.0 - second_parameter) * normal)
        append_vertex_gradient(row, second + 1, -second_parameter * normal)

    curvature_row_offset = len(active_contacts)
    for offset, angle_index in enumerate(active_curvatures):
        vertex = int(angle_index) + 1
        incoming = points[vertex] - points[vertex - 1]
        outgoing = points[vertex + 1] - points[vertex]
        incoming_length = float(np.linalg.norm(incoming))
        outgoing_length = float(np.linalg.norm(outgoing))
        assert incoming_length > 0.0 and outgoing_length > 0.0
        incoming_unit = incoming / incoming_length
        outgoing_unit = outgoing / outgoing_length
        cosine = float(np.clip(incoming_unit @ outgoing_unit, -1.0, 1.0))
        sine = float(np.sqrt(max(1e-12, 1.0 - cosine * cosine)))
        incoming_angle_gradient = -(outgoing_unit - cosine * incoming_unit) / (
            sine * incoming_length
        )
        outgoing_angle_gradient = -(incoming_unit - cosine * outgoing_unit) / (
            sine * outgoing_length
        )
        row = curvature_row_offset + offset
        slacks[row] = target_angle - float(angles[angle_index])
        # The feasible constraint is target_angle - angle >= 0.
        append_vertex_gradient(row, vertex - 1, incoming_angle_gradient)
        append_vertex_gradient(
            row,
            vertex,
            -incoming_angle_gradient + outgoing_angle_gradient,
        )
        append_vertex_gradient(row, vertex + 1, -outgoing_angle_gradient)

    matrix = csr_matrix(
        (values, (rows, columns)),
        shape=(row_count, 3 * len(points)),
        dtype=np.float64,
    )
    matrix.sum_duplicates()
    return ActiveConstraintJacobian(
        matrix=matrix,
        slacks=slacks,
        vertex_count=len(points),
        contact_count=len(active_contacts),
        curvature_count=len(active_curvatures),
        contact_pairs=tuple(pair for pair, _ in active_contacts),
        contact_parameters=np.asarray(
            [parameters for _, parameters in active_contacts], dtype=np.float64
        ).reshape((-1, 3)),
        curvature_vertices=np.asarray(active_curvatures + 1, dtype=np.int64),
    )
