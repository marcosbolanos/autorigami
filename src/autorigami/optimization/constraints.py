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
    parameter_data = np.asarray(
        segment_segment_distance_parameters(
            np.asarray(points, dtype=np.float32), pairs
        ),
        dtype=np.float64,
    ).reshape((-1, 3))
    pair_data = np.asarray(pairs, dtype=np.int64).reshape((-1, 2))
    active_contact_mask = parameter_data[:, 0] <= contact_limit
    active_parameters = parameter_data[active_contact_mask]
    active_pairs = pair_data[active_contact_mask]

    angles = get_polyline_angles(np.asarray(points, dtype=np.float32))
    active_curvatures = np.flatnonzero(
        angles >= target_angle - curvature_activation_angle
    )
    contact_count = len(active_pairs)
    row_count = contact_count + len(active_curvatures)
    slacks = np.empty(row_count, dtype=np.float64)

    contact_rows, contact_columns, contact_values = _contact_jacobian_entries(
        points,
        active_pairs,
        active_parameters,
    )
    slacks[:contact_count] = active_parameters[:, 0] - min_distance
    curvature_rows, curvature_columns, curvature_values = (
        _curvature_jacobian_entries(
            points,
            active_curvatures,
            row_offset=contact_count,
        )
    )
    slacks[contact_count:] = target_angle - angles[active_curvatures]
    rows = np.concatenate((contact_rows, curvature_rows))
    columns = np.concatenate((contact_columns, curvature_columns))
    values = np.concatenate((contact_values, curvature_values))

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
        contact_count=contact_count,
        curvature_count=len(active_curvatures),
        contact_pairs=tuple(map(tuple, active_pairs.tolist())),
        contact_parameters=active_parameters,
        curvature_vertices=np.asarray(active_curvatures + 1, dtype=np.int64),
    )


def _contact_jacobian_entries(
    points: FloatArray,
    pairs: npt.NDArray[np.int64],
    parameters: FloatArray,
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64], FloatArray]:
    """Vectorize four endpoint blocks for exact segment-distance rows."""
    assert pairs.shape == (len(parameters), 2)
    assert parameters.shape == (len(pairs), 3)
    if not len(pairs):
        empty_indices = np.empty(0, dtype=np.int64)
        return empty_indices, empty_indices, np.empty(0, dtype=np.float64)
    distance = parameters[:, 0]
    assert np.all(distance > 0.0), "active contact distance must be positive"
    first_parameter = parameters[:, 1]
    second_parameter = parameters[:, 2]
    first = pairs[:, 0]
    second = pairs[:, 1]
    first_point = (
        (1.0 - first_parameter[:, None]) * points[first]
        + first_parameter[:, None] * points[first + 1]
    )
    second_point = (
        (1.0 - second_parameter[:, None]) * points[second]
        + second_parameter[:, None] * points[second + 1]
    )
    normal = (first_point - second_point) / distance[:, None]
    gradients = np.stack(
        (
            (1.0 - first_parameter[:, None]) * normal,
            first_parameter[:, None] * normal,
            -(1.0 - second_parameter[:, None]) * normal,
            -second_parameter[:, None] * normal,
        ),
        axis=1,
    )
    vertices = np.stack((first, first + 1, second, second + 1), axis=1)
    rows = np.broadcast_to(
        np.arange(len(pairs), dtype=np.int64)[:, None, None],
        gradients.shape,
    )
    columns = 3 * vertices[:, :, None] + np.arange(3, dtype=np.int64)
    return rows.reshape(-1), columns.reshape(-1), gradients.reshape(-1)


def _curvature_jacobian_entries(
    points: FloatArray,
    angle_indices: npt.NDArray[np.int64],
    *,
    row_offset: int,
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64], FloatArray]:
    """Vectorize three vertex blocks for turning-angle slack rows."""
    assert angle_indices.ndim == 1
    assert row_offset >= 0
    if not len(angle_indices):
        empty_indices = np.empty(0, dtype=np.int64)
        return empty_indices, empty_indices, np.empty(0, dtype=np.float64)
    vertices = angle_indices + 1
    incoming = points[vertices] - points[vertices - 1]
    outgoing = points[vertices + 1] - points[vertices]
    incoming_length = np.linalg.norm(incoming, axis=1)
    outgoing_length = np.linalg.norm(outgoing, axis=1)
    assert np.all(incoming_length > 0.0) and np.all(outgoing_length > 0.0)
    incoming_unit = incoming / incoming_length[:, None]
    outgoing_unit = outgoing / outgoing_length[:, None]
    cosine = np.clip(np.sum(incoming_unit * outgoing_unit, axis=1), -1.0, 1.0)
    sine = np.sqrt(np.maximum(1e-12, 1.0 - cosine * cosine))
    incoming_gradient = -(
        outgoing_unit - cosine[:, None] * incoming_unit
    ) / (sine * incoming_length)[:, None]
    outgoing_gradient = -(
        incoming_unit - cosine[:, None] * outgoing_unit
    ) / (sine * outgoing_length)[:, None]
    gradients = np.stack(
        (
            incoming_gradient,
            -incoming_gradient + outgoing_gradient,
            -outgoing_gradient,
        ),
        axis=1,
    )
    gradient_vertices = np.stack(
        (vertices - 1, vertices, vertices + 1), axis=1
    )
    rows = np.broadcast_to(
        (row_offset + np.arange(len(vertices), dtype=np.int64))[:, None, None],
        gradients.shape,
    )
    columns = 3 * gradient_vertices[:, :, None] + np.arange(3, dtype=np.int64)
    return rows.reshape(-1), columns.reshape(-1), gradients.reshape(-1)
