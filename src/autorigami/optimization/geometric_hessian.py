from __future__ import annotations

# pyright: reportAttributeAccessIssue=false, reportCallIssue=false

import autograd.numpy as anp
import numpy as np
import numpy.typing as npt
from autograd import hessian

from autorigami.optimization.constraints import ActiveConstraintJacobian

FloatArray = npt.NDArray[np.float64]


class ClosestFeatureChange(RuntimeError):
    """Raised when segment distance is not twice differentiable in one regime."""


def active_geometric_hessian_product(
    points: FloatArray,
    constraints: ActiveConstraintJacobian,
    multipliers: FloatArray,
    direction: FloatArray,
    *,
    feature_tolerance: float = 1e-5,
) -> FloatArray:
    """Apply exact AD Hessians of active distance and angle slacks.

    Segment distance is differentiated in its current closest-feature regime.
    A regime boundary or a direction crossing that boundary is reported
    explicitly because the classical Hessian is undefined there.
    """
    assert points.dtype == direction.dtype == np.float64
    assert points.shape == direction.shape == (constraints.vertex_count, 3)
    assert multipliers.shape == (constraints.constraint_count,)
    assert len(constraints.contact_pairs) == constraints.contact_count
    assert constraints.contact_parameters.shape == (constraints.contact_count, 3)
    assert constraints.curvature_vertices.shape == (constraints.curvature_count,)
    result = np.zeros_like(points)
    for row, ((first, second), parameters) in enumerate(
        zip(constraints.contact_pairs, constraints.contact_parameters, strict=True)
    ):
        local_indices = np.array([first, first + 1, second, second + 1])
        local_points = points[local_indices]
        local_direction = direction[local_indices]
        feature = _classify_feature(float(parameters[1]), float(parameters[2]), feature_tolerance)
        _assert_feature_stable(local_points, local_direction, feature, feature_tolerance)
        local_hessian = np.asarray(
            hessian(lambda values: _distance_in_feature(values, feature))(local_points.reshape(-1)),
            dtype=np.float64,
        )
        local_product = multipliers[row] * (local_hessian @ local_direction.reshape(-1))
        for local, vertex in enumerate(local_indices):
            result[vertex] += local_product.reshape((4, 3))[local]
    for offset, vertex_value in enumerate(constraints.curvature_vertices):
        vertex = int(vertex_value)
        local_indices = np.array([vertex - 1, vertex, vertex + 1])
        local_points = points[local_indices]
        local_direction = direction[local_indices]
        local_hessian = np.asarray(
            hessian(_negative_turning_angle)(local_points.reshape(-1)),
            dtype=np.float64,
        )
        local_product = multipliers[constraints.contact_count + offset] * (
            local_hessian @ local_direction.reshape(-1)
        )
        for local, index in enumerate(local_indices):
            result[index] += local_product.reshape((3, 3))[local]
    return result


def _classify_feature(first: float, second: float, tolerance: float) -> tuple[int, int]:
    def classify(value: float) -> int:
        if abs(value) <= tolerance:
            return 0
        if abs(value - 1.0) <= tolerance:
            return 1
        if tolerance < value < 1.0 - tolerance:
            return 2
        raise ClosestFeatureChange("closest point lies at an ambiguous feature boundary")

    return classify(first), classify(second)


def _distance_in_feature(values: npt.ArrayLike, feature: tuple[int, int]):
    points = anp.reshape(values, (4, 3))
    first_start, first_end, second_start, second_end = points
    first_edge = first_end - first_start
    second_edge = second_end - second_start
    first_kind, second_kind = feature
    if first_kind == 2 and second_kind == 2:
        offset = first_start - second_start
        system = anp.array(
            [
                [anp.dot(first_edge, first_edge), -anp.dot(first_edge, second_edge)],
                [-anp.dot(first_edge, second_edge), anp.dot(second_edge, second_edge)],
            ]
        )
        rhs = anp.array([-anp.dot(first_edge, offset), anp.dot(second_edge, offset)])
        parameters = anp.linalg.solve(system, rhs)
        first_parameter, second_parameter = parameters[0], parameters[1]
    elif first_kind == 2:
        second_parameter = float(second_kind)
        second_point = second_start + second_parameter * second_edge
        first_parameter = anp.dot(first_edge, second_point - first_start) / anp.dot(first_edge, first_edge)
    elif second_kind == 2:
        first_parameter = float(first_kind)
        first_point = first_start + first_parameter * first_edge
        second_parameter = anp.dot(second_edge, first_point - second_start) / anp.dot(second_edge, second_edge)
    else:
        first_parameter = float(first_kind)
        second_parameter = float(second_kind)
    separation = (
        first_start + first_parameter * first_edge
        - second_start - second_parameter * second_edge
    )
    return anp.sqrt(anp.dot(separation, separation))


def _negative_turning_angle(values: npt.ArrayLike):
    points = anp.reshape(values, (3, 3))
    incoming = points[1] - points[0]
    outgoing = points[2] - points[1]
    cosine = anp.dot(incoming, outgoing) / (
        anp.sqrt(anp.dot(incoming, incoming)) * anp.sqrt(anp.dot(outgoing, outgoing))
    )
    return -anp.arccos(cosine)


def _assert_feature_stable(
    points: FloatArray,
    direction: FloatArray,
    feature: tuple[int, int],
    tolerance: float,
) -> None:
    scale = max(1.0, float(np.max(np.linalg.norm(direction, axis=1))))
    for sign in (-1.0, 1.0):
        probe = points + sign * (1e-6 / scale) * direction
        parameters = _unclamped_parameters(probe)
        if _classify_feature(parameters[0], parameters[1], tolerance) != feature:
            raise ClosestFeatureChange("Hessian direction crosses a closest-feature boundary")


def _unclamped_parameters(points: FloatArray) -> tuple[float, float]:
    first_edge = points[1] - points[0]
    second_edge = points[3] - points[2]
    offset = points[0] - points[2]
    system = np.array(
        [
            [first_edge @ first_edge, -(first_edge @ second_edge)],
            [-(first_edge @ second_edge), second_edge @ second_edge],
        ]
    )
    if abs(np.linalg.det(system)) < 1e-14:
        raise ClosestFeatureChange("parallel closest segments have a non-unique feature")
    parameters = np.linalg.solve(
        system,
        np.array([-(first_edge @ offset), second_edge @ offset]),
    )
    return float(np.clip(parameters[0], 0.0, 1.0)), float(np.clip(parameters[1], 0.0, 1.0))
