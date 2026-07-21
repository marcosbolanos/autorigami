import inspect
from typing import cast

import numpy as np
from scipy.optimize import brentq

from autorigami.geometry.curvature import get_polyline_angles
from autorigami.geometry.reparametrize import (
    ArcLengthProjection,
    validate_arc_length_sampling,
)
from autorigami.geometry.separation import check_self_intersections
from autorigami.optimization.workloads import optimize_separation


def test_optimize_separation_uses_dna_geometry_defaults() -> None:
    parameters = inspect.signature(optimize_separation).parameters

    assert parameters["edge_length"].default == 0.34
    assert parameters["target_angle"].default == np.deg2rad(3.25)
    assert parameters["min_distance"].default == 2.6
    assert parameters["local_exclusion_length"].default == 34.0
    assert parameters["steps"].default == 5
    assert parameters["maximum_vertex_step"].default == 0.02


def test_arc_length_sampling_allows_terminal_remainder() -> None:
    polyline = np.array(
        [[0.0, 0.0, 0.0], [0.34, 0.0, 0.0], [0.5, 0.0, 0.0]],
        dtype=np.float32,
    )

    validate_arc_length_sampling(polyline, interval=0.34)


def test_arc_length_projection_is_identity_on_its_reference_curve() -> None:
    polyline = np.array(
        [[0.0, 0.0, 0.0], [0.34, 0.0, 0.0], [0.68, 0.0, 0.0], [0.8, 0.0, 0.0]],
        dtype=np.float32,
    )

    projected = ArcLengthProjection.from_polyline(polyline, 0.34).project(polyline)

    np.testing.assert_allclose(projected, polyline, atol=1e-10)


def test_optimize_separation_attracts_without_violating_constraints() -> None:
    polyline = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [2.0, 1.0, 0.0],
            [2.0, 2.0, 0.0],
            [1.0, 2.0, 0.0],
            [0.0, 2.0, 0.0],
        ],
        dtype=np.float32,
    )

    result = optimize_separation(
        polyline,
        target_angle=np.pi / 2.0 + 0.001,
        edge_length=1.0,
        min_distance=1.0,
        local_exclusion_length=2.0,
        steps=15,
        maximum_vertex_step=0.05,
    )

    assert result.iterations
    assert result.iterations[-1].energy < result.iterations[0].energy
    assert result.iterations[-1].minimum_separation >= 1.0
    assert result.iterations[-1].maximum_angle <= np.pi / 2.0 + 0.00101
    assert result.iterations[-1].maximum_edge_length_error <= 1e-3
    assert result.iterations[0].active_curvature_count == 2
    assert result.iterations[0].kkt_iterations > 0
    assert result.iterations[0].kkt_residual < 1e-6


def test_large_curve_uses_the_only_production_path() -> None:
    radius = 7.0
    pitch = 3.0
    edge_length = 0.34
    rise_per_radian = pitch / (2.0 * np.pi)
    angle_step = cast(
        float,
        brentq(
            lambda angle: (
                np.sqrt(
                    (2.0 * radius * np.sin(0.5 * angle)) ** 2
                    + (rise_per_radian * angle) ** 2
                )
                - edge_length
            ),
            1e-3,
            0.2,
        ),
    )
    parameters = np.arange(260) * angle_step
    polyline = np.column_stack(
        (
            radius * np.cos(parameters),
            radius * np.sin(parameters),
            rise_per_radian * parameters,
        )
    ).astype(np.float32)

    result = optimize_separation(
        polyline,
        steps=1,
        maximum_vertex_step=0.01,
    )

    assert result.termination == "step_limit"
    assert len(result.iterations) == 1
    assert result.iterations[0].approximated_cluster_count > 0
    lengths = np.linalg.norm(np.diff(result.polyline, axis=0), axis=1)
    assert np.max(np.abs(lengths - edge_length)) <= 2e-4
    assert np.max(get_polyline_angles(result.polyline)) <= np.deg2rad(3.25) + 2e-4
    assert check_self_intersections(result.polyline, 2.6, 100)["edges"] is None
