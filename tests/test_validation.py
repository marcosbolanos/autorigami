from __future__ import annotations

import numpy as np

from autorigami._native import (
    convert_trimesh_to_manifold_surface_mesh,
    piecewise_hermite_generator,
    validate_piecewise_curve_curvature,
)
from autorigami.parametrization import PiecewiseHermite


def _sample_trimesh_arrays() -> tuple[np.ndarray, np.ndarray]:
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    faces = np.array(
        [
            [0, 2, 1],
            [0, 1, 3],
            [1, 2, 3],
            [2, 0, 3],
        ],
        dtype=np.int64,
    )
    return vertices, faces


def test_validate_piecewise_curve_curvature_accepts_straight_segments() -> None:
    piecewise_hermite = PiecewiseHermite(
        points=np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
            ],
            dtype=float,
        ),
        tangents=np.array(
            [
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
            ],
            dtype=float,
        ),
    )

    assert validate_piecewise_curve_curvature(
        piecewise_hermite=piecewise_hermite,
        max_curvature=2.0,
        curvature_tolerance=0.01,
    )


def test_piecewise_hermite_generator_returns_dataclass() -> None:
    vertices, faces = _sample_trimesh_arrays()
    axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    generated, run_data = piecewise_hermite_generator(vertices, faces, axis)

    assert isinstance(generated, PiecewiseHermite)
    assert generated.points.shape == generated.tangents.shape
    assert generated.points.shape[0] >= 2
    assert generated.points.shape[1] == 3
    assert run_data["cpp_point_count"] == generated.points.shape[0]
    assert run_data["cpp_segment_count"] == generated.points.shape[0] - 1
    assert run_data["cpp_parameter_step"] == 1.0
    assert run_data["input_mesh_vertex_count"] == int(vertices.shape[0])
    assert run_data["input_mesh_face_count"] == int(faces.shape[0])
    assert run_data["input_axis_x"] == axis[0]
    assert run_data["input_axis_y"] == axis[1]
    assert run_data["input_axis_z"] == axis[2]
    assert validate_piecewise_curve_curvature(
        piecewise_hermite=generated,
        max_curvature=100.0,
        curvature_tolerance=1e-6,
    )


def test_convert_trimesh_to_manifold_surface_mesh() -> None:
    vertices, faces = _sample_trimesh_arrays()
    info = convert_trimesh_to_manifold_surface_mesh(vertices, faces)

    assert info["vertex_count"] == int(vertices.shape[0])
    assert info["face_count"] == int(faces.shape[0])
