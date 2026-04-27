from __future__ import annotations

from typing import TypeAlias

from autorigami.parametrization import PiecewiseHermite
from autorigami.types import FloatArray, IndexArray

GeneratorRunData: TypeAlias = dict[str, int | float]
ManifoldMeshInfo: TypeAlias = dict[str, int]

def add(left: int, right: int) -> int: ...
def convert_trimesh_to_manifold_surface_mesh(
    vertices: FloatArray,
    faces: IndexArray,
) -> ManifoldMeshInfo: ...
def piecewise_hermite_generator(
    vertices: FloatArray,
    faces: IndexArray,
    axis_origin: FloatArray,
    axis_direction: FloatArray,
    spacing_world: float = 2.6,
    nonlocal_window_world: float = 4.1,
    max_curvature: float = 1e6,
    curvature_tolerance: float = 0.0,
    extension_step_world: float = 0.5,
    outer_iterations: int = 4,
) -> tuple[PiecewiseHermite, GeneratorRunData]: ...
def validate_polyline_nonlocal_distance(
    points: FloatArray,
    minimum_separation: float,
    nonlocal_window: float,
    stop_on_first_violation: bool = False,
) -> dict[str, int | float]: ...
def validate_piecewise_curve_curvature(
    piecewise_hermite: PiecewiseHermite,
    max_curvature: float,
    curvature_tolerance: float,
) -> bool: ...
