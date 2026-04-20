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
) -> tuple[PiecewiseHermite, GeneratorRunData]: ...
def validate_piecewise_curve_curvature(
    piecewise_hermite: PiecewiseHermite,
    max_curvature: float,
    curvature_tolerance: float,
) -> bool: ...
