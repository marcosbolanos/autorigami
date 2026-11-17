from __future__ import annotations

from typing import TypeAlias

import numpy as np
import numpy.typing as npt

from autorigami.types import EdgeIndex, Polyline, Vector3

SegmentSegmentDistance: TypeAlias = tuple[float, Vector3, Vector3]

def apply_separation_correction(
    polyline: Polyline,
    passes: int,
    candidate_pairs: list[tuple[EdgeIndex, EdgeIndex]],
    min_distance: float,
    fixed_step: float = 0.0,
    coordinate_mask: tuple[bool, bool, bool] = (True, True, True),
    reverse_order: bool = False,
) -> tuple[Polyline, int]: ...
def segment_segment_distance(
    polyline: Polyline,
    candidate_pairs: list[tuple[EdgeIndex, EdgeIndex]],
) -> list[SegmentSegmentDistance]: ...
def segment_segment_distance_parameters(
    polyline: Polyline,
    candidate_pairs: list[tuple[EdgeIndex, EdgeIndex]]
    | tuple[tuple[EdgeIndex, EdgeIndex], ...],
) -> npt.NDArray[np.float32]: ...
def find_close_edge_pairs(
    polyline: Polyline,
    max_distance: float,
    ignored_adjacent_edges: int,
    leaf_size: int = 16,
) -> list[tuple[EdgeIndex, EdgeIndex]]: ...

class AxialGravitySimulation:
    def __init__(
        self,
        *,
        points: Polyline,
        pinned: npt.NDArray[np.bool_],
        curve_indices: npt.NDArray[np.uint32],
        collision_radius: float,
        ignored_adjacent_vertices: int,
        tension_stiffness: float,
        tension_damping: float,
        floor_height: float,
        gravity: float,
        linear_damping: float,
        maximum_depenetration_velocity: float,
        maximum_linear_velocity: float,
        position_iterations: int,
        velocity_iterations: int,
        thread_count: int,
    ) -> None: ...
    def step(self, step_count: int, time_step: float) -> None: ...
    def set_gravity(self, gravity: float) -> None: ...
    def points(self) -> Polyline: ...
    def maximum_speed(self) -> float: ...
