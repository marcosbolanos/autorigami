from __future__ import annotations

from typing import TypeAlias, TypedDict

import numpy as np
import numpy.typing as npt

from autorigami.types import EdgeIndex, Polyline, Vector3

SegmentSegmentDistance: TypeAlias = tuple[float, Vector3, Vector3]

class TangentPointEvaluation(TypedDict):
    energy: float
    repulsive_energy: float
    attractive_energy: float
    differential: npt.NDArray[np.float64]
    exact_pair_count: int
    approximated_cluster_count: int

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
def evaluate_tangent_point_exact(
    polyline: Polyline,
    target_distance: float,
    attraction_strength: float,
    local_exclusion_length: float,
) -> TangentPointEvaluation: ...
def evaluate_tangent_point_hierarchical(
    polyline: Polyline,
    target_distance: float,
    attraction_strength: float,
    local_exclusion_length: float,
    opening_angle: float = 0.25,
    leaf_size: int = 8,
) -> TangentPointEvaluation: ...
