from __future__ import annotations

from typing import Literal, TypeAlias, overload

from autorigami.types import EdgeIndex, Polyline, Vector3

SegmentSegmentDistance: TypeAlias = tuple[float, Vector3, Vector3]
SegmentSegmentOptimizationDistance: TypeAlias = tuple[float, Vector3, Vector3, float, float]

@overload
def segment_segment_distance(
    polyline: Polyline,
    candidate_pairs: list[tuple[EdgeIndex, EdgeIndex]],
    include_optimization_data: Literal[False] = False,
) -> list[SegmentSegmentDistance]: ...

@overload
def segment_segment_distance(
    polyline: Polyline,
    candidate_pairs: list[tuple[EdgeIndex, EdgeIndex]],
    include_optimization_data: Literal[True],
) -> list[SegmentSegmentOptimizationDistance]: ...
