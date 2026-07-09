from __future__ import annotations

from typing import TypeAlias

from autorigami.types import Vector3

Edge: TypeAlias = tuple[Vector3, Vector3]
SegmentSegmentDistance: TypeAlias = tuple[float, Vector3, Vector3]

def segment_segment_distance(
    candidate_pairs: list[tuple[Edge, Edge]],
) -> list[SegmentSegmentDistance]: ...
