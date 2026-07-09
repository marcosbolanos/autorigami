from __future__ import annotations

from typing import TypeAlias

from autorigami.types import EdgeIndexPair, Polyline, Vector3

SegmentSegmentDistance: TypeAlias = tuple[float, Vector3, Vector3]

def segment_segment_distance(
    polyline: Polyline,
    candidate_pairs: list[EdgeIndexPair],
) -> list[SegmentSegmentDistance]: ...
