from __future__ import annotations

from typing import TypeAlias

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
