from typing import TypedDict

from scipy.spatial import KDTree
import numpy as np

from autorigami._native import segment_segment_distance
from autorigami.types import EdgeIndex, Polyline


class SelfIntersectionCheckResult(TypedDict):
    edges: list[tuple[EdgeIndex, EdgeIndex]] | None
    distances: list[float] | None


def check_self_intersections(
    polyline: Polyline,
    min_euclid_distance: float,
    n_ignored_adjacent_edges: int = 1
) -> SelfIntersectionCheckResult:
    """
    Check polyline for self-intersections, return indexes and pairwise distances for culprit edges
    
    """
    candidate_edges = get_candidate_intersecting_edges(
        polyline,
        min_distance=min_euclid_distance,
        n_ignored_adjacent_edges=n_ignored_adjacent_edges
    )
    distances = segment_segment_distance(polyline, candidate_edges)
    culprit_edges_and_distances = [
        (edge, distance)
        for edge, (distance, _, _) in zip(candidate_edges, distances, strict=True)
        if distance < min_euclid_distance
    ]
    if not culprit_edges_and_distances:
        return {"edges": None, "distances": None}

    return {
        "edges": [edge for edge, _ in culprit_edges_and_distances],
        "distances": [distance for _, distance in culprit_edges_and_distances],
    }

# Get pairs that might self intersect
def get_candidate_intersecting_edges(
    polyline: Polyline,
    min_distance: float,
    n_ignored_adjacent_edges: int=1
) -> list[tuple[EdgeIndex, EdgeIndex]]:
    """
    Filter a Polyline for edges that might self-intersect using a KDTree
    This reduces the amount of pairs to test from n² to around nlogn

    Inputs:
    polyline: Polyline object to search
    min_distance: Distance under which the curve is considered to self-intersect
    n_ignored_adjacent_edges: how many edges around each edge should be ingored during search

    Outputs:
    A list of edge index tuples (i, j), where edge index comes from the polyline's ith and i+1th vertices
    """
    # Find segment midpoints and half-lengths
    starts = polyline[:-1]
    ends = polyline[1:]
    midpoints = 0.5 * (starts + ends)
    half_lengths = 0.5 * np.linalg.norm(ends - starts, axis=1)

    # A KDtree efficiently finds all midpoints within a radius of each other
    # Search radius compensates for half-lengths to guarantee finding all violating edges
    search_radius = 2 * half_lengths.max() + 2 * min_distance
    tree = KDTree(midpoints)
    candidate_pairs = tree.query_pairs(search_radius)

    # Remove duplicate and neighboring edges
    candidate_pairs = [(i, j) for i, j in candidate_pairs if abs(i - j) > n_ignored_adjacent_edges]

    return candidate_pairs
