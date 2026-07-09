from scipy.spatial import KDTree
import numpy as np

from autorigami._native import segment_segment_distance
from autorigami.types import Polyline


def validate_non_self_intersection(
    polyline: Polyline,
    min_euclid_distance: float,
    n_ignored_adjacent_edges: int = 1
):
    pairs = get_candidate_intersecting_pairs(
        polyline,
        min_distance=min_euclid_distance,
        n_ignored_adjacent_edges=n_ignored_adjacent_edges
    )
    return

# Get pairs that might self intersect
def get_candidate_intersecting_pairs(
    polyline: Polyline,
    min_distance: float,
    n_ignored_adjacent_edges: int=1
):
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
