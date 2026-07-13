import numpy as np

from autorigami.types import Polyline


def reparametrize_arc_length(polyline: Polyline, interval: float) -> Polyline:
    """
    Reparametrize a polyline so each point is at equal arc length
    NOTE: the last edge will be shorter than the rest
    """
    assert polyline.ndim == 2 and polyline.shape[1] == 3, (
        "polyline must have shape (n, 3)"
    )
    assert len(polyline) >= 2, "polyline must contain at least 2 points"
    assert interval > 0.0, "interval must be positive"

    # Get total arc length of the polyline
    edge_lengths = np.linalg.norm(polyline[1:] - polyline[:-1], axis=1)
    assert np.all(edge_lengths > 0.0), (
        "polyline must not contain repeated neighboring points"
    )
    cumulative_lengths = np.r_[0, np.cumsum(edge_lengths)]
    total_arc_length = cumulative_lengths[-1]

    # Define the array of target arc lengths at equal intervals
    # The last edge is concatenated to complete the curve, it can be shorter
    targets = np.r_[np.arange(0, total_arc_length, interval), total_arc_length]

    # Define coordinates for every target arc length
    indices = np.searchsorted(cumulative_lengths, targets, side="right") - 1
    indices = np.minimum(indices, len(edge_lengths) - 1)
    new_edges = (targets - cumulative_lengths[indices]) / edge_lengths[indices]
    output = (
        (1.0 - new_edges[:, None]) * polyline[indices]
        + new_edges[:, None] * polyline[indices + 1]
    )

    return np.array(output, dtype=np.float32)
