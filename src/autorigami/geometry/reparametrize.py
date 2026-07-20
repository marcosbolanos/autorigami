from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from autorigami.types import Polyline

FloatArray = npt.NDArray[np.float64]


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
    output = (1.0 - new_edges[:, None]) * polyline[indices] + new_edges[
        :, None
    ] * polyline[indices + 1]

    return np.array(output, dtype=np.float32)


def reparametrize_vertex_count(polyline: Polyline, vertex_count: int) -> Polyline:
    """Sample a polyline uniformly in arc length with a fixed vertex count."""
    assert polyline.ndim == 2 and polyline.shape[1] == 3, (
        "polyline must have shape (n, 3)"
    )
    assert len(polyline) >= 2, "polyline must contain at least 2 points"
    assert vertex_count >= 2, "vertex_count must be at least 2"
    edge_lengths = np.linalg.norm(polyline[1:] - polyline[:-1], axis=1)
    assert np.all(edge_lengths > 0.0), (
        "polyline must not contain repeated neighboring points"
    )
    cumulative_lengths = np.r_[0.0, np.cumsum(edge_lengths)]
    targets = np.linspace(0.0, cumulative_lengths[-1], vertex_count)
    indices = np.searchsorted(cumulative_lengths, targets, side="right") - 1
    indices = np.clip(indices, 0, len(edge_lengths) - 1)
    fractions = (targets - cumulative_lengths[indices]) / edge_lengths[indices]
    output = (1.0 - fractions[:, None]) * polyline[indices] + fractions[
        :, None
    ] * polyline[indices + 1]
    return np.asarray(output, dtype=np.float32)


def validate_arc_length_sampling(
    polyline: Polyline | FloatArray,
    *,
    interval: float,
    tolerance: float = 1e-3,
) -> None:
    """Validate uniform arc sampling with one permitted terminal remainder."""
    assert interval > 0.0
    assert tolerance > 0.0
    points = np.asarray(polyline, dtype=np.float64)
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    assert len(lengths) >= 1
    if np.max(np.abs(lengths[:-1] - interval), initial=0.0) > tolerance:
        raise ValueError("polyline is not sampled at the requested arc-length interval")
    if not 0.0 < lengths[-1] <= interval + tolerance:
        raise ValueError("terminal edge must be a positive sampling remainder")


@dataclass(frozen=True)
class ArcLengthProjection:
    """Project deformations back to a fixed-length, fixed-size centerline."""

    vertex_count: int
    total_length: float
    barycenter: FloatArray
    sampling_interval: float

    @classmethod
    def from_polyline(
        cls,
        polyline: Polyline | FloatArray,
        sampling_interval: float,
    ) -> ArcLengthProjection:
        points = np.asarray(polyline, dtype=np.float64)
        return cls(
            vertex_count=len(points),
            total_length=float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum()),
            barycenter=np.mean(points, axis=0),
            sampling_interval=sampling_interval,
        )

    def project(self, polyline: Polyline | FloatArray) -> FloatArray:
        """Restore length, uniform sampling, vertex count, and barycenter."""
        projected = np.asarray(polyline, dtype=np.float64)
        for _ in range(3):
            current_length = float(
                np.linalg.norm(np.diff(projected, axis=0), axis=1).sum()
            )
            center = np.mean(projected, axis=0)
            projected = center + (self.total_length / current_length) * (
                projected - center
            )
            projected = np.asarray(
                reparametrize_vertex_count(
                    np.asarray(projected, dtype=np.float32),
                    self.vertex_count,
                ),
                dtype=np.float64,
            )
        return projected + self.barycenter - np.mean(projected, axis=0)

    def validate(self, polyline: Polyline | FloatArray) -> None:
        """Raise when a projected centerline changed its chain geometry."""
        points = np.asarray(polyline, dtype=np.float64)
        if len(points) != self.vertex_count:
            raise ValueError("arc-length projection changed the vertex count")
        validate_arc_length_sampling(points, interval=self.sampling_interval)
        total_length = float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
        if abs(total_length - self.total_length) > 0.1:
            raise ValueError("arc-length projection changed the total length")
