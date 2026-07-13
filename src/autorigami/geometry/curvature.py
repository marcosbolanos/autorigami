import numpy as np
from numpy.typing import NDArray

from autorigami.types import Polyline


def get_polyline_angles(polyline: Polyline) -> NDArray[np.float32]:
    """
    Returns an array of sclar angles for each inner vertex
    Length: len(polyline) - 2
    """
    vectors = polyline[1:] - polyline[:-1]
    # normalize all vectors so dot product gives us cosines
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

    # stream dot products across all consecutive vectors
    cosines = np.sum(vectors[:-1] * vectors[1:], axis=1)
    cosines = np.clip(cosines, -1.0, 1.0)

    angles = np.arccos(cosines)
    return angles
