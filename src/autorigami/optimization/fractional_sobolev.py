from __future__ import annotations

import numpy as np
import numpy.typing as npt
from scipy.fft import dct, idct

from autorigami.types import Polyline

FloatArray = npt.NDArray[np.float64]


class FractionalSobolevPreconditioner:
    """Apply an inverse fractional chain Laplacian componentwise.

    The cosine basis is the open-chain analogue of a Fourier basis. It gives
    the inexpensive Sobolev-smoothed descent direction used by separation
    optimization without assembling a dense metric matrix.
    """

    def __init__(
        self,
        polyline: Polyline | FloatArray,
        *,
        sigma: float = 0.75,
    ) -> None:
        assert polyline.ndim == 2 and polyline.shape[1] == 3
        assert len(polyline) >= 2
        assert 0.0 < sigma < 1.0
        points = np.asarray(polyline, dtype=np.float64)
        edge_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
        assert np.all(edge_lengths > 0.0)
        interval = float(np.median(edge_lengths))
        frequencies = np.arange(len(points), dtype=np.float64)
        laplacian_eigenvalues = (
            4.0 * np.sin(0.5 * np.pi * frequencies / len(points)) ** 2 / interval**2
        )
        self._eigenvalues = laplacian_eigenvalues ** (sigma + 1.0)
        self._eigenvalues[0] = max(1e-8, 1e-6 * self._eigenvalues[1])
        self._vertex_count = len(points)

    def apply_inverse(self, differential: FloatArray) -> FloatArray:
        """Return the Sobolev descent vector for a Euclidean differential."""
        assert differential.shape == (self._vertex_count, 3)
        coefficients = np.asarray(
            dct(differential, type=2, axis=0, norm="ortho"),
            dtype=np.float64,
        )
        coefficients /= self._eigenvalues[:, None]
        return np.asarray(
            idct(coefficients, type=2, axis=0, norm="ortho"),
            dtype=np.float64,
        )
