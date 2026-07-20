import numpy as np
from scipy.fft import dct

from autorigami.optimization.fractional_sobolev import (
    FractionalSobolevPreconditioner,
)


def test_fractional_preconditioner_smooths_high_frequencies_more() -> None:
    polyline = np.column_stack(
        (np.arange(8, dtype=np.float32), np.zeros((8, 2), dtype=np.float32))
    )
    low_frequency = np.zeros_like(polyline, dtype=np.float64)
    low_frequency[:, 1] = np.cos(np.pi * (np.arange(8) + 0.5) / 8)
    high_frequency = np.zeros_like(polyline, dtype=np.float64)
    high_frequency[:, 1] = np.cos(7 * np.pi * (np.arange(8) + 0.5) / 8)
    preconditioner = FractionalSobolevPreconditioner(polyline)

    low_result = preconditioner.apply_inverse(low_frequency)
    high_result = preconditioner.apply_inverse(high_frequency)

    assert np.linalg.norm(low_result) > np.linalg.norm(high_result)
    assert np.argmax(np.abs(dct(low_result[:, 1], norm="ortho"))) == 1
    assert np.argmax(np.abs(dct(high_result[:, 1], norm="ortho"))) == 7


def test_fractional_preconditioner_preserves_coordinate_independence() -> None:
    polyline = np.column_stack(
        (np.arange(6, dtype=np.float32), np.zeros((6, 2), dtype=np.float32))
    )
    differential = np.zeros_like(polyline, dtype=np.float64)
    differential[:, 2] = np.linspace(-1.0, 1.0, len(polyline))

    result = FractionalSobolevPreconditioner(polyline).apply_inverse(differential)

    np.testing.assert_allclose(result[:, :2], 0.0, atol=1e-12)
    assert np.linalg.norm(result[:, 2]) > 0.0
