import numpy as np
import pytest

from src.curvepack.sdf import mask_to_sdf, polygon_to_mask, sample_interior_points


def _world_to_pixel(point: np.ndarray, origin: np.ndarray, h: float) -> tuple[int, int]:
    col = int(np.floor((point[0] - origin[0]) / h))
    row = int(np.floor((origin[1] - point[1]) / h))
    return row, col


@pytest.mark.parametrize(
    ("V", "inside", "outside"),
    [
        (
            np.array(
                [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=np.float32
            ),
            np.array([0.5, 0.5], dtype=np.float32),
            np.array([1.5, 1.5], dtype=np.float32),
        ),
        (
            np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]], dtype=np.float32),
            np.array([0.25, 0.25], dtype=np.float32),
            np.array([1.5, 1.5], dtype=np.float32),
        ),
    ],
)
def test_mask_to_sdf_sign_simple_polygons(
    V: np.ndarray, inside: np.ndarray, outside: np.ndarray
) -> None:
    mask, origin, h = polygon_to_mask(V, h=0.1, pad=1.0)
    sdf = mask_to_sdf(mask, h=h)

    in_row, in_col = _world_to_pixel(inside, origin, h)
    out_row, out_col = _world_to_pixel(outside, origin, h)

    assert mask[in_row, in_col]
    assert not mask[out_row, out_col]
    assert sdf[in_row, in_col] > 0.0
    assert sdf[out_row, out_col] < 0.0


def test_sample_interior_points_inside_square() -> None:
    V = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]], dtype=np.float32)
    mask, origin, h = polygon_to_mask(V, h=0.1, pad=1.0)
    rng = np.random.default_rng(0)

    Y = sample_interior_points(mask, origin, h, Q=200, rng=rng)
    sdf = mask_to_sdf(mask, h=h)
    cols = np.floor((Y[:, 0] - origin[0]) / h).astype(int)
    rows = np.floor((origin[1] - Y[:, 1]) / h).astype(int)

    assert rows.min() >= 0
    assert cols.min() >= 0
    assert rows.max() < mask.shape[0]
    assert cols.max() < mask.shape[1]
    assert bool(np.all(mask[rows, cols]))
    assert bool(np.all(sdf[rows, cols] > 0.0))
