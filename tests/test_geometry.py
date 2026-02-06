import numpy as np
import jax.numpy as jnp

from src.curvepack.geometry import (
    bilinear_sample_sdf,
    squared_hinge,
    discrete_curvature,
    point_segment_dist2,
    segseg_dist2,
    polyline_length,
    curves_length,
)


def test_bilinear_sample_sdf_center() -> None:
    sdf = jnp.array([[0.0, 1.0], [2.0, 3.0]], dtype=jnp.float32)
    origin = jnp.array([0.0, 1.0], dtype=jnp.float32)
    val = bilinear_sample_sdf(
        sdf, origin, 1.0, jnp.array([0.5, 0.5], dtype=jnp.float32)
    )
    assert np.isclose(float(val), 1.5)


def test_squared_hinge() -> None:
    z = jnp.array([-1.0, 0.0, 2.0], dtype=jnp.float32)
    out = squared_hinge(z)
    np.testing.assert_allclose(
        np.array(out), np.array([0.0, 0.0, 4.0], dtype=np.float32)
    )


def test_discrete_curvature_straight_line() -> None:
    x = jnp.linspace(0.0, 4.0, 5, dtype=jnp.float32)
    y = jnp.zeros_like(x)
    points = jnp.stack([x, y], axis=1)
    X = points[None, ...]
    kappa = discrete_curvature(X)
    assert kappa.shape == (1, 3)
    assert float(jnp.max(jnp.abs(kappa))) < 1e-6


def test_point_segment_dist2_on_segment() -> None:
    a = jnp.array([0.0, 0.0], dtype=jnp.float32)
    b = jnp.array([2.0, 0.0], dtype=jnp.float32)
    p = jnp.array([1.0, 0.0], dtype=jnp.float32)
    d2 = point_segment_dist2(p, a, b)
    assert np.isclose(float(d2), 0.0)


def test_point_segment_dist2_off_segment() -> None:
    a = jnp.array([0.0, 0.0], dtype=jnp.float32)
    b = jnp.array([2.0, 0.0], dtype=jnp.float32)
    p = jnp.array([1.0, 3.0], dtype=jnp.float32)
    d2 = point_segment_dist2(p, a, b)
    assert np.isclose(float(d2), 9.0)


def test_segseg_dist2_intersecting() -> None:
    a = jnp.array([0.0, 0.0], dtype=jnp.float32)
    b = jnp.array([1.0, 0.0], dtype=jnp.float32)
    c = jnp.array([0.5, -1.0], dtype=jnp.float32)
    d = jnp.array([0.5, 1.0], dtype=jnp.float32)
    d2 = segseg_dist2(a, b, c, d)
    assert np.isclose(float(d2), 0.0)


def test_segseg_dist2_parallel_symmetry() -> None:
    a = jnp.array([0.0, 0.0], dtype=jnp.float32)
    b = jnp.array([1.0, 0.0], dtype=jnp.float32)
    c = jnp.array([0.0, 2.0], dtype=jnp.float32)
    d = jnp.array([1.0, 2.0], dtype=jnp.float32)
    d2 = segseg_dist2(a, b, c, d)
    d2_swap = segseg_dist2(c, d, a, b)
    assert np.isclose(float(d2), 4.0)
    assert np.isclose(float(d2), float(d2_swap))


def test_polyline_length_open() -> None:
    x = jnp.array([[0.0, 0.0], [3.0, 0.0], [3.0, 4.0]], dtype=jnp.float32)
    L = polyline_length(x, closed=False)
    assert np.isclose(float(L), 7.0)


def test_polyline_length_closed() -> None:
    x = jnp.array(
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
        dtype=jnp.float32,
    )
    L_open = polyline_length(x, closed=False)
    L_closed = polyline_length(x, closed=True)
    assert np.isclose(float(L_open), 3.0)
    assert np.isclose(float(L_closed), 4.0)


def test_curves_length_batch() -> None:
    X = jnp.array(
        [
            [[0.0, 0.0], [1.0, 0.0]],
            [[0.0, 0.0], [0.0, 2.0]],
        ],
        dtype=jnp.float32,
    )
    lengths = curves_length(X, closed=False)
    np.testing.assert_allclose(
        np.array(lengths), np.array([1.0, 2.0], dtype=np.float32)
    )
