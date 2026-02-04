import numpy as np
import jax.numpy as jnp

from src.curvepack.loss import (
    inside_loss,
    curvature_loss,
    separation_loss,
    fill_reward,
)


def test_inside_loss_sanity_and_distance() -> None:
    X = jnp.array([[[0.5, 0.5]]], dtype=jnp.float32)
    r = jnp.array([1.0], dtype=jnp.float32)
    origin = jnp.array([0.0, 1.0], dtype=jnp.float32)
    h = 1.0

    sdf_near = jnp.full((2, 2), 0.25, dtype=jnp.float32)
    sdf_far = jnp.full((2, 2), 2.0, dtype=jnp.float32)

    loss_near = inside_loss(X, sdf_near, origin, h, r)
    loss_far = inside_loss(X, sdf_far, origin, h, r)

    assert np.isfinite(float(loss_near))
    assert np.isfinite(float(loss_far))
    assert float(loss_near) >= 0.0
    assert float(loss_far) >= 0.0
    assert float(loss_far) < float(loss_near)


def test_curvature_loss_sanity_and_bend() -> None:
    X_straight = jnp.array([[[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]], dtype=jnp.float32)
    X_bent = jnp.array([[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]], dtype=jnp.float32)
    Rmin = 2.0

    loss_straight = curvature_loss(X_straight, Rmin)
    loss_bent = curvature_loss(X_bent, Rmin)

    assert np.isfinite(float(loss_straight))
    assert np.isfinite(float(loss_bent))
    assert float(loss_straight) >= 0.0
    assert float(loss_bent) >= 0.0
    assert float(loss_bent) > float(loss_straight)


def test_separation_loss_sanity_and_distance() -> None:
    X_near = jnp.array(
        [
            [[0.0, 0.0], [1.0, 0.0]],
            [[0.0, 0.2], [1.0, 0.2]],
        ],
        dtype=jnp.float32,
    )
    X_far = jnp.array(
        [
            [[0.0, 0.0], [1.0, 0.0]],
            [[0.0, 2.0], [1.0, 2.0]],
        ],
        dtype=jnp.float32,
    )
    r = jnp.array([0.3, 0.3], dtype=jnp.float32)
    pairs = (
        jnp.array([0], dtype=jnp.int32),
        jnp.array([0], dtype=jnp.int32),
        jnp.array([1], dtype=jnp.int32),
        jnp.array([0], dtype=jnp.int32),
    )

    loss_near = separation_loss(X_near, r, pairs)
    loss_far = separation_loss(X_far, r, pairs)

    assert np.isfinite(float(loss_near))
    assert np.isfinite(float(loss_far))
    assert float(loss_near) >= 0.0
    assert float(loss_far) >= 0.0
    assert float(loss_far) < float(loss_near)


def test_fill_reward_sanity_and_distance() -> None:
    X_near = jnp.array([[[0.0, 0.0], [2.0, 0.0]]], dtype=jnp.float32)
    X_far = jnp.array([[[0.0, 2.0], [2.0, 2.0]]], dtype=jnp.float32)
    Y = jnp.array([[0.5, 0.0], [1.5, 0.0]], dtype=jnp.float32)

    loss_near = fill_reward(X_near, Y, r_fill=0.5, tau=0.5)
    loss_far = fill_reward(X_far, Y, r_fill=0.5, tau=0.5)

    assert np.isfinite(float(loss_near))
    assert np.isfinite(float(loss_far))
    assert float(loss_near) <= 0.0
    assert float(loss_far) <= 0.0
    assert float(loss_near) < float(loss_far)
