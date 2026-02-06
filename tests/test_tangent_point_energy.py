import numpy as np
import jax
import jax.numpy as jnp

from src.curvepack.tangent_point.tangent_point_energy import (
    tangent_point_energy_full,
    tangent_point_energy_curves,
)


def test_tangent_point_energy_straight_line_near_zero() -> None:
    n = 16
    x = jnp.stack(
        [
            jnp.linspace(0.0, 1.0, n, dtype=jnp.float32),
            jnp.zeros((n,), dtype=jnp.float32),
        ],
        axis=1,
    )
    E = tangent_point_energy_full(x, closed=False, ignore_k=2, eps2=1e-12)
    assert np.isfinite(float(E))
    assert float(E) >= 0.0
    assert float(E) < 1e-4


def test_tangent_point_energy_bent_greater_than_straight() -> None:
    x_straight = jnp.array(
        [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0]],
        dtype=jnp.float32,
    )
    x_bent = jnp.array(
        [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [2.0, 1.0], [2.0, 2.0]],
        dtype=jnp.float32,
    )
    E0 = tangent_point_energy_full(x_straight, closed=False, ignore_k=1)
    E1 = tangent_point_energy_full(x_bent, closed=False, ignore_k=1)
    assert np.isfinite(float(E0))
    assert np.isfinite(float(E1))
    assert float(E1) > float(E0) + 1e-6


def test_tangent_point_energy_grad_finite() -> None:
    x = jnp.array(
        [[0.0, 0.0], [1.0, 0.2], [2.0, 0.0], [3.0, -0.1], [4.0, 0.0]],
        dtype=jnp.float32,
    )

    def energy_fn(xx: jax.Array) -> jax.Array:
        return tangent_point_energy_full(xx, closed=False, ignore_k=1)

    g = jax.grad(energy_fn)(x)
    assert g.shape == x.shape
    assert np.isfinite(np.array(g)).all()


def test_tangent_point_energy_curves_matches_sum() -> None:
    x0 = jnp.array(
        [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]],
        dtype=jnp.float32,
    )
    x1 = jnp.array(
        [[0.0, 0.0], [1.0, 0.2], [2.0, 0.0], [3.0, -0.2]],
        dtype=jnp.float32,
    )
    X = jnp.stack([x0, x1], axis=0)
    E_sum = tangent_point_energy_full(x0, ignore_k=1) + tangent_point_energy_full(
        x1, ignore_k=1
    )
    E_batch = tangent_point_energy_curves(X, ignore_k=1)
    np.testing.assert_allclose(float(E_batch), float(E_sum), rtol=1e-6, atol=1e-6)
