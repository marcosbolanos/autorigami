from __future__ import annotations

from typing import TypeAlias

import jax
from jax import Array
import jax.numpy as jnp
from beartype import beartype
from jaxtyping import Float, jaxtyped

PointsND: TypeAlias = Float[Array, "N D"]
TangentsND: TypeAlias = Float[Array, "N D"]
WeightsN: TypeAlias = Float[Array, "N"]
Scalar: TypeAlias = Float[Array, ""]


@jaxtyped(typechecker=beartype)
def tangents_and_weights(
    x: PointsND,
    *,
    closed: bool = False,
    eps: float = 1e-12,
) -> tuple[TangentsND, WeightsN]:
    """Approximate unit tangents and arc-length weights at polyline vertices.

    x: (N, D) vertices.

    Returns:
      t: (N, D) approximate unit tangents.
      w: (N,) vertex weights ~ local arc-length measure.
    """

    if closed:
        xp = jnp.concatenate([x[-1:], x, x[:1]], axis=0)  # (N+2, D)
        d_prev = xp[1:-1] - xp[:-2]
        d_next = xp[2:] - xp[1:-1]
    else:
        d_prev = jnp.concatenate([x[1:2] - x[:1], x[1:] - x[:-1]], axis=0)
        d_next = jnp.concatenate([x[1:] - x[:-1], x[-1:] - x[-2:-1]], axis=0)

    t_raw = d_prev + d_next
    t_norm = jnp.sqrt(jnp.sum(t_raw * t_raw, axis=-1, keepdims=True) + (eps * eps))
    t = t_raw / t_norm

    seg = x[1:] - x[:-1]
    seglen = jnp.sqrt(jnp.sum(seg * seg, axis=-1) + (eps * eps))
    if closed:
        closing_vec = x[:1] - x[-1:]
        closing = jnp.sqrt(jnp.sum(closing_vec * closing_vec, axis=-1) + (eps * eps))
        seglen = jnp.concatenate([seglen, closing], axis=0)
        w = 0.5 * (seglen + jnp.roll(seglen, 1))
    else:
        w = jnp.empty((x.shape[0],), dtype=x.dtype)
        w = w.at[1:-1].set(0.5 * (seglen[:-1] + seglen[1:]))
        w = w.at[0].set(seglen[0])
        w = w.at[-1].set(seglen[-1])
    return t, w


@jaxtyped(typechecker=beartype)
def tangent_point_energy_full(
    x: PointsND,
    *,
    closed: bool = False,
    ignore_k: int = 0,
    eps2: float = 1e-12,
) -> Scalar:
    """Tangent-point energy (vertex discretization).

    O(N^2) time and O(N^2) memory.

    ignore_k masks pairs with |i-j|<=ignore_k (and wrap-around if closed).
    eps2 is an additive regularizer for squared distances.
    """

    t, w = tangents_and_weights(x, closed=closed)

    N = x.shape[0]
    ii, jj = jnp.triu_indices(N, k=1)
    keep = None
    if ignore_k > 0:
        dist = jj - ii
        if closed:
            dist = jnp.minimum(dist, N - dist)
        keep = dist > ignore_k

    d = x[ii, :] - x[jj, :]  # (P, D)
    r2 = jnp.sum(d * d, axis=-1) + eps2  # (P,)
    inv_r4 = 1.0 / (r2 * r2)

    td_i = jnp.sum(t[ii, :] * d, axis=-1)
    td_j = jnp.sum(t[jj, :] * d, axis=-1)
    num_i = r2 - td_i * td_i
    num_j = r2 - td_j * td_j

    ww = w[ii] * w[jj]
    contrib = ww * (num_i + num_j) * inv_r4
    if keep is not None:
        contrib = jnp.where(keep, contrib, jnp.zeros_like(contrib))
    return jnp.sum(contrib)


@jaxtyped(typechecker=beartype)
def tangent_point_energy_curves(
    X: Float[Array, "C M 2"],
    *,
    closed: bool = False,
    ignore_k: int = 0,
    eps2: float = 1e-12,
) -> Scalar:
    """Sum of tangent-point energies over a batch of curves."""

    def per_curve(x: Float[Array, "M 2"]) -> Scalar:
        return tangent_point_energy_full(
            x,
            closed=closed,
            ignore_k=ignore_k,
            eps2=eps2,
        )

    return jnp.sum(jax.vmap(per_curve)(X))
