from __future__ import annotations

from typing import TypeAlias

import jax
import jax.numpy as jnp
from jax import Array
from beartype import beartype
from jaxtyping import Float, Int, jaxtyped

from .geometry import (
    bilinear_sample_sdf,
    squared_hinge,
    discrete_curvature,
    segseg_dist2,
    point_segment_dist2,
)

Pairs: TypeAlias = tuple[
    Int[Array, "P"],
    Int[Array, "P"],
    Int[Array, "P"],
    Int[Array, "P"],
]


@jaxtyped(typechecker=beartype)
def inside_loss(
    X: Float[Array, "C M 2"],
    sdf: Float[Array, "H W"],
    origin: Float[Array, "2"],
    h: float,
    r: Float[Array, "C"],
    w: float = 1.0,
) -> Float[Array, ""]:
    """
    Enforce SDF(x) >= r for all curve sample points.
    X: (C,M,2)
    r: (C,) radii
    """
    sdf_x = bilinear_sample_sdf(sdf, origin, h, X)  # (C,M)
    # broadcast r over samples
    viol = r[:, None] - sdf_x
    return w * jnp.sum(squared_hinge(viol))


@jaxtyped(typechecker=beartype)
def curvature_loss(
    X: Float[Array, "C M 2"],
    Rmin: float,
    w: float = 1.0,
) -> Float[Array, ""]:
    kappa = discrete_curvature(X)  # (C,M-2)
    kappa_max = 1.0 / Rmin
    return w * jnp.sum(squared_hinge(kappa - kappa_max))


@jaxtyped(typechecker=beartype)
def separation_loss(
    X: Float[Array, "C M 2"],
    r: Float[Array, "C"],
    pairs: Pairs,
    delta: float = 0.0,
    w: float = 1.0,
) -> Float[Array, ""]:
    """
    X: (C,M,2), segments are [X[:,k], X[:,k+1]]
    pairs: (pair_i, pair_k, pair_j, pair_l) arrays length P
    """
    pair_i, pair_k, pair_j, pair_l = pairs
    a = X[pair_i, pair_k, :]
    b = X[pair_i, pair_k + 1, :]
    c = X[pair_j, pair_l, :]
    d = X[pair_j, pair_l + 1, :]

    dist = jnp.sqrt(segseg_dist2(a, b, c, d) + 1e-12)
    dmin = r[pair_i] + r[pair_j] + delta
    # Normalize by number of pairs to keep scale stable
    pair_count = jnp.maximum(pair_i.shape[0], 1)
    return w * jnp.sum(squared_hinge(dmin - dist)) / pair_count


@jaxtyped(typechecker=beartype)
def fill_loss(
    X: Float[Array, "C M 2"],
    Y: Float[Array, "Q 2"],
    r_fill: float,
    tau: float | Array = 1.0,
    w: float | Array = 1.0,
) -> Float[Array, ""]:
    """
    Penalty for missing coverage: points in Y should be within r_fill of some curve.
    X: (C,M,2) sampled points along curves
    Y: (Q,2) domain points inside polygon
    r_fill: scalar (typically same as tube radius)
    tau: softness for sigmoid
    We approximate curves by their segments. We compute min distance from each y to all segments.
    For moderate sizes, this is OK; for large sizes use neighbor restriction.
    """
    # segments: (C, M-1, 2) endpoints
    A = X[:, :-1, :]  # (C,S,2)
    B = X[:, 1:, :]  # (C,S,2)

    # Compute distances from Y (Q,2) to all segments (C,S)
    # We'll vectorize: for each y, compute dist2 to all segments.
    def dist2_to_all_segments(y: Float[Array, "2"]) -> Float[Array, "C S"]:
        # y: (2,)
        # broadcast over (C,S,2)
        d2 = point_segment_dist2(y[None, None, :], A, B)  # (C,S)
        return d2.reshape(-1)  # (C*S,)

    d2_all = jax.vmap(dist2_to_all_segments)(Y)  # (Q, C*S)
    # soft-min of distances (smoother than hard min)
    # softmin(d) = -tau_s * log sum exp(-d/tau_s)
    tau_s = 0.5 * tau
    d2_soft = -tau_s * jnp.log(jnp.sum(jnp.exp(-d2_all / tau_s), axis=1) + 1e-12)
    d_soft = jnp.sqrt(jnp.maximum(d2_soft, 0.0) + 1e-12)  # (Q,)

    # Smooth occupancy
    occ = jax.nn.sigmoid((r_fill - d_soft) / tau)
    # Penalty = 1 - mean occupancy
    return w * (1.0 - jnp.mean(occ))


@jaxtyped(typechecker=beartype)
def fill_fullness(
    X: Float[Array, "C M 2"],
    Y: Float[Array, "Q 2"],
    r_fill: float,
) -> Float[Array, ""]:
    """
    Fraction of Y points within r_fill of any curve segment.
    X: (C,M,2) sampled points along curves
    Y: (Q,2) domain points inside polygon
    r_fill: scalar radius in world units
    """
    A = X[:, :-1, :]
    B = X[:, 1:, :]

    def dist2_to_all_segments(y: Float[Array, "2"]) -> Float[Array, "C S"]:
        d2 = point_segment_dist2(y[None, None, :], A, B)
        return d2.reshape(-1)

    d2_all = jax.vmap(dist2_to_all_segments)(Y)
    d2_min = jnp.min(d2_all, axis=1)
    r2 = r_fill * r_fill
    return jnp.mean(d2_min <= r2)
