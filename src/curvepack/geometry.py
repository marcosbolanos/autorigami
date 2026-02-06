from __future__ import annotations

from typing import cast

from jax import Array
import jax.numpy as jnp
from beartype import beartype
from jaxtyping import Float, jaxtyped


@jaxtyped(typechecker=beartype)
def bilinear_sample_sdf(
    sdf: Float[Array, "H W"],
    origin: Float[Array, "2"],
    h: float,
    X: Float[Array, "... 2"],
) -> Float[Array, "..."]:
    """
    sdf: (H,W) float32
    origin: (2,) where origin[0]=minx, origin[1]=maxy (top in world coords)
    h: pixel size
    X: (...,2) world coords
    Returns sdf(X): (...) via bilinear interpolation
    """
    H, W = sdf.shape

    # Convert world -> continuous pixel coordinates
    # px = (x - minx)/h
    # py = (maxy - y)/h
    px = (X[..., 0] - origin[0]) / h
    py = (origin[1] - X[..., 1]) / h

    x0 = jnp.floor(px).astype(jnp.int32)
    y0 = jnp.floor(py).astype(jnp.int32)
    x1 = x0 + 1
    y1 = y0 + 1

    # Clip
    x0c = jnp.clip(x0, 0, W - 1)
    x1c = jnp.clip(x1, 0, W - 1)
    y0c = jnp.clip(y0, 0, H - 1)
    y1c = jnp.clip(y1, 0, H - 1)

    # Weights
    wx = px - x0.astype(px.dtype)
    wy = py - y0.astype(py.dtype)

    v00 = sdf[y0c, x0c]
    v10 = sdf[y0c, x1c]
    v01 = sdf[y1c, x0c]
    v11 = sdf[y1c, x1c]

    v0 = v00 * (1 - wx) + v10 * wx
    v1 = v01 * (1 - wx) + v11 * wx
    v = v0 * (1 - wy) + v1 * wy
    return v


@jaxtyped(typechecker=beartype)
def squared_hinge(z: Float[Array, "..."]) -> Float[Array, "..."]:
    return jnp.square(jnp.maximum(0.0, z))


@jaxtyped(typechecker=beartype)
def discrete_curvature(
    X: Float[Array, "C M 2"],
    eps: float = 1e-9,
) -> Float[Array, "C M1"]:
    """
    X: (C, M, 2) polyline samples
    Returns kappa: (C, M-2)
    """
    u = X[:, 1:-1, :] - X[:, 0:-2, :]
    v = X[:, 2:, :] - X[:, 1:-1, :]

    nu = cast(Float[Array, "C M1"], jnp.linalg.norm(u, axis=-1) + eps)
    nv = cast(Float[Array, "C M1"], jnp.linalg.norm(v, axis=-1) + eps)
    cos_th = jnp.sum(u * v, axis=-1) / (nu * nv)
    cos_th = jnp.clip(cos_th, -1.0, 1.0)
    theta = jnp.arccos(cos_th)
    ds = 0.5 * (nu + nv)
    return theta / (ds + eps)


@jaxtyped(typechecker=beartype)
def point_segment_dist2(
    p: Float[Array, "... 2"],
    a: Float[Array, "... 2"],
    b: Float[Array, "... 2"],
    eps: float = 1e-9,
) -> Float[Array, "..."]:
    """
    p: (...,2)
    a,b: (...,2) broadcastable
    """
    ab = b - a
    t = jnp.sum((p - a) * ab, axis=-1) / (jnp.sum(ab * ab, axis=-1) + eps)
    t = jnp.clip(t, 0.0, 1.0)
    q = a + t[..., None] * ab
    d = p - q
    return jnp.sum(d * d, axis=-1)


@jaxtyped(typechecker=beartype)
def segseg_dist2(
    a: Float[Array, "... 2"],
    b: Float[Array, "... 2"],
    c: Float[Array, "... 2"],
    d: Float[Array, "... 2"],
    eps: float = 1e-9,
) -> Float[Array, "..."]:
    """
    a,b,c,d: (...,2)
    returns (...,) squared distance between segments
    """
    u = b - a
    v = d - c
    w0 = a - c
    A = jnp.sum(u * u, axis=-1) + eps
    B = jnp.sum(u * v, axis=-1)
    C = jnp.sum(v * v, axis=-1) + eps
    D = jnp.sum(u * w0, axis=-1)
    E = jnp.sum(v * w0, axis=-1)
    denom = A * C - B * B

    s = jnp.where(denom > eps, (B * E - C * D) / denom, 0.0)
    t = jnp.where(denom > eps, (A * E - B * D) / denom, 0.0)
    s = jnp.clip(s, 0.0, 1.0)
    t = jnp.clip(t, 0.0, 1.0)

    p = a + s[..., None] * u
    q = c + t[..., None] * v
    diff = p - q
    return jnp.sum(diff * diff, axis=-1)


@jaxtyped(typechecker=beartype)
def polyline_length(
    x: Float[Array, "M 2"],
    *,
    closed: bool = False,
    eps: float = 1e-12,
) -> Float[Array, ""]:
    """Polyline length in world units."""

    seg = x[1:, :] - x[:-1, :]
    seglen = jnp.linalg.norm(seg, axis=-1) + eps
    length = jnp.sum(seglen)
    if closed:
        length = length + jnp.linalg.norm(x[0, :] - x[-1, :])
    return length


@jaxtyped(typechecker=beartype)
def curves_length(
    X: Float[Array, "C M 2"],
    *,
    closed: bool = False,
    eps: float = 1e-12,
) -> Float[Array, "C"]:
    """Per-curve polyline length for X: (C, M, 2)."""

    seg = X[:, 1:, :] - X[:, :-1, :]
    seglen = jnp.linalg.norm(seg, axis=-1) + eps
    length = jnp.sum(seglen, axis=-1)
    if closed:
        closing = jnp.linalg.norm(X[:, 0, :] - X[:, -1, :], axis=-1)
        length = length + closing
    return length
