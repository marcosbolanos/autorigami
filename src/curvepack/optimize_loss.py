from __future__ import annotations

import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Callable
from jaxtyping import jaxtyped

from .loss import inside_loss, curvature_loss, separation_loss, fill_loss
from .optimize_types import (
    JaxControlPoints,
    JaxRadii,
    JaxScalar,
    LossFn,
    NpBasisMatrix,
    NpOrigin,
    NpSamplePoints,
    NpSdfGrid,
    PairsJax,
)


@jaxtyped(typechecker=beartype)
def make_loss_fn(
    B: NpBasisMatrix,
    sdf: NpSdfGrid,
    origin: NpOrigin,
    h: float,
    Y: NpSamplePoints,
    Rmin: float,
    delta: float,
    r_fill: float,
    w_inside: float,
    w_curv: float,
    w_sep: float,
) -> LossFn:
    """
    Returns a JAX function loss(P, pairs, r, w_fill, tau_fill) -> scalar.
    """
    B_j = jnp.asarray(B)
    sdf_j = jnp.asarray(sdf)
    origin_j = jnp.asarray(origin)
    Y_j = jnp.asarray(Y)

    def loss(
        P: JaxControlPoints,
        pairs: PairsJax,
        r: JaxRadii,
        w_fill_t: JaxScalar,
        tau_fill_t: JaxScalar,
    ) -> JaxScalar:
        X = jnp.einsum("mn,cnd->cmd", B_j, P)

        Li = inside_loss(X, sdf_j, origin_j, h, r, w=w_inside)
        Lc = curvature_loss(X, Rmin, w=w_curv)
        Ls = separation_loss(X, r, pairs, delta=delta, w=w_sep)
        Lf = fill_loss(X, Y_j, r_fill=r_fill, tau=tau_fill_t, w=w_fill_t)

        return Li + Lc + Ls + Lf

    return loss


def make_loss_terms_fn(
    B: NpBasisMatrix,
    sdf: NpSdfGrid,
    origin: NpOrigin,
    h: float,
    Y: NpSamplePoints,
    Rmin: float,
    delta: float,
    r_fill: float,
    w_inside: float,
    w_curv: float,
    w_sep: float,
) -> Callable[
    [JaxControlPoints, PairsJax, JaxRadii, JaxScalar, JaxScalar],
    tuple[JaxScalar, ...],
]:
    B_j = jnp.asarray(B)
    sdf_j = jnp.asarray(sdf)
    origin_j = jnp.asarray(origin)
    Y_j = jnp.asarray(Y)

    def loss_terms(
        P: JaxControlPoints,
        pairs: PairsJax,
        r: JaxRadii,
        w_fill_t: JaxScalar,
        tau_fill_t: JaxScalar,
    ) -> tuple[JaxScalar, ...]:
        X = jnp.einsum("mn,cnd->cmd", B_j, P)
        Li = inside_loss(X, sdf_j, origin_j, h, r, w=w_inside)
        Lc = curvature_loss(X, Rmin, w=w_curv)
        Ls = separation_loss(X, r, pairs, delta=delta, w=w_sep)
        Lf = fill_loss(X, Y_j, r_fill=r_fill, tau=tau_fill_t, w=w_fill_t)
        return Li, Lc, Ls, Lf

    return loss_terms
