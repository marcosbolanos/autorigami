import numpy as np
import jax.numpy as jnp

from src.curvepack.bspline import make_basis_matrix
from src.curvepack.optimize import (
    sample_init_control_points,
    build_pairs_spatial_hash,
    make_loss_terms_fn,
    optimize_curves,
)
from src.curvepack.sdf import polygon_to_mask, mask_to_sdf, sample_interior_points


def test_optimize_smoke_no_nans_and_finite_terms() -> None:
    rng = np.random.default_rng(0)

    V = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]], dtype=np.float32)
    h = 0.5
    mask, origin, h = polygon_to_mask(V, h=h, pad=1.0)
    sdf = mask_to_sdf(mask, h)
    Y = sample_interior_points(mask, origin, h, Q=12, rng=rng)

    C = 2
    n_ctrl = 4
    n_samples = 6
    B = make_basis_matrix(n_ctrl, n_samples)
    P0 = sample_init_control_points(Y, C=C, n_ctrl=n_ctrl, rng=rng)
    r = np.full((C,), 0.2, dtype=np.float32)

    Rmin = 1.0
    delta = 0.1
    r_fill = 0.3
    tau_fill = 0.5
    w_inside = 1.0
    w_curv = 1.0
    w_sep = 1.0
    w_fill = 1.0

    P1, _ = optimize_curves(
        P0,
        B,
        sdf,
        origin,
        h,
        Y,
        r,
        steps=2,
        lr=1e-2,
        weight_decay=0.0,
        Rmin=Rmin,
        delta=delta,
        r_fill=r_fill,
        tau_fill=tau_fill,
        w_inside=w_inside,
        w_curv=w_curv,
        w_sep=w_sep,
        w_fill=w_fill,
        log_every=1000,
        checkpoint_every=None,
        seed=0,
    )

    assert np.isfinite(P0).all()
    assert np.isfinite(P1).all()

    loss_terms_fn = make_loss_terms_fn(
        B,
        sdf,
        origin,
        h,
        Y,
        Rmin,
        delta,
        r_fill,
        w_inside,
        w_curv,
        w_sep,
    )

    pair_cell = float(np.max(r) + delta)
    for P in (P0, P1):
        X_np = np.einsum("mn,cnd->cmd", B, P)
        pairs_np = build_pairs_spatial_hash(
            X_np,
            r_max=float(np.max(r) + delta),
            cell=pair_cell,
        )
        pair_i_jax = jnp.asarray(pairs_np[0])
        pair_k_jax = jnp.asarray(pairs_np[1])
        pair_j_jax = jnp.asarray(pairs_np[2])
        pair_l_jax = jnp.asarray(pairs_np[3])
        pairs_j = (pair_i_jax, pair_k_jax, pair_j_jax, pair_l_jax)
        Li, Lc, Ls, Lf = loss_terms_fn(
            jnp.asarray(P),
            pairs_j,
            jnp.asarray(r),
            jnp.asarray(w_fill, dtype=jnp.float32),
            jnp.asarray(tau_fill, dtype=jnp.float32),
        )
        terms = np.array([Li, Lc, Ls, Lf], dtype=np.float32)
        assert np.isfinite(terms).all()
