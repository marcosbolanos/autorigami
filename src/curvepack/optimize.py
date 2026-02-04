from __future__ import annotations

from typing import Callable, TypeAlias, cast

import numpy as np
import jax
import jax.numpy as jnp
import optax  # type: ignore[reportMissingTypeStubs]
from beartype import beartype
from jaxtyping import Float, Int, jaxtyped

from .loss import inside_loss, curvature_loss, separation_loss, fill_reward

NpControlPoints: TypeAlias = Float[np.ndarray, "C n_ctrl 2"]
NpBasisMatrix: TypeAlias = Float[np.ndarray, "M n_ctrl"]
NpSdfGrid: TypeAlias = Float[np.ndarray, "H W"]
NpOrigin: TypeAlias = Float[np.ndarray, "2"]
NpSamplePoints: TypeAlias = Float[np.ndarray, "Q 2"]
NpCurveSamples: TypeAlias = Float[np.ndarray, "C M 2"]
NpRadii: TypeAlias = Float[np.ndarray, "C"]
PairsNp: TypeAlias = tuple[
    Int[np.ndarray, "P"],
    Int[np.ndarray, "P"],
    Int[np.ndarray, "P"],
    Int[np.ndarray, "P"],
]
JaxControlPoints: TypeAlias = Float[jax.Array, "C n_ctrl 2"]
JaxRadii: TypeAlias = Float[jax.Array, "C"]
PairsJax: TypeAlias = tuple[
    Int[jax.Array, "P"],
    Int[jax.Array, "P"],
    Int[jax.Array, "P"],
    Int[jax.Array, "P"],
]
JaxScalar: TypeAlias = Float[jax.Array, ""]
LossFn: TypeAlias = Callable[[JaxControlPoints, PairsJax, JaxRadii], JaxScalar]


@jaxtyped(typechecker=beartype)
def sample_init_control_points(
    Y: NpSamplePoints,
    C: int,
    n_ctrl: int,
    rng: np.random.Generator,
) -> NpControlPoints:
    """
    Initialize control points by picking random interior points and smoothing them.
    Returns P0: (C,n_ctrl,2)
    """
    Q = Y.shape[0]
    P0 = np.zeros((C, n_ctrl, 2), dtype=np.float32)
    for i in range(C):
        idx = rng.integers(0, Q, size=n_ctrl)
        pts = Y[idx].copy()
        # mild smoothing along index
        for _ in range(5):
            pts[1:-1] = 0.25 * pts[:-2] + 0.5 * pts[1:-1] + 0.25 * pts[2:]
        P0[i] = pts
    return P0


@jaxtyped(typechecker=beartype)
def build_pairs_spatial_hash(
    X: NpCurveSamples,
    r_max: float,
    cell: float,
    exclude_adj: int = 2,
    max_pairs: int = 200000,
) -> PairsNp:
    """
    Build candidate segment pairs for separation loss using a uniform grid hash.
    X: (C,M,2) samples in numpy
    r_max: max tube radius + clearance
    cell: grid cell size (recommend ~ r_max)
    exclude_adj: for same curve, exclude segment pairs with |k-l| <= exclude_adj
    Returns (pair_i, pair_k, pair_j, pair_l) int32 arrays.
    """
    C, M, _ = X.shape
    S = M - 1

    # Segment midpoints for hashing
    A = X[:, :-1, :]
    B = X[:, 1:, :]
    mid = 0.5 * (A + B)  # (C,S,2)

    mins = mid.reshape(-1, 2).min(axis=0)
    # map to grid coordinates
    gx = np.floor((mid[..., 0] - mins[0]) / cell).astype(np.int32)
    gy = np.floor((mid[..., 1] - mins[1]) / cell).astype(np.int32)

    # hash dict: (gx,gy)-> list of (curve, seg)
    buckets: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for i in range(C):
        for k in range(S):
            key = (gx[i, k], gy[i, k])
            buckets.setdefault(key, []).append((i, k))

    # For each bucket, check within neighboring buckets
    pair_i: list[int] = []
    pair_k: list[int] = []
    pair_j: list[int] = []
    pair_l: list[int] = []

    def add_pair(i: int, k: int, j: int, l: int) -> None:
        pair_i.append(i)
        pair_k.append(k)
        pair_j.append(j)
        pair_l.append(l)

    neigh: list[tuple[int, int]] = [
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 0),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    ]
    for key, items in buckets.items():
        bx, by = key
        # collect candidate neighbor items
        cand: list[tuple[int, int]] = []
        for dx, dy in neigh:
            cand.extend(buckets.get((bx + dx, by + dy), []))
        # brute force within candidates
        for i, k in items:
            for j, l in cand:
                # avoid duplicates by ordering
                if (j < i) or (j == i and l <= k):
                    continue
                if i == j and abs(k - l) <= exclude_adj:
                    continue
                add_pair(i, k, j, l)
                if len(pair_i) >= max_pairs:
                    break
            if len(pair_i) >= max_pairs:
                break
        if len(pair_i) >= max_pairs:
            break

    return (
        np.asarray(pair_i, np.int32),
        np.asarray(pair_k, np.int32),
        np.asarray(pair_j, np.int32),
        np.asarray(pair_l, np.int32),
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
    tau_fill: float,
    w_inside: float,
    w_curv: float,
    w_sep: float,
    w_fill: float,
) -> LossFn:
    """
    Returns a JAX function loss(P, pairs, r) -> scalar.
    """
    B_j = jnp.asarray(B)
    sdf_j = jnp.asarray(sdf)
    origin_j = jnp.asarray(origin)
    Y_j = jnp.asarray(Y)

    def loss(P: JaxControlPoints, pairs: PairsJax, r: JaxRadii) -> JaxScalar:
        # sample: X = B @ P (per curve)
        X = jnp.einsum("mn,cnd->cmd", B_j, P)

        Li = inside_loss(X, sdf_j, origin_j, h, r, w=w_inside)
        Lc = curvature_loss(X, Rmin, w=w_curv)
        Ls = separation_loss(X, r, pairs, delta=delta, w=w_sep)
        Lf = fill_reward(X, Y_j, r_fill=r_fill, tau=tau_fill, w=w_fill)

        return Li + Lc + Ls + Lf

    return loss


@jaxtyped(typechecker=beartype)
def optimize_curves(
    P0: NpControlPoints,
    B: NpBasisMatrix,
    sdf: NpSdfGrid,
    origin: NpOrigin,
    h: float,
    Y: NpSamplePoints,
    r: NpRadii,
    *,
    steps: int = 2000,
    lr: float = 1e-2,
    weight_decay: float = 1e-4,
    Rmin: float = 20.0,
    delta: float = 2.0,
    r_fill: float = 6.0,
    tau_fill: float = 2.0,
    w_inside: float = 5.0,
    w_curv: float = 1.0,
    w_sep: float = 5.0,
    w_fill: float = 2.0,
    pair_cell: float | None = None,
    seed: int = 0,
    log_every: int = 50,
) -> tuple[NpControlPoints, float]:
    """
    Optimize control points P (C,n_ctrl,2).
    """
    P_ctrl: JaxControlPoints = jnp.asarray(P0)

    pair_cell_value = (
        float(np.max(r) + delta) if pair_cell is None else float(pair_cell)
    )

    loss_fn = make_loss_fn(
        B,
        sdf,
        origin,
        h,
        Y,
        Rmin,
        delta,
        r_fill,
        tau_fill,
        w_inside,
        w_curv,
        w_sep,
        w_fill,
    )
    grad_fn = jax.grad(loss_fn)

    opt = optax.adamw(learning_rate=lr, weight_decay=weight_decay)
    opt_state = opt.init(P_ctrl)

    # JIT the update step (pairs fed as dynamic arrays; ok)
    @jax.jit
    def step(
        P_ctrl: JaxControlPoints,
        opt_state: optax.OptState,
        pairs_j: PairsJax,
        r_j: JaxRadii,
    ) -> tuple[JaxControlPoints, optax.OptState, JaxScalar]:
        L = loss_fn(P_ctrl, pairs_j, r_j)
        g = cast(JaxControlPoints, grad_fn(P_ctrl, pairs_j, r_j))
        updates, opt_state2 = opt.update(g, opt_state, P_ctrl)
        P2 = cast(JaxControlPoints, optax.apply_updates(P_ctrl, updates))
        return P2, opt_state2, L

    r_j: JaxRadii = jnp.asarray(r)

    last_L = float("nan")
    for t in range(steps):
        # Build pairs outside JAX using current sampled X (numpy)
        X_np = np.einsum("mn,cnd->cmd", B, np.array(P_ctrl))  # (C,M,2)
        pairs_np: PairsNp = build_pairs_spatial_hash(
            X_np,
            r_max=float(np.max(r) + delta),
            cell=pair_cell_value,
        )
        pair_i_jax = jnp.asarray(pairs_np[0])
        pair_k_jax = jnp.asarray(pairs_np[1])
        pair_j_jax = jnp.asarray(pairs_np[2])
        pair_l_jax = jnp.asarray(pairs_np[3])
        pairs_j: PairsJax = (
            pair_i_jax,
            pair_k_jax,
            pair_j_jax,
            pair_l_jax,
        )

        P_ctrl, opt_state, L = step(P_ctrl, opt_state, pairs_j, r_j)

        if (t % log_every) == 0 or t == steps - 1:
            last_L = float(L)
            print(f"step {t:5d}  loss={last_L:.6g}  pairs={pairs_np[0].shape[0]}")

        # Optional: continuation on fill softness / weights (simple schedule)
        # (kept out for clarity; you can ramp w_fill and decrease tau_fill over time)

    return np.array(P_ctrl), last_L
