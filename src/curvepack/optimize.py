from __future__ import annotations

import time
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
import optax  # type: ignore[reportMissingTypeStubs]
from beartype import beartype
from jaxtyping import jaxtyped

from .optimize_checkpoint import (
    CheckpointRun,
    CHECKPOINT_CSV_FIELDS,
    append_metrics_csv,
    init_checkpoint_run,
    save_checkpoint_npz,
    save_checkpoint_svg,
)
from . import optimize_init
from .optimize_loss import make_loss_fn, make_loss_terms_fn
from .optimize_pairs import build_pairs_spatial_hash
from .optimize_types import (
    JaxControlPoints,
    JaxRadii,
    JaxScalar,
    NpBasisMatrix,
    NpControlPoints,
    NpOrigin,
    NpRadii,
    NpSamplePoints,
    NpSdfGrid,
    PairsJax,
    PairsNp,
)
from .. import PROJECT_ROOT
from ..utils import debug, debug_helpers

__all__ = [
    "sample_init_control_points",
    "build_pairs_spatial_hash",
    "make_loss_fn",
    "make_loss_terms_fn",
    "optimize_curves",
]

sample_init_control_points = optimize_init.sample_init_control_points


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
    Rmin: float = 6.0,
    delta: float = 0.6,
    r_fill: float = 1.0,
    tau_fill: float = 2.0,
    w_inside: float = 5.0,
    w_curv: float = 1.0,
    w_sep: float = 5.0,
    w_fill: float = 2.0,
    pair_cell: float | None = None,
    seed: int = 0,
    log_every: int = 50,
    checkpoint_every: int | None = 500,
    checkpoint_svgs: bool = True,
    checkpoint_csv: bool = True,
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
    loss_terms_fn = make_loss_terms_fn(
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

    warmup_steps = max(200, int(0.03 * steps))
    warmup_steps = min(warmup_steps, max(1, steps - 1))
    decay_steps = max(1, steps - warmup_steps)
    end_lr = lr * 0.02
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=lr,
        warmup_steps=warmup_steps,
        decay_steps=decay_steps,
        end_value=end_lr,
    )
    if debug.is_verbose():
        debug.log(
            "lr_schedule "
            f"warmup_steps={warmup_steps} decay_steps={decay_steps} "
            f"peak_lr={lr:.6g} end_lr={end_lr:.6g}"
        )

    opt = optax.chain(
        optax.zero_nans(),
        optax.clip_by_global_norm(1e2),
        optax.adamw(learning_rate=schedule, weight_decay=weight_decay),
    )
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

    if checkpoint_every is not None and checkpoint_every <= 0:
        checkpoint_every = None

    checkpoint_run: CheckpointRun | None
    if checkpoint_every is None:
        checkpoint_run = None
    else:
        metadata = {
            "seed": int(seed),
            "steps": int(steps),
            "lr": float(lr),
            "weight_decay": float(weight_decay),
            "Rmin": float(Rmin),
            "delta": float(delta),
            "r_fill": float(r_fill),
            "tau_fill": float(tau_fill),
            "w_inside": float(w_inside),
            "w_curv": float(w_curv),
            "w_sep": float(w_sep),
            "w_fill": float(w_fill),
            "pair_cell": float(pair_cell_value),
            "warmup_steps": int(warmup_steps),
            "decay_steps": int(decay_steps),
            "end_lr": float(end_lr),
            "checkpoint_every": int(checkpoint_every),
            "checkpoint_svgs": bool(checkpoint_svgs),
            "checkpoint_csv": bool(checkpoint_csv),
            "shapes": {
                "P0": list(P0.shape),
                "B": list(B.shape),
                "sdf": list(sdf.shape),
                "origin": list(origin.shape),
                "Y": list(Y.shape),
                "r": list(r.shape),
            },
            "r_stats": {
                "min": float(np.min(r)),
                "max": float(np.max(r)),
                "mean": float(np.mean(r)),
            },
        }
        checkpoint_run = init_checkpoint_run(
            PROJECT_ROOT / "data" / "checkpoints", metadata
        )

    last_L = float("nan")
    start_time = time.perf_counter()
    debug_steps = 5
    for t in range(steps):
        step_start = time.perf_counter()
        # Build pairs outside JAX using current sampled X (numpy)
        X_np = np.einsum("mn,cnd->cmd", B, np.array(P_ctrl))  # (C,M,2)
        if debug.is_verbose() and t == 0:
            debug_helpers.log_array("X_np", X_np)
            debug.log(
                f"pair_hash_params: pair_cell={pair_cell_value:.6g} "
                f"r_max={float(np.max(r) + delta):.6g} delta={delta:.6g}"
            )
        if debug.is_verbose() and not np.isfinite(X_np).all():
            debug_helpers.log_once(
                "X_np_nonfinite",
                f"X_np non-finite at step {t}",
            )
            debug_helpers.log_array_once("X_np_nonfinite_arr", "X_np", X_np)
            debug_helpers.log_array_once(
                "P_ctrl_nonfinite_arr",
                "P_ctrl",
                np.array(P_ctrl),
            )
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

        if debug.is_verbose() and t < debug_steps:
            Li, Lc, Ls, Lf = loss_terms_fn(P_ctrl, pairs_j, r_j)
            li_val = float(Li)
            lc_val = float(Lc)
            ls_val = float(Ls)
            lf_val = float(Lf)
            total = li_val + lc_val + ls_val + lf_val
            debug.log(
                "loss_terms "
                f"step={t} inside={li_val:.6g} curv={lc_val:.6g} "
                f"sep={ls_val:.6g} fill={lf_val:.6g} total={total:.6g}"
            )
            if not np.isfinite([li_val, lc_val, ls_val, lf_val]).all():
                raise ValueError(
                    "Non-finite loss terms at step "
                    f"{t}: inside={li_val} curv={lc_val} sep={ls_val} fill={lf_val}"
                )

        P_ctrl, opt_state, L = step(P_ctrl, opt_state, pairs_j, r_j)

        if checkpoint_run is not None and checkpoint_every is not None:
            step_idx = t + 1
            if (step_idx % checkpoint_every) == 0:
                P_np = np.array(P_ctrl)
                save_checkpoint_npz(
                    checkpoint_run.run_dir,
                    step_idx,
                    P_np,
                    np.array(r),
                    float(L),
                )
                X_np_ckpt = np.einsum("mn,cnd->cmd", B, P_np)
                if checkpoint_csv:
                    pairs_np_ckpt = build_pairs_spatial_hash(
                        X_np_ckpt,
                        r_max=float(np.max(r) + delta),
                        cell=pair_cell_value,
                    )
                    pair_i_jax = jnp.asarray(pairs_np_ckpt[0])
                    pair_k_jax = jnp.asarray(pairs_np_ckpt[1])
                    pair_j_jax = jnp.asarray(pairs_np_ckpt[2])
                    pair_l_jax = jnp.asarray(pairs_np_ckpt[3])
                    pairs_j_ckpt: PairsJax = (
                        pair_i_jax,
                        pair_k_jax,
                        pair_j_jax,
                        pair_l_jax,
                    )
                    Li, Lc, Ls, Lf = loss_terms_fn(P_ctrl, pairs_j_ckpt, r_j)
                    loss_total = float(Li) + float(Lc) + float(Ls) + float(Lf)
                    elapsed_total = time.perf_counter() - start_time
                    elapsed_step = time.perf_counter() - step_start
                    row = {
                        "step": step_idx,
                        "loss": loss_total,
                        "loss_inside": float(Li),
                        "loss_curv": float(Lc),
                        "loss_sep": float(Ls),
                        "loss_fill": float(Lf),
                        "pairs": int(pairs_np_ckpt[0].shape[0]),
                        "lr": float(schedule(step_idx)),
                        "weight_decay": float(weight_decay),
                        "elapsed_s": float(elapsed_total),
                        "step_s": float(elapsed_step),
                    }
                    append_metrics_csv(
                        checkpoint_run.csv_path,
                        CHECKPOINT_CSV_FIELDS,
                        row,
                    )
                if checkpoint_svgs:
                    save_checkpoint_svg(checkpoint_run.run_dir, step_idx, X_np_ckpt)

        if (t % log_every) == 0 or t == steps - 1:
            last_L = float(L)
            print(f"step {t:5d}  loss={last_L:.6g}  pairs={pairs_np[0].shape[0]}")

        # Optional: continuation on fill softness / weights (simple schedule)
        # (kept out for clarity; you can ramp w_fill and decrease tau_fill over time)

    return np.array(P_ctrl), last_L
