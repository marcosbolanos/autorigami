from __future__ import annotations

import time
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
import optax  # type: ignore[reportMissingTypeStubs]
from beartype import beartype
from beartype.typing import Callable
from jaxtyping import jaxtyped

from .optimize_checkpoint import (
    CheckpointRun,
    CHECKPOINT_CSV_FIELDS,
    GRADIENTS_DEBUG_FIELDS,
    append_metrics_csv,
    init_checkpoint_run,
    save_checkpoint_npz,
    save_checkpoint_svg,
)
from .loss import inside_loss, curvature_loss, separation_loss, fill_reward
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
    checkpoint_viewbox: tuple[float, float, float, float] | None = None,
    checkpoint_canvas_size: tuple[float, float] | tuple[str, str] | None = None,
    checkpoint_stroke_width: float | str = "1pt",
    checkpoint_scale_nm_per_unit: float | None = None,
    checkpoint_scale_nm_per_cm: float | None = None,
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
        w_inside,
        w_curv,
        w_sep,
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
        w_inside,
        w_curv,
        w_sep,
    )
    grad_fn = jax.grad(loss_fn)

    fill_ramp_steps = max(1, int(0.3 * steps))
    tau_fill_start = tau_fill * 3.0
    if debug.is_verbose():
        debug.log(
            "fill_schedule "
            f"ramp_steps={fill_ramp_steps} tau_start={tau_fill_start:.6g} "
            f"tau_end={tau_fill:.6g} w_fill_target={w_fill:.6g}"
        )

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
        w_fill_t: JaxScalar,
        tau_fill_t: JaxScalar,
    ) -> tuple[JaxControlPoints, optax.OptState, JaxScalar]:
        L = loss_fn(P_ctrl, pairs_j, r_j, w_fill_t, tau_fill_t)
        g = cast(JaxControlPoints, grad_fn(P_ctrl, pairs_j, r_j, w_fill_t, tau_fill_t))
        updates, opt_state2 = opt.update(g, opt_state, P_ctrl)
        P2 = cast(JaxControlPoints, optax.apply_updates(P_ctrl, updates))
        return P2, opt_state2, L

    r_j: JaxRadii = jnp.asarray(r)
    sdf_j = jnp.asarray(sdf)
    origin_j = jnp.asarray(origin)
    Y_j = jnp.asarray(Y)

    def grad_stats(
        term_fn: Callable[[JaxControlPoints], JaxScalar], X_j: JaxControlPoints
    ) -> tuple[float, float]:
        g = jax.grad(term_fn)(X_j)
        g_norm = jnp.linalg.norm(g, axis=-1)
        return float(jnp.mean(g_norm)), float(jnp.max(g_norm))

    def compute_fill_schedule(
        step_index: int,
    ) -> tuple[float, float, JaxScalar, JaxScalar]:
        if fill_ramp_steps <= 1:
            fill_alpha = 1.0
        else:
            fill_alpha = min(1.0, step_index / float(fill_ramp_steps - 1))
        w_fill_t = w_fill * fill_alpha
        tau_fill_t = tau_fill_start * (1.0 - fill_alpha) + tau_fill * fill_alpha
        return w_fill_t, tau_fill_t, jnp.asarray(w_fill_t), jnp.asarray(tau_fill_t)

    def write_gradients_debug(
        step_idx: int,
        X_np_ckpt: np.ndarray,
        pairs_np_ckpt: PairsNp,
        w_fill_t: float,
        tau_fill_t: float,
        w_fill_j: JaxScalar,
        tau_fill_j: JaxScalar,
        step_start: float,
    ) -> None:
        if checkpoint_run is None:
            return
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

        X_j_ckpt = jnp.asarray(X_np_ckpt)

        def term_inside(X: JaxControlPoints) -> JaxScalar:
            return inside_loss(X, sdf_j, origin_j, h, r_j, w=w_inside)

        def term_curv(X: JaxControlPoints) -> JaxScalar:
            return curvature_loss(X, Rmin, w=w_curv)

        def term_sep(X: JaxControlPoints) -> JaxScalar:
            return separation_loss(X, r_j, pairs_j_ckpt, delta=delta, w=w_sep)

        def term_fill(X: JaxControlPoints) -> JaxScalar:
            return fill_reward(X, Y_j, r_fill=r_fill, tau=tau_fill_j, w=w_fill_j)

        Li_dbg = term_inside(X_j_ckpt)
        Lc_dbg = term_curv(X_j_ckpt)
        Ls_dbg = term_sep(X_j_ckpt)
        Lf_dbg = term_fill(X_j_ckpt)

        gi_mean, gi_max = grad_stats(term_inside, X_j_ckpt)
        gc_mean, gc_max = grad_stats(term_curv, X_j_ckpt)
        gs_mean, gs_max = grad_stats(term_sep, X_j_ckpt)
        gf_mean, gf_max = grad_stats(term_fill, X_j_ckpt)

        row_dbg = {
            "step": step_idx,
            "loss": float(Li_dbg + Lc_dbg + Ls_dbg + Lf_dbg),
            "loss_inside": float(Li_dbg),
            "loss_curv": float(Lc_dbg),
            "loss_sep": float(Ls_dbg),
            "loss_fill": float(Lf_dbg),
            "tau_fill": float(tau_fill_t),
            "w_fill": float(w_fill_t),
            "pairs": int(pairs_np_ckpt[0].shape[0]),
            "lr": float(schedule(step_idx)),
            "elapsed_s": float(time.perf_counter() - start_time),
            "step_s": float(time.perf_counter() - step_start),
            "grad_inside_mean": gi_mean,
            "grad_inside_max": gi_max,
            "grad_curv_mean": gc_mean,
            "grad_curv_max": gc_max,
            "grad_sep_mean": gs_mean,
            "grad_sep_max": gs_max,
            "grad_fill_mean": gf_mean,
            "grad_fill_max": gf_max,
        }
        append_metrics_csv(
            checkpoint_run.gradients_csv_path,
            GRADIENTS_DEBUG_FIELDS,
            row_dbg,
        )

    if checkpoint_every is not None and checkpoint_every <= 0:
        checkpoint_every = None

    save_checkpoints = checkpoint_every is not None

    checkpoint_run: CheckpointRun | None
    if not (checkpoint_csv or checkpoint_svgs or save_checkpoints):
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
            "checkpoint_every": (
                int(checkpoint_every) if checkpoint_every is not None else None
            ),
            "checkpoint_svgs": bool(checkpoint_svgs),
            "checkpoint_csv": bool(checkpoint_csv),
            "checkpoint_viewbox": (
                list(checkpoint_viewbox) if checkpoint_viewbox is not None else None
            ),
            "checkpoint_canvas_size": (
                list(checkpoint_canvas_size)
                if checkpoint_canvas_size is not None
                else None
            ),
            "checkpoint_stroke_width": str(checkpoint_stroke_width),
            "svg_scale_nm_per_unit": (
                float(checkpoint_scale_nm_per_unit)
                if checkpoint_scale_nm_per_unit is not None
                else None
            ),
            "svg_scale_nm_per_cm": (
                float(checkpoint_scale_nm_per_cm)
                if checkpoint_scale_nm_per_cm is not None
                else None
            ),
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

    if checkpoint_run is not None and save_checkpoints:
        step_idx = 0
        step_start = time.perf_counter()
        w_fill_t, tau_fill_t, w_fill_j, tau_fill_j = compute_fill_schedule(0)
        P_np0 = np.array(P_ctrl)
        X_np0 = np.einsum("mn,cnd->cmd", B, P_np0)
        pairs_np0 = build_pairs_spatial_hash(
            X_np0,
            r_max=float(np.max(r) + delta),
            cell=pair_cell_value,
        )
        pair_i_jax = jnp.asarray(pairs_np0[0])
        pair_k_jax = jnp.asarray(pairs_np0[1])
        pair_j_jax = jnp.asarray(pairs_np0[2])
        pair_l_jax = jnp.asarray(pairs_np0[3])
        pairs_j0: PairsJax = (
            pair_i_jax,
            pair_k_jax,
            pair_j_jax,
            pair_l_jax,
        )
        loss0 = float("nan")
        if checkpoint_csv:
            Li0, Lc0, Ls0, Lf0 = loss_terms_fn(
                P_ctrl, pairs_j0, r_j, w_fill_j, tau_fill_j
            )
            loss_total = float(Li0) + float(Lc0) + float(Ls0) + float(Lf0)
            loss0 = loss_total
            elapsed_total = time.perf_counter() - start_time
            elapsed_step = time.perf_counter() - step_start
            row = {
                "step": step_idx,
                "loss": loss_total,
                "loss_inside": float(Li0),
                "loss_curv": float(Lc0),
                "loss_sep": float(Ls0),
                "loss_fill": float(Lf0),
                "tau_fill": float(tau_fill_t),
                "w_fill": float(w_fill_t),
                "pairs": int(pairs_np0[0].shape[0]),
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
        else:
            loss0 = float(loss_fn(P_ctrl, pairs_j0, r_j, w_fill_j, tau_fill_j))

        write_gradients_debug(
            step_idx,
            X_np0,
            pairs_np0,
            w_fill_t,
            tau_fill_t,
            w_fill_j,
            tau_fill_j,
            step_start,
        )
        save_checkpoint_npz(
            checkpoint_run.run_dir,
            step_idx,
            P_np0,
            np.array(r),
            loss0,
        )
        if checkpoint_svgs:
            save_checkpoint_svg(
                checkpoint_run.run_dir,
                step_idx,
                X_np0,
                viewbox=checkpoint_viewbox,
                canvas_size=checkpoint_canvas_size,
                stroke_width=checkpoint_stroke_width,
            )

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

        w_fill_t, tau_fill_t, w_fill_j, tau_fill_j = compute_fill_schedule(t)

        li_val = lc_val = ls_val = lf_val = float("nan")
        need_terms = checkpoint_csv or (debug.is_verbose() and t < debug_steps)
        if need_terms:
            Li, Lc, Ls, Lf = loss_terms_fn(P_ctrl, pairs_j, r_j, w_fill_j, tau_fill_j)
            li_val = float(Li)
            lc_val = float(Lc)
            ls_val = float(Ls)
            lf_val = float(Lf)
            total = li_val + lc_val + ls_val + lf_val
            if debug.is_verbose() and t < debug_steps:
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

        P_ctrl, opt_state, L = step(
            P_ctrl, opt_state, pairs_j, r_j, w_fill_j, tau_fill_j
        )

        step_idx = t + 1
        if checkpoint_run is not None and checkpoint_csv:
            loss_total = li_val + lc_val + ls_val + lf_val
            elapsed_total = time.perf_counter() - start_time
            elapsed_step = time.perf_counter() - step_start
            row = {
                "step": step_idx,
                "loss": loss_total,
                "loss_inside": li_val,
                "loss_curv": lc_val,
                "loss_sep": ls_val,
                "loss_fill": lf_val,
                "tau_fill": float(tau_fill_t),
                "w_fill": float(w_fill_t),
                "pairs": int(pairs_np[0].shape[0]),
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

        save_step = save_checkpoints and (step_idx % checkpoint_every == 0)
        if checkpoint_run is not None and save_step:
            P_np = np.array(P_ctrl)
            X_np_ckpt = np.einsum("mn,cnd->cmd", B, P_np)
            pairs_np_ckpt = build_pairs_spatial_hash(
                X_np_ckpt,
                r_max=float(np.max(r) + delta),
                cell=pair_cell_value,
            )
            write_gradients_debug(
                step_idx,
                X_np_ckpt,
                pairs_np_ckpt,
                w_fill_t,
                tau_fill_t,
                w_fill_j,
                tau_fill_j,
                step_start,
            )

            save_checkpoint_npz(
                checkpoint_run.run_dir,
                step_idx,
                P_np,
                np.array(r),
                float(L),
            )
            if checkpoint_svgs:
                save_checkpoint_svg(
                    checkpoint_run.run_dir,
                    step_idx,
                    X_np_ckpt,
                    viewbox=checkpoint_viewbox,
                    canvas_size=checkpoint_canvas_size,
                    stroke_width=checkpoint_stroke_width,
                )

        if (t % log_every) == 0 or t == steps - 1:
            last_L = float(L)
            print(f"step {t:5d}  loss={last_L:.6g}  pairs={pairs_np[0].shape[0]}")

        # Optional: continuation on fill softness / weights (simple schedule)
        # (kept out for clarity; you can ramp w_fill and decrease tau_fill over time)

    return np.array(P_ctrl), last_L
