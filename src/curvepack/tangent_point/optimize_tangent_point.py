from __future__ import annotations

import time
from typing import Mapping, cast

import jax
import jax.numpy as jnp
import numpy as np
import optax  # type: ignore[reportMissingTypeStubs]
from beartype import beartype
from jaxtyping import Float, jaxtyped

from ..geometry import curves_length
from ..optimize_checkpoint import (
    append_metrics_csv,
    init_checkpoint_run,
    save_checkpoint_npz_arrays,
    save_checkpoint_svg,
)
from ..optimize_types import JaxCurveSamples, JaxScalar, NpCurveSamples
from .tangent_point_energy import tangent_point_energy_curves
from ... import PROJECT_ROOT


TP_CHECKPOINT_CSV_FIELDS = [
    "step",
    "loss",
    "energy_tpe",
    "len_mean",
    "len_target_mean",
    "len_rmse",
    "lambda_mean",
    "rho",
    "lr",
    "weight_decay",
    "update_norm",
    "elapsed_s",
    "step_s",
]


@jaxtyped(typechecker=beartype)
def _compute_lengths_np(
    X: NpCurveSamples,
    *,
    closed: bool,
) -> Float[np.ndarray, "C"]:
    seg = X[:, 1:, :] - X[:, :-1, :]
    seglen = np.linalg.norm(seg, axis=-1)
    length = np.sum(seglen, axis=-1)
    if closed:
        length = length + np.linalg.norm(X[:, 0, :] - X[:, -1, :], axis=-1)
    return length.astype(np.float32)


@jaxtyped(typechecker=beartype)
def optimize_curves_tangent_point(
    X0: NpCurveSamples,
    *,
    steps: int = 2000,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    closed: bool = False,
    ignore_k: int = 2,
    eps2: float = 1e-12,
    len_final_mul: float = 1.5,
    len_rho: float = 1.0,
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
    checkpoint_shape: Float[np.ndarray, "N 2"] | None = None,
    metadata_extra: Mapping[str, object] | None = None,
) -> tuple[NpCurveSamples, float]:
    """Optimize curve samples X directly using tangent-point energy + length constraint.

    X0: (C, M, 2) numpy array.

    Constraint: each curve length tracks a schedule that increases linearly from
    its initial length to `len_final_mul * initial_length` over `steps`.

    We enforce the constraint via an augmented Lagrangian with per-curve multipliers.
    """

    if steps <= 0:
        raise ValueError("steps must be positive")
    if lr <= 0:
        raise ValueError("lr must be positive")
    if ignore_k < 0:
        raise ValueError("ignore_k must be >= 0")
    if len_final_mul <= 0:
        raise ValueError("len_final_mul must be positive")
    if len_rho <= 0:
        raise ValueError("len_rho must be positive")
    if checkpoint_every is not None and checkpoint_every <= 0:
        checkpoint_every = None

    rng = np.random.default_rng(seed)
    _ = rng  # reserved for future stochastic variants

    X_j: JaxCurveSamples = jnp.asarray(X0)
    C = int(X0.shape[0])

    len0_np = _compute_lengths_np(X0, closed=closed)
    lenf_np = len0_np * float(len_final_mul)

    schedule = optax.constant_schedule(lr)
    opt = optax.chain(
        optax.zero_nans(),
        optax.clip_by_global_norm(1e2),
        optax.adamw(learning_rate=schedule, weight_decay=weight_decay),
    )
    opt_state = opt.init(X_j)

    lambda_len: Float[jax.Array, "C"] = jnp.zeros((C,), dtype=X_j.dtype)
    rho_j: JaxScalar = jnp.asarray(float(len_rho), dtype=X_j.dtype)

    def loss_aug(
        X_in: JaxCurveSamples,
        lambda_in: Float[jax.Array, "C"],
        len_target: Float[jax.Array, "C"],
    ) -> tuple[
        JaxScalar, tuple[JaxScalar, Float[jax.Array, "C"], Float[jax.Array, "C"]]
    ]:
        energy = tangent_point_energy_curves(
            X_in,
            closed=closed,
            ignore_k=ignore_k,
            eps2=eps2,
        )
        length = curves_length(X_in, closed=closed)
        g = length - len_target
        penalty = jnp.sum(lambda_in * g + 0.5 * rho_j * g * g)
        return energy + penalty, (energy, length, g)

    def loss_aug_scalar(
        X_in: JaxCurveSamples,
        lambda_in: Float[jax.Array, "C"],
        len_target: Float[jax.Array, "C"],
    ) -> JaxScalar:
        return loss_aug(X_in, lambda_in, len_target)[0]

    grad_fn = jax.grad(loss_aug_scalar)

    @jax.jit
    def step(
        X_in: JaxCurveSamples,
        opt_state_in: optax.OptState,
        lambda_in: Float[jax.Array, "C"],
        len_target: Float[jax.Array, "C"],
    ) -> tuple[
        JaxCurveSamples,
        optax.OptState,
        Float[jax.Array, "C"],
        JaxScalar,
        JaxScalar,
        Float[jax.Array, "C"],
        Float[jax.Array, "C"],
        JaxScalar,
    ]:
        loss_val, (energy, _length, _g) = loss_aug(X_in, lambda_in, len_target)
        gX = cast(JaxCurveSamples, grad_fn(X_in, lambda_in, len_target))
        updates, opt_state_out = opt.update(gX, opt_state_in, X_in)
        X_out = cast(JaxCurveSamples, optax.apply_updates(X_in, updates))

        length2 = curves_length(X_out, closed=closed)
        g2 = length2 - len_target
        lambda_out = lambda_in + rho_j * g2
        update_norm = jnp.linalg.norm(X_out - X_in)
        return (
            X_out,
            opt_state_out,
            lambda_out,
            loss_val,
            energy,
            length2,
            g2,
            update_norm,
        )

    checkpoint_run = None
    if checkpoint_csv or checkpoint_svgs or checkpoint_every is not None:
        metadata: dict[str, object] = {
            "seed": int(seed),
            "steps": int(steps),
            "lr": float(lr),
            "weight_decay": float(weight_decay),
            "closed": bool(closed),
            "ignore_k": int(ignore_k),
            "eps2": float(eps2),
            "len_final_mul": float(len_final_mul),
            "len_rho": float(len_rho),
            "len0": [float(v) for v in len0_np],
            "lenf": [float(v) for v in lenf_np],
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
                "X0": list(X0.shape),
            },
        }
        if metadata_extra:
            metadata["run_params"] = cast(object, dict(metadata_extra))
        checkpoint_run = init_checkpoint_run(
            PROJECT_ROOT / "data" / "checkpoints", metadata
        )

    start_time = time.perf_counter()
    last_loss = float("nan")

    def length_targets(step_idx: int) -> np.ndarray:
        if steps <= 1:
            alpha = 1.0
        else:
            alpha = float(step_idx) / float(steps - 1)
        return (len0_np + alpha * (lenf_np - len0_np)).astype(np.float32)

    # step 0 checkpoint
    if checkpoint_run is not None and checkpoint_every is not None:
        X_np0 = np.array(X_j)
        len_t0 = length_targets(0)
        L0 = float(
            loss_aug(
                jnp.asarray(X_np0),
                lambda_len,
                jnp.asarray(len_t0),
            )[0]
        )
        save_checkpoint_npz_arrays(
            checkpoint_run.run_dir,
            0,
            loss=L0,
            X=X_np0,
            lambda_len=np.array(lambda_len),
            len_target=len_t0,
        )
        if checkpoint_svgs:
            save_checkpoint_svg(
                checkpoint_run.run_dir,
                0,
                X_np0,
                viewbox=checkpoint_viewbox,
                canvas_size=checkpoint_canvas_size,
                stroke_width=checkpoint_stroke_width,
                reference_shape=checkpoint_shape,
            )

    for t in range(steps):
        step_start = time.perf_counter()
        len_t_np = length_targets(t)
        len_t_j = jnp.asarray(len_t_np)

        X_j, opt_state, lambda_len, loss_val, energy_val, length_val, g_val, upd = step(
            X_j,
            opt_state,
            lambda_len,
            len_t_j,
        )

        step_idx = t + 1
        save_step = (
            checkpoint_run is not None
            and checkpoint_every is not None
            and (step_idx % checkpoint_every == 0)
        )
        if checkpoint_run is not None and (
            checkpoint_csv or save_step or (t % log_every) == 0
        ):
            len_mean = float(jnp.mean(length_val))
            len_target_mean = float(np.mean(len_t_np))
            len_rmse = float(jnp.sqrt(jnp.mean(jnp.square(g_val))))
            lambda_mean = float(jnp.mean(lambda_len))
            elapsed_total = time.perf_counter() - start_time
            elapsed_step = time.perf_counter() - step_start
            lr_val = float(schedule(step_idx))
            update_norm = float(upd)

            if checkpoint_csv:
                row = {
                    "step": step_idx,
                    "loss": float(loss_val),
                    "energy_tpe": float(energy_val),
                    "len_mean": len_mean,
                    "len_target_mean": len_target_mean,
                    "len_rmse": len_rmse,
                    "lambda_mean": lambda_mean,
                    "rho": float(len_rho),
                    "lr": lr_val,
                    "weight_decay": float(weight_decay),
                    "update_norm": update_norm,
                    "elapsed_s": float(elapsed_total),
                    "step_s": float(elapsed_step),
                }
                append_metrics_csv(
                    checkpoint_run.csv_path,
                    TP_CHECKPOINT_CSV_FIELDS,
                    row,
                )

        if save_step and checkpoint_run is not None:
            X_np = np.array(X_j)
            save_checkpoint_npz_arrays(
                checkpoint_run.run_dir,
                step_idx,
                loss=float(loss_val),
                X=X_np,
                lambda_len=np.array(lambda_len),
                len_target=len_t_np,
            )
            if checkpoint_svgs:
                save_checkpoint_svg(
                    checkpoint_run.run_dir,
                    step_idx,
                    X_np,
                    viewbox=checkpoint_viewbox,
                    canvas_size=checkpoint_canvas_size,
                    stroke_width=checkpoint_stroke_width,
                    reference_shape=checkpoint_shape,
                )

        if (t % log_every) == 0 or t == steps - 1:
            last_loss = float(loss_val)
            print(
                f"step {t:5d}  loss={last_loss:.6g}  energy={float(energy_val):.6g} "
                f"len_rmse={float(jnp.sqrt(jnp.mean(jnp.square(g_val)))):.6g} "
                f"upd={float(upd):.6g}"
            )

    return np.array(X_j), last_loss
