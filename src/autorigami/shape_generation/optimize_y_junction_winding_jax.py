"""Optimize a Y-junction winding centerline with differentiable trajectory costs."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

from autorigami.shape_generation.generate_y_junction_winding import (
    YJunctionWindingSpec,
    _resample_centerline,
    _smooth_centerline_from_samples,
    _tube_mesh_from_centerline,
    _reference_y_mesh,
    _y_branch_segments,
    _y_clearance_values,
    generate_y_junction_winding_centerline,
)
import trimesh


@dataclass(frozen=True)
class TrajectoryOptimizationSpec:
    control_points: int = 300
    dense_points: int = 900
    iterations: int = 3500
    learning_rate: float = 0.025
    offset_bound_nm: float = 10.0
    y_margin_nm: float = 0.18
    self_margin_nm: float = 0.08
    upper_branch_contact_start_z_nm: float = 0.0
    upper_branch_contact_target_nm: float = 0.8
    maximum_upper_axis_tangent_fraction: float = 0.35
    local_exclusion_points: int = 4
    y_weight: float = 260.0
    upper_branch_contact_weight: float = 0.0
    upper_branch_circumferential_weight: float = 0.0
    self_weight: float = 150.0
    packing_weight: float = 0.4
    curvature_weight: float = 80.0
    smoothness_weight: float = 0.9
    jerk_weight: float = 0.16
    segment_length_weight: float = 4.0
    shape_weight: float = 0.28
    monotonic_z_weight: float = 240.0
    endpoint_tangent_weight: float = 2.0
    fix_endpoints: bool = False
    report_every: int = 250
    smooth_export_points: int = 7600
    smooth_export_method: str = "pchip"


def _sample_polyline_jax(control: jnp.ndarray, dense_points: int) -> jnp.ndarray:
    positions = jnp.linspace(0.0, control.shape[0] - 1.0, dense_points)
    left = jnp.floor(positions).astype(jnp.int32)
    left = jnp.clip(left, 0, control.shape[0] - 2)
    right = left + 1
    t = (positions - left.astype(jnp.float32))[:, None]
    return (1.0 - t) * control[left] + t * control[right]


def _distance_to_y_axes_jax(points: jnp.ndarray, starts: jnp.ndarray, ends: jnp.ndarray) -> jnp.ndarray:
    axes = ends - starts
    lengths_squared = jnp.sum(axes * axes, axis=1)
    projections = jnp.clip(((points[:, None, :] - starts[None, :, :]) * axes[None, :, :]).sum(axis=2) / lengths_squared[None, :], 0.0, 1.0)
    closest = starts[None, :, :] + projections[:, :, None] * axes[None, :, :]
    return jnp.linalg.norm(points[:, None, :] - closest, axis=2)


def _distance_to_y_axes_with_axes_jax(
    points: jnp.ndarray,
    starts: jnp.ndarray,
    ends: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    axes = ends - starts
    unit_axes = axes / jnp.linalg.norm(axes, axis=1, keepdims=True)
    lengths_squared = jnp.sum(axes * axes, axis=1)
    projections = jnp.clip(((points[:, None, :] - starts[None, :, :]) * axes[None, :, :]).sum(axis=2) / lengths_squared[None, :], 0.0, 1.0)
    closest = starts[None, :, :] + projections[:, :, None] * axes[None, :, :]
    return jnp.linalg.norm(points[:, None, :] - closest, axis=2), unit_axes


def _curvature_radii_jax(points: jnp.ndarray) -> jnp.ndarray:
    previous = points[:-2]
    current = points[1:-1]
    following = points[2:]
    a = current - previous
    b = following - current
    c = following - previous
    cross = jnp.linalg.norm(jnp.cross(a, b), axis=1)
    denom = jnp.linalg.norm(a, axis=1) * jnp.linalg.norm(b, axis=1) * jnp.linalg.norm(c, axis=1)
    curvature = jnp.where(denom > 1e-8, 2.0 * cross / denom, 0.0)
    return jnp.where(curvature > 1e-8, 1.0 / curvature, jnp.inf)


def _make_loss(
    initial_control: np.ndarray,
    winding_spec: YJunctionWindingSpec,
    optimization_spec: TrajectoryOptimizationSpec,
):
    starts_np, ends_np = _y_branch_segments(winding_spec)
    starts = jnp.asarray(starts_np, dtype=jnp.float32)
    ends = jnp.asarray(ends_np, dtype=jnp.float32)
    initial = jnp.asarray(initial_control, dtype=jnp.float32)
    required_y_distance = winding_spec.channel_radius_nm + 0.5 * winding_spec.rod_diameter_nm + winding_spec.rod_clearance_nm
    self_target = winding_spec.rod_diameter_nm + optimization_spec.self_margin_nm
    curvature_limit = 1.0 / winding_spec.minimum_curvature_radius_nm
    pair_indices = jnp.arange(optimization_spec.dense_points)
    symmetric_nonlocal_mask = jnp.abs(pair_indices[:, None] - pair_indices[None, :]) > optimization_spec.local_exclusion_points
    pair_penalty_mask = jnp.triu(symmetric_nonlocal_mask, k=1)

    def unpack(offsets: jnp.ndarray) -> jnp.ndarray:
        bounded_offsets = optimization_spec.offset_bound_nm * jnp.tanh(offsets)
        if optimization_spec.fix_endpoints:
            internal = initial[1:-1] + bounded_offsets
            return jnp.vstack((initial[0][None, :], internal, initial[-1][None, :]))
        return initial + bounded_offsets

    def loss(offsets: jnp.ndarray) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
        control = unpack(offsets)
        dense = _sample_polyline_jax(control, optimization_spec.dense_points)
        y_distances, y_unit_axes = _distance_to_y_axes_with_axes_jax(dense, starts, ends)
        y_clearance = jnp.min(y_distances - required_y_distance, axis=1)
        y_violation = jax.nn.softplus(-(y_clearance - optimization_spec.y_margin_nm) * 3.0) / 3.0
        upper_branch_clearance = jnp.min(y_distances[:, 1:] - required_y_distance, axis=1)
        upper_branch_active = jax.nn.sigmoid((dense[:, 2] - optimization_spec.upper_branch_contact_start_z_nm) * 0.7)
        upper_branch_contact_excess = upper_branch_active * (
            jax.nn.softplus((upper_branch_clearance - optimization_spec.upper_branch_contact_target_nm) * 1.6) / 1.6
        )

        diff = dense[:, None, :] - dense[None, :, :]
        distances = jnp.linalg.norm(diff + 1e-8, axis=2)
        masked_distances = jnp.where(symmetric_nonlocal_mask, distances, jnp.inf)
        nearest = jnp.min(masked_distances, axis=1)
        pair_violation = jnp.where(pair_penalty_mask, jax.nn.softplus((self_target - distances) * 2.0) / 2.0, 0.0)
        packing_excess = jax.nn.softplus((nearest - self_target) * 0.6) / 0.6

        tangents = dense[1:] - dense[:-1]
        tangent_norms = jnp.linalg.norm(tangents, axis=1)
        upper_branch_axis_index = jnp.argmin(0.5 * (y_distances[:-1, 1:] + y_distances[1:, 1:]), axis=1) + 1
        upper_branch_axes = y_unit_axes[upper_branch_axis_index]
        upper_axis_tangent_fraction = jnp.abs(jnp.sum(tangents * upper_branch_axes, axis=1)) / jnp.maximum(tangent_norms, 1e-8)
        upper_branch_near = jax.nn.sigmoid((2.8 - upper_branch_clearance) * 1.2)
        upper_circumferential_active = 0.5 * (
            upper_branch_active[:-1] * upper_branch_near[:-1]
            + upper_branch_active[1:] * upper_branch_near[1:]
        )
        upper_branch_circumferential_violation = upper_circumferential_active * (
            jax.nn.softplus((upper_axis_tangent_fraction - optimization_spec.maximum_upper_axis_tangent_fraction) * 8.0) / 8.0
        )

        radii = _curvature_radii_jax(dense)
        curvature = jnp.where(jnp.isfinite(radii), 1.0 / radii, 0.0)
        curvature_violation = jax.nn.softplus((curvature - curvature_limit) * 24.0) / 24.0

        second = control[:-2] - 2.0 * control[1:-1] + control[2:]
        third = control[:-3] - 3.0 * control[1:-2] + 3.0 * control[2:-1] - control[3:]
        segment_lengths = jnp.linalg.norm(control[1:] - control[:-1], axis=1)
        target_segment_length = jnp.mean(jnp.linalg.norm(initial[1:] - initial[:-1], axis=1))
        displacement = control - initial
        dz = control[1:, 2] - control[:-1, 2]
        z_violation = jax.nn.softplus((-dz) * 6.0) / 6.0
        tangent_residual = (control[1] - control[0] - (initial[1] - initial[0]))
        tangent_residual = jnp.concatenate((tangent_residual, control[-1] - control[-2] - (initial[-1] - initial[-2])))

        terms = {
            "y": optimization_spec.y_weight * jnp.mean(y_violation**2),
            "upper_branch_contact": optimization_spec.upper_branch_contact_weight * jnp.mean(upper_branch_contact_excess**2),
            "upper_branch_circumferential": optimization_spec.upper_branch_circumferential_weight * jnp.mean(upper_branch_circumferential_violation**2),
            "self": optimization_spec.self_weight * jnp.sum(pair_violation**2) / optimization_spec.dense_points,
            "packing": optimization_spec.packing_weight * jnp.mean(packing_excess**2),
            "curvature": optimization_spec.curvature_weight * jnp.mean(curvature_violation**2),
            "smoothness": optimization_spec.smoothness_weight * jnp.mean(jnp.sum(second**2, axis=1)),
            "jerk": optimization_spec.jerk_weight * jnp.mean(jnp.sum(third**2, axis=1)),
            "segment_length": optimization_spec.segment_length_weight * jnp.mean((segment_lengths - target_segment_length) ** 2),
            "shape": optimization_spec.shape_weight * jnp.mean(jnp.sum(displacement**2, axis=1)),
            "monotonic_z": optimization_spec.monotonic_z_weight * jnp.mean(z_violation**2),
            "endpoint_tangent": optimization_spec.endpoint_tangent_weight * jnp.mean(tangent_residual**2),
        }
        total = sum(terms.values())
        metrics = {
            **terms,
            "total": total,
            "min_y_clearance": jnp.min(y_clearance),
            "mean_upper_branch_clearance": jnp.mean(upper_branch_clearance),
            "nearest_nonlocal": jnp.min(nearest),
            "min_curvature_radius": jnp.min(radii),
            "mean_upper_axis_tangent_fraction": jnp.mean(upper_axis_tangent_fraction),
            "max_offset": jnp.max(jnp.linalg.norm(displacement, axis=1)),
        }
        return total, metrics

    return unpack, loss


def optimize_centerline(
    initial_centerline: np.ndarray,
    winding_spec: YJunctionWindingSpec,
    optimization_spec: TrajectoryOptimizationSpec,
) -> tuple[np.ndarray, dict[str, object]]:
    initial_control = _resample_centerline(initial_centerline, optimization_spec.control_points).astype(np.float32)
    unpack, loss = _make_loss(initial_control, winding_spec, optimization_spec)
    value_and_grad = jax.jit(jax.value_and_grad(lambda values: loss(values)[0]))
    metrics_fn = jax.jit(lambda values: loss(values)[1])
    offset_count = optimization_spec.control_points - 2 if optimization_spec.fix_endpoints else optimization_spec.control_points
    offsets = jnp.zeros((offset_count, 3), dtype=jnp.float32)
    optimizer = optax.chain(optax.clip_by_global_norm(3.0), optax.adam(optimization_spec.learning_rate))
    state = optimizer.init(offsets)
    history: list[dict[str, float]] = []

    for step in range(optimization_spec.iterations + 1):
        value, gradient = value_and_grad(offsets)
        updates, state = optimizer.update(gradient, state, offsets)
        offsets = optax.apply_updates(offsets, updates)
        if step % optimization_spec.report_every == 0 or step == optimization_spec.iterations:
            metrics = {key: float(value) for key, value in metrics_fn(offsets).items()}
            metrics["step"] = float(step)
            history.append(metrics)
            print(
                "step={step:.0f} total={total:.3f} y={y:.3f} self={self:.3f} "
                "near={nearest_nonlocal:.3f} min_y={min_y_clearance:.3f} min_R={min_curvature_radius:.3f}".format(**metrics),
                flush=True,
            )

    optimized_control = np.asarray(unpack(offsets), dtype=np.float64)
    if optimized_control.shape[0] == winding_spec.path_samples:
        optimized_centerline = optimized_control
    else:
        optimized_centerline = _resample_centerline(optimized_control, winding_spec.path_samples)
    diagnostics = {
        "optimizer": "jax_optax_adam_dense_trajectory",
        "optimization_spec": asdict(optimization_spec),
        "history": history,
    }
    return optimized_centerline, diagnostics


def export_optimized_winding(
    output_path: str | Path,
    winding_spec: YJunctionWindingSpec,
    optimization_spec: TrajectoryOptimizationSpec,
    initial_centerline_path: str | Path | None = None,
) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    initial = (
        generate_y_junction_winding_centerline(winding_spec)
        if initial_centerline_path is None
        else np.loadtxt(initial_centerline_path, delimiter=",", skiprows=1)
    )
    optimized, diagnostics = optimize_centerline(initial, winding_spec, optimization_spec)
    smooth_centerline = _smooth_centerline_from_samples(
        optimized,
        optimization_spec.smooth_export_points,
        method=optimization_spec.smooth_export_method,
    )
    mesh, validation = _tube_mesh_from_centerline(
        smooth_centerline.points,
        winding_spec,
        curvature_radii_override=smooth_centerline.curvature_radii,
    )
    validation["trajectory_optimization"] = diagnostics
    validation["smooth_export"] = {
        "method": optimization_spec.smooth_export_method,
        "input_point_count": int(optimized.shape[0]),
        "export_point_count": int(smooth_centerline.points.shape[0]),
        "curvature_source": "spline_derivatives",
    }

    scene = trimesh.Scene()
    scene.add_geometry(mesh, geom_name="trajectory_optimized_winding_tube")
    if winding_spec.include_reference_y:
        scene.add_geometry(_reference_y_mesh(winding_spec), geom_name="reference_y")
    scene.export(path)
    mesh.export(path.with_suffix(".obj"))
    centerline_path = path.with_name(f"{path.stem}_centerline.csv")
    np.savetxt(centerline_path, smooth_centerline.points, delimiter=",", header="x_nm,y_nm,z_nm", comments="")
    control_path = path.with_name(f"{path.stem}_optimized_control_points.csv")
    np.savetxt(control_path, optimized, delimiter=",", header="x_nm,y_nm,z_nm", comments="")
    path.with_suffix(".json").write_text(
        json.dumps(
            {
                "glb_preview": str(path),
                "obj_preview": str(path.with_suffix(".obj")),
                "centerline_csv": str(centerline_path),
                "optimized_control_points_csv": str(control_path),
                "units": "nm",
                "winding_spec": asdict(winding_spec),
                "optimization_spec": asdict(optimization_spec),
                "initial_centerline_csv": None if initial_centerline_path is None else str(initial_centerline_path),
                "validation": validation,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run dense JAX trajectory optimization for the Y winding.")
    parser.add_argument("--out", default="outputs/y_junction_winding/y_junction_winding_jax_optimized.glb")
    parser.add_argument("--initial-centerline-csv", default=None)
    parser.add_argument("--control-points", type=int, default=300)
    parser.add_argument("--dense-points", type=int, default=900)
    parser.add_argument("--iterations", type=int, default=3500)
    parser.add_argument("--learning-rate", type=float, default=0.025)
    parser.add_argument("--offset-bound-nm", type=float, default=10.0)
    parser.add_argument("--y-weight", type=float, default=260.0)
    parser.add_argument("--upper-branch-contact-start-z-nm", type=float, default=0.0)
    parser.add_argument("--upper-branch-contact-target-nm", type=float, default=0.8)
    parser.add_argument("--maximum-upper-axis-tangent-fraction", type=float, default=0.35)
    parser.add_argument("--upper-branch-contact-weight", type=float, default=0.0)
    parser.add_argument("--upper-branch-circumferential-weight", type=float, default=0.0)
    parser.add_argument("--self-weight", type=float, default=150.0)
    parser.add_argument("--packing-weight", type=float, default=0.4)
    parser.add_argument("--curvature-weight", type=float, default=80.0)
    parser.add_argument("--smoothness-weight", type=float, default=0.9)
    parser.add_argument("--jerk-weight", type=float, default=0.16)
    parser.add_argument("--segment-length-weight", type=float, default=4.0)
    parser.add_argument("--shape-weight", type=float, default=0.28)
    parser.add_argument("--monotonic-z-weight", type=float, default=240.0)
    parser.add_argument("--endpoint-tangent-weight", type=float, default=2.0)
    parser.add_argument("--fix-endpoints", action="store_true")
    parser.add_argument("--report-every", type=int, default=250)
    parser.add_argument("--smooth-export-points", type=int, default=7600)
    parser.add_argument("--smooth-export-method", choices=("pchip", "cubic"), default="pchip")
    parser.add_argument("--centerline-smooth-sigma", type=float, default=2.0)
    parser.add_argument("--hide-reference-y", action="store_true")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    winding_spec = YJunctionWindingSpec(
        centerline_smooth_sigma=args.centerline_smooth_sigma,
        include_reference_y=not args.hide_reference_y,
    )
    optimization_spec = TrajectoryOptimizationSpec(
        control_points=args.control_points,
        dense_points=args.dense_points,
        iterations=args.iterations,
        learning_rate=args.learning_rate,
        offset_bound_nm=args.offset_bound_nm,
        y_weight=args.y_weight,
        upper_branch_contact_start_z_nm=args.upper_branch_contact_start_z_nm,
        upper_branch_contact_target_nm=args.upper_branch_contact_target_nm,
        maximum_upper_axis_tangent_fraction=args.maximum_upper_axis_tangent_fraction,
        upper_branch_contact_weight=args.upper_branch_contact_weight,
        upper_branch_circumferential_weight=args.upper_branch_circumferential_weight,
        self_weight=args.self_weight,
        packing_weight=args.packing_weight,
        curvature_weight=args.curvature_weight,
        smoothness_weight=args.smoothness_weight,
        jerk_weight=args.jerk_weight,
        segment_length_weight=args.segment_length_weight,
        shape_weight=args.shape_weight,
        monotonic_z_weight=args.monotonic_z_weight,
        endpoint_tangent_weight=args.endpoint_tangent_weight,
        fix_endpoints=args.fix_endpoints,
        report_every=args.report_every,
        smooth_export_points=args.smooth_export_points,
        smooth_export_method=args.smooth_export_method,
    )
    path = export_optimized_winding(args.out, winding_spec, optimization_spec, args.initial_centerline_csv)
    print(f"Saved optimized GLB preview to: {path}")
    print(f"Saved OBJ preview to: {path.with_suffix('.obj')}")
    print(f"Saved metadata to: {path.with_suffix('.json')}")


if __name__ == "__main__":
    main()
