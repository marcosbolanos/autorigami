"""Turnwise trajectory optimization for a Y-junction winding."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import trimesh

from autorigami.shape_generation.generate_y_junction_winding import (
    YJunctionWindingSpec,
    _reference_y_mesh,
    _self_contact_flags,
    _smooth_centerline_from_samples,
    _tube_mesh_from_centerline,
    _y_branch_segments,
    _y_clearance_values,
    generate_y_junction_winding_centerline,
)


@dataclass(frozen=True)
class TurnwiseOptimizationSpec:
    turn_count: int = 22
    start_window: int = 0
    end_window: int | None = None
    passes: int = 2
    iterations_per_window: int = 220
    learning_rate: float = 0.015
    offset_bound_nm: float = 7.0
    local_exclusion_points: int = 7
    y_margin_nm: float = 0.18
    self_margin_nm: float = 0.08
    wall_contact_target_nm: float = 0.45
    maximum_axis_tangent_fraction: float = 0.24
    upper_branch_contact_start_z_nm: float = 18.0
    upper_branch_contact_target_nm: float = 0.5
    y_weight: float = 900.0
    wall_contact_weight: float = 130.0
    circumferential_weight: float = 180.0
    upper_branch_contact_weight: float = 0.0
    upper_branch_circumferential_weight: float = 0.0
    self_weight: float = 700.0
    packing_weight: float = 0.12
    curvature_weight: float = 1400.0
    smoothness_weight: float = 2.0
    jerk_weight: float = 0.65
    segment_length_weight: float = 4.0
    shape_weight: float = 0.7
    monotonic_z_weight: float = 450.0
    boundary_weight: float = 18.0
    report_every: int = 55
    smooth_export_points: int = 7600
    smooth_export_method: str = "cubic"


def _window_bounds(point_count: int, turn_count: int) -> list[tuple[int, int]]:
    edges = np.linspace(0, point_count, turn_count + 1, dtype=int)
    return [(int(edges[index]), int(edges[index + 1])) for index in range(turn_count) if edges[index + 1] > edges[index]]


def _axis_wrapping_diagnostics(centerline: np.ndarray, winding_spec: YJunctionWindingSpec) -> dict[str, object]:
    starts, ends = _y_branch_segments(winding_spec)
    axes = ends - starts
    unit_axes = axes / np.linalg.norm(axes, axis=1, keepdims=True)
    axis_lengths_squared = np.sum(axes * axes, axis=1)
    midpoints = 0.5 * (centerline[1:] + centerline[:-1])
    distances = np.empty((midpoints.shape[0], starts.shape[0]), dtype=np.float64)
    for axis_index, (start, axis, length_squared) in enumerate(zip(starts, axes, axis_lengths_squared, strict=True)):
        projection = np.clip(((midpoints - start) @ axis) / length_squared, 0.0, 1.0)
        closest = start + projection[:, None] * axis
        distances[:, axis_index] = np.linalg.norm(midpoints - closest, axis=1)
    nearest_axis = np.argmin(distances, axis=1)
    nearest_upper_axis = np.argmin(distances[:, 1:], axis=1) + 1
    tangents = centerline[1:] - centerline[:-1]
    tangent_norms = np.linalg.norm(tangents, axis=1)
    axis_fraction = np.abs(np.sum(tangents * unit_axes[nearest_axis], axis=1)) / np.maximum(tangent_norms, 1e-12)
    upper_axis_fraction = np.abs(np.sum(tangents * unit_axes[nearest_upper_axis], axis=1)) / np.maximum(tangent_norms, 1e-12)
    clearance = np.min(distances - (winding_spec.channel_radius_nm + 0.5 * winding_spec.rod_diameter_nm + winding_spec.rod_clearance_nm), axis=1)
    upper_clearance = np.min(distances[:, 1:] - (winding_spec.channel_radius_nm + 0.5 * winding_spec.rod_diameter_nm + winding_spec.rod_clearance_nm), axis=1)
    z_values = midpoints[:, 2]
    regions = {
        "stem": z_values < 0.0,
        "early_junction": (z_values >= 0.0) & (z_values < 18.0),
        "upper_transition": (z_values >= 18.0) & (z_values < 38.0),
        "upper_crossing": z_values >= 38.0,
    }
    output: dict[str, object] = {}
    for name, mask in regions.items():
        if bool(np.any(mask)):
            output[name] = {
                "mean_clearance_nm": float(np.mean(clearance[mask])),
                "p90_clearance_nm": float(np.percentile(clearance[mask], 90)),
                "mean_axis_tangent_fraction": float(np.mean(axis_fraction[mask])),
                "p90_axis_tangent_fraction": float(np.percentile(axis_fraction[mask], 90)),
                "mean_upper_branch_clearance_nm": float(np.mean(upper_clearance[mask])),
                "p90_upper_branch_clearance_nm": float(np.percentile(upper_clearance[mask], 90)),
                "mean_upper_axis_tangent_fraction": float(np.mean(upper_axis_fraction[mask])),
                "p90_upper_axis_tangent_fraction": float(np.percentile(upper_axis_fraction[mask], 90)),
            }
    output["overall"] = {
        "mean_clearance_nm": float(np.mean(clearance)),
        "p90_clearance_nm": float(np.percentile(clearance, 90)),
        "mean_axis_tangent_fraction": float(np.mean(axis_fraction)),
        "p90_axis_tangent_fraction": float(np.percentile(axis_fraction, 90)),
        "mean_upper_branch_clearance_nm": float(np.mean(upper_clearance)),
        "p90_upper_branch_clearance_nm": float(np.percentile(upper_clearance, 90)),
        "mean_upper_axis_tangent_fraction": float(np.mean(upper_axis_fraction)),
        "p90_upper_axis_tangent_fraction": float(np.percentile(upper_axis_fraction, 90)),
    }
    return output


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
    distances = jnp.linalg.norm(points[:, None, :] - closest, axis=2)
    return distances, unit_axes


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


def _optimize_window(
    path: np.ndarray,
    start: int,
    end: int,
    winding_spec: YJunctionWindingSpec,
    optimization_spec: TurnwiseOptimizationSpec,
) -> tuple[np.ndarray, dict[str, float]]:
    starts_np, ends_np = _y_branch_segments(winding_spec)
    y_starts = jnp.asarray(starts_np, dtype=jnp.float32)
    y_ends = jnp.asarray(ends_np, dtype=jnp.float32)
    full_initial = jnp.asarray(path, dtype=jnp.float32)
    initial_window = full_initial[start:end]
    required_y_distance = winding_spec.channel_radius_nm + 0.5 * winding_spec.rod_diameter_nm + winding_spec.rod_clearance_nm
    self_target = winding_spec.rod_diameter_nm + optimization_spec.self_margin_nm
    curvature_limit = 1.0 / winding_spec.minimum_curvature_radius_nm
    full_indices = jnp.arange(path.shape[0])
    window_indices = jnp.arange(start, end)
    sequence_distance = jnp.abs(window_indices[:, None] - full_indices[None, :])
    nonlocal_mask = sequence_distance > optimization_spec.local_exclusion_points
    target_segment_length = jnp.mean(jnp.linalg.norm(full_initial[1:] - full_initial[:-1], axis=1))

    context_before = full_initial[:start]
    context_after = full_initial[end:]

    def unpack(offsets: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        bounded_offsets = optimization_spec.offset_bound_nm * jnp.tanh(offsets)
        window = initial_window + bounded_offsets
        candidate = jnp.vstack((context_before, window, context_after))
        return window, candidate

    def loss(offsets: jnp.ndarray) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
        window, candidate = unpack(offsets)
        expanded_start = max(0, start - 4)
        expanded_end = min(path.shape[0], end + 4)
        expanded = candidate[expanded_start:expanded_end]

        y_distances, y_unit_axes = _distance_to_y_axes_with_axes_jax(window, y_starts, y_ends)
        nearest_axis_index = jnp.argmin(y_distances, axis=1)
        y_clearance = jnp.min(y_distances - required_y_distance, axis=1)
        y_violation = jax.nn.softplus(-(y_clearance - optimization_spec.y_margin_nm) * 3.0) / 3.0
        wall_contact_excess = jax.nn.softplus((y_clearance - optimization_spec.wall_contact_target_nm) * 1.8) / 1.8
        upper_branch_clearance = jnp.min(y_distances[:, 1:] - required_y_distance, axis=1)
        upper_branch_active = jax.nn.sigmoid((window[:, 2] - optimization_spec.upper_branch_contact_start_z_nm) * 0.7)
        upper_branch_near = jax.nn.sigmoid((2.8 - upper_branch_clearance) * 1.2)
        upper_branch_contact_excess = upper_branch_active * (
            jax.nn.softplus((upper_branch_clearance - optimization_spec.upper_branch_contact_target_nm) * 1.6) / 1.6
        )

        distances = jnp.linalg.norm(window[:, None, :] - candidate[None, :, :] + 1e-8, axis=2)
        masked_distances = jnp.where(nonlocal_mask, distances, jnp.inf)
        nearest = jnp.min(masked_distances, axis=1)
        self_violation = jnp.where(nonlocal_mask, jax.nn.softplus((self_target - distances) * 2.0) / 2.0, 0.0)
        packing_excess = jax.nn.softplus((nearest - self_target) * 0.65) / 0.65

        tangents = window[1:] - window[:-1]
        tangent_norms = jnp.linalg.norm(tangents, axis=1)
        midpoint_axis_indices = nearest_axis_index[:-1]
        nearest_axes = y_unit_axes[midpoint_axis_indices]
        axis_tangent_fraction = jnp.abs(jnp.sum(tangents * nearest_axes, axis=1)) / jnp.maximum(tangent_norms, 1e-8)
        circumferential_violation = jax.nn.softplus(
            (axis_tangent_fraction - optimization_spec.maximum_axis_tangent_fraction) * 8.0
        ) / 8.0
        upper_branch_axis_index = jnp.argmin(0.5 * (y_distances[:-1, 1:] + y_distances[1:, 1:]), axis=1) + 1
        upper_branch_axes = y_unit_axes[upper_branch_axis_index]
        upper_axis_tangent_fraction = jnp.abs(jnp.sum(tangents * upper_branch_axes, axis=1)) / jnp.maximum(tangent_norms, 1e-8)
        upper_circumferential_active = 0.5 * (
            upper_branch_active[:-1] * upper_branch_near[:-1]
            + upper_branch_active[1:] * upper_branch_near[1:]
        )
        upper_branch_circumferential_violation = upper_circumferential_active * (
            jax.nn.softplus((upper_axis_tangent_fraction - optimization_spec.maximum_axis_tangent_fraction) * 8.0) / 8.0
        )

        radii = _curvature_radii_jax(expanded)
        curvature = jnp.where(jnp.isfinite(radii), 1.0 / radii, 0.0)
        curvature_violation = jax.nn.softplus((curvature - curvature_limit) * 24.0) / 24.0

        second = expanded[:-2] - 2.0 * expanded[1:-1] + expanded[2:]
        third = expanded[:-3] - 3.0 * expanded[1:-2] + 3.0 * expanded[2:-1] - expanded[3:]
        segment_lengths = jnp.linalg.norm(expanded[1:] - expanded[:-1], axis=1)
        displacement = window - initial_window
        dz = window[1:, 2] - window[:-1, 2]
        z_violation = jax.nn.softplus((-dz) * 5.0) / 5.0
        boundary_residuals = []
        if start > 0:
            boundary_residuals.append(window[0] - initial_window[0])
        if end < path.shape[0]:
            boundary_residuals.append(window[-1] - initial_window[-1])
        boundary = jnp.concatenate(boundary_residuals) if boundary_residuals else jnp.zeros((1,), dtype=jnp.float32)

        terms = {
            "y": optimization_spec.y_weight * jnp.mean(y_violation**2),
            "wall_contact": optimization_spec.wall_contact_weight * jnp.mean(wall_contact_excess**2),
            "circumferential": optimization_spec.circumferential_weight * jnp.mean(circumferential_violation**2),
            "upper_branch_contact": optimization_spec.upper_branch_contact_weight * jnp.mean(upper_branch_contact_excess**2),
            "upper_branch_circumferential": optimization_spec.upper_branch_circumferential_weight * jnp.mean(upper_branch_circumferential_violation**2),
            "self": optimization_spec.self_weight * jnp.sum(self_violation**2) / window.shape[0],
            "packing": optimization_spec.packing_weight * jnp.mean(packing_excess**2),
            "curvature": optimization_spec.curvature_weight * jnp.mean(curvature_violation**2),
            "smoothness": optimization_spec.smoothness_weight * jnp.mean(jnp.sum(second**2, axis=1)),
            "jerk": optimization_spec.jerk_weight * jnp.mean(jnp.sum(third**2, axis=1)),
            "segment_length": optimization_spec.segment_length_weight * jnp.mean((segment_lengths - target_segment_length) ** 2),
            "shape": optimization_spec.shape_weight * jnp.mean(jnp.sum(displacement**2, axis=1)),
            "monotonic_z": optimization_spec.monotonic_z_weight * jnp.mean(z_violation**2),
            "boundary": optimization_spec.boundary_weight * jnp.mean(boundary**2),
        }
        total = sum(terms.values())
        metrics = {
            **terms,
            "total": total,
            "min_y_clearance": jnp.min(y_clearance),
            "mean_upper_branch_clearance": jnp.mean(upper_branch_clearance),
            "nearest_nonlocal": jnp.min(nearest),
            "min_curvature_radius": jnp.min(radii),
            "mean_y_clearance": jnp.mean(y_clearance),
            "mean_axis_tangent_fraction": jnp.mean(axis_tangent_fraction),
            "mean_upper_axis_tangent_fraction": jnp.mean(upper_axis_tangent_fraction),
            "max_offset": jnp.max(jnp.linalg.norm(displacement, axis=1)),
        }
        return total, metrics

    value_and_grad = jax.jit(jax.value_and_grad(lambda values: loss(values)[0]))
    metrics_fn = jax.jit(lambda values: loss(values)[1])
    offsets = jnp.zeros_like(initial_window)
    optimizer = optax.chain(optax.clip_by_global_norm(3.0), optax.adam(optimization_spec.learning_rate))
    state = optimizer.init(offsets)

    for step in range(optimization_spec.iterations_per_window + 1):
        value, gradient = value_and_grad(offsets)
        updates, state = optimizer.update(gradient, state, offsets)
        offsets = optax.apply_updates(offsets, updates)
        if step % optimization_spec.report_every == 0 or step == optimization_spec.iterations_per_window:
            metrics = {key: float(value) for key, value in metrics_fn(offsets).items()}
            print(
                f"window={start}:{end} step={step} total={metrics['total']:.2f} "
                f"near={metrics['nearest_nonlocal']:.3f} min_y={metrics['min_y_clearance']:.3f} "
                f"min_R={metrics['min_curvature_radius']:.3f}",
                flush=True,
            )

    window, _ = unpack(offsets)
    return np.asarray(window, dtype=np.float64), {key: float(value) for key, value in metrics_fn(offsets).items()}


def optimize_turnwise(
    initial_path: np.ndarray,
    winding_spec: YJunctionWindingSpec,
    optimization_spec: TurnwiseOptimizationSpec,
) -> tuple[np.ndarray, dict[str, object]]:
    path = initial_path.copy()
    windows = _window_bounds(path.shape[0], optimization_spec.turn_count)
    active_end = optimization_spec.turn_count if optimization_spec.end_window is None else optimization_spec.end_window
    windows = windows[optimization_spec.start_window:active_end]
    history: list[dict[str, object]] = []
    for pass_index in range(optimization_spec.passes):
        order = windows if pass_index % 2 == 0 else list(reversed(windows))
        for start, end in order:
            optimized_window, metrics = _optimize_window(path, start, end, winding_spec, optimization_spec)
            path[start:end] = optimized_window
            flags, nearest = _self_contact_flags(path, winding_spec.rod_diameter_nm)
            history.append(
                {
                    "pass": pass_index,
                    "start": start,
                    "end": end,
                    "window_metrics": metrics,
                    "global_nearest_nonlocal_centerline_distance_nm": float(nearest),
                    "global_self_contact_points": int(np.count_nonzero(flags)),
                    "global_minimum_y_clearance_nm": float(np.min(_y_clearance_values(path, winding_spec))),
                }
            )
    return path, {"optimizer": "turnwise_jax_window_adam", "optimization_spec": asdict(optimization_spec), "history": history}


def export_turnwise_winding(
    output_path: str | Path,
    winding_spec: YJunctionWindingSpec,
    optimization_spec: TurnwiseOptimizationSpec,
    initial_centerline_path: str | Path | None = None,
) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    initial = (
        generate_y_junction_winding_centerline(winding_spec)
        if initial_centerline_path is None
        else np.loadtxt(initial_centerline_path, delimiter=",", skiprows=1)
    )
    optimized, diagnostics = optimize_turnwise(initial, winding_spec, optimization_spec)
    smooth = _smooth_centerline_from_samples(
        optimized,
        optimization_spec.smooth_export_points,
        method=optimization_spec.smooth_export_method,
    )
    mesh, validation = _tube_mesh_from_centerline(smooth.points, winding_spec, curvature_radii_override=smooth.curvature_radii)
    validation["turnwise_optimization"] = diagnostics
    validation["axis_wrapping_diagnostics"] = _axis_wrapping_diagnostics(smooth.points, winding_spec)
    validation["smooth_export"] = {
        "method": optimization_spec.smooth_export_method,
        "input_point_count": int(optimized.shape[0]),
        "export_point_count": int(smooth.points.shape[0]),
        "curvature_source": "spline_derivatives",
    }

    scene = trimesh.Scene()
    scene.add_geometry(mesh, geom_name="turnwise_optimized_winding_tube")
    if winding_spec.include_reference_y:
        scene.add_geometry(_reference_y_mesh(winding_spec), geom_name="reference_y")
    scene.export(path)
    mesh.export(path.with_suffix(".obj"))
    centerline_path = path.with_name(f"{path.stem}_centerline.csv")
    control_path = path.with_name(f"{path.stem}_turnwise_points.csv")
    np.savetxt(centerline_path, smooth.points, delimiter=",", header="x_nm,y_nm,z_nm", comments="")
    np.savetxt(control_path, optimized, delimiter=",", header="x_nm,y_nm,z_nm", comments="")
    path.with_suffix(".json").write_text(
        json.dumps(
            {
                "glb_preview": str(path),
                "obj_preview": str(path.with_suffix(".obj")),
                "centerline_csv": str(centerline_path),
                "turnwise_points_csv": str(control_path),
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
    parser = argparse.ArgumentParser(description="Optimize a Y-junction winding one turn-sized window at a time.")
    parser.add_argument("--out", default="outputs/y_junction_winding/y_junction_winding_turnwise.glb")
    parser.add_argument("--initial-centerline-csv", default=None)
    parser.add_argument("--turn-count", type=int, default=22)
    parser.add_argument("--start-window", type=int, default=0)
    parser.add_argument("--end-window", type=int, default=None)
    parser.add_argument("--passes", type=int, default=2)
    parser.add_argument("--iterations-per-window", type=int, default=220)
    parser.add_argument("--learning-rate", type=float, default=0.015)
    parser.add_argument("--offset-bound-nm", type=float, default=7.0)
    parser.add_argument("--local-exclusion-points", type=int, default=7)
    parser.add_argument("--self-margin-nm", type=float, default=0.08)
    parser.add_argument("--y-margin-nm", type=float, default=0.18)
    parser.add_argument("--wall-contact-target-nm", type=float, default=0.45)
    parser.add_argument("--maximum-axis-tangent-fraction", type=float, default=0.24)
    parser.add_argument("--upper-branch-contact-start-z-nm", type=float, default=18.0)
    parser.add_argument("--upper-branch-contact-target-nm", type=float, default=0.5)
    parser.add_argument("--y-weight", type=float, default=900.0)
    parser.add_argument("--wall-contact-weight", type=float, default=130.0)
    parser.add_argument("--circumferential-weight", type=float, default=180.0)
    parser.add_argument("--upper-branch-contact-weight", type=float, default=0.0)
    parser.add_argument("--upper-branch-circumferential-weight", type=float, default=0.0)
    parser.add_argument("--self-weight", type=float, default=700.0)
    parser.add_argument("--packing-weight", type=float, default=0.12)
    parser.add_argument("--curvature-weight", type=float, default=1400.0)
    parser.add_argument("--smoothness-weight", type=float, default=2.0)
    parser.add_argument("--jerk-weight", type=float, default=0.65)
    parser.add_argument("--segment-length-weight", type=float, default=4.0)
    parser.add_argument("--shape-weight", type=float, default=0.7)
    parser.add_argument("--monotonic-z-weight", type=float, default=450.0)
    parser.add_argument("--boundary-weight", type=float, default=18.0)
    parser.add_argument("--report-every", type=int, default=55)
    parser.add_argument("--smooth-export-points", type=int, default=7600)
    parser.add_argument("--smooth-export-method", choices=("cubic", "pchip"), default="cubic")
    parser.add_argument("--hide-reference-y", action="store_true")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    winding_spec = YJunctionWindingSpec(include_reference_y=not args.hide_reference_y)
    optimization_spec = TurnwiseOptimizationSpec(
        turn_count=args.turn_count,
        start_window=args.start_window,
        end_window=args.end_window,
        passes=args.passes,
        iterations_per_window=args.iterations_per_window,
        learning_rate=args.learning_rate,
        offset_bound_nm=args.offset_bound_nm,
        local_exclusion_points=args.local_exclusion_points,
        self_margin_nm=args.self_margin_nm,
        y_margin_nm=args.y_margin_nm,
        wall_contact_target_nm=args.wall_contact_target_nm,
        maximum_axis_tangent_fraction=args.maximum_axis_tangent_fraction,
        upper_branch_contact_start_z_nm=args.upper_branch_contact_start_z_nm,
        upper_branch_contact_target_nm=args.upper_branch_contact_target_nm,
        y_weight=args.y_weight,
        wall_contact_weight=args.wall_contact_weight,
        circumferential_weight=args.circumferential_weight,
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
        boundary_weight=args.boundary_weight,
        report_every=args.report_every,
        smooth_export_points=args.smooth_export_points,
        smooth_export_method=args.smooth_export_method,
    )
    path = export_turnwise_winding(args.out, winding_spec, optimization_spec, args.initial_centerline_csv)
    print(f"Saved turnwise optimized GLB preview to: {path}")
    print(f"Saved OBJ preview to: {path.with_suffix('.obj')}")
    print(f"Saved metadata to: {path.with_suffix('.json')}")


if __name__ == "__main__":
    main()
