"""Generate a dense diagnostic tube winding around a Y junction."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline, PchipInterpolator
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
import trimesh


Vec3 = np.ndarray


@dataclass(frozen=True)
class YJunctionWindingSpec:
    channel_radius_nm: float = 24.0
    branch_length_nm: float = 90.0
    upper_branch_angle_degrees: float = 60.0
    rod_diameter_nm: float = 5.2
    rod_clearance_nm: float = 0.2
    minimum_curvature_radius_nm: float = 6.0
    lower_z_nm: float = -90.0
    top_z_nm: float = 52.0
    inside_start_z_nm: float = 34.0
    inside_blend_height_nm: float = 14.0
    upper_follow_start_z_nm: float = 0.0
    upper_follow_blend_height_nm: float = 45.0
    junction_flare_nm: float = 0.0
    y_projection_iterations: int = 14
    y_projection_smooth_sigma: float = 0.0
    outside_y_scale: float = 1.25
    inside_y_scale: float = 1.9
    winding_pitch_nm: float = 5.35
    pitch_search_max_nm: float = 14.0
    pitch_candidate_count: int = 36
    turn_samples: int = 80
    target_contact_gap_nm: float = 0.06
    upper_funnel_arc_degrees: float = 70.0
    upper_vertical_undulation_nm: float = 0.4
    optimize_control_points: bool = False
    optimizer_control_points: int = 96
    optimizer_max_nfev: int = 160
    optimizer_xy_bound_nm: float = 3.0
    y_clearance_weight: float = 16.0
    self_contact_weight: float = 10.0
    packing_weight: float = 1.6
    smoothness_weight: float = 0.18
    shape_preservation_weight: float = 0.12
    phase_offset_degrees: float = 0.0
    path_samples: int = 2600
    tube_radial_samples: int = 28
    centerline_smooth_sigma: float = 2.0
    include_reference_y: bool = True


@dataclass(frozen=True)
class WindingResult:
    centerline: np.ndarray
    mesh: trimesh.Trimesh
    validation: dict[str, object]


@dataclass(frozen=True)
class SmoothCenterline:
    points: np.ndarray
    curvature_radii: np.ndarray


def _smoothstep(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, 0.0, 1.0)
    return clipped * clipped * (3.0 - 2.0 * clipped)


def _branch_top_x(spec: YJunctionWindingSpec) -> float:
    return spec.branch_length_nm * np.sin(np.deg2rad(spec.upper_branch_angle_degrees))


def _branch_top_z(spec: YJunctionWindingSpec) -> float:
    return spec.branch_length_nm * np.cos(np.deg2rad(spec.upper_branch_angle_degrees))


def _y_branch_segments(spec: YJunctionWindingSpec) -> tuple[np.ndarray, np.ndarray]:
    theta = np.deg2rad(spec.upper_branch_angle_degrees)
    starts = np.array(
        [
            [0.0, 0.0, -spec.branch_length_nm],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    ends = np.array(
        [
            [0.0, 0.0, 0.0],
            spec.branch_length_nm * np.array([-np.sin(theta), 0.0, np.cos(theta)], dtype=np.float64),
            spec.branch_length_nm * np.array([np.sin(theta), 0.0, np.cos(theta)], dtype=np.float64),
        ],
        dtype=np.float64,
    )
    return starts, ends


def _distance_to_y_axes(points: np.ndarray, spec: YJunctionWindingSpec) -> np.ndarray:
    starts, ends = _y_branch_segments(spec)
    axes = ends - starts
    axis_lengths_squared = np.sum(axes * axes, axis=1)
    distances = np.empty((points.shape[0], starts.shape[0]), dtype=np.float64)
    for index, (start, axis, length_squared) in enumerate(zip(starts, axes, axis_lengths_squared, strict=True)):
        projection = np.clip(((points - start) @ axis) / length_squared, 0.0, 1.0)
        closest = start + projection[:, None] * axis
        distances[:, index] = np.linalg.norm(points - closest, axis=1)
    return distances


def _y_clearance_values(centerline: np.ndarray, spec: YJunctionWindingSpec) -> np.ndarray:
    required_radius = spec.channel_radius_nm + 0.5 * spec.rod_diameter_nm + spec.rod_clearance_nm
    return np.min(_distance_to_y_axes(centerline, spec) - required_radius, axis=1)


def _push_points_outside_y(points: np.ndarray, spec: YJunctionWindingSpec) -> np.ndarray:
    corrected = points.copy()
    starts, ends = _y_branch_segments(spec)
    axes = ends - starts
    axis_lengths_squared = np.sum(axes * axes, axis=1)
    required_radius = spec.channel_radius_nm + 0.5 * spec.rod_diameter_nm + spec.rod_clearance_nm
    from scipy.ndimage import gaussian_filter1d

    for _ in range(spec.y_projection_iterations):
        distances = _distance_to_y_axes(corrected, spec)
        nearest_segment = np.argmin(distances, axis=1)
        nearest_distance = distances[np.arange(corrected.shape[0]), nearest_segment]
        penetrating = nearest_distance < required_radius
        if not bool(np.any(penetrating)):
            break
        displacement = np.zeros_like(corrected)
        for point_index in np.where(penetrating)[0]:
            segment_index = int(nearest_segment[point_index])
            start = starts[segment_index]
            axis = axes[segment_index]
            projection = np.clip(
                ((corrected[point_index] - start) @ axis) / axis_lengths_squared[segment_index],
                0.0,
                1.0,
            )
            closest = start + projection * axis
            direction = corrected[point_index] - closest
            norm = float(np.linalg.norm(direction))
            if norm < 1e-9:
                direction = np.cross(axis, np.array([0.0, 1.0, 0.0], dtype=np.float64))
                norm = float(np.linalg.norm(direction))
            direction /= norm
            target = closest + direction * (required_radius + 0.08)
            if spec.y_projection_smooth_sigma == 0.0:
                corrected[point_index] = target
            else:
                displacement[point_index] = target - corrected[point_index]
        if spec.y_projection_smooth_sigma > 0.0:
            displacement = gaussian_filter1d(
                displacement,
                sigma=spec.y_projection_smooth_sigma,
                axis=0,
                mode="nearest",
            )
            corrected += displacement
    return corrected


def _upper_branch_x_at_z(z_values: np.ndarray, spec: YJunctionWindingSpec) -> np.ndarray:
    top_x = _branch_top_x(spec)
    top_z = _branch_top_z(spec)
    return top_x * _smoothstep(np.clip(z_values, 0.0, top_z) / top_z)


def _resample_centerline(centerline: np.ndarray, point_count: int) -> np.ndarray:
    segment_lengths = np.linalg.norm(np.diff(centerline, axis=0), axis=1)
    arclength = np.concatenate((np.array([0.0], dtype=np.float64), np.cumsum(segment_lengths)))
    if arclength[-1] <= 0.0:
        raise ValueError("centerline arclength must be > 0")
    target = np.linspace(0.0, float(arclength[-1]), point_count)
    return np.column_stack(
        (
            np.interp(target, arclength, centerline[:, 0]),
            np.interp(target, arclength, centerline[:, 1]),
            np.interp(target, arclength, centerline[:, 2]),
        )
    )


def _smooth_centerline_from_samples(
    centerline: np.ndarray,
    point_count: int,
    method: str = "pchip",
) -> SmoothCenterline:
    segment_lengths = np.linalg.norm(np.diff(centerline, axis=0), axis=1)
    arclength = np.concatenate((np.array([0.0], dtype=np.float64), np.cumsum(segment_lengths)))
    keep = np.concatenate(([True], np.diff(arclength) > 1e-9))
    arclength = arclength[keep]
    points = centerline[keep]
    if arclength[-1] <= 0.0:
        raise ValueError("centerline arclength must be > 0")
    if method == "cubic":
        splines = [CubicSpline(arclength, points[:, axis], bc_type="natural") for axis in range(3)]
    elif method == "pchip":
        splines = [PchipInterpolator(arclength, points[:, axis]) for axis in range(3)]
    else:
        raise ValueError("method must be 'cubic' or 'pchip'")
    target = np.linspace(0.0, float(arclength[-1]), point_count)
    sampled = np.column_stack([spline(target) for spline in splines])
    first = np.column_stack([spline(target, 1) for spline in splines])
    second = np.column_stack([spline(target, 2) for spline in splines])
    speed = np.linalg.norm(first, axis=1)
    cross = np.linalg.norm(np.cross(first, second), axis=1)
    curvature = np.divide(cross, speed**3, out=np.zeros_like(cross), where=speed > 1e-9)
    radii = np.divide(1.0, curvature, out=np.full_like(curvature, np.inf), where=curvature > 1e-12)
    radii[:3] = np.inf
    radii[-3:] = np.inf
    return SmoothCenterline(points=sampled, curvature_radii=radii)


def _turn_segment(
    z_start: float,
    z_end: float,
    phase_start: float,
    spec: YJunctionWindingSpec,
) -> np.ndarray:
    q = np.linspace(0.0, 1.0, spec.turn_samples + 1)
    phase = phase_start + 2.0 * np.pi * q
    z_linear = z_start + (z_end - z_start) * q
    upper_amount = _smoothstep(
        (z_linear - spec.upper_follow_start_z_nm) / spec.upper_follow_blend_height_nm
    )
    inside_amount = _smoothstep((z_linear - spec.inside_start_z_nm) / spec.inside_blend_height_nm)
    z_values = z_linear + spec.upper_vertical_undulation_nm * upper_amount * inside_amount * np.sin(2.0 * np.pi * q)

    wrap_radius = spec.channel_radius_nm + 0.5 * spec.rod_diameter_nm + spec.rod_clearance_nm
    flare_amount = (1.0 - upper_amount) * _smoothstep(z_values / max(spec.upper_follow_start_z_nm, 1e-9))
    lower_radius = wrap_radius + spec.junction_flare_nm * flare_amount
    branch_x = _upper_branch_x_at_z(z_values, spec)
    funnel_angle = np.deg2rad(spec.upper_funnel_arc_degrees) * (upper_amount - 0.5) * np.sin(phase)
    funnel_x_scale = 1.0 + 0.18 * upper_amount * np.sin(funnel_angle)
    funnel_y_scale = 1.0 + 0.12 * upper_amount * np.cos(funnel_angle)

    lower = np.column_stack(
        (
            lower_radius * np.cos(phase),
            lower_radius * np.sin(phase),
            z_values,
        )
    )
    outside = np.column_stack(
        (
            funnel_x_scale * (branch_x + wrap_radius) * np.cos(phase),
            funnel_y_scale * spec.outside_y_scale * wrap_radius * np.sin(phase),
            z_values,
        )
    )
    inside = np.column_stack(
        (
            funnel_x_scale * (branch_x + wrap_radius) * np.cos(phase),
            funnel_y_scale * spec.inside_y_scale * wrap_radius * np.sin(phase) * np.cos(phase),
            z_values,
        )
    )

    upper = (1.0 - inside_amount[:, None]) * outside + inside_amount[:, None] * inside
    return _push_points_outside_y((1.0 - upper_amount[:, None]) * lower + upper_amount[:, None] * upper, spec)


def _candidate_metrics(centerline: np.ndarray, spec: YJunctionWindingSpec) -> tuple[bool, float, float, int, float]:
    radii = _centerline_curvature_radii(centerline)
    self_flags, nearest_nonlocal = _self_contact_flags(centerline, spec.rod_diameter_nm)
    y_clearance = _y_clearance_values(centerline, spec)
    minimum_radius = float(np.min(radii))
    self_contact_points = int(np.count_nonzero(self_flags))
    minimum_y_clearance = float(np.min(y_clearance))
    feasible = (
        minimum_radius >= spec.minimum_curvature_radius_nm
        and self_contact_points == 0
        and minimum_y_clearance >= 0.0
    )
    return feasible, nearest_nonlocal, minimum_radius, self_contact_points, minimum_y_clearance


def _generate_turn_by_turn_centerline(
    spec: YJunctionWindingSpec,
) -> tuple[np.ndarray, dict[str, object]]:
    pitch_min = spec.rod_diameter_nm + spec.target_contact_gap_nm
    pitch_candidates = np.linspace(pitch_min, spec.pitch_search_max_nm, spec.pitch_candidate_count)
    points = [_turn_segment(spec.lower_z_nm, spec.lower_z_nm + pitch_min, np.deg2rad(spec.phase_offset_degrees), spec)[0]]
    phase = np.deg2rad(spec.phase_offset_degrees)
    current_z = spec.lower_z_nm
    chosen_pitches: list[float] = []
    rejected_candidates = 0

    while spec.top_z_nm - current_z >= pitch_min - 1e-9:
        best_segment: np.ndarray | None = None
        best_score = np.inf
        best_pitch = pitch_min
        best_nearest = np.inf
        best_minimum_radius = 0.0
        fallback_segment: np.ndarray | None = None
        fallback_score: tuple[int, int, float, float] = (-1, -1, -np.inf, -np.inf)
        fallback_pitch = pitch_min

        remaining_height = spec.top_z_nm - current_z
        for pitch in pitch_candidates[pitch_candidates <= remaining_height + 1e-9]:
            next_z = current_z + float(pitch)
            segment = _turn_segment(current_z, next_z, phase, spec)[1:]
            trial = np.vstack((np.asarray(points), segment))
            feasible, nearest_nonlocal, minimum_radius, self_contact_points, minimum_y_clearance = _candidate_metrics(trial, spec)
            if feasible:
                score = abs(nearest_nonlocal - (spec.rod_diameter_nm + spec.target_contact_gap_nm))
                if score < best_score:
                    best_score = score
                    best_segment = segment
                    best_pitch = next_z - current_z
                    best_nearest = nearest_nonlocal
                    best_minimum_radius = minimum_radius
            else:
                rejected_candidates += 1
                no_contact = int(self_contact_points == 0)
                no_y_collision = int(minimum_y_clearance >= 0.0)
                score = (no_y_collision, no_contact, minimum_radius, nearest_nonlocal)
                if score > fallback_score:
                    fallback_score = score
                    fallback_segment = segment
                    fallback_pitch = next_z - current_z

        if best_segment is None:
            if fallback_segment is None:
                raise RuntimeError("turn search produced no candidate segment")
            best_segment = fallback_segment
            best_pitch = fallback_pitch

        points.extend(best_segment)
        chosen_pitches.append(best_pitch)
        current_z += best_pitch
        phase += 2.0 * np.pi

    raw_centerline = np.asarray(points, dtype=np.float64)
    centerline = _push_points_outside_y(_resample_centerline(raw_centerline, spec.path_samples), spec)
    if spec.centerline_smooth_sigma > 0.0:
        from scipy.ndimage import gaussian_filter1d

        centerline[:, :2] = gaussian_filter1d(
            centerline[:, :2],
            sigma=spec.centerline_smooth_sigma,
            axis=0,
            mode="nearest",
        )
        centerline = _push_points_outside_y(centerline, spec)
    diagnostics = {
        "construction": "turn_by_turn_hard_feasibility",
        "turn_count": len(chosen_pitches),
        "mean_chosen_pitch_nm": float(np.mean(chosen_pitches)),
        "minimum_chosen_pitch_nm": float(np.min(chosen_pitches)),
        "maximum_chosen_pitch_nm": float(np.max(chosen_pitches)),
        "candidate_pitches_nm": pitch_candidates.tolist(),
        "rejected_candidate_count": rejected_candidates,
        "last_feasible_nearest_nonlocal_centerline_distance_nm": float(best_nearest),
        "last_feasible_minimum_curvature_radius_nm": float(best_minimum_radius),
    }
    return centerline, diagnostics


def _nonlocal_nearest_distances(points: np.ndarray, rod_diameter_nm: float) -> np.ndarray:
    tree = cKDTree(points)
    neighbor_count = min(48, points.shape[0])
    distances, indices = tree.query(points, k=neighbor_count)
    segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    arclength = np.concatenate((np.array([0.0], dtype=np.float64), np.cumsum(segment_lengths)))
    nearest = np.full(points.shape[0], np.inf, dtype=np.float64)
    for index, neighbors in enumerate(indices):
        for distance, neighbor in zip(distances[index, 1:], neighbors[1:], strict=True):
            if abs(arclength[index] - arclength[int(neighbor)]) < 2.0 * rod_diameter_nm:
                continue
            nearest[index] = float(distance)
            break
    return nearest


def _optimize_centerline_against_y(
    centerline: np.ndarray,
    spec: YJunctionWindingSpec,
) -> tuple[np.ndarray, dict[str, object]]:
    control = _resample_centerline(centerline, spec.optimizer_control_points)
    z_values = control[:, 2].copy()
    initial_xy = control[:, :2].copy()
    moving = np.arange(1, control.shape[0] - 1)
    target_contact = spec.rod_diameter_nm + spec.target_contact_gap_nm

    def unpack(values: np.ndarray) -> np.ndarray:
        candidate = control.copy()
        candidate[moving, :2] = values.reshape((-1, 2))
        candidate[:, 2] = z_values
        return candidate

    def residual(values: np.ndarray) -> np.ndarray:
        candidate = unpack(values)
        y_clearance = _y_clearance_values(candidate, spec)
        nearest = _nonlocal_nearest_distances(candidate, spec.rod_diameter_nm)
        finite_nearest = nearest[np.isfinite(nearest)]
        second_difference = candidate[:-2] - 2.0 * candidate[1:-1] + candidate[2:]
        displacement = candidate[:, :2] - initial_xy
        residuals = [
            spec.y_clearance_weight * np.maximum(0.0, -y_clearance),
            spec.self_contact_weight * np.maximum(0.0, target_contact - finite_nearest),
            spec.packing_weight * np.maximum(0.0, finite_nearest - target_contact),
            spec.smoothness_weight * second_difference.reshape(-1),
            spec.shape_preservation_weight * displacement.reshape(-1),
        ]
        return np.concatenate(residuals)

    initial_values = initial_xy[moving].reshape(-1)
    lower = initial_values - spec.optimizer_xy_bound_nm
    upper = initial_values + spec.optimizer_xy_bound_nm
    result = least_squares(
        residual,
        initial_values,
        bounds=(lower, upper),
        max_nfev=spec.optimizer_max_nfev,
        x_scale="jac",
        loss="soft_l1",
        f_scale=1.0,
        verbose=0,
    )
    optimized_control = unpack(result.x)
    optimized = _push_points_outside_y(_resample_centerline(optimized_control, spec.path_samples), spec)
    diagnostics = {
        "optimizer": "scipy.optimize.least_squares",
        "control_point_count": spec.optimizer_control_points,
        "optimized_xy_only": True,
        "success": bool(result.success),
        "status": int(result.status),
        "message": str(result.message),
        "nfev": int(result.nfev),
        "initial_cost": float(0.5 * np.sum(residual(initial_values) ** 2)),
        "final_cost": float(result.cost),
        "minimum_y_clearance_nm": float(np.min(_y_clearance_values(optimized, spec))),
    }
    return optimized, diagnostics


def generate_y_junction_winding_centerline(
    spec: YJunctionWindingSpec = YJunctionWindingSpec(),
) -> np.ndarray:
    _validate_spec(spec)
    centerline, _ = _generate_turn_by_turn_centerline(spec)
    if spec.optimize_control_points:
        centerline, _ = _optimize_centerline_against_y(centerline, spec)
    return centerline


def _parallel_transport_frames(centerline: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tangents = np.gradient(centerline, axis=0)
    tangents /= np.linalg.norm(tangents, axis=1, keepdims=True)
    normals = np.empty_like(tangents)
    binormals = np.empty_like(tangents)

    reference = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(float(np.dot(reference, tangents[0]))) > 0.95:
        reference = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    normals[0] = reference - np.dot(reference, tangents[0]) * tangents[0]
    normals[0] /= np.linalg.norm(normals[0])
    binormals[0] = np.cross(tangents[0], normals[0])

    for index in range(1, centerline.shape[0]):
        candidate = normals[index - 1] - np.dot(normals[index - 1], tangents[index]) * tangents[index]
        if np.linalg.norm(candidate) < 1e-10:
            candidate = binormals[index - 1] - np.dot(binormals[index - 1], tangents[index]) * tangents[index]
        normals[index] = candidate / np.linalg.norm(candidate)
        binormals[index] = np.cross(tangents[index], normals[index])
    return tangents, normals, binormals


def _centerline_curvature_radii(centerline: np.ndarray) -> np.ndarray:
    previous = centerline[:-2]
    current = centerline[1:-1]
    following = centerline[2:]
    a = current - previous
    b = following - current
    c = following - previous
    cross = np.linalg.norm(np.cross(a, b), axis=1)
    denom = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) * np.linalg.norm(c, axis=1)
    curvature = np.divide(2.0 * cross, denom, out=np.zeros_like(cross), where=denom > 1e-12)
    radii = np.full(centerline.shape[0], np.inf, dtype=np.float64)
    radii[1:-1] = np.divide(1.0, curvature, out=np.full_like(curvature, np.inf), where=curvature > 1e-12)
    radii[:3] = np.inf
    radii[-3:] = np.inf
    return radii


def _self_contact_flags(centerline: np.ndarray, rod_diameter_nm: float) -> tuple[np.ndarray, float]:
    tree = cKDTree(centerline)
    neighbor_count = min(96, centerline.shape[0])
    distances, indices = tree.query(centerline, k=neighbor_count)
    segment_lengths = np.linalg.norm(np.diff(centerline, axis=0), axis=1)
    arclength = np.concatenate((np.array([0.0], dtype=np.float64), np.cumsum(segment_lengths)))
    flags = np.zeros(centerline.shape[0], dtype=bool)
    nearest = np.inf
    for index, neighbors in enumerate(indices):
        for distance, neighbor in zip(distances[index, 1:], neighbors[1:], strict=True):
            if abs(arclength[index] - arclength[int(neighbor)]) < 2.0 * rod_diameter_nm:
                continue
            nearest = min(nearest, float(distance))
            if distance < rod_diameter_nm:
                flags[index] = True
                flags[int(neighbor)] = True
    return flags, nearest


def _tube_mesh_from_centerline(
    centerline: np.ndarray,
    spec: YJunctionWindingSpec,
    curvature_radii_override: np.ndarray | None = None,
) -> tuple[trimesh.Trimesh, dict[str, object]]:
    _, normals, binormals = _parallel_transport_frames(centerline)
    tube_radius = 0.5 * spec.rod_diameter_nm
    ring_angles = np.linspace(0.0, 2.0 * np.pi, spec.tube_radial_samples, endpoint=False)
    vertices = np.empty((centerline.shape[0], spec.tube_radial_samples, 3), dtype=np.float64)
    for index, center in enumerate(centerline):
        ring = (
            np.cos(ring_angles)[:, None] * normals[index]
            + np.sin(ring_angles)[:, None] * binormals[index]
        )
        vertices[index] = center + tube_radius * ring

    faces: list[tuple[int, int, int]] = []
    radial_count = spec.tube_radial_samples
    for index in range(centerline.shape[0] - 1):
        base = index * radial_count
        next_base = (index + 1) * radial_count
        for side in range(radial_count):
            a = base + side
            b = base + (side + 1) % radial_count
            c = next_base + side
            d = next_base + (side + 1) % radial_count
            faces.append((a, c, b))
            faces.append((b, c, d))

    curvature_radii = (
        _centerline_curvature_radii(centerline)
        if curvature_radii_override is None
        else curvature_radii_override
    )
    curvature_flags = curvature_radii < spec.minimum_curvature_radius_nm
    self_flags, nearest_nonlocal = _self_contact_flags(centerline, spec.rod_diameter_nm)
    y_clearance = _y_clearance_values(centerline, spec)
    y_flags = y_clearance < 0.0
    colors = np.empty((centerline.shape[0], 4), dtype=np.uint8)
    colors[:] = np.array([60, 180, 255, 255], dtype=np.uint8)
    colors[curvature_flags] = np.array([255, 60, 60, 255], dtype=np.uint8)
    colors[self_flags] = np.array([255, 190, 40, 255], dtype=np.uint8)
    colors[curvature_flags & self_flags] = np.array([255, 0, 220, 255], dtype=np.uint8)
    colors[y_flags] = np.array([110, 60, 255, 255], dtype=np.uint8)

    mesh = trimesh.Trimesh(
        vertices=vertices.reshape((-1, 3)),
        faces=np.asarray(faces),
        process=False,
    )
    mesh.visual.vertex_colors = np.repeat(colors, radial_count, axis=0)
    validation = {
        "minimum_centerline_curvature_radius_nm": float(np.min(curvature_radii)),
        "curvature_violation_points": int(np.count_nonzero(curvature_flags)),
        "curvature_point_count": int(curvature_flags.size),
        "self_contact_detected": bool(np.any(self_flags)),
        "self_contact_points": int(np.count_nonzero(self_flags)),
        "nearest_nonlocal_centerline_distance_nm": float(nearest_nonlocal),
        "minimum_y_clearance_nm": float(np.min(y_clearance)),
        "y_collision_detected": bool(np.any(y_flags)),
        "y_collision_points": int(np.count_nonzero(y_flags)),
        "rod_diameter_nm": spec.rod_diameter_nm,
        "path_point_count": int(centerline.shape[0]),
        "tube_vertex_count": int(mesh.vertices.shape[0]),
        "tube_face_count": int(mesh.faces.shape[0]),
        "color_legend": {
            "cyan": "passes curvature and self-contact checks",
            "red": "centerline curvature radius below threshold",
            "orange": "nonlocal centerline distance below rod diameter",
            "magenta": "both curvature and self-contact violation",
            "purple": "rod centerline is too close to the Y scaffold",
        },
    }
    return mesh, validation


def _rotation_from_z(axis: np.ndarray) -> np.ndarray:
    source = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    target = axis / np.linalg.norm(axis)
    cross = np.cross(source, target)
    norm = np.linalg.norm(cross)
    if norm < 1e-12:
        return np.eye(4)
    angle = float(np.arccos(np.clip(np.dot(source, target), -1.0, 1.0)))
    return trimesh.transformations.rotation_matrix(angle, cross / norm)


def _reference_y_mesh(spec: YJunctionWindingSpec) -> trimesh.Trimesh:
    theta = np.deg2rad(spec.upper_branch_angle_degrees)
    branches = [
        (
            np.array([0.0, 0.0, -spec.branch_length_nm], dtype=np.float64),
            np.array([0.0, 0.0, 0.0], dtype=np.float64),
        ),
        (
            np.array([0.0, 0.0, 0.0], dtype=np.float64),
            spec.branch_length_nm * np.array([-np.sin(theta), 0.0, np.cos(theta)], dtype=np.float64),
        ),
        (
            np.array([0.0, 0.0, 0.0], dtype=np.float64),
            spec.branch_length_nm * np.array([np.sin(theta), 0.0, np.cos(theta)], dtype=np.float64),
        ),
    ]
    meshes: list[trimesh.Trimesh] = []
    for start, end in branches:
        axis = end - start
        cylinder = trimesh.creation.cylinder(
            radius=spec.channel_radius_nm,
            height=float(np.linalg.norm(axis)),
            sections=48,
        )
        transform = _rotation_from_z(axis)
        transform[:3, 3] = 0.5 * (start + end)
        cylinder.apply_transform(transform)
        cylinder.visual.vertex_colors = np.tile(np.array([[170, 170, 170, 80]], dtype=np.uint8), (cylinder.vertices.shape[0], 1))
        meshes.append(cylinder)
    return trimesh.util.concatenate(meshes)


def generate_y_junction_winding(
    spec: YJunctionWindingSpec = YJunctionWindingSpec(),
) -> WindingResult:
    _validate_spec(spec)
    centerline, turn_diagnostics = _generate_turn_by_turn_centerline(spec)
    optimizer_diagnostics: dict[str, object] | None = None
    if spec.optimize_control_points:
        centerline, optimizer_diagnostics = _optimize_centerline_against_y(centerline, spec)
    mesh, validation = _tube_mesh_from_centerline(centerline, spec)
    validation["turn_diagnostics"] = turn_diagnostics
    if optimizer_diagnostics is not None:
        validation["optimizer_diagnostics"] = optimizer_diagnostics
    return WindingResult(centerline=centerline, mesh=mesh, validation=validation)


def export_y_junction_winding(
    output_path: str | Path = "outputs/y_junction_winding/y_junction_winding.glb",
    spec: YJunctionWindingSpec = YJunctionWindingSpec(),
) -> Path:
    path = Path(output_path)
    if path.suffix == "":
        path = path / "y_junction_winding.glb"
    path.parent.mkdir(parents=True, exist_ok=True)
    result = generate_y_junction_winding(spec)

    scene = trimesh.Scene()
    scene.add_geometry(result.mesh, geom_name="colored_winding_tube")
    if spec.include_reference_y:
        scene.add_geometry(_reference_y_mesh(spec), geom_name="reference_y")
    scene.export(path)
    result.mesh.export(path.with_suffix(".obj"))
    np.savetxt(
        path.with_name(f"{path.stem}_centerline.csv"),
        result.centerline,
        delimiter=",",
        header="x_nm,y_nm,z_nm",
        comments="",
    )
    path.with_suffix(".json").write_text(
        json.dumps(
            {
                "glb_preview": str(path),
                "obj_preview": str(path.with_suffix(".obj")),
                "centerline_csv": str(path.with_name(f"{path.stem}_centerline.csv")),
                "units": "nm",
                "primary_geometry": "5.2 nm diameter diagnostic winding tube around a 3-cylinder Y",
                "spec": asdict(spec),
                "validation": result.validation,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def _validate_spec(spec: YJunctionWindingSpec) -> None:
    if spec.channel_radius_nm <= 0.0:
        raise ValueError("channel_radius_nm must be > 0")
    if spec.branch_length_nm <= 0.0:
        raise ValueError("branch_length_nm must be > 0")
    if not 0.0 < spec.upper_branch_angle_degrees < 90.0:
        raise ValueError("upper_branch_angle_degrees must be between 0 and 90")
    if spec.rod_diameter_nm <= 0.0:
        raise ValueError("rod_diameter_nm must be > 0")
    if spec.minimum_curvature_radius_nm <= 0.0:
        raise ValueError("minimum_curvature_radius_nm must be > 0")
    if spec.top_z_nm <= spec.lower_z_nm:
        raise ValueError("top_z_nm must be > lower_z_nm")
    if spec.inside_blend_height_nm <= 0.0:
        raise ValueError("inside_blend_height_nm must be > 0")
    if spec.upper_follow_blend_height_nm <= 0.0:
        raise ValueError("upper_follow_blend_height_nm must be > 0")
    if spec.junction_flare_nm < 0.0:
        raise ValueError("junction_flare_nm must be >= 0")
    if spec.y_projection_iterations < 1:
        raise ValueError("y_projection_iterations must be >= 1")
    if spec.y_projection_smooth_sigma < 0.0:
        raise ValueError("y_projection_smooth_sigma must be >= 0")
    if spec.outside_y_scale <= 0.0:
        raise ValueError("outside_y_scale must be > 0")
    if spec.inside_y_scale <= 0.0:
        raise ValueError("inside_y_scale must be > 0")
    if spec.winding_pitch_nm <= spec.rod_diameter_nm:
        raise ValueError("winding_pitch_nm must be > rod_diameter_nm")
    if spec.pitch_search_max_nm <= spec.rod_diameter_nm:
        raise ValueError("pitch_search_max_nm must be > rod_diameter_nm")
    if spec.pitch_candidate_count < 2:
        raise ValueError("pitch_candidate_count must be >= 2")
    if spec.turn_samples < 8:
        raise ValueError("turn_samples must be >= 8")
    if spec.target_contact_gap_nm < 0.0:
        raise ValueError("target_contact_gap_nm must be >= 0")
    if spec.upper_funnel_arc_degrees <= 0.0:
        raise ValueError("upper_funnel_arc_degrees must be > 0")
    if spec.upper_vertical_undulation_nm < 0.0:
        raise ValueError("upper_vertical_undulation_nm must be >= 0")
    if spec.optimizer_control_points < 8:
        raise ValueError("optimizer_control_points must be >= 8")
    if spec.optimizer_max_nfev < 1:
        raise ValueError("optimizer_max_nfev must be >= 1")
    if spec.optimizer_xy_bound_nm <= 0.0:
        raise ValueError("optimizer_xy_bound_nm must be > 0")
    if spec.y_clearance_weight <= 0.0:
        raise ValueError("y_clearance_weight must be > 0")
    if spec.self_contact_weight <= 0.0:
        raise ValueError("self_contact_weight must be > 0")
    if spec.packing_weight < 0.0:
        raise ValueError("packing_weight must be >= 0")
    if spec.smoothness_weight < 0.0:
        raise ValueError("smoothness_weight must be >= 0")
    if spec.shape_preservation_weight < 0.0:
        raise ValueError("shape_preservation_weight must be >= 0")
    if spec.path_samples < 16:
        raise ValueError("path_samples must be >= 16")
    if spec.tube_radial_samples < 8:
        raise ValueError("tube_radial_samples must be >= 8")
    if spec.centerline_smooth_sigma < 0.0:
        raise ValueError("centerline_smooth_sigma must be >= 0")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a colored rod winding around a Y junction.")
    parser.add_argument("--out", default="outputs/y_junction_winding/y_junction_winding.glb")
    parser.add_argument("--channel-radius-nm", type=float, default=24.0)
    parser.add_argument("--branch-length-nm", type=float, default=90.0)
    parser.add_argument("--upper-branch-angle-degrees", type=float, default=60.0)
    parser.add_argument("--rod-diameter-nm", type=float, default=5.2)
    parser.add_argument("--rod-clearance-nm", type=float, default=0.2)
    parser.add_argument("--minimum-curvature-radius-nm", type=float, default=6.0)
    parser.add_argument("--lower-z-nm", type=float, default=-90.0)
    parser.add_argument("--top-z-nm", type=float, default=52.0)
    parser.add_argument("--inside-start-z-nm", type=float, default=34.0)
    parser.add_argument("--inside-blend-height-nm", type=float, default=14.0)
    parser.add_argument("--upper-follow-start-z-nm", type=float, default=0.0)
    parser.add_argument("--upper-follow-blend-height-nm", type=float, default=45.0)
    parser.add_argument("--junction-flare-nm", type=float, default=0.0)
    parser.add_argument("--y-projection-iterations", type=int, default=14)
    parser.add_argument("--y-projection-smooth-sigma", type=float, default=0.0)
    parser.add_argument("--outside-y-scale", type=float, default=1.25)
    parser.add_argument("--inside-y-scale", type=float, default=1.9)
    parser.add_argument("--winding-pitch-nm", type=float, default=5.35)
    parser.add_argument("--pitch-search-max-nm", type=float, default=14.0)
    parser.add_argument("--pitch-candidate-count", type=int, default=36)
    parser.add_argument("--turn-samples", type=int, default=80)
    parser.add_argument("--target-contact-gap-nm", type=float, default=0.06)
    parser.add_argument("--upper-funnel-arc-degrees", type=float, default=70.0)
    parser.add_argument("--upper-vertical-undulation-nm", type=float, default=0.4)
    parser.add_argument("--optimize-control-points", action="store_true")
    parser.add_argument("--optimizer-control-points", type=int, default=96)
    parser.add_argument("--optimizer-max-nfev", type=int, default=160)
    parser.add_argument("--optimizer-xy-bound-nm", type=float, default=3.0)
    parser.add_argument("--y-clearance-weight", type=float, default=16.0)
    parser.add_argument("--self-contact-weight", type=float, default=10.0)
    parser.add_argument("--packing-weight", type=float, default=1.6)
    parser.add_argument("--smoothness-weight", type=float, default=0.18)
    parser.add_argument("--shape-preservation-weight", type=float, default=0.12)
    parser.add_argument("--phase-offset-degrees", type=float, default=0.0)
    parser.add_argument("--path-samples", type=int, default=2600)
    parser.add_argument("--tube-radial-samples", type=int, default=28)
    parser.add_argument("--centerline-smooth-sigma", type=float, default=2.0)
    parser.add_argument("--hide-reference-y", action="store_true")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    path = export_y_junction_winding(
        args.out,
        YJunctionWindingSpec(
            channel_radius_nm=args.channel_radius_nm,
            branch_length_nm=args.branch_length_nm,
            upper_branch_angle_degrees=args.upper_branch_angle_degrees,
            rod_diameter_nm=args.rod_diameter_nm,
            rod_clearance_nm=args.rod_clearance_nm,
            minimum_curvature_radius_nm=args.minimum_curvature_radius_nm,
            lower_z_nm=args.lower_z_nm,
            top_z_nm=args.top_z_nm,
            inside_start_z_nm=args.inside_start_z_nm,
            inside_blend_height_nm=args.inside_blend_height_nm,
            upper_follow_start_z_nm=args.upper_follow_start_z_nm,
            upper_follow_blend_height_nm=args.upper_follow_blend_height_nm,
            junction_flare_nm=args.junction_flare_nm,
            y_projection_iterations=args.y_projection_iterations,
            y_projection_smooth_sigma=args.y_projection_smooth_sigma,
            outside_y_scale=args.outside_y_scale,
            inside_y_scale=args.inside_y_scale,
            winding_pitch_nm=args.winding_pitch_nm,
            pitch_search_max_nm=args.pitch_search_max_nm,
            pitch_candidate_count=args.pitch_candidate_count,
            turn_samples=args.turn_samples,
            target_contact_gap_nm=args.target_contact_gap_nm,
            upper_funnel_arc_degrees=args.upper_funnel_arc_degrees,
            upper_vertical_undulation_nm=args.upper_vertical_undulation_nm,
            optimize_control_points=args.optimize_control_points,
            optimizer_control_points=args.optimizer_control_points,
            optimizer_max_nfev=args.optimizer_max_nfev,
            optimizer_xy_bound_nm=args.optimizer_xy_bound_nm,
            y_clearance_weight=args.y_clearance_weight,
            self_contact_weight=args.self_contact_weight,
            packing_weight=args.packing_weight,
            smoothness_weight=args.smoothness_weight,
            shape_preservation_weight=args.shape_preservation_weight,
            phase_offset_degrees=args.phase_offset_degrees,
            path_samples=args.path_samples,
            tube_radial_samples=args.tube_radial_samples,
            centerline_smooth_sigma=args.centerline_smooth_sigma,
            include_reference_y=not args.hide_reference_y,
        ),
    )
    print(f"Saved GLB preview to: {path}")
    print(f"Saved OBJ preview to: {path.with_suffix('.obj')}")
    print(f"Saved centerline CSV to: {path.with_name(f'{path.stem}_centerline.csv')}")
    print(f"Saved metadata to: {path.with_suffix('.json')}")


if __name__ == "__main__":
    main()
