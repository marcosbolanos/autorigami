"""Optimize one continuous cylinder surface to cover an exact Y constraint."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.spatial import cKDTree
import trimesh


Vec3 = np.ndarray


@dataclass(frozen=True)
class YJunctionWrapperSpec:
    channel_radius_nm: float = 8.0
    branch_length_nm: float = 30.0
    upper_branch_angle_degrees: float = 60.0
    min_curvature_radius_nm: float = 6.0
    curvature_mode: str = "all"
    z_samples: int = 40
    radial_samples: int = 48
    control_z_samples: int = 6
    control_theta_samples: int = 8
    target_axis_samples: int = 28
    target_radial_samples: int = 24
    radial_search_samples: int = 160
    initial_margin_nm: float = 9.0
    stem_clearance_nm: float = 1.5
    stem_blend_start_z_nm: float = -10.0
    stem_blend_end_z_nm: float = 4.0
    base_gaussian_sigma_z: float = 3.0
    base_gaussian_sigma_theta: float = 2.0
    init_self_overlap: bool = False
    overlap_start_fraction: float = 0.67
    overlap_end_fraction: float = 0.95
    overlap_swap_nm: float = 1.0
    overlap_swap_fraction: float = 0.0
    overlap_angular_width_degrees: float = 35.0
    overlap_exact_columns: bool = False
    overlap_axis: str = "x"
    overlap_outward_nm: float = 0.0
    overlap_handle_outward_nm: float = 0.0
    overlap_handle_offset_degrees: float = 26.0
    overlap_handle_width_degrees: float = 10.0
    overlap_sparse_controls: bool = False
    overlap_bezier_lobes: bool = False
    overlap_crossing_width_degrees: float = 24.0
    overlap_edge_support_nm: float = 0.0
    overlap_edge_support_width_degrees: float = 10.0
    overlap_handle_tangent_nm: float = 0.0
    overlap_control_width_degrees: float = 8.0
    overlap_radial_floor_fraction: float = 0.0
    overlap_smooth_displacement: bool = False
    overlap_smooth_sigma_z: float = 1.2
    overlap_smooth_sigma_theta: float = 1.8
    overlap_lift_nm: float = 0.0
    coverage_scale_nm: float = 4.0
    coverage_schedule: tuple[float, ...] = (0.15, 0.15, 0.15)
    max_iterations_per_phase: int = 10
    skip_optimization: bool = False
    penetration_weight: float = 10000.0
    curvature_weight: float = 500.0
    curvature_barrier_margin_nm: float = 1.0
    curvature_barrier_sharpness: float = 12.0
    smoothness_weight: float = 0.08
    displacement_weight: float = 0.004
    radial_nonshrink_weight: float = 0.0
    upper_extension_nm: float = 0.0
    upper_extension_samples: int = 0
    upper_extension_crossing_nm: float = 0.0
    upper_extension_crossing_width_degrees: float = 24.0


@dataclass(frozen=True)
class Branch:
    start: Vec3
    end: Vec3

    @property
    def axis(self) -> Vec3:
        vector = self.end - self.start
        length = float(np.linalg.norm(vector))
        if length <= 0.0:
            raise ValueError("branch length must be > 0")
        return vector / length


@dataclass(frozen=True)
class OptimizedWrapper:
    surface: np.ndarray
    mesh: trimesh.Trimesh
    validation: dict[str, object]
    objective: dict[str, float | int | bool]
    trace: list[dict[str, float | int | str | bool]]


def _branches(spec: YJunctionWrapperSpec) -> tuple[Branch, Branch, Branch]:
    theta = np.deg2rad(spec.upper_branch_angle_degrees)
    lower = Branch(
        start=np.array([0.0, 0.0, -spec.branch_length_nm], dtype=np.float64),
        end=np.array([0.0, 0.0, 0.0], dtype=np.float64),
    )
    left = Branch(
        start=np.array([0.0, 0.0, 0.0], dtype=np.float64),
        end=spec.branch_length_nm
        * np.array([-np.sin(theta), 0.0, np.cos(theta)], dtype=np.float64),
    )
    right = Branch(
        start=np.array([0.0, 0.0, 0.0], dtype=np.float64),
        end=spec.branch_length_nm
        * np.array([np.sin(theta), 0.0, np.cos(theta)], dtype=np.float64),
    )
    return lower, left, right


def _distance_to_branch(points: np.ndarray, branch: Branch) -> np.ndarray:
    axis = branch.end - branch.start
    length_squared = float(np.dot(axis, axis))
    relative = points - branch.start
    t = np.clip(np.sum(relative * axis, axis=-1) / length_squared, 0.0, 1.0)
    closest = branch.start + t[..., None] * axis
    return np.linalg.norm(points - closest, axis=-1)


def _y_sdf(points: np.ndarray, spec: YJunctionWrapperSpec) -> np.ndarray:
    return np.minimum.reduce(
        [
            _distance_to_branch(points, branch) - spec.channel_radius_nm
            for branch in _branches(spec)
        ]
    )


def _orthonormal_frame(axis: Vec3) -> tuple[Vec3, Vec3]:
    reference = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    if abs(float(np.dot(axis, reference))) > 0.95:
        reference = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    normal = np.cross(reference, axis)
    normal /= np.linalg.norm(normal)
    binormal = np.cross(axis, normal)
    return normal, binormal


def _target_y_surface_points(spec: YJunctionWrapperSpec) -> np.ndarray:
    targets: list[np.ndarray] = []
    t_values = np.linspace(0.0, 1.0, spec.target_axis_samples)
    angles = np.linspace(0.0, 2.0 * np.pi, spec.target_radial_samples, endpoint=False)
    for branch in _branches(spec):
        normal, binormal = _orthonormal_frame(branch.axis)
        centers = branch.start + t_values[:, None] * (branch.end - branch.start)
        rings = []
        for angle in angles:
            radial = np.cos(angle) * normal + np.sin(angle) * binormal
            rings.append(centers + spec.channel_radius_nm * radial)
        targets.append(np.concatenate(rings, axis=0))
    return np.concatenate(targets, axis=0)


def _surface_from_radii(z_values: np.ndarray, angles: np.ndarray, radii: np.ndarray) -> np.ndarray:
    surface = np.empty((z_values.size, angles.size, 3), dtype=np.float64)
    for z_index, z in enumerate(z_values):
        surface[z_index, :, 0] = radii[z_index] * np.cos(angles)
        surface[z_index, :, 1] = radii[z_index] * np.sin(angles)
        surface[z_index, :, 2] = z
    return surface


def _smoothstep(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, 0.0, 1.0)
    return clipped * clipped * (3.0 - 2.0 * clipped)


def _smooth_overlap_displacement(
    base_surface: np.ndarray,
    deformed_surface: np.ndarray,
    window: np.ndarray,
    spec: YJunctionWrapperSpec,
) -> np.ndarray:
    if not spec.overlap_smooth_displacement:
        return deformed_surface

    from scipy.ndimage import gaussian_filter

    displacement = deformed_surface - base_surface
    smoothed = gaussian_filter(
        displacement,
        sigma=(spec.overlap_smooth_sigma_z, spec.overlap_smooth_sigma_theta, 0.0),
        mode=("nearest", "wrap", "nearest"),
    )
    blend = _smoothstep(window)[:, None, None]
    return base_surface + blend * smoothed + (1.0 - blend) * displacement


def _initial_surface(spec: YJunctionWrapperSpec) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from scipy.ndimage import gaussian_filter

    lower, left, _ = _branches(spec)
    z_values = np.linspace(lower.start[2], left.end[2] + spec.upper_extension_nm, spec.z_samples)
    angles = np.linspace(0.0, 2.0 * np.pi, spec.radial_samples, endpoint=False)
    max_extent = spec.branch_length_nm * np.sin(np.deg2rad(spec.upper_branch_angle_degrees))
    scan = np.linspace(0.0, max_extent + 3.0 * spec.channel_radius_nm, spec.radial_search_samples)
    radii = np.empty((spec.z_samples, spec.radial_samples), dtype=np.float64)
    for z_index, z in enumerate(z_values):
        for theta_index, theta in enumerate(angles):
            ray = np.column_stack(
                (
                    scan * np.cos(theta),
                    scan * np.sin(theta),
                    np.full_like(scan, z),
                )
            )
            inside = _y_sdf(ray, spec) <= 0.0
            if np.any(inside):
                radii[z_index, theta_index] = scan[np.nonzero(inside)[0].max()] + spec.initial_margin_nm
            else:
                radii[z_index, theta_index] = spec.channel_radius_nm + spec.initial_margin_nm

    stem_radius = spec.channel_radius_nm + spec.stem_clearance_nm
    blend = _smoothstep(
        (z_values - spec.stem_blend_start_z_nm)
        / (spec.stem_blend_end_z_nm - spec.stem_blend_start_z_nm)
    )
    radii = (1.0 - blend[:, None]) * stem_radius + blend[:, None] * radii
    radii = gaussian_filter(
        radii,
        sigma=(spec.base_gaussian_sigma_z, spec.base_gaussian_sigma_theta),
        mode=("nearest", "wrap"),
    )
    return z_values, angles, _surface_from_radii(z_values, angles, radii)


def _apply_initial_self_overlap(
    surface: np.ndarray,
    z_values: np.ndarray,
    angles: np.ndarray,
    spec: YJunctionWrapperSpec,
) -> np.ndarray:
    if not spec.init_self_overlap:
        return surface

    z_min = float(z_values[0])
    z_max = float(z_values[-1])
    u = (z_values - z_min) / (z_max - z_min)
    rise = _smoothstep(
        (u - spec.overlap_start_fraction)
        / (spec.overlap_end_fraction - spec.overlap_start_fraction)
    )
    fall = 1.0 - _smoothstep((u - spec.overlap_end_fraction) / 0.08)
    window = rise * fall

    deformed = surface.copy()
    if spec.overlap_axis == "x":
        angular_profile = np.cos(angles)
        coordinate_index = 0
        positive_angle = 0.0
        negative_angle = np.pi
    elif spec.overlap_axis == "y":
        angular_profile = np.sin(angles)
        coordinate_index = 1
        positive_angle = 0.5 * np.pi
        negative_angle = 1.5 * np.pi
    else:
        raise ValueError("overlap_axis must be 'x' or 'y'")

    if spec.overlap_bezier_lobes:
        if spec.overlap_axis != "y":
            raise ValueError("overlap_bezier_lobes currently requires overlap_axis='y'")

        t = window
        swap_progress = t * t * t * (10.0 + t * (-15.0 + 6.0 * t))
        bell = np.sin(np.pi * t) ** 2
        base_offset = np.deg2rad(45.0)
        anchor_offsets = np.deg2rad(np.array([-45.0, -38.0, -28.0, -18.0, 0.0, 18.0, 28.0, 38.0, 45.0]))
        deformed_xy = deformed[:, :, :2].copy()

        for center, side_sign in ((positive_angle, 1.0), (negative_angle, -1.0)):
            local_angles = (angles - center + np.pi) % (2.0 * np.pi) - np.pi
            side_mask = np.abs(local_angles) <= base_offset
            if not np.any(side_mask):
                continue

            center_index = int(np.argmin(np.abs(local_angles)))
            center_initial_y = surface[:, center_index, 1]
            center_target_y = (
                center_initial_y * (1.0 - 2.0 * spec.overlap_swap_fraction * swap_progress)
                + side_sign * spec.overlap_outward_nm * bell
            )
            anchor_displacements = np.zeros((z_values.size, anchor_offsets.size, 2), dtype=np.float64)
            for anchor_index, offset in enumerate(anchor_offsets):
                anchor_angle = center + offset
                radial = np.array([np.cos(anchor_angle), np.sin(anchor_angle)], dtype=np.float64)
                abs_offset_degrees = abs(float(np.rad2deg(offset)))
                if abs_offset_degrees == 28.0:
                    radial_scale = 1.0
                elif abs_offset_degrees == 38.0:
                    radial_scale = 0.45
                elif abs_offset_degrees == 18.0:
                    radial_scale = 0.65
                elif abs_offset_degrees == 0.0:
                    radial_scale = 0.35
                else:
                    radial_scale = 0.0
                anchor_displacements[:, anchor_index, :] = (
                    spec.overlap_handle_outward_nm * radial_scale * bell[:, None] * radial[None, :]
                )

            selected_indices = np.nonzero(side_mask)[0]
            for theta_index in selected_indices:
                local_angle = local_angles[theta_index]
                segment_index = int(np.searchsorted(anchor_offsets, local_angle, side="right") - 1)
                segment_index = max(0, min(segment_index, anchor_offsets.size - 2))
                left = anchor_offsets[segment_index]
                right = anchor_offsets[segment_index + 1]
                alpha = (local_angle - left) / (right - left)
                blend = float(_smoothstep(np.array([alpha], dtype=np.float64))[0])
                displacement = (
                    (1.0 - blend) * anchor_displacements[:, segment_index, :]
                    + blend * anchor_displacements[:, segment_index + 1, :]
                )
                deformed_xy[:, theta_index, :] += displacement

            center_width = np.deg2rad(spec.overlap_crossing_width_degrees)
            center_distance = np.minimum(
                np.abs(local_angles),
                2.0 * np.pi - np.abs(local_angles),
            )
            center_basis = np.exp(-0.5 * (center_distance / center_width) ** 2)
            crossing_delta = np.column_stack(
                (
                    np.zeros_like(center_target_y),
                    center_target_y - center_initial_y,
                )
            )
            deformed_xy += center_basis[None, :, None] * crossing_delta[:, None, :]

            if spec.overlap_edge_support_nm > 0.0:
                edge_width = np.deg2rad(spec.overlap_edge_support_width_degrees)
                edge_support = np.zeros_like(angles)
                for edge_offset in (-spec.overlap_crossing_width_degrees, spec.overlap_crossing_width_degrees):
                    edge_center = np.deg2rad(edge_offset)
                    edge_distance = np.minimum(
                        np.abs(local_angles - edge_center),
                        2.0 * np.pi - np.abs(local_angles - edge_center),
                    )
                    edge_support += np.exp(-0.5 * (edge_distance / edge_width) ** 2)
                edge_support = np.minimum(edge_support, 1.0)
                deformed_xy += (
                    spec.overlap_edge_support_nm
                    * bell[:, None, None]
                    * edge_support[None, :, None]
                    * np.column_stack((np.cos(angles), np.sin(angles)))[None, :, :]
                )

        deformed[:, :, :2] = deformed_xy
        deformed[:, :, 2] += spec.overlap_lift_nm * window[:, None] * (
            1.0 - angular_profile[None, :] * angular_profile[None, :]
        )
        return _smooth_overlap_displacement(surface, deformed, window, spec)

    if spec.overlap_sparse_controls:
        t = window[:, None]
        swap_progress = t * t * t * (10.0 + t * (-15.0 + 6.0 * t))
        bell = np.sin(np.pi * t) ** 2
        width = np.deg2rad(spec.overlap_control_width_degrees)
        offset = np.deg2rad(spec.overlap_handle_offset_degrees)
        displacement = np.zeros_like(deformed[:, :, :2])

        for center, side_sign in ((positive_angle, 1.0), (negative_angle, -1.0)):
            center_distance = np.minimum(
                np.abs(angles - center),
                2.0 * np.pi - np.abs(angles - center),
            )
            center_basis = np.exp(-0.5 * (center_distance / width) ** 2)
            center_coordinate = surface[:, :, coordinate_index]
            center_target = (
                center_coordinate * (1.0 - 2.0 * spec.overlap_swap_fraction * swap_progress)
                + side_sign * spec.overlap_outward_nm * bell
            )
            displacement[:, :, coordinate_index] += center_basis[None, :] * (
                center_target - center_coordinate
            )

            for offset_sign in (-1.0, 1.0):
                handle_center = (center + offset_sign * offset) % (2.0 * np.pi)
                handle_distance = np.minimum(
                    np.abs(angles - handle_center),
                    2.0 * np.pi - np.abs(angles - handle_center),
                )
                handle_basis = np.exp(-0.5 * (handle_distance / width) ** 2)
                tangent = np.array(
                    [-np.sin(handle_center), np.cos(handle_center)],
                    dtype=np.float64,
                )
                radial = np.array(
                    [np.cos(handle_center), np.sin(handle_center)],
                    dtype=np.float64,
                )
                handle_vector = (
                    offset_sign * spec.overlap_handle_tangent_nm * tangent
                    + spec.overlap_handle_outward_nm * radial
                )
                displacement += (
                    bell[:, :, None]
                    * handle_basis[None, :, None]
                    * handle_vector[None, None, :]
                )

        deformed[:, :, :2] += displacement
        deformed[:, :, 2] += spec.overlap_lift_nm * window[:, None] * (
            1.0 - angular_profile[None, :] * angular_profile[None, :]
        )
        return _smooth_overlap_displacement(surface, deformed, window, spec)

    if spec.overlap_exact_columns:
        angular_mask = np.zeros_like(angles)
        positive_distance = np.minimum(
            np.abs(angles - positive_angle),
            2.0 * np.pi - np.abs(angles - positive_angle),
        )
        negative_distance = np.minimum(
            np.abs(angles - negative_angle),
            2.0 * np.pi - np.abs(angles - negative_angle),
        )
        angular_mask[int(np.argmin(positive_distance))] = 1.0
        angular_mask[int(np.argmin(negative_distance))] = 1.0
    else:
        positive_distance = np.minimum(
            np.abs(angles - positive_angle),
            2.0 * np.pi - np.abs(angles - positive_angle),
        )
        negative_distance = np.minimum(
            np.abs(angles - negative_angle),
            2.0 * np.pi - np.abs(angles - negative_angle),
        )
        angular_distance = np.minimum(positive_distance, negative_distance)
        angular_width = np.deg2rad(spec.overlap_angular_width_degrees)
        angular_mask = np.exp(-0.5 * (angular_distance / angular_width) ** 2)
    deformed[:, :, coordinate_index] -= (
        spec.overlap_swap_nm
        * window[:, None]
        * angular_mask[None, :]
        * angular_profile[None, :]
    )
    if spec.overlap_outward_nm > 0.0:
        t = window[:, None]
        coordinate = surface[:, :, coordinate_index]
        side_sign = np.sign(angular_profile)[None, :]
        swap_progress = t * t * t * (10.0 + t * (-15.0 + 6.0 * t))
        outward_profile = np.sin(np.pi * t) ** 2
        target = (
            coordinate * (1.0 - 2.0 * spec.overlap_swap_fraction * swap_progress)
            + side_sign * spec.overlap_outward_nm * outward_profile
        )
        deformed[:, :, coordinate_index] += angular_mask[None, :] * (target - coordinate)
    else:
        deformed[:, :, coordinate_index] += (
            -2.0
            * spec.overlap_swap_fraction
            * window[:, None]
            * angular_mask[None, :]
            * surface[:, :, coordinate_index]
        )
    if spec.overlap_handle_outward_nm > 0.0:
        handle_offset = np.deg2rad(spec.overlap_handle_offset_degrees)
        handle_width = np.deg2rad(spec.overlap_handle_width_degrees)
        handle_mask = np.zeros_like(angles)
        for center in (positive_angle, negative_angle):
            for handle_center in (center - handle_offset, center + handle_offset):
                wrapped_center = handle_center % (2.0 * np.pi)
                distance = np.minimum(
                    np.abs(angles - wrapped_center),
                    2.0 * np.pi - np.abs(angles - wrapped_center),
                )
                handle_mask += np.exp(-0.5 * (distance / handle_width) ** 2)
        handle_mask = np.minimum(handle_mask, 1.0)
        deformed[:, :, coordinate_index] += (
            spec.overlap_handle_outward_nm
            * window[:, None]
            * handle_mask[None, :]
            * angular_profile[None, :]
        )
    if spec.overlap_radial_floor_fraction > 0.0:
        xy = deformed[:, :, :2]
        radius = np.linalg.norm(xy, axis=2)
        row_lobe_radius = np.percentile(radius, 90.0, axis=1)
        radius_floor = spec.overlap_radial_floor_fraction * row_lobe_radius[:, None]
        active_floor = window[:, None] * radius_floor + (1.0 - window[:, None]) * radius
        below_floor = radius < active_floor
        original_direction = surface[:, :, :2]
        original_direction_norm = np.linalg.norm(original_direction, axis=2, keepdims=True)
        radial_direction = np.divide(
            xy,
            radius[:, :, None],
            out=np.zeros_like(xy),
            where=radius[:, :, None] > 1e-12,
        )
        fallback_direction = np.divide(
            original_direction,
            original_direction_norm,
            out=np.zeros_like(original_direction),
            where=original_direction_norm > 1e-12,
        )
        radial_direction = np.where(
            radius[:, :, None] > 1e-12,
            radial_direction,
            fallback_direction,
        )
        deformed[:, :, :2] = np.where(
            below_floor[:, :, None],
            radial_direction * active_floor[:, :, None],
            xy,
        )
    deformed[:, :, 2] += spec.overlap_lift_nm * window[:, None] * (
        1.0 - angular_profile[None, :] * angular_profile[None, :]
    )
    return _smooth_overlap_displacement(surface, deformed, window, spec)


def _interpolate_controls(
    controls: np.ndarray,
    z_samples: int,
    radial_samples: int,
) -> np.ndarray:
    control_z, control_theta, _ = controls.shape
    z_pos = np.linspace(0.0, control_z - 1, z_samples)
    theta_pos = np.linspace(0.0, control_theta, radial_samples, endpoint=False)
    output = np.empty((z_samples, radial_samples, 3), dtype=np.float64)
    for zi, z in enumerate(z_pos):
        z0 = int(np.floor(z))
        z1 = min(z0 + 1, control_z - 1)
        wz = z - z0
        for ti, theta in enumerate(theta_pos):
            t0 = int(np.floor(theta)) % control_theta
            t1 = (t0 + 1) % control_theta
            wt = theta - np.floor(theta)
            a = (1.0 - wt) * controls[z0, t0] + wt * controls[z0, t1]
            b = (1.0 - wt) * controls[z1, t0] + wt * controls[z1, t1]
            output[zi, ti] = (1.0 - wz) * a + wz * b
    return output


def _curve_curvatures(points: np.ndarray, periodic: bool) -> np.ndarray:
    if periodic:
        previous = np.roll(points, 1, axis=0)
        current = points
        following = np.roll(points, -1, axis=0)
    else:
        previous = points[:-2]
        current = points[1:-1]
        following = points[2:]
    a = current - previous
    b = following - current
    c = following - previous
    cross = np.linalg.norm(np.cross(a, b), axis=1)
    denom = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) * np.linalg.norm(c, axis=1)
    return np.divide(2.0 * cross, denom, out=np.zeros_like(cross), where=denom > 1e-12)


def _surface_curvatures(surface: np.ndarray, mode: str = "all") -> np.ndarray:
    curves: list[np.ndarray] = []
    if mode in ("all", "vertical"):
        curves.extend(_curve_curvatures(surface[:, side, :], periodic=False) for side in range(surface.shape[1]))
    if mode in ("all", "horizontal"):
        curves.extend(_curve_curvatures(surface[z_index], periodic=True) for z_index in range(surface.shape[0]))
    if not curves:
        raise ValueError("curvature mode must be 'all', 'vertical', or 'horizontal'")
    return np.concatenate(curves)


def _smoothness_loss(controls: np.ndarray) -> float:
    dz = controls[2:, :, :] - 2.0 * controls[1:-1, :, :] + controls[:-2, :, :]
    dt = np.roll(controls, -1, axis=1) - 2.0 * controls + np.roll(controls, 1, axis=1)
    return float(np.mean(dz * dz) + np.mean(dt * dt))


def _terms(
    variables: np.ndarray,
    base_surface: np.ndarray,
    targets: np.ndarray,
    spec: YJunctionWrapperSpec,
    coverage_weight: float,
) -> tuple[dict[str, float], np.ndarray]:
    controls = variables.reshape((spec.control_z_samples, spec.control_theta_samples, 3))
    surface = base_surface + _interpolate_controls(controls, spec.z_samples, spec.radial_samples)
    flat_surface = surface.reshape((-1, 3))
    sdf = _y_sdf(flat_surface, spec)
    penetration = np.minimum(sdf, 0.0)

    tree = cKDTree(flat_surface)
    target_distances, _ = tree.query(targets, k=1)
    coverage_loss = float(np.mean(np.minimum(target_distances, spec.coverage_scale_nm) ** 2))

    curvatures = _surface_curvatures(surface, spec.curvature_mode)
    curvature_limit = 1.0 / spec.min_curvature_radius_nm
    barrier_limit = 1.0 / (spec.min_curvature_radius_nm + spec.curvature_barrier_margin_nm)
    scaled_excess = spec.curvature_barrier_sharpness * (curvatures - barrier_limit)
    softplus = np.logaddexp(0.0, scaled_excess) / spec.curvature_barrier_sharpness
    curvature_loss = float(np.mean(softplus * softplus))

    terms = {
        "coverage_loss": coverage_loss,
        "weighted_coverage_loss": float(coverage_weight * coverage_loss),
        "penetration_loss": float(np.mean(penetration * penetration)),
        "curvature_loss": curvature_loss,
        "smoothness_loss": _smoothness_loss(controls),
        "displacement_loss": float(np.mean(controls * controls)),
        "radial_nonshrink_loss": float(
            np.mean(
                np.maximum(
                    np.linalg.norm(base_surface[:, :, :2], axis=2)
                    - np.linalg.norm(surface[:, :, :2], axis=2),
                    0.0,
                )
                ** 2
            )
        ),
    }
    return terms, surface


def _trace_record(
    phase: str,
    kind: str,
    index: int,
    variables: np.ndarray,
    base_surface: np.ndarray,
    targets: np.ndarray,
    spec: YJunctionWrapperSpec,
    coverage_weight: float,
) -> dict[str, float | int | str | bool]:
    terms, surface = _terms(variables, base_surface, targets, spec, coverage_weight)
    validation = _surface_validation(surface, targets, spec)
    return {
        "phase": phase,
        "kind": kind,
        "index": index,
        "coverage_weight": coverage_weight,
        "objective": float(
            terms["weighted_coverage_loss"]
            + spec.penetration_weight * terms["penetration_loss"]
            + spec.curvature_weight * terms["curvature_loss"]
            + spec.smoothness_weight * terms["smoothness_loss"]
            + spec.displacement_weight * terms["displacement_loss"]
            + spec.radial_nonshrink_weight * terms["radial_nonshrink_loss"]
        ),
        "coverage_loss": terms["coverage_loss"],
        "penetration_loss": terms["penetration_loss"],
        "curvature_loss": terms["curvature_loss"],
        "smoothness_loss": terms["smoothness_loss"],
        "displacement_loss": terms["displacement_loss"],
        "radial_nonshrink_loss": terms["radial_nonshrink_loss"],
        "minimum_distance_to_forbidden_y_nm": validation["minimum_distance_to_forbidden_y_nm"],
        "intersects_forbidden_y": validation["intersects_forbidden_y"],
        "minimum_curvature_radius_nm": validation["minimum_curvature_radius_nm"],
        "curvature_constraint_passes": validation["curvature_constraint_passes"],
        "mean_y_target_to_wrapper_distance_nm": validation["mean_y_target_to_wrapper_distance_nm"],
        "target_points_within_1nm": validation["target_points_within_1nm"],
        "self_intersection_detected": validation["self_intersection_detected"],
    }


def _objective(
    variables: np.ndarray,
    base_surface: np.ndarray,
    targets: np.ndarray,
    spec: YJunctionWrapperSpec,
    coverage_weight: float,
) -> float:
    terms, _ = _terms(variables, base_surface, targets, spec, coverage_weight)
    return (
        terms["weighted_coverage_loss"]
        + spec.penetration_weight * terms["penetration_loss"]
        + spec.curvature_weight * terms["curvature_loss"]
        + spec.smoothness_weight * terms["smoothness_loss"]
        + spec.displacement_weight * terms["displacement_loss"]
        + spec.radial_nonshrink_weight * terms["radial_nonshrink_loss"]
    )


def _mesh_from_surface(surface: np.ndarray) -> trimesh.Trimesh:
    z_count, radial_count, _ = surface.shape
    vertices = surface.reshape((-1, 3))
    faces: list[tuple[int, int, int]] = []
    for z_index in range(z_count - 1):
        base = z_index * radial_count
        next_base = (z_index + 1) * radial_count
        for side in range(radial_count):
            a = base + side
            b = base + (side + 1) % radial_count
            c = next_base + side
            d = next_base + (side + 1) % radial_count
            faces.append((a, c, b))
            faces.append((b, c, d))
    return trimesh.Trimesh(vertices=vertices, faces=np.asarray(faces), process=False)


def _self_intersection_score(surface: np.ndarray) -> tuple[bool, float]:
    flat = surface.reshape((-1, 3))
    tree = cKDTree(flat)
    distances, indices = tree.query(flat, k=20)
    z_count, radial_count, _ = surface.shape
    nearest = np.inf
    detected = False
    for index, neighbors in enumerate(indices):
        zi = index // radial_count
        ti = index % radial_count
        for distance, neighbor in zip(distances[index, 1:], neighbors[1:], strict=True):
            nz = int(neighbor) // radial_count
            nt = int(neighbor) % radial_count
            dz = abs(zi - nz)
            dt = min(abs(ti - nt), radial_count - abs(ti - nt))
            if dz <= 2 and dt <= 2:
                continue
            nearest = min(nearest, float(distance))
            detected = detected or bool(distance < 2.0)
    return detected, nearest


def _surface_validation(surface: np.ndarray, targets: np.ndarray, spec: YJunctionWrapperSpec) -> dict[str, object]:
    flat_surface = surface.reshape((-1, 3))
    sdf = _y_sdf(flat_surface, spec)
    curvatures = _surface_curvatures(surface, spec.curvature_mode)
    horizontal_curvatures = _surface_curvatures(surface, "horizontal")
    vertical_curvatures = _surface_curvatures(surface, "vertical")
    min_radius = float(1.0 / np.max(curvatures)) if np.max(curvatures) > 0.0 else float("inf")
    min_horizontal_radius = (
        float(1.0 / np.max(horizontal_curvatures)) if np.max(horizontal_curvatures) > 0.0 else float("inf")
    )
    min_vertical_radius = (
        float(1.0 / np.max(vertical_curvatures)) if np.max(vertical_curvatures) > 0.0 else float("inf")
    )
    target_distances, _ = cKDTree(flat_surface).query(targets, k=1)
    self_intersects, nearest_nonlocal = _self_intersection_score(surface)
    return {
        "minimum_distance_to_forbidden_y_nm": float(np.min(sdf)),
        "intersects_forbidden_y": bool(np.min(sdf) < -1e-6),
        "minimum_curvature_radius_nm": min_radius,
        "minimum_horizontal_curvature_radius_nm": min_horizontal_radius,
        "minimum_vertical_curvature_radius_nm": min_vertical_radius,
        "curvature_mode": spec.curvature_mode,
        "curvature_constraint_passes": bool(min_radius >= spec.min_curvature_radius_nm),
        "self_intersection_detected": self_intersects,
        "nearest_nonlocal_surface_distance_nm": nearest_nonlocal,
        "mean_y_target_to_wrapper_distance_nm": float(np.mean(target_distances)),
        "max_y_target_to_wrapper_distance_nm": float(np.max(target_distances)),
        "target_points_within_1nm": int(np.count_nonzero(target_distances <= 1.0)),
        "target_point_count": int(targets.shape[0]),
        "surface_point_count": int(flat_surface.shape[0]),
    }


def _extend_upper_crossing(surface: np.ndarray, spec: YJunctionWrapperSpec) -> np.ndarray:
    if spec.upper_extension_samples == 0:
        return surface

    radial_count = surface.shape[1]
    angles = np.linspace(0.0, 2.0 * np.pi, radial_count, endpoint=False)
    step_count = min(4, surface.shape[0] - 1)
    tangent_per_row = (surface[-1] - surface[-1 - step_count]) / float(step_count)
    extension = np.empty((spec.upper_extension_samples, radial_count, 3), dtype=np.float64)
    width = np.deg2rad(spec.upper_extension_crossing_width_degrees)
    positive_distance = np.minimum(np.abs(angles - 0.5 * np.pi), 2.0 * np.pi - np.abs(angles - 0.5 * np.pi))
    negative_distance = np.minimum(np.abs(angles - 1.5 * np.pi), 2.0 * np.pi - np.abs(angles - 1.5 * np.pi))
    crossing_basis = np.exp(-0.5 * (np.minimum(positive_distance, negative_distance) / width) ** 2)
    side_sign = np.sign(np.sin(angles))
    for index in range(spec.upper_extension_samples):
        progress = float(index + 1) / float(spec.upper_extension_samples)
        row = surface[-1] + float(index + 1) * tangent_per_row
        row[:, 2] = surface[-1, :, 2] + progress * spec.upper_extension_nm
        row[:, 1] -= (
            side_sign
            * crossing_basis
            * spec.upper_extension_crossing_nm
            * float(_smoothstep(np.array([progress], dtype=np.float64))[0])
        )
        extension[index] = row
    return np.concatenate((surface, extension), axis=0)


def _initial_validation(spec: YJunctionWrapperSpec) -> dict[str, object]:
    z_values, angles, surface = _initial_surface(spec)
    surface = _apply_initial_self_overlap(surface, z_values, angles, spec)
    return _surface_validation(surface, _target_y_surface_points(spec), spec)


def optimize_y_junction_wrapper(
    spec: YJunctionWrapperSpec = YJunctionWrapperSpec(),
) -> OptimizedWrapper:
    _validate_spec(spec)
    z_values, angles, base_surface = _initial_surface(spec)
    base_surface = _apply_initial_self_overlap(base_surface, z_values, angles, spec)
    targets = _target_y_surface_points(spec)
    variables = np.zeros((spec.control_z_samples, spec.control_theta_samples, 3), dtype=np.float64).reshape(-1)
    last_result = None
    trace: list[dict[str, float | int | str | bool]] = [
        _trace_record(
            "initial",
            "accepted",
            0,
            variables,
            base_surface,
            targets,
            spec,
            coverage_weight=1.0,
        )
    ]
    if not spec.skip_optimization:
        for phase_index, coverage_weight in enumerate(spec.coverage_schedule):
            callback_count = 0

            def callback(xk: np.ndarray) -> None:
                nonlocal callback_count
                callback_count += 1
                trace.append(
                    _trace_record(
                        f"phase_{phase_index}",
                        "accepted",
                        callback_count,
                        xk,
                        base_surface,
                        targets,
                        spec,
                        coverage_weight,
                    )
                )

            last_result = minimize(
                _objective,
                variables,
                args=(base_surface, targets, spec, coverage_weight),
                method="L-BFGS-B",
                callback=callback,
                options={"maxiter": spec.max_iterations_per_phase, "ftol": 1e-5, "maxls": 10},
            )
            variables = last_result.x
            trace.append(
                _trace_record(
                    f"phase_{phase_index}",
                    "final",
                    int(last_result.nit),
                    variables,
                    base_surface,
                    targets,
                    spec,
                    coverage_weight,
                )
            )

    terms, surface = _terms(variables, base_surface, targets, spec, coverage_weight=1.0)
    mesh = _mesh_from_surface(surface)
    objective = {
        "success": bool(last_result.success) if last_result is not None else False,
        "iterations": int(last_result.nit) if last_result is not None else 0,
        "final_loss": float(last_result.fun) if last_result is not None else _objective(
            variables, base_surface, targets, spec, coverage_weight=1.0
        ),
        **terms,
    }
    return OptimizedWrapper(
        surface=surface,
        mesh=mesh,
        validation=_surface_validation(surface, targets, spec),
        objective=objective,
        trace=trace,
    )


def generate_y_junction_wrapper_mesh(
    spec: YJunctionWrapperSpec = YJunctionWrapperSpec(),
) -> trimesh.Trimesh:
    return optimize_y_junction_wrapper(spec).mesh


def export_y_junction_wrapper(
    output_path: str | Path = "outputs/y_junction_wrapper/y_junction_wrapper.obj",
    spec: YJunctionWrapperSpec = YJunctionWrapperSpec(),
) -> Path:
    path = _resolve_output_mesh_path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    result = optimize_y_junction_wrapper(spec)

    surface_points_path = path.with_name(f"{path.stem}_surface_points.csv")
    np.savetxt(
        surface_points_path,
        result.surface.reshape((-1, 3)),
        delimiter=",",
        header="x_nm,y_nm,z_nm",
        comments="",
    )

    result.mesh.export(path)
    glb_path = path.with_suffix(".glb")
    result.mesh.export(glb_path)

    metadata_path = path.with_suffix(".json")
    metadata_path.write_text(
        json.dumps(
            {
                "mesh_preview": str(path),
                "glb_preview": str(glb_path),
                "surface_points_csv": str(surface_points_path),
                "units": "nm",
                "primary_geometry": "coverage-optimized continuous cylinder surface coordinates",
                "forbidden_volume": "exact union of three 8 nm Y cylinders",
                "spec": asdict(spec),
                "objective": result.objective,
                "initial_validation": _initial_validation(spec),
                "validation": result.validation,
                "trace": result.trace,
                "vertex_count": int(result.mesh.vertices.shape[0]),
                "face_count": int(result.mesh.faces.shape[0]),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def _resolve_output_mesh_path(output_path: str | Path) -> Path:
    path = Path(output_path)
    if path.suffix == "":
        return path / "y_junction_wrapper.obj"
    if path.parent == Path("outputs"):
        return path.parent / path.stem / path.name
    return path


def _validate_spec(spec: YJunctionWrapperSpec) -> None:
    if spec.channel_radius_nm <= 0.0:
        raise ValueError("channel_radius_nm must be > 0")
    if spec.branch_length_nm <= 0.0:
        raise ValueError("branch_length_nm must be > 0")
    if not 0.0 < spec.upper_branch_angle_degrees < 90.0:
        raise ValueError("upper_branch_angle_degrees must be between 0 and 90")
    if spec.min_curvature_radius_nm <= 0.0:
        raise ValueError("min_curvature_radius_nm must be > 0")
    if spec.curvature_mode not in ("all", "vertical", "horizontal"):
        raise ValueError("curvature_mode must be 'all', 'vertical', or 'horizontal'")
    if spec.z_samples < 8:
        raise ValueError("z_samples must be >= 8")
    if spec.radial_samples < 8:
        raise ValueError("radial_samples must be >= 8")
    if spec.control_z_samples < 3:
        raise ValueError("control_z_samples must be >= 3")
    if spec.control_theta_samples < 4:
        raise ValueError("control_theta_samples must be >= 4")
    if spec.target_axis_samples < 2:
        raise ValueError("target_axis_samples must be >= 2")
    if spec.target_radial_samples < 8:
        raise ValueError("target_radial_samples must be >= 8")
    if spec.stem_clearance_nm < 0.0:
        raise ValueError("stem_clearance_nm must be >= 0")
    if spec.stem_blend_end_z_nm <= spec.stem_blend_start_z_nm:
        raise ValueError("stem_blend_end_z_nm must be > stem_blend_start_z_nm")
    if not 0.0 <= spec.overlap_start_fraction < spec.overlap_end_fraction <= 1.0:
        raise ValueError("overlap fractions must satisfy 0 <= start < end <= 1")
    if spec.overlap_swap_nm < 0.0:
        raise ValueError("overlap_swap_nm must be >= 0")
    if not 0.0 <= spec.overlap_swap_fraction <= 1.0:
        raise ValueError("overlap_swap_fraction must be between 0 and 1")
    if spec.overlap_angular_width_degrees <= 0.0:
        raise ValueError("overlap_angular_width_degrees must be > 0")
    if spec.overlap_axis not in ("x", "y"):
        raise ValueError("overlap_axis must be 'x' or 'y'")
    if spec.overlap_outward_nm < 0.0:
        raise ValueError("overlap_outward_nm must be >= 0")
    if spec.overlap_handle_outward_nm < 0.0:
        raise ValueError("overlap_handle_outward_nm must be >= 0")
    if spec.overlap_handle_offset_degrees <= 0.0:
        raise ValueError("overlap_handle_offset_degrees must be > 0")
    if spec.overlap_handle_width_degrees <= 0.0:
        raise ValueError("overlap_handle_width_degrees must be > 0")
    if spec.overlap_bezier_lobes and spec.overlap_axis != "y":
        raise ValueError("overlap_bezier_lobes requires overlap_axis='y'")
    if spec.overlap_crossing_width_degrees <= 0.0:
        raise ValueError("overlap_crossing_width_degrees must be > 0")
    if spec.overlap_edge_support_nm < 0.0:
        raise ValueError("overlap_edge_support_nm must be >= 0")
    if spec.overlap_edge_support_width_degrees <= 0.0:
        raise ValueError("overlap_edge_support_width_degrees must be > 0")
    if spec.overlap_handle_tangent_nm < 0.0:
        raise ValueError("overlap_handle_tangent_nm must be >= 0")
    if spec.overlap_control_width_degrees <= 0.0:
        raise ValueError("overlap_control_width_degrees must be > 0")
    if not 0.0 <= spec.overlap_radial_floor_fraction <= 1.0:
        raise ValueError("overlap_radial_floor_fraction must be between 0 and 1")
    if spec.overlap_smooth_sigma_z <= 0.0:
        raise ValueError("overlap_smooth_sigma_z must be > 0")
    if spec.overlap_smooth_sigma_theta <= 0.0:
        raise ValueError("overlap_smooth_sigma_theta must be > 0")
    if spec.overlap_lift_nm < 0.0:
        raise ValueError("overlap_lift_nm must be >= 0")
    if spec.radial_nonshrink_weight < 0.0:
        raise ValueError("radial_nonshrink_weight must be >= 0")
    if spec.upper_extension_nm < 0.0:
        raise ValueError("upper_extension_nm must be >= 0")
    if spec.upper_extension_samples < 0:
        raise ValueError("upper_extension_samples must be >= 0")
    if spec.upper_extension_crossing_nm < 0.0:
        raise ValueError("upper_extension_crossing_nm must be >= 0")
    if spec.upper_extension_crossing_width_degrees <= 0.0:
        raise ValueError("upper_extension_crossing_width_degrees must be > 0")
    if spec.coverage_scale_nm <= 0.0:
        raise ValueError("coverage_scale_nm must be > 0")
    if not spec.coverage_schedule:
        raise ValueError("coverage_schedule must contain at least one value")
    if any(weight < 0.0 for weight in spec.coverage_schedule):
        raise ValueError("coverage_schedule values must be >= 0")
    if spec.max_iterations_per_phase < 0:
        raise ValueError("max_iterations_per_phase must be >= 0")
    if spec.curvature_barrier_margin_nm < 0.0:
        raise ValueError("curvature_barrier_margin_nm must be >= 0")
    if spec.curvature_barrier_sharpness <= 0.0:
        raise ValueError("curvature_barrier_sharpness must be > 0")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Optimize a continuous cylinder surface to cover an exact Y."
    )
    parser.add_argument("--out", default="outputs/y_junction_wrapper/y_junction_wrapper.obj")
    parser.add_argument("--channel-radius-nm", type=float, default=8.0)
    parser.add_argument("--branch-length-nm", type=float, default=30.0)
    parser.add_argument("--upper-branch-angle-degrees", type=float, default=60.0)
    parser.add_argument("--min-curvature-radius-nm", type=float, default=6.0)
    parser.add_argument("--curvature-mode", choices=("all", "vertical", "horizontal"), default="all")
    parser.add_argument("--z-samples", type=int, default=40)
    parser.add_argument("--radial-samples", type=int, default=48)
    parser.add_argument("--control-z-samples", type=int, default=6)
    parser.add_argument("--control-theta-samples", type=int, default=8)
    parser.add_argument("--target-axis-samples", type=int, default=28)
    parser.add_argument("--target-radial-samples", type=int, default=24)
    parser.add_argument("--radial-search-samples", type=int, default=160)
    parser.add_argument("--initial-margin-nm", type=float, default=9.0)
    parser.add_argument("--stem-clearance-nm", type=float, default=1.5)
    parser.add_argument("--stem-blend-start-z-nm", type=float, default=-10.0)
    parser.add_argument("--stem-blend-end-z-nm", type=float, default=4.0)
    parser.add_argument("--base-gaussian-sigma-z", type=float, default=3.0)
    parser.add_argument("--base-gaussian-sigma-theta", type=float, default=2.0)
    parser.add_argument("--init-self-overlap", action="store_true")
    parser.add_argument("--overlap-start-fraction", type=float, default=0.67)
    parser.add_argument("--overlap-end-fraction", type=float, default=0.95)
    parser.add_argument("--overlap-swap-nm", type=float, default=1.0)
    parser.add_argument("--overlap-swap-fraction", type=float, default=0.0)
    parser.add_argument("--overlap-angular-width-degrees", type=float, default=35.0)
    parser.add_argument("--overlap-exact-columns", action="store_true")
    parser.add_argument("--overlap-axis", choices=("x", "y"), default="x")
    parser.add_argument("--overlap-outward-nm", type=float, default=0.0)
    parser.add_argument("--overlap-handle-outward-nm", type=float, default=0.0)
    parser.add_argument("--overlap-handle-offset-degrees", type=float, default=26.0)
    parser.add_argument("--overlap-handle-width-degrees", type=float, default=10.0)
    parser.add_argument("--overlap-sparse-controls", action="store_true")
    parser.add_argument("--overlap-bezier-lobes", action="store_true")
    parser.add_argument("--overlap-crossing-width-degrees", type=float, default=24.0)
    parser.add_argument("--overlap-edge-support-nm", type=float, default=0.0)
    parser.add_argument("--overlap-edge-support-width-degrees", type=float, default=10.0)
    parser.add_argument("--overlap-handle-tangent-nm", type=float, default=0.0)
    parser.add_argument("--overlap-control-width-degrees", type=float, default=8.0)
    parser.add_argument("--overlap-radial-floor-fraction", type=float, default=0.0)
    parser.add_argument("--overlap-smooth-displacement", action="store_true")
    parser.add_argument("--overlap-smooth-sigma-z", type=float, default=1.2)
    parser.add_argument("--overlap-smooth-sigma-theta", type=float, default=1.8)
    parser.add_argument("--overlap-lift-nm", type=float, default=0.0)
    parser.add_argument("--coverage-scale-nm", type=float, default=4.0)
    parser.add_argument("--coverage-schedule", type=float, nargs="+", default=[0.15, 0.15, 0.15])
    parser.add_argument("--max-iterations-per-phase", type=int, default=10)
    parser.add_argument("--skip-optimization", action="store_true")
    parser.add_argument("--penetration-weight", type=float, default=10000.0)
    parser.add_argument("--curvature-weight", type=float, default=500.0)
    parser.add_argument("--curvature-barrier-margin-nm", type=float, default=1.0)
    parser.add_argument("--curvature-barrier-sharpness", type=float, default=12.0)
    parser.add_argument("--smoothness-weight", type=float, default=0.08)
    parser.add_argument("--displacement-weight", type=float, default=0.004)
    parser.add_argument("--radial-nonshrink-weight", type=float, default=0.0)
    parser.add_argument("--upper-extension-nm", type=float, default=0.0)
    parser.add_argument("--upper-extension-samples", type=int, default=0)
    parser.add_argument("--upper-extension-crossing-nm", type=float, default=0.0)
    parser.add_argument("--upper-extension-crossing-width-degrees", type=float, default=24.0)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    spec = YJunctionWrapperSpec(
        channel_radius_nm=args.channel_radius_nm,
        branch_length_nm=args.branch_length_nm,
        upper_branch_angle_degrees=args.upper_branch_angle_degrees,
        min_curvature_radius_nm=args.min_curvature_radius_nm,
        curvature_mode=args.curvature_mode,
        z_samples=args.z_samples,
        radial_samples=args.radial_samples,
        control_z_samples=args.control_z_samples,
        control_theta_samples=args.control_theta_samples,
        target_axis_samples=args.target_axis_samples,
        target_radial_samples=args.target_radial_samples,
        radial_search_samples=args.radial_search_samples,
        initial_margin_nm=args.initial_margin_nm,
        stem_clearance_nm=args.stem_clearance_nm,
        stem_blend_start_z_nm=args.stem_blend_start_z_nm,
        stem_blend_end_z_nm=args.stem_blend_end_z_nm,
        base_gaussian_sigma_z=args.base_gaussian_sigma_z,
        base_gaussian_sigma_theta=args.base_gaussian_sigma_theta,
        init_self_overlap=args.init_self_overlap,
        overlap_start_fraction=args.overlap_start_fraction,
        overlap_end_fraction=args.overlap_end_fraction,
        overlap_swap_nm=args.overlap_swap_nm,
        overlap_swap_fraction=args.overlap_swap_fraction,
        overlap_angular_width_degrees=args.overlap_angular_width_degrees,
        overlap_exact_columns=args.overlap_exact_columns,
        overlap_axis=args.overlap_axis,
        overlap_outward_nm=args.overlap_outward_nm,
        overlap_handle_outward_nm=args.overlap_handle_outward_nm,
        overlap_handle_offset_degrees=args.overlap_handle_offset_degrees,
        overlap_handle_width_degrees=args.overlap_handle_width_degrees,
        overlap_sparse_controls=args.overlap_sparse_controls,
        overlap_bezier_lobes=args.overlap_bezier_lobes,
        overlap_crossing_width_degrees=args.overlap_crossing_width_degrees,
        overlap_edge_support_nm=args.overlap_edge_support_nm,
        overlap_edge_support_width_degrees=args.overlap_edge_support_width_degrees,
        overlap_handle_tangent_nm=args.overlap_handle_tangent_nm,
        overlap_control_width_degrees=args.overlap_control_width_degrees,
        overlap_radial_floor_fraction=args.overlap_radial_floor_fraction,
        overlap_smooth_displacement=args.overlap_smooth_displacement,
        overlap_smooth_sigma_z=args.overlap_smooth_sigma_z,
        overlap_smooth_sigma_theta=args.overlap_smooth_sigma_theta,
        overlap_lift_nm=args.overlap_lift_nm,
        coverage_scale_nm=args.coverage_scale_nm,
        coverage_schedule=tuple(args.coverage_schedule),
        max_iterations_per_phase=args.max_iterations_per_phase,
        skip_optimization=args.skip_optimization,
        penetration_weight=args.penetration_weight,
        curvature_weight=args.curvature_weight,
        curvature_barrier_margin_nm=args.curvature_barrier_margin_nm,
        curvature_barrier_sharpness=args.curvature_barrier_sharpness,
        smoothness_weight=args.smoothness_weight,
        displacement_weight=args.displacement_weight,
        radial_nonshrink_weight=args.radial_nonshrink_weight,
        upper_extension_nm=args.upper_extension_nm,
        upper_extension_samples=args.upper_extension_samples,
        upper_extension_crossing_nm=args.upper_extension_crossing_nm,
        upper_extension_crossing_width_degrees=args.upper_extension_crossing_width_degrees,
    )
    output_path = export_y_junction_wrapper(args.out, spec)
    print(f"Saved OBJ preview to: {output_path}")
    print(f"Saved GLB preview to: {output_path.with_suffix('.glb')}")
    print(
        "Saved surface point CSV to: "
        f"{output_path.with_name(f'{output_path.stem}_surface_points.csv')}"
    )
    print(f"Saved metadata to: {output_path.with_suffix('.json')}")


if __name__ == "__main__":
    main()
