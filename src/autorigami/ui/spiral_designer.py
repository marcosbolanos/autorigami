from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from streamlit_plotly_events import plotly_events

from autorigami._native import validate_piecewise_curve_curvature
from autorigami.spiral_generation.control_point_spiral import (
    SpiralValidationSummary,
    default_control_points,
    generate_natural_control_curve,
    piecewise_hermite_from_polyline,
    summarize_spiral_validation,
)
from autorigami.types import FloatArray

DNA_RADIUS_NM = 2.6
MIN_CENTERLINE_SEPARATION_NM = 2.6
MIN_CURVATURE_RADIUS_NM = 6.0
MAX_CURVATURE_NM_INV = 1.0 / MIN_CURVATURE_RADIUS_NM
CURVATURE_TOLERANCE = 1e-6
DEFAULT_SAMPLE_COUNT = 360
DEFAULT_NEIGHBOR_SKIP = 18
DEFAULT_SLICE_HALF_WIDTH_NM = 25.0
SELECTED_Z_TOLERANCE_NM = 0.35


def main() -> None:
    st.set_page_config(page_title="Autorigami Spiral Designer", layout="wide")
    st.title("Autorigami Spiral Designer")

    if "control_points" not in st.session_state:
        st.session_state.control_points = default_control_points()
    if "z_range" not in st.session_state:
        points = np.asarray(st.session_state.control_points, dtype=np.float64)
        st.session_state.z_range = (
            float(points[:, 2].min()),
            float(points[:, 2].max()),
        )
    if "active_z" not in st.session_state:
        st.session_state.active_z = float(st.session_state.z_range[0])

    left, right = st.columns([0.95, 1.05], gap="large")
    with left:
        control_points = _normalized_control_points(st.session_state.control_points)
        z_min, z_max = _z_range_picker(control_points)
        active_z = st.slider(
            "z slice",
            min_value=z_min,
            max_value=z_max,
            value=float(np.clip(st.session_state.active_z, z_min, z_max)),
            step=0.1,
            format="%.1f nm",
        )
        st.session_state.active_z = active_z

        control_points = _trim_control_points_to_z_range(control_points, z_min, z_max)
        st.session_state.control_points = control_points

        st.subheader("Slice")
        clicked = plotly_events(
            _slice_figure(control_points, active_z, _slice_half_width(control_points)),
            click_event=True,
            hover_event=False,
            select_event=False,
            override_height=620,
            key="slice_click",
        )
        if clicked:
            point = clicked[0]
            st.session_state.control_points = _upsert_control_point(
                control_points,
                x=float(point["x"]),
                y=float(point["y"]),
                z=active_z,
            )
            st.rerun()

        button_a, button_b = st.columns(2)
        if button_a.button("Delete slice point", width="stretch"):
            st.session_state.control_points = _delete_nearest_z_point(
                control_points,
                active_z,
            )
            st.rerun()
        if button_b.button("Reset", width="stretch"):
            st.session_state.control_points = default_control_points()
            st.session_state.z_range = (
                float(st.session_state.control_points[:, 2].min()),
                float(st.session_state.control_points[:, 2].max()),
            )
            st.session_state.active_z = float(st.session_state.z_range[0])
            st.rerun()

        _control_point_list(st.session_state.control_points)

    with right:
        st.subheader("3D Render")
        try:
            polyline = generate_natural_control_curve(
                control_points=_normalized_control_points(
                    st.session_state.control_points
                ),
                sample_count=DEFAULT_SAMPLE_COUNT,
            )
            hermite = piecewise_hermite_from_polyline(polyline)
            curvature_passes = validate_piecewise_curve_curvature(
                piecewise_hermite=hermite,
                max_curvature=MAX_CURVATURE_NM_INV,
                curvature_tolerance=CURVATURE_TOLERANCE,
            )
            summary = summarize_spiral_validation(
                polyline=polyline,
                curvature_passes=curvature_passes,
                min_separation=MIN_CENTERLINE_SEPARATION_NM,
                neighbor_skip=DEFAULT_NEIGHBOR_SKIP,
            )
        except ValueError as error:
            st.error(str(error))
            return

        st.plotly_chart(
            _tube_figure(
                centerline=polyline.points,
                control_points=_normalized_control_points(
                    st.session_state.control_points
                ),
                radius=DNA_RADIUS_NM,
            ),
            width="stretch",
            config={"scrollZoom": True, "displaylogo": False},
        )
        _validation_panel(summary)


def _z_range_picker(control_points: FloatArray) -> tuple[float, float]:
    current_min = float(control_points[:, 2].min())
    current_max = float(control_points[:, 2].max())
    range_floor = min(0.0, current_min)
    range_ceiling = max(120.0, current_max)
    selected = st.slider(
        "z range",
        min_value=range_floor,
        max_value=range_ceiling,
        value=st.session_state.z_range,
        step=0.5,
        format="%.1f nm",
    )
    z_min, z_max = float(selected[0]), float(selected[1])
    if z_max <= z_min:
        z_max = z_min + 0.5
    st.session_state.z_range = (z_min, z_max)
    return z_min, z_max


def _slice_figure(
    control_points: FloatArray,
    active_z: float,
    half_width: float,
) -> go.Figure:
    nearby_mask = np.abs(control_points[:, 2] - active_z) <= SELECTED_Z_TOLERANCE_NM
    figure = go.Figure()
    click_grid = _slice_click_grid(half_width)
    figure.add_trace(
        go.Scatter(
            x=click_grid[:, 0],
            y=click_grid[:, 1],
            mode="markers",
            name="click target",
            marker={
                "size": 8,
                "color": "rgba(15, 23, 42, 0.01)",
            },
            hoverinfo="skip",
            showlegend=False,
        )
    )
    figure.add_trace(
        go.Scatter(
            x=control_points[:, 0],
            y=control_points[:, 1],
            mode="markers",
            name="all z controls",
            marker={"size": 7, "color": "#94a3b8"},
            hovertemplate="x=%{x:.2f}<br>y=%{y:.2f}<extra></extra>",
        )
    )
    if np.any(nearby_mask):
        selected = control_points[nearby_mask]
        figure.add_trace(
            go.Scatter(
                x=selected[:, 0],
                y=selected[:, 1],
                mode="markers",
                name="slice control",
                marker={"size": 14, "color": "#dc2626", "symbol": "x"},
                hovertemplate="x=%{x:.2f}<br>y=%{y:.2f}<extra></extra>",
            )
        )
    figure.update_layout(
        margin={"l": 8, "r": 8, "t": 8, "b": 8},
        clickmode="event+select",
        showlegend=False,
        xaxis={
            "title": "x nm",
            "range": [-half_width, half_width],
            "zeroline": True,
            "scaleanchor": "y",
            "scaleratio": 1,
        },
        yaxis={
            "title": "y nm",
            "range": [-half_width, half_width],
            "zeroline": True,
        },
        plot_bgcolor="#f8fafc",
    )
    return figure


def _slice_click_grid(half_width: float) -> FloatArray:
    coordinates = np.linspace(-half_width, half_width, 121, dtype=np.float64)
    x_values, y_values = np.meshgrid(coordinates, coordinates)
    return np.column_stack([x_values.ravel(), y_values.ravel()]).astype(np.float64)


def _tube_figure(
    centerline: FloatArray,
    control_points: FloatArray,
    radius: float,
) -> go.Figure:
    vertices, faces = _tube_mesh(centerline, radius=radius, sides=18)
    i, j, k = faces[:, 0], faces[:, 1], faces[:, 2]
    figure = go.Figure()
    figure.add_trace(
        go.Mesh3d(
            x=vertices[:, 0],
            y=vertices[:, 1],
            z=vertices[:, 2],
            i=i,
            j=j,
            k=k,
            name="DNA tube",
            color="#2563eb",
            opacity=0.92,
            flatshading=False,
            lighting={"ambient": 0.45, "diffuse": 0.8, "specular": 0.25},
        )
    )
    figure.add_trace(
        go.Scatter3d(
            x=centerline[:, 0],
            y=centerline[:, 1],
            z=centerline[:, 2],
            mode="lines",
            name="centerline",
            line={"color": "#0f172a", "width": 3},
        )
    )
    figure.add_trace(
        go.Scatter3d(
            x=control_points[:, 0],
            y=control_points[:, 1],
            z=control_points[:, 2],
            mode="markers",
            name="controls",
            marker={"color": "#dc2626", "size": 4},
        )
    )
    figure.update_layout(
        margin={"l": 0, "r": 0, "t": 8, "b": 0},
        height=720,
        scene={
            "aspectmode": "data",
            "xaxis_title": "x nm",
            "yaxis_title": "y nm",
            "zaxis_title": "z nm",
        },
        legend={"orientation": "h", "y": 0.98},
    )
    return figure


def _tube_mesh(
    centerline: FloatArray,
    radius: float,
    sides: int,
) -> tuple[FloatArray, np.ndarray]:
    tangents = np.gradient(centerline, axis=0)
    norms = np.linalg.norm(tangents, axis=1)
    tangents = tangents / np.maximum(norms[:, None], 1e-12)

    vertices: list[np.ndarray] = []
    previous_normal = _initial_normal(tangents[0])
    angles = np.linspace(0.0, 2.0 * np.pi, sides, endpoint=False)
    for point, tangent in zip(centerline, tangents):
        normal = previous_normal - tangent * float(np.dot(previous_normal, tangent))
        normal_norm = np.linalg.norm(normal)
        if normal_norm < 1e-9:
            normal = _initial_normal(tangent)
        else:
            normal = normal / normal_norm
        binormal = np.cross(tangent, normal)
        previous_normal = normal
        ring = point + radius * (
            np.cos(angles)[:, None] * normal[None, :]
            + np.sin(angles)[:, None] * binormal[None, :]
        )
        vertices.extend(ring)

    face_rows: list[tuple[int, int, int]] = []
    for ring_index in range(centerline.shape[0] - 1):
        start = ring_index * sides
        next_start = (ring_index + 1) * sides
        for side in range(sides):
            a = start + side
            b = start + (side + 1) % sides
            c = next_start + side
            d = next_start + (side + 1) % sides
            face_rows.append((a, c, b))
            face_rows.append((b, c, d))

    return np.asarray(vertices, dtype=np.float64), np.asarray(face_rows, dtype=np.int64)


def _initial_normal(tangent: FloatArray) -> FloatArray:
    candidate = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    if abs(float(np.dot(candidate, tangent))) > 0.85:
        candidate = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    normal = candidate - tangent * float(np.dot(candidate, tangent))
    return np.asarray(normal / np.linalg.norm(normal), dtype=np.float64)


def _normalized_control_points(points: FloatArray) -> FloatArray:
    points = np.asarray(points, dtype=np.float64)
    order = np.argsort(points[:, 2], kind="stable")
    sorted_points = points[order]
    _, unique_indices = np.unique(sorted_points[:, 2], return_index=True)
    return np.asarray(sorted_points[np.sort(unique_indices)], dtype=np.float64)


def _slice_half_width(control_points: FloatArray) -> float:
    xy_extent = float(np.max(np.abs(control_points[:, :2]))) + DNA_RADIUS_NM
    return max(DEFAULT_SLICE_HALF_WIDTH_NM, np.ceil(xy_extent / 5.0) * 5.0)


def _trim_control_points_to_z_range(
    control_points: FloatArray,
    z_min: float,
    z_max: float,
) -> FloatArray:
    keep = (control_points[:, 2] >= z_min) & (control_points[:, 2] <= z_max)
    points = control_points[keep]
    if points.shape[0] == 0:
        points = np.array(
            [
                [0.0, 0.0, z_min],
                [0.0, 0.0, z_max],
            ],
            dtype=np.float64,
        )
    return _ensure_range_endpoints(_normalized_control_points(points), z_min, z_max)


def _ensure_range_endpoints(
    control_points: FloatArray,
    z_min: float,
    z_max: float,
) -> FloatArray:
    points = control_points
    if not np.any(np.isclose(points[:, 2], z_min)):
        points = _upsert_control_point(points, x=0.0, y=0.0, z=z_min)
    if not np.any(np.isclose(points[:, 2], z_max)):
        points = _upsert_control_point(points, x=0.0, y=0.0, z=z_max)
    return _normalized_control_points(points)


def _upsert_control_point(
    control_points: FloatArray,
    x: float,
    y: float,
    z: float,
) -> FloatArray:
    keep = np.abs(control_points[:, 2] - z) > SELECTED_Z_TOLERANCE_NM
    points = np.vstack(
        [
            control_points[keep],
            np.array([[x, y, z]], dtype=np.float64),
        ]
    )
    return _normalized_control_points(points)


def _delete_nearest_z_point(
    control_points: FloatArray,
    active_z: float,
) -> FloatArray:
    if control_points.shape[0] <= 2:
        return control_points
    distances = np.abs(control_points[:, 2] - active_z)
    nearest = int(np.argmin(distances))
    if distances[nearest] > SELECTED_Z_TOLERANCE_NM:
        return control_points
    return np.delete(control_points, nearest, axis=0)


def _control_point_list(control_points: FloatArray) -> None:
    st.subheader("Controls")
    display = [
        {"z": f"{point[2]:.1f}", "x": f"{point[0]:.2f}", "y": f"{point[1]:.2f}"}
        for point in _normalized_control_points(control_points)
    ]
    st.dataframe(display, hide_index=True, width="stretch")


def _validation_panel(summary: SpiralValidationSummary) -> None:
    curvature_label = "pass" if summary.curvature_passes else "fail"
    separation_label = "pass" if summary.separation_passes else "fail"
    col_a, col_b = st.columns(2)
    col_a.metric("Min radius", _format_radius(summary.approximate_min_curvature_radius))
    col_b.metric("Curvature", curvature_label)
    col_c, col_d = st.columns(2)
    col_c.metric(
        "Min centerline separation", f"{summary.approximate_min_separation:.2f} nm"
    )
    col_d.metric("Separation", separation_label)
    st.caption(
        "Fixed validation: DNA tube radius 2.6 nm, minimum centerline separation "
        "2.6 nm, minimum curvature radius 6.0 nm."
    )


def _format_radius(radius: float) -> str:
    if np.isinf(radius):
        return "inf"
    return f"{radius:.2f} nm"


if __name__ == "__main__":
    main()
