from __future__ import annotations

import numpy as np

from autorigami.acap_integration import compute_toolpath_stats, load_acap_polyline


def test_load_acap_obj_vertices(tmp_path) -> None:
    path = tmp_path / "path.obj"
    path.write_text(
        "v 0 0 0\n"
        "v 1.5 0 0\n"
        "l 1 2\n",
        encoding="utf-8",
    )

    points = load_acap_polyline(path)

    np.testing.assert_allclose(points, np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]]))


def test_load_acap_gcode_motion_points(tmp_path) -> None:
    path = tmp_path / "path.gcode"
    path.write_text(
        "G1 X0 Y0 Z0\n"
        "G1 X1.25\n"
        "G0 Y2.5 ; travel still records geometry for QC\n",
        encoding="utf-8",
    )

    points = load_acap_polyline(path)

    np.testing.assert_allclose(
        points,
        np.array(
            [
                [0.0, 0.0, 0.0],
                [1.25, 0.0, 0.0],
                [1.25, 2.5, 0.0],
            ]
        ),
    )


def test_compute_toolpath_stats_reports_native_separation() -> None:
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [2.0, 0.4, 0.0],
            [0.0, 0.4, 0.0],
        ],
        dtype=np.float64,
    )

    stats = compute_toolpath_stats(
        points=points,
        minimum_separation_world=0.5,
        nonlocal_window_world=0.2,
        world_to_nm=20.0,
    )

    assert stats["format_family"] == "acap_polyline_toolpath"
    assert stats["point_count"] == 5
    assert stats["nonlocal_violation_count"] > 0
    assert stats["minimum_checked_distance_nm"] < 10.0
