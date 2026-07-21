from pathlib import Path

import numpy as np
from PIL import Image

from autorigami.geometry.validation import (
    PolylineViolationMasks,
    validate_polyline,
)
from autorigami.mesh_io import render_dna_molecule_png


def test_standard_validation_returns_both_violation_masks() -> None:
    polyline = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [1.0, 0.5, 0.0],
            [0.0, 0.5, 0.0],
        ],
        dtype=np.float32,
    )

    result = validate_polyline(
        polyline,
        maximum_angle=0.5,
        minimum_distance=0.6,
        valid_curvature_ignored_adjacent_edges=3,
        include_violation_masks=True,
    )

    assert not result.valid
    assert result.curvature_violation_count > 0
    assert result.separation_violation_count > 0
    assert result.ignored_adjacent_edges == 1
    assert result.violation_masks is not None
    assert np.any(result.violation_masks.curvature_vertices)
    assert np.any(result.violation_masks.separation_edges)


def test_dna_renderer_draws_both_violation_overlays(tmp_path: Path) -> None:
    polyline = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.5, 0.5, 0.0],
            [2.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    masks = PolylineViolationMasks(
        separation_edges=np.array([False, True, False]),
        curvature_vertices=np.array([False, False, True, False]),
    )
    output = tmp_path / "annotated.png"

    render_dna_molecule_png(
        polyline,
        output,
        violation_masks=masks,
        camera_view="xy",
        window_size=(400, 300),
    )

    image = np.asarray(Image.open(output).convert("RGB"))
    red_pixels = (
        (image[:, :, 0] > 180) & (image[:, :, 1] < 100) & (image[:, :, 2] < 100)
    )
    assert np.any(red_pixels)
