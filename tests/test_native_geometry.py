import numpy as np

from autorigami._native import (
    apply_separation_correction,
    evaluate_tangent_point_exact,
    evaluate_tangent_point_hierarchical,
    find_close_edge_pairs,
    segment_segment_distance,
    segment_segment_distance_parameters,
)


def test_find_close_edge_pairs_filters_by_exact_distance() -> None:
    polyline = np.array(
        [[0, 0, 0], [1, 0, 0], [2, 0, 0], [0, 0.5, 0], [1, 0.5, 0]],
        dtype=np.float32,
    )
    assert find_close_edge_pairs(polyline, 0.6, 1) == [(0, 2), (0, 3), (1, 3)]


def test_find_close_edge_pairs_matches_exhaustive_oracle() -> None:
    rng = np.random.default_rng(41)
    polyline = np.cumsum(rng.normal(size=(40, 3)), axis=0).astype(np.float32)
    candidate_pairs = [
        (first, second)
        for first in range(len(polyline) - 1)
        for second in range(first + 4, len(polyline) - 1)
    ]
    distances = segment_segment_distance(polyline, candidate_pairs)
    expected = [
        pair
        for pair, (distance, _, _) in zip(candidate_pairs, distances, strict=True)
        if distance <= 0.75
    ]
    assert find_close_edge_pairs(polyline, 0.75, 3, leaf_size=3) == expected


def test_segment_segment_distance_accepts_candidate_edge_pairs() -> None:
    polyline = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, -1.0, 0.0],
            [0.5, 1.0, 0.0],
            [0.0, 2.0, 0.0],
            [1.0, 2.0, 0.0],
        ],
        dtype=np.float32,
    )
    candidate_pairs = [(0, 2), (0, 4)]

    distances = segment_segment_distance(polyline, candidate_pairs)
    parameter_data = segment_segment_distance_parameters(polyline, candidate_pairs)

    assert len(distances) == 2
    np.testing.assert_allclose(parameter_data[:, 0], [item[0] for item in distances])
    assert parameter_data.shape == (2, 3)
    crossing_distance, crossing_p, crossing_q = distances[0]
    parallel_distance, parallel_p, parallel_q = distances[1]

    assert np.isclose(crossing_distance, 0.0)
    assert np.allclose(crossing_p, [0.5, 0.0, 0.0])
    assert np.allclose(crossing_q, [0.5, 0.0, 0.0])
    assert np.isclose(parallel_distance, 2.0)
    assert np.allclose(parallel_p, [0.5, 0.0, 0.0])
    assert np.allclose(parallel_q, [0.5, 2.0, 0.0])


def test_apply_separation_correction_corrects_an_isolated_pair() -> None:
    polyline = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, -1.0, 0.0],
            [0.5, 1.0, 0.0],
        ],
        dtype=np.float32,
    )

    corrected, correction_count = apply_separation_correction(
        polyline,
        1,
        [(0, 2)],
        min_distance=0.25,
    )
    corrected_array = np.asarray(corrected, dtype=np.float32)
    corrected_distance = segment_segment_distance(corrected_array, [(0, 2)])[0][0]

    assert correction_count == 1
    assert np.isclose(corrected_distance, 0.25)


def test_native_tangent_point_evaluation_matches_python_oracle() -> None:
    polyline = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.1, 0.0],
            [2.0, 0.4, 0.2],
            [2.0, 2.0, 0.3],
            [1.0, 2.2, 0.1],
            [0.0, 2.0, 0.0],
        ],
        dtype=np.float32,
    )
    exact = evaluate_tangent_point_exact(polyline, 1.3, 0.7, 1.8)
    hierarchical = evaluate_tangent_point_hierarchical(
        polyline, 1.3, 0.7, 1.8, opening_angle=1e-6, leaf_size=1
    )

    assert np.isclose(hierarchical["energy"], exact["energy"], atol=1e-12)
    np.testing.assert_allclose(
        hierarchical["differential"], exact["differential"], atol=1e-12
    )
