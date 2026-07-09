import numpy as np

from autorigami._native import segment_segment_distance


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

    assert len(distances) == 2
    crossing_distance, crossing_p, crossing_q = distances[0]
    parallel_distance, parallel_p, parallel_q = distances[1]

    assert np.isclose(crossing_distance, 0.0)
    assert np.allclose(crossing_p, [0.5, 0.0, 0.0])
    assert np.allclose(crossing_q, [0.5, 0.0, 0.0])
    assert np.isclose(parallel_distance, 2.0)
    assert np.allclose(parallel_p, [0.0, 0.0, 0.0])
    assert np.allclose(parallel_q, [0.0, 2.0, 0.0])
