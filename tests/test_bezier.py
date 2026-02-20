import numpy as np

from src.curvepack.bezier import beziers_to_svg_path_d, polyline_to_cubic_beziers


def test_polyline_to_beziers_endpoints_and_count() -> None:
    P = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 1.0], [3.0, 1.0]], dtype=np.float32)
    segs = polyline_to_cubic_beziers(P, handle_scale=1.0)
    assert len(segs) == P.shape[0] - 1
    assert np.allclose(segs[0][0], P[0])
    assert np.allclose(segs[-1][3], P[-1])
    for i in range(len(segs)):
        assert np.allclose(segs[i][0], P[i])
        assert np.allclose(segs[i][3], P[i + 1])


def test_handle_scale_zero_gives_straight_segments() -> None:
    P = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]], dtype=np.float32)
    segs = polyline_to_cubic_beziers(P, handle_scale=0.0)
    for p0, c1, c2, p3 in segs:
        # Collinear along the segment, handles collapse to endpoints
        assert np.allclose(c1, p0)
        assert np.allclose(c2, p3)


def test_svg_path_string_nonempty() -> None:
    P = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 1.0]], dtype=np.float32)
    segs = polyline_to_cubic_beziers(P, handle_scale=0.5)
    d = beziers_to_svg_path_d(segs)
    assert d.startswith("M ")
    assert " C " in d


def test_c1_continuity_when_not_clamped() -> None:
    # Straight-ish polyline with equal spacing: no clamping should be active.
    P = np.array(
        [[0.0, 0.0], [1.0, 0.2], [2.0, 0.0], [3.0, 0.2], [4.0, 0.0]], dtype=np.float32
    )
    segs = polyline_to_cubic_beziers(P, handle_scale=1.0, max_handle_ratio=10.0)
    # Derivative continuity at internal knots:
    # left derivative at p = 3*(p - c2_prev)
    # right derivative at p = 3*(c1_next - p)
    for i in range(1, len(P) - 1):
        _p0, _c1, c2_prev, p = segs[i - 1]
        p0_next, c1_next, _c2_next, _p3_next = segs[i]
        assert np.allclose(p0_next, p)
        d_left = 3.0 * (p - c2_prev)
        d_right = 3.0 * (c1_next - p)
        assert np.allclose(d_left, d_right, atol=1e-5, rtol=1e-5)
