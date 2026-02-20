import numpy as np
import pytest
from scipy.spatial import cKDTree  # type: ignore[reportMissingTypeStubs]
from shapely.geometry import Point, Polygon
from shapely.prepared import prep

from src.lattice import honeycomb_lattice_points


def _covers_all(poly: Polygon, P: np.ndarray) -> bool:
    prepped = prep(poly)
    covers_fn = getattr(prepped, "covers", None)
    if covers_fn is None:
        covers_fn = poly.covers
    return all(bool(covers_fn(Point(float(x), float(y)))) for x, y in P)


def test_honeycomb_lattice_points_inside_polygon() -> None:
    V = np.array(
        [[0.0, 0.0], [100.0, 0.0], [100.0, 80.0], [0.0, 80.0]], dtype=np.float32
    )
    d_units = 2.6
    P = honeycomb_lattice_points(V, neighbor_dist=d_units)

    assert P.ndim == 2
    assert P.shape[1] == 2
    assert P.dtype == np.float32
    assert P.shape[0] > 0

    poly = Polygon(V)
    assert _covers_all(poly, P)

    tree = cKDTree(P)
    dists, _ = tree.query(P, k=2)
    nn = dists[:, 1]
    assert float(nn.min()) > 0.9 * d_units
    assert float(np.median(nn)) == pytest.approx(d_units, rel=1e-4, abs=1e-4)


def test_honeycomb_lattice_core_has_three_neighbors() -> None:
    V = np.array(
        [[0.0, 0.0], [120.0, 0.0], [120.0, 120.0], [0.0, 120.0]], dtype=np.float32
    )
    d_units = 2.6
    P = honeycomb_lattice_points(V, neighbor_dist=d_units)
    assert P.shape[0] > 50

    poly = Polygon(V)
    core = poly.buffer(-1.5 * d_units)
    assert not core.is_empty

    prepped = prep(core)
    covers_fn = getattr(prepped, "covers", None)
    if covers_fn is None:
        covers_fn = core.covers
    core_mask = np.array(
        [bool(covers_fn(Point(float(x), float(y)))) for x, y in P], dtype=bool
    )
    P_core = P[core_mask]
    assert P_core.shape[0] > 20

    tree = cKDTree(P)
    # self + 3 nearest neighbors at distance d, then next shell at ~sqrt(3)*d.
    dists, _ = tree.query(P_core, k=5)
    np.testing.assert_allclose(dists[:, 1:4], d_units, rtol=1e-4, atol=1e-4)
    assert bool(np.all(dists[:, 4] > 1.5 * d_units))


def test_honeycomb_lattice_scale_nm_per_unit() -> None:
    # Simulate an SVG where 1 user unit = 0.5 nm.
    nm_per_unit = 0.5
    d_nm = 2.6
    d_units = d_nm / nm_per_unit

    V = np.array(
        [[0.0, 0.0], [300.0, 0.0], [300.0, 200.0], [0.0, 200.0]], dtype=np.float32
    )
    P = honeycomb_lattice_points(V, neighbor_dist=d_units)
    assert P.shape[0] > 0

    tree = cKDTree(P)
    dists, _ = tree.query(P, k=2)
    nn_nm = dists[:, 1] * nm_per_unit
    assert float(np.median(nn_nm)) == pytest.approx(d_nm, rel=1e-4, abs=1e-3)


def test_honeycomb_lattice_rejects_invalid_inputs() -> None:
    V = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]], dtype=np.float32)
    with pytest.raises(ValueError):
        honeycomb_lattice_points(V, neighbor_dist=0.0)
    with pytest.raises(ValueError):
        honeycomb_lattice_points(V, neighbor_dist=-1.0)
    with pytest.raises(ValueError):
        honeycomb_lattice_points(V[:2], neighbor_dist=2.6)
