from __future__ import annotations

import numpy as np
from beartype import beartype
from jaxtyping import jaxtyped

from ..utils import debug, debug_helpers
from .optimize_types import NpCurveSamples, PairsNp


@jaxtyped(typechecker=beartype)
def build_pairs_spatial_hash(
    X: NpCurveSamples,
    r_max: float,
    cell: float,
    exclude_adj: int = 2,
    max_pairs: int = 200000,
) -> PairsNp:
    """
    Build candidate segment pairs for separation loss using a uniform grid hash.
    X: (C,M,2) samples in numpy
    r_max: max tube radius + clearance
    cell: grid cell size (recommend ~ r_max)
    exclude_adj: for same curve, exclude segment pairs with |k-l| <= exclude_adj
    Returns (pair_i, pair_k, pair_j, pair_l) int32 arrays.
    """
    C, M, _ = X.shape
    S = M - 1

    A = X[:, :-1, :]
    B = X[:, 1:, :]
    mid = 0.5 * (A + B)

    if debug.is_verbose():
        finite_mid = bool(np.isfinite(mid).all())
        if np.isfinite(mid).any():
            mid_min = float(np.min(mid))
            mid_max = float(np.max(mid))
        else:
            mid_min = float("nan")
            mid_max = float("nan")
        debug_helpers.log_once(
            "pair_hash_stats",
            f"pair_hash: cell={cell:.6g} r_max={r_max:.6g} "
            f"mid_finite={finite_mid} mid_min={mid_min:.6g} mid_max={mid_max:.6g}",
        )
        if not np.isfinite(cell) or cell <= 0:
            debug_helpers.log_once("pair_hash_bad_cell", f"pair_hash bad cell: {cell}")
        if not finite_mid:
            debug_helpers.log_once(
                "pair_hash_mid_nonfinite", "pair_hash mid non-finite"
            )
            debug_helpers.log_array("mid", mid)

    mins = mid.reshape(-1, 2).min(axis=0)
    if debug.is_verbose() and not np.isfinite(mins).all():
        debug_helpers.log_once(
            "pair_hash_mins_nonfinite",
            f"pair_hash mins non-finite: mins=({mins[0]}, {mins[1]})",
        )
    if debug.is_verbose() and np.isfinite(cell) and cell > 0:
        coords = (mid - mins) / cell
        if not np.isfinite(coords).all():
            debug_helpers.log_once(
                "pair_hash_coords_nonfinite", "pair_hash coords non-finite"
            )
            debug_helpers.log_array("pair_coords", coords)
        else:
            max_abs = float(np.max(np.abs(coords)))
            if max_abs > 2**30:
                debug_helpers.log_once(
                    "pair_hash_coords_huge",
                    f"pair_hash coords huge: max_abs={max_abs:.6g}",
                )
    gx = np.floor((mid[..., 0] - mins[0]) / cell).astype(np.int32)
    gy = np.floor((mid[..., 1] - mins[1]) / cell).astype(np.int32)

    buckets: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for i in range(C):
        for k in range(S):
            key = (gx[i, k], gy[i, k])
            buckets.setdefault(key, []).append((i, k))

    pair_i: list[int] = []
    pair_k: list[int] = []
    pair_j: list[int] = []
    pair_l: list[int] = []

    def add_pair(i: int, k: int, j: int, l: int) -> None:
        pair_i.append(i)
        pair_k.append(k)
        pair_j.append(j)
        pair_l.append(l)

    neigh: list[tuple[int, int]] = [
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 0),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    ]
    for key, items in buckets.items():
        bx, by = key
        cand: list[tuple[int, int]] = []
        for dx, dy in neigh:
            cand.extend(buckets.get((bx + dx, by + dy), []))
        for i, k in items:
            for j, l in cand:
                if (j < i) or (j == i and l <= k):
                    continue
                if i == j and abs(k - l) <= exclude_adj:
                    continue
                add_pair(i, k, j, l)
                if len(pair_i) >= max_pairs:
                    break
            if len(pair_i) >= max_pairs:
                break
        if len(pair_i) >= max_pairs:
            break

    return (
        np.asarray(pair_i, np.int32),
        np.asarray(pair_k, np.int32),
        np.asarray(pair_j, np.int32),
        np.asarray(pair_l, np.int32),
    )
