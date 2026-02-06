from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from ..curvepack.bspline import make_basis_matrix
from ..curvepack.sdf import sample_interior_points


def _find_latest_step(run_dir: Path) -> int:
    steps: list[int] = []
    for path in run_dir.glob("step_*.npz"):
        name = path.stem
        try:
            step = int(name.split("_")[-1])
        except ValueError:
            continue
        steps.append(step)
    if not steps:
        raise ValueError(f"No checkpoint npz files found in {run_dir}")
    return max(steps)


def _load_metadata(run_dir: Path) -> dict[str, object]:
    metadata_path = run_dir / "metadata.json"
    if not metadata_path.exists():
        raise ValueError(f"Missing metadata.json in {run_dir}")
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def _load_rasters(run_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    raster_path = run_dir / "rasters.npz"
    if not raster_path.exists():
        raise ValueError(f"Missing rasters.npz in {run_dir}")
    data = np.load(raster_path)
    if "mask" not in data:
        raise ValueError("rasters.npz missing mask")
    mask = data["mask"].astype(bool)
    sdf = data["sdf"].astype(np.float32)
    origin = data["origin"].astype(np.float32)
    h = float(data["h"])
    return mask, sdf, origin, h


def _border_from_mask(mask: np.ndarray) -> np.ndarray:
    mask_pad = np.pad(mask, 1, mode="constant", constant_values=False)
    core = mask_pad[1:-1, 1:-1]
    neighbors = (
        mask_pad[:-2, 1:-1]
        & mask_pad[2:, 1:-1]
        & mask_pad[1:-1, :-2]
        & mask_pad[1:-1, 2:]
    )
    return core & (~neighbors)


def _point_segment_dist2(
    p: np.ndarray, a: np.ndarray, b: np.ndarray, eps: float = 1e-9
) -> np.ndarray:
    ab = b - a
    denom = np.sum(ab * ab, axis=-1) + eps
    t = np.sum((p - a) * ab, axis=-1) / denom
    t = np.clip(t, 0.0, 1.0)
    q = a + t[..., None] * ab
    d = p - q
    return np.sum(d * d, axis=-1)


def _min_dist2_points_to_segments(
    points: np.ndarray, a: np.ndarray, b: np.ndarray, chunk: int = 1024
) -> np.ndarray:
    min_d2 = np.full((points.shape[0],), np.inf, dtype=np.float32)
    for i in range(0, points.shape[0], chunk):
        p = points[i : i + chunk]
        p_exp = p[:, None, :]
        a_exp = a[None, :, :]
        b_exp = b[None, :, :]
        d2 = _point_segment_dist2(p_exp, a_exp, b_exp)
        min_d2[i : i + chunk] = np.min(d2, axis=1)
    return min_d2


def _draw_raster(
    out_path: Path,
    mask: np.ndarray,
    origin: np.ndarray,
    h: float,
    Y: np.ndarray,
    full_mask: np.ndarray,
    max_size: int,
) -> None:
    H, W = mask.shape
    scale = min(1.0, float(max_size) / float(max(H, W)))
    out_w = max(1, int(round(W * scale)))
    out_h = max(1, int(round(H * scale)))

    img = np.full((out_h, out_w, 3), 255, dtype=np.uint8)

    border = _border_from_mask(mask)
    by, bx = np.nonzero(border)
    bx_s = np.clip((bx * scale).astype(int), 0, out_w - 1)
    by_s = np.clip((by * scale).astype(int), 0, out_h - 1)

    # Map Y points to pixel coords
    px = (Y[:, 0] - origin[0]) / h
    py = (origin[1] - Y[:, 1]) / h
    px_s = np.clip((px * scale).astype(int), 0, out_w - 1)
    py_s = np.clip((py * scale).astype(int), 0, out_h - 1)

    full_px = px_s[full_mask]
    full_py = py_s[full_mask]

    img[full_py, full_px] = np.array([34, 139, 34], dtype=np.uint8)
    img[by_s, bx_s] = np.array([0, 0, 0], dtype=np.uint8)

    Image.fromarray(img).save(out_path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, help="Checkpoint run directory")
    ap.add_argument("--step", type=int, default=None, help="Checkpoint step")
    ap.add_argument("--out", default=None, help="Output PNG path")
    ap.add_argument(
        "--max-size",
        type=int,
        default=512,
        help="Max width/height for raster output",
    )
    ap.add_argument(
        "--use-clearance",
        action="store_true",
        help="Use r_fill + 0.5*delta for fullness",
    )
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise ValueError(f"Run dir does not exist: {run_dir}")

    step = args.step if args.step is not None else _find_latest_step(run_dir)
    checkpoint_path = run_dir / f"step_{step:06d}.npz"
    if not checkpoint_path.exists():
        raise ValueError(f"Missing checkpoint: {checkpoint_path}")

    metadata = _load_metadata(run_dir)
    shapes = metadata.get("shapes", {})
    B_shape = shapes.get("B") if isinstance(shapes, dict) else None
    Y_shape = shapes.get("Y") if isinstance(shapes, dict) else None

    if isinstance(B_shape, list) and len(B_shape) == 2:
        M = int(B_shape[0])
        n_ctrl = int(B_shape[1])
    else:
        raise ValueError("metadata.json missing shapes.B")

    if isinstance(Y_shape, list) and len(Y_shape) >= 1:
        Q = int(Y_shape[0])
    else:
        raise ValueError("metadata.json missing shapes.Y")

    seed = int(metadata.get("seed", 0))
    r_fill = float(metadata.get("r_fill", 0.0))
    delta = float(metadata.get("delta", 0.0))
    if args.use_clearance:
        r_fill = r_fill + 0.5 * delta

    mask, _sdf, origin, h = _load_rasters(run_dir)
    rng = np.random.default_rng(seed)
    Y = sample_interior_points(mask, origin, h, Q=Q, rng=rng)

    ckpt = np.load(checkpoint_path)
    P = ckpt["P"]
    r = ckpt["r"]
    if r_fill <= 0:
        r_fill = float(np.mean(r))

    B = make_basis_matrix(n_ctrl=n_ctrl, n_samples=M, degree=3)
    X = np.einsum("mn,cnd->cmd", B, P)

    A = X[:, :-1, :].reshape(-1, 2)
    B_seg = X[:, 1:, :].reshape(-1, 2)
    d2_min = _min_dist2_points_to_segments(Y, A, B_seg)
    full_mask = d2_min <= (r_fill * r_fill)

    out_path = (
        Path(args.out) if args.out is not None else run_dir / f"raster_{step:06d}.png"
    )
    _draw_raster(out_path, mask, origin, h, Y, full_mask, max_size=args.max_size)


if __name__ == "__main__":
    main()
