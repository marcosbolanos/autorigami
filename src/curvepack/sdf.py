from __future__ import annotations

from typing import cast

import numpy as np
from beartype import beartype
from jaxtyping import Bool, Float, jaxtyped
from shapely.geometry import Polygon
from PIL import Image, ImageDraw
from scipy.ndimage import distance_transform_edt  # type: ignore[reportMissingTypeStubs]


@jaxtyped(typechecker=beartype)
def polygon_to_mask(
    V: Float[np.ndarray, "N 2"],
    h: float,
    pad: float = 10.0,
) -> tuple[Bool[np.ndarray, "H W"], Float[np.ndarray, "2"], float]:
    """
    V: (N,2) polygon vertices in world coords (closed)
    h: pixel size in world units
    Returns:
      mask: (H,W) bool, True inside polygon
      origin: (2,) world coord of pixel (0,0) center
      h: pixel size
    """
    poly = Polygon(V)
    if not poly.is_valid:
        poly = poly.buffer(0)  # attempt to fix self-intersections
    if poly.is_empty:
        raise ValueError("Polygon is empty after fixing.")

    minx, miny, maxx, maxy = poly.bounds
    minx -= pad
    miny -= pad
    maxx += pad
    maxy += pad

    W = int(np.ceil((maxx - minx) / h))
    H = int(np.ceil((maxy - miny) / h))

    # Raster coordinates: image y increases downward, world y increases upward
    # We'll map world->pixel: px = (x-minx)/h, py = (maxy-y)/h
    img = Image.new("L", (W, H), 0)
    draw = ImageDraw.Draw(img)

    # Convert polygon vertices to pixel coords
    pts: list[tuple[float, float]] = []
    for x, y in V:
        px = (x - minx) / h
        py = (maxy - y) / h
        pts.append((px, py))
    draw.polygon(pts, outline=1, fill=1)

    mask = np.array(img, dtype=np.uint8) > 0

    origin = np.array(
        [minx, maxy], dtype=np.float32
    )  # world coord of pixel top-left corner in x, and top in y
    return mask, origin, h


@jaxtyped(typechecker=beartype)
def mask_to_sdf(mask: Bool[np.ndarray, "H W"], h: float) -> Float[np.ndarray, "H W"]:
    """
    Signed distance in world units, positive inside.
    """
    inside = mask.astype(np.uint8)
    outside = (~mask).astype(np.uint8)

    d_in = cast(np.ndarray, distance_transform_edt(inside)) * h
    d_out = cast(np.ndarray, distance_transform_edt(outside)) * h
    sdf = d_in - d_out
    return sdf.astype(np.float32)


@jaxtyped(typechecker=beartype)
def sample_interior_points(
    mask: Bool[np.ndarray, "H W"],
    origin: Float[np.ndarray, "2"],
    h: float,
    Q: int,
    rng: np.random.Generator,
) -> Float[np.ndarray, "Q 2"]:
    """
    Uniform-ish sampling inside mask by rejection on pixels then jitter within pixel.
    Returns Y: (Q,2) world coords.
    """
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        raise ValueError("Mask has no inside pixels.")
    idx = rng.integers(0, len(xs), size=Q)
    px = xs[idx].astype(np.float32)
    py = ys[idx].astype(np.float32)

    # Jitter inside pixel
    jx = rng.random(Q).astype(np.float32)
    jy = rng.random(Q).astype(np.float32)

    # Convert pixel center-ish to world:
    # world x = origin_x + (px + jx)*h
    # world y = origin_y - (py + jy)*h   (since origin_y = maxy)
    xw = origin[0] + (px + jx) * h
    yw = origin[1] - (py + jy) * h
    Y = np.stack([xw, yw], axis=1)
    return Y.astype(np.float32)
