import logging, platform, sys
import numpy as np
sys.path.insert(0, sys.argv[1])
from test_gravity_packing import _spiral
from autorigami.packing import gravity as g
logging.basicConfig(level=logging.INFO, format="%(message)s")
print(platform.machine(), platform.platform(), flush=True)
for args in ((16.0, 16.0, 5.2), (10.0, 22.0, 4.0)):
    spiral = _spiral(*args, turns=6.0)
    points = g._repair_horizontal_kinks(spiral)
    pinned = np.zeros(len(points), dtype=np.bool_)
    pinned[: g._first_turn_vertex_count(points)] = True
    print(args, "n", len(points), "pinned", int(pinned.sum()), "nan in", int(np.isnan(points).sum()), flush=True)
    out = g._settle(points, pinned, g._GRAVITY * (28.0 / 0.34) ** 2)
    edges = np.linalg.norm(np.diff(out, axis=0), axis=1)
    print("nan out", int(np.isnan(out).sum()), "zero edges", int((edges == 0).sum()), "min edge", float(np.nanmin(edges)),
          "xy drift", float(np.nanmax(np.abs(out[:, :2] - points[:, :2]))), "z range", float(np.nanmin(out[:, 2])), float(np.nanmax(out[:, 2])), flush=True)
    bad = np.flatnonzero(~(edges > 0))
    print("bad edge indices", bad[:20], "of", len(edges), flush=True)
    if len(bad):
        i = int(bad[0]); print(points[i - 1 : i + 3], out[i - 1 : i + 3], flush=True)
