from __future__ import annotations

import numpy as np

from . import debug

_seen: set[str] = set()


def log_once(key: str, message: str) -> None:
    if debug.is_verbose() and key not in _seen:
        _seen.add(key)
        debug.log(message)


def log_array(name: str, arr: np.ndarray) -> None:
    if not debug.is_verbose():
        return
    if arr.size == 0:
        finite_any = False
        finite_all = True
        min_val = float("nan")
        max_val = float("nan")
    else:
        finite_mask = np.isfinite(arr)
        finite_any = bool(finite_mask.any())
        finite_all = bool(finite_mask.all())
        if finite_any:
            finite_vals = arr[finite_mask]
            min_val = float(np.min(finite_vals))
            max_val = float(np.max(finite_vals))
        else:
            min_val = float("nan")
            max_val = float("nan")
    debug.log(
        f"{name}: shape={arr.shape} dtype={arr.dtype} "
        f"finite_all={finite_all} min={min_val:.6g} max={max_val:.6g}"
    )


def log_array_once(key: str, name: str, arr: np.ndarray) -> None:
    if debug.is_verbose() and key not in _seen:
        _seen.add(key)
        log_array(name, arr)
