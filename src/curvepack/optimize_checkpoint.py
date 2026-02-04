from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .export_svg import export_curves_svg

CHECKPOINT_CSV_FIELDS = [
    "step",
    "loss",
    "loss_inside",
    "loss_curv",
    "loss_sep",
    "loss_fill",
    "pairs",
    "lr",
    "weight_decay",
    "elapsed_s",
    "step_s",
]


@dataclass(frozen=True)
class CheckpointRun:
    run_dir: Path
    csv_path: Path


def init_checkpoint_run(base_dir: Path, metadata: dict[str, Any]) -> CheckpointRun:
    base_dir.mkdir(parents=True, exist_ok=True)
    now = time.localtime()
    timestamp = time.strftime("%Y%m%d_%H%M%S", now)
    nonce = time.time_ns() % 1_000_000_000
    run_id = f"run_{timestamp}_{nonce:09d}"
    run_dir = base_dir / run_id
    while run_dir.exists():
        nonce = (nonce + 1) % 1_000_000_000
        run_id = f"run_{timestamp}_{nonce:09d}"
        run_dir = base_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    metadata_out = dict(metadata)
    metadata_out["run_id"] = run_id
    metadata_out["created_at"] = time.strftime("%Y-%m-%dT%H:%M:%S", now)
    metadata_path = run_dir / "metadata.json"
    metadata_path.write_text(
        json.dumps(metadata_out, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"checkpoint run dir={run_dir}")

    return CheckpointRun(run_dir=run_dir, csv_path=run_dir / "metrics.csv")


def save_checkpoint_npz(
    run_dir: Path,
    step_idx: int,
    P: np.ndarray,
    r: np.ndarray,
    loss: float,
) -> Path:
    checkpoint_path = run_dir / f"step_{step_idx:06d}.npz"
    np.savez_compressed(
        checkpoint_path,
        step=step_idx,
        loss=loss,
        P=P,
        r=r,
    )
    return checkpoint_path


def append_metrics_csv(
    csv_path: Path,
    fieldnames: list[str],
    row: dict[str, Any],
) -> None:
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def save_checkpoint_svg(run_dir: Path, step_idx: int, curves: np.ndarray) -> None:
    svg_path = run_dir / f"step_{step_idx:06d}.svg"
    start_svg = time.perf_counter()
    export_curves_svg(str(svg_path), curves)
    svg_elapsed = time.perf_counter() - start_svg
    print(
        f"checkpoint svg saved step={step_idx} path={svg_path} time={svg_elapsed:.3f}s"
    )
