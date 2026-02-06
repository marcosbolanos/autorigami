from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
import numpy as np

from .plots import (
    plot_fill_schedule,
    plot_fill_hard,
    plot_fill_soft,
    plot_gradient_norms,
    plot_loss_lr,
    plot_loss_terms,
    plot_run_summary,
    plot_separation_pairs,
    plot_update_norm,
)

REQUIRED_FIELDS = [
    "step",
    "loss",
    "loss_inside",
    "loss_curv",
    "loss_sep",
    "loss_fill",
    "pairs",
    "lr",
]
OPTIONAL_FIELDS = [
    "fill_soft",
    "tau_fill",
    "w_fill",
    "update_norm",
]

GRAD_REQUIRED_FIELDS = [
    "step",
    "grad_inside_mean",
    "grad_inside_max",
    "grad_curv_mean",
    "grad_curv_max",
    "grad_sep_mean",
    "grad_sep_max",
    "grad_fill_mean",
    "grad_fill_max",
]

GRAD_OPTIONAL_FIELDS = [
    "fill_soft",
    "fill_hard",
    "fill_hard_clearance",
]


def read_metrics(csv_path: Path) -> dict[str, np.ndarray]:
    with csv_path.open(newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames is None:
            raise ValueError("metrics.csv is missing a header row")
        missing = [field for field in REQUIRED_FIELDS if field not in reader.fieldnames]
        if missing:
            raise ValueError(f"metrics.csv missing columns: {', '.join(missing)}")
        optional = [field for field in OPTIONAL_FIELDS if field in reader.fieldnames]
        rows = list(reader)

    if not rows:
        raise ValueError("metrics.csv has no data rows")

    steps = np.array([int(row["step"]) for row in rows], dtype=np.int32)
    order = np.argsort(steps)

    def col_float(name: str) -> np.ndarray:
        return np.array([float(row[name]) for row in rows], dtype=np.float32)[order]

    data = {
        "step": steps[order],
        "loss": col_float("loss"),
        "loss_inside": col_float("loss_inside"),
        "loss_curv": col_float("loss_curv"),
        "loss_sep": col_float("loss_sep"),
        "loss_fill": col_float("loss_fill"),
        "pairs": col_float("pairs"),
        "lr": col_float("lr"),
    }
    for name in optional:
        data[name] = col_float(name)
    return data


def read_gradients(csv_path: Path) -> dict[str, np.ndarray]:
    with csv_path.open(newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames is None:
            raise ValueError("gradients_debug.csv is missing a header row")
        missing = [
            field for field in GRAD_REQUIRED_FIELDS if field not in reader.fieldnames
        ]
        if missing:
            raise ValueError(
                f"gradients_debug.csv missing columns: {', '.join(missing)}"
            )
        optional = [
            field for field in GRAD_OPTIONAL_FIELDS if field in reader.fieldnames
        ]
        rows = list(reader)

    if not rows:
        raise ValueError("gradients_debug.csv has no data rows")

    steps = np.array([int(row["step"]) for row in rows], dtype=np.int32)
    order = np.argsort(steps)

    def col_float(name: str) -> np.ndarray:
        return np.array([float(row[name]) for row in rows], dtype=np.float32)[order]

    data = {"step": steps[order]}
    for name in GRAD_REQUIRED_FIELDS[1:]:
        data[name] = col_float(name)
    for name in optional:
        data[name] = col_float(name)
    return data


def plot_metrics(
    csv_path: Path,
    out_dir: Path,
    prefix: str,
    show: bool,
    start_step: int | None,
) -> None:
    if not show:
        matplotlib.use("Agg")
    data = read_metrics(csv_path)
    gradients_path = csv_path.with_name("gradients_debug.csv")
    gradients_data: dict[str, np.ndarray] | None = None
    if gradients_path.exists():
        gradients_data = read_gradients(gradients_path)
    base_out_dir = out_dir
    base_out_dir.mkdir(parents=True, exist_ok=True)

    steps = data["step"]
    loss = data["loss"]
    loss_inside = data["loss_inside"]
    loss_curv = data["loss_curv"]
    loss_sep = data["loss_sep"]
    loss_fill = data["loss_fill"]
    pairs = data["pairs"]
    lr = data["lr"]
    update_norm = data.get("update_norm")

    suffix = ""
    mask: np.ndarray | None = None
    if start_step is not None:
        mask = steps >= start_step
        if not np.any(mask):
            raise ValueError(f"No rows with step >= {start_step}")
        steps = steps[mask]
        loss = loss[mask]
        loss_inside = loss_inside[mask]
        loss_curv = loss_curv[mask]
        loss_sep = loss_sep[mask]
        loss_fill = loss_fill[mask]
        pairs = pairs[mask]
        lr = lr[mask]
        if update_norm is not None:
            update_norm = update_norm[mask]
        suffix = f"_from_step{start_step}"

        if gradients_data is not None:
            g_steps = gradients_data["step"]
            g_mask = g_steps >= start_step
            if np.any(g_mask):
                gradients_data = {
                    key: val[g_mask] for key, val in gradients_data.items()
                }
            else:
                gradients_data = None

    start_label = int(steps[0])
    end_label = int(steps[-1])
    plot_dir = base_out_dir / f"start_{start_label}_end_{end_label}"
    plot_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = csv_path.with_name("metadata.json")
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata_text = json.dumps(metadata, indent=2, sort_keys=True)
        summary_lines = [
            f"run_dir: {csv_path.parent}",
            f"start_step: {start_label}",
            f"end_step: {end_label}",
            "",
            "metadata.json:",
            *metadata_text.splitlines(),
        ]
        plot_run_summary(
            plot_dir / f"{prefix}{suffix}_1_run_summary.png",
            "1 Run Summary",
            summary_lines,
        )

    plot_loss_terms(
        plot_dir / f"{prefix}{suffix}_loss_terms.png",
        steps,
        loss,
        loss_inside,
        loss_curv,
        loss_sep,
        loss_fill,
    )
    plot_loss_lr(plot_dir / f"{prefix}{suffix}_loss_lr.png", steps, loss, lr)
    if update_norm is not None:
        plot_update_norm(
            plot_dir / f"{prefix}{suffix}_update_norm.png",
            steps,
            update_norm,
        )

    if "tau_fill" not in data or "w_fill" not in data:
        raise ValueError("metrics.csv missing fill schedule columns (tau_fill, w_fill)")

    tau_fill = data["tau_fill"]
    w_fill = data["w_fill"]
    fill_soft = data.get("fill_soft")
    if mask is not None:
        tau_fill = tau_fill[mask]
        w_fill = w_fill[mask]
        if fill_soft is not None:
            fill_soft = fill_soft[mask]

    plot_fill_schedule(
        plot_dir / f"{prefix}{suffix}_fill_schedule.png",
        steps,
        tau_fill,
        w_fill,
    )
    if fill_soft is not None:
        plot_fill_soft(
            plot_dir / f"{prefix}{suffix}_fill_soft.png",
            steps,
            fill_soft,
        )
    plot_separation_pairs(
        plot_dir / f"{prefix}{suffix}_separation_pairs.png",
        steps,
        loss_sep,
        pairs,
    )

    if gradients_data is not None:
        plot_gradient_norms(
            plot_dir / f"{prefix}{suffix}_gradient_norms.png",
            gradients_data["step"],
            gradients_data["grad_inside_mean"],
            gradients_data["grad_inside_max"],
            gradients_data["grad_curv_mean"],
            gradients_data["grad_curv_max"],
            gradients_data["grad_sep_mean"],
            gradients_data["grad_sep_max"],
            gradients_data["grad_fill_mean"],
            gradients_data["grad_fill_max"],
        )

        if "fill_hard" in gradients_data:
            plot_fill_hard(
                plot_dir / f"{prefix}{suffix}_fill_hard.png",
                gradients_data["step"],
                gradients_data["fill_hard"],
                gradients_data.get("fill_hard_clearance"),
            )

    if show:
        import matplotlib.pyplot as plt

        plt.show()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to metrics.csv")
    ap.add_argument(
        "--out-dir",
        default=None,
        help="Output directory for plots (defaults to CSV directory)",
    )
    ap.add_argument(
        "--prefix",
        default=None,
        help="Output filename prefix (defaults to CSV stem)",
    )
    ap.add_argument("--show", action="store_true", help="Show plots interactively")
    ap.add_argument(
        "--start-step",
        type=int,
        default=None,
        help="Only plot rows with step >= this value",
    )
    args = ap.parse_args()

    csv_path = Path(args.input)
    out_dir = Path(args.out_dir) if args.out_dir is not None else csv_path.parent
    prefix = args.prefix if args.prefix is not None else csv_path.stem

    plot_metrics(csv_path, out_dir, prefix, args.show, args.start_step)


if __name__ == "__main__":
    main()
