from __future__ import annotations

from pathlib import Path

import numpy as np


def plot_fill_schedule(
    out_path: Path,
    steps: np.ndarray,
    tau_fill: np.ndarray,
    w_fill: np.ndarray,
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.5, 5.0), dpi=120)
    ax.plot(steps, tau_fill, label="tau_fill", color="#2ca02c", linewidth=2.0)
    ax.set_xlabel("step")
    ax.set_ylabel("tau_fill", color="#2ca02c")
    ax.tick_params(axis="y", labelcolor="#2ca02c")
    ax.grid(True, alpha=0.3)

    ax_w = ax.twinx()
    ax_w.plot(steps, w_fill, label="w_fill", color="#ff7f0e", linewidth=2.0)
    ax_w.set_ylabel("w_fill", color="#ff7f0e")
    ax_w.tick_params(axis="y", labelcolor="#ff7f0e")

    lines = ax.get_lines() + ax_w.get_lines()
    labels = [str(line.get_label()) for line in lines]
    ax.legend(lines, labels, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
