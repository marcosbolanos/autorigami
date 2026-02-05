from __future__ import annotations

from pathlib import Path

import numpy as np


def plot_separation_pairs(
    out_path: Path,
    steps: np.ndarray,
    loss_sep: np.ndarray,
    pairs: np.ndarray,
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.5, 5.0), dpi=120)
    ax.plot(steps, loss_sep, label="loss_sep", color="#1f77b4", linewidth=2.0)
    ax.set_xlabel("step")
    ax.set_ylabel("loss_sep", color="#1f77b4")
    ax.tick_params(axis="y", labelcolor="#1f77b4")
    ax.grid(True, alpha=0.3)

    ax_pairs = ax.twinx()
    ax_pairs.plot(steps, pairs, label="pairs", color="#9467bd", linewidth=2.0)
    ax_pairs.set_ylabel("pairs", color="#9467bd")
    ax_pairs.tick_params(axis="y", labelcolor="#9467bd")

    lines = ax.get_lines() + ax_pairs.get_lines()
    labels = [str(line.get_label()) for line in lines]
    ax.legend(lines, labels, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
