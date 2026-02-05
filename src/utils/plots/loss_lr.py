from __future__ import annotations

from pathlib import Path

import numpy as np


def plot_loss_lr(
    out_path: Path,
    steps: np.ndarray,
    loss: np.ndarray,
    lr: np.ndarray,
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.5, 5.0), dpi=120)
    ax.plot(steps, loss, label="loss", color="#1f77b4", linewidth=2.0)
    ax.set_xlabel("step")
    ax.set_ylabel("loss", color="#1f77b4")
    ax.tick_params(axis="y", labelcolor="#1f77b4")
    ax.grid(True, alpha=0.3)

    ax_lr = ax.twinx()
    ax_lr.plot(steps, lr, label="lr", color="#d62728", linewidth=2.0)
    ax_lr.set_ylabel("learning rate", color="#d62728")
    ax_lr.tick_params(axis="y", labelcolor="#d62728")

    lines = ax.get_lines() + ax_lr.get_lines()
    labels = [str(line.get_label()) for line in lines]
    ax.legend(lines, labels, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
