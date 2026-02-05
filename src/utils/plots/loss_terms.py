from __future__ import annotations

from pathlib import Path

import numpy as np


def plot_loss_terms(
    out_path: Path,
    steps: np.ndarray,
    loss: np.ndarray,
    loss_inside: np.ndarray,
    loss_curv: np.ndarray,
    loss_sep: np.ndarray,
    loss_fill: np.ndarray,
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.5, 5.0), dpi=120)
    ax.plot(steps, loss, label="loss", linewidth=2.0)
    ax.plot(steps, loss_inside, label="loss_inside")
    ax.plot(steps, loss_curv, label="loss_curv")
    ax.plot(steps, loss_sep, label="loss_sep")
    ax.plot(steps, loss_fill, label="loss_fill")
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
