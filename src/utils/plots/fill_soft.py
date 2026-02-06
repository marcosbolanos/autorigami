from __future__ import annotations

from pathlib import Path

import numpy as np


def plot_fill_soft(
    out_path: Path,
    steps: np.ndarray,
    fill_soft: np.ndarray,
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.5, 5.0), dpi=120)
    ax.plot(steps, fill_soft, label="fill_soft", color="#2ca02c", linewidth=2.0)
    ax.set_xlabel("step")
    ax.set_ylabel("fill_soft")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
