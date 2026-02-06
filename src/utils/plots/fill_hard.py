from __future__ import annotations

from pathlib import Path

import numpy as np


def plot_fill_hard(
    out_path: Path,
    steps: np.ndarray,
    fill_hard: np.ndarray,
    fill_hard_clearance: np.ndarray | None = None,
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.5, 5.0), dpi=120)
    ax.plot(steps, fill_hard, label="fill_hard", color="#2ca02c", linewidth=2.0)
    if fill_hard_clearance is not None:
        ax.plot(
            steps,
            fill_hard_clearance,
            label="fill_hard_clearance",
            color="#1f77b4",
            linewidth=2.0,
        )
    ax.set_xlabel("step")
    ax.set_ylabel("fullness")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
