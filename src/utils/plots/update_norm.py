from pathlib import Path

import numpy as np


def plot_update_norm(
    out_path: Path,
    steps: np.ndarray,
    update_norm: np.ndarray,
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.5, 5.0), dpi=120)
    ax.plot(steps, update_norm, label="update_norm", color="#17becf", linewidth=2.0)
    ax.set_xlabel("step")
    ax.set_ylabel("update_norm")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
