from __future__ import annotations

from pathlib import Path

import numpy as np


def plot_gradient_norms(
    out_path: Path,
    steps: np.ndarray,
    grad_inside_mean: np.ndarray,
    grad_inside_max: np.ndarray,
    grad_curv_mean: np.ndarray,
    grad_curv_max: np.ndarray,
    grad_sep_mean: np.ndarray,
    grad_sep_max: np.ndarray,
    grad_fill_mean: np.ndarray,
    grad_fill_max: np.ndarray,
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.5, 5.0), dpi=120)
    ax.plot(steps, grad_inside_mean, label="inside_mean", color="#d62728")
    ax.plot(steps, grad_inside_max, label="inside_max", color="#d62728", linestyle="--")
    ax.plot(steps, grad_curv_mean, label="curv_mean", color="#ff7f0e")
    ax.plot(steps, grad_curv_max, label="curv_max", color="#ff7f0e", linestyle="--")
    ax.plot(steps, grad_sep_mean, label="sep_mean", color="#1f77b4")
    ax.plot(steps, grad_sep_max, label="sep_max", color="#1f77b4", linestyle="--")
    ax.plot(steps, grad_fill_mean, label="fill_mean", color="#2ca02c")
    ax.plot(steps, grad_fill_max, label="fill_max", color="#2ca02c", linestyle="--")
    ax.set_xlabel("step")
    ax.set_ylabel("grad_norm")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
