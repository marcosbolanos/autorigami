from __future__ import annotations

from pathlib import Path


def plot_run_summary(out_path: Path, title: str, lines: list[str]) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.5, 11.0), dpi=120)
    ax.axis("off")
    fig.suptitle(title, fontsize=12, y=0.98)

    text = "\n".join(lines)
    ax.text(
        0.01,
        0.98,
        text,
        va="top",
        ha="left",
        family="monospace",
        fontsize=8,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
