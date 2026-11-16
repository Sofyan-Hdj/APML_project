"""Shared matplotlib style helpers."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

MODEL_COLORS = {"logreg": "#9aa5b1", "random_forest": "#e9a03b", "lightgbm": "#2a6f97"}


def setup_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 160,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "font.size": 9,
    })


def savefig(fig, path: Path) -> Path:
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path
