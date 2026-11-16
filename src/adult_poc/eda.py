"""Exploratory data analysis: summary tables and figures used in the report."""
from __future__ import annotations

import json
import logging

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .plotting import savefig, setup_style

log = logging.getLogger(__name__)


def run_eda(X: pd.DataFrame, y: pd.Series, cfg: dict) -> dict:
    setup_style()
    plots, tables = cfg["paths"]["plots"], cfg["paths"]["tables"]
    sens = cfg["data"]["sensitive_attribute"]
    df = X.assign(target=y.values)

    summary = {
        "n_rows": int(len(df)),
        "n_features": int(X.shape[1]),
        "positive_rate": float(y.mean()),
        "missing_by_column": {k: int(v) for k, v in X.isna().sum().items() if v > 0},
        "rows_with_missing": int(X.isna().any(axis=1).sum()),
        "positive_rate_by_sex": df.groupby(sens, observed=True)["target"].mean().round(4).to_dict(),
        "share_by_sex": df[sens].value_counts(normalize=True).round(4).to_dict(),
        "positive_rate_by_race": df.groupby("race", observed=True)["target"].mean().round(4).to_dict(),
        "capital_gain_zero_share": float((df["capital-gain"] == 0).mean()),
        "capital_loss_zero_share": float((df["capital-loss"] == 0).mean()),
    }
    with open(tables / "eda_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    X.describe().T.round(2).to_csv(tables / "eda_numeric_describe.csv")

    # 1. Class balance
    fig, ax = plt.subplots(figsize=(4, 3))
    counts = y.value_counts().sort_index()
    ax.bar(["<=50K", ">50K"], counts.values, color=["#9aa5b1", "#2a6f97"])
    for i, v in enumerate(counts.values):
        ax.text(i, v, f"{v:,}\n({v / len(y):.1%})", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("Count")
    ax.set_title("Target distribution")
    ax.set_ylim(0, counts.max() * 1.2)
    savefig(fig, plots / "eda_target.png")

    # 2. Positive rate by sensitive attribute and by key categoricals
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
    for ax, col in zip(axes, [sens, "relationship", "education-num"]):
        rate = df.groupby(col, observed=True)["target"].mean().sort_values()
        ax.barh(rate.index.astype(str), rate.values, color="#2a6f97")
        ax.set_title(f"P(>50K) by {col}")
        ax.set_xlim(0, 1)
    savefig(fig, plots / "eda_rate_by_group.png")

    # 3. Numeric distributions by class
    fig, axes = plt.subplots(1, 3, figsize=(11, 3))
    for ax, col in zip(axes, ["age", "hours-per-week", "education-num"]):
        for cls, color in [(0, "#9aa5b1"), (1, "#2a6f97")]:
            ax.hist(df.loc[df.target == cls, col], bins=30, alpha=0.6, density=True,
                    color=color, label=">50K" if cls else "<=50K")
        ax.set_title(col)
    axes[0].legend()
    savefig(fig, plots / "eda_numeric_by_class.png")

    # 4. Missingness
    miss = X.isna().mean().loc[lambda s: s > 0].sort_values()
    fig, ax = plt.subplots(figsize=(4.5, 2.2))
    ax.barh(miss.index, miss.values * 100, color="#c8553d")
    ax.set_xlabel("% missing")
    ax.set_title("Missing values")
    savefig(fig, plots / "eda_missing.png")

    log.info("EDA summary: %s", summary)
    return summary
