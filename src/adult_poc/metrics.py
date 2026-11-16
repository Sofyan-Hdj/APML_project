"""Evaluation metrics shared by every model and every step."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score, brier_score_loss,
    f1_score, precision_recall_curve, precision_score, recall_score, roc_auc_score,
)

# Scorers used inside cross-validation (threshold-free + default-threshold metrics).
CV_SCORING = {
    "pr_auc": "average_precision",
    "roc_auc": "roc_auc",
    "f1": "f1",
    "balanced_acc": "balanced_accuracy",
    "brier": "neg_brier_score",
}


def best_f1_threshold(y_true, proba) -> float:
    """Threshold maximising F1 on the given (validation) data."""
    p, r, t = precision_recall_curve(y_true, proba)
    f1 = 2 * p * r / np.clip(p + r, 1e-12, None)
    return float(t[np.argmax(f1[:-1])])


def expected_calibration_error(y_true, proba, n_bins: int = 10) -> float:
    """Equal-width ECE: weighted mean |accuracy - confidence| over probability bins."""
    y_true, proba = np.asarray(y_true), np.asarray(proba)
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(proba, bins[1:-1]), 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            ece += m.mean() * abs(y_true[m].mean() - proba[m].mean())
    return float(ece)


def evaluate(y_true, proba, threshold: float = 0.5) -> dict:
    pred = (np.asarray(proba) >= threshold).astype(int)
    return {
        "pr_auc": average_precision_score(y_true, proba),
        "roc_auc": roc_auc_score(y_true, proba),
        "f1": f1_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred),
        "balanced_acc": balanced_accuracy_score(y_true, pred),
        "accuracy": accuracy_score(y_true, pred),
        "brier": brier_score_loss(y_true, proba),
        "ece": expected_calibration_error(y_true, proba),
        "threshold": threshold,
    }
