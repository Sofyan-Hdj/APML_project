"""Unit tests for the decision-threshold selection used in the imbalance and tuning sections."""
import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from adult_poc.metrics import best_f1_threshold  # noqa: E402


def _brute_force_best_f1(y, proba):
    """Highest F1 over every observed score used as a threshold (reference implementation)."""
    return max(f1_score(y, (proba >= c).astype(int), zero_division=0) for c in np.unique(proba))


def test_best_f1_threshold_on_a_hand_checked_example():
    # Thresholds 0.4, 0.6, 0.9 and 0.1 give F1 = 0.8, 0.5, 0.667 and 0.667: the best is 0.4.
    y = np.array([0, 0, 1, 1])
    proba = np.array([0.1, 0.6, 0.4, 0.9])
    assert best_f1_threshold(y, proba) == pytest.approx(0.4)


def test_best_f1_threshold_reaches_the_maximum_f1():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=400)
    proba = np.clip(0.35 * y + rng.uniform(0, 0.7, size=400), 0, 1)
    thr = best_f1_threshold(y, proba)
    achieved = f1_score(y, (proba >= thr).astype(int), zero_division=0)
    assert achieved == pytest.approx(_brute_force_best_f1(y, proba))
