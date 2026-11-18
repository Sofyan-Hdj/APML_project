"""Regression checks for data ordering and rebuilding a tuned configuration."""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import optuna
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from adult_poc import data  # noqa: E402
from adult_poc.models import build_model, suggest_params  # noqa: E402


def test_openml_and_local_have_identical_data_and_splits(monkeypatch, tmp_path):
    from test_pipeline import ROW

    raw = pd.DataFrame([{**ROW, "age": 20 + i % 60, "fnlwgt": i} for i in range(100)])
    labels = pd.Series(np.where(np.arange(100) % 3 == 0, ">50K", "<=50K"))
    order = list(range(60, 100)) + list(range(60))
    monkeypatch.setattr(data, "UCI_TRAIN_ROWS", 60)
    monkeypatch.setattr(data, "UCI_TEST_ROWS", 40)
    monkeypatch.setattr("sklearn.datasets.fetch_openml", lambda **kw: SimpleNamespace(
        data=raw.iloc[order].reset_index(drop=True), target=labels.iloc[order].reset_index(drop=True)))
    path = tmp_path / "adult.csv"
    raw.assign(**{"class": labels}).to_csv(path, columns=data.COLUMNS, index=False, header=False)
    cfg = {"seed": 42, "data": {"openml_name": "adult", "openml_version": 2, "local_csv": path,
                               "positive_label": ">50K", "test_size": 0.2, "val_size": 0.2}}
    monkeypatch.delenv("ADULT_OFFLINE", raising=False)
    xo, yo = data.load_raw(cfg)
    cfg["data"]["sha256"] = data.data_fingerprint(xo, yo)
    monkeypatch.setenv("ADULT_OFFLINE", "1")
    xl, yl = data.load_raw(cfg)
    assert data.data_fingerprint(xl, yl) == data.data_fingerprint(xo, yo)
    for left, right in zip(data.split(xo, yo, cfg), data.split(xl, yl, cfg)):
        assert left.index.tolist() == right.index.tolist()
    raw.assign(**{"class": labels}).iloc[::-1].to_csv(path, columns=data.COLUMNS, index=False, header=False)
    with pytest.raises(ValueError, match="data/order differs"):
        data.load_raw(cfg)


def test_rebuilt_lightgbm_keeps_fixed_tuning_parameters():
    trial = optuna.trial.FixedTrial({"n_estimators": 100, "learning_rate": 0.05, "num_leaves": 16,
                                    "min_child_samples": 20, "subsample": 0.7,
                                    "colsample_bytree": 0.8, "reg_lambda": 1.0})
    during_cv = build_model("lightgbm", suggest_params("lightgbm", trial), 42)
    rebuilt = build_model("lightgbm", trial.params, 42)
    assert during_cv.get_params() == rebuilt.get_params()
    assert rebuilt.subsample_freq == 1


def test_logistic_regression_uses_l1_ratio():
    """The pinned sklearn must actually use the parameter the search varies."""
    from sklearn.datasets import make_classification

    X, y = make_classification(n_samples=100, n_features=6, random_state=42)
    a = build_model("logreg", {"C": 0.1, "l1_ratio": 0.0}, 42).fit(X, y)
    b = build_model("logreg", {"C": 0.1, "l1_ratio": 1.0}, 42).fit(X, y)
    assert not np.allclose(a.coef_, b.coef_)
