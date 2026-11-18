"""Unit tests for the feature engineering and the serving endpoint."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from adult_poc.features import FeatureEngineer  # noqa: E402
from adult_poc.metrics import expected_calibration_error  # noqa: E402

ROW = {"age": 45, "workclass": "Private", "fnlwgt": 1.0, "education": "Bachelors", "education-num": 13,
       "marital-status": "Married-civ-spouse", "occupation": "Exec-managerial", "relationship": "Husband",
       "race": "White", "sex": "Male", "capital-gain": 5000, "capital-loss": 0, "hours-per-week": 50,
       "native-country": "United-States"}


def test_feature_engineer_builds_expected_columns():
    df = pd.DataFrame([ROW, {**ROW, "native-country": "Atlantis", "capital-gain": 0, "capital-loss": 1000,
                             "marital-status": "Divorced", "hours-per-week": 30}])
    fe = FeatureEngineer(rare_min_count=1).fit(df.iloc[[0]])
    out = fe.transform(df)
    assert "fnlwgt" not in out and "education" not in out
    assert out["is_married"].tolist() == [1, 0]
    assert out["overtime"].tolist() == [1, 0]
    assert out["native-country"].tolist() == ["United-States", "Other"]
    assert out["capital_net_log"].iloc[0] == pytest.approx(np.log1p(5000))
    assert out["capital_net_log"].iloc[1] == pytest.approx(-np.log1p(1000))


def test_ece_is_zero_for_perfect_calibration():
    p = np.repeat([0.1, 0.9], 1000)
    y = np.concatenate([np.r_[np.ones(100), np.zeros(900)], np.r_[np.ones(900), np.zeros(100)]])
    assert expected_calibration_error(y, p) == pytest.approx(0.0, abs=1e-9)


@pytest.fixture
def api_client(monkeypatch):
    """Exercise a real fitted pipeline without downloading data or skipping tests."""
    from fastapi.testclient import TestClient
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.frozen import FrozenEstimator
    from sklearn.linear_model import LogisticRegression

    from adult_poc import api
    from adult_poc.features import make_pipeline

    rows = pd.DataFrame([ROW, {**ROW, "age": 22, "capital-gain": 0, "hours-per-week": 20},
                         {**ROW, "age": 55}, {**ROW, "age": 18, "capital-gain": 0, "hours-per-week": 10}])
    pipe = make_pipeline(LogisticRegression(random_state=42)).fit(rows, [1, 0, 1, 0])
    cal = CalibratedClassifierCV(FrozenEstimator(pipe), method="isotonic", cv=2).fit(rows, [1, 0, 1, 0])
    monkeypatch.setattr(api, "_load", lambda: (pipe, {"model": "test_model", "threshold": 0.6}, cal))
    return TestClient(api.app), pipe, cal


def test_api_predict(api_client):
    client, pipe, cal = api_client
    assert client.get("/health").json()["status"] == "ok"
    body = {k: v for k, v in ROW.items() if k != "fnlwgt"}
    r = client.post("/predict", json=[body, {**body, "workclass": None, "occupation": None}])
    assert r.status_code == 200
    out = r.json()
    assert len(out) == 2
    p = pipe.predict_proba(pd.DataFrame([ROW]))[0, 1]
    pc = cal.predict_proba(pd.DataFrame([ROW]))[0, 1]
    assert out[0]["probability_gt_50k"] == pytest.approx(round(p, 4))
    assert out[0]["calibrated_probability_gt_50k"] == pytest.approx(round(pc, 4))
    assert out[0]["prediction"] == (">50K" if p >= 0.6 else "<=50K")


@pytest.mark.parametrize("body", [[], [{**ROW, "capital_gian": 5000}],
                                  [{**ROW, "capital-gain": "Infinity"}],
                                  [{**ROW, "capital-loss": "NaN"}],
                                  [{**ROW, "age": -1}], [{**ROW, "sex": "Unknown"}]])
def test_api_rejects_invalid_inputs(api_client, body):
    client, _, _ = api_client
    assert client.post("/predict", json=body).status_code == 422


def test_api_reports_missing_model(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient
    from adult_poc import api

    monkeypatch.setattr(api, "MODELS", tmp_path)
    api._load.cache_clear()
    assert TestClient(api.app).get("/health").status_code == 503
    api._load.cache_clear()


def test_api_rejects_overflowing_json_number(api_client):
    import json

    client, _, _ = api_client
    body = json.dumps([ROW]).replace('5000', '1e999')
    assert client.post("/predict", content=body, headers={"Content-Type": "application/json"}).status_code == 422
