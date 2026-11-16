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


@pytest.mark.skipif(not (Path(__file__).resolve().parents[1] / "artifacts/models/final_pipeline.joblib").exists(),
                    reason="train the model first (make train)")
def test_api_predict():
    from fastapi.testclient import TestClient

    from adult_poc.api import app

    client = TestClient(app)
    assert client.get("/health").json()["status"] == "ok"
    body = {k: v for k, v in ROW.items() if k != "fnlwgt"}
    r = client.post("/predict", json=[body, {**body, "workclass": None, "occupation": None}])
    assert r.status_code == 200
    out = r.json()
    assert len(out) == 2 and 0 <= out[0]["probability_gt_50k"] <= 1
