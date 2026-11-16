"""Prediction endpoint (FastAPI) serving the final pipeline.

Run:  uvicorn adult_poc.api:app --app-dir src --port 8000
Docs: http://localhost:8000/docs
"""
from __future__ import annotations

import json
from typing import Literal, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field

from .config import ROOT

MODELS = ROOT / "artifacts" / "models"


class Person(BaseModel):
    """One census record, using the original column names (hyphens allowed via aliases)."""
    model_config = ConfigDict(populate_by_name=True, json_schema_extra={"example": {
        "age": 45, "workclass": "Private", "education": "Bachelors", "education-num": 13,
        "marital-status": "Married-civ-spouse", "occupation": "Exec-managerial",
        "relationship": "Husband", "race": "White", "sex": "Male", "capital-gain": 0,
        "capital-loss": 0, "hours-per-week": 50, "native-country": "United-States"}})

    age: int = Field(ge=16, le=100)
    workclass: Optional[str] = None
    fnlwgt: Optional[float] = None
    education: Optional[str] = None
    education_num: int = Field(alias="education-num", ge=1, le=16)
    marital_status: str = Field(alias="marital-status")
    occupation: Optional[str] = None
    relationship: str
    race: str
    sex: Literal["Male", "Female"]
    capital_gain: float = Field(0, alias="capital-gain", ge=0)
    capital_loss: float = Field(0, alias="capital-loss", ge=0)
    hours_per_week: int = Field(alias="hours-per-week", ge=1, le=99)
    native_country: Optional[str] = Field(None, alias="native-country")


class Prediction(BaseModel):
    probability_gt_50k: float
    calibrated_probability_gt_50k: Optional[float]
    prediction: Literal[">50K", "<=50K"]
    threshold: float
    model: str


def _load():
    pipe = joblib.load(MODELS / "final_pipeline.joblib")
    meta = json.load(open(MODELS / "final_meta.json"))
    cal_path = MODELS / "final_pipeline_calibrated.joblib"
    cal = joblib.load(cal_path) if cal_path.exists() else None
    return pipe, meta, cal


app = FastAPI(title="Adult income PoC", version="1.0")
PIPE, META, CAL = _load()


def _frame(people: list[Person]) -> pd.DataFrame:
    df = pd.DataFrame([p.model_dump(by_alias=True) for p in people])
    df["fnlwgt"] = df["fnlwgt"].fillna(0.0)  # dropped by the pipeline anyway
    return df.replace({None: np.nan})


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model": META["model"], "threshold": META["threshold"]}


@app.post("/predict", response_model=list[Prediction])
def predict(people: list[Person]) -> list[Prediction]:
    df = _frame(people)
    p = PIPE.predict_proba(df)[:, 1]
    pc = CAL.predict_proba(df)[:, 1] if CAL is not None else [None] * len(p)
    thr = META["threshold"]
    return [Prediction(probability_gt_50k=round(float(a), 4),
                       calibrated_probability_gt_50k=None if b is None else round(float(b), 4),
                       prediction=">50K" if a >= thr else "<=50K", threshold=round(thr, 4), model=META["model"])
            for a, b in zip(p, pc)]
