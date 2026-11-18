"""Data loading and the train / validation / test split.

The dataset is loaded with ``fetch_openml(name="adult", version=2)`` as required.
If OpenML cannot be reached (offline machine, firewalled network) we fall back
to the identical UCI file stored in ``data/raw/adult.csv``.
"""
from __future__ import annotations

import hashlib
import logging
import os

import pandas as pd
from sklearn.model_selection import train_test_split

log = logging.getLogger(__name__)

COLUMNS = [
    "age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
    "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss",
    "hours-per-week", "native-country", "class",
]
NUMERIC_RAW = ["age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
UCI_TRAIN_ROWS = 32561
UCI_TEST_ROWS = 16281


def _load_openml(cfg: dict) -> tuple[pd.DataFrame, pd.Series]:
    from sklearn.datasets import fetch_openml

    if os.environ.get("ADULT_OFFLINE") == "1":
        raise ConnectionError("ADULT_OFFLINE=1")
    ds = fetch_openml(name=cfg["data"]["openml_name"], version=cfg["data"]["openml_version"], as_frame=True,
                      n_retries=1, delay=0.5)
    # Adult v2 on OpenML is adult.test followed by adult.data. Our original
    # local experiment used adult.data followed by adult.test. Keep that order
    # on both routes so seed 42 selects the same people, not just the same sizes.
    if len(ds.data) != UCI_TRAIN_ROWS + UCI_TEST_ROWS:
        raise ValueError("Unexpected Adult v2 row count")
    order = list(range(UCI_TEST_ROWS, len(ds.data))) + list(range(UCI_TEST_ROWS))
    return ds.data.iloc[order].reset_index(drop=True), ds.target.iloc[order].reset_index(drop=True)


def _load_local(cfg: dict) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(cfg["data"]["local_csv"], header=None, names=COLUMNS,
                     na_values="?", skipinitialspace=True)
    if len(df) != UCI_TRAIN_ROWS + UCI_TEST_ROWS:
        raise ValueError("Expected 48,842 rows: adult.data followed by adult.test")
    y = df.pop("class").str.strip().str.rstrip(".")
    for c in df.columns:
        if c not in NUMERIC_RAW:
            df[c] = df[c].astype("category")
    return df, y


def load_raw(cfg: dict) -> tuple[pd.DataFrame, pd.Series]:
    """Return features X and binary target y (1 = income >50K)."""
    try:
        X, y = _load_openml(cfg)
        log.info("Loaded Adult from OpenML: %s", X.shape)
    except Exception as exc:  # network errors, proxy, etc.
        log.warning("OpenML unavailable (%s); using local copy %s", type(exc).__name__, cfg["data"]["local_csv"])
        X, y = _load_local(cfg)
    labels = y.astype(str).str.strip().str.rstrip(".")
    if not labels.isin(["<=50K", ">50K"]).all():
        raise ValueError("Unexpected Adult target labels")
    y = (labels == cfg["data"]["positive_label"]).astype(int).rename("target")
    # Exact duplicate rows carry no new information and could leak across splits.
    dup = X.assign(_y=y.values).duplicated()
    log.info("Dropping %d exact duplicate rows", int(dup.sum()))
    X, y = X.loc[~dup].reset_index(drop=True), y.loc[~dup].reset_index(drop=True)
    expected = cfg["data"].get("sha256")
    if expected and data_fingerprint(X, y) != expected:
        raise ValueError("Adult data/order differs from the recorded experiment; check the local CSV order")
    return X, y


def data_fingerprint(X: pd.DataFrame, y: pd.Series) -> str:
    """Hash row values in order; comparable across OpenML and local dtypes."""
    rows = X[COLUMNS[:-1]].copy()
    for c in NUMERIC_RAW:
        rows[c] = pd.to_numeric(rows[c]).astype("float64")
    rows = rows.astype(object).where(rows.notna(), "<Missing>")
    rows["target"] = y.to_numpy()
    return hashlib.sha256(rows.to_csv(index=False).encode("utf-8")).hexdigest()


def split(X: pd.DataFrame, y: pd.Series, cfg: dict):
    """Stratified 60 / 20 / 20 split.

    * train      -> Optuna tuning with k-fold CV, final fit
    * validation -> model selection, decision threshold, calibration, fairness post-processing
    * test       -> held out from fitting and selection, used for final evaluation
    """
    seed = cfg["seed"]
    test_size = cfg["data"]["test_size"]
    val_rel = cfg["data"]["val_size"] / (1 - test_size)
    X_tmp, X_test, y_tmp, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)
    X_train, X_val, y_train, y_val = train_test_split(X_tmp, y_tmp, test_size=val_rel, stratify=y_tmp, random_state=seed)
    return X_train, X_val, X_test, y_train, y_val, y_test
