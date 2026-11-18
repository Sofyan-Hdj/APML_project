"""Feature engineering and the preprocessing ColumnTransformer.

Everything here is a scikit-learn transformer so that it lives *inside* the
Pipeline: anything learned from data (rare-country list, imputation medians,
scaling statistics, one-hot vocabularies) is fitted on the training fold only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CATEGORICAL = ["workclass", "marital-status", "occupation", "relationship", "race", "sex", "native-country"]
MARRIED = {"Married-civ-spouse", "Married-AF-spouse"}


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Domain features for the Adult census data.

    Engineered features (justified in the report, section 3):
      * ``capital_net_log``  signed log1p(capital-gain - capital-loss): the two raw
        columns are >90% zeros with a long tail up to 99,999; the net amount on a
        log scale is a far better behaved signal of investment income.
      * ``has_capital_gain`` / ``has_capital_loss``  owning investments at all is
        itself informative, independently of the amount.
      * ``is_married``  collapses the 7 marital-status levels to the one split
        that matters (cohabiting spouse => household income pooling).
      * ``overtime``  works more than the 40h legal week.
      * ``age_sq``  lets the linear model capture the inverted-U age/income curve.
      * ``native-country`` levels with < ``rare_min_count`` training rows are
        grouped into ``Other`` (learned on the training data only).

    Dropped: ``fnlwgt`` (a census sampling weight, not a property of the person)
    and ``education`` (a 1-to-1 label of the ordinal ``education-num``).
    """

    def __init__(self, drop: tuple = ("fnlwgt", "education"), rare_min_count: int = 50):
        self.drop = drop
        self.rare_min_count = rare_min_count

    def fit(self, X: pd.DataFrame, y=None):
        counts = X["native-country"].astype("object").value_counts()
        self.frequent_countries_ = set(counts[counts >= self.rare_min_count].index)
        self.feature_names_in_ = np.array(X.columns, dtype=object)
        self.feature_names_out_ = np.array(self.transform(X.head(2)).columns, dtype=object)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for c in CATEGORICAL:
            X[c] = X[c].astype("object")
            X.loc[X[c].isna(), c] = np.nan
        net = X["capital-gain"].astype(float) - X["capital-loss"].astype(float)
        X["capital_net_log"] = np.sign(net) * np.log1p(np.abs(net))
        X["has_capital_gain"] = (X["capital-gain"] > 0).astype(int)
        X["has_capital_loss"] = (X["capital-loss"] > 0).astype(int)
        X["is_married"] = X["marital-status"].isin(MARRIED).astype(int)
        X["overtime"] = (X["hours-per-week"] > 40).astype(int)
        X["age_sq"] = (X["age"].astype(float) ** 2) / 100.0
        nc = X["native-country"]
        X["native-country"] = np.where(nc.isna(), np.nan, np.where(nc.isin(self.frequent_countries_), nc, "Other"))
        X = X.drop(columns=[c for c in self.drop if c in X.columns])
        X = X.drop(columns=["capital-gain", "capital-loss"])
        return X

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_out_


NUMERIC = ["age", "age_sq", "education-num", "hours-per-week", "capital_net_log"]
BINARY = ["has_capital_gain", "has_capital_loss", "is_married", "overtime"]


def make_preprocessor(scale: bool = True) -> ColumnTransformer:
    """Imputation + encoding.

    Missing ``workclass`` / ``occupation`` get an explicit ``Missing`` category
    rather than the mode, preserving their informative absence without assuming
    a particular statistical missingness mechanism.
    """
    num_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        num_steps.append(("scale", StandardScaler()))
    cat = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="Missing")),
        ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=20, sparse_output=False)),
    ])
    return ColumnTransformer(
        [("num", Pipeline(num_steps), NUMERIC), ("bin", "passthrough", BINARY), ("cat", cat, CATEGORICAL)],
        verbose_feature_names_out=False,
    )


def make_pipeline(model, fe_kwargs: dict | None = None, scale: bool = True) -> Pipeline:
    """Single end-to-end Pipeline: raw dataframe in, probability out."""
    return Pipeline([
        ("features", FeatureEngineer(**(fe_kwargs or {}))),
        ("preprocess", make_preprocessor(scale=scale)),
        ("model", model),
    ])
