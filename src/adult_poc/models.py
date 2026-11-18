"""The three model families and their Optuna search spaces.

Progression: a regularised linear baseline -> bagged trees (non-linear,
interactions, low variance) -> gradient boosting (sequential bias reduction).

Class imbalance (24% positives) is handled the same way for all three:
cost-sensitive learning through ``class_weight="balanced"`` (positives weighted
~3.2x) and a decision threshold chosen on the validation set. The ablation in
``scripts/imbalance_ablation.py`` compares this to resampling (SMOTE, random
under-sampling) and to doing nothing.
"""
from __future__ import annotations

import optuna
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

MODEL_NAMES = ["logreg", "random_forest", "lightgbm"]
PRETTY = {"logreg": "Logistic Regression", "random_forest": "Random Forest", "lightgbm": "LightGBM"}


def default_model(name: str, seed: int, class_weight="balanced"):
    if name == "logreg":
        return LogisticRegression(max_iter=2000, class_weight=class_weight, random_state=seed)
    if name == "random_forest":
        return RandomForestClassifier(n_estimators=300, min_samples_leaf=5, class_weight=class_weight,
                                      n_jobs=-1, random_state=seed)
    if name == "lightgbm":
        return LGBMClassifier(class_weight=class_weight, random_state=seed, verbose=-1, n_jobs=-1)
    raise ValueError(name)


def suggest_params(name: str, trial: optuna.Trial) -> dict:
    if name == "logreg":
        return {
            "C": trial.suggest_float("C", 1e-3, 10.0, log=True),
            "l1_ratio": trial.suggest_float("l1_ratio", 0.0, 1.0),
        }
    if name == "random_forest":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 400, step=50),
            "max_depth": trial.suggest_int("max_depth", 6, 30),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
            "max_features": trial.suggest_float("max_features", 0.1, 0.7),
            "max_samples": trial.suggest_float("max_samples", 0.4, 1.0),
        }
    if name == "lightgbm":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 800, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 8, 128, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        }
    raise ValueError(name)


def build_model(name: str, params: dict, seed: int):
    if name == "logreg":
        # In pinned scikit-learn 1.9, l1_ratio selects L2 / elastic net / L1.
        return LogisticRegression(solver="saga", max_iter=3000, class_weight="balanced",
                                  random_state=seed, **params)
    if name == "random_forest":
        return RandomForestClassifier(class_weight="balanced", n_jobs=-1, random_state=seed, **params)
    if name == "lightgbm":
        # Fixed settings belong here so CV and reconstruction use the same model.
        return LGBMClassifier(class_weight="balanced", random_state=seed, verbose=-1, n_jobs=-1,
                              **{"subsample_freq": 1, **params})
    raise ValueError(name)


def uses_scaling(name: str) -> bool:
    return name == "logreg"
