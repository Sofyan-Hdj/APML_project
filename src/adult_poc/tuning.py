"""Optuna hyper-parameter search with stratified k-fold CV on the training set.

Only the training split is seen during tuning: each trial is scored by the mean
of 5 out-of-fold scores, so the validation and test splits never influence the
choice of hyper-parameters and the reported CV score is not optimistic.
Every trial is appended to ``artifacts/runs/trials.csv`` (the run log).
"""
from __future__ import annotations

import csv
import json
import logging
import subprocess
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
from sklearn.model_selection import StratifiedKFold, cross_validate

from .features import make_pipeline
from .metrics import CV_SCORING
from .models import build_model, suggest_params, uses_scaling
from .plotting import MODEL_COLORS, savefig, setup_style

log = logging.getLogger(__name__)
RUN_LOG_FIELDS = [
    "run_id", "timestamp_utc", "git_commit", "model", "trial", "state", "duration_s", "params",
    "cv_pr_auc_mean", "cv_pr_auc_std", "cv_roc_auc_mean", "cv_f1_mean", "cv_balanced_acc_mean",
    "cv_brier_mean", "is_best", "artifacts",
]


def _git_commit() -> str:
    # Run git inside this project, not in the caller's working directory, so the run log
    # records this repository's commit even if the script is launched from elsewhere.
    repo = Path(__file__).resolve().parents[2]
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True,
                                       cwd=repo, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "n/a"


def tune_model(name: str, X, y, cfg: dict) -> optuna.Study:
    seed = cfg["seed"]
    tcfg = cfg["tuning"]
    cv = StratifiedKFold(n_splits=tcfg["cv_folds"], shuffle=True, random_state=seed)
    fe_kwargs = {"drop": tuple(cfg["features"]["drop"]),
                 "rare_min_count": cfg["features"]["rare_country_min_count"]}

    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(name, trial)
        pipe = make_pipeline(build_model(name, params, seed), fe_kwargs, scale=uses_scaling(name))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = cross_validate(pipe, X, y, cv=cv, scoring=CV_SCORING, n_jobs=1)
        for k in CV_SCORING:
            s = res[f"test_{k}"]
            if k == "brier":
                s = -s
            trial.set_user_attr(f"{k}_mean", float(np.mean(s)))
            trial.set_user_attr(f"{k}_std", float(np.std(s)))
        return float(np.mean(res["test_pr_auc"]))

    study = optuna.create_study(direction="maximize", study_name=name,
                                sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=tcfg["n_trials"][name], timeout=tcfg["timeout_per_model_s"],
                   show_progress_bar=False)
    log.info("%s best CV PR-AUC=%.4f params=%s", name, study.best_value, study.best_params)
    return study


def plot_history(study: optuna.Study, name: str, path: Path) -> Path:
    setup_style()
    done = [t for t in study.trials if t.value is not None]
    x = [t.number for t in done]
    v = [t.value for t in done]
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.scatter(x, v, s=14, color=MODEL_COLORS[name], alpha=0.7, label="trial")
    ax.plot(x, np.maximum.accumulate(v), color="black", lw=1.2, label="best so far")
    ax.set_xlabel("Trial")
    ax.set_ylabel("5-fold CV PR-AUC")
    ax.set_title(f"Optuna history - {name}")
    ax.legend()
    return savefig(fig, path)


def append_run_log(study: optuna.Study, name: str, run_id: str, log_path: Path, artifacts: dict) -> None:
    """Write one row per trial: parameters, CV metrics, and the artifacts it maps to."""
    new = not log_path.exists()
    commit = _git_commit()
    with open(log_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=RUN_LOG_FIELDS)
        if new:
            w.writeheader()
        for t in study.trials:
            best = t.number == study.best_trial.number
            arts = [artifacts["history_plot"], artifacts["trial_json"]]
            if best:
                arts += artifacts["best"]
            ua = t.user_attrs
            w.writerow({
                "run_id": run_id,
                "timestamp_utc": (t.datetime_start or datetime.now(timezone.utc)).isoformat(timespec="seconds"),
                "git_commit": commit,
                "model": name,
                "trial": t.number,
                "state": t.state.name,
                "duration_s": round(t.duration.total_seconds(), 1) if t.duration else "",
                "params": json.dumps(t.params, sort_keys=True),
                "cv_pr_auc_mean": round(ua.get("pr_auc_mean", float("nan")), 5),
                "cv_pr_auc_std": round(ua.get("pr_auc_std", float("nan")), 5),
                "cv_roc_auc_mean": round(ua.get("roc_auc_mean", float("nan")), 5),
                "cv_f1_mean": round(ua.get("f1_mean", float("nan")), 5),
                "cv_balanced_acc_mean": round(ua.get("balanced_acc_mean", float("nan")), 5),
                "cv_brier_mean": round(ua.get("brier_mean", float("nan")), 5),
                "is_best": int(best),
                "artifacts": ";".join(str(a) for a in arts),
            })


def run_tuning(name: str, X, y, cfg: dict, run_id: str | None = None) -> dict:
    run_id = run_id or time.strftime("%Y%m%d-%H%M%S")
    study = tune_model(name, X, y, cfg)
    paths = cfg["paths"]
    rel = lambda p: Path(p).relative_to(paths["artifacts"].parent)  # noqa: E731
    hist = plot_history(study, name, paths["plots"] / f"optuna_history_{name}.png")
    trial_json = paths["runs"] / f"trials_{name}_{run_id}.json"
    with open(trial_json, "w") as f:
        json.dump([{"trial": t.number, "params": t.params, "value": t.value, **t.user_attrs}
                   for t in study.trials], f, indent=1)
    best = {"model": name, "params": study.best_params, "cv": study.best_trial.user_attrs,
            "cv_pr_auc": study.best_value, "best_trial": study.best_trial.number, "run_id": run_id}
    best_json = paths["runs"] / f"best_params_{name}.json"
    with open(best_json, "w") as f:
        json.dump(best, f, indent=2)
    append_run_log(study, name, run_id, paths["runs"] / "trials.csv", {
        "history_plot": rel(hist), "trial_json": rel(trial_json),
        "best": [rel(best_json), Path("artifacts/models") / f"{name}.joblib"],
    })
    return best
