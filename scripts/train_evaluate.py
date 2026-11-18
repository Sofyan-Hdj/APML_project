"""Step 4/5 - refit the three tuned pipelines, compare them, pick the final model.

* Each pipeline is refit on the full training split with its best Optuna params.
* The decision threshold of each model is the F1-optimal one on *validation*.
* The final model is chosen on validation PR-AUC only.
* The test split is scored once, for the comparison table in the report.
"""
import json
import logging
import sys
import warnings
from importlib.metadata import version
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import joblib  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.dummy import DummyClassifier  # noqa: E402
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve  # noqa: E402
from sklearn.model_selection import StratifiedKFold, cross_validate  # noqa: E402

from adult_poc.config import load_config  # noqa: E402
from adult_poc.data import data_fingerprint, load_raw, split  # noqa: E402
from adult_poc.features import make_pipeline  # noqa: E402
from adult_poc.metrics import CV_SCORING, best_f1_threshold, evaluate  # noqa: E402
from adult_poc.models import MODEL_NAMES, PRETTY, build_model, uses_scaling  # noqa: E402
from adult_poc.plotting import MODEL_COLORS, savefig, setup_style  # noqa: E402

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    warnings.simplefilter("ignore")
    setup_style()
    cfg = load_config()
    P = cfg["paths"]
    X, y = load_raw(cfg)
    X_tr, X_va, X_te, y_tr, y_va, y_te = split(X, y, cfg)

    rows, preds, pipes = [], {"y_val": y_va.values, "y_test": y_te.values}, {}
    # Recheck the originally selected configurations without rerunning or replacing
    # the historical 100-trial search. This also records the current environment.
    cv = StratifiedKFold(n_splits=cfg["tuning"]["cv_folds"], shuffle=True, random_state=cfg["seed"])
    verification = {"dataset_sha256": data_fingerprint(X, y), "seed": cfg["seed"],
                    "versions": {name: version(name) for name in
                                 ["numpy", "pandas", "scikit-learn", "lightgbm", "optuna"]},
                    "split_rows": {"train": len(X_tr), "validation": len(X_va), "test": len(X_te)},
                    "split_sha256": {name: data_fingerprint(xs, ys) for name, xs, ys in
                                     [("train", X_tr, y_tr), ("validation", X_va, y_va), ("test", X_te, y_te)]},
                    "models": {}}
    dummy = DummyClassifier(strategy="prior").fit(X_tr, y_tr)
    for split_name, Xs, ys in [("val", X_va, y_va), ("test", X_te, y_te)]:
        rows.append({"model": "dummy_prior", "split": split_name, **evaluate(ys, dummy.predict_proba(Xs)[:, 1], 0.5)})

    for name in MODEL_NAMES:
        best = json.load(open(P["runs"] / f"best_params_{name}.json"))
        fe_kwargs = {"drop": tuple(cfg["features"]["drop"]),
                     "rare_min_count": cfg["features"]["rare_country_min_count"]}
        pipe = make_pipeline(build_model(name, best["params"], cfg["seed"]), fe_kwargs,
                             scale=uses_scaling(name))
        scores = cross_validate(pipe, X_tr, y_tr, cv=cv, scoring=CV_SCORING, n_jobs=1,
                                error_score="raise")
        current_cv = {key: float((-scores[f"test_{key}"] if key == "brier" else scores[f"test_{key}"]).mean())
                      for key in CV_SCORING}
        verification["models"][name] = {"historical_run_id": best["run_id"], "trial": best["best_trial"],
                                        "effective_model_params": pipe[-1].get_params(), "cv": current_cv}
        pipe.fit(X_tr, y_tr)
        p_va, p_te = pipe.predict_proba(X_va)[:, 1], pipe.predict_proba(X_te)[:, 1]
        thr = best_f1_threshold(y_va, p_va)
        rows.append({"model": name, "split": "cv_train", **current_cv, "threshold": 0.5})
        rows.append({"model": name, "split": "val", **evaluate(y_va, p_va, thr)})
        rows.append({"model": name, "split": "test", **evaluate(y_te, p_te, thr)})
        preds[f"{name}_val"], preds[f"{name}_test"] = p_va, p_te
        joblib.dump(pipe, P["models"] / f"{name}.joblib")
        pipes[name] = (pipe, thr)

    res = pd.DataFrame(rows)
    res.round(4).to_csv(P["tables"] / "model_comparison.csv", index=False)
    val = res[(res.split == "val") & res.model.isin(MODEL_NAMES)].set_index("model")
    final = val["pr_auc"].idxmax()
    pipe, thr = pipes[final]
    joblib.dump(pipe, P["models"] / "final_pipeline.joblib")
    meta = {"model": final, "threshold": thr, "selected_on": "validation PR-AUC",
            "val_pr_auc": float(val.loc[final, "pr_auc"])}
    json.dump(meta, open(P["models"] / "final_meta.json", "w"), indent=2)
    with open(P["runs"] / "refit_verification.json", "w") as f:
        json.dump(verification, f, indent=2)
    pd.DataFrame({k: v for k, v in preds.items() if "val" in k}).to_csv(P["tables"] / "preds_val.csv", index=False)
    pd.DataFrame({k: v for k, v in preds.items() if "test" in k}).to_csv(P["tables"] / "preds_test.csv", index=False)
    print(res.round(4).to_string(index=False))
    print("FINAL:", meta)

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for name in MODEL_NAMES:
        p = preds[f"{name}_val"]
        prec, rec, _ = precision_recall_curve(y_va, p)
        fpr, tpr, _ = roc_curve(y_va, p)
        axes[0].plot(rec, prec, color=MODEL_COLORS[name],
                     label=f"{PRETTY[name]} (AP={average_precision_score(y_va, p):.3f})")
        axes[1].plot(fpr, tpr, color=MODEL_COLORS[name], label=f"{PRETTY[name]} (AUC={roc_auc_score(y_va, p):.3f})")
    axes[0].set_xlabel("Recall")
    axes[0].set_ylabel("Precision")
    axes[1].set_xlabel("False positive rate")
    axes[1].set_ylabel("True positive rate")
    axes[0].axhline(y_va.mean(), ls="--", color="grey", lw=0.8, label="prevalence")
    axes[0].set_title("Precision-Recall (validation)")
    axes[1].plot([0, 1], [0, 1], ls="--", color="grey", lw=0.8)
    axes[1].set_title("ROC (validation)")
    for ax in axes:
        ax.legend(fontsize=7)
    savefig(fig, P["plots"] / "model_pr_roc_val.png")
