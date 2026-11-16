"""Step 10 - fairness audit on the sensitive attribute ``sex`` and mitigation.

Fairness definitions reported (test set, final model at its validation threshold):
  * Demographic parity  - P(Yhat=1 | A=a) equal across groups (selection rate)
  * Equal opportunity   - TPR equal across groups
  * Equalized odds      - TPR *and* FPR equal across groups (max of the two gaps)
  * Predictive parity   - precision equal across groups

Mitigation: post-processing with fairlearn ``ThresholdOptimizer`` under an
equalized-odds constraint, fitted on the *validation* split (the trained model
is untouched), then compared with the unmitigated model on the untouched test set.
A "fairness through unawareness" variant (sex removed from the inputs, model
re-trained) is also reported to show that dropping the attribute is not enough.
"""
import json
import logging
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import joblib  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from fairlearn.metrics import (  # noqa: E402
    MetricFrame, demographic_parity_difference, demographic_parity_ratio,
    equalized_odds_difference, false_positive_rate, selection_rate, true_positive_rate,
)
from fairlearn.postprocessing import ThresholdOptimizer  # noqa: E402
from sklearn.base import clone  # noqa: E402
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score  # noqa: E402

from adult_poc.config import load_config  # noqa: E402
from adult_poc.data import load_raw, split  # noqa: E402
from adult_poc.metrics import best_f1_threshold  # noqa: E402
from adult_poc.plotting import savefig, setup_style  # noqa: E402

GROUP_METRICS = {
    "selection_rate": selection_rate,
    "tpr": true_positive_rate,
    "fpr": false_positive_rate,
    "precision": lambda yt, yp: precision_score(yt, yp, zero_division=0),
    "f1": f1_score,
    "accuracy": accuracy_score,
}


def audit(y_true, y_pred, A) -> tuple[dict, pd.DataFrame]:
    mf = MetricFrame(metrics=GROUP_METRICS, y_true=y_true, y_pred=y_pred, sensitive_features=A)
    by = mf.by_group
    summary = {
        "f1": f1_score(y_true, y_pred),
        "balanced_acc": balanced_accuracy_score(y_true, y_pred),
        "accuracy": accuracy_score(y_true, y_pred),
        "dp_difference": demographic_parity_difference(y_true, y_pred, sensitive_features=A),
        "dp_ratio": demographic_parity_ratio(y_true, y_pred, sensitive_features=A),
        "tpr_difference": float(by["tpr"].max() - by["tpr"].min()),
        "fpr_difference": float(by["fpr"].max() - by["fpr"].min()),
        "eo_difference": equalized_odds_difference(y_true, y_pred, sensitive_features=A),
        "precision_difference": float(by["precision"].max() - by["precision"].min()),
    }
    return summary, by


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    warnings.simplefilter("ignore")
    setup_style()
    cfg = load_config()
    P = cfg["paths"]
    s = cfg["data"]["sensitive_attribute"]
    X, y = load_raw(cfg)
    X_tr, X_va, X_te, y_tr, y_va, y_te = split(X, y, cfg)
    A_va, A_te = X_va[s].astype(str), X_te[s].astype(str)
    pipe = joblib.load(P["models"] / "final_pipeline.joblib")
    meta = json.load(open(P["models"] / "final_meta.json"))

    results, groups = {}, {}

    # 1. Baseline: final model, single global threshold
    yhat = (pipe.predict_proba(X_te)[:, 1] >= meta["threshold"]).astype(int)
    results["baseline"], groups["baseline"] = audit(y_te, yhat, A_te)
    results["ground_truth_base_rate"] = y_te.groupby(A_te.values).mean().to_dict()

    # 2. Unawareness: same model family / params, sex blanked out of the inputs
    blind = clone(pipe).fit(X_tr.assign(**{s: "NA"}), y_tr)
    p_va_b = blind.predict_proba(X_va.assign(**{s: "NA"}))[:, 1]
    yhat_b = (blind.predict_proba(X_te.assign(**{s: "NA"}))[:, 1] >= best_f1_threshold(y_va, p_va_b)).astype(int)
    results["unawareness"], groups["unawareness"] = audit(y_te, yhat_b, A_te)

    # 3. Mitigation: ThresholdOptimizer (equalized odds), fitted on validation
    to = ThresholdOptimizer(estimator=pipe, constraints="equalized_odds", objective="balanced_accuracy_score",
                            predict_method="predict_proba", prefit=True)
    to.fit(X_va, y_va, sensitive_features=A_va)
    yhat_m = to.predict(X_te, sensitive_features=A_te, random_state=cfg["seed"])
    results["threshold_optimizer_eo"], groups["threshold_optimizer_eo"] = audit(y_te, yhat_m, A_te)
    joblib.dump(to, P["models"] / "fair_threshold_optimizer.joblib")

    # Secondary attribute: race (audit only)
    results["race_baseline"], groups["race_baseline"] = audit(y_te, yhat, X_te["race"].astype(str))

    by_tables = pd.concat({k: v for k, v in groups.items()}, names=["variant", "group"]).round(4)
    by_tables.to_csv(P["tables"] / "fairness_by_group.csv")
    json.dump(results, open(P["tables"] / "fairness_summary.json", "w"), indent=2, default=float)
    print(pd.DataFrame({k: v for k, v in results.items() if k != "ground_truth_base_rate"}).round(4).to_string())
    print(by_tables.to_string())
    print("base rates", results["ground_truth_base_rate"])

    # Plot: per-group TPR / FPR / selection rate, before vs after
    variants = ["baseline", "unawareness", "threshold_optimizer_eo"]
    labels = ["Baseline", "Unawareness", "ThresholdOpt (EO)"]
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2), sharey=True)
    for ax, metric in zip(axes, ["selection_rate", "tpr", "fpr"]):
        xs = np.arange(len(variants))
        for j, (g, color) in enumerate([("Female", "#c8553d"), ("Male", "#2a6f97")]):
            vals = [groups[v].loc[g, metric] for v in variants]
            ax.bar(xs + (j - 0.5) * 0.35, vals, 0.35, color=color, label=g)
        ax.set_xticks(xs, labels, fontsize=7)
        ax.set_title(metric.replace("_", " ").upper() if metric != "selection_rate" else "Selection rate")
    axes[0].legend()
    savefig(fig, P["plots"] / "fairness_groups.png")
