"""Step 8 - SHAP explanations of the final model (global + local)."""
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
import shap  # noqa: E402

from adult_poc.config import load_config  # noqa: E402
from adult_poc.data import load_raw, split  # noqa: E402
from adult_poc.features import BINARY, CATEGORICAL, NUMERIC  # noqa: E402
from adult_poc.plotting import savefig, setup_style  # noqa: E402


def parent_feature(col: str) -> str:
    for c in CATEGORICAL:
        if col.startswith(c + "_"):
            return c
    return col


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    warnings.simplefilter("ignore")
    setup_style()
    cfg = load_config()
    P = cfg["paths"]
    X, y = load_raw(cfg)
    _, _, X_te, _, _, y_te = split(X, y, cfg)
    pipe = joblib.load(P["models"] / "final_pipeline.joblib")
    meta = json.load(open(P["models"] / "final_meta.json"))
    thr = meta["threshold"]

    pre = pipe[:-1]
    model = pipe[-1]
    names = list(pre.get_feature_names_out())
    sample = X_te.sample(2000, random_state=cfg["seed"])
    Z = pd.DataFrame(pre.transform(sample), columns=names, index=sample.index)

    if meta["model"] == "logreg":
        explainer = shap.LinearExplainer(model, Z)
    else:
        explainer = shap.TreeExplainer(model)
    sv = explainer(Z)
    if sv.values.ndim == 3:  # RF returns one slice per class
        sv = sv[:, :, 1]

    # Global: mean |SHAP| per encoded column and aggregated to the original feature
    imp = pd.Series(np.abs(sv.values).mean(0), index=names).sort_values(ascending=False)
    agg = imp.groupby([parent_feature(c) for c in imp.index]).sum().sort_values(ascending=False)
    imp.round(5).to_csv(P["tables"] / "shap_importance_encoded.csv", header=["mean_abs_shap"])
    agg.round(5).to_csv(P["tables"] / "shap_importance_grouped.csv", header=["mean_abs_shap"])

    fig, ax = plt.subplots(figsize=(5, 3.6))
    top = agg.head(14)[::-1]
    ax.barh(top.index, top.values, color="#2a6f97")
    ax.set_xlabel("mean |SHAP| (log-odds), summed over one-hot levels")
    ax.set_title("Global importance by original feature")
    savefig(fig, P["plots"] / "shap_global_grouped.png")

    plt.figure()
    shap.plots.beeswarm(sv, max_display=15, show=False)
    fig = plt.gcf()
    fig.set_size_inches(7, 5)
    savefig(fig, P["plots"] / "shap_beeswarm.png")

    # Local: pick representative individual predictions from the sample
    proba = pipe.predict_proba(sample)[:, 1]
    yt = y_te.loc[sample.index].values
    pred = proba >= thr
    pick = {
        "true_positive_confident": int(np.argmax(np.where(pred & (yt == 1), proba, -1))),
        "true_negative_confident": int(np.argmin(np.where(~pred & (yt == 0), proba, 2))),
        "false_positive": int(np.argmax(np.where(pred & (yt == 0), proba, -1))),
        "false_negative": int(np.argmin(np.where(~pred & (yt == 1), proba, 2))),
        "borderline": int(np.argmin(np.abs(proba - thr))),
    }
    cases = []
    for label, i in pick.items():
        plt.figure()
        shap.plots.waterfall(sv[i], max_display=10, show=False)
        fig = plt.gcf()
        fig.set_size_inches(7, 4)
        fig.suptitle(f"{label}: p(>50K)={proba[i]:.2f}, true={'>50K' if yt[i] else '<=50K'}", fontsize=9)
        savefig(fig, P["plots"] / f"shap_local_{label}.png")
        row = sample.iloc[i]
        top3 = pd.Series(sv.values[i], index=names).sort_values(key=np.abs, ascending=False).head(4)
        cases.append({"case": label, "proba": round(float(proba[i]), 3), "true": int(yt[i]),
                      "profile": {k: (str(v)) for k, v in row.items() if k not in ("fnlwgt", "education")},
                      "top_contributions": top3.round(3).to_dict()})
    json.dump(cases, open(P["tables"] / "shap_local_cases.json", "w"), indent=2)
    print(agg.round(4).to_string())
    print(json.dumps(cases, indent=1))
