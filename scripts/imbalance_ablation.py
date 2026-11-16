"""Step 3 - class-imbalance remedies compared on the validation set.

Same LightGBM (default params) and LogisticRegression, varying only the remedy:
  none            : no correction, threshold 0.5
  none+thr        : no correction, F1-optimal threshold chosen on validation
  class_weight    : cost-sensitive learning (balanced), threshold 0.5
  class_weight+thr: cost-sensitive learning + tuned threshold   <- retained
  smote           : SMOTE oversampling of the training data, threshold 0.5
  undersample     : random under-sampling of the majority class, threshold 0.5

Resamplers are inserted in an imblearn Pipeline *after* preprocessing so that
synthetic points are created only from training rows (never from validation).
"""
import json
import logging
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402
from imblearn.over_sampling import SMOTE  # noqa: E402
from imblearn.pipeline import Pipeline as ImbPipeline  # noqa: E402
from imblearn.under_sampling import RandomUnderSampler  # noqa: E402

from adult_poc.config import load_config  # noqa: E402
from adult_poc.data import load_raw, split  # noqa: E402
from adult_poc.features import FeatureEngineer, make_pipeline, make_preprocessor  # noqa: E402
from adult_poc.metrics import best_f1_threshold, evaluate  # noqa: E402
from adult_poc.models import default_model  # noqa: E402

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    warnings.simplefilter("ignore")
    cfg = load_config()
    seed = cfg["seed"]
    X, y = load_raw(cfg)
    X_tr, X_va, _, y_tr, y_va, _ = split(X, y, cfg)
    rows = []
    for name in ["logreg", "lightgbm"]:
        scale = name == "logreg"
        variants = {
            "none": make_pipeline(default_model(name, seed, class_weight=None), scale=scale),
            "class_weight": make_pipeline(default_model(name, seed, class_weight="balanced"), scale=scale),
            "smote": ImbPipeline([("features", FeatureEngineer()), ("preprocess", make_preprocessor(scale)),
                                  ("resample", SMOTE(random_state=seed)),
                                  ("model", default_model(name, seed, class_weight=None))]),
            "undersample": ImbPipeline([("features", FeatureEngineer()), ("preprocess", make_preprocessor(scale)),
                                        ("resample", RandomUnderSampler(random_state=seed)),
                                        ("model", default_model(name, seed, class_weight=None))]),
        }
        for remedy, pipe in variants.items():
            proba = pipe.fit(X_tr, y_tr).predict_proba(X_va)[:, 1]
            rows.append({"model": name, "remedy": remedy, **evaluate(y_va, proba, 0.5)})
            if remedy in ("none", "class_weight"):
                thr = best_f1_threshold(y_va, proba)
                rows.append({"model": name, "remedy": remedy + "+thr", **evaluate(y_va, proba, thr)})
    df = pd.DataFrame(rows).round(4)
    out = cfg["paths"]["tables"] / "imbalance_ablation.csv"
    df.to_csv(out, index=False)
    print(df[["model", "remedy", "pr_auc", "f1", "precision", "recall", "balanced_acc", "accuracy", "brier", "threshold"]]
          .to_string(index=False))
    json.dump({"note": "validation set, default hyper-parameters"}, open(out.with_suffix(".json"), "w"))
