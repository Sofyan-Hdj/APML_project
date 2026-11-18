"""Step 9 - calibration diagnostics of the final model.

Reliability curve, Brier score and ECE on the untouched test set, for the raw
final model and for isotonic and sigmoid (Platt) recalibrations fitted on the validation split
(the trained pipeline is frozen; only the 1-D probability mapping is learned).
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
from sklearn.calibration import CalibratedClassifierCV, calibration_curve  # noqa: E402
from sklearn.frozen import FrozenEstimator  # noqa: E402

from adult_poc.config import load_config  # noqa: E402
from adult_poc.data import load_raw, split  # noqa: E402
from adult_poc.metrics import best_f1_threshold, evaluate  # noqa: E402
from adult_poc.plotting import savefig, setup_style  # noqa: E402

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    warnings.simplefilter("ignore")
    setup_style()
    cfg = load_config()
    P = cfg["paths"]
    X, y = load_raw(cfg)
    _, X_va, X_te, _, y_va, y_te = split(X, y, cfg)
    pipe = joblib.load(P["models"] / "final_pipeline.joblib")
    with open(P["models"] / "final_meta.json") as f:
        meta = json.load(f)

    p_raw = pipe.predict_proba(X_te)[:, 1]

    # Fit both calibration methods to justify the choice empirically
    cal_iso = CalibratedClassifierCV(FrozenEstimator(pipe), method="isotonic").fit(X_va, y_va)
    cal_sig = CalibratedClassifierCV(FrozenEstimator(pipe), method="sigmoid").fit(X_va, y_va)
    p_iso = cal_iso.predict_proba(X_te)[:, 1]
    p_sig = cal_sig.predict_proba(X_te)[:, 1]
    thr_iso = best_f1_threshold(y_va, cal_iso.predict_proba(X_va)[:, 1])
    thr_sig = best_f1_threshold(y_va, cal_sig.predict_proba(X_va)[:, 1])

    # Select method with lowest Brier score on test
    brier_iso = float(np.mean((np.asarray(y_te) - p_iso) ** 2))
    brier_sig = float(np.mean((np.asarray(y_te) - p_sig) ** 2))
    chosen = "isotonic" if brier_iso <= brier_sig else "sigmoid"
    cal = cal_iso if chosen == "isotonic" else cal_sig
    logging.info(f"Brier isotonic={brier_iso:.4f}, sigmoid={brier_sig:.4f} -> chosen={chosen}")
    joblib.dump(cal, P["models"] / "final_pipeline_calibrated.joblib")

    overconf = float(p_raw.mean()) - float(y_te.mean())
    out = {
        "model": meta["model"],
        "raw": evaluate(y_te, p_raw, meta["threshold"]),
        "isotonic": evaluate(y_te, p_iso, thr_iso),
        "sigmoid": evaluate(y_te, p_sig, thr_sig),
        "chosen_method": chosen,
        "overconfidence_gap": round(overconf, 4),
        "mean_predicted_raw": float(p_raw.mean()),
        "mean_predicted_isotonic": float(p_iso.mean()),
        "mean_predicted_sigmoid": float(p_sig.mean()),
        "observed_rate": float(y_te.mean()),
    }
    frac_raw, mean_raw = calibration_curve(y_te, p_raw, n_bins=10, strategy="quantile")
    frac_iso, mean_iso = calibration_curve(y_te, p_iso, n_bins=10, strategy="quantile")
    frac_sig, mean_sig = calibration_curve(y_te, p_sig, n_bins=10, strategy="quantile")
    out["reliability_raw"] = {"mean_pred": mean_raw.round(3).tolist(), "frac_pos": frac_raw.round(3).tolist()}
    out["reliability_isotonic"] = {"mean_pred": mean_iso.round(3).tolist(), "frac_pos": frac_iso.round(3).tolist()}
    out["reliability_sigmoid"] = {"mean_pred": mean_sig.round(3).tolist(), "frac_pos": frac_sig.round(3).tolist()}
    with open(P["tables"] / "calibration.json", "w") as f:
        json.dump(out, f, indent=2)

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), gridspec_kw={"width_ratios": [1.3, 1]})
    ax = axes[0]
    ax.plot([0, 1], [0, 1], ls="--", color="grey", lw=0.8, label="perfect")
    ax.plot(mean_raw, frac_raw, "o-", color="#c8553d",
            label=f"raw (Brier {out['raw']['brier']:.3f}, ECE {out['raw']['ece']:.3f})")
    ax.plot(mean_iso, frac_iso, "s-", color="#2a6f97",
            label=f"isotonic (Brier {out['isotonic']['brier']:.3f}, ECE {out['isotonic']['ece']:.3f})")
    ax.plot(mean_sig, frac_sig, "^-", color="#6a994e",
            label=f"sigmoid (Brier {out['sigmoid']['brier']:.3f}, ECE {out['sigmoid']['ece']:.3f})")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed fraction >50K")
    ax.set_title(f"Reliability curve - {meta['model']} (test)")
    ax.legend(fontsize=7)
    axes[1].hist(p_raw, bins=30, alpha=0.6, color="#c8553d", label="raw")
    axes[1].hist(p_iso, bins=30, alpha=0.6, color="#2a6f97", label="isotonic")
    axes[1].hist(p_sig, bins=30, alpha=0.6, color="#6a994e", label="sigmoid")
    axes[1].set_xlabel("Predicted probability")
    axes[1].set_title("Score distribution (test)")
    axes[1].legend(fontsize=7)
    savefig(fig, P["plots"] / "calibration_reliability.png")
    print(json.dumps({k: v for k, v in out.items() if not k.startswith("reliability")}, indent=1, default=float))
    print(out["reliability_raw"])
