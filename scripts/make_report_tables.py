"""Export result tables to LaTeX for the report (report/tables/*.tex)."""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
A, OUT = ROOT / "artifacts", ROOT / "report" / "tables"
OUT.mkdir(parents=True, exist_ok=True)
PRETTY = {"logreg": "Logistic Reg.", "random_forest": "Random Forest", "lightgbm": "LightGBM",
          "dummy_prior": "Dummy (prior)"}


def esc(s: str) -> str:
    return str(s).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")


# Model comparison
mc = pd.read_csv(A / "tables/model_comparison.csv")
lines = []
for _, r in mc.iterrows():
    f = lambda c: "--" if pd.isna(r[c]) else f"{r[c]:.3f}"  # noqa: E731
    lines.append(f"{PRETTY[r.model]} & {r.split.replace('_', ' ')} & {f('pr_auc')} & {f('roc_auc')} & {f('f1')} & "
                 f"{f('precision')} & {f('recall')} & {f('balanced_acc')} & {f('brier')} & {f('threshold')} \\\\")
(OUT / "model_comparison.tex").write_text("\n".join(lines))

# Imbalance ablation
ab = pd.read_csv(A / "tables/imbalance_ablation.csv")
lines = [f"{PRETTY[r.model]} & {esc(r.remedy)} & {r.pr_auc:.3f} & {r.f1:.3f} & {r.precision:.3f} & {r.recall:.3f} & "
         f"{r.balanced_acc:.3f} & {r.accuracy:.3f} & {r.brier:.3f} & {r.threshold:.2f} \\\\" for r in ab.itertuples()]
(OUT / "imbalance_ablation.tex").write_text("\n".join(lines))

# Fairness
fs = json.load(open(A / "tables/fairness_summary.json"))
rows = [("f1", "F1"), ("balanced_acc", "Balanced acc."), ("accuracy", "Accuracy"),
        ("dp_difference", "Demographic parity diff."), ("dp_ratio", "Demographic parity ratio"),
        ("tpr_difference", "TPR diff. (equal opportunity)"), ("fpr_difference", "FPR diff."),
        ("eo_difference", "Equalized odds diff."), ("precision_difference", "Precision diff. (pred. parity)")]
lines = [f"{lab} & {fs['baseline'][k]:.3f} & {fs['unawareness'][k]:.3f} & {fs['threshold_optimizer_eo'][k]:.3f} \\\\"
         for k, lab in rows]
(OUT / "fairness_summary.tex").write_text("\n".join(lines))
fg = pd.read_csv(A / "tables/fairness_by_group.csv")
fg = fg[fg.variant != "race_baseline"]
lines = [f"{esc(r.variant).replace('threshold\\_optimizer\\_eo', 'ThresholdOpt (EO)')} & {r.group} & {r.selection_rate:.3f} & "
         f"{r.tpr:.3f} & {r.fpr:.3f} & {r.precision:.3f} & {r.f1:.3f} \\\\" for r in fg.itertuples()]
(OUT / "fairness_by_group.tex").write_text("\n".join(lines))

# Run log (appendix): every trial
tl = pd.read_csv(A / "runs/trials.csv")
lines = []
for r in tl.itertuples():
    p = json.loads(r.params)
    ps = ", ".join(f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}" for k, v in p.items())
    star = r"$\star$" if r.is_best else ""
    art = "hist" + (" + model" if r.is_best else "")
    lines.append(f"{PRETTY[r.model]} & {r.trial}{star} & {esc(ps)} & {r.cv_pr_auc_mean:.4f} & {r.cv_pr_auc_std:.4f} & "
                 f"{r.cv_roc_auc_mean:.4f} & {r.cv_f1_mean:.3f} & {r.cv_brier_mean:.4f} & {r.duration_s:.0f} & {art} \\\\")
(OUT / "run_log.tex").write_text("\n".join(lines))
print("tables written to", OUT, file=sys.stderr)
