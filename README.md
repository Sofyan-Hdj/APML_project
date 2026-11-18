# Adult / Census Income — supervised-learning proof-of-concept

Applied ML: Foundations, Engineering & Evaluation (MSA-DATI07-05) — Albert School.
Group: Sofyan Hadj-Ali · Martin Vallée · Henry Falcon de Longevialle · Kaled Goatache.

**Dataset:** Adult / Census Income (`fetch_openml(name="adult", version=2)`), 48,842 rows,
binary target `income > 50K` (24% positives), sensitive attribute `sex`.

End-to-end pipeline: EDA → feature engineering → class-imbalance handling → three models
(Logistic Regression, Random Forest, LightGBM) tuned with Optuna under 5-fold CV → SHAP
explanations → calibration diagnostics → fairness audit and mitigation → FastAPI endpoint.
Methodological choices and corrected results are in `report/report.tex`; its
preview compiles in the LaTeX editor. The tracked `report/report.pdf` is the
previous export and must be refreshed from the editor before the PR/submission.

## Setup

Use Python 3.12 and the tested versions below. On macOS, LightGBM also requires
the OpenMP runtime (`brew install libomp`).

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

The data is downloaded from OpenML on first run. For the offline route, run
`make download-data` once while online to prepare `data/raw/adult.csv` from the
official UCI archive, then set `ADULT_OFFLINE=1`. This file must contain
`adult.data` followed by `adult.test`, without its comment line or any header.
The loader reorders OpenML Adult v2 to that same order and checks a dataset hash
after deduplication. A different input order is rejected rather than silently
changing the held-out people.

## Run the whole project

```bash
make reproduce    # rebuild from the three saved configurations, including a fresh CV check
make all          # also rerun the full Optuna search (30 / 30 / 40 trials)
```

| Step | Command | Main outputs |
|---|---|---|
| 1. EDA & split | `python scripts/run_eda.py` | `artifacts/plots/eda_*.png`, `artifacts/tables/eda_summary.json` |
| 3. Imbalance ablation | `python scripts/imbalance_ablation.py` | `artifacts/tables/imbalance_ablation.csv` |
| 4/6/7. Optuna tuning + run log | `python scripts/tune.py` | `artifacts/runs/trials.csv`, `best_params_*.json`, `optuna_history_*.png` |
| 4/5. Refit, compare, select | `python scripts/train_evaluate.py` | `artifacts/models/*.joblib`, `artifacts/tables/model_comparison.csv` |
| 9. Calibration | `python scripts/calibration.py` | `artifacts/plots/calibration_reliability.png`, `calibration.json` |
| 8. SHAP | `python scripts/explain.py` | `artifacts/plots/shap_*.png`, `shap_*.csv/json` |
| 10. Fairness | `python scripts/fairness.py` | `artifacts/plots/fairness_groups.png`, `fairness_*.csv/json` |
| Tests | `pytest -q tests` | |
| Serve | `make serve` | API on http://localhost:8000/docs |

The original tuning took about 30–40 minutes on a 2-core machine. `make reproduce`
keeps the original 100-trial log, rechecks the chosen configurations with five-fold
CV, and rebuilds all downstream results. `artifacts/runs/refit_verification.json`
records effective model parameters, versions, and dataset/split hashes. For a
full new search, `make all` appends a run to the log. `make report` exports the
tables and compiles the LaTeX source (requires `pdflatex`). Narrative text should
be checked when results change.

## Prediction endpoint

```bash
uvicorn adult_poc.api:app --app-dir src --port 8000
curl -X POST localhost:8000/predict -H "Content-Type: application/json" -d '[{
  "age": 45, "workclass": "Private", "education-num": 13, "marital-status": "Married-civ-spouse",
  "occupation": "Exec-managerial", "relationship": "Husband", "race": "White", "sex": "Male",
  "capital-gain": 0, "capital-loss": 0, "hours-per-week": 50, "native-country": "United-States"}]'
```

It returns the raw score, the isotonic-calibrated probability (or `null` before
calibration), and the decision at the raw-score validation threshold. This endpoint
serves the **baseline global-threshold policy**; the fairness post-processor is
evaluated offline. Calibration does not make a decision rule fair.

Run `make train` before serving, and `make calibrate` for calibrated probabilities.
Without a trained model, `/health` and valid prediction requests return HTTP 503.
Empty batches, unknown fields, nonfinite numbers, and invalid types/ranges return
HTTP 422. API regression tests fit a tiny local pipeline and run without model
artifacts or network access.

## Repository layout

```
config.yaml              seeds, split sizes, tuning budget, paths
src/adult_poc/
  data.py                loading (OpenML + offline fallback), dedup, 60/20/20 stratified split
  features.py            FeatureEngineer transformer + ColumnTransformer + make_pipeline
  models.py              the 3 model families and their Optuna search spaces
  tuning.py              Optuna + StratifiedKFold CV, run log writer
  metrics.py             metrics, F1-optimal threshold, ECE
  eda.py, plotting.py    figures
  api.py                 FastAPI app
scripts/                 one script per project step (see table above)
tests/                   pytest unit tests
artifacts/               plots, tables, run log (models are rebuilt by the scripts)
report/                  LaTeX source and PDF of the report
```

## Data splits

Stratified 60 / 20 / 20 (seed 42), after removing 52 exact duplicate rows:
* **train** (29,274) — Optuna tuning with 5-fold CV, then final refit;
* **validation** (9,758) — model selection, decision threshold, isotonic calibration, fairness post-processing;
* **test** (9,758) — held out from fitting, threshold tuning, calibration fitting,
  and mitigation fitting. The original descriptive EDA used the whole dataset;
  test data was therefore not completely unseen during exploration.

## Results (held-out test set)

| Model | PR-AUC | F1 | Precision | Recall |
|---|---|---|---|---|
| Logistic regression | 0.777 | 0.692 | 0.660 | 0.728 |
| Random forest | 0.815 | 0.715 | 0.650 | 0.793 |
| **LightGBM (final)** | **0.840** | **0.730** | 0.680 | 0.787 |

Calibration (isotonic): ECE 0.105 → 0.007. Fairness (sex, ThresholdOptimizer): equalized-odds gap 0.163 → 0.037.
These are corrected refits; the original 100-trial search remains unchanged.
Full analysis in `report/report.pdf`.
