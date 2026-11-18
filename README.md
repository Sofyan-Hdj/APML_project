# Adult / Census Income — supervised-learning proof-of-concept

Applied ML: Foundations, Engineering & Evaluation (MSA-DATI07-05) — Albert School.
Group: Sofyan Hadj-Ali · Martin Vallée · Henry Falcon de Longevialle · Kaled Goatache.

**Dataset:** Adult / Census Income (`fetch_openml(name="adult", version=2)`), 48,842 rows,
binary target `income > 50K` (24% positives), sensitive attribute `sex`.

End-to-end pipeline: EDA → feature engineering → class-imbalance handling → three models
(Logistic Regression, Random Forest, LightGBM) tuned with Optuna under 5-fold CV → SHAP
explanations → calibration diagnostics → fairness audit and mitigation → FastAPI endpoint.
All methodological choices and results are discussed in `report/report.pdf`.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

The data is downloaded from OpenML on first run. On a machine without access to
openml.org, place the UCI file (`adult-all.csv`, 48,842 rows, no header) at
`data/raw/adult.csv` and set `ADULT_OFFLINE=1`; the loader falls back to it automatically.

## Run the whole project

```bash
make all          # or run the scripts below in order
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

Tuning takes about 30–40 minutes on a 2-core laptop (the number of trials is set in `config.yaml`).

## Prediction endpoint

```bash
uvicorn adult_poc.api:app --app-dir src --port 8000
curl -X POST localhost:8000/predict -H "Content-Type: application/json" -d '[{
  "age": 45, "workclass": "Private", "education-num": 13, "marital-status": "Married-civ-spouse",
  "occupation": "Exec-managerial", "relationship": "Husband", "race": "White", "sex": "Male",
  "capital-gain": 0, "capital-loss": 0, "hours-per-week": 50, "native-country": "United-States"}]'
```

It returns the raw score, the isotonic-calibrated probability and the decision at the
validation-tuned threshold.

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
* **test** (9,758) — used once, for the final reported numbers.

## Results (test set, used once)

| Model | PR-AUC | F1 | Precision | Recall |
|---|---|---|---|---|
| Logistic regression | 0.777 | 0.692 | 0.660 | 0.728 |
| Random forest | 0.815 | 0.714 | 0.643 | 0.803 |
| **LightGBM (final)** | **0.840** | **0.735** | 0.709 | 0.763 |

Calibration (isotonic): ECE 0.106 → 0.006. Fairness (sex, ThresholdOptimizer): equalized-odds gap 0.157 → 0.037.
Full analysis in `report/report.pdf`.