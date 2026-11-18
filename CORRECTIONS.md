# Reproducibility and API corrections

This branch starts from `3365daa` on `main`. The project still uses the same three
model families, feature engineering, isotonic calibration and equalized-odds
post-processing taught in the course. No optional appendix experiments were added.

## Changes

- Normalize OpenML Adult v2 to the original local order: `adult.data`, then
  `adult.test`. Check the normalized dataset hash before splitting.
- Provide a small downloader for the official UCI offline copy.
- Keep LightGBM's fixed `subsample_freq=1` in both CV and the final refit.
- Pin the tested Python dependencies, including the missing `imbalanced-learn`.
  Scikit-learn 1.9 uses `l1_ratio` to choose logistic-regression regularization;
  the old minimum of 1.6 silently ignored this setting without `penalty='elasticnet'`.
- Recheck the three historically selected configurations with training-only CV
  and refit them. Store current versions, parameters and data/split hashes in
  `artifacts/runs/refit_verification.json`.
- Reject empty API batches, unknown fields and nonfinite numbers with HTTP 422.
  Return HTTP 503 with a useful instruction when the model has not been trained.
- Run API tests with a tiny fitted pipeline so they do not silently skip in a
  fresh clone. Keep fairness mitigation as an offline comparison; `/health`
  identifies the served baseline decision policy.
- Refresh downstream outputs and correct the report's interpretation and
  contribution wording. Original EDA included all rows; that limitation is
  disclosed rather than describing the test data as entirely unseen.

## Preserving the original experiment

The 100-trial CSV/JSON logs, best-parameter files and optimization plots are
unchanged. Appendix A describes that historical search. The model comparison now
uses a current CV check and refit. Earlier figures, tables and predictions remain
available at the base commit; refreshed outputs do not claim to be the old run.
In particular, the initial Random Forest predictions could not be reproduced
exactly, and their original environment has not been recovered.

Run `make reproduce` to rebuild from the saved configurations. `make all` also
performs a new full tuning search and appends a new trial run. For an offline run,
prepare the data with `make download-data`, then set `ADULT_OFFLINE=1`.

## Separate branch still requiring review

`Henry-SHAP-and-calibration` is not merged here. Before a future merge, its
calibration-method selection must stop using test Brier scores, its quantile ECE
must handle constant scores correctly, and its SHAP model-name check should use
`random_forest`. The single isotonic method on this branch has no test-based
method selection.

## Validation

- Clean Python 3.12 installation and `pip check`: passed.
- `make reproduce`: passed, including all result-generating stages and 14 tests
  with no skipped tests. A macOS OpenMP library was supplied through the test
  command; normal macOS setup is documented as `brew install libomp`.
- Actual cached OpenML and downloaded UCI inputs: identical dataset and split hashes.
- One-trial tuning/logging smoke check for each model: passed in a temporary output folder.
- LaTeX source: compiled successfully with the desktop editor using the repository as project root.

The full historical 100-trial search was not rerun. The tracked `report/report.pdf`
is still the previous export; export the corrected editor preview before the
later PR/submission. The built-in compiler verifies the preview but does not
export a replacement PDF.
