PY ?= python

.PHONY: all eda ablation tune train explain calibrate fairness test serve report

all: eda ablation tune train calibrate explain fairness test

eda:        ; $(PY) scripts/run_eda.py
ablation:   ; $(PY) scripts/imbalance_ablation.py
tune:       ; $(PY) scripts/tune.py
train:      ; $(PY) scripts/train_evaluate.py
calibrate:  ; $(PY) scripts/calibration.py
explain:    ; $(PY) scripts/explain.py
fairness:   ; $(PY) scripts/fairness.py
test:       ; $(PY) -m pytest -q tests
serve:      ; uvicorn adult_poc.api:app --app-dir src --port 8000
report:     ; $(PY) scripts/make_report_tables.py && cd report && pdflatex -interaction=nonstopmode report.tex && pdflatex -interaction=nonstopmode report.tex
