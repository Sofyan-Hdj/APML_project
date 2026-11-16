"""Step 4/6/7 - Optuna tuning of the three models with 5-fold CV + run log.

Usage: python scripts/tune.py [logreg random_forest lightgbm]
"""
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import optuna  # noqa: E402

from adult_poc.config import load_config  # noqa: E402
from adult_poc.data import load_raw, split  # noqa: E402
from adult_poc.models import MODEL_NAMES  # noqa: E402
from adult_poc.tuning import run_tuning  # noqa: E402

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    optuna.logging.set_verbosity(optuna.logging.INFO)
    cfg = load_config()
    X, y = load_raw(cfg)
    X_train, _, _, y_train, _, _ = split(X, y, cfg)
    run_id = time.strftime("%Y%m%d-%H%M%S")
    for name in sys.argv[1:] or MODEL_NAMES:
        run_tuning(name, X_train, y_train, cfg, run_id=run_id)
