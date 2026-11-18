"""Step 1 - EDA and problem framing."""
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from adult_poc.config import load_config  # noqa: E402
from adult_poc.data import load_raw, split  # noqa: E402
from adult_poc.eda import run_eda  # noqa: E402

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    cfg = load_config()
    X, y = load_raw(cfg)
    # Retain whole-data descriptive figures from the initial experiment. These
    # are not used for fitting; disclose their scope rather than calling test unseen.
    run_eda(X, y, cfg)
    X_tr, X_va, X_te, y_tr, y_va, y_te = split(X, y, cfg)
    for name, part in [("train", y_tr), ("val", y_va), ("test", y_te)]:
        logging.info("%-5s n=%6d  positive rate=%.4f", name, len(part), part.mean())
