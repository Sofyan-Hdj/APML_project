"""Prepare the optional offline copy from the official UCI archive."""
import io
import sys
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from adult_poc.config import load_config  # noqa: E402


if __name__ == "__main__":
    path = load_config()["data"]["local_csv"]
    if path.exists():
        raise SystemExit(f"Already exists: {path}; preserving the existing data")
    url = "https://archive.ics.uci.edu/static/public/2/adult.zip"
    with urllib.request.urlopen(url, timeout=60) as response:
        archive = zipfile.ZipFile(io.BytesIO(response.read()))
    train = archive.read("adult.data").decode("utf-8").strip()
    test = "\n".join(line for line in archive.read("adult.test").decode("utf-8").splitlines()
                     if line.strip() and not line.startswith("|"))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(train + "\n" + test + "\n", encoding="utf-8")
    print(f"Saved adult.data followed by adult.test to {path}")
