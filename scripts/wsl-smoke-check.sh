#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="${1:-classical}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${ROOT_DIR}/.venv-wsl"

usage() {
  echo "Usage: bash scripts/wsl-smoke-check.sh [classical|dnn]"
}

if [[ "${MODE}" != "classical" && "${MODE}" != "dnn" ]]; then
  usage
  exit 1
fi

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "Python interpreter not found: ${PYTHON_BIN}"
  exit 1
fi

cd "${ROOT_DIR}"

"${PYTHON_BIN}" - <<'PY'
import csv
import py_compile
from pathlib import Path

ignored_parts = {".git", ".idea", ".ipynb_checkpoints", ".vs", ".vscode", ".venv-wsl"}
ignored_names = {"tempCodeRunnerFile.py"}
required_paths = [
    Path("all.py"),
    Path("temp1.py"),
    Path("classical/accuclassical.py"),
    Path("dnn/dnn1.py"),
    Path("dnn/dnn5.py"),
    Path("kddtrain.csv"),
    Path("kddtest.csv"),
    Path("dnn/kdd/binary/Training.csv"),
    Path("dnn/kdd/binary/Testing.csv"),
]
csv_paths = [
    Path("kddtrain.csv"),
    Path("kddtest.csv"),
    Path("dnn/kdd/binary/Training.csv"),
    Path("dnn/kdd/binary/Testing.csv"),
]

failures = []

for path in Path(".").rglob("*.py"):
    if any(part in ignored_parts for part in path.parts):
        continue
    if path.name in ignored_names:
        continue
    try:
        py_compile.compile(str(path), doraise=True)
    except Exception as exc:
        failures.append(f"{path}: {exc}")

missing = [str(path) for path in required_paths if not path.exists()]
if missing:
    failures.append(f"Missing required files: {missing}")

for path in csv_paths:
    with path.open(newline="", encoding="utf-8", errors="ignore") as handle:
        row = next(csv.reader(handle))
    if len(row) < 42:
        failures.append(f"{path} has too few columns: expected at least 42, got {len(row)}")

if failures:
    raise SystemExit("\n".join(failures))

print("Source files and datasets passed the base checks.")
PY

if [[ ! -d "${VENV_DIR}" ]]; then
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

python -m pip install --upgrade pip

if [[ "${MODE}" == "dnn" ]]; then
  python - <<'PY'
import sys

if sys.version_info[:2] not in {(3, 10), (3, 11)}:
    raise SystemExit(
        "DNN mode requires Python 3.10 or 3.11 because tensorflow==2.15.1 "
        f"does not support Python {sys.version_info.major}.{sys.version_info.minor}."
    )
PY

  pip install -r requirements-dnn.txt

  python - <<'PY'
import h5py
import keras
import pandas as pd

train = pd.read_csv("dnn/kdd/binary/Training.csv", header=None, nrows=16)
test = pd.read_csv("dnn/kdd/binary/Testing.csv", header=None, nrows=16)

if train.shape[1] < 42 or test.shape[1] < 42:
    raise SystemExit(f"Unexpected DNN CSV shapes: train={train.shape}, test={test.shape}")

print(f"keras={keras.__version__}")
print(f"h5py={h5py.__version__}")
print(f"train_shape={train.shape}")
print(f"test_shape={test.shape}")
PY
else
  pip install -r requirements-classical.txt

  python - <<'PY'
import numpy as np
import pandas as pd
import sklearn

train = pd.read_csv("kddtrain.csv", header=None, nrows=16)
test = pd.read_csv("kddtest.csv", header=None, nrows=16)

if train.shape[1] < 42 or test.shape[1] < 42:
    raise SystemExit(f"Unexpected CSV shapes: train={train.shape}, test={test.shape}")

print(f"numpy={np.__version__}")
print(f"pandas={pd.__version__}")
print(f"scikit-learn={sklearn.__version__}")
print(f"train_shape={train.shape}")
print(f"test_shape={test.shape}")
PY
fi
