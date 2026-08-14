from __future__ import annotations

import csv
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from .config import LEGACY_CLASSICAL_FILES, LEGACY_DNN_FILES, ROOT_DIR
from .metrics import binary_metrics


def _load_labels(path: Path) -> np.ndarray:
    return np.loadtxt(path).astype(int).reshape(-1)


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _load_history(path: Path) -> dict[str, float | int] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return None

    accuracy_values = [
        value
        for value in (_safe_float(row.get("accuracy")) for row in rows)
        if value is not None
    ]
    loss_values = [
        value
        for value in (_safe_float(row.get("loss")) for row in rows)
        if value is not None
    ]
    final_accuracy = _safe_float(rows[-1].get("accuracy"))
    final_loss = _safe_float(rows[-1].get("loss"))

    if (
        not accuracy_values
        or not loss_values
        or final_accuracy is None
        or final_loss is None
    ):
        return None

    return {
        "epochs_logged": int(len(rows)),
        "best_accuracy": round(max(accuracy_values), 6),
        "best_loss": round(min(loss_values), 6),
        "final_accuracy": round(final_accuracy, 6),
        "final_loss": round(final_loss, 6),
    }


@lru_cache(maxsize=1)
def evaluate_legacy_predictions() -> dict[str, object]:
    classical_expected_path = ROOT_DIR / "classical" / "expected.txt"
    dnn_expected_path = ROOT_DIR / "dnn" / "dnnres" / "expected.txt"
    if not dnn_expected_path.exists():
        dnn_expected_path = classical_expected_path

    classical_expected = _load_labels(classical_expected_path)
    dnn_expected = _load_labels(dnn_expected_path)

    classical_results = []
    for slug, (label, path) in LEGACY_CLASSICAL_FILES.items():
        if not path.exists():
            continue
        metrics = binary_metrics(classical_expected, _load_labels(path))
        classical_results.append(
            {
                "id": slug,
                "label": label,
                "source": str(path.relative_to(ROOT_DIR)),
                "metrics": metrics,
            }
        )

    dnn_results = []
    for slug, (label, path) in LEGACY_DNN_FILES.items():
        if not path.exists():
            continue
        metrics = binary_metrics(dnn_expected, _load_labels(path))
        dnn_results.append(
            {
                "id": slug,
                "label": label,
                "source": str(path.relative_to(ROOT_DIR)),
                "metrics": metrics,
            }
        )

    history_map = {
        "legacy_dnn1": ROOT_DIR
        / "dnn"
        / "kddresults"
        / "dnn1layer"
        / "training_set_dnnanalysis.csv",
        "legacy_dnn2": ROOT_DIR
        / "dnn"
        / "kddresults"
        / "dnn2layer"
        / "training_set_dnnanalysis.csv",
        "legacy_dnn3": ROOT_DIR
        / "dnn"
        / "kddresults"
        / "dnn3layer"
        / "training_set_dnnanalysis.csv",
        "legacy_dnn4": ROOT_DIR
        / "dnn"
        / "kddresults"
        / "dnn4layer"
        / "training_set_dnnanalysis.csv",
        "legacy_dnn5": ROOT_DIR
        / "dnn"
        / "kddresults"
        / "dnn5layer"
        / "training_set_dnnanalysis.csv",
    }
    for result in dnn_results:
        result["history"] = _load_history(history_map[result["id"]])

    classical_results.sort(key=lambda item: item["metrics"]["f1"], reverse=True)
    dnn_results.sort(key=lambda item: item["metrics"]["f1"], reverse=True)

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "classical": classical_results,
        "dnn": dnn_results,
    }
