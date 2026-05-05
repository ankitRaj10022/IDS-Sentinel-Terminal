from __future__ import annotations

from datetime import datetime, timezone
from functools import lru_cache

import numpy as np
import pandas as pd

from .config import LEGACY_CLASSICAL_FILES, LEGACY_DNN_FILES, ROOT_DIR
from .metrics import binary_metrics


def _load_labels(path) -> np.ndarray:
    return np.loadtxt(path).astype(int).reshape(-1)


def _load_history(path) -> dict[str, float | int] | None:
    if not path.exists():
        return None
    try:
        frame = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return None
    if frame.empty:
        return None
    return {
        "epochs_logged": int(len(frame)),
        "best_accuracy": round(float(frame["accuracy"].max()), 6),
        "best_loss": round(float(frame["loss"].min()), 6),
        "final_accuracy": round(float(frame["accuracy"].iloc[-1]), 6),
        "final_loss": round(float(frame["loss"].iloc[-1]), 6),
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
        "legacy_dnn1": ROOT_DIR / "dnn" / "kddresults" / "dnn1layer" / "training_set_dnnanalysis.csv",
        "legacy_dnn2": ROOT_DIR / "dnn" / "kddresults" / "dnn2layer" / "training_set_dnnanalysis.csv",
        "legacy_dnn3": ROOT_DIR / "dnn" / "kddresults" / "dnn3layer" / "training_set_dnnanalysis.csv",
        "legacy_dnn4": ROOT_DIR / "dnn" / "kddresults" / "dnn4layer" / "training_set_dnnanalysis.csv",
        "legacy_dnn5": ROOT_DIR / "dnn" / "kddresults" / "dnn5layer" / "training_set_dnnanalysis.csv",
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
