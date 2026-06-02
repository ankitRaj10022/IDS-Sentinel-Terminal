from __future__ import annotations

import numpy as np


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | int]:
    y_true = np.asarray(y_true).astype(int).reshape(-1)
    y_pred = np.asarray(y_pred).astype(int).reshape(-1)
    if y_true.size != y_pred.size:
        raise ValueError(
            f"metric arrays must have same length: y_true={y_true.size}, y_pred={y_pred.size}"
        )

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    support = int(y_true.size)
    accuracy = (tp + tn) / support if support else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    )

    return {
        "accuracy": round(float(accuracy), 6),
        "precision": round(float(precision), 6),
        "recall": round(float(recall), 6),
        "f1": round(float(f1), 6),
        "support": support,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def probability_summary(values: np.ndarray | None) -> dict[str, float] | None:
    if values is None:
        return None
    flattened = np.asarray(values, dtype=float).reshape(-1)
    return {
        "min": round(float(flattened.min()), 6),
        "mean": round(float(flattened.mean()), 6),
        "max": round(float(flattened.max()), 6),
    }
