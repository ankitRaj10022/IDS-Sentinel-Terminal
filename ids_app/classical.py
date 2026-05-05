from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import joblib
import numpy as np
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from .config import CLASSICAL_MODEL_LABELS, CLASSICAL_PROFILES
from .data import load_classical_split
from .metrics import binary_metrics, probability_summary
from .storage import run_dir, run_summary_path, write_json


MODEL_BUILDERS = {
    "logistic_regression": lambda: LogisticRegression(max_iter=400, solver="lbfgs"),
    "naive_bayes": lambda: GaussianNB(),
    "decision_tree": lambda: DecisionTreeClassifier(random_state=42),
    "adaboost": lambda: AdaBoostClassifier(n_estimators=100, random_state=42, algorithm="SAMME"),
    "random_forest": lambda: RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42),
    "knn": lambda: KNeighborsClassifier(n_neighbors=5, weights="distance", n_jobs=-1),
}


def resolve_classical_config(config: dict[str, object] | None) -> dict[str, object]:
    payload = dict(config or {})
    profile = str(payload.get("profile", "fast"))
    if profile not in CLASSICAL_PROFILES:
        profile = "fast"
    merged = dict(CLASSICAL_PROFILES[profile])
    merged["profile"] = profile
    if payload.get("models"):
        selected = [name for name in payload["models"] if name in MODEL_BUILDERS]
        if selected:
            merged["models"] = selected
    for key in ("train_sample", "test_sample", "random_state"):
        if key in payload:
            merged[key] = payload.get(key)
    merged.setdefault("random_state", 42)
    return merged


def train_classical_suite(job_id: str, config: dict[str, object] | None = None) -> dict[str, object]:
    resolved = resolve_classical_config(config)
    split = load_classical_split(
        train_sample=resolved.get("train_sample"),
        test_sample=resolved.get("test_sample"),
        random_state=int(resolved.get("random_state", 42)),
    )

    generated_run_id = f"classical-{uuid4().hex[:12]}"
    target_dir = run_dir(generated_run_id)
    models_dir = target_dir / "models"
    predictions_dir = target_dir / "predictions"
    models_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)

    y_test = split["y_test"]
    results = []

    for model_name in resolved["models"]:
        builder = MODEL_BUILDERS[model_name]
        estimator = builder()
        started = time.perf_counter()
        estimator.fit(split["X_train"], split["y_train"])
        training_seconds = round(time.perf_counter() - started, 3)

        predicted = estimator.predict(split["X_test"]).astype(int)
        probabilities = None
        if hasattr(estimator, "predict_proba"):
            probabilities = estimator.predict_proba(split["X_test"])[:, 1]

        metrics = binary_metrics(y_test, predicted)
        joblib.dump(estimator, models_dir / f"{model_name}.joblib")
        np.savetxt(predictions_dir / f"{model_name}_labels.txt", predicted, fmt="%01d")
        if probabilities is not None:
            np.savetxt(predictions_dir / f"{model_name}_probabilities.txt", probabilities)

        results.append(
            {
                "id": model_name,
                "label": CLASSICAL_MODEL_LABELS[model_name],
                "training_seconds": training_seconds,
                "metrics": metrics,
                "probability_summary": probability_summary(probabilities),
                "model_path": str((models_dir / f"{model_name}.joblib").relative_to(target_dir.parent.parent)),
            }
        )

    results.sort(key=lambda item: item["metrics"]["f1"], reverse=True)
    summary = {
        "run_id": generated_run_id,
        "job_id": job_id,
        "kind": "classical_train",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": resolved,
        "dataset": {
            "train_rows": split["train_rows"],
            "test_rows": split["test_rows"],
            "feature_count": split["feature_count"],
        },
        "results": results,
        "best_model": results[0]["id"] if results else None,
    }
    write_json(run_summary_path(generated_run_id), summary)
    return summary
