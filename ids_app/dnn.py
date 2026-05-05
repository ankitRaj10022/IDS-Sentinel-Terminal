from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from uuid import uuid4

import numpy as np

from .config import DNN_LAYER_SPECS, DNN_PROFILES
from .data import load_dnn_split
from .metrics import binary_metrics, probability_summary
from .storage import run_dir, run_summary_path, write_json


def _tensorflow():
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    import tensorflow as tf

    return tf


def resolve_dnn_config(config: dict[str, object] | None) -> dict[str, object]:
    payload = dict(config or {})
    profile = str(payload.get("profile", "fast"))
    if profile not in DNN_PROFILES:
        profile = "fast"
    merged = dict(DNN_PROFILES[profile])
    merged["profile"] = profile
    if payload.get("architectures"):
        selected = [int(value) for value in payload["architectures"] if int(value) in DNN_LAYER_SPECS]
        if selected:
            merged["architectures"] = selected
    for key in ("train_sample", "test_sample", "epochs", "batch_size", "random_state"):
        if key in payload:
            merged[key] = payload.get(key)
    merged.setdefault("random_state", 42)
    return merged


def _build_model(layer_count: int, dropout_rate: float = 0.01):
    tf = _tensorflow()

    tf.keras.backend.clear_session()
    model = tf.keras.Sequential()
    for index, units in enumerate(DNN_LAYER_SPECS[layer_count]):
        kwargs = {"input_dim": 41} if index == 0 else {}
        model.add(tf.keras.layers.Dense(units, activation="relu", **kwargs))
        model.add(tf.keras.layers.Dropout(dropout_rate))
    model.add(tf.keras.layers.Dense(1, activation="sigmoid"))
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model


def train_dnn_suite(job_id: str, config: dict[str, object] | None = None) -> dict[str, object]:
    tf = _tensorflow()

    resolved = resolve_dnn_config(config)
    tf.random.set_seed(int(resolved.get("random_state", 42)))
    np.random.seed(int(resolved.get("random_state", 42)))

    split = load_dnn_split(
        train_sample=resolved.get("train_sample"),
        test_sample=resolved.get("test_sample"),
        random_state=int(resolved.get("random_state", 42)),
    )

    generated_run_id = f"dnn-{uuid4().hex[:12]}"
    target_dir = run_dir(generated_run_id)
    models_dir = target_dir / "models"
    history_dir = target_dir / "history"
    predictions_dir = target_dir / "predictions"
    models_dir.mkdir(parents=True, exist_ok=True)
    history_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)

    y_test = split["y_test"]
    results = []

    for layer_count in resolved["architectures"]:
        model = _build_model(layer_count)
        checkpoint_path = models_dir / f"dnn{layer_count}_best.keras"
        csv_logger_path = history_dir / f"dnn{layer_count}_history.csv"

        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(filepath=str(checkpoint_path), save_best_only=True, monitor="loss", verbose=0),
            tf.keras.callbacks.CSVLogger(str(csv_logger_path), separator=",", append=False),
        ]

        started = time.perf_counter()
        history = model.fit(
            split["X_train"],
            split["y_train"],
            batch_size=int(resolved["batch_size"]),
            epochs=int(resolved["epochs"]),
            verbose=0,
            callbacks=callbacks,
        )
        training_seconds = round(time.perf_counter() - started, 3)
        best_model = tf.keras.models.load_model(checkpoint_path, compile=False)
        probabilities = best_model.predict(split["X_test"], verbose=0).reshape(-1)
        predicted = (probabilities >= 0.5).astype(int)
        metrics = binary_metrics(y_test, predicted)

        model_path = models_dir / f"dnn{layer_count}.keras"
        best_model.save(model_path)
        np.savetxt(predictions_dir / f"dnn{layer_count}_labels.txt", predicted, fmt="%01d")
        np.savetxt(predictions_dir / f"dnn{layer_count}_probabilities.txt", probabilities)

        results.append(
            {
                "id": f"dnn_{layer_count}_layer",
                "label": f"DNN {layer_count} Layer",
                "training_seconds": training_seconds,
                "metrics": metrics,
                "probability_summary": probability_summary(probabilities),
                "history": {
                    "epochs": int(len(history.history["loss"])),
                    "best_accuracy": round(float(max(history.history["accuracy"])), 6),
                    "best_loss": round(float(min(history.history["loss"])), 6),
                    "final_accuracy": round(float(history.history["accuracy"][-1]), 6),
                    "final_loss": round(float(history.history["loss"][-1]), 6),
                },
                "model_path": str(model_path.relative_to(target_dir.parent.parent)),
            }
        )

    results.sort(key=lambda item: item["metrics"]["f1"], reverse=True)
    summary = {
        "run_id": generated_run_id,
        "job_id": job_id,
        "kind": "dnn_train",
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
