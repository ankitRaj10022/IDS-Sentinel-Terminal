from __future__ import annotations

import csv
import hashlib
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer

from .config import CLASSICAL_TEST_DATA, CLASSICAL_TRAIN_DATA, DNN_TEST_DATA, DNN_TRAIN_DATA, ROOT_DIR


def _sample_frame(frame: pd.DataFrame, sample_size: int | None, random_state: int) -> pd.DataFrame:
    if sample_size is None or sample_size >= len(frame):
        return frame
    sampled, _ = train_test_split(frame, train_size=sample_size, stratify=frame.iloc[:, 0], random_state=random_state)
    return sampled.reset_index(drop=True)


def load_dataset_split(
    train_path: Path,
    test_path: Path,
    train_sample: int | None = None,
    test_sample: int | None = None,
    random_state: int = 42,
) -> dict[str, np.ndarray | int | str]:
    train_frame = pd.read_csv(train_path, header=None)
    test_frame = pd.read_csv(test_path, header=None)

    train_frame = _sample_frame(train_frame, train_sample, random_state)
    test_frame = _sample_frame(test_frame, test_sample, random_state)

    X_train = train_frame.iloc[:, 1:42].to_numpy(dtype=np.float32)
    y_train = train_frame.iloc[:, 0].to_numpy(dtype=np.int32)
    X_test = test_frame.iloc[:, 1:42].to_numpy(dtype=np.float32)
    y_test = test_frame.iloc[:, 0].to_numpy(dtype=np.int32)

    scaler = Normalizer()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "train_rows": int(X_train.shape[0]),
        "test_rows": int(X_test.shape[0]),
        "feature_count": int(X_train.shape[1]),
    }


def load_classical_split(train_sample: int | None = None, test_sample: int | None = None, random_state: int = 42) -> dict[str, np.ndarray | int]:
    return load_dataset_split(CLASSICAL_TRAIN_DATA, CLASSICAL_TEST_DATA, train_sample, test_sample, random_state)


def load_dnn_split(train_sample: int | None = None, test_sample: int | None = None, random_state: int = 42) -> dict[str, np.ndarray | int]:
    return load_dataset_split(DNN_TRAIN_DATA, DNN_TEST_DATA, train_sample, test_sample, random_state)


def _dataset_file_summary(path: Path) -> dict[str, object]:
    sha256 = hashlib.sha256()
    row_count = 0
    columns = 0
    label_counts: dict[str, int] = {}

    with path.open("rb") as raw_handle:
        for chunk in iter(lambda: raw_handle.read(1024 * 1024), b""):
            sha256.update(chunk)

    with path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if not row:
                continue
            row_count += 1
            if columns == 0:
                columns = len(row)
            label = row[0]
            label_counts[label] = label_counts.get(label, 0) + 1

    return {
        "path": _relative_path(path),
        "rows": row_count,
        "columns": columns,
        "size_mb": round(path.stat().st_size / (1024 * 1024), 2),
        "sha256": sha256.hexdigest(),
        "label_counts": label_counts,
    }


def _relative_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT_DIR))
    except ValueError:
        return str(path)


@lru_cache(maxsize=1)
def dataset_summary() -> dict[str, object]:
    classical_train = _dataset_file_summary(CLASSICAL_TRAIN_DATA)
    classical_test = _dataset_file_summary(CLASSICAL_TEST_DATA)
    dnn_train = _dataset_file_summary(DNN_TRAIN_DATA)
    dnn_test = _dataset_file_summary(DNN_TEST_DATA)

    classical_train["path"] = _relative_path(CLASSICAL_TRAIN_DATA)
    classical_test["path"] = _relative_path(CLASSICAL_TEST_DATA)
    dnn_train["path"] = _relative_path(DNN_TRAIN_DATA)
    dnn_test["path"] = _relative_path(DNN_TEST_DATA)

    return {
        "classical_train": classical_train,
        "classical_test": classical_test,
        "dnn_train": dnn_train,
        "dnn_test": dnn_test,
        "duplicates": {
            "train_files_match": classical_train["sha256"] == dnn_train["sha256"],
            "test_files_match": classical_test["sha256"] == dnn_test["sha256"],
        },
    }
