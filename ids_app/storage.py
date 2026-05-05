from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import AUTOMATION_DIR, JOBS_DIR, LEGACY_DIR, RUNS_DIR


def ensure_directories() -> None:
    for path in (AUTOMATION_DIR, JOBS_DIR, RUNS_DIR, LEGACY_DIR):
        path.mkdir(parents=True, exist_ok=True)


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=False)


def job_path(job_id: str) -> Path:
    return JOBS_DIR / f"{job_id}.json"


def run_dir(run_id: str) -> Path:
    return RUNS_DIR / run_id


def run_summary_path(run_id: str) -> Path:
    return run_dir(run_id) / "summary.json"


def list_run_summaries(limit: int = 20) -> list[dict[str, Any]]:
    ensure_directories()
    summaries: list[dict[str, Any]] = []
    for path in sorted(RUNS_DIR.glob("*/summary.json"), key=lambda item: item.stat().st_mtime, reverse=True):
        payload = read_json(path)
        if payload:
            summaries.append(payload)
        if len(summaries) >= limit:
            break
    return summaries

