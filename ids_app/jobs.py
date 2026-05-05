from __future__ import annotations

import threading
import traceback
from datetime import datetime, timezone
from typing import Any, Callable
from uuid import uuid4

from .storage import ensure_directories, job_path, read_json, write_json


JobTask = Callable[[str, dict[str, Any]], dict[str, Any]]


class JobManager:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._jobs: dict[str, dict[str, Any]] = {}
        ensure_directories()
        self._load_existing_jobs()

    def _load_existing_jobs(self) -> None:
        for path in job_path("").parent.glob("*.json"):
            payload = read_json(path)
            if payload:
                self._jobs[payload["id"]] = payload

    def _persist(self, payload: dict[str, Any]) -> None:
        write_json(job_path(payload["id"]), payload)

    def _update(self, job_id: str, **fields: Any) -> None:
        with self._lock:
            payload = dict(self._jobs[job_id])
            payload.update(fields)
            self._jobs[job_id] = payload
            self._persist(payload)

    def submit(self, kind: str, config: dict[str, Any], task: JobTask) -> dict[str, Any]:
        created = {
            "id": uuid4().hex,
            "kind": kind,
            "status": "queued",
            "config": config,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        with self._lock:
            self._jobs[created["id"]] = created
            self._persist(created)

        thread = threading.Thread(target=self._run_job, args=(created["id"], task), daemon=True)
        thread.start()
        return created

    def _run_job(self, job_id: str, task: JobTask) -> None:
        payload = self._jobs[job_id]
        self._update(job_id, status="running", started_at=datetime.now(timezone.utc).isoformat())
        try:
            result = task(job_id, payload["config"])
            self._update(
                job_id,
                status="completed",
                completed_at=datetime.now(timezone.utc).isoformat(),
                run_id=result.get("run_id"),
                summary=result,
            )
        except Exception as exc:  # pragma: no cover - surfaced to UI
            self._update(
                job_id,
                status="failed",
                completed_at=datetime.now(timezone.utc).isoformat(),
                error=str(exc),
                traceback=traceback.format_exc(),
            )

    def get(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            payload = self._jobs.get(job_id)
            return dict(payload) if payload else None

    def list(self, limit: int = 20) -> list[dict[str, Any]]:
        with self._lock:
            jobs = sorted(self._jobs.values(), key=lambda item: item.get("created_at", ""), reverse=True)
            return [dict(job) for job in jobs[:limit]]


job_manager = JobManager()

