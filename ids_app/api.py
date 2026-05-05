from __future__ import annotations

from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .classical import train_classical_suite
from .config import CLASSICAL_PROFILES, DNN_PROFILES, FRONTEND_DIR
from .data import dataset_summary
from .dnn import train_dnn_suite
from .jobs import job_manager
from .legacy import evaluate_legacy_predictions
from .storage import ensure_directories, list_run_summaries, read_json, run_summary_path


class ClassicalJobRequest(BaseModel):
    profile: Literal["fast", "balanced", "full"] = "fast"
    models: list[str] | None = None
    train_sample: int | None = Field(default=None, ge=1000)
    test_sample: int | None = Field(default=None, ge=1000)
    random_state: int = 42


class DnnJobRequest(BaseModel):
    profile: Literal["fast", "balanced", "full"] = "fast"
    architectures: list[int] | None = None
    epochs: int | None = Field(default=None, ge=1, le=100)
    batch_size: int | None = Field(default=None, ge=16, le=512)
    train_sample: int | None = Field(default=None, ge=1000)
    test_sample: int | None = Field(default=None, ge=1000)
    random_state: int = 42


app = FastAPI(title="IDS Automation Console", version="1.0.0")
ensure_directories()
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/overview")
def overview() -> dict[str, object]:
    return {
        "datasets": dataset_summary(),
        "legacy": evaluate_legacy_predictions(),
        "jobs": job_manager.list(limit=12),
        "runs": list_run_summaries(limit=12),
        "profiles": {
            "classical": CLASSICAL_PROFILES,
            "dnn": DNN_PROFILES,
        },
    }


@app.get("/api/jobs")
def list_jobs() -> list[dict[str, object]]:
    return job_manager.list(limit=40)


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict[str, object]:
    payload = job_manager.get(job_id)
    if not payload:
        raise HTTPException(status_code=404, detail="Job not found")
    return payload


@app.get("/api/runs")
def list_runs() -> list[dict[str, object]]:
    return list_run_summaries(limit=40)


@app.get("/api/runs/{run_id}")
def get_run(run_id: str) -> dict[str, object]:
    payload = read_json(run_summary_path(run_id))
    if not payload:
        raise HTTPException(status_code=404, detail="Run not found")
    return payload


@app.post("/api/jobs/legacy-evaluation")
def launch_legacy_evaluation() -> dict[str, object]:
    return job_manager.submit("legacy_evaluation", {}, lambda _job_id, _config: {"run_id": "legacy-snapshot", "kind": "legacy_evaluation", "results": evaluate_legacy_predictions()})


@app.post("/api/jobs/classical")
def launch_classical(request: ClassicalJobRequest) -> dict[str, object]:
    return job_manager.submit("classical_train", request.model_dump(exclude_none=True), train_classical_suite)


@app.post("/api/jobs/dnn")
def launch_dnn(request: DnnJobRequest) -> dict[str, object]:
    return job_manager.submit("dnn_train", request.model_dump(exclude_none=True), train_dnn_suite)

