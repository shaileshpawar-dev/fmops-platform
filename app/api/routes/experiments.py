"""Experiment tracking endpoints."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Query

from app.core.config import get_settings
from app.core.logging import get_logger
from app.data.versioning import get_dataset_registry
from app.tracking.factory import build_tracker

logger = get_logger(__name__)
router = APIRouter(prefix="/api/v1", tags=["experiments"])


@router.get("/experiments", summary="List experiments")
def list_experiments() -> dict[str, Any]:
    tracker = build_tracker()
    return {"backend": tracker.backend, "experiments": tracker.list_experiments()}


@router.get("/experiments/runs", summary="List runs in an experiment")
def list_runs(
    experiment: str | None = Query(default=None),
    limit: int = Query(default=25, le=200),
) -> dict[str, Any]:
    tracker = build_tracker()
    runs = tracker.search_runs(experiment, max_results=limit)
    return {
        "backend": tracker.backend,
        "experiment": experiment or get_settings().tracking.experiment_name,
        "count": len(runs),
        "runs": [
            {
                "run_id": r.run_id,
                "run_name": r.run_name,
                "status": r.status,
                "start_time": r.start_time,
                "end_time": r.end_time,
                "metrics": r.metrics,
                "params": r.params,
                "tags": r.tags,
            }
            for r in runs
        ],
    }


@router.get("/experiments/runs/{run_id}", summary="Get one run")
def get_run(run_id: str) -> dict[str, Any]:
    tracker = build_tracker()
    run = tracker.get_run(run_id)
    if run is None:
        return {"run_id": run_id, "found": False}
    return {"found": True, "run": run.__dict__}


@router.get("/datasets", summary="List dataset versions")
def list_datasets(name: str | None = None) -> dict[str, Any]:
    """Every registered dataset version, newest last.

    Each entry carries the content hash a model version points at, which is what
    makes a training run reproducible.
    """
    registry = get_dataset_registry()
    versions = registry.list_versions(name)
    return {
        "count": len(versions),
        "dvc": registry.dvc.status(),
        "versions": [v.model_dump(mode="json") for v in versions],
    }


@router.get("/datasets/{version}", summary="Get one dataset version")
def get_dataset(version: str) -> dict[str, Any]:
    return get_dataset_registry().get(version).model_dump(mode="json")
