"""Background jobs: list, inspect, read logs, cancel, retry."""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Query, Request

from app.api.security import request_actor
from app.core.audit import audit
from app.core.exceptions import FMOpsError
from app.jobs.runner import get_job_runner
from app.jobs.store import CANCELLED, FAILED, RUNNING, TERMINAL, get_job_store

router = APIRouter(prefix="/api/v1/jobs", tags=["jobs"])
automation_router = APIRouter(prefix="/api/v1/automation", tags=["jobs"])


class JobNotFoundError(FMOpsError):
    code = "job_not_found"
    http_status = 404


class JobStateError(FMOpsError):
    code = "job_state_conflict"
    http_status = 409


def _job(job_id: str) -> dict[str, Any]:
    job = get_job_store().get(job_id)
    if job is None:
        raise JobNotFoundError(f"no job with id {job_id!r}", job_id=job_id)
    return job


@router.get("", summary="List jobs")
def list_jobs(
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    status: str | None = None,
    kind: str | None = None,
    model: str | None = None,
) -> dict[str, Any]:
    store = get_job_store()
    jobs = store.list(limit=limit, status=status, kind=kind, model_name=model)
    return {"counts": store.counts(), "count": len(jobs), "jobs": jobs}


@router.get("/{job_id}", summary="Get one job")
def get_job(job_id: str) -> dict[str, Any]:
    return _job(job_id)


@router.get("/{job_id}/logs", summary="Log lines captured while the job ran")
def job_logs(
    job_id: str,
    after: Annotated[int, Query(ge=0, description="Return lines after this id.")] = 0,
    limit: Annotated[int, Query(ge=1, le=1000)] = 500,
) -> dict[str, Any]:
    """Polling with ``after`` set to the last id seen streams new lines only."""
    job = _job(job_id)
    lines = get_job_store().logs(job_id, after_id=after, limit=limit)
    return {
        "job_id": job_id,
        "status": job["status"],
        "finished": job["status"] in TERMINAL,
        "lines": lines,
        "next_after": lines[-1]["id"] if lines else after,
    }


@router.post("/{job_id}/cancel", summary="Cancel a queued or running job")
def cancel_job(job_id: str, request: Request) -> dict[str, Any]:
    """A queued job is cancelled at once. A running job stops at its next
    checkpoint -- between AutoML candidates, between canary steps, around the
    training fit -- and a canary that is interrupted restores the live version."""
    job = _job(job_id)
    if job["status"] in TERMINAL:
        raise JobStateError(f"job {job_id} is already {job['status']}", status=job["status"])
    updated = get_job_store().request_cancel(job_id)
    audit("job.cancel", "job", job_id, kind=job["kind"])
    return {
        "job": updated,
        "detail": (
            "cancelled"
            if updated and updated["status"] == CANCELLED
            else (
                "cancellation requested; the job stops at its next checkpoint"
                if updated and updated["status"] == RUNNING
                else f"job is {updated['status'] if updated else 'gone'}"
            )
        ),
    }


@router.post("/{job_id}/retry", summary="Retry a failed or cancelled job", status_code=202)
def retry_job(job_id: str, request: Request) -> dict[str, Any]:
    """Queues a new job with the same parameters. The failed one is kept as-is."""
    job = _job(job_id)
    if job["status"] not in (FAILED, CANCELLED):
        raise JobStateError(
            f"only failed or cancelled jobs can be retried; this one is {job['status']}",
            status=job["status"],
        )
    new = get_job_runner().retry(job_id, requested_by=request_actor(request))
    audit("job.retry", "job", new["id"], retry_of=job_id, kind=job["kind"])
    return {"job": new, "retry_of": job_id}


@automation_router.get("", summary="Scheduled automation: settings and last pass")
def automation_status() -> dict[str, Any]:
    from app.core.config import get_settings
    from app.core.db import get_database
    from app.jobs.automation import _KEY

    settings = get_settings()
    last = get_database().scalar("SELECT value FROM schema_meta WHERE key = ?", (_KEY,), None)
    return {
        "enabled": settings.automation.enabled,
        "interval_minutes": settings.automation.interval_minutes,
        "last_tick_at": last,
        "does": [
            "queue a drift scan per serving model once drift.min_samples new predictions "
            "have arrived since its last scan",
            "evaluate the retraining trigger per model and queue a retraining job when it "
            "fires (never within the cooldown, never without new labelled data)",
            "run the latency/error-rate SLO watchdog on deployed endpoints",
        ],
    }


@automation_router.post("/run", summary="Run the automation pass now")
def automation_run() -> dict[str, Any]:
    """Runs the same pass the scheduler runs. It only queues jobs, so it returns
    at once; the jobs it queued are listed in the response."""
    from app.jobs.automation import run_tick

    return run_tick()
