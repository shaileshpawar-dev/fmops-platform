"""Health endpoints.

Three distinct checks, because they answer different questions:

``/health``        overall summary for humans and dashboards: every component,
                   every model, every deployed endpoint
``/health/live``   is the process up? (liveness)
``/health/ready``  should this instance receive traffic? (readiness)

Readiness is a property of the *service*, not of any one model. A platform with
no models yet must still accept uploads and training, so "no model is serving"
is reported -- per model, in ``/health`` -- but does not take the instance out
of the load balancer. What does: a database it cannot reach, or a job worker
that has died, because then work would be accepted and never done.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Response, status

from app.core.config import get_settings
from app.core.db import get_database
from app.core.logging import get_logger
from app.deployment.manager import get_deployment_manager
from app.registry.context import endpoint_for
from app.registry.factory import get_registry
from app.schemas.common import HealthStatus

logger = get_logger(__name__)
router = APIRouter(tags=["health"])


@router.get("/health/live", summary="Liveness probe")
def live() -> dict[str, str]:
    """The process is running. Never touches dependencies."""
    return {"status": "alive"}


@router.get("/health/ready", summary="Readiness probe")
def ready(response: Response) -> dict[str, Any]:
    """Ready to receive traffic: the database answers and jobs can execute."""
    checks: dict[str, bool] = {}
    detail: dict[str, Any] = {}

    try:
        get_database().scalar("SELECT 1")
        checks["database"] = True
    except Exception as exc:
        checks["database"] = False
        detail["database_error"] = str(exc)

    runner = _job_runner_state()
    checks["job_worker"] = runner["alive"]
    if not runner["alive"]:
        detail["job_worker_error"] = runner["detail"]

    try:
        models = get_registry().list_models()
        serving = [
            m["name"]
            for m in models
            if m.get("production_version") or m.get("staging_version")
        ]
        detail["models_registered"] = len(models)
        detail["models_serving"] = len(serving)
    except Exception as exc:  # informational only
        detail["models_error"] = str(exc)

    ok = all(checks.values())
    if not ok:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return {"status": "ready" if ok else "not_ready", "checks": checks, "detail": detail}


@router.get("/health", summary="Service health summary")
def health() -> dict[str, Any]:
    settings = get_settings()
    payload: dict[str, Any] = {
        "status": "healthy",
        "service": settings.service_name,
        "version": settings.version,
        "environment": settings.environment,
        "git_commit": settings.git_commit,
        "components": {},
        "models": [],
    }

    try:
        tables = get_database().scalar(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='table'", default=0
        )
        payload["components"]["database"] = {"status": "ok", "tables": int(tables)}
    except Exception as exc:
        payload["status"] = "degraded"
        payload["components"]["database"] = {"status": "error", "detail": str(exc)}

    runner = _job_runner_state()
    payload["components"]["jobs"] = {
        "status": "ok" if runner["alive"] else "error",
        **runner,
    }
    if not runner["alive"]:
        payload["status"] = "degraded"

    try:
        registry = get_registry()
        manager = get_deployment_manager()
        payload["components"]["registry"] = {"status": "ok", "backend": registry.backend}
        payload["components"]["deployment"] = {
            "status": "ok",
            "provider": manager.provider.name,
        }
        for model in registry.list_models():
            name = model["name"]
            endpoint = endpoint_for(name, settings)
            serving = registry.get_serving(name)
            deployment = manager.status(endpoint)
            entry: dict[str, Any] = {
                "name": name,
                "endpoint": endpoint,
                "serving_version": serving.version if serving else None,
                "serving_stage": serving.stage.value if serving else None,
                "deployment_state": deployment.state.value if deployment else None,
                "endpoint_health": None,
            }
            if deployment is not None:
                endpoint_health = manager.health(endpoint)
                entry["endpoint_health"] = endpoint_health.status.value
                # Only a deployed endpoint that is unhealthy degrades the
                # service; a model nobody has deployed yet is not a fault.
                if endpoint_health.status == HealthStatus.UNHEALTHY:
                    payload["status"] = "degraded"
                    payload["components"]["deployment"]["status"] = "degraded"
            payload["models"].append(entry)
    except Exception as exc:
        payload["status"] = "degraded"
        payload["components"]["registry"] = {"status": "error", "detail": str(exc)}

    return payload


def _job_runner_state() -> dict[str, Any]:
    from app.jobs.runner import get_job_runner
    from app.jobs.store import get_job_store

    runner = get_job_runner()
    inline = runner.config.inline
    alive = inline or runner.threads_alive()
    try:
        counts = get_job_store().counts()
    except Exception:
        counts = {}
    return {
        "alive": alive,
        "mode": "inline" if inline else "worker",
        "max_running": runner.config.max_running,
        "counts": counts,
        "detail": None if alive else "the job worker thread is not running",
    }
