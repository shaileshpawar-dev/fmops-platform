"""Health endpoints.

Three distinct checks, because they answer different questions:

``/health``        overall summary for humans and dashboards
``/health/live``   is the process up? (Kubernetes liveness / ECS health check)
``/health/ready``  can it actually serve? (readiness -- fails while no model is
                   loaded, which keeps a starting container out of the load
                   balancer instead of serving 503s to users)
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Response, status

from app.core.config import get_settings
from app.core.db import get_database
from app.core.logging import get_logger
from app.deployment.manager import get_deployment_manager
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
    """Ready to serve traffic: database reachable and a servable model exists."""
    settings = get_settings()
    checks: dict[str, bool] = {}
    detail: dict[str, Any] = {}

    try:
        get_database().scalar("SELECT 1")
        checks["database"] = True
    except Exception as exc:
        checks["database"] = False
        detail["database_error"] = str(exc)

    try:
        registry = get_registry()
        serving = registry.get_serving(settings.tracking.registered_model_name)
        checks["model_available"] = serving is not None
        if serving is not None:
            detail["model_version"] = serving.version
            detail["model_stage"] = serving.stage.value
        else:
            detail["model_error"] = "no Production or Staging model version is registered"
    except Exception as exc:
        checks["model_available"] = False
        detail["model_error"] = str(exc)

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
    }

    try:
        tables = get_database().scalar(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='table'", default=0
        )
        payload["components"]["database"] = {"status": "ok", "tables": int(tables)}
    except Exception as exc:
        payload["status"] = "degraded"
        payload["components"]["database"] = {"status": "error", "detail": str(exc)}

    try:
        registry = get_registry()
        model_name = settings.tracking.registered_model_name
        serving = registry.get_serving(model_name)
        payload["components"]["registry"] = {
            "status": "ok",
            "backend": registry.backend,
            "model_name": model_name,
            "serving_version": serving.version if serving else None,
            "serving_stage": serving.stage.value if serving else None,
        }
        if serving is None:
            payload["status"] = "degraded"
    except Exception as exc:
        payload["status"] = "degraded"
        payload["components"]["registry"] = {"status": "error", "detail": str(exc)}

    try:
        manager = get_deployment_manager()
        endpoint_health = manager.health()
        deployment = manager.status()
        payload["components"]["deployment"] = {
            "status": endpoint_health.status.value,
            "endpoint": endpoint_health.endpoint_name,
            "provider": manager.provider.name,
            "checks": endpoint_health.checks,
            "current_version": deployment.current_version if deployment else None,
            "state": deployment.state.value if deployment else None,
        }
        if endpoint_health.status == HealthStatus.UNHEALTHY and deployment is not None:
            payload["status"] = "degraded"
    except Exception as exc:
        payload["components"]["deployment"] = {"status": "unknown", "detail": str(exc)}

    return payload
