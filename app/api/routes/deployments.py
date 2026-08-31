"""Deployment endpoints."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Body, Query

from app.core.config import get_settings
from app.core.logging import get_logger
from app.deployment.manager import get_deployment_manager
from app.monitoring.metrics import set_deployment_metrics
from app.schemas.deployment import (
    Deployment,
    DeploymentRequest,
    DeploymentResult,
    EndpointHealth,
    RollbackRequest,
    RollbackResult,
)

logger = get_logger(__name__)
router = APIRouter(prefix="/api/v1/deployments", tags=["deployments"])


@router.get("", response_model=list[Deployment], summary="List deployments")
def list_deployments(
    endpoint: str | None = Query(default=None), limit: int = Query(default=50, le=200)
) -> list[Deployment]:
    return get_deployment_manager().list(endpoint, limit)


@router.get("/current", summary="Active deployment for an endpoint")
def current(endpoint: str | None = None) -> dict[str, Any]:
    manager = get_deployment_manager()
    deployment = manager.status(endpoint)
    if deployment is None:
        return {
            "endpoint": endpoint or get_settings().deployment.endpoint_name,
            "deployment": None,
            "detail": "no deployment has been created for this endpoint",
        }
    set_deployment_metrics(
        deployment.endpoint_name,
        deployment.model_name,
        deployment.traffic,
        deployment.state.value,
    )
    return {
        "endpoint": deployment.endpoint_name,
        "deployment": deployment.model_dump(mode="json"),
        "provider_status": manager.provider_status(deployment.endpoint_name),
        "versions": {
            "current": deployment.current_version,
            "previous": deployment.previous_version,
            "candidate": deployment.candidate_version,
            "shadow": deployment.shadow_version,
        },
    }


@router.get("/health", response_model=EndpointHealth, summary="Endpoint health")
def health(endpoint: str | None = None) -> EndpointHealth:
    return get_deployment_manager().health(endpoint)


@router.get("/{deployment_id}", response_model=Deployment, summary="Get one deployment")
def get_deployment(deployment_id: str) -> Deployment:
    return get_deployment_manager().store.get(deployment_id)


@router.post("", response_model=DeploymentResult, summary="Deploy a model version")
def deploy(
    payload: DeploymentRequest,
    force: bool = Query(
        default=False,
        description=(
            "Deploy a version that has not cleared the approval gate. Recorded "
            "in the audit log."
        ),
    ),
) -> DeploymentResult:
    """Roll a model version out using the requested (or configured) strategy.

    Refuses versions that are not in a deployable stage unless ``force=true``.
    """
    return get_deployment_manager().deploy(payload, force=force, actor="api")


@router.post("/rollback", response_model=RollbackResult, summary="Roll back an endpoint")
def rollback(payload: RollbackRequest = Body(default=RollbackRequest())) -> RollbackResult:
    """Restore the previous version.

    Target selection: the explicit ``to_version`` if given, else the recorded
    previous version, else the registry's most recently archived production
    version. If none exist the call fails rather than guessing.
    """
    return get_deployment_manager().rollback(
        endpoint_name=payload.endpoint_name,
        to_version=payload.to_version,
        reason=payload.reason,
        actor="api",
    )


@router.post("/terminate", summary="Tear down an endpoint")
def terminate(endpoint: str | None = None) -> dict[str, str]:
    manager = get_deployment_manager()
    manager.terminate(endpoint)
    return {
        "status": "terminated",
        "endpoint": endpoint or get_settings().deployment.endpoint_name,
    }
