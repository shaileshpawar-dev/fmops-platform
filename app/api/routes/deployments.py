"""Deployment endpoints.

Deploying runs as a background job: a canary spends minutes observing traffic
between steps, which no HTTP request should be held open for (the load balancer
in front of the API times out at 60 s). Everything that can refuse a deployment
is checked first, so a refusal is an immediate 409 rather than a failed job.

Rollback is synchronous: it restores a version that is already loaded, takes
seconds, and is exactly when a caller wants the answer in the response.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Body, Query, Request

from app.api.security import request_actor
from app.core.logging import get_logger
from app.deployment.manager import default_endpoint, get_deployment_manager
from app.jobs.runner import get_job_runner
from app.monitoring.metrics import set_deployment_metrics
from app.registry.context import endpoint_for
from app.schemas.deployment import (
    Deployment,
    DeploymentRequest,
    EndpointHealth,
    RollbackRequest,
    RollbackResult,
)

logger = get_logger(__name__)
router = APIRouter(prefix="/api/v1/deployments", tags=["deployments"])


def _endpoint(endpoint: str | None, model: str | None) -> str:
    if endpoint:
        return endpoint
    manager = get_deployment_manager()
    return (
        endpoint_for(model, manager.settings) if model else default_endpoint(manager.settings)
    )


@router.get("", response_model=list[Deployment], summary="List deployments")
def list_deployments(
    endpoint: str | None = Query(default=None),
    model: str | None = Query(default=None),
    limit: int = Query(default=50, le=200),
) -> list[Deployment]:
    target = endpoint or (endpoint_for(model) if model else None)
    return get_deployment_manager().list(target, limit)


@router.get("/current", summary="Active deployment for a model's endpoint")
def current(endpoint: str | None = None, model: str | None = None) -> dict[str, Any]:
    manager = get_deployment_manager()
    target = _endpoint(endpoint, model)
    deployment = manager.status(target)
    if deployment is None:
        return {
            "endpoint": target,
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
def health(endpoint: str | None = None, model: str | None = None) -> EndpointHealth:
    return get_deployment_manager().health(_endpoint(endpoint, model))


@router.get("/{deployment_id}", response_model=Deployment, summary="Get one deployment")
def get_deployment(deployment_id: str) -> Deployment:
    return get_deployment_manager().store.get(deployment_id)


@router.post("", status_code=202, summary="Deploy a model version")
def deploy(payload: DeploymentRequest, request: Request) -> dict[str, Any]:
    """Queue a rollout of an approved version onto its model's endpoint.

    Refused at once (409) when the version has not been approved into Staging
    or Production, when ``endpoint_name`` names another model's endpoint, or
    when a canary/shadow would mix two versions that take different inputs.
    There is no override: a version reaches a deployable stage only through the
    approval gate.
    """
    manager = get_deployment_manager()
    model_name, endpoint, strategy = manager.validate_request(payload)
    actor = request_actor(request)
    job = get_job_runner().submit(
        "deployment",
        {
            "request": payload.model_copy(update={"model_name": model_name}).model_dump(
                mode="json"
            ),
            "actor": actor,
        },
        model_name=model_name,
        requested_by=actor,
    )
    return {
        "job": job,
        "model_name": model_name,
        "endpoint": endpoint,
        "strategy": strategy.value,
        "poll": f"/api/v1/jobs/{job['id']}",
    }


@router.post("/rollback", response_model=RollbackResult, summary="Roll back an endpoint")
def rollback(
    request: Request, payload: RollbackRequest = Body(default=RollbackRequest())
) -> RollbackResult:
    """Restore the previous version of one model's endpoint.

    Target selection: the explicit ``to_version`` if given, else the recorded
    previous version, else the registry's most recently archived production
    version. If none exist the call fails rather than guessing.
    """
    return get_deployment_manager().rollback(
        endpoint_name=_endpoint(payload.endpoint_name, payload.model_name),
        to_version=payload.to_version,
        reason=payload.reason,
        actor=request_actor(request),
    )


@router.post("/terminate", summary="Tear down an endpoint")
def terminate(endpoint: str | None = None, model: str | None = None) -> dict[str, str]:
    target = _endpoint(endpoint, model)
    get_deployment_manager().terminate(target)
    return {"status": "terminated", "endpoint": target}
