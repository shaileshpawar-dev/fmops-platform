"""Model registry endpoints."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Body, Query

from app.core.config import get_settings
from app.core.logging import get_logger
from app.registry.factory import get_registry
from app.schemas.common import ModelStage
from app.schemas.model import ModelVersion, StageTransition
from app.training.approval import evaluate_approval
from app.training.model_factory import available_algorithms

logger = get_logger(__name__)
router = APIRouter(prefix="/api/v1/models", tags=["models"])


@router.get("", summary="List registered models")
def list_models() -> dict[str, Any]:
    registry = get_registry()
    return {"backend": registry.backend, "models": registry.list_models()}


@router.get("/algorithms", summary="Algorithms available in this environment")
def algorithms() -> dict[str, Any]:
    """Which estimators can actually be instantiated here.

    Optional backends (xgboost, lightgbm) report ``false`` when their extra is
    not installed, rather than failing only at training time.
    """
    return {"algorithms": available_algorithms()}


@router.get("/{name}/versions", response_model=list[ModelVersion], summary="List versions")
def list_versions(
    name: str, stage: ModelStage | None = Query(default=None)
) -> list[ModelVersion]:
    return get_registry().list_versions(name, stage)


@router.get("/{name}/versions/{version}", response_model=ModelVersion, summary="Get a version")
def get_version(name: str, version: int) -> ModelVersion:
    return get_registry().get(name, version)


@router.get("/{name}/production", summary="Current production version")
def production(name: str) -> dict[str, Any]:
    registry = get_registry()
    current = registry.get_production(name)
    previous = registry.previous_production(name)
    return {
        "model_name": name,
        "production": current.model_dump(mode="json") if current else None,
        "previous_production": previous.model_dump(mode="json") if previous else None,
        "rollback_target": previous.version if previous else None,
    }


@router.get(
    "/{name}/history", response_model=list[StageTransition], summary="Stage transition history"
)
def history(name: str, version: int | None = None) -> list[StageTransition]:
    return get_registry().history(name, version)


@router.post(
    "/{name}/versions/{version}/stage",
    response_model=ModelVersion,
    summary="Transition a version to a stage",
)
def transition(
    name: str,
    version: int,
    stage: ModelStage = Body(..., embed=True),
    reason: str = Body(default="manual transition", embed=True),
    actor: str = Body(default="api", embed=True),
) -> ModelVersion:
    """Move a version through the stage machine.

    Illegal moves (for example Development straight to Production) are rejected
    with ``invalid_stage_transition`` and the list of legal targets.
    """
    return get_registry().transition_stage(name, version, stage, reason=reason, actor=actor)


@router.post(
    "/{name}/versions/{version}/evaluate-gate",
    summary="Run the approval gate against a version",
)
def evaluate_gate(name: str, version: int) -> dict[str, Any]:
    """Re-run the approval gate using the version's recorded metrics.

    Useful for auditing: it shows exactly which thresholds a version would
    clear under the *current* configuration, which may differ from the
    thresholds in force when it was trained.
    """
    registry = get_registry()
    model_version = registry.get(name, version)
    settings = get_settings()
    result = evaluate_approval(
        model_version.metrics,
        name,
        version,
        validation_passed=True,
        settings=settings,
    )
    production = registry.get_production(name)
    from app.training.approval import compare_to_production

    comparison = compare_to_production(
        model_version.metrics,
        production.metrics if production and production.version != version else None,
        candidate_version=version,
        baseline_version=production.version if production else None,
        settings=settings,
    )
    return {
        "approval": result.model_dump(mode="json"),
        "comparison": comparison.model_dump(mode="json"),
        "thresholds": settings.approval.model_dump(mode="json"),
        "report": result.render_text(),
    }


@router.post(
    "/{name}/versions/{version}/tags", response_model=ModelVersion, summary="Set tags"
)
def set_tags(name: str, version: int, tags: dict[str, str] = Body(...)) -> ModelVersion:
    return get_registry().set_tags(name, version, tags)
