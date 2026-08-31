"""Retraining endpoints."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, BackgroundTasks, Query

from app.core.logging import get_logger
from app.retraining.pipeline import RetrainingPipeline
from app.retraining.trigger import evaluate_trigger, get_event_store
from app.schemas.evaluation import RetrainingDecision, RetrainingEvent

logger = get_logger(__name__)
router = APIRouter(prefix="/api/v1/retraining", tags=["retraining"])


@router.get("", response_model=list[RetrainingEvent], summary="Recent retraining events")
def list_events(limit: int = Query(default=25, le=200)) -> list[RetrainingEvent]:
    return get_event_store().recent(limit)


@router.get("/{event_id}", summary="Get one retraining event")
def get_event(event_id: str) -> dict[str, Any]:
    event = get_event_store().get(event_id)
    if event is None:
        return {"event_id": event_id, "found": False}
    return {"found": True, "event": event.model_dump(mode="json")}


@router.get("/trigger/evaluate", summary="Would retraining fire right now?")
def evaluate(force: bool = False) -> dict[str, Any]:
    """Evaluate every configured trigger without running anything.

    Returns the per-trigger evidence, so you can see *why* retraining would or
    would not fire (including whether it is suppressed by the cooldown).
    """
    decision = evaluate_trigger(force=force)
    return {
        "should_retrain": decision.should_retrain,
        "trigger": decision.trigger.value if decision.trigger else None,
        "reason": decision.reason,
        "suppressed_by_cooldown": decision.suppressed_by_cooldown,
        "checks": decision.checks,
        "evidence": decision.evidence,
        "report": decision.render_text(),
    }


@router.post("/run", response_model=RetrainingDecision, summary="Run the retraining pipeline")
def run(
    force: bool = Query(default=False, description="Run even if no trigger fired."),
    deploy: bool | None = Query(
        default=None,
        description="Override auto_deploy_if_better for this run.",
    ),
) -> RetrainingDecision:
    """Run one retraining cycle synchronously.

    The candidate is deployed only if it clears the approval gate **and** beats
    the production model by the configured minimum improvement; otherwise the
    production model keeps serving and the response explains why.
    """
    return RetrainingPipeline().run(force=force, deploy=deploy)


@router.post("/run-async", summary="Run the retraining pipeline in the background")
def run_async(
    background: BackgroundTasks,
    force: bool = False,
    deploy: bool | None = None,
) -> dict[str, str]:
    """Kick off retraining without blocking the caller.

    Training can take minutes; poll ``GET /api/v1/retraining`` for the event and
    its outcome. Suitable for a single-process deployment -- a multi-replica
    production setup should dispatch to a job runner (SageMaker Pipelines, Step
    Functions, or a queue) instead. See docs/architecture.md.
    """

    def _run() -> None:
        try:
            RetrainingPipeline().run(force=force, deploy=deploy)
        except Exception as exc:
            logger.error("retraining.background_failed", extra={"error": str(exc)})

    background.add_task(_run)
    return {
        "status": "started",
        "detail": "poll GET /api/v1/retraining for the resulting event",
    }
