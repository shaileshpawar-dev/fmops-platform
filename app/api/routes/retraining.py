"""Retraining endpoints.

A retraining run trains for minutes, so it always executes as a background job;
the request returns the job at once. Trigger evaluation is synchronous and has
no side effects.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Query, Request
from pydantic import BaseModel, ConfigDict, Field

from app.api.security import request_actor
from app.core.logging import get_logger
from app.jobs.runner import get_job_runner
from app.registry.context import default_model_name
from app.retraining.trigger import evaluate_trigger, get_event_store
from app.schemas.evaluation import RetrainingEvent

logger = get_logger(__name__)
router = APIRouter(prefix="/api/v1/retraining", tags=["retraining"])


class RetrainingRunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_name: str | None = Field(default=None, max_length=64)
    force: bool = Field(
        default=False, description="Run even if no trigger fired (a manual retrain)."
    )
    dataset_version: str | None = Field(
        default=None,
        max_length=128,
        description="New training data to add to the serving version's training set. "
        "Implies a manual retrain.",
    )
    deploy: bool | None = Field(
        default=None, description="Override auto_deploy_if_better for this run."
    )


@router.get("", response_model=list[RetrainingEvent], summary="Recent retraining events")
def list_events(
    limit: int = Query(default=25, le=200), model: str | None = None
) -> list[RetrainingEvent]:
    return get_event_store().recent(limit, model_name=model)


@router.get("/{event_id}", summary="Get one retraining event")
def get_event(event_id: str) -> dict[str, Any]:
    event = get_event_store().get(event_id)
    if event is None:
        return {"event_id": event_id, "found": False}
    return {"found": True, "event": event.model_dump(mode="json")}


@router.get("/trigger/evaluate", summary="Would retraining fire right now?")
def evaluate(force: bool = False, model: str | None = None) -> dict[str, Any]:
    """Evaluate every configured trigger for one model without running anything.

    Returns the per-trigger evidence, so you can see *why* retraining would or
    would not fire -- including a cooldown, or a trigger that fired with no new
    labelled data to learn from.
    """
    decision = evaluate_trigger(force=force, model_name=model)
    return {
        "model_name": model or default_model_name(),
        "should_retrain": decision.should_retrain,
        "trigger": decision.trigger.value if decision.trigger else None,
        "reason": decision.reason,
        "suppressed_by_cooldown": decision.suppressed_by_cooldown,
        "checks": decision.checks,
        "evidence": decision.evidence,
        "report": decision.render_text(),
    }


@router.post("/run", status_code=202, summary="Queue a retraining run")
def run(payload: RetrainingRunRequest, request: Request) -> dict[str, Any]:
    """Queue one retraining cycle for a model.

    The candidate replaces nothing unless it clears the approval gate **and**
    beats the production model by the configured minimum improvement; where a
    human sign-off is required it is held for approval instead. Poll the job,
    then read the retraining event it links to.
    """
    model_name = payload.model_name or default_model_name()
    job = get_job_runner().submit(
        "retraining",
        payload.model_dump(mode="json") | {"model_name": model_name},
        model_name=model_name,
        requested_by=request_actor(request),
    )
    return {
        "job": job,
        "poll": f"/api/v1/jobs/{job['id']}",
        "detail": "retraining queued; the job links to its retraining event once it starts",
    }
