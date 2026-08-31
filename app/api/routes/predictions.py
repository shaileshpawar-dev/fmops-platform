"""Prediction endpoints."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request

from app.api.serving import get_prediction_service
from app.core.config import get_settings
from app.core.exceptions import PredictionError
from app.core.logging import get_logger, new_request_id
from app.monitoring.inference_log import get_inference_log
from app.schemas.prediction import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    FeedbackRequest,
    FeedbackResponse,
    ModelInfoResponse,
    PredictionRequest,
    PredictionResponse,
)

logger = get_logger(__name__)
router = APIRouter(prefix="/api/v1", tags=["predictions"])


def _request_id(request: Request) -> str:
    return getattr(request.state, "request_id", None) or new_request_id()


@router.post("/predict", response_model=PredictionResponse, summary="Score one application")
def predict(payload: PredictionRequest, request: Request) -> PredictionResponse:
    """Score a single loan application.

    The response carries the model version and variant that actually served the
    request, so a canary or shadow rollout is visible per response rather than
    only in aggregate.
    """
    service = get_prediction_service()
    return service.predict_one(
        features=payload.features.model_dump(),
        request_id=_request_id(request),
        version=payload.model_version,
        threshold=payload.threshold,
        explain=payload.explain,
    )


@router.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    summary="Score many applications",
)
def predict_batch(
    payload: BatchPredictionRequest, request: Request
) -> BatchPredictionResponse:
    settings = get_settings()
    limit = settings.security.max_batch_rows
    if len(payload.instances) > limit:
        raise PredictionError(
            f"batch of {len(payload.instances)} exceeds the configured maximum "
            f"of {limit} rows; split the request or raise "
            "FMOPS_SECURITY__MAX_BATCH_ROWS",
            n_instances=len(payload.instances),
            max_batch_rows=limit,
        )
    service = get_prediction_service()
    return service.predict_batch(
        instances=[i.model_dump() for i in payload.instances],
        request_id=_request_id(request),
        version=payload.model_version,
        threshold=payload.threshold,
    )


@router.get("/model", response_model=ModelInfoResponse, summary="Currently serving model")
def current_model() -> ModelInfoResponse:
    service = get_prediction_service()
    model, _variant = service.resolve_model()
    return ModelInfoResponse(
        model_name=model.model_name,
        model_version=model.version,
        model_stage=model.stage,
        algorithm=model.algorithm,
        artifact_uri=model.artifact_uri,
        dataset_version=model.dataset_version,
        git_commit=model.git_commit,
        metrics=model.metrics,
        params=model.params,
        loaded_at=model.loaded_at,
        feature_names=model.feature_columns,
    )


@router.post(
    "/feedback",
    response_model=FeedbackResponse,
    summary="Submit ground truth for a prediction",
)
def feedback(payload: FeedbackRequest) -> FeedbackResponse:
    """Attach an observed outcome to a served prediction.

    This is the only route by which the platform can compute *live* model
    quality or measure concept drift; without labels both are reported as
    unavailable rather than estimated.
    """
    log = get_inference_log()
    total = log.record_feedback(payload.request_id, payload.actual_label, payload.source)
    return FeedbackResponse(request_id=payload.request_id, recorded=True, labelled_total=total)


@router.get("/predictions/recent", summary="Recent predictions")
def recent_predictions(limit: int = 50, include_shadow: bool = False) -> dict[str, Any]:
    log = get_inference_log()
    rows = log.recent(limit=min(limit, 500), include_shadow=include_shadow, only_ok=False)
    return {"count": len(rows), "predictions": rows}
