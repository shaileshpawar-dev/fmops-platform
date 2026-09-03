"""Start and inspect training runs.

Thin by design. The route validates the request, records a run and hands off to
:func:`app.training.jobs.execute_training_run`, which calls the existing
training pipeline. No training, evaluation, registration or promotion logic
lives in this module.

Runs execute as FastAPI background tasks in the API process -- the mechanism
``POST /api/v1/retraining/run-async`` already uses. See
:mod:`app.training.jobs` for what that does and does not guarantee.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from fastapi import APIRouter, BackgroundTasks, Query, Response
from pydantic import BaseModel, ConfigDict, Field

from app.core.config import get_settings
from app.core.exceptions import FMOpsError
from app.core.logging import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/api/v1/training", tags=["training"])


class UnknownAlgorithmError(FMOpsError):
    """The requested algorithm is not one the model factory can build."""

    code = "unknown_algorithm"
    http_status = 422


class TrainingRunNotFoundError(FMOpsError):
    code = "training_run_not_found"
    http_status = 404


class UnknownDatasetError(FMOpsError):
    """The requested dataset version is not registered."""

    code = "dataset_not_found"
    http_status = 404


class TrainingRunRequest(BaseModel):
    """What to train, and what to do with the result.

    Every field maps onto an existing pipeline argument. There is deliberately
    no free-form parameter passthrough: the request must not be able to reach
    arbitrary code, a shell, or a filesystem path. Unknown fields are rejected
    rather than ignored, so a caller aiming at a passthrough that does not
    exist is told so instead of getting a silently different run.
    """

    model_config = ConfigDict(extra="forbid")

    dataset_version: str | None = Field(
        default=None,
        description="Registered dataset version. Omit to use the latest.",
        max_length=128,
    )
    algorithm: str | None = Field(
        default=None,
        description="Algorithm id from GET /api/v1/models/algorithms. Omit for the configured default.",
        max_length=64,
    )
    tune: bool = Field(
        default=False, description="Run hyperparameter search before the final fit."
    )
    promote: bool = Field(
        default=False,
        description=(
            "Register the model and put it through the approval gate. Promotion still "
            "requires clearing the gate and beating the incumbent -- this only asks for "
            "the attempt."
        ),
    )
    target_stage: Literal["Staging", "Production"] = Field(
        default="Staging",
        description="Stage to promote into if the gate passes. Defaults to Staging.",
    )


class TrainingRunAccepted(BaseModel):
    run_id: str
    status: str
    poll: str
    detail: str


@router.post(
    "/runs",
    status_code=202,
    response_model=TrainingRunAccepted,
    summary="Start a training run in the background",
)
def start_training_run(
    request: TrainingRunRequest,
    background: BackgroundTasks,
    response: Response,
) -> TrainingRunAccepted:
    """Queue a training run and return immediately.

    202 rather than 201: the run has been accepted, not completed. Training
    takes minutes, so poll ``GET /api/v1/training/runs/{run_id}`` for the
    outcome.
    """
    settings = get_settings()

    if request.algorithm:
        # available_algorithms() reports which estimators can actually be
        # instantiated here, so an optional backend whose extra is not
        # installed is refused now rather than minutes into a background run.
        from app.training.model_factory import available_algorithms

        available = available_algorithms()
        if not available.get(request.algorithm, False):
            raise UnknownAlgorithmError(
                (
                    f"algorithm {request.algorithm!r} is not available in this environment"
                    if request.algorithm in available
                    else f"unknown algorithm {request.algorithm!r}"
                ),
                requested=request.algorithm,
                available=sorted(k for k, v in available.items() if v),
            )

    if request.dataset_version:
        from app.data.versioning import get_dataset_registry

        known_versions = {v.version for v in get_dataset_registry().list_versions()}
        if request.dataset_version not in known_versions:
            raise UnknownDatasetError(
                f"dataset version {request.dataset_version!r} is not registered",
                requested=request.dataset_version,
            )

    from app.training.jobs import execute_training_run, get_training_run_store

    run = get_training_run_store().create(
        dataset_version=request.dataset_version,
        algorithm=request.algorithm or settings.training.algorithm,
        tune=request.tune,
        promote=request.promote,
        target_stage=request.target_stage,
    )
    background.add_task(execute_training_run, run.id)

    response.headers["Location"] = f"/api/v1/training/runs/{run.id}"
    return TrainingRunAccepted(
        run_id=run.id,
        status=run.status,
        poll=f"/api/v1/training/runs/{run.id}",
        detail="training started in the background; poll the run for its status",
    )


@router.get("/runs", summary="List training runs")
def list_training_runs(
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
) -> dict[str, Any]:
    """Most recent runs first."""
    from app.training.jobs import get_training_run_store

    runs = get_training_run_store().list(limit=limit)
    return {"count": len(runs), "runs": [r.to_dict() for r in runs]}


@router.get("/runs/{run_id}", summary="Get one training run")
def get_training_run(run_id: str) -> dict[str, Any]:
    """Full record, including the pipeline report once the run has finished."""
    from app.training.jobs import get_training_run_store

    run = get_training_run_store().get(run_id)
    if run is None:
        raise TrainingRunNotFoundError(f"no training run with id {run_id!r}", run_id=run_id)
    return run.to_dict()


__all__ = ["router"]
