"""AutoML: profile a dataset, then search candidate models over it.

Thin. Profiling is `app.automl.profiler`, recommendation is
`app.automl.recommend`, and execution is `app.automl.runner`, which drives the
existing training function and the existing approval gate. This module
validates requests and moves JSON.
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, BackgroundTasks, Query, Response
from pydantic import BaseModel, ConfigDict, Field

from app.core.exceptions import FMOpsError
from app.core.logging import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/api/v1/automl", tags=["automl"])

MAX_MODELS_LIMIT = 5


class AutoMLRunNotFoundError(FMOpsError):
    code = "automl_run_not_found"
    http_status = 404


class UnsupportedProblemError(FMOpsError):
    """The target implies a problem the training stack cannot fit."""

    code = "unsupported_problem_type"
    http_status = 422


class InvalidAutoMLRequestError(FMOpsError):
    code = "invalid_automl_request"
    http_status = 422


@router.get("/profile/{version}", summary="Profile a dataset and recommend a target")
def profile_dataset(
    version: str,
    target: Annotated[
        str | None, Query(description="Pin the target instead of taking the suggestion.")
    ] = None,
) -> dict[str, Any]:
    """Profile a registered dataset version.

    Returns the column profile, the recommended target with the evidence behind
    it, the inferred problem type, data-quality warnings, and the candidate
    models that suit the result. Everything is a recommendation: nothing is
    trained and nothing is decided here.
    """
    from app.automl.profiler import profile_frame
    from app.automl.recommend import (
        default_selection,
        primary_metric_for,
        recommend_candidates,
        supported_problem_types,
    )
    from app.data.versioning import get_dataset_registry

    registry = get_dataset_registry()
    registry.get(version)  # 404s on an unknown version
    frame = registry.load(version)

    try:
        profile = profile_frame(frame, target_override=target)
    except KeyError as exc:
        raise InvalidAutoMLRequestError(str(exc), version=version, target=target) from exc

    suggestion = profile.suggested_target
    problem = suggestion.problem_type if suggestion else "unknown"
    recommendations = recommend_candidates(profile, problem)
    metric, secondary, metric_note = primary_metric_for(problem, profile)

    return {
        "dataset_version": version,
        "profile": profile.to_dict(),
        "problem_type": problem,
        "problem_supported": problem in supported_problem_types(),
        "supported_problem_types": supported_problem_types(),
        "candidates": [c.to_dict() for c in recommendations],
        "default_selection": default_selection(recommendations),
        "primary_metric": metric,
        "secondary_metrics": secondary,
        "metric_note": metric_note,
        "max_models_limit": MAX_MODELS_LIMIT,
    }


class AutoMLRunRequest(BaseModel):
    """What to search over.

    Closed surface: unknown fields are rejected, and there is no passthrough
    into training configuration beyond the fields below.
    """

    model_config = ConfigDict(extra="forbid")

    dataset_version: str = Field(max_length=128)
    target_column: str = Field(max_length=128, description="Confirmed target. Required.")
    algorithms: list[str] = Field(
        default_factory=list,
        description="Candidates to try. Empty means the recommended selection.",
        max_length=MAX_MODELS_LIMIT,
    )
    primary_metric: str = Field(default="roc_auc", max_length=32)
    tune: bool = Field(default=False, description="Run hyperparameter search per candidate.")
    target_stage: str = Field(default="Staging", pattern="^(Staging|Production)$")
    max_models: int = Field(default=3, ge=1, le=MAX_MODELS_LIMIT)


class AutoMLRunAccepted(BaseModel):
    run_id: str
    status: str
    algorithms: list[str]
    poll: str
    detail: str


@router.post(
    "/runs", status_code=202, response_model=AutoMLRunAccepted, summary="Start an AutoML run"
)
def start_automl_run(
    request: AutoMLRunRequest, background: BackgroundTasks, response: Response
) -> AutoMLRunAccepted:
    """Queue a candidate search.

    The target must be confirmed by the caller -- the profiler recommends, it
    does not decide, and training never starts on a guess.
    """
    from app.automl.profiler import profile_frame
    from app.automl.recommend import (
        default_selection,
        recommend_candidates,
        supported_problem_types,
    )
    from app.automl.runner import execute_automl_run, get_automl_store
    from app.data.versioning import get_dataset_registry
    from app.training.model_factory import available_algorithms

    registry = get_dataset_registry()
    registry.get(request.dataset_version)
    frame = registry.load(request.dataset_version)

    if request.target_column not in frame.columns:
        raise InvalidAutoMLRequestError(
            f"target column {request.target_column!r} is not in dataset {request.dataset_version}",
            available_columns=[str(c) for c in frame.columns][:50],
        )

    profile = profile_frame(frame, target_override=request.target_column)
    suggestion = profile.suggested_target
    problem = suggestion.problem_type if suggestion else "unknown"
    if problem not in supported_problem_types():
        raise UnsupportedProblemError(
            f"target {request.target_column!r} looks like {problem}; this platform's training "
            f"stack fits binary classification only",
            problem_type=problem,
            supported=supported_problem_types(),
        )

    chosen = request.algorithms or default_selection(
        recommend_candidates(profile, problem), request.max_models
    )
    chosen = chosen[: request.max_models]
    if not chosen:
        raise InvalidAutoMLRequestError(
            "no candidate algorithms are available in this environment"
        )

    availability = available_algorithms()
    unavailable = [a for a in chosen if not availability.get(a, False)]
    if unavailable:
        raise InvalidAutoMLRequestError(
            f"these algorithms are not available in this environment: {', '.join(unavailable)}",
            requested=chosen,
            available=sorted(k for k, v in availability.items() if v),
        )

    store = get_automl_store()
    run_id = store.create(
        dataset_version=request.dataset_version,
        target=request.target_column,
        problem_type=problem,
        algorithms=chosen,
        primary_metric=request.primary_metric,
        tune=request.tune,
        target_stage=request.target_stage,
        max_models=request.max_models,
    )
    background.add_task(execute_automl_run, run_id)

    response.headers["Location"] = f"/api/v1/automl/runs/{run_id}"
    return AutoMLRunAccepted(
        run_id=run_id,
        status="queued",
        algorithms=chosen,
        poll=f"/api/v1/automl/runs/{run_id}",
        detail=f"AutoML will train up to {len(chosen)} candidate model(s) in the background",
    )


@router.get("/runs", summary="List AutoML runs")
def list_automl_runs(limit: Annotated[int, Query(ge=1, le=200)] = 50) -> dict[str, Any]:
    from app.automl.runner import get_automl_store

    runs = get_automl_store().list(limit=limit)
    # The full column profile is large and useless in a list view.
    for run in runs:
        run.pop("profile", None)
    return {"count": len(runs), "runs": runs}


@router.get("/runs/{run_id}", summary="Get one AutoML run")
def get_automl_run(run_id: str) -> dict[str, Any]:
    """The whole record: configuration, profile, candidates, leaderboard, gate outcome."""
    import contextlib
    import json

    from app.automl.runner import get_automl_store, rank_candidates

    run = get_automl_store().get(run_id)
    if run is None:
        raise AutoMLRunNotFoundError(f"no AutoML run with id {run_id!r}", run_id=run_id)

    if run.get("promotion"):
        with contextlib.suppress(TypeError, ValueError):
            run["promotion"] = json.loads(run["promotion"])

    # The leaderboard is derived, not stored twice.
    from app.automl.runner import Candidate

    cands = [
        Candidate(**{k: v for k, v in c.items() if k in Candidate.__dataclass_fields__})
        for c in run.get("candidates", [])
    ]
    run["leaderboard"] = [
        c.to_dict() for c in rank_candidates(cands, run.get("primary_metric", "roc_auc"))
    ]
    return run


@router.get("/runs/{run_id}/candidates", summary="Candidates for one AutoML run")
def get_automl_candidates(run_id: str) -> dict[str, Any]:
    from app.automl.runner import Candidate, get_automl_store, rank_candidates

    run = get_automl_store().get(run_id)
    if run is None:
        raise AutoMLRunNotFoundError(f"no AutoML run with id {run_id!r}", run_id=run_id)

    cands = [
        Candidate(**{k: v for k, v in c.items() if k in Candidate.__dataclass_fields__})
        for c in run.get("candidates", [])
    ]
    ranked = rank_candidates(cands, run.get("primary_metric", "roc_auc"))
    from app.automl.runner import RANKING_RULE

    return {
        "run_id": run_id,
        "primary_metric": run.get("primary_metric"),
        "ranking_rule": RANKING_RULE,
        "candidates": [c.to_dict() for c in cands],
        "leaderboard": [c.to_dict() for c in ranked],
        "best": ranked[0].to_dict() if ranked else None,
    }


__all__ = ["router"]
