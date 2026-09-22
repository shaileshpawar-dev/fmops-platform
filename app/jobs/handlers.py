"""What each kind of job does.

Handlers are thin: they call the same library code the CLI and the pipelines
use, link the job to the record that code produces, and turn its outcome into
the job's result. Each also knows how to fail its record if the job is lost
(the worker died) or cancelled, so a run never sits in "running" forever.
"""

from __future__ import annotations

from typing import Any

from app.core.logging import get_logger
from app.jobs.runner import JobContext, handler

logger = get_logger(__name__)

TRAINING = "training"
AUTOML = "automl"
RETRAINING = "retraining"
DEPLOYMENT = "deployment"
DRIFT_SCAN = "drift_scan"


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def _lose_training(job: dict[str, Any], reason: str) -> None:
    from app.training.jobs import get_training_run_store

    if job.get("resource_id"):
        get_training_run_store().fail(job["resource_id"], reason)


@handler(TRAINING, on_lost=_lose_training)
def run_training(ctx: JobContext) -> dict[str, Any]:
    from app.training.jobs import FAILED, execute_training_run, get_training_run_store

    run_id = ctx.payload["run_id"]
    ctx.link(run_id)
    status = execute_training_run(run_id, checkpoint=ctx.checkpoint)
    run = get_training_run_store().get(run_id)
    if status == FAILED:
        # The run records why; the job fails so the failure is visible as one.
        raise RuntimeError(run.error if run and run.error else "training failed")
    return {
        "run_id": run_id,
        "status": status,
        "model_name": run.model_name if run else None,
        "model_version": run.model_version if run else None,
    }


# --------------------------------------------------------------------------- #
# AutoML
# --------------------------------------------------------------------------- #
def _lose_automl(job: dict[str, Any], reason: str) -> None:
    from app.automl.runner import get_automl_store

    if job.get("resource_id"):
        get_automl_store().fail(job["resource_id"], reason)


@handler(AUTOML, on_lost=_lose_automl)
def run_automl(ctx: JobContext) -> dict[str, Any]:
    from app.automl.runner import FAILED, execute_automl_run, get_automl_store

    run_id = ctx.payload["run_id"]
    ctx.link(run_id)
    status = execute_automl_run(run_id, checkpoint=ctx.checkpoint)
    run = get_automl_store().get(run_id) or {}
    if status == FAILED:
        raise RuntimeError(run.get("error") or "AutoML run failed")
    return {
        "run_id": run_id,
        "status": status,
        "model_name": run.get("model_name"),
        "best_algorithm": run.get("best_algorithm"),
        "best_model_version": run.get("best_model_version"),
    }


# --------------------------------------------------------------------------- #
# Retraining
# --------------------------------------------------------------------------- #
def _lose_retraining(job: dict[str, Any], reason: str) -> None:
    from app.retraining.trigger import get_event_store
    from app.schemas.common import RetrainingStatus

    event_id = job.get("resource_id")
    if not event_id:
        return
    store = get_event_store()
    event = store.get(event_id)
    if event is not None and event.status in (
        RetrainingStatus.CREATED,
        RetrainingStatus.RUNNING,
    ):
        store.update(
            event_id,
            status=RetrainingStatus.FAILED,
            decision="failed",
            detail={**event.detail, "error": reason},
        )


@handler(RETRAINING, on_lost=_lose_retraining)
def run_retraining_job(ctx: JobContext) -> dict[str, Any]:
    from app.retraining.pipeline import RetrainingPipeline

    payload = ctx.payload
    decision = RetrainingPipeline().run(
        force=bool(payload.get("force")),
        deploy=payload.get("deploy"),
        model_name=payload.get("model_name"),
        dataset_version=payload.get("dataset_version"),
        checkpoint=ctx.checkpoint,
        on_event=ctx.link,
    )
    return decision.model_dump(mode="json")


# --------------------------------------------------------------------------- #
# Deployment
# --------------------------------------------------------------------------- #
def _lose_deployment(job: dict[str, Any], reason: str) -> None:
    from app.deployment.base import get_deployment_store
    from app.schemas.common import DeploymentState

    deployment_id = job.get("resource_id")
    if not deployment_id:
        return
    store = get_deployment_store()
    try:
        deployment = store.get(deployment_id)
    except Exception:
        return
    if deployment.state in (DeploymentState.PENDING, DeploymentState.IN_PROGRESS):
        store.update(deployment_id, state=DeploymentState.FAILED, message=reason)


@handler(DEPLOYMENT, on_lost=_lose_deployment)
def run_deployment(ctx: JobContext) -> dict[str, Any]:
    from app.deployment.manager import get_deployment_manager
    from app.schemas.deployment import DeploymentRequest

    request = DeploymentRequest.model_validate(ctx.payload["request"])
    ctx.checkpoint()
    result = get_deployment_manager().deploy(
        request,
        actor=ctx.payload.get("actor") or "system",
        sleep=ctx.cancellable_sleep,
        on_created=ctx.link,
    )
    return result.model_dump(mode="json")


# --------------------------------------------------------------------------- #
# Drift scan
# --------------------------------------------------------------------------- #
@handler(DRIFT_SCAN)
def run_drift_scan(ctx: JobContext) -> dict[str, Any]:
    from app.monitoring.service import get_monitoring_service

    report = get_monitoring_service().run_drift_scan(model_name=ctx.payload["model_name"])
    ctx.link(report.id)
    return {
        "report_id": report.id,
        "drift_detected": report.drift_detected,
        "dataset_drift_score": report.dataset_drift_score,
        "drifted_features": report.drifted_features,
    }


# --------------------------------------------------------------------------- #
# Retry
# --------------------------------------------------------------------------- #
def prepare_retry(job: dict[str, Any]) -> tuple[dict[str, Any], str | None]:
    """The payload for a new attempt.

    Training and AutoML runs are records of one attempt each, so a retry gets a
    fresh record with the same parameters -- the failed one stays as it was.
    Retraining and deployment payloads are already self-contained.
    """
    kind = job["kind"]
    if kind == TRAINING:
        from app.training.jobs import get_training_run_store

        runs = get_training_run_store()
        previous = runs.get(job["payload"]["run_id"])
        if previous is None:
            raise ValueError("the original training run no longer exists")
        fresh = runs.create(
            dataset_version=previous.dataset_version,
            algorithm=previous.algorithm,
            tune=previous.tune,
            promote=previous.promote,
            target_stage=previous.target_stage or "Staging",
            model_name=previous.model_name,
            target_column=previous.target_column,
            positive_label=previous.positive_label,
            requested_by=job.get("requested_by"),
        )
        return {"run_id": fresh.id}, fresh.id
    if kind == AUTOML:
        from app.automl.runner import get_automl_store

        automl_runs = get_automl_store()
        original = automl_runs.get(job["payload"]["run_id"])
        if original is None:
            raise ValueError("the original AutoML run no longer exists")
        new_id = automl_runs.create(
            dataset_version=original["dataset_version"],
            target=original["target_column"],
            problem_type=original["problem_type"],
            algorithms=original["algorithms"],
            primary_metric=original["primary_metric"],
            tune=original["tune"],
            target_stage=original["target_stage"],
            max_models=original["max_models"],
            model_name=original.get("model_name"),
            positive_label=original.get("positive_label"),
        )
        return {"run_id": new_id}, new_id
    return dict(job["payload"]), None


def reconcile_unowned() -> int:
    """Fail domain records left in flight by a process that no longer exists."""
    from app.automl.runner import get_automl_store
    from app.training.jobs import get_training_run_store

    return (
        get_training_run_store().reconcile_orphans() + get_automl_store().reconcile_orphans()
    )
