"""Training runs started from the API.

This is a thin job record around :func:`pipelines.training_pipeline.run`. It
adds no training logic of its own -- the pipeline still owns validation,
feature engineering, training, tuning, evaluation, registration, the
champion/challenger comparison and promotion. What is added here is the bit an
HTTP caller needs and the CLI does not: a durable id to poll, because training
takes minutes and holding the request open for that is not an option.

Execution model: FastAPI ``BackgroundTasks`` in the API process, the same
mechanism ``POST /api/v1/retraining/run-async`` already uses. That is honest
for the current single-task deployment and nothing more -- a multi-replica
setup needs a real job runner (Step Functions, SageMaker Pipelines, a queue),
and a run in flight is lost if the process dies. Runs left ``running`` by a
restart are reconciled to ``failed`` at startup rather than lying about being
in progress forever.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from app.core.audit import audit
from app.core.db import get_database
from app.core.logging import get_logger

logger = get_logger(__name__)

# Terminal states never change again; the rest are in flight.
QUEUED = "queued"
RUNNING = "running"
COMPLETED = "completed"
FAILED = "failed"
REJECTED = "rejected"
TERMINAL = frozenset({COMPLETED, FAILED, REJECTED})


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


@dataclass
class TrainingRun:
    """One API-initiated training run."""

    id: str
    status: str
    dataset_version: str | None = None
    algorithm: str | None = None
    tune: bool = False
    promote: bool = False
    target_stage: str | None = None
    model_name: str | None = None
    # Set for a user model: the column it predicts and its positive class. The
    # reference model's contract comes from configuration instead.
    target_column: str | None = None
    positive_label: str | None = None
    model_version: int | None = None
    exit_code: int | None = None
    error: str | None = None
    report: dict[str, Any] = field(default_factory=dict)
    requested_by: str | None = None
    created_at: str = ""
    started_at: str | None = None
    completed_at: str | None = None
    updated_at: str = ""

    @property
    def duration_seconds(self) -> float | None:
        """Wall time, or ``None`` while the run has not finished.

        Deliberately not computed from ``created_at``: time spent queued is not
        training time, and reporting it as such would overstate how long the
        model took to fit.
        """
        if not self.started_at or not self.completed_at:
            return None
        try:
            start = datetime.fromisoformat(self.started_at.replace("Z", "+00:00"))
            end = datetime.fromisoformat(self.completed_at.replace("Z", "+00:00"))
        except ValueError:
            return None
        return round((end - start).total_seconds(), 2)

    def to_dict(self) -> dict[str, Any]:
        metrics: dict[str, Any] = {}
        if isinstance(self.report, dict):
            metrics = self.report.get("metrics") or {}
        return {
            "run_id": self.id,
            "status": self.status,
            "dataset_version": self.dataset_version,
            "algorithm": self.algorithm,
            "tune": self.tune,
            "promote": self.promote,
            "target_stage": self.target_stage,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "target_column": self.target_column,
            "positive_label": self.positive_label,
            "exit_code": self.exit_code,
            "error": self.error,
            "metrics": metrics,
            "report": self.report,
            "requested_by": self.requested_by,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "duration_seconds": self.duration_seconds,
        }


def _row_to_run(row: Any) -> TrainingRun:
    data = dict(row)
    try:
        report = json.loads(data.get("report") or "{}")
    except (TypeError, ValueError):
        report = {}
    return TrainingRun(
        id=data["id"],
        status=data["status"],
        dataset_version=data.get("dataset_version"),
        algorithm=data.get("algorithm"),
        tune=bool(data.get("tune")),
        promote=bool(data.get("promote")),
        target_stage=data.get("target_stage"),
        model_name=data.get("model_name"),
        model_version=data.get("model_version"),
        exit_code=data.get("exit_code"),
        error=data.get("error"),
        report=report if isinstance(report, dict) else {},
        requested_by=data.get("requested_by"),
        target_column=data.get("target_column"),
        positive_label=data.get("positive_label"),
        created_at=data.get("created_at") or "",
        started_at=data.get("started_at"),
        completed_at=data.get("completed_at"),
        updated_at=data.get("updated_at") or "",
    )


class TrainingRunStore:
    """Persistence for training runs. No training logic lives here."""

    def create(
        self,
        dataset_version: str | None,
        algorithm: str | None,
        tune: bool,
        promote: bool,
        target_stage: str,
        requested_by: str | None = None,
        model_name: str | None = None,
        target_column: str | None = None,
        positive_label: str | None = None,
    ) -> TrainingRun:
        run = TrainingRun(
            id=f"train-{uuid.uuid4().hex[:16]}",
            status=QUEUED,
            dataset_version=dataset_version,
            algorithm=algorithm,
            tune=tune,
            promote=promote,
            target_stage=target_stage,
            model_name=model_name,
            target_column=target_column,
            positive_label=positive_label,
            requested_by=requested_by,
            created_at=_now(),
            updated_at=_now(),
        )
        get_database().execute(
            """
            INSERT INTO training_runs
                (id, status, dataset_version, algorithm, tune, promote, target_stage,
                 model_name, target_column, positive_label, report, requested_by,
                 created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, '{}', ?, ?, ?)
            """,
            (
                run.id,
                run.status,
                run.dataset_version,
                run.algorithm,
                int(run.tune),
                int(run.promote),
                run.target_stage,
                run.model_name,
                run.target_column,
                run.positive_label,
                run.requested_by,
                run.created_at,
                run.updated_at,
            ),
        )
        audit(
            "training.run_requested",
            "training_run",
            run.id,
            dataset_version=dataset_version,
            algorithm=algorithm,
            tune=tune,
            promote=promote,
        )
        logger.info(
            "training.run_queued",
            extra={
                "run_id": run.id,
                "dataset_version": dataset_version,
                "algorithm": algorithm,
                "tune": tune,
            },
        )
        return run

    def update(self, run_id: str, **fields: Any) -> None:
        if not fields:
            return
        if "report" in fields and not isinstance(fields["report"], str):
            fields["report"] = json.dumps(fields["report"], default=str)
        fields["updated_at"] = _now()
        assignments = ", ".join(f"{k} = ?" for k in fields)
        get_database().execute(
            f"UPDATE training_runs SET {assignments} WHERE id = ?",  # noqa: S608 - keys are literals
            (*fields.values(), run_id),
        )

    def get(self, run_id: str) -> TrainingRun | None:
        row = get_database().query_one("SELECT * FROM training_runs WHERE id = ?", (run_id,))
        return _row_to_run(row) if row else None

    def list(self, limit: int = 50) -> list[TrainingRun]:
        rows = get_database().query(
            "SELECT * FROM training_runs ORDER BY created_at DESC LIMIT ?", (int(limit),)
        )
        return [_row_to_run(r) for r in rows]

    def fail(self, run_id: str, reason: str) -> None:
        """Mark one in-flight run failed -- its job was lost or cancelled."""
        run = self.get(run_id)
        if run is None or run.status in TERMINAL:
            return
        self.update(run_id, status=FAILED, error=reason, completed_at=_now())

    def reconcile_orphans(self) -> int:
        """Fail in-flight runs that no queued or running job owns.

        Those are runs whose executing process restarted. A run owned by a live
        job -- possibly in another worker process -- is left alone; failing it
        would kill work that is still happening.
        """
        rows = get_database().query(
            "SELECT id FROM training_runs WHERE status IN (?, ?) "
            "AND id NOT IN (SELECT resource_id FROM jobs WHERE resource_id IS NOT NULL "
            "AND status IN ('queued', 'running'))",
            (QUEUED, RUNNING),
        )
        for row in rows:
            self.fail(
                row["id"], "no job is executing this run: the process that ran it restarted"
            )
        if rows:
            logger.warning("training.orphans_reconciled", extra={"count": len(rows)})
        return len(rows)


_store = TrainingRunStore()


def get_training_run_store() -> TrainingRunStore:
    return _store


def execute_training_run(run_id: str, checkpoint: Any = None) -> str:
    """Run the training pipeline and record the outcome. Returns the run status.

    Every decision -- whether the data validates, whether the model clears the
    approval gate, whether it beats the incumbent -- belongs to the pipeline.
    This function builds the run's data contract (the reference one, or one
    derived from the dataset and the chosen target) and translates the
    pipeline's exit code into a run status.
    """
    from pipelines import training_pipeline

    store = get_training_run_store()
    run = store.get(run_id)
    if run is None:
        logger.error("training.run_missing", extra={"run_id": run_id})
        return FAILED

    store.update(run_id, status=RUNNING, started_at=_now())
    try:
        if checkpoint is not None:
            checkpoint()
        run_settings = None
        if run.target_column and run.dataset_version:
            from app.data.contract import contract_for_target
            from app.data.versioning import get_dataset_registry

            frame = get_dataset_registry().load(run.dataset_version)
            run_settings = contract_for_target(frame, run.target_column, run.positive_label)
        exit_code, report = training_pipeline.run(
            dataset_version=run.dataset_version,
            algorithm=run.algorithm,
            tune=run.tune,
            promote=run.promote,
            target_stage=run.target_stage or "Production",
            model_name=run.model_name,
            settings=run_settings,
        )
    except Exception as exc:
        if exc.__class__.__name__ == "JobCancelled":
            raise
        logger.exception("training.run_failed", extra={"run_id": run_id})
        message = getattr(exc, "message", None) or str(exc)
        store.update(run_id, status=FAILED, error=message, exit_code=1, completed_at=_now())
        audit("training.run_failed", "training_run", run_id, outcome="failure", error=message)
        return FAILED

    # The pipeline uses exit code 2 for "ran fine, candidate not accepted",
    # which is a result rather than a failure and must not be shown as one.
    if exit_code == 0:
        status = COMPLETED
    elif exit_code == 2:
        status = REJECTED
    else:
        status = FAILED

    model_version = report.get("model_version")
    store.update(
        run_id,
        status=status,
        exit_code=exit_code,
        error=report.get("error"),
        report=report,
        model_name=report.get("model_name") if model_version is not None else run.model_name,
        model_version=model_version,
        algorithm=report.get("algorithm") or run.algorithm,
        dataset_version=report.get("dataset_version") or run.dataset_version,
        completed_at=_now(),
    )
    audit(
        "training.run_completed",
        "training_run",
        run_id,
        outcome="success" if status in (COMPLETED,) else "failure",
        status=status,
        exit_code=exit_code,
        model_version=model_version,
        promoted=report.get("promoted"),
        approval_decision=report.get("approval_decision"),
    )
    logger.info(
        "training.run_finished",
        extra={"run_id": run_id, "status": status, "exit_code": exit_code},
    )
    return status
