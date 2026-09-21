"""AutoML run execution.

Orchestration only. Each candidate is fitted by ``app.training.train.train_model``
-- the same function the CLI, the training API and the retraining pipeline use
-- and the winner is registered and judged by ``register_and_promote``, the
same approval gate as everything else. No training, evaluation, registration or
promotion logic is reimplemented here, and the production thresholds are not
relaxed to make a run look successful.

The one thing this module does that the pipeline cannot is point training at a
target column and feature set chosen at runtime. ``train_model`` accepts a
``Settings`` override and ``split_features_target`` reads its ``DataConfig``,
so a per-run copy of the settings with the profiled target and features is
enough. Nothing global is mutated.

Partial failure is a result, not a catastrophe: if one candidate raises and
another succeeds the run completes with warnings, because a leaderboard with
two entries is still a leaderboard.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from app.automl.profiler import profile_frame
from app.automl.recommend import supported_problem_types
from app.core.audit import audit
from app.core.db import get_database
from app.core.logging import get_logger

logger = get_logger(__name__)

QUEUED = "queued"
PROFILING = "profiling"
TRAINING = "training"
RANKING = "ranking"
COMPLETED = "completed"
COMPLETED_WITH_WARNINGS = "completed_with_warnings"
FAILED = "failed"
TERMINAL = frozenset({COMPLETED, COMPLETED_WITH_WARNINGS, FAILED})

MAX_CANDIDATES = 5


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


class AutoMLError(Exception):
    """The run cannot proceed, with a reason worth showing the user."""


@dataclass
class Candidate:
    algorithm: str
    status: str = "queued"
    metrics: dict[str, float] = field(default_factory=dict)
    duration_seconds: float | None = None
    model_version: int | None = None
    run_id: str | None = None
    error: str | None = None
    rank: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "algorithm": self.algorithm,
            "status": self.status,
            "metrics": self.metrics,
            "duration_seconds": self.duration_seconds,
            "model_version": self.model_version,
            "run_id": self.run_id,
            "error": self.error,
            "rank": self.rank,
        }


class AutoMLRunStore:
    """Persistence for AutoML runs. No ML logic."""

    def create(
        self,
        dataset_version: str,
        target: str,
        problem_type: str,
        algorithms: list[str],
        primary_metric: str,
        tune: bool,
        target_stage: str,
        max_models: int,
        model_name: str | None = None,
        positive_label: str | None = None,
    ) -> str:
        run_id = f"automl-{uuid.uuid4().hex[:16]}"
        get_database().execute(
            """
            INSERT INTO automl_runs
                (id, status, dataset_version, target_column, problem_type, algorithms,
                 primary_metric, tune, target_stage, max_models, profile, candidates,
                 created_at, updated_at, model_name, positive_label)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, '{}', '[]', ?, ?, ?, ?)
            """,
            (
                run_id,
                QUEUED,
                dataset_version,
                target,
                problem_type,
                json.dumps(algorithms),
                primary_metric,
                int(tune),
                target_stage,
                int(max_models),
                _now(),
                _now(),
                model_name,
                positive_label,
            ),
        )
        audit(
            "automl.run_requested",
            "automl_run",
            run_id,
            model_name=model_name,
            dataset_version=dataset_version,
            target=target,
            problem_type=problem_type,
            algorithms=algorithms,
            tune=tune,
        )
        logger.info(
            "automl.run_queued",
            extra={
                "run_id": run_id,
                "dataset_version": dataset_version,
                "target": target,
                "algorithms": algorithms,
            },
        )
        return run_id

    def update(self, run_id: str, **fields: Any) -> None:
        if not fields:
            return
        for key in ("profile", "candidates", "ranking_rule"):
            if key in fields and not isinstance(fields[key], str):
                fields[key] = json.dumps(fields[key], default=str)
        fields["updated_at"] = _now()
        assignments = ", ".join(f"{k} = ?" for k in fields)
        get_database().execute(
            f"UPDATE automl_runs SET {assignments} WHERE id = ?",  # noqa: S608 - keys are literals
            (*fields.values(), run_id),
        )

    def get(self, run_id: str) -> dict[str, Any] | None:
        row = get_database().query_one("SELECT * FROM automl_runs WHERE id = ?", (run_id,))
        return _row_to_run(row) if row else None

    def list(self, limit: int = 50) -> list[dict[str, Any]]:
        rows = get_database().query(
            "SELECT * FROM automl_runs ORDER BY created_at DESC LIMIT ?", (int(limit),)
        )
        return [_row_to_run(r) for r in rows]

    def fail(self, run_id: str, reason: str) -> None:
        run = self.get(run_id)
        if run is None or run["status"] in TERMINAL:
            return
        self.update(run_id, status=FAILED, error=reason, completed_at=_now())

    def reconcile_orphans(self) -> int:
        """Fail in-flight runs that no queued or running job owns (see the
        training store: a live job in another process must not be touched)."""
        rows = get_database().query(
            "SELECT id FROM automl_runs WHERE status NOT IN (?, ?, ?) "
            "AND id NOT IN (SELECT resource_id FROM jobs WHERE resource_id IS NOT NULL "
            "AND status IN ('queued', 'running'))",
            (COMPLETED, COMPLETED_WITH_WARNINGS, FAILED),
        )
        for row in rows:
            self.fail(
                row["id"], "no job is executing this run: the process that ran it restarted"
            )
        if rows:
            logger.warning("automl.orphans_reconciled", extra={"count": len(rows)})
        return len(rows)


def _row_to_run(row: Any) -> dict[str, Any]:
    d = dict(row)
    for key, default in (("profile", {}), ("candidates", []), ("algorithms", [])):
        try:
            d[key] = json.loads(d.get(key) or json.dumps(default))
        except (TypeError, ValueError):
            d[key] = default
    d["tune"] = bool(d.get("tune"))
    d["run_id"] = d.pop("id")
    return d


_store = AutoMLRunStore()


def get_automl_store() -> AutoMLRunStore:
    return _store


# --------------------------------------------------------------------------- #
# Ranking
# --------------------------------------------------------------------------- #
RANKING_RULE = (
    "Candidates that trained and evaluated successfully are ranked by the configured "
    "primary metric, descending. Ties are broken by F1, then by shorter training time, "
    "then by algorithm name so the order is stable across runs. Failed candidates are "
    "never ranked and can never win."
)


def rank_candidates(candidates: list[Candidate], primary_metric: str) -> list[Candidate]:
    successful = [c for c in candidates if c.status == "completed" and c.metrics]
    successful.sort(
        key=lambda c: (
            -float(c.metrics.get(primary_metric, float("-inf"))),
            -float(c.metrics.get("f1", float("-inf"))),
            c.duration_seconds if c.duration_seconds is not None else float("inf"),
            c.algorithm,
        )
    )
    for position, cand in enumerate(successful, start=1):
        cand.rank = position
    return successful


# --------------------------------------------------------------------------- #
# Execution
# --------------------------------------------------------------------------- #
def execute_automl_run(run_id: str, checkpoint: Any = None) -> str:
    """Profile, train every candidate, rank them, register the winner.

    ``checkpoint`` is called before each candidate; it raises when the job has
    been cancelled, which stops the run between fits rather than mid-write.
    Returns the run's final status.
    """
    from app.data.contract import InvalidTrainingTargetError, contract_for_target
    from app.data.versioning import get_dataset_registry
    from app.schemas.model import ModelStage, TrainingRequest
    from app.training.registry import register_and_promote
    from app.training.train import train_model

    store = get_automl_store()
    run = store.get(run_id)
    if run is None:
        logger.error("automl.run_missing", extra={"run_id": run_id})
        return FAILED

    started = time.perf_counter()
    try:
        store.update(run_id, status=PROFILING, started_at=_now())
        frame = get_dataset_registry().load(run["dataset_version"])
        profile = profile_frame(frame, target_override=run["target_column"])

        if profile.suggested_target is None:
            raise AutoMLError(f"column {run['target_column']!r} cannot be used as a target")
        problem = profile.suggested_target.problem_type
        if problem not in supported_problem_types():
            raise AutoMLError(
                f"this platform trains binary classification only; target "
                f"{run['target_column']!r} looks like {problem}"
            )

        features = [c.name for c in profile.columns if c.role == "feature"]
        if not features:
            raise AutoMLError("no usable feature columns remain after profiling")

        store.update(run_id, profile=profile.to_dict(), status=TRAINING)
        try:
            settings = contract_for_target(
                frame, run["target_column"], run.get("positive_label"), profile=profile
            )
        except InvalidTrainingTargetError as exc:
            raise AutoMLError(exc.message) from exc
        model_name = run.get("model_name") or settings.tracking.registered_model_name

        candidates = [Candidate(algorithm=a) for a in run["algorithms"][:MAX_CANDIDATES]]
        # TrainingRunResult holds paths and metrics, not the fitted estimator,
        # so keeping one per candidate costs almost nothing -- and it means the
        # model that is registered is the exact model that won, rather than a
        # refit that could land somewhere slightly different.
        results: dict[str, Any] = {}
        store.update(run_id, candidates=[c.to_dict() for c in candidates])

        for cand in candidates:
            if checkpoint is not None:
                checkpoint()
            cand.status = "training"
            store.update(run_id, candidates=[c.to_dict() for c in candidates])
            t0 = time.perf_counter()
            try:
                result = train_model(
                    TrainingRequest(
                        model_name=model_name,
                        dataset_version=run["dataset_version"],
                        algorithm=cand.algorithm,
                        tune=run["tune"],
                    ),
                    settings=settings,
                )
                results[cand.algorithm] = result
                cand.metrics = result.evaluation.metrics.as_dict() if result.evaluation else {}
                cand.model_version = result.registered_version
                cand.run_id = result.run_id
                cand.duration_seconds = round(time.perf_counter() - t0, 2)
                cand.status = "completed"
                logger.info(
                    "automl.candidate_completed",
                    extra={
                        "run_id": run_id,
                        "algorithm": cand.algorithm,
                        "metrics": {k: round(v, 4) for k, v in list(cand.metrics.items())[:4]},
                    },
                )
            except Exception as exc:  # one bad candidate must not sink the run
                if exc.__class__.__name__ == "JobCancelled":
                    raise
                cand.status = "failed"
                cand.error = _message(exc)[:400]
                cand.duration_seconds = round(time.perf_counter() - t0, 2)
                logger.warning(
                    "automl.candidate_failed",
                    extra={
                        "run_id": run_id,
                        "algorithm": cand.algorithm,
                        "error": str(exc)[:200],
                    },
                )
            store.update(run_id, candidates=[c.to_dict() for c in candidates])

        store.update(run_id, status=RANKING)
        ranked = rank_candidates(candidates, run["primary_metric"])
        store.update(run_id, candidates=[c.to_dict() for c in candidates])

        if not ranked:
            store.update(
                run_id,
                status=FAILED,
                error="every candidate failed to train; see the individual errors",
                completed_at=_now(),
                duration_seconds=round(time.perf_counter() - started, 2),
            )
            audit(
                "automl.run_failed",
                "automl_run",
                run_id,
                outcome="failure",
                reason="no candidate succeeded",
            )
            return FAILED

        best = ranked[0]

        # The winner goes through the ordinary gate. AutoML picks a candidate;
        # it does not decide what reaches production.
        promotion: dict[str, Any] = {}
        best_result = results.get(best.algorithm)
        if best_result is not None:
            outcome = register_and_promote(
                best_result,
                target_stage=ModelStage(run["target_stage"]),
                compare=True,
                settings=settings,
                actor="automl",
            )
            promotion = {
                "decision": outcome.approval.decision.value,
                "promoted": outcome.promoted,
                "final_stage": outcome.final_stage.value,
                "reason": outcome.reason,
                "failed_checks": [c.name for c in outcome.approval.failed_checks],
                "comparison": (
                    outcome.comparison.model_dump(mode="json") if outcome.comparison else None
                ),
            }

        failures = [c for c in candidates if c.status == "failed"]
        status = COMPLETED_WITH_WARNINGS if failures else COMPLETED
        store.update(
            run_id,
            status=status,
            best_algorithm=best.algorithm,
            best_model_version=best.model_version,
            promotion=json.dumps(promotion, default=str),
            ranking_rule=RANKING_RULE,
            completed_at=_now(),
            duration_seconds=round(time.perf_counter() - started, 2),
        )
        audit(
            "automl.run_completed",
            "automl_run",
            run_id,
            outcome="success",
            best_algorithm=best.algorithm,
            best_model_version=best.model_version,
            candidates_failed=len(failures),
            promoted=promotion.get("promoted"),
        )
        logger.info(
            "automl.run_finished",
            extra={
                "run_id": run_id,
                "status": status,
                "best": best.algorithm,
                "promoted": promotion.get("promoted"),
            },
        )
        return status

    except AutoMLError as exc:
        store.update(
            run_id,
            status=FAILED,
            error=str(exc),
            completed_at=_now(),
            duration_seconds=round(time.perf_counter() - started, 2),
        )
        audit("automl.run_failed", "automl_run", run_id, outcome="failure", error=str(exc))
        logger.warning("automl.run_rejected", extra={"run_id": run_id, "error": str(exc)})
        return FAILED
    except Exception as exc:
        if exc.__class__.__name__ == "JobCancelled":
            raise
        logger.exception("automl.run_failed", extra={"run_id": run_id})
        store.update(
            run_id,
            status=FAILED,
            error=_message(exc)[:500],
            completed_at=_now(),
            duration_seconds=round(time.perf_counter() - started, 2),
        )
        audit(
            "automl.run_failed",
            "automl_run",
            run_id,
            outcome="failure",
            error=_message(exc)[:200],
        )
        return FAILED


def _message(exc: Exception) -> str:
    """The human sentence of an error, without a platform error's detail dump."""
    return str(getattr(exc, "message", None) or exc)
