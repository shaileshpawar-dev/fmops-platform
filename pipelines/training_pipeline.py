"""Training pipeline entrypoint.

The unit CI and SageMaker Pipelines invoke. It composes the stages that already
exist as library code -- it does not reimplement them -- and exits non-zero on
any gate failure so a pipeline runner can act on it.

Exit codes:

    0  model trained, and (if requested) promoted
    1  a stage failed: bad data, training error, registry unavailable
    2  the model was trained but rejected by the approval gate

Distinguishing 1 from 2 matters in CI: a rejected model is a *successful*
pipeline run that made a correct negative decision, and should not page anyone.
"""

from __future__ import annotations

import argparse
import sys
from typing import Any

from app.core.config import get_settings
from app.core.exceptions import DataValidationError, FMOpsError
from app.core.logging import configure_from_settings, get_logger
from app.core.utils import write_json
from app.schemas.common import ModelStage
from app.schemas.model import TrainingRequest
from app.training.registry import register_and_promote
from app.training.train import train_model

logger = get_logger("fmops.pipeline.training")


def run(
    dataset_version: str | None = None,
    dataset_path: str | None = None,
    algorithm: str | None = None,
    tune: bool | None = None,
    promote: bool = False,
    target_stage: str = "Production",
    compare: bool = True,
    report_path: str | None = None,
) -> tuple[int, dict[str, Any]]:
    settings = get_settings()
    report: dict[str, Any] = {"pipeline": "training", "environment": settings.environment}

    try:
        result = train_model(
            TrainingRequest(
                dataset_version=dataset_version,
                dataset_path=dataset_path,
                algorithm=algorithm,
                tune=tune,
            )
        )
    except DataValidationError as exc:
        logger.error(
            "pipeline.stopped_at_validation_gate",
            extra={"failed": exc.details.get("failed", [])},
        )
        report.update(
            stage="data_validation",
            status="failed",
            error=exc.message,
            failed_expectations=exc.details.get("failed", []),
        )
        _write(report, report_path)
        return 1, report
    except FMOpsError as exc:
        logger.error("pipeline.training_failed", extra={"error": exc.message})
        report.update(stage="training", status="failed", error=exc.message)
        _write(report, report_path)
        return 1, report

    report.update(
        stage="training",
        status="succeeded",
        run_id=result.run_id,
        model_version=result.registered_version,
        algorithm=result.algorithm,
        dataset_version=result.dataset_version,
        git_commit=result.git_commit,
        duration_seconds=result.duration_seconds,
        metrics=result.evaluation.metrics.as_dict() if result.evaluation else {},
        threshold=result.evaluation.threshold if result.evaluation else None,
        tuning=(
            {
                "backend": result.tuning.backend,
                "trials": result.tuning.n_trials,
                "best_score": result.tuning.best_score,
                "best_params": result.tuning.best_params,
            }
            if result.tuning
            else None
        ),
    )

    if not promote:
        _write(report, report_path)
        return 0, report

    outcome = register_and_promote(
        result, target_stage=ModelStage(target_stage), compare=compare
    )
    report.update(
        stage="approval",
        promoted=outcome.promoted,
        final_stage=outcome.final_stage.value,
        approval_decision=outcome.approval.decision.value,
        approval_reason=outcome.reason,
        failed_checks=[c.name for c in outcome.approval.failed_checks],
        comparison=(
            outcome.comparison.model_dump(mode="json") if outcome.comparison else None
        ),
    )
    print(outcome.render_text())
    _write(report, report_path)

    if outcome.promoted:
        return 0, report
    if outcome.approval.approved:
        # Approved on absolute quality but did not beat production. A correct
        # decision, not a failure.
        logger.info("pipeline.candidate_not_promoted", extra={"reason": outcome.reason})
        return 0, report
    return 2, report


def _write(report: dict[str, Any], path: str | None) -> None:
    if path:
        write_json(path, report)
        logger.info("pipeline.report_written", extra={"path": path})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="FMOps training pipeline")
    parser.add_argument("--dataset-version")
    parser.add_argument("--dataset-path")
    parser.add_argument("--algorithm")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--tune", dest="tune", action="store_true", default=None)
    group.add_argument("--no-tune", dest="tune", action="store_false")
    parser.add_argument("--promote", action="store_true")
    parser.add_argument("--target-stage", default="Production")
    parser.add_argument("--no-compare", action="store_true")
    parser.add_argument("--report", help="write a JSON run report here")
    args = parser.parse_args(argv)

    configure_from_settings(force=True)
    code, _report = run(
        dataset_version=args.dataset_version,
        dataset_path=args.dataset_path,
        algorithm=args.algorithm,
        tune=args.tune,
        promote=args.promote,
        target_stage=args.target_stage,
        compare=not args.no_compare,
        report_path=args.report,
    )
    return code


if __name__ == "__main__":
    sys.exit(main())
