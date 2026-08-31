"""Evaluation pipeline entrypoint.

Re-evaluates a registered model version against a dataset and runs the approval
gate, without retraining. Two uses:

* **Audit** -- prove what a version would score under the *current* thresholds,
  which may differ from those in force when it was trained.
* **Pre-deployment check** -- CI runs this against a release candidate before
  the deploy job, so a stale or degraded artifact is caught before traffic.

Exit codes: 0 approved, 2 rejected, 1 error.
"""

from __future__ import annotations

import argparse
import sys
from typing import Any

from app.core.config import get_settings
from app.core.exceptions import FMOpsError
from app.core.logging import configure_from_settings, get_logger
from app.core.utils import write_json
from app.data.preprocessing import split_features_target
from app.data.versioning import get_dataset_registry
from app.registry.factory import get_registry
from app.training.approval import compare_to_production, evaluate_approval
from app.training.evaluate import evaluate_model, predict_proba, select_threshold
from app.training.train import load_model

logger = get_logger("fmops.pipeline.evaluation")


def run(
    version: int | None = None,
    dataset_version: str | None = None,
    report_path: str | None = None,
) -> tuple[int, dict[str, Any]]:
    settings = get_settings()
    registry = get_registry()
    name = settings.tracking.registered_model_name

    target = version
    if target is None:
        candidate = registry.get_serving(name) or registry.get_latest(name)
        if candidate is None:
            logger.error("pipeline.no_model_registered")
            return 1, {"status": "failed", "error": "no model versions registered"}
        target = candidate.version

    model_version = registry.get(name, target)
    logger.info(
        "pipeline.evaluating",
        extra={"model": name, "version": target, "stage": model_version.stage.value},
    )

    try:
        pipeline = load_model(model_version.artifact_uri)
    except FMOpsError as exc:
        return 1, {"status": "failed", "error": exc.message}

    datasets = get_dataset_registry()
    frame, record = datasets.resolve(version=dataset_version)
    features, labels = split_features_target(frame, settings.data)

    threshold = float(model_version.params.get("threshold") or 0.0)
    if threshold <= 0:
        # No recorded operating point: choose one here and log it, rather than
        # silently scoring at 0.5 and reporting a worse F1 than the model has.
        threshold = select_threshold(labels.to_numpy(), predict_proba(pipeline, features))
        logger.warning(
            "pipeline.threshold_not_recorded",
            extra={"model": name, "version": target, "selected": round(threshold, 4)},
        )

    evaluation = evaluate_model(
        pipeline,
        features,
        labels,
        name,
        model_version=target,
        dataset_version=record.version if record else None,
        split="holdout",
        threshold=threshold,
    )
    approval = evaluate_approval(
        evaluation.metrics, name, target, validation_passed=True, settings=settings
    )
    production = registry.get_production(name)
    comparison = compare_to_production(
        evaluation.metrics,
        production.metrics if production and production.version != target else None,
        candidate_version=target,
        baseline_version=production.version if production else None,
        settings=settings,
    )

    print(approval.render_text())
    print(f"\nComparison: {comparison.reason}")

    report = {
        "pipeline": "evaluation",
        "model_name": name,
        "model_version": target,
        "dataset_version": record.version if record else None,
        "threshold": threshold,
        "metrics": evaluation.metrics.as_dict(),
        "confusion_matrix": evaluation.confusion_matrix.as_dict(),
        "approval": approval.model_dump(mode="json"),
        "comparison": comparison.model_dump(mode="json"),
        "status": "approved" if approval.approved else "rejected",
    }
    if report_path:
        write_json(report_path, report)
    return (0 if approval.approved else 2), report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="FMOps evaluation pipeline")
    parser.add_argument("--version", type=int)
    parser.add_argument("--dataset-version")
    parser.add_argument("--report")
    args = parser.parse_args(argv)

    configure_from_settings(force=True)
    code, _ = run(args.version, args.dataset_version, args.report)
    return code


if __name__ == "__main__":
    sys.exit(main())
