"""Approval gates.

A model does **not** reach production because training succeeded. It reaches
production because it cleared an explicit, configurable set of checks, each of
which is recorded on the resulting :class:`~app.schemas.evaluation.ApprovalResult`
so the decision can be explained months later.

Two gates live here:

:func:`evaluate_approval`
    Absolute quality bar. F1, ROC-AUC, precision, recall, inference latency,
    clean data validation, and acceptable drift.

:func:`compare_to_production`
    Relative bar (champion vs challenger). A retrained model must beat the
    incumbent on the configured metric by at least ``min_improvement``. This is
    what stops a worse model from replacing a working one.

Both thresholds come from :class:`~app.core.config.ApprovalConfig`, so
development, staging and production can hold models to different bars.
"""

from __future__ import annotations

from app.core.config import ApprovalConfig, Settings, get_settings
from app.core.logging import get_logger
from app.registry.base import ModelRegistry
from app.schemas.common import ApprovalDecision, ModelStage
from app.schemas.evaluation import ApprovalResult, GateCheck, ModelComparison
from app.schemas.model import EvaluationResult, Metrics, ModelVersion

logger = get_logger(__name__)


def _check(
    name: str,
    observed: float,
    threshold: float,
    comparison: str = ">=",
    blocking: bool = True,
) -> GateCheck:
    passed = observed >= threshold if comparison == ">=" else observed <= threshold
    return GateCheck(
        name=name,
        passed=passed,
        observed=round(float(observed), 6),
        threshold=round(float(threshold), 6),
        blocking=blocking,
        message="" if passed else f"{observed:.4f} fails {comparison} {threshold:.4f}",
    )


def evaluate_approval(
    metrics: Metrics | dict[str, float],
    model_name: str,
    model_version: int | None = None,
    validation_passed: bool = True,
    drift_score: float | None = None,
    config: ApprovalConfig | None = None,
    settings: Settings | None = None,
) -> ApprovalResult:
    """Run the absolute quality gate over a candidate model."""
    settings = settings or get_settings()
    config = config or settings.approval

    values = metrics.as_dict() if isinstance(metrics, Metrics) else dict(metrics)

    if not config.enabled:
        logger.warning(
            "approval.gate_disabled",
            extra={"model": model_name, "version": model_version},
        )
        return ApprovalResult(
            model_name=model_name,
            model_version=model_version,
            decision=ApprovalDecision.APPROVED,
            reason="approval gate disabled by configuration",
            evaluated_metrics=values,
        )

    checks: list[GateCheck] = [
        _check("f1", values.get("f1", 0.0), config.min_f1),
        _check("roc_auc", values.get("roc_auc", 0.0), config.min_roc_auc),
    ]
    if config.min_precision > 0:
        checks.append(_check("precision", values.get("precision", 0.0), config.min_precision))
    if config.min_recall > 0:
        checks.append(_check("recall", values.get("recall", 0.0), config.min_recall))

    checks.append(
        _check(
            "inference_latency_p95_ms",
            values.get("inference_latency_p95_ms", 0.0),
            config.max_inference_latency_ms,
            comparison="<=",
        )
    )

    if config.require_clean_validation:
        checks.append(
            GateCheck(
                name="data_validation",
                passed=bool(validation_passed),
                observed="passed" if validation_passed else "failed",
                threshold="passed",
                message=(
                    ""
                    if validation_passed
                    else "the training dataset did not clear the validation suite"
                ),
            )
        )

    if drift_score is not None:
        checks.append(
            _check(
                "drift_score",
                drift_score,
                config.max_drift_score,
                comparison="<=",
            )
        )

    failed = [c for c in checks if not c.passed and c.blocking]
    if failed:
        decision = ApprovalDecision.REJECTED
        reason = "failed gate checks: " + ", ".join(c.name for c in failed)
    elif config.require_manual_approval:
        decision = ApprovalDecision.PENDING_MANUAL
        reason = (
            "all automated checks passed; manual approval is required in this "
            "environment before promotion to Production"
        )
    else:
        decision = ApprovalDecision.APPROVED
        reason = "all gate checks passed"

    result = ApprovalResult(
        model_name=model_name,
        model_version=model_version,
        decision=decision,
        checks=checks,
        reason=reason,
        evaluated_metrics=values,
    )
    logger.info(
        "approval.evaluated",
        extra={
            "model": model_name,
            "version": model_version,
            "decision": decision.value,
            "failed_checks": [c.name for c in failed],
            "reason": reason,
        },
    )
    return result


def compare_to_production(
    candidate_metrics: Metrics | dict[str, float],
    baseline_metrics: Metrics | dict[str, float] | None,
    candidate_version: int | None = None,
    baseline_version: int | None = None,
    config: ApprovalConfig | None = None,
    settings: Settings | None = None,
) -> ModelComparison:
    """Champion vs challenger on the configured comparison metric.

    With no incumbent the candidate wins by default -- there is nothing to
    protect. With an incumbent, the candidate must exceed it by at least
    ``min_improvement``; a tie or a regression keeps the incumbent live.
    """
    settings = settings or get_settings()
    config = config or settings.approval
    metric = config.comparison_metric

    candidate_values = (
        candidate_metrics.as_dict()
        if isinstance(candidate_metrics, Metrics)
        else dict(candidate_metrics)
    )
    candidate_score = float(candidate_values.get(metric, 0.0))

    if baseline_metrics is None:
        comparison = ModelComparison(
            metric=metric,
            baseline_version=baseline_version,
            candidate_version=candidate_version,
            baseline_score=0.0,
            candidate_score=candidate_score,
            improvement=candidate_score,
            min_improvement=config.min_improvement,
            candidate_is_better=True,
            decision="promote",
            reason="no production model exists; the candidate becomes the baseline",
        )
        logger.info(
            "approval.comparison",
            extra={
                "metric": metric,
                "candidate": round(candidate_score, 5),
                "baseline": None,
                "decision": "promote",
            },
        )
        return comparison

    baseline_values = (
        baseline_metrics.as_dict()
        if isinstance(baseline_metrics, Metrics)
        else dict(baseline_metrics)
    )
    baseline_score = float(baseline_values.get(metric, 0.0))
    improvement = candidate_score - baseline_score
    is_better = improvement >= config.min_improvement

    if is_better:
        reason = (
            f"candidate improves {metric} by {improvement:+.4f} "
            f"(>= required {config.min_improvement:.4f})"
        )
        decision = "promote"
    else:
        reason = (
            f"candidate {metric} {candidate_score:.4f} does not beat production "
            f"{baseline_score:.4f} by the required margin "
            f"{config.min_improvement:.4f} (actual {improvement:+.4f}); "
            "keeping the production model"
        )
        decision = "reject"

    comparison = ModelComparison(
        metric=metric,
        baseline_version=baseline_version,
        candidate_version=candidate_version,
        baseline_score=round(baseline_score, 6),
        candidate_score=round(candidate_score, 6),
        improvement=round(improvement, 6),
        min_improvement=config.min_improvement,
        candidate_is_better=is_better,
        decision=decision,
        reason=reason,
    )
    logger.info(
        "approval.comparison",
        extra={
            "metric": metric,
            "candidate_version": candidate_version,
            "baseline_version": baseline_version,
            "candidate": round(candidate_score, 5),
            "baseline": round(baseline_score, 5),
            "improvement": round(improvement, 5),
            "decision": decision,
        },
    )
    return comparison


def approve_and_compare(
    evaluation: EvaluationResult,
    model_name: str,
    model_version: int,
    registry: ModelRegistry,
    validation_passed: bool = True,
    drift_score: float | None = None,
    settings: Settings | None = None,
) -> tuple[ApprovalResult, ModelComparison]:
    """Run both gates against the current production model, if any."""
    settings = settings or get_settings()
    approval = evaluate_approval(
        evaluation.metrics,
        model_name,
        model_version,
        validation_passed=validation_passed,
        drift_score=drift_score,
        settings=settings,
    )
    from app.training.holdout import compare_on_shared_holdout

    comparison = compare_on_shared_holdout(
        registry.get(model_name, model_version),
        registry.get_production(model_name),
        settings=settings,
    )
    return approval, comparison


def promote_if_eligible(
    model_name: str,
    model_version: int,
    approval: ApprovalResult,
    comparison: ModelComparison | None,
    registry: ModelRegistry,
    target_stage: ModelStage = ModelStage.PRODUCTION,
    actor: str = "system",
) -> ModelVersion:
    """Walk an approved model up the stage chain, or leave it where it is.

    Promotion is stepwise (Development -> Validation -> Staging -> Production)
    because the stage machine forbids jumps; each hop is recorded in the
    transition history.
    """
    current = registry.get(model_name, model_version)

    if not approval.approved:
        registry.update_status(
            model_name,
            model_version,
            "rejected" if approval.decision.value == "rejected" else "pending",
        )
        logger.info(
            "approval.promotion_skipped",
            extra={
                "model": model_name,
                "version": model_version,
                "decision": approval.decision.value,
                "reason": approval.reason,
            },
        )
        # Re-read: ``current`` was fetched before update_status, so returning it
        # would report the pre-rejection status to the caller.
        return registry.get(model_name, model_version)

    if comparison is not None and not comparison.candidate_is_better:
        registry.update_status(model_name, model_version, "rejected")
        registry.set_tags(
            model_name,
            model_version,
            {"rejected_reason": "did not beat production baseline"},
        )
        logger.info(
            "approval.promotion_skipped",
            extra={
                "model": model_name,
                "version": model_version,
                "reason": comparison.reason,
            },
        )
        return registry.get(model_name, model_version)

    chain = [
        ModelStage.DEVELOPMENT,
        ModelStage.VALIDATION,
        ModelStage.STAGING,
        ModelStage.PRODUCTION,
    ]
    try:
        start = chain.index(current.stage)
        end = chain.index(target_stage)
    except ValueError:
        start, end = 0, chain.index(target_stage)

    updated = current
    for stage in chain[start + 1 : end + 1]:
        updated = registry.transition_stage(
            model_name,
            model_version,
            stage,
            reason=approval.reason,
            actor=actor,
        )
    registry.update_status(model_name, model_version, "approved")
    logger.info(
        "approval.promoted",
        extra={
            "model": model_name,
            "version": model_version,
            "from_stage": current.stage.value,
            "to_stage": updated.stage.value,
        },
    )
    return registry.get(model_name, model_version)
