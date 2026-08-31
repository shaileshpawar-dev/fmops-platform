"""Approval gate and champion/challenger comparison tests.

The rule these protect: a model reaches production only by clearing an absolute
quality bar AND beating the incumbent by a configured margin. A regression here
would let a worse model replace a working one, which is the single most
expensive failure mode this platform exists to prevent.
"""

from __future__ import annotations

import pytest

from app.core.exceptions import InvalidStageTransitionError
from app.registry.base import ALLOWED_TRANSITIONS, assert_transition
from app.schemas.common import ApprovalDecision, ModelStage
from app.schemas.model import Metrics
from app.training.approval import (
    compare_to_production,
    evaluate_approval,
    promote_if_eligible,
)

pytestmark = pytest.mark.unit


def good_metrics(**overrides) -> Metrics:
    base = {
        "accuracy": 0.85,
        "precision": 0.62,
        "recall": 0.70,
        "f1": 0.66,
        "roc_auc": 0.88,
        "pr_auc": 0.68,
        "inference_latency_p95_ms": 20.0,
    }
    base.update(overrides)
    return Metrics(**base)


# --------------------------------------------------------------------------- #
# Absolute gate
# --------------------------------------------------------------------------- #
def test_good_model_is_approved(settings):
    result = evaluate_approval(good_metrics(), "m", 1, settings=settings)
    assert result.approved
    assert result.decision == ApprovalDecision.APPROVED
    assert result.failed_checks == []


def test_low_f1_is_rejected(settings):
    config = settings.approval.model_copy(update={"min_f1": 0.80})
    result = evaluate_approval(good_metrics(f1=0.55), "m", 1, config=config, settings=settings)
    assert not result.approved
    assert "f1" in [c.name for c in result.failed_checks]


def test_low_roc_auc_is_rejected(settings):
    config = settings.approval.model_copy(update={"min_roc_auc": 0.95})
    result = evaluate_approval(good_metrics(), "m", 1, config=config, settings=settings)
    assert not result.approved
    assert "roc_auc" in [c.name for c in result.failed_checks]


def test_slow_model_is_rejected_on_latency(settings):
    config = settings.approval.model_copy(update={"max_inference_latency_ms": 10.0})
    result = evaluate_approval(
        good_metrics(inference_latency_p95_ms=900.0), "m", 1, config=config, settings=settings
    )
    assert not result.approved
    assert "inference_latency_p95_ms" in [c.name for c in result.failed_checks]


def test_failed_data_validation_blocks_approval(settings):
    result = evaluate_approval(
        good_metrics(), "m", 1, validation_passed=False, settings=settings
    )
    assert not result.approved
    assert "data_validation" in [c.name for c in result.failed_checks]


def test_excessive_drift_blocks_approval(settings):
    config = settings.approval.model_copy(update={"max_drift_score": 0.10})
    result = evaluate_approval(
        good_metrics(), "m", 1, drift_score=0.55, config=config, settings=settings
    )
    assert not result.approved
    assert "drift_score" in [c.name for c in result.failed_checks]


def test_manual_approval_holds_the_model_back(settings):
    config = settings.approval.model_copy(update={"require_manual_approval": True})
    result = evaluate_approval(good_metrics(), "m", 1, config=config, settings=settings)
    assert result.decision == ApprovalDecision.PENDING_MANUAL
    assert not result.approved
    assert result.failed_checks == [], "it passed the checks; it just needs a human"


def test_disabled_gate_approves_but_says_so(settings):
    config = settings.approval.model_copy(update={"enabled": False})
    result = evaluate_approval(
        good_metrics(f1=0.01, roc_auc=0.01), "m", 1, config=config, settings=settings
    )
    assert result.approved
    assert "disabled" in result.reason


def test_every_check_is_recorded_for_audit(settings):
    result = evaluate_approval(good_metrics(), "m", 1, settings=settings)
    names = {c.name for c in result.checks}
    assert {"f1", "roc_auc", "inference_latency_p95_ms", "data_validation"} <= names
    for check in result.checks:
        assert check.observed is not None
        assert check.threshold is not None


def test_render_text_shows_pass_and_fail(settings):
    config = settings.approval.model_copy(update={"min_f1": 0.99})
    result = evaluate_approval(good_metrics(), "m", 1, config=config, settings=settings)
    text = result.render_text()
    assert "REJECTED" in text
    assert "[FAIL] f1" in text
    assert "[PASS] roc_auc" in text


# --------------------------------------------------------------------------- #
# Champion / challenger
# --------------------------------------------------------------------------- #
def test_no_baseline_means_the_candidate_wins(settings):
    comparison = compare_to_production(
        good_metrics(), None, candidate_version=1, settings=settings
    )
    assert comparison.candidate_is_better
    assert comparison.decision == "promote"
    assert "no production model" in comparison.reason


def test_clearly_better_candidate_is_promoted(settings):
    comparison = compare_to_production(
        good_metrics(roc_auc=0.92),
        good_metrics(roc_auc=0.85).as_dict(),
        candidate_version=2,
        baseline_version=1,
        settings=settings,
    )
    assert comparison.candidate_is_better
    assert comparison.improvement == pytest.approx(0.07, abs=1e-6)


def test_worse_candidate_is_rejected(settings):
    """The headline behaviour: a worse model must not replace production."""
    comparison = compare_to_production(
        good_metrics(roc_auc=0.86),
        good_metrics(roc_auc=0.88).as_dict(),
        candidate_version=2,
        baseline_version=1,
        settings=settings,
    )
    assert not comparison.candidate_is_better
    assert comparison.decision == "reject"
    assert comparison.improvement < 0
    assert "keeping the production model" in comparison.reason


def test_marginally_better_candidate_is_still_rejected(settings):
    """Beating production by less than min_improvement is not enough.

    Without this rule, ordinary run-to-run noise would churn production models.
    """
    config = settings.approval.model_copy(update={"min_improvement": 0.01})
    comparison = compare_to_production(
        good_metrics(roc_auc=0.8802),
        good_metrics(roc_auc=0.8800).as_dict(),
        candidate_version=2,
        baseline_version=1,
        config=config,
        settings=settings,
    )
    assert not comparison.candidate_is_better
    assert comparison.improvement == pytest.approx(0.0002, abs=1e-6)


def test_identical_scores_are_rejected(settings):
    comparison = compare_to_production(
        good_metrics(roc_auc=0.88),
        good_metrics(roc_auc=0.88).as_dict(),
        candidate_version=2,
        baseline_version=1,
        settings=settings,
    )
    assert not comparison.candidate_is_better


def test_comparison_metric_is_configurable(settings):
    config = settings.approval.model_copy(
        update={"comparison_metric": "f1", "min_improvement": 0.01}
    )
    # Better F1, worse ROC-AUC: with metric=f1 the candidate should win.
    comparison = compare_to_production(
        good_metrics(f1=0.75, roc_auc=0.80),
        good_metrics(f1=0.66, roc_auc=0.95).as_dict(),
        candidate_version=2,
        baseline_version=1,
        config=config,
        settings=settings,
    )
    assert comparison.metric == "f1"
    assert comparison.candidate_is_better


# --------------------------------------------------------------------------- #
# Stage machine
# --------------------------------------------------------------------------- #
def test_legal_transitions_are_allowed():
    assert_transition(ModelStage.DEVELOPMENT, ModelStage.VALIDATION)
    assert_transition(ModelStage.VALIDATION, ModelStage.STAGING)
    assert_transition(ModelStage.STAGING, ModelStage.PRODUCTION)
    assert_transition(ModelStage.PRODUCTION, ModelStage.STAGING)  # rollback
    assert_transition(ModelStage.PRODUCTION, ModelStage.ARCHIVED)


def test_stage_jumps_are_rejected():
    with pytest.raises(InvalidStageTransitionError) as excinfo:
        assert_transition(ModelStage.DEVELOPMENT, ModelStage.PRODUCTION)
    assert "legal moves" in str(excinfo.value)
    assert excinfo.value.details["allowed"]


def test_archived_can_only_be_reinstated_to_development():
    assert_transition(ModelStage.ARCHIVED, ModelStage.DEVELOPMENT)
    with pytest.raises(InvalidStageTransitionError):
        assert_transition(ModelStage.ARCHIVED, ModelStage.PRODUCTION)


def test_same_stage_is_a_no_op():
    assert_transition(ModelStage.PRODUCTION, ModelStage.PRODUCTION)


def test_every_stage_has_a_transition_rule():
    for stage in ModelStage:
        assert stage in ALLOWED_TRANSITIONS


# --------------------------------------------------------------------------- #
# Promotion
# --------------------------------------------------------------------------- #
def test_promotion_walks_the_full_chain(registry):
    version = registry.register("m", "file:///tmp/m.joblib", metrics=good_metrics().as_dict())
    approval = evaluate_approval(good_metrics(), "m", version.version)
    comparison = compare_to_production(good_metrics(), None, candidate_version=version.version)

    promoted = promote_if_eligible("m", version.version, approval, comparison, registry)
    assert promoted.stage == ModelStage.PRODUCTION

    stages = [t.to_stage for t in registry.history("m", version.version)]
    assert ModelStage.VALIDATION in stages
    assert ModelStage.STAGING in stages
    assert ModelStage.PRODUCTION in stages


def test_rejected_model_is_not_promoted(registry, settings):
    version = registry.register("m", "file:///tmp/m.joblib", metrics=good_metrics().as_dict())
    config = settings.approval.model_copy(update={"min_f1": 0.99})
    approval = evaluate_approval(good_metrics(), "m", version.version, config=config)

    result = promote_if_eligible("m", version.version, approval, None, registry)
    assert result.stage == ModelStage.DEVELOPMENT
    assert result.status.value == "rejected"


def test_worse_candidate_is_not_promoted(registry):
    baseline = registry.register(
        "m", "file:///a", metrics=good_metrics(roc_auc=0.90).as_dict()
    )
    registry.transition_stage("m", baseline.version, ModelStage.VALIDATION)
    registry.transition_stage("m", baseline.version, ModelStage.STAGING)
    registry.transition_stage("m", baseline.version, ModelStage.PRODUCTION)

    candidate = registry.register(
        "m", "file:///b", metrics=good_metrics(roc_auc=0.80).as_dict()
    )
    approval = evaluate_approval(good_metrics(roc_auc=0.80), "m", candidate.version)
    comparison = compare_to_production(
        good_metrics(roc_auc=0.80),
        good_metrics(roc_auc=0.90).as_dict(),
        candidate_version=candidate.version,
        baseline_version=baseline.version,
    )

    result = promote_if_eligible("m", candidate.version, approval, comparison, registry)
    assert result.stage == ModelStage.DEVELOPMENT
    assert registry.get_production("m").version == baseline.version


def test_promoting_to_production_archives_the_incumbent(registry):
    first = registry.register("m", "file:///a", metrics=good_metrics().as_dict())
    for stage in (ModelStage.VALIDATION, ModelStage.STAGING, ModelStage.PRODUCTION):
        registry.transition_stage("m", first.version, stage)

    second = registry.register("m", "file:///b", metrics=good_metrics(roc_auc=0.95).as_dict())
    for stage in (ModelStage.VALIDATION, ModelStage.STAGING, ModelStage.PRODUCTION):
        registry.transition_stage("m", second.version, stage)

    assert registry.get("m", first.version).stage == ModelStage.ARCHIVED
    assert registry.get_production("m").version == second.version
    # The displaced version becomes the rollback target.
    assert registry.previous_production("m").version == first.version
