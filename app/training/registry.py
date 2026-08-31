"""Training-side registry helpers.

Thin orchestration over :mod:`app.registry` and :mod:`app.training.approval`:
the pipeline calls :func:`register_and_promote` and gets back a single decision
object describing what happened and why.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.core.config import Settings, get_settings
from app.core.logging import get_logger
from app.registry.base import ModelRegistry
from app.registry.factory import get_registry
from app.schemas.common import ModelStage
from app.schemas.evaluation import ApprovalResult, ModelComparison
from app.schemas.model import ModelVersion, TrainingRunResult
from app.training.approval import approve_and_compare, promote_if_eligible

logger = get_logger(__name__)


@dataclass
class PromotionOutcome:
    """What the gates decided about one trained model."""

    model_version: ModelVersion
    approval: ApprovalResult
    comparison: ModelComparison | None
    promoted: bool
    final_stage: ModelStage
    reason: str

    def render_text(self) -> str:
        lines = [
            self.approval.render_text(),
            "",
            f"Promotion: {'YES' if self.promoted else 'NO'} "
            f"-> stage {self.final_stage.value}",
            f"  reason: {self.reason}",
        ]
        if self.comparison:
            lines.insert(
                1,
                f"Comparison ({self.comparison.metric}): candidate "
                f"{self.comparison.candidate_score:.4f} vs baseline "
                f"{self.comparison.baseline_score:.4f} "
                f"({self.comparison.improvement:+.4f}) -> "
                f"{self.comparison.decision}",
            )
        return "\n".join(lines)


def register_and_promote(
    run: TrainingRunResult,
    registry: ModelRegistry | None = None,
    target_stage: ModelStage = ModelStage.PRODUCTION,
    drift_score: float | None = None,
    compare: bool = True,
    settings: Settings | None = None,
    actor: str = "system",
) -> PromotionOutcome:
    """Run both approval gates over a finished training run and act on them."""
    settings = settings or get_settings()
    registry = registry or get_registry()

    if run.registered_version is None:
        raise ValueError("training run was not registered; re-run with register_model=True")
    if run.evaluation is None:
        raise ValueError("training run has no evaluation result to gate on")

    approval, comparison = approve_and_compare(
        run.evaluation,
        run.model_name,
        run.registered_version,
        registry,
        validation_passed=run.validation_passed,
        drift_score=drift_score,
        settings=settings,
    )
    effective_comparison = comparison if compare else None

    before = registry.get(run.model_name, run.registered_version)
    after = promote_if_eligible(
        run.model_name,
        run.registered_version,
        approval,
        effective_comparison,
        registry,
        target_stage=target_stage,
        actor=actor,
    )
    promoted = after.stage != before.stage

    if promoted or not approval.approved:
        reason = approval.reason
    elif effective_comparison and not effective_comparison.candidate_is_better:
        reason = effective_comparison.reason
    else:
        reason = "model already at the target stage"

    return PromotionOutcome(
        model_version=after,
        approval=approval,
        comparison=comparison,
        promoted=promoted,
        final_stage=after.stage,
        reason=reason,
    )
