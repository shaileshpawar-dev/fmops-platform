"""The automatic retraining pipeline.

    drift / performance / volume trigger
              |
              v
      create retraining event
              |
              v
      collect the latest data  (training set + labelled production traffic)
              |
              v
      validate  -- invalid data aborts here, the production model keeps serving
              |
              v
      train + tune + evaluate   (the standard training pipeline)
              |
              v
      compare against the production model
              |
       +------+------+
       |             |
   better?        worse or equal?
       |             |
       v             v
  approval gate   REJECT: keep the production model,
       |          archive the candidate with the reason
       v
  deploy + monitor

The rule that matters: **a new model never replaces the incumbent just because
it is new.** It must clear the absolute approval gate *and* beat production on
the configured metric by at least ``min_improvement``. Both decisions, with
their numbers, are recorded on the retraining event.
"""

from __future__ import annotations

import time
from typing import Any

import pandas as pd

from app.core.config import Settings, get_settings
from app.core.exceptions import DataValidationError, FMOpsError
from app.core.logging import StageTimer, get_logger, log_context
from app.data.generator import build_production_dataset
from app.data.versioning import get_dataset_registry
from app.deployment.manager import DeploymentManager, get_deployment_manager
from app.monitoring.alerts import get_alert_manager
from app.monitoring.inference_log import get_inference_log
from app.monitoring.metrics import get_metrics
from app.registry.base import ModelRegistry
from app.registry.factory import get_registry
from app.retraining.trigger import (
    RetrainingEventStore,
    TriggerDecision,
    evaluate_trigger,
    get_event_store,
)
from app.schemas.common import (
    AlertCategory,
    DeploymentStrategy,
    ModelStage,
    RetrainingStatus,
    RetrainingTrigger,
    Severity,
)
from app.schemas.deployment import DeploymentRequest
from app.schemas.evaluation import RetrainingDecision
from app.schemas.model import TrainingRequest
from app.training.approval import compare_to_production, evaluate_approval
from app.training.train import train_model

logger = get_logger(__name__)


class RetrainingPipeline:
    """Runs one retraining cycle end to end."""

    def __init__(
        self,
        settings: Settings | None = None,
        registry: ModelRegistry | None = None,
        store: RetrainingEventStore | None = None,
        deployment_manager: DeploymentManager | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self._registry = registry
        self.store = store or get_event_store()
        self._deployments = deployment_manager

    @property
    def registry(self) -> ModelRegistry:
        if self._registry is None:
            self._registry = get_registry()
        return self._registry

    @property
    def deployments(self) -> DeploymentManager:
        if self._deployments is None:
            self._deployments = get_deployment_manager()
        return self._deployments

    # -- entry point --------------------------------------------------------- #
    def run(
        self,
        force: bool = False,
        decision: TriggerDecision | None = None,
        deploy: bool | None = None,
        collect_production_data: bool = True,
    ) -> RetrainingDecision:
        model_name = self.settings.tracking.registered_model_name
        decision = decision or evaluate_trigger(force=force)

        if not decision.should_retrain:
            logger.info("retraining.skipped", extra={"reason": decision.reason})
            return RetrainingDecision(
                event_id="",
                triggered=False,
                reason=decision.reason,
                status=RetrainingStatus.SKIPPED,
                message=decision.reason,
            )

        baseline = self.registry.get_production(model_name)
        event = self.store.create(
            trigger=decision.trigger or RetrainingTrigger.MANUAL,
            reason=decision.reason,
            model_name=model_name,
            baseline_version=baseline.version if baseline else None,
            detail=decision.evidence,
        )
        metrics = get_metrics()

        with log_context(retraining_event=event.id, model=model_name):
            self.store.update(event.id, status=RetrainingStatus.RUNNING)
            started = time.perf_counter()
            try:
                result = self._execute(
                    event_id=event.id,
                    model_name=model_name,
                    baseline_version=baseline.version if baseline else None,
                    baseline_metrics=baseline.metrics if baseline else None,
                    trigger=decision.trigger or RetrainingTrigger.MANUAL,
                    reason=decision.reason,
                    deploy=deploy,
                    collect_production_data=collect_production_data,
                )
            except DataValidationError as exc:
                # The retraining data is unusable. Production keeps serving.
                self._fail(
                    event.id,
                    "retraining aborted: the collected data failed validation; "
                    "the production model is unchanged",
                    exc,
                )
                metrics.retraining_events_total.labels(
                    trigger=(decision.trigger or RetrainingTrigger.MANUAL).value,
                    status=RetrainingStatus.FAILED.value,
                    decision="data_validation_failed",
                ).inc()
                return RetrainingDecision(
                    event_id=event.id,
                    triggered=True,
                    trigger=decision.trigger,
                    reason=decision.reason,
                    status=RetrainingStatus.FAILED,
                    production_version=baseline.version if baseline else None,
                    message=(
                        "retraining aborted at the data-validation gate; production "
                        f"version {baseline.version if baseline else 'n/a'} still serving"
                    ),
                )
            except FMOpsError as exc:
                self._fail(event.id, f"retraining failed: {exc.message}", exc)
                metrics.retraining_events_total.labels(
                    trigger=(decision.trigger or RetrainingTrigger.MANUAL).value,
                    status=RetrainingStatus.FAILED.value,
                    decision="error",
                ).inc()
                raise
            except Exception as exc:
                self._fail(event.id, f"retraining failed: {exc}", exc)
                metrics.retraining_events_total.labels(
                    trigger=(decision.trigger or RetrainingTrigger.MANUAL).value,
                    status=RetrainingStatus.FAILED.value,
                    decision="error",
                ).inc()
                raise

            logger.info(
                "retraining.completed",
                extra={
                    "event_id": event.id,
                    "status": result.status.value,
                    "candidate_version": result.candidate_version,
                    "deployed": result.deployed,
                    "production_version": result.production_version,
                    "duration_s": round(time.perf_counter() - started, 2),
                },
            )
            metrics.retraining_events_total.labels(
                trigger=(decision.trigger or RetrainingTrigger.MANUAL).value,
                status=result.status.value,
                decision=(result.comparison.decision if result.comparison else "n/a"),
            ).inc()
            return result

    # -- stages -------------------------------------------------------------- #
    def _execute(
        self,
        event_id: str,
        model_name: str,
        baseline_version: int | None,
        baseline_metrics: dict[str, float] | None,
        trigger: RetrainingTrigger,
        reason: str,
        deploy: bool | None,
        collect_production_data: bool,
    ) -> RetrainingDecision:
        # ---- 1. collect the latest data --------------------------------- #
        with StageTimer(logger, "retraining.collect_data"):
            dataset_version = self._collect_data(collect_production_data)

        # ---- 2/3. validate + train (the standard pipeline does both) ----- #
        with StageTimer(logger, "retraining.train"):
            run = train_model(
                TrainingRequest(
                    dataset_version=dataset_version,
                    run_name=f"retrain-{event_id[:12]}",
                    tags={
                        "fmops.retraining_event": event_id,
                        "fmops.trigger": trigger.value,
                    },
                )
            )

        candidate_version = run.registered_version
        self.store.update(event_id, candidate_version=candidate_version)
        if run.evaluation is None or candidate_version is None:
            raise FMOpsError("retraining produced no evaluable model version")

        # ---- 4. compare against production ------------------------------- #
        with StageTimer(logger, "retraining.compare"):
            comparison = compare_to_production(
                run.evaluation.metrics,
                baseline_metrics,
                candidate_version=candidate_version,
                baseline_version=baseline_version,
                settings=self.settings,
            )
            approval = evaluate_approval(
                run.evaluation.metrics,
                model_name,
                candidate_version,
                validation_passed=run.validation_passed,
                settings=self.settings,
            )
            get_metrics().approval_decisions_total.labels(
                model_name=model_name, decision=approval.decision.value
            ).inc()

        detail: dict[str, Any] = {
            "trigger_reason": reason,
            "dataset_version": run.dataset_version,
            "candidate_metrics": run.evaluation.metrics.as_dict(),
            "baseline_metrics": baseline_metrics or {},
            "comparison": comparison.model_dump(mode="json"),
            "approval": approval.model_dump(mode="json"),
        }

        # ---- 5. decide ---------------------------------------------------- #
        if not approval.approved:
            return self._reject(
                event_id,
                model_name,
                candidate_version,
                baseline_version,
                trigger,
                reason,
                approval.reason,
                comparison,
                approval,
                detail,
                category="approval_gate",
            )

        if not comparison.candidate_is_better:
            return self._reject(
                event_id,
                model_name,
                candidate_version,
                baseline_version,
                trigger,
                reason,
                comparison.reason,
                comparison,
                approval,
                detail,
                category="worse_than_production",
            )

        # ---- 6. promote and deploy ---------------------------------------- #
        should_deploy = (
            self.settings.retraining.auto_deploy_if_better if deploy is None else deploy
        )
        self.registry.transition_stage(
            model_name,
            candidate_version,
            ModelStage.VALIDATION,
            reason=f"retraining candidate cleared the gates ({reason})",
            actor="retraining",
        )
        self.registry.transition_stage(
            model_name,
            candidate_version,
            ModelStage.STAGING,
            reason="promoted by the retraining pipeline",
            actor="retraining",
        )
        self.registry.update_status(model_name, candidate_version, "approved")

        deployed = False
        production_version = baseline_version
        message = (
            f"candidate version {candidate_version} beat production on "
            f"{comparison.metric} ({comparison.improvement:+.4f}) and was promoted "
            "to Staging"
        )

        if should_deploy:
            with StageTimer(logger, "retraining.deploy"):
                result = self.deployments.deploy(
                    DeploymentRequest(
                        model_name=model_name,
                        model_version=candidate_version,
                        strategy=DeploymentStrategy(self.settings.deployment.strategy),
                        reason=f"automatic retraining: {reason}",
                    ),
                    actor="retraining",
                )
                deployed = result.succeeded
                if deployed:
                    production_version = candidate_version
                    message = (
                        f"candidate version {candidate_version} beat production "
                        f"({comparison.improvement:+.4f} {comparison.metric}) and is "
                        "now live"
                    )
                else:
                    message = (
                        f"candidate version {candidate_version} cleared the gates but "
                        f"its deployment did not succeed: {result.message}"
                    )
                detail["deployment"] = {
                    "succeeded": result.succeeded,
                    "rolled_back": result.rolled_back,
                    "message": result.message,
                }
        else:
            message += "; automatic deployment is disabled in this environment"

        self.store.update(
            event_id,
            status=RetrainingStatus.SUCCEEDED,
            decision="promoted",
            detail=detail,
        )
        get_alert_manager().raise_alert(
            Severity.INFO,
            AlertCategory.RETRAINING,
            f"Retraining promoted {model_name} v{candidate_version}",
            message,
            context={
                "event_id": event_id,
                "model_name": model_name,
                "candidate_version": candidate_version,
                "baseline_version": baseline_version,
                "improvement": comparison.improvement,
                "deployed": deployed,
            },
        )
        return RetrainingDecision(
            event_id=event_id,
            triggered=True,
            trigger=trigger,
            reason=reason,
            status=RetrainingStatus.SUCCEEDED,
            comparison=comparison,
            approval=approval,
            candidate_version=candidate_version,
            deployed=deployed,
            production_version=production_version,
            message=message,
        )

    def _collect_data(self, collect_production_data: bool) -> str | None:
        """Assemble the retraining dataset.

        Combines the existing training set with labelled production traffic when
        any exists. When there is no labelled traffic (the common case early on),
        a fresh sample from the current production distribution is generated so
        the retrained model sees the *shifted* distribution rather than the stale
        one -- which is the entire point of retraining after drift.
        """
        registry = get_dataset_registry()
        base = registry.latest()
        if base is None:
            raise FMOpsError("no dataset is registered; cannot retrain")

        frames: list[pd.DataFrame] = [registry.load(base.version)]

        if collect_production_data:
            labelled = get_inference_log().labelled_frame(
                self.settings.tracking.registered_model_name
            )
            appended = 0
            if not labelled.empty and "actual_label" in labelled:
                columns = [
                    *self.settings.data.feature_columns,
                    self.settings.data.target_column,
                ]
                production = labelled.rename(
                    columns={"actual_label": self.settings.data.target_column}
                )
                available = [c for c in columns if c in production.columns]
                if self.settings.data.target_column in available:
                    frames.append(production[available])
                    appended = len(production)
            logger.info(
                "retraining.production_data_collected",
                extra={
                    "labelled_rows": appended,
                    "detail": (
                        "labelled production traffic appended"
                        if appended
                        else "no labelled production traffic available; "
                        "sampling the current production distribution instead"
                    ),
                },
            )
            if appended < self.settings.retraining.min_new_samples:
                fresh = build_production_dataset(
                    n_rows=max(self.settings.retraining.min_new_samples, 1500),
                    drift="none",
                    seed=int(time.time()) % 10_000,
                )
                frames.append(fresh)

        combined = pd.concat(frames, ignore_index=True)
        combined = (
            combined.drop_duplicates(subset=[self.settings.data.id_column], keep="last")
            if self.settings.data.id_column in combined.columns
            else combined
        )

        record = registry.register_frame(
            combined,
            filename=f"{self.settings.data.dataset_name}_retrain.csv",
            description="retraining set: base training data + recent production data",
        )
        logger.info(
            "retraining.dataset_ready",
            extra={"dataset_version": record.version, "rows": record.n_rows},
        )
        return record.version

    def _reject(
        self,
        event_id: str,
        model_name: str,
        candidate_version: int,
        baseline_version: int | None,
        trigger: RetrainingTrigger,
        trigger_reason: str,
        rejection_reason: str,
        comparison,
        approval,
        detail: dict[str, Any],
        category: str,
    ) -> RetrainingDecision:
        """Keep the production model and archive the candidate with the reason."""
        self.registry.update_status(model_name, candidate_version, "rejected")
        self.registry.set_tags(
            model_name,
            candidate_version,
            {"rejected_by": "retraining", "rejected_reason": category},
        )
        self.store.update(
            event_id,
            status=RetrainingStatus.REJECTED,
            decision=category,
            detail=detail,
        )
        logger.warning(
            "retraining.candidate_rejected",
            extra={
                "event_id": event_id,
                "candidate_version": candidate_version,
                "baseline_version": baseline_version,
                "category": category,
                "reason": rejection_reason,
            },
        )
        get_alert_manager().raise_alert(
            Severity.WARNING,
            AlertCategory.RETRAINING,
            f"Retraining candidate rejected for {model_name}",
            (
                f"Version {candidate_version} was rejected ({category}): "
                f"{rejection_reason}. Production version "
                f"{baseline_version if baseline_version else 'n/a'} is unchanged."
            ),
            context={
                "event_id": event_id,
                "model_name": model_name,
                "candidate_version": candidate_version,
                "baseline_version": baseline_version,
                "category": category,
            },
        )
        return RetrainingDecision(
            event_id=event_id,
            triggered=True,
            trigger=trigger,
            reason=trigger_reason,
            status=RetrainingStatus.REJECTED,
            comparison=comparison,
            approval=approval,
            candidate_version=candidate_version,
            deployed=False,
            production_version=baseline_version,
            message=(
                f"candidate version {candidate_version} rejected ({category}): "
                f"{rejection_reason}"
            ),
        )

    def _fail(self, event_id: str, message: str, exc: Exception) -> None:
        logger.error("retraining.failed", exc_info=exc, extra={"event_id": event_id})
        self.store.update(
            event_id,
            status=RetrainingStatus.FAILED,
            decision="failed",
            detail={"error": str(exc), "message": message},
        )
        get_alert_manager().raise_alert(
            Severity.CRITICAL,
            AlertCategory.RETRAINING,
            "Retraining run failed",
            message,
            context={"event_id": event_id, "error": str(exc)},
        )


def run_retraining(force: bool = False, deploy: bool | None = None) -> RetrainingDecision:
    return RetrainingPipeline().run(force=force, deploy=deploy)
