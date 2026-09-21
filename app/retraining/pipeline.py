"""The automatic retraining pipeline.

    drift / performance / volume trigger
              |
              v
      create retraining event
              |
              v
      collect the data  (serving version's training set + labelled production
                         traffic + an optional new dataset version -- never
                         anything generated)
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
the configured metric by at least ``min_improvement``. Where the environment
requires a human sign-off, a candidate that clears both is held for approval
rather than promoted. Every decision, with its numbers, is recorded on the
retraining event and as a gate decision on the candidate version.
"""

from __future__ import annotations

import time
from typing import Any

import pandas as pd

from app.core.config import Settings, get_settings
from app.core.exceptions import DataValidationError, FMOpsError
from app.core.logging import StageTimer, get_logger, log_context
from app.data.versioning import get_dataset_registry
from app.deployment.manager import DeploymentManager, get_deployment_manager
from app.monitoring.alerts import get_alert_manager
from app.monitoring.inference_log import get_inference_log
from app.monitoring.metrics import get_metrics
from app.registry.base import ModelRegistry
from app.registry.context import data_config_for, default_model_name, signature_of
from app.registry.factory import get_registry
from app.retraining.trigger import (
    RetrainingEventStore,
    TriggerDecision,
    evaluate_trigger,
    get_event_store,
)
from app.schemas.common import (
    AlertCategory,
    ApprovalDecision,
    DeploymentStrategy,
    ModelStage,
    RetrainingStatus,
    RetrainingTrigger,
    Severity,
)
from app.schemas.deployment import DeploymentRequest
from app.schemas.evaluation import RetrainingDecision
from app.schemas.model import TrainingRequest
from app.training.approval import evaluate_approval
from app.training.decisions import get_gate_decisions
from app.training.holdout import compare_on_shared_holdout
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
        model_name: str | None = None,
        dataset_version: str | None = None,
        checkpoint: Any = None,
        on_event: Any = None,
    ) -> RetrainingDecision:
        """One retraining cycle for one model.

        ``dataset_version`` is new training data the operator uploaded; it is
        combined with the serving version's training set and any labelled
        production traffic. Supplying it counts as a manual trigger.

        ``checkpoint`` (from the job runner) is called between stages and raises
        when the job was cancelled; ``on_event`` receives the event id as soon as
        the event exists, so the job links to it before the long part starts.
        """
        model_name = model_name or default_model_name(self.settings)
        if dataset_version is not None:
            force = True
        decision = decision or evaluate_trigger(force=force, model_name=model_name)

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
        base = self.registry.get_serving(model_name) or self.registry.get_latest(model_name)
        if base is None:
            raise FMOpsError(
                f"model {model_name} has no registered versions; train it before retraining",
                model=model_name,
            )
        event = self.store.create(
            trigger=decision.trigger or RetrainingTrigger.MANUAL,
            reason=decision.reason,
            model_name=model_name,
            baseline_version=baseline.version if baseline else None,
            detail=decision.evidence,
        )
        if on_event is not None:
            on_event(event.id)
        metrics = get_metrics()

        with log_context(retraining_event=event.id, model=model_name):
            self.store.update(event.id, status=RetrainingStatus.RUNNING)
            started = time.perf_counter()
            try:
                result = self._execute(
                    event_id=event.id,
                    model_name=model_name,
                    base_version=base,
                    new_dataset_version=dataset_version,
                    baseline_version=baseline.version if baseline else None,
                    baseline_metrics=baseline.metrics if baseline else None,
                    trigger=decision.trigger or RetrainingTrigger.MANUAL,
                    reason=decision.reason,
                    deploy=deploy,
                    collect_production_data=collect_production_data,
                    checkpoint=checkpoint,
                )
            except DataValidationError as exc:
                # The retraining data is unusable. Production keeps serving.
                self._fail(
                    event.id,
                    "retraining aborted: the collected data failed validation; "
                    "the production model is unchanged",
                    exc,
                    model_name,
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
                self._fail(event.id, f"retraining failed: {exc.message}", exc, model_name)
                metrics.retraining_events_total.labels(
                    trigger=(decision.trigger or RetrainingTrigger.MANUAL).value,
                    status=RetrainingStatus.FAILED.value,
                    decision="error",
                ).inc()
                raise
            except Exception as exc:
                if exc.__class__.__name__ == "JobCancelled":
                    self.store.update(
                        event.id,
                        status=RetrainingStatus.FAILED,
                        decision="cancelled",
                        detail={"message": "cancelled by request; production is unchanged"},
                    )
                    raise
                self._fail(event.id, f"retraining failed: {exc}", exc, model_name)
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
        base_version: Any,
        new_dataset_version: str | None,
        baseline_version: int | None,
        baseline_metrics: dict[str, float] | None,
        trigger: RetrainingTrigger,
        reason: str,
        deploy: bool | None,
        collect_production_data: bool,
        checkpoint: Any = None,
    ) -> RetrainingDecision:
        def check() -> None:
            if checkpoint is not None:
                checkpoint()

        # ---- 1. collect the data ----------------------------------------- #
        data_config = data_config_for(base_version, self.settings)
        with StageTimer(logger, "retraining.collect_data"):
            dataset_version, sources = self._collect_data(
                model_name,
                base_version,
                data_config,
                collect_production_data,
                new_dataset_version,
            )
        self.store.update(event_id, detail={"data_sources": sources, "trigger_reason": reason})
        check()

        # ---- 2/3. validate + train (the standard pipeline does both) ----- #
        # Under the model's own data contract: the reference model's settings
        # would validate a churn dataset against the loan schema.
        with StageTimer(logger, "retraining.train"):
            run = train_model(
                TrainingRequest(
                    model_name=model_name,
                    dataset_version=dataset_version,
                    run_name=f"retrain-{event_id[:12]}",
                    tags={
                        "fmops.retraining_event": event_id,
                        "fmops.trigger": trigger.value,
                    },
                ),
                settings=self.settings.model_copy(update={"data": data_config}, deep=True),
            )

        candidate_version = run.registered_version
        self.store.update(event_id, candidate_version=candidate_version)
        if run.evaluation is None or candidate_version is None:
            raise FMOpsError("retraining produced no evaluable model version")

        # ---- 4. compare against production ------------------------------- #
        with StageTimer(logger, "retraining.compare"):
            # Both scored on the rows of the candidate's holdout that the
            # incumbent never trained on -- not each on its own test split.
            comparison = compare_on_shared_holdout(
                self.registry.get(model_name, candidate_version),
                self.registry.get(model_name, baseline_version) if baseline_version else None,
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
            "data_sources": sources,
            "dataset_version": run.dataset_version,
            "candidate_metrics": run.evaluation.metrics.as_dict(),
            "baseline_metrics": baseline_metrics or {},
            "comparison": comparison.model_dump(mode="json"),
            "approval": approval.model_dump(mode="json"),
        }

        # ---- 5. decide ---------------------------------------------------- #
        if approval.decision == ApprovalDecision.REJECTED:
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

        if approval.decision == ApprovalDecision.PENDING_MANUAL:
            return self._hold_for_approval(
                event_id,
                model_name,
                candidate_version,
                baseline_version,
                trigger,
                reason,
                comparison,
                approval,
                detail,
            )

        # ---- 6. promote and deploy ---------------------------------------- #
        self._record_decision(
            model_name,
            candidate_version,
            "approved",
            f"retraining candidate cleared the gate and beat production ({reason})",
            approval,
            comparison,
            ModelStage.STAGING,
        )
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
            check()
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

    def _collect_data(
        self,
        model_name: str,
        base_version: Any,
        data_config: Any,
        collect_production_data: bool,
        new_dataset_version: str | None,
    ) -> tuple[str, dict[str, Any]]:
        """Assemble the retraining dataset from real rows only.

        Three sources, all observed: the training set the base version was
        fitted on, an optional new dataset version supplied by the operator, and
        this model's labelled production traffic. Nothing is generated. If the
        combination adds no rows beyond the base training set, the run says so
        on the event rather than pretending the data is new.
        """
        registry = get_dataset_registry()
        if not base_version.dataset_version:
            raise FMOpsError(
                f"{model_name} v{base_version.version} has no recorded training dataset; "
                "retraining needs one to start from",
                model=model_name,
            )
        target = data_config.target_column
        base_record = registry.get(base_version.dataset_version)
        base_frame = registry.load(base_version.dataset_version)
        # Identity columns travel with every row when the base dataset has them;
        # a declared schema (the reference dataset's) may require them.
        identity = [
            c
            for c in (data_config.id_column, data_config.timestamp_column)
            if c and c in base_frame.columns
        ]
        columns = [*data_config.feature_columns, target, *identity]
        frames: list[pd.DataFrame] = [base_frame]
        observed: pd.DataFrame | None = None
        sources: dict[str, Any] = {
            "base_dataset": base_version.dataset_version,
            "base_rows": len(base_frame),
            "new_dataset": new_dataset_version,
            "new_dataset_rows": 0,
            "labelled_production_rows": 0,
        }

        if new_dataset_version:
            fresh = registry.load(new_dataset_version)
            missing = [c for c in columns if c not in fresh.columns]
            if missing:
                raise FMOpsError(
                    f"dataset {new_dataset_version} lacks columns {model_name} was trained on: "
                    f"{', '.join(missing[:8])}",
                    model=model_name,
                    dataset_version=new_dataset_version,
                )
            frames.append(fresh)
            sources["new_dataset_rows"] = len(fresh)

        if collect_production_data:
            labelled = get_inference_log().labelled_frame(model_name)
            if not labelled.empty and "actual_label" in labelled:
                # Feedback is stored as 0/1; the training frame speaks the
                # target's own labels, so map back before combining.
                signature = signature_of(base_version)
                raw_labels = signature.class_labels if signature else ["0", "1"]
                production = labelled.copy()
                production[target] = production["actual_label"].map(
                    lambda v: raw_labels[int(v)] if pd.notna(v) else None
                )
                if not data_config.class_labels:
                    production[target] = pd.to_numeric(production[target], errors="coerce")
                # A production row has an identity and a time of its own: the
                # request id and when it was served. Carrying them keeps every
                # row traceable back to the inference log.
                if data_config.id_column in identity:
                    production[data_config.id_column] = production["request_id"]
                if data_config.timestamp_column in identity:
                    production[data_config.timestamp_column] = production["created_at"]
                available = [c for c in columns if c in production.columns]
                if target in available:
                    observed = production[available]
                    sources["labelled_production_rows"] = len(production)

        # Tables can overlap -- a new dataset version is often a full re-export
        # that repeats the base rows -- so those are de-duplicated: by id where
        # a row has one, otherwise by content. Production rows are separate
        # events and are appended as they are; they carry no id, and treating
        # every missing id as "the same id" would keep one row out of hundreds.
        tables = pd.concat(frames, ignore_index=True)
        before = len(tables)
        id_column = data_config.id_column
        if id_column and id_column in tables.columns:
            has_id = tables[id_column].notna()
            tables = pd.concat(
                [
                    tables[has_id].drop_duplicates(subset=[id_column], keep="last"),
                    tables[~has_id],
                ],
                ignore_index=True,
            )
        tables = tables.drop_duplicates(keep="last")
        sources["duplicates_dropped"] = before - len(tables)
        # Keep what training consumes plus the identity columns, which every
        # row -- base, new dataset or production -- now has.
        keep = [c for c in columns if c in tables.columns]
        combined = (
            tables[keep]
            if observed is None
            else pd.concat(
                [tables[keep], observed[[c for c in keep if c in observed]]], ignore_index=True
            )
        )
        sources["rows"] = len(combined)
        sources["new_rows"] = len(combined) - len(base_frame.drop_duplicates())

        record = registry.register_frame(
            combined,
            dataset_name=base_record.dataset_name,
            parent_version=base_version.dataset_version,
            filename=f"{model_name}_retrain.csv",
            description=(
                f"retraining set for {model_name}: {base_version.dataset_version} "
                f"+ {sources['new_dataset_rows']} rows from {new_dataset_version or 'no new dataset'} "
                f"+ {sources['labelled_production_rows']} labelled production rows"
            ),
        )
        sources["dataset_version"] = record.version
        logger.info(
            "retraining.dataset_ready",
            extra={"model": model_name, **sources},
        )
        return record.version, sources

    def _record_decision(
        self,
        model_name: str,
        version: int,
        decision: str,
        reason: str,
        approval: Any,
        comparison: Any,
        target_stage: ModelStage | None,
    ) -> None:
        current = self.registry.get(model_name, version)
        get_gate_decisions().record(
            model_name=model_name,
            model_version=version,
            source="pipeline",
            decision=decision,
            reason=reason,
            approval=approval,
            comparison=comparison,
            thresholds=self.settings.approval.model_dump(mode="json"),
            target_stage=target_stage.value if target_stage else None,
            final_stage=current.stage.value,
            actor="retraining",
        )

    def _hold_for_approval(
        self,
        event_id: str,
        model_name: str,
        candidate_version: int,
        baseline_version: int | None,
        trigger: RetrainingTrigger,
        reason: str,
        comparison: Any,
        approval: Any,
        detail: dict[str, Any],
    ) -> RetrainingDecision:
        """Every automated check passed; this environment needs a human to say yes."""
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
        self.registry.update_status(model_name, candidate_version, "pending")
        self._record_decision(
            model_name,
            candidate_version,
            ApprovalDecision.PENDING_MANUAL.value,
            "cleared every automated check and beat production; awaiting human approval",
            approval,
            comparison,
            ModelStage.PRODUCTION,
        )
        self.store.update(
            event_id,
            status=RetrainingStatus.SUCCEEDED,
            decision="awaiting_approval",
            detail=detail,
        )
        message = (
            f"candidate version {candidate_version} beat production on {comparison.metric} "
            f"({comparison.improvement:+.4f}) and passed every automated check; it is held "
            f"for approval: POST /api/v1/models/{model_name}/versions/{candidate_version}/approve"
        )
        get_alert_manager().raise_alert(
            Severity.INFO,
            AlertCategory.RETRAINING,
            f"Retraining candidate for {model_name} awaits approval",
            message,
            context={
                "event_id": event_id,
                "model_name": model_name,
                "candidate_version": candidate_version,
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
            deployed=False,
            production_version=baseline_version,
            message=message,
        )

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
        self._record_decision(
            model_name,
            candidate_version,
            ApprovalDecision.REJECTED.value,
            f"retraining candidate rejected ({category}): {rejection_reason}",
            approval,
            comparison,
            None,
        )
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

    def _fail(
        self, event_id: str, message: str, exc: Exception, model_name: str | None = None
    ) -> None:
        logger.error(
            "retraining.failed",
            exc_info=exc,
            extra={"event_id": event_id, "model": model_name},
        )
        self.store.update(
            event_id,
            status=RetrainingStatus.FAILED,
            decision="failed",
            detail={"error": str(exc), "message": message},
        )
        get_alert_manager().raise_alert(
            Severity.CRITICAL,
            AlertCategory.RETRAINING,
            f"Retraining failed for {model_name}" if model_name else "Retraining run failed",
            message,
            context={"event_id": event_id, "model_name": model_name, "error": str(exc)},
        )


def run_retraining(
    force: bool = False,
    deploy: bool | None = None,
    model_name: str | None = None,
    dataset_version: str | None = None,
) -> RetrainingDecision:
    return RetrainingPipeline().run(
        force=force, deploy=deploy, model_name=model_name, dataset_version=dataset_version
    )
