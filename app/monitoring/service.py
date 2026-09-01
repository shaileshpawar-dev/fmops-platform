"""Monitoring service.

Assembles the pieces the API and the retraining trigger both need:

* the **reference window** -- the training distribution a model was fitted on,
  reconstructed from the dataset version recorded on the model version, so a
  drift comparison is always against the data that model actually saw,
* the **current window** -- recent production traffic from the inference log,
* live performance from labelled traffic (or an explicit "no labels" answer),
* the aggregate monitoring summary shown on the dashboard.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from app.core.config import Settings, get_settings
from app.core.exceptions import InsufficientDataError
from app.core.logging import get_logger
from app.data.versioning import get_dataset_registry
from app.deployment.base import get_deployment_store
from app.monitoring.alerts import get_alert_manager
from app.monitoring.drift import DriftDetector, latest_drift_report
from app.monitoring.inference_log import get_inference_log
from app.monitoring.metrics import set_live_performance
from app.monitoring.resource_monitor import latest_resources
from app.registry.base import ModelRegistry
from app.registry.factory import get_registry
from app.schemas.evaluation import (
    DriftReport,
    LatencyStats,
    LivePerformance,
    MonitoringSummary,
    ServiceMetrics,
)

logger = get_logger(__name__)


class MonitoringService:
    """Read-side of the monitoring layer."""

    def __init__(
        self, settings: Settings | None = None, registry: ModelRegistry | None = None
    ) -> None:
        self.settings = settings or get_settings()
        self._registry = registry

    @property
    def registry(self) -> ModelRegistry:
        if self._registry is None:
            self._registry = get_registry()
        return self._registry

    # -- windows ------------------------------------------------------------- #
    def reference_window(self, model_version: int | None = None) -> pd.DataFrame:
        """The training data the serving model was fitted on.

        Falls back to the latest registered dataset when the model's own dataset
        version is no longer resolvable, and says so in the log -- a drift
        comparison against the wrong reference is worse than none.
        """
        model_name = self.settings.tracking.registered_model_name
        datasets = get_dataset_registry()

        version = None
        if model_version is not None:
            version = self.registry.get(model_name, model_version).dataset_version
        else:
            serving = self.registry.get_serving(model_name)
            version = serving.dataset_version if serving else None

        if version:
            try:
                return datasets.load(version)
            except Exception as exc:
                logger.warning(
                    "monitoring.reference_dataset_unavailable",
                    extra={
                        "dataset_version": version,
                        "error": str(exc),
                        "fallback": "latest registered dataset",
                    },
                )

        latest = datasets.latest()
        if latest is None:
            return pd.DataFrame()
        return datasets.load(latest.version)

    def current_window(
        self, model_version: int | None = None, limit: int | None = None
    ) -> pd.DataFrame:
        log = get_inference_log()
        return log.feature_frame(
            model_name=self.settings.tracking.registered_model_name,
            model_version=model_version,
            limit=limit or self.settings.drift.detection_window,
        )

    # -- drift ---------------------------------------------------------------- #
    def run_drift_scan(
        self, model_version: int | None = None, persist: bool = True
    ) -> DriftReport:
        model_name = self.settings.tracking.registered_model_name
        serving = self.registry.get_serving(model_name)
        version = model_version or (serving.version if serving else None)

        current = self.current_window(version)
        if current.empty:
            raise InsufficientDataError(
                "no production predictions have been logged yet; send traffic to "
                "/api/v1/predict (or run 'make simulate-traffic') before scanning "
                "for drift",
                model=model_name,
            )

        reference = self.reference_window(version)
        reference = reference.head(self.settings.drift.reference_window)

        # Reference probabilities let prediction drift be measured. They come
        # from scoring the reference sample with the serving model.
        reference = self._add_reference_predictions(reference, version)

        labelled = get_inference_log().labelled_frame(model_name, version)
        baseline = None
        if serving is not None:
            baseline = serving.metrics.get("roc_auc")

        detector = DriftDetector(self.settings)
        return detector.detect(
            reference=reference,
            current=current,
            model_name=model_name,
            model_version=version,
            labelled=labelled if not labelled.empty else None,
            baseline_metric=baseline,
            persist=persist,
        )

    def _add_reference_predictions(
        self, reference: pd.DataFrame, version: int | None
    ) -> pd.DataFrame:
        if reference.empty or version is None or "_probability" in reference.columns:
            return reference
        try:
            from app.api.serving import get_prediction_service
            from app.data.preprocessing import prepare_inference_frame

            model, _ = get_prediction_service().resolve_model(version)
            sample = reference.head(2000)
            frame = prepare_inference_frame(sample, self.settings.data)
            probabilities = model.pipeline.predict_proba(frame)[:, 1]
            enriched = sample.copy()
            enriched["_probability"] = probabilities
            enriched["_prediction"] = (probabilities >= model.threshold).astype(int)
            return enriched
        except Exception as exc:
            logger.debug("monitoring.reference_scoring_skipped", extra={"error": str(exc)})
            return reference

    # -- performance ---------------------------------------------------------- #
    def live_performance(self, model_version: int | None = None) -> LivePerformance:
        """Model quality from labelled production traffic only.

        Returns ``available=False`` with an explanation when no labels exist.
        The platform never estimates production accuracy from unlabelled data.
        """
        model_name = self.settings.tracking.registered_model_name
        labelled = get_inference_log().labelled_frame(model_name, model_version)
        if labelled.empty:
            return LivePerformance(
                labelled_samples=0,
                available=False,
                detail=(
                    "No ground-truth labels have been submitted. Production "
                    "accuracy cannot be computed without them; submit outcomes "
                    "via POST /api/v1/feedback."
                ),
            )

        y_true = pd.to_numeric(labelled.get("actual_label"), errors="coerce")
        y_pred = pd.to_numeric(labelled.get("prediction"), errors="coerce")
        probabilities = pd.to_numeric(labelled.get("probability"), errors="coerce")
        frame = pd.DataFrame({"y": y_true, "p": y_pred, "prob": probabilities}).dropna()

        if len(frame) < 20:
            return LivePerformance(
                labelled_samples=len(frame),
                available=False,
                detail=(
                    f"Only {len(frame)} labelled predictions available; at least 20 "
                    "are required before live metrics are reported."
                ),
            )

        from sklearn.metrics import (
            accuracy_score,
            f1_score,
            precision_score,
            recall_score,
            roc_auc_score,
        )

        both_classes = frame["y"].nunique() > 1
        result = LivePerformance(
            labelled_samples=len(frame),
            available=True,
            accuracy=float(accuracy_score(frame["y"], frame["p"])),
            precision=float(precision_score(frame["y"], frame["p"], zero_division=0)),
            recall=float(recall_score(frame["y"], frame["p"], zero_division=0)),
            f1=float(f1_score(frame["y"], frame["p"], zero_division=0)),
            roc_auc=(
                float(roc_auc_score(frame["y"], frame["prob"])) if both_classes else None
            ),
            detail=(
                f"Computed from {len(frame)} labelled production predictions."
                + ("" if both_classes else " ROC-AUC undefined: only one class present.")
            ),
        )
        if model_version is not None:
            set_live_performance(
                model_name,
                model_version,
                {
                    k: v
                    for k, v in result.model_dump().items()
                    if isinstance(v, (int, float)) and k != "labelled_samples"
                },
            )
        return result

    def service_metrics(self, window_minutes: int = 60) -> ServiceMetrics:
        stats = get_inference_log().window_stats(
            model_name=self.settings.tracking.registered_model_name,
            minutes=window_minutes,
        )
        monitoring = self.settings.monitoring
        latency = LatencyStats(
            count=stats["request_count"],
            p50_ms=round(stats["latency_p50_ms"], 3),
            p95_ms=round(stats["latency_p95_ms"], 3),
            p99_ms=round(stats["latency_p99_ms"], 3),
            max_ms=round(stats["latency_max_ms"], 3),
            mean_ms=round(stats["latency_mean_ms"], 3),
        )
        return ServiceMetrics(
            window_minutes=window_minutes,
            request_count=stats["request_count"],
            error_count=stats["error_count"],
            error_rate=round(stats["error_rate"], 5),
            throughput_rpm=round(stats["throughput_rpm"], 3),
            latency=latency,
            slo_latency_ms=monitoring.latency_slo_ms,
            slo_error_rate=monitoring.error_rate_slo,
            latency_slo_met=latency.p95_ms <= monitoring.latency_slo_ms
            or stats["request_count"] == 0,
            error_slo_met=stats["error_rate"] <= monitoring.error_rate_slo,
        )

    # -- summary --------------------------------------------------------------- #
    def summary(self, window_minutes: int = 60) -> MonitoringSummary:
        model_name = self.settings.tracking.registered_model_name
        serving = self.registry.get_serving(model_name)
        version = serving.version if serving else None

        stats = get_inference_log().window_stats(model_name=model_name, minutes=window_minutes)
        return MonitoringSummary(
            model_name=model_name,
            model_version=version,
            model_stage=serving.stage.value if serving else None,
            service=self.service_metrics(window_minutes),
            resources=latest_resources(),
            live_performance=self.live_performance(version),
            prediction_positive_rate=stats.get("positive_rate"),
            latest_drift=latest_drift_report(model_name),
            open_alerts=get_alert_manager().open_count(),
        )

    def health_watchdog(self, window_minutes: int = 15) -> dict[str, Any]:
        """Check live SLOs and raise alerts (and optionally roll back).

        This is what makes "production health deteriorates -> rollback" real
        rather than manual. It is invoked by the monitoring endpoint and by the
        scheduled retraining/health workflow.
        """
        from app.schemas.common import AlertCategory, Severity

        metrics = self.service_metrics(window_minutes)
        alerts = get_alert_manager()
        breaches: list[str] = []

        if metrics.request_count == 0:
            return {
                "checked": True,
                "breaches": [],
                "detail": "no traffic in the window; nothing to evaluate",
                "metrics": metrics.model_dump(mode="json"),
            }

        if not metrics.error_slo_met:
            breaches.append("error_rate")
            alerts.raise_alert(
                Severity.CRITICAL,
                AlertCategory.ERROR_RATE,
                f"Error-rate SLO breached on {self.settings.deployment.endpoint_name}",
                (
                    f"Error rate {metrics.error_rate:.2%} over {metrics.request_count} "
                    f"requests exceeds the SLO of {metrics.slo_error_rate:.2%}."
                ),
                context={
                    "endpoint": self.settings.deployment.endpoint_name,
                    "error_rate": metrics.error_rate,
                    "slo": metrics.slo_error_rate,
                    "requests": metrics.request_count,
                },
                dedupe_keys=("endpoint",),
            )

        if not metrics.latency_slo_met:
            breaches.append("latency")
            alerts.raise_alert(
                Severity.WARNING,
                AlertCategory.LATENCY,
                f"Latency SLO breached on {self.settings.deployment.endpoint_name}",
                (
                    f"p95 latency {metrics.latency.p95_ms:.1f}ms exceeds the SLO of "
                    f"{metrics.slo_latency_ms:.1f}ms over {metrics.request_count} requests."
                ),
                context={
                    "endpoint": self.settings.deployment.endpoint_name,
                    "latency_p95_ms": metrics.latency.p95_ms,
                    "slo": metrics.slo_latency_ms,
                },
                dedupe_keys=("endpoint",),
            )

        return {
            "checked": True,
            "breaches": breaches,
            "healthy": not breaches,
            "metrics": metrics.model_dump(mode="json"),
        }

    def deployment_snapshot(self) -> dict[str, Any]:
        deployment = get_deployment_store().active(self.settings.deployment.endpoint_name)
        if deployment is None:
            return {"active": False}
        return {
            "active": True,
            "id": deployment.id,
            "endpoint": deployment.endpoint_name,
            "state": deployment.state.value,
            "strategy": deployment.strategy.value,
            "health": deployment.health.value,
            "current_version": deployment.current_version,
            "previous_version": deployment.previous_version,
            "candidate_version": deployment.candidate_version,
            "shadow_version": deployment.shadow_version,
            "traffic": deployment.traffic,
        }


_SERVICE: MonitoringService | None = None


def get_monitoring_service() -> MonitoringService:
    global _SERVICE
    if _SERVICE is None:
        _SERVICE = MonitoringService()
    return _SERVICE
