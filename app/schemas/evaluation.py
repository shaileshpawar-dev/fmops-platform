"""Schemas for approval gates, drift, alerts, monitoring and retraining.

These are the types that make the platform's *decisions* auditable: an approval
gate produces an :class:`ApprovalResult` listing every check it ran, a drift
scan produces a :class:`DriftReport` naming every feature it measured, and a
retraining run produces a :class:`ModelComparison` showing exactly why the
candidate was promoted or rejected.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from app.core.utils import new_id, utcnow_iso
from app.schemas.common import (
    AlertCategory,
    ApprovalDecision,
    DriftType,
    RetrainingStatus,
    RetrainingTrigger,
    Severity,
)


# --------------------------------------------------------------------------- #
# Approval
# --------------------------------------------------------------------------- #
class GateCheck(BaseModel):
    name: str
    passed: bool
    observed: float | str | None = None
    threshold: float | str | None = None
    message: str = ""
    blocking: bool = True


class ApprovalResult(BaseModel):
    """Outcome of running a candidate model through the approval gate."""

    model_name: str
    model_version: int | None = None
    decision: ApprovalDecision
    checks: list[GateCheck] = Field(default_factory=list)
    reason: str = ""
    evaluated_metrics: dict[str, float] = Field(default_factory=dict)
    created_at: str = Field(default_factory=utcnow_iso)

    @property
    def approved(self) -> bool:
        return self.decision == ApprovalDecision.APPROVED

    @property
    def failed_checks(self) -> list[GateCheck]:
        return [c for c in self.checks if not c.passed]

    def render_text(self) -> str:
        lines = [
            f"Approval gate -- {self.model_name}"
            + (f" v{self.model_version}" if self.model_version else ""),
            f"  decision: {self.decision.value.upper()}",
        ]
        for check in self.checks:
            mark = "PASS" if check.passed else "FAIL"
            lines.append(
                f"  [{mark}] {check.name}: observed={check.observed} "
                f"threshold={check.threshold} {check.message}".rstrip()
            )
        if self.reason:
            lines.append(f"  reason: {self.reason}")
        return "\n".join(lines)


class ModelComparison(BaseModel):
    """Champion vs challenger. Drives promote-or-reject after retraining."""

    metric: str
    baseline_version: int | None
    candidate_version: int | None
    baseline_score: float
    candidate_score: float
    improvement: float
    min_improvement: float
    candidate_is_better: bool
    decision: str
    reason: str = ""
    created_at: str = Field(default_factory=utcnow_iso)


# --------------------------------------------------------------------------- #
# Drift
# --------------------------------------------------------------------------- #
class FeatureDrift(BaseModel):
    """Drift measurement for a single feature.

    ``score`` is normalised to [0, 1] regardless of the underlying test so
    features of different types can be compared and averaged.
    """

    feature: str
    feature_type: str
    test: str
    statistic: float
    p_value: float | None = None
    score: float
    threshold: float
    drifted: bool
    reference_summary: dict[str, float] = Field(default_factory=dict)
    current_summary: dict[str, float] = Field(default_factory=dict)


class DriftReport(BaseModel):
    """Result of one drift scan.

    Note on concept drift: it cannot be measured from unlabelled production
    data.  ``concept_drift_status`` is ``measured`` only when ground-truth
    labels were available; otherwise it is ``unavailable`` and
    ``prediction_drift_score`` acts as a *proxy signal*, not a measurement.
    """

    id: str = Field(default_factory=lambda: new_id("drift"))
    model_name: str
    model_version: int | None = None
    engine: str = "native"
    drift_detected: bool = False
    dataset_drift_score: float = 0.0
    dataset_drift_share: float = 0.0
    threshold: float = 0.2
    prediction_drift_score: float | None = None
    prediction_drift_detected: bool = False
    concept_drift_score: float | None = None
    concept_drift_status: str = "unavailable"
    concept_drift_detail: str = (
        "Concept drift requires ground-truth labels; none were available for "
        "this window. Prediction drift is reported as a proxy signal only."
    )
    feature_drift: list[FeatureDrift] = Field(default_factory=list)
    drifted_features: list[str] = Field(default_factory=list)
    n_reference: int = 0
    n_current: int = 0
    report_uri: str | None = None
    created_at: str = Field(default_factory=utcnow_iso)

    def by_type(self, drift_type: DriftType) -> float | None:
        if drift_type in (DriftType.DATA, DriftType.FEATURE):
            return self.dataset_drift_score
        if drift_type == DriftType.PREDICTION:
            return self.prediction_drift_score
        return self.concept_drift_score

    def render_text(self) -> str:
        lines = [
            f"Drift report -- {self.model_name}"
            + (f" v{self.model_version}" if self.model_version else ""),
            f"  engine            : {self.engine}",
            f"  windows           : reference={self.n_reference} current={self.n_current}",
            f"  dataset drift     : {self.dataset_drift_score:.4f} "
            f"(threshold {self.threshold:.2f})",
            f"  drifted features  : {len(self.drifted_features)}/"
            f"{len(self.feature_drift)} -> {', '.join(self.drifted_features) or 'none'}",
            "  prediction drift  : "
            + (
                f"{self.prediction_drift_score:.4f}"
                if self.prediction_drift_score is not None
                else "n/a"
            ),
            f"  concept drift     : {self.concept_drift_status}"
            + (
                f" ({self.concept_drift_score:.4f})"
                if self.concept_drift_score is not None
                else ""
            ),
            f"  verdict           : {'DRIFT DETECTED' if self.drift_detected else 'stable'}",
        ]
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Alerts
# --------------------------------------------------------------------------- #
class Alert(BaseModel):
    id: str = Field(default_factory=lambda: new_id("alert"))
    severity: Severity
    category: AlertCategory
    title: str
    message: str
    fingerprint: str = ""
    context: dict[str, Any] = Field(default_factory=dict)
    acknowledged: bool = False
    created_at: str = Field(default_factory=utcnow_iso)


# --------------------------------------------------------------------------- #
# Monitoring
# --------------------------------------------------------------------------- #
class LatencyStats(BaseModel):
    count: int = 0
    p50_ms: float = 0.0
    p95_ms: float = 0.0
    p99_ms: float = 0.0
    max_ms: float = 0.0
    mean_ms: float = 0.0


class ServiceMetrics(BaseModel):
    window_minutes: int = 60
    request_count: int = 0
    error_count: int = 0
    error_rate: float = 0.0
    throughput_rpm: float = 0.0
    latency: LatencyStats = Field(default_factory=LatencyStats)
    slo_latency_ms: float = 250.0
    slo_error_rate: float = 0.02
    latency_slo_met: bool = True
    error_slo_met: bool = True


class ResourceUsage(BaseModel):
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    memory_used_mb: float = 0.0
    memory_total_mb: float = 0.0
    disk_percent: float = 0.0
    process_rss_mb: float = 0.0
    open_files: int = 0
    threads: int = 0
    gpu_available: bool = False
    gpu: list[dict[str, Any]] = Field(default_factory=list)
    sampled_at: str = Field(default_factory=utcnow_iso)


class LivePerformance(BaseModel):
    """Model quality computed from labelled production traffic only."""

    labelled_samples: int = 0
    available: bool = False
    accuracy: float | None = None
    precision: float | None = None
    recall: float | None = None
    f1: float | None = None
    roc_auc: float | None = None
    detail: str = "No ground-truth labels received for this window."


class MonitoringSummary(BaseModel):
    model_name: str
    model_version: int | None = None
    model_stage: str | None = None
    service: ServiceMetrics = Field(default_factory=ServiceMetrics)
    resources: ResourceUsage = Field(default_factory=ResourceUsage)
    live_performance: LivePerformance = Field(default_factory=LivePerformance)
    prediction_positive_rate: float | None = None
    latest_drift: DriftReport | None = None
    open_alerts: int = 0
    created_at: str = Field(default_factory=utcnow_iso)


# --------------------------------------------------------------------------- #
# Retraining
# --------------------------------------------------------------------------- #
class RetrainingEvent(BaseModel):
    id: str = Field(default_factory=lambda: new_id("retrain"))
    trigger: RetrainingTrigger
    reason: str
    status: RetrainingStatus = RetrainingStatus.CREATED
    model_name: str
    baseline_version: int | None = None
    candidate_version: int | None = None
    decision: str | None = None
    detail: dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default_factory=utcnow_iso)
    updated_at: str = Field(default_factory=utcnow_iso)


class RetrainingDecision(BaseModel):
    """Final word on a retraining run: what happened and what is live now."""

    event_id: str
    triggered: bool
    trigger: RetrainingTrigger | None = None
    reason: str = ""
    status: RetrainingStatus = RetrainingStatus.SKIPPED
    comparison: ModelComparison | None = None
    approval: ApprovalResult | None = None
    candidate_version: int | None = None
    deployed: bool = False
    production_version: int | None = None
    message: str = ""
    created_at: str = Field(default_factory=utcnow_iso)
