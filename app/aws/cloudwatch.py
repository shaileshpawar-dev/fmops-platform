"""CloudWatch metrics adapter.

Mirrors a **deliberately small subset** of the platform's Prometheus metrics to
CloudWatch. Not everything: CloudWatch custom metrics are billed per metric per
month, and mirroring a per-feature drift gauge across 14 features and 3
environments costs real money for no operational benefit. Prometheus keeps the
high-cardinality detail; CloudWatch gets the handful of series that alarms and
the AWS console dashboard are built on.

Metrics are buffered and flushed in batches of 20 (the PutMetricData limit),
because one API call per metric would be both slow and expensive.

A CloudWatch outage must never break a prediction, so every failure here is
logged and swallowed.
"""

from __future__ import annotations

import threading
from typing import Any

from app.aws.client import get_client, require_boto3
from app.core.config import Settings, get_settings
from app.core.logging import get_logger
from app.core.utils import utcnow

logger = get_logger(__name__)

# PutMetricData accepts at most 20 metric data items per call.
MAX_BATCH = 20


class CloudWatchMetrics:
    """Publishes selected platform metrics to CloudWatch."""

    def __init__(self, settings: Settings | None = None, client: Any = None) -> None:
        self.settings = settings or get_settings()
        self.namespace = self.settings.monitoring.cloudwatch_namespace
        self._client = client
        self._buffer: list[dict[str, Any]] = []
        self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        return self.settings.monitoring.cloudwatch_enabled and self.settings.aws.enabled

    @property
    def client(self):
        if self._client is None:
            require_boto3()
            self._client = get_client("cloudwatch", self.settings.aws.region)
        return self._client

    # -- buffering ----------------------------------------------------------- #
    def put(
        self,
        name: str,
        value: float,
        unit: str = "None",
        dimensions: dict[str, str] | None = None,
        flush: bool = False,
    ) -> None:
        """Buffer one datum. Flushes automatically at MAX_BATCH."""
        if not self.enabled:
            return

        datum: dict[str, Any] = {
            "MetricName": name,
            "Value": float(value),
            "Unit": unit,
            "Timestamp": utcnow(),
        }
        if dimensions:
            datum["Dimensions"] = [
                {"Name": str(k), "Value": str(v)[:255]}
                for k, v in dimensions.items()
                if v is not None
            ]

        with self._lock:
            self._buffer.append(datum)
            should_flush = flush or len(self._buffer) >= MAX_BATCH

        if should_flush:
            self.flush()

    def flush(self) -> int:
        """Send the buffer. Returns how many data points were published."""
        if not self.enabled:
            return 0

        with self._lock:
            pending, self._buffer = self._buffer, []
        if not pending:
            return 0

        published = 0
        for start in range(0, len(pending), MAX_BATCH):
            batch = pending[start : start + MAX_BATCH]
            try:
                self.client.put_metric_data(Namespace=self.namespace, MetricData=batch)
                published += len(batch)
            except Exception as exc:
                # Losing metrics is bad; failing the caller is worse.
                logger.error(
                    "cloudwatch.put_metric_data_failed",
                    extra={
                        "namespace": self.namespace,
                        "batch_size": len(batch),
                        "error": str(exc),
                    },
                )
        return published

    # -- the mirrored subset -------------------------------------------------- #
    def record_prediction(
        self, model_name: str, model_version: int, latency_ms: float, success: bool
    ) -> None:
        dimensions = {"ModelName": model_name, "ModelVersion": str(model_version)}
        self.put("Predictions", 1, "Count", dimensions)
        self.put("PredictionLatency", latency_ms, "Milliseconds", dimensions)
        if not success:
            self.put("PredictionErrors", 1, "Count", dimensions)

    def record_drift(
        self,
        model_name: str,
        dataset_drift_score: float,
        detected: bool,
        drifted_feature_count: int,
    ) -> None:
        dimensions = {"ModelName": model_name}
        self.put("DriftScore", dataset_drift_score, "None", dimensions)
        self.put("DriftDetected", 1 if detected else 0, "None", dimensions)
        self.put("DriftedFeatures", drifted_feature_count, "Count", dimensions, flush=True)

    def record_model_metrics(
        self, model_name: str, model_version: int, metrics: dict[str, float]
    ) -> None:
        """Only the headline quality metrics, not the full metric set."""
        dimensions = {"ModelName": model_name, "ModelVersion": str(model_version)}
        for metric in ("roc_auc", "f1", "precision", "recall"):
            if metric in metrics:
                self.put(
                    f"Model{metric.replace('_', '').title()}",
                    float(metrics[metric]),
                    "None",
                    dimensions,
                )
        self.flush()

    def record_deployment(
        self, endpoint: str, model_version: int, succeeded: bool, rolled_back: bool
    ) -> None:
        dimensions = {"Endpoint": endpoint}
        self.put("Deployments", 1, "Count", dimensions)
        if not succeeded:
            self.put("DeploymentFailures", 1, "Count", dimensions)
        if rolled_back:
            self.put("Rollbacks", 1, "Count", dimensions)
        self.put("LiveModelVersion", model_version, "None", dimensions, flush=True)

    def record_retraining(self, trigger: str, status: str, promoted: bool) -> None:
        dimensions = {"Trigger": trigger, "Status": status}
        self.put("RetrainingEvents", 1, "Count", dimensions)
        self.put(
            "RetrainingCandidatesPromoted" if promoted else "RetrainingCandidatesRejected",
            1,
            "Count",
            {"Trigger": trigger},
            flush=True,
        )

    def record_llm_call(
        self, provider: str, model: str, tokens: int, cost_usd: float, latency_ms: float
    ) -> None:
        dimensions = {"Provider": provider, "Model": model}
        self.put("LLMCalls", 1, "Count", dimensions)
        self.put("LLMTokens", tokens, "Count", dimensions)
        self.put("LLMCostUSD", cost_usd, "None", dimensions)
        self.put("LLMLatency", latency_ms, "Milliseconds", dimensions, flush=True)


_METRICS: CloudWatchMetrics | None = None


def get_cloudwatch_metrics() -> CloudWatchMetrics:
    global _METRICS
    if _METRICS is None:
        _METRICS = CloudWatchMetrics()
    return _METRICS
