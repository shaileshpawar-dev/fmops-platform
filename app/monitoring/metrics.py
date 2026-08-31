"""Prometheus metrics and the metrics probe.

Two distinct things live here, and the distinction matters:

* **Counters/histograms** exported at ``/metrics`` for Prometheus to scrape.
  These are process-local and reset when the process restarts.
* :class:`MetricsProbe`, which reads the *persisted* inference log. Canary
  promotion and rollback decisions use the probe, not the in-process counters,
  because a decision must survive a restart and must be identical across
  workers.

The CloudWatch backend mirrors a small, deliberately chosen subset of metrics
(not everything -- CloudWatch custom metrics are billed per metric per month).
"""

from __future__ import annotations

import threading
import time
from typing import Any

from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from prometheus_client.core import REGISTRY as GLOBAL_REGISTRY

from app.core.config import Settings, get_settings
from app.core.logging import get_logger
from app.monitoring.inference_log import InferenceLog, get_inference_log

logger = get_logger(__name__)

# Latency buckets tuned for a sub-second scoring endpoint; the default
# prometheus buckets top out too coarsely to see a 150ms SLO breach.
LATENCY_BUCKETS = (
    0.005,
    0.01,
    0.025,
    0.05,
    0.075,
    0.1,
    0.15,
    0.25,
    0.5,
    0.75,
    1.0,
    2.5,
    5.0,
)


class PlatformMetrics:
    """All Prometheus metrics exported by the platform."""

    def __init__(self, registry: CollectorRegistry | None = None, namespace: str = "fmops"):
        self.registry = registry if registry is not None else GLOBAL_REGISTRY
        ns = namespace

        # --- serving --------------------------------------------------------- #
        self.predictions_total = Counter(
            f"{ns}_predictions_total",
            "Predictions served",
            ["model_name", "model_version", "variant", "outcome"],
            registry=self.registry,
        )
        self.prediction_latency = Histogram(
            f"{ns}_prediction_latency_seconds",
            "End-to-end prediction latency",
            ["model_name", "model_version", "variant"],
            buckets=LATENCY_BUCKETS,
            registry=self.registry,
        )
        self.prediction_probability = Histogram(
            f"{ns}_prediction_probability",
            "Distribution of predicted probabilities (prediction drift signal)",
            ["model_name", "model_version"],
            buckets=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
            registry=self.registry,
        )
        self.http_requests_total = Counter(
            f"{ns}_http_requests_total",
            "HTTP requests",
            ["method", "path", "status"],
            registry=self.registry,
        )
        self.http_request_latency = Histogram(
            f"{ns}_http_request_latency_seconds",
            "HTTP request latency",
            ["method", "path"],
            buckets=LATENCY_BUCKETS,
            registry=self.registry,
        )
        self.errors_total = Counter(
            f"{ns}_errors_total",
            "Errors by code",
            ["code", "component"],
            registry=self.registry,
        )

        # --- model / deployment ---------------------------------------------- #
        self.model_info = Gauge(
            f"{ns}_model_info",
            "Serving model metadata (always 1; labels carry the information)",
            ["model_name", "model_version", "stage", "algorithm"],
            registry=self.registry,
        )
        self.model_metric = Gauge(
            f"{ns}_model_metric",
            "Offline evaluation metrics of the serving model",
            ["model_name", "model_version", "metric"],
            registry=self.registry,
        )
        self.live_model_metric = Gauge(
            f"{ns}_live_model_metric",
            "Model quality computed from labelled production traffic",
            ["model_name", "model_version", "metric"],
            registry=self.registry,
        )
        self.deployment_traffic = Gauge(
            f"{ns}_deployment_traffic_percent",
            "Percentage of live traffic per model version",
            ["endpoint", "model_name", "model_version"],
            registry=self.registry,
        )
        self.deployment_state = Gauge(
            f"{ns}_deployment_state",
            "Deployment state (1 for the active state)",
            ["endpoint", "state"],
            registry=self.registry,
        )
        self.rollbacks_total = Counter(
            f"{ns}_rollbacks_total",
            "Rollbacks performed",
            ["endpoint", "reason"],
            registry=self.registry,
        )
        self.deployments_total = Counter(
            f"{ns}_deployments_total",
            "Deployments attempted",
            ["endpoint", "strategy", "outcome"],
            registry=self.registry,
        )

        # --- drift / data quality --------------------------------------------- #
        self.drift_score = Gauge(
            f"{ns}_drift_score",
            "Drift score by type (data, prediction, concept)",
            ["model_name", "drift_type"],
            registry=self.registry,
        )
        self.feature_drift_score = Gauge(
            f"{ns}_feature_drift_score",
            "Per-feature drift score",
            ["model_name", "feature"],
            registry=self.registry,
        )
        self.drift_detected = Gauge(
            f"{ns}_drift_detected",
            "1 when the most recent drift scan exceeded the threshold",
            ["model_name"],
            registry=self.registry,
        )
        self.drift_checks_total = Counter(
            f"{ns}_drift_checks_total",
            "Drift scans run",
            ["model_name", "result"],
            registry=self.registry,
        )
        self.validation_runs_total = Counter(
            f"{ns}_data_validation_runs_total",
            "Data validation suite runs",
            ["dataset", "result"],
            registry=self.registry,
        )

        # --- training / retraining -------------------------------------------- #
        self.training_runs_total = Counter(
            f"{ns}_training_runs_total",
            "Training runs",
            ["model_name", "outcome"],
            registry=self.registry,
        )
        self.training_duration = Histogram(
            f"{ns}_training_duration_seconds",
            "Training run duration",
            ["model_name"],
            buckets=(10, 30, 60, 120, 300, 600, 1800, 3600),
            registry=self.registry,
        )
        self.retraining_events_total = Counter(
            f"{ns}_retraining_events_total",
            "Retraining events",
            ["trigger", "status", "decision"],
            registry=self.registry,
        )
        self.approval_decisions_total = Counter(
            f"{ns}_approval_decisions_total",
            "Approval gate decisions",
            ["model_name", "decision"],
            registry=self.registry,
        )

        # --- resources --------------------------------------------------------- #
        self.cpu_percent = Gauge(
            f"{ns}_process_cpu_percent", "Process CPU usage", registry=self.registry
        )
        self.memory_percent = Gauge(
            f"{ns}_system_memory_percent", "System memory usage", registry=self.registry
        )
        self.process_memory_mb = Gauge(
            f"{ns}_process_memory_mb", "Process RSS in MB", registry=self.registry
        )
        self.gpu_utilization = Gauge(
            f"{ns}_gpu_utilization_percent",
            "GPU utilisation (only exported when a GPU is present)",
            ["device"],
            registry=self.registry,
        )

        # --- LLMOps ------------------------------------------------------------ #
        self.llm_requests_total = Counter(
            f"{ns}_llm_requests_total",
            "LLM invocations",
            ["provider", "model", "prompt_name", "prompt_version", "outcome"],
            registry=self.registry,
        )
        self.llm_tokens_total = Counter(
            f"{ns}_llm_tokens_total",
            "LLM tokens consumed",
            ["provider", "model", "direction"],
            registry=self.registry,
        )
        self.llm_cost_usd_total = Counter(
            f"{ns}_llm_cost_usd_total",
            "Estimated LLM spend in USD",
            ["provider", "model"],
            registry=self.registry,
        )
        self.llm_latency = Histogram(
            f"{ns}_llm_latency_seconds",
            "LLM call latency",
            ["provider", "model"],
            buckets=(0.1, 0.25, 0.5, 1, 2, 5, 10, 20, 30, 60),
            registry=self.registry,
        )
        self.llm_eval_score = Gauge(
            f"{ns}_llm_evaluation_score",
            "LLM evaluation scores",
            ["suite", "model", "prompt_version", "metric"],
            registry=self.registry,
        )
        self.llm_safety_findings_total = Counter(
            f"{ns}_llm_safety_findings_total",
            "Safety screen findings",
            ["check", "severity"],
            registry=self.registry,
        )

        # --- alerts ------------------------------------------------------------- #
        self.alerts_total = Counter(
            f"{ns}_alerts_total",
            "Alerts raised",
            ["severity", "category"],
            registry=self.registry,
        )

    def render(self) -> bytes:
        return generate_latest(self.registry)


_METRICS: PlatformMetrics | None = None
_METRICS_LOCK = threading.Lock()


def get_metrics() -> PlatformMetrics:
    """Process-wide metrics registry."""
    global _METRICS
    if _METRICS is None:
        with _METRICS_LOCK:
            if _METRICS is None:
                settings = get_settings()
                _METRICS = PlatformMetrics(namespace=settings.monitoring.metrics_namespace)
    return _METRICS


def reset_metrics(registry: CollectorRegistry | None = None) -> PlatformMetrics:
    """Rebuild the metric set against a fresh registry (used by tests)."""
    global _METRICS
    with _METRICS_LOCK:
        _METRICS = PlatformMetrics(
            registry=registry if registry is not None else CollectorRegistry(),
            namespace=get_settings().monitoring.metrics_namespace,
        )
    return _METRICS


def render_metrics() -> bytes:
    return get_metrics().render()


# --------------------------------------------------------------------------- #
# Probe
# --------------------------------------------------------------------------- #
class MetricsProbe:
    """Observed serving health, read from the persisted inference log.

    Implements :class:`~app.deployment.strategy.HealthProbe`. Reading from the
    database rather than in-process counters means a canary decision is the same
    no matter which worker evaluates it, and survives a restart mid-rollout.
    """

    def __init__(
        self, log: InferenceLog | None = None, settings: Settings | None = None
    ) -> None:
        self._log = log
        self.settings = settings or get_settings()

    @property
    def log(self) -> InferenceLog:
        if self._log is None:
            self._log = get_inference_log()
        return self._log

    def observe(
        self, endpoint_name: str, model_name: str, version: int, window_seconds: int
    ) -> tuple[int, float, float]:
        minutes = max(1, round(window_seconds / 60))
        stats = self.log.window_stats(
            model_name=model_name, model_version=version, minutes=minutes
        )
        return (
            int(stats["request_count"]),
            float(stats["error_rate"]),
            float(stats["latency_p95_ms"]),
        )


# --------------------------------------------------------------------------- #
# Recording helpers -- the only places that touch metric objects directly
# --------------------------------------------------------------------------- #
def record_prediction(
    model_name: str,
    model_version: int | None,
    variant: str,
    latency_seconds: float,
    probability: float | None = None,
    outcome: str = "success",
) -> None:
    metrics = get_metrics()
    version_label = str(model_version if model_version is not None else "unknown")
    metrics.predictions_total.labels(
        model_name=model_name,
        model_version=version_label,
        variant=variant,
        outcome=outcome,
    ).inc()
    metrics.prediction_latency.labels(
        model_name=model_name, model_version=version_label, variant=variant
    ).observe(latency_seconds)
    if probability is not None:
        metrics.prediction_probability.labels(
            model_name=model_name, model_version=version_label
        ).observe(float(probability))


def record_error(code: str, component: str) -> None:
    get_metrics().errors_total.labels(code=code, component=component).inc()


def record_http(method: str, path: str, status: int, latency_seconds: float) -> None:
    metrics = get_metrics()
    metrics.http_requests_total.labels(method=method, path=path, status=str(status)).inc()
    metrics.http_request_latency.labels(method=method, path=path).observe(latency_seconds)


def set_model_info(
    model_name: str,
    model_version: int,
    stage: str,
    algorithm: str,
    metrics_values: dict[str, float] | None = None,
) -> None:
    metrics = get_metrics()
    metrics.model_info.labels(
        model_name=model_name,
        model_version=str(model_version),
        stage=stage,
        algorithm=algorithm,
    ).set(1)
    for name, value in (metrics_values or {}).items():
        if isinstance(value, (int, float)):
            metrics.model_metric.labels(
                model_name=model_name, model_version=str(model_version), metric=name
            ).set(float(value))


def set_deployment_metrics(
    endpoint: str, model_name: str, traffic: dict[str, float], state: str
) -> None:
    metrics = get_metrics()
    for version, percent in traffic.items():
        metrics.deployment_traffic.labels(
            endpoint=endpoint, model_name=model_name, model_version=str(version)
        ).set(float(percent))
    for candidate in (
        "pending",
        "in_progress",
        "live",
        "rolling_back",
        "rolled_back",
        "failed",
        "terminated",
    ):
        metrics.deployment_state.labels(endpoint=endpoint, state=candidate).set(
            1 if candidate == state else 0
        )


def set_drift_metrics(
    model_name: str,
    dataset_score: float,
    prediction_score: float | None,
    concept_score: float | None,
    detected: bool,
    feature_scores: dict[str, float] | None = None,
) -> None:
    metrics = get_metrics()
    metrics.drift_score.labels(model_name=model_name, drift_type="data").set(dataset_score)
    if prediction_score is not None:
        metrics.drift_score.labels(model_name=model_name, drift_type="prediction").set(
            prediction_score
        )
    if concept_score is not None:
        metrics.drift_score.labels(model_name=model_name, drift_type="concept").set(
            concept_score
        )
    metrics.drift_detected.labels(model_name=model_name).set(1 if detected else 0)
    metrics.drift_checks_total.labels(
        model_name=model_name, result="drifted" if detected else "stable"
    ).inc()
    for feature, score in (feature_scores or {}).items():
        metrics.feature_drift_score.labels(model_name=model_name, feature=feature).set(
            float(score)
        )


def set_live_performance(
    model_name: str, model_version: int, values: dict[str, float]
) -> None:
    metrics = get_metrics()
    for name, value in values.items():
        if isinstance(value, (int, float)):
            metrics.live_model_metric.labels(
                model_name=model_name, model_version=str(model_version), metric=name
            ).set(float(value))


def record_llm_call(
    provider: str,
    model: str,
    prompt_name: str,
    prompt_version: str,
    input_tokens: int,
    output_tokens: int,
    cost_usd: float,
    latency_seconds: float,
    outcome: str = "success",
) -> None:
    metrics = get_metrics()
    metrics.llm_requests_total.labels(
        provider=provider,
        model=model,
        prompt_name=prompt_name or "adhoc",
        prompt_version=prompt_version or "none",
        outcome=outcome,
    ).inc()
    metrics.llm_tokens_total.labels(provider=provider, model=model, direction="input").inc(
        input_tokens
    )
    metrics.llm_tokens_total.labels(provider=provider, model=model, direction="output").inc(
        output_tokens
    )
    if cost_usd > 0:
        metrics.llm_cost_usd_total.labels(provider=provider, model=model).inc(cost_usd)
    metrics.llm_latency.labels(provider=provider, model=model).observe(latency_seconds)


def record_alert(severity: str, category: str) -> None:
    get_metrics().alerts_total.labels(severity=severity, category=category).inc()


class Timer:
    """Context manager returning elapsed seconds."""

    def __init__(self) -> None:
        self.elapsed = 0.0
        self._start = 0.0

    def __enter__(self) -> Timer:
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc: Any) -> bool:
        self.elapsed = time.perf_counter() - self._start
        return False

    @property
    def elapsed_ms(self) -> float:
        return self.elapsed * 1000.0
