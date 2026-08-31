"""Monitoring, drift and alert endpoints."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Query, Response

from app.core.config import get_settings
from app.core.logging import get_logger
from app.monitoring.alerts import get_alert_manager
from app.monitoring.drift import recent_drift_reports
from app.monitoring.metrics import render_metrics
from app.monitoring.resource_monitor import sample_resources
from app.monitoring.service import get_monitoring_service
from app.schemas.common import AlertCategory, Severity
from app.schemas.evaluation import (
    Alert,
    DriftReport,
    LivePerformance,
    MonitoringSummary,
    ResourceUsage,
    ServiceMetrics,
)

logger = get_logger(__name__)
router = APIRouter(tags=["monitoring"])

PROMETHEUS_CONTENT_TYPE = "text/plain; version=0.0.4; charset=utf-8"


@router.get("/metrics", summary="Prometheus metrics", include_in_schema=False)
def metrics() -> Response:
    """Prometheus scrape endpoint (OpenMetrics text exposition)."""
    return Response(content=render_metrics(), media_type=PROMETHEUS_CONTENT_TYPE)


@router.get(
    "/api/v1/monitoring/summary",
    response_model=MonitoringSummary,
    summary="Monitoring summary",
)
def summary(window_minutes: int = Query(default=60, ge=1, le=10080)) -> MonitoringSummary:
    """Everything the dashboard shows in one call."""
    return get_monitoring_service().summary(window_minutes)


@router.get(
    "/api/v1/monitoring/service",
    response_model=ServiceMetrics,
    summary="Latency, throughput and error rate",
)
def service_metrics(window_minutes: int = Query(default=60, ge=1)) -> ServiceMetrics:
    return get_monitoring_service().service_metrics(window_minutes)


@router.get(
    "/api/v1/monitoring/performance",
    response_model=LivePerformance,
    summary="Live model quality from labelled traffic",
)
def live_performance(model_version: int | None = None) -> LivePerformance:
    """Production accuracy, computed only from labelled predictions.

    When no labels have been submitted this returns ``available=false`` with an
    explanation rather than an estimate.
    """
    return get_monitoring_service().live_performance(model_version)


@router.get(
    "/api/v1/monitoring/resources",
    response_model=ResourceUsage,
    summary="CPU / memory / disk / GPU",
)
def resources() -> ResourceUsage:
    return sample_resources()


@router.post("/api/v1/monitoring/watchdog", summary="Evaluate SLOs and raise alerts")
def watchdog(window_minutes: int = Query(default=15, ge=1)) -> dict[str, Any]:
    return get_monitoring_service().health_watchdog(window_minutes)


# --------------------------------------------------------------------------- #
# Drift
# --------------------------------------------------------------------------- #
@router.get("/api/v1/drift", summary="Recent drift reports")
def list_drift(limit: int = Query(default=20, le=100)) -> dict[str, Any]:
    settings = get_settings()
    reports = recent_drift_reports(settings.tracking.registered_model_name, limit)
    return {
        "count": len(reports),
        "threshold": settings.drift.threshold,
        "engine": settings.drift.engine,
        "reports": [{k: v for k, v in r.items() if k != "report"} for r in reports],
    }


@router.get("/api/v1/drift/latest", summary="Most recent drift report")
def latest_drift() -> dict[str, Any]:
    settings = get_settings()
    reports = recent_drift_reports(settings.tracking.registered_model_name, 1)
    if not reports:
        return {
            "found": False,
            "detail": "no drift scan has been run yet; POST /api/v1/drift/scan",
        }
    return {"found": True, "report": reports[0]["report"]}


@router.post("/api/v1/drift/scan", response_model=DriftReport, summary="Run a drift scan")
def scan_drift(
    model_version: int | None = None,
    persist: bool = Query(default=True),
) -> DriftReport:
    """Compare recent production traffic against the training reference window.

    Reports data, feature and prediction drift. Concept drift is reported only
    when ground-truth labels are available; otherwise its status is
    ``unavailable`` and prediction drift is offered as a proxy signal.
    """
    return get_monitoring_service().run_drift_scan(model_version, persist=persist)


# --------------------------------------------------------------------------- #
# Alerts
# --------------------------------------------------------------------------- #
@router.get("/api/v1/alerts", response_model=list[Alert], summary="Recent alerts")
def list_alerts(
    limit: int = Query(default=50, le=500),
    severity: Severity | None = None,
    category: AlertCategory | None = None,
    unacknowledged_only: bool = False,
) -> list[Alert]:
    return get_alert_manager().recent(limit, severity, category, unacknowledged_only)


@router.post("/api/v1/alerts/{alert_id}/acknowledge", summary="Acknowledge an alert")
def acknowledge(alert_id: str) -> dict[str, Any]:
    ok = get_alert_manager().acknowledge(alert_id)
    return {"alert_id": alert_id, "acknowledged": ok}


@router.post("/api/v1/alerts/acknowledge-all", summary="Acknowledge every open alert")
def acknowledge_all() -> dict[str, int]:
    return {"acknowledged": get_alert_manager().acknowledge_all()}


# --------------------------------------------------------------------------- #
# Audit
# --------------------------------------------------------------------------- #
@router.get("/api/v1/audit", summary="Audit log")
def audit_log(limit: int = Query(default=100, le=1000), action: str | None = None) -> dict:
    from app.core.audit import get_audit_log

    entries = get_audit_log().recent(limit, action)
    return {"count": len(entries), "entries": entries}
