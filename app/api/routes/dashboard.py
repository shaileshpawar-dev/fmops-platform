"""Operations console and its aggregate data endpoint.

``/api/v1/dashboard`` returns the whole control-plane view in one call, each
section degrading independently. ``/dashboard`` serves the console that renders
it, plus the rest of the API, from ``static/console.html``.

The console is deliberately dependency-free -- no build step, no CDN, no
framework -- so it works in an air-gapped container and there is no second
toolchain to keep alive. Grafana remains the tool for time-series depth; this
answers "what is the platform doing right now?": which model is live, what the
gates decided, what drifted, what got rolled back and why.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Query
from fastapi.responses import HTMLResponse

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)
router = APIRouter(tags=["dashboard"])


@router.get("/api/v1/dashboard", summary="Everything the dashboard renders")
def dashboard_data(
    model: str | None = Query(
        default=None,
        description="Model the per-model sections describe. Defaults to the most "
        "recently active model.",
    ),
) -> dict[str, Any]:
    """Aggregated state for the dashboard, in one call.

    Platform-wide sections (every model, jobs, counts, activity) plus the
    per-model sections for one focus model. Each section degrades
    independently: a failure gathering LLM stats must not blank out the model
    section.
    """
    settings = get_settings()
    name = model or _focus_model()
    payload: dict[str, Any] = {
        "service": {
            "name": settings.service_name,
            "version": settings.version,
            "environment": settings.environment,
            "git_commit": settings.git_commit[:12],
            "deployment_provider": settings.deployment.provider,
            "registry_backend": settings.tracking.registry_backend,
            "llm_provider": settings.llm.provider,
            "aws_enabled": settings.aws.enabled,
        }
    }

    # Live model quality is the single most expensive thing on this page (a
    # join over the inference log plus sklearn metrics). Both the system panel
    # and the retraining trigger need it for the same model version at the same
    # instant, so compute it once here and hand the same value to both. This is
    # de-duplication within one request, not a cache: nothing is retained
    # between requests and the numbers are identical to computing it twice.
    live = _safe_value(lambda: _live_performance(name), "live_performance")

    payload["focus_model"] = name
    payload["platform"] = _safe(_platform_section, "platform")
    payload["models"] = _safe(_models_section, "models")
    payload["jobs"] = _safe(_jobs_section, "jobs")
    payload["activity"] = _safe(_activity_section, "activity")
    payload["model"] = _safe(lambda: _model_section(name), "model")
    payload["deployment"] = _safe(lambda: _deployment_section(name), "deployment")
    payload["drift"] = _safe(lambda: _drift_section(name), "drift")
    payload["system"] = _safe(lambda: _system_section(name, live), "system")
    payload["llm"] = _safe(_llm_section, "llm")
    payload["retraining"] = _safe(lambda: _retraining_section(name, live), "retraining")
    payload["alerts"] = _safe(_alerts_section, "alerts")
    return payload


def _focus_model() -> str:
    """The model with the most recent registry activity, else the reference model."""
    from app.core.db import get_database
    from app.registry.context import default_model_name

    row = get_database().query_one(
        "SELECT name FROM model_versions ORDER BY updated_at DESC, id DESC LIMIT 1"
    )
    return row["name"] if row else default_model_name()


def _live_performance(name: str):
    """Live quality for the focus model, computed once per request."""
    from app.monitoring.service import get_monitoring_service

    return get_monitoring_service().live_performance(model_name=name)


def _platform_section() -> dict[str, Any]:
    """Counts of what actually exists. Every number is a COUNT over a table."""
    from app.core.db import get_database
    from app.core.utils import iso_days_ago
    from app.data.versioning import get_dataset_registry

    db = get_database()

    def count(sql: str, params: tuple = ()) -> int:
        return int(db.scalar(sql, params, 0))

    week = iso_days_ago(7)
    return {
        "available": True,
        "models": count("SELECT COUNT(DISTINCT name) FROM model_versions"),
        "model_versions": count("SELECT COUNT(*) FROM model_versions"),
        "datasets": len(get_dataset_registry().list_versions()),
        "training_runs": count("SELECT COUNT(*) FROM training_runs")
        + count("SELECT COUNT(*) FROM automl_runs"),
        "active_deployments": count(
            "SELECT COUNT(*) FROM (SELECT endpoint_name FROM deployments "
            "WHERE state IN ('live', 'in_progress') GROUP BY endpoint_name)"
        ),
        "predictions_7d": count(
            "SELECT COUNT(*) FROM inference_log WHERE shadow = 0 AND created_at >= ?", (week,)
        ),
        "drift_detected_7d": count(
            "SELECT COUNT(*) FROM drift_reports WHERE drift_detected = 1 AND created_at >= ?",
            (week,),
        ),
        "retraining_events": count("SELECT COUNT(*) FROM retraining_events"),
        "open_alerts": count("SELECT COUNT(*) FROM alerts WHERE acknowledged = 0"),
    }


def _models_section() -> dict[str, Any]:
    """One line per registered model: what serves, where, and whether it drifted."""
    from app.core.db import get_database
    from app.deployment.manager import get_deployment_manager
    from app.registry.context import endpoint_for, signature_of
    from app.registry.factory import get_registry

    registry = get_registry()
    manager = get_deployment_manager()
    db = get_database()
    out = []
    for item in registry.list_models():
        name = item["name"]
        serving = registry.get_serving(name)
        deployment = manager.status(endpoint_for(name))
        described = serving or registry.get_latest(name)
        signature = signature_of(described) if described else None
        drift = db.query_one(
            "SELECT drift_detected, created_at FROM drift_reports WHERE model_name = ? "
            "ORDER BY created_at DESC LIMIT 1",
            (name,),
        )
        pending = int(
            db.scalar(
                "SELECT COUNT(*) FROM model_versions WHERE name = ? AND status = 'pending'",
                (name,),
                0,
            )
        )
        out.append(
            {
                "name": name,
                "versions": item.get("versions"),
                "endpoint": endpoint_for(name),
                "serving_version": serving.version if serving else None,
                "serving_stage": serving.stage.value if serving else None,
                "roc_auc": float(serving.metrics.get("roc_auc", 0.0)) if serving else None,
                "target": signature.target if signature else None,
                "deployment_state": deployment.state.value if deployment else None,
                "live_version": deployment.current_version if deployment else None,
                "last_drift_detected": bool(drift["drift_detected"]) if drift else None,
                "last_drift_at": drift["created_at"] if drift else None,
                "awaiting_approval": pending,
                "updated_at": item.get("updated_at"),
            }
        )
    return {"available": True, "models": out}


def _jobs_section() -> dict[str, Any]:
    from app.jobs.store import get_job_store

    store = get_job_store()
    return {
        "available": True,
        "counts": store.counts(),
        "recent": [
            {
                k: job.get(k)
                for k in (
                    "id",
                    "kind",
                    "status",
                    "model_name",
                    "resource_id",
                    "error",
                    "created_at",
                    "started_at",
                    "finished_at",
                )
            }
            for job in store.list(limit=8)
        ],
    }


def _activity_section() -> dict[str, Any]:
    """What happened, newest first -- straight from the audit log."""
    from app.core.audit import get_audit_log

    entries = get_audit_log().recent(20)
    return {
        "available": True,
        "events": [
            {
                "action": e.get("action"),
                "resource_type": e.get("resource_type"),
                "resource_id": e.get("resource_id"),
                "outcome": e.get("outcome"),
                "actor": e.get("actor"),
                "created_at": e.get("created_at"),
            }
            for e in entries
        ],
    }


def _safe_value(fn, name: str):
    """Like :func:`_safe` but for a value the sections share; None on failure."""
    try:
        return fn()
    except Exception as exc:
        logger.warning("dashboard.section_failed", extra={"section": name, "error": str(exc)})
        return None


def _safe(fn, name: str) -> dict[str, Any]:
    try:
        return fn()
    except Exception as exc:
        logger.warning("dashboard.section_failed", extra={"section": name, "error": str(exc)})
        return {"error": str(exc), "available": False}


def _model_section(name: str) -> dict[str, Any]:
    from app.registry.factory import get_registry

    settings = get_settings()
    registry = get_registry()

    production = registry.get_production(name)
    previous = registry.previous_production(name)
    versions = registry.list_versions(name)

    return {
        "available": production is not None,
        "model_name": name,
        "current_version": production.version if production else None,
        "current_stage": production.stage.value if production else None,
        "previous_version": previous.version if previous else None,
        "algorithm": production.algorithm if production else None,
        "dataset_version": production.dataset_version if production else None,
        "git_commit": (production.git_commit[:12] if production else None),
        "metrics": production.metrics if production else {},
        "total_versions": len(versions),
        "versions": [
            {
                "version": v.version,
                "stage": v.stage.value,
                "status": v.status.value,
                "roc_auc": round(float(v.metrics.get("roc_auc", 0.0)), 4),
                "f1": round(float(v.metrics.get("f1", 0.0)), 4),
                "created_at": v.created_at,
            }
            for v in versions[:10]
        ],
        "thresholds": settings.approval.model_dump(mode="json"),
    }


def _deployment_section(name: str) -> dict[str, Any]:
    from app.deployment.manager import get_deployment_manager
    from app.registry.context import endpoint_for

    manager = get_deployment_manager()
    endpoint = endpoint_for(name)
    deployment = manager.status(endpoint)
    health = manager.health(endpoint)
    recent = manager.list(endpoint, limit=5)

    return {
        "available": deployment is not None,
        "endpoint": health.endpoint_name,
        "provider": manager.provider.name,
        "state": deployment.state.value if deployment else None,
        "strategy": deployment.strategy.value if deployment else None,
        "health": health.status.value,
        "checks": health.checks,
        "current_version": deployment.current_version if deployment else None,
        "previous_version": deployment.previous_version if deployment else None,
        "candidate_version": deployment.candidate_version if deployment else None,
        "shadow_version": deployment.shadow_version if deployment else None,
        "traffic": deployment.traffic if deployment else {},
        "rolled_back": (
            deployment.state.value in ("rolled_back", "rolling_back") if deployment else False
        ),
        "history": [
            {
                "id": d.id,
                "strategy": d.strategy.value,
                "state": d.state.value,
                "current_version": d.current_version,
                "created_at": d.created_at,
                "message": d.message,
            }
            for d in recent
        ],
    }


def _drift_section(name: str) -> dict[str, Any]:
    from app.monitoring.drift import recent_drift_reports

    settings = get_settings()
    reports = recent_drift_reports(name, limit=10)
    latest = reports[0] if reports else None
    detail = latest.get("report", {}) if latest else {}

    return {
        "available": latest is not None,
        "threshold": settings.drift.threshold,
        "engine": settings.drift.engine,
        "drift_detected": bool(latest["drift_detected"]) if latest else False,
        "dataset_drift_score": latest["dataset_drift_score"] if latest else None,
        "prediction_drift_score": latest["prediction_drift_score"] if latest else None,
        "concept_drift_status": latest["concept_drift_status"] if latest else "unavailable",
        "concept_drift_detail": detail.get("concept_drift_detail", ""),
        "drifted_features": latest["drifted_features"] if latest else [],
        "feature_drift": [
            {
                "feature": f["feature"],
                "score": f["score"],
                "drifted": f["drifted"],
                "test": f["test"],
                "statistic": f["statistic"],
            }
            for f in detail.get("feature_drift", [])
        ],
        "checked_at": latest["created_at"] if latest else None,
        "history": [
            {
                "created_at": r["created_at"],
                "score": r["dataset_drift_score"],
                "detected": bool(r["drift_detected"]),
            }
            for r in reports
        ],
    }


def _system_section(name: str, performance: Any = None) -> dict[str, Any]:
    from app.monitoring.resource_monitor import latest_resources
    from app.monitoring.service import get_monitoring_service

    # Call each collector exactly once. Going through summary() here would
    # recompute service_metrics and live_performance a second and third time,
    # which on a busy inference log is the most expensive thing on the page.
    service = get_monitoring_service()
    metrics = service.service_metrics(60, name)
    resources = latest_resources()
    if performance is None:
        performance = service.live_performance(model_name=name)

    return {
        "available": True,
        "requests": metrics.request_count,
        "errors": metrics.error_count,
        "error_rate": metrics.error_rate,
        "throughput_rpm": metrics.throughput_rpm,
        "latency_p50_ms": metrics.latency.p50_ms,
        "latency_p95_ms": metrics.latency.p95_ms,
        "latency_p99_ms": metrics.latency.p99_ms,
        "slo_latency_ms": metrics.slo_latency_ms,
        "slo_error_rate": metrics.slo_error_rate,
        "latency_slo_met": metrics.latency_slo_met,
        "error_slo_met": metrics.error_slo_met,
        "cpu_percent": resources.cpu_percent,
        "memory_percent": resources.memory_percent,
        "process_rss_mb": resources.process_rss_mb,
        "gpu_available": resources.gpu_available,
        "gpu": resources.gpu,
        "live_performance": performance.model_dump(mode="json"),
    }


def _llm_section() -> dict[str, Any]:
    from app.llmops.cost import get_cost_tracker
    from app.llmops.evaluation.runner import recent_evaluations
    from app.llmops.prompts.registry import get_prompt_registry
    from app.llmops.token_tracking import get_trace_store

    settings = get_settings()
    store = get_trace_store()
    totals = store.token_totals(30)
    cost = get_cost_tracker().summary()
    evaluations = recent_evaluations(5)
    registry = get_prompt_registry()

    prompts = {}
    for name in registry.list_names():
        versions = registry.list_versions(name)
        prompts[name] = {
            "latest": versions[-1].version,
            "versions": [v.version for v in versions],
        }

    return {
        "available": True,
        "provider": settings.llm.provider,
        "model": settings.llm.model,
        "is_mock": settings.llm.provider == "mock",
        "calls": totals["calls"],
        "error_rate": totals["error_rate"],
        "total_tokens": totals["total_tokens"],
        "input_tokens": totals["input_tokens"],
        "output_tokens": totals["output_tokens"],
        "avg_latency_ms": totals["avg_latency_ms"],
        "avg_tokens_per_call": totals["avg_tokens_per_call"],
        "today_cost_usd": cost.today_cost_usd,
        "month_cost_usd": cost.month_cost_usd,
        "daily_budget_usd": cost.daily_budget_usd,
        "daily_budget_used_pct": cost.daily_budget_used_pct,
        "monthly_budget_used_pct": cost.monthly_budget_used_pct,
        "by_model": {k: v.model_dump(mode="json") for k, v in cost.by_model.items()},
        "by_prompt_version": store.by_prompt_version(30),
        "prompts": prompts,
        "evaluations": [
            {
                "id": e["id"],
                "dataset": e["dataset"],
                "model": e["model"],
                "prompt_version": e["prompt_version"],
                "overall": round(float(e["aggregate"].get("overall", 0.0)), 4),
                "created_at": e["created_at"],
            }
            for e in evaluations
        ],
    }


def _retraining_section(name: str, live_performance: Any = None) -> dict[str, Any]:
    from app.retraining.trigger import evaluate_trigger, get_event_store

    events = get_event_store().recent(5, model_name=name)
    decision = evaluate_trigger(live_performance=live_performance, model_name=name)
    return {
        "available": True,
        "would_trigger": decision.should_retrain,
        "trigger": decision.trigger.value if decision.trigger else None,
        "reason": decision.reason,
        "suppressed_by_cooldown": decision.suppressed_by_cooldown,
        "checks": decision.checks,
        "events": [
            {
                "id": e.id,
                "trigger": e.trigger.value,
                "status": e.status.value,
                "decision": e.decision,
                "baseline_version": e.baseline_version,
                "candidate_version": e.candidate_version,
                "reason": e.reason,
                "created_at": e.created_at,
            }
            for e in events
        ],
    }


def _alerts_section() -> dict[str, Any]:
    from app.monitoring.alerts import get_alert_manager

    manager = get_alert_manager()
    alerts = manager.recent(15)
    return {
        "available": True,
        "open_count": manager.open_count(),
        "alerts": [
            {
                "id": a.id,
                "severity": a.severity.value,
                "category": a.category.value,
                "title": a.title,
                "message": a.message,
                "acknowledged": a.acknowledged,
                "created_at": a.created_at,
            }
            for a in alerts
        ],
    }


_CONSOLE_PATH = Path(__file__).resolve().parents[1] / "static" / "console.html"


@lru_cache(maxsize=1)
def _console_html() -> str:
    """Read the console once and keep it.

    The file is part of the image and cannot change under a running process,
    so re-reading it per request would be pure syscall overhead on a page that
    is polled. If it is missing the API must still serve -- the console is a
    convenience over the JSON API, not a dependency of it.
    """
    try:
        return _CONSOLE_PATH.read_text(encoding="utf-8")
    except OSError as exc:  # pragma: no cover - only when the image is broken
        logger.error(
            "dashboard.console_missing", extra={"path": str(_CONSOLE_PATH), "error": str(exc)}
        )
        return (
            "<!doctype html><html><head><title>FMOps Platform</title></head><body>"
            "<h1>FMOps Platform</h1><p>The console asset is not present in this build. "
            'The JSON API is unaffected: see <a href="/api/v1/dashboard">/api/v1/dashboard</a> '
            'and <a href="/docs">/docs</a>.</p></body></html>'
        )


@router.get("/dashboard", response_class=HTMLResponse, include_in_schema=False)
def dashboard_page() -> HTMLResponse:
    return HTMLResponse(content=_console_html())
