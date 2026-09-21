"""Scheduled automation: the loop that makes monitoring act on its own.

Every ``automation.interval_minutes``, for each model that has a serving
version, the platform:

1. queues a **drift scan** when enough predictions have arrived since the last
   one (``drift.min_samples``) -- a scan over the same traffic twice tells
   nothing new;
2. evaluates the **retraining trigger** and queues a retraining job when it
   fires -- which it only does when there is new labelled data to learn from,
   and never inside the cooldown, so drift cannot become a retraining loop;
3. runs the **SLO watchdog** on a deployed endpoint, raising alerts on breach.

Exactly one process does this per interval: the tick is claimed with the same
database write lock the job queue uses. Work it starts is ordinary jobs --
visible, logged, cancellable -- and idempotency keys stop a slow job from being
queued twice by consecutive ticks.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from app.core.config import Settings, get_settings
from app.core.db import get_database
from app.core.logging import get_logger
from app.core.utils import utcnow_iso

logger = get_logger(__name__)

_KEY = "automation.last_tick"


def claim_tick(settings: Settings) -> bool:
    """True for exactly one caller per interval, across every process."""
    interval = settings.automation.interval_minutes
    if not settings.automation.enabled or interval <= 0:
        return False
    now = datetime.now(UTC)
    with get_database().transaction() as conn:
        row = conn.execute("SELECT value FROM schema_meta WHERE key = ?", (_KEY,)).fetchone()
        if row is not None:
            last = datetime.fromisoformat(row["value"].replace("Z", "+00:00"))
            if now - last < timedelta(minutes=interval):
                return False
        conn.execute(
            "INSERT INTO schema_meta (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (_KEY, utcnow_iso()),
        )
    return True


def run_tick(settings: Settings | None = None) -> dict[str, Any]:
    """One pass over every serving model. Returns what it did, for the log."""
    from app.deployment.manager import get_deployment_manager
    from app.jobs.runner import get_job_runner
    from app.monitoring.service import get_monitoring_service
    from app.registry.context import endpoint_for
    from app.registry.factory import get_registry
    from app.retraining.trigger import evaluate_trigger

    settings = settings or get_settings()
    registry = get_registry()
    runner = get_job_runner()
    manager = get_deployment_manager()
    summary: dict[str, Any] = {
        "models": 0,
        "drift_scans": [],
        "retraining": [],
        "watchdog": [],
    }

    for model in registry.list_models():
        name = model["name"]
        serving = registry.get_serving(name)
        if serving is None:
            continue
        summary["models"] += 1
        try:
            if _new_predictions_since_last_scan(name) >= settings.drift.min_samples:
                job = runner.submit(
                    "drift_scan",
                    {"model_name": name},
                    model_name=name,
                    requested_by="automation",
                    idempotency_key=f"automation:drift:{name}",
                )
                summary["drift_scans"].append(job["id"])

            decision = evaluate_trigger(model_name=name)
            if decision.should_retrain:
                job = runner.submit(
                    "retraining",
                    {"model_name": name, "force": False},
                    model_name=name,
                    requested_by="automation",
                    idempotency_key=f"automation:retrain:{name}",
                )
                summary["retraining"].append(job["id"])

            if manager.status(endpoint_for(name, settings)) is not None:
                result = get_monitoring_service().health_watchdog(15, model_name=name)
                if result.get("breaches"):
                    summary["watchdog"].append({"model": name, "breaches": result["breaches"]})
        except Exception as exc:  # one model's failure must not stop the others
            logger.error("automation.model_failed", extra={"model": name, "error": str(exc)})

    logger.info("automation.tick", extra=summary)
    return summary


def _new_predictions_since_last_scan(model_name: str) -> int:
    db = get_database()
    last = db.scalar(
        "SELECT MAX(created_at) FROM drift_reports WHERE model_name = ?", (model_name,), None
    )
    if last is None:
        return int(
            db.scalar(
                "SELECT COUNT(*) FROM inference_log WHERE model_name = ? AND shadow = 0 "
                "AND status = 'ok'",
                (model_name,),
                0,
            )
        )
    return int(
        db.scalar(
            "SELECT COUNT(*) FROM inference_log WHERE model_name = ? AND shadow = 0 "
            "AND status = 'ok' AND created_at > ?",
            (model_name, last),
            0,
        )
    )
