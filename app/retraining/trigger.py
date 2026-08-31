"""Retraining triggers.

Decides *whether* retraining should run. Four automatic triggers plus manual:

``drift``        the latest drift scan exceeded the configured threshold
``performance``  live quality (labelled traffic only) dropped below the model's
                 own offline baseline by more than the tolerance
``volume``       enough new production rows have accumulated
``schedule``     a cron-driven periodic refresh (evaluated by CI, not here)
``manual``       someone asked

Two properties matter operationally:

* **Cooldown.** After a retraining event, further automatic triggers are
  suppressed for ``cooldown_minutes``. Without this, a persistent drift
  condition would launch a retraining run on every scan.
* **Trigger evaluation never trains.** It returns a decision object. That
  separation makes the trigger logic trivially testable, and lets the API expose
  "would this fire, and why?" without side effects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.core.config import Settings, get_settings
from app.core.db import Database, dumps, get_database, loads
from app.core.logging import get_logger
from app.core.utils import iso_minutes_ago, utcnow_iso
from app.monitoring.drift import latest_drift_report
from app.monitoring.inference_log import get_inference_log
from app.monitoring.service import get_monitoring_service
from app.registry.factory import get_registry
from app.schemas.common import RetrainingStatus, RetrainingTrigger
from app.schemas.evaluation import RetrainingEvent

logger = get_logger(__name__)


@dataclass
class TriggerDecision:
    """Whether retraining should run, and the evidence for that."""

    should_retrain: bool
    trigger: RetrainingTrigger | None = None
    reason: str = ""
    evidence: dict[str, Any] = field(default_factory=dict)
    suppressed_by_cooldown: bool = False
    checks: list[dict[str, Any]] = field(default_factory=list)

    def render_text(self) -> str:
        lines = [
            f"Retraining trigger: {'FIRE' if self.should_retrain else 'no action'}",
        ]
        if self.trigger:
            lines.append(f"  trigger: {self.trigger.value}")
        lines.append(f"  reason : {self.reason}")
        for check in self.checks:
            mark = "HIT " if check.get("fired") else "    "
            lines.append(f"  [{mark}] {check['name']}: {check['detail']}")
        return "\n".join(lines)


class RetrainingTriggerEvaluator:
    """Evaluates every configured trigger without side effects."""

    def __init__(self, settings: Settings | None = None, db: Database | None = None) -> None:
        self.settings = settings or get_settings()
        self._db = db

    @property
    def db(self) -> Database:
        if self._db is None:
            self._db = get_database()
        return self._db

    def evaluate(self, force: bool = False) -> TriggerDecision:
        config = self.settings.retraining
        model_name = self.settings.tracking.registered_model_name
        checks: list[dict[str, Any]] = []

        if force:
            return TriggerDecision(
                should_retrain=True,
                trigger=RetrainingTrigger.MANUAL,
                reason="manual retraining requested",
                checks=checks,
            )

        if not config.enabled:
            return TriggerDecision(
                should_retrain=False,
                reason="automatic retraining is disabled by configuration",
                checks=checks,
            )

        in_cooldown, last_at = self._in_cooldown()
        if in_cooldown:
            return TriggerDecision(
                should_retrain=False,
                reason=(
                    f"a retraining event ran at {last_at}; automatic triggers are "
                    f"suppressed for {config.cooldown_minutes} minutes"
                ),
                suppressed_by_cooldown=True,
                checks=checks,
            )

        fired: TriggerDecision | None = None

        # --- drift ---------------------------------------------------------- #
        if "drift" in config.triggers:
            report = latest_drift_report(model_name)
            if report is None:
                checks.append(
                    {
                        "name": "drift",
                        "fired": False,
                        "detail": "no drift scan has been run yet",
                    }
                )
            else:
                hit = report.drift_detected
                checks.append(
                    {
                        "name": "drift",
                        "fired": hit,
                        "detail": (
                            f"dataset drift {report.dataset_drift_score:.4f} vs "
                            f"threshold {report.threshold:.2f}; "
                            f"{len(report.drifted_features)} features drifted"
                        ),
                    }
                )
                if hit and fired is None:
                    fired = TriggerDecision(
                        should_retrain=True,
                        trigger=RetrainingTrigger.DRIFT,
                        reason=(
                            "drift detected: "
                            f"{len(report.drifted_features)} of "
                            f"{len(report.feature_drift)} features drifted "
                            f"({report.dataset_drift_share:.0%}), mean drift score "
                            f"{report.dataset_drift_score:.4f} "
                            f"(threshold {report.threshold:.2f}); drifted: "
                            f"{', '.join(report.drifted_features[:6])}"
                        ),
                        evidence={
                            "drift_report_id": report.id,
                            "dataset_drift_score": report.dataset_drift_score,
                            "drifted_features": report.drifted_features,
                            "concept_drift_status": report.concept_drift_status,
                        },
                    )

        # --- performance ----------------------------------------------------- #
        if "performance" in config.triggers:
            decision = self._performance_check(model_name, config.performance_drop_tolerance)
            checks.append(decision[0])
            if decision[1] is not None and fired is None:
                fired = decision[1]

        # --- volume ----------------------------------------------------------- #
        if "volume" in config.triggers:
            new_rows = get_inference_log().count(model_name)
            hit = new_rows >= config.min_new_samples
            checks.append(
                {
                    "name": "volume",
                    "fired": hit,
                    "detail": f"{new_rows} logged predictions vs minimum {config.min_new_samples}",
                }
            )
            if hit and fired is None:
                fired = TriggerDecision(
                    should_retrain=True,
                    trigger=RetrainingTrigger.VOLUME,
                    reason=(
                        f"{new_rows} new production samples accumulated "
                        f"(threshold {config.min_new_samples})"
                    ),
                    evidence={"new_samples": new_rows},
                )

        if fired is not None:
            fired.checks = checks
            logger.info(
                "retraining.trigger_fired",
                extra={
                    "trigger": fired.trigger.value if fired.trigger else None,
                    "reason": fired.reason,
                },
            )
            return fired

        return TriggerDecision(
            should_retrain=False,
            reason="no retraining trigger fired",
            checks=checks,
        )

    def _performance_check(
        self, model_name: str, tolerance: float
    ) -> tuple[dict[str, Any], TriggerDecision | None]:
        registry = get_registry()
        serving = registry.get_serving(model_name)
        if serving is None:
            return (
                {
                    "name": "performance",
                    "fired": False,
                    "detail": "no serving model to evaluate",
                },
                None,
            )

        live = get_monitoring_service().live_performance(serving.version)
        if not live.available or live.roc_auc is None:
            return (
                {
                    "name": "performance",
                    "fired": False,
                    "detail": (
                        f"not evaluable: {live.detail} "
                        "(production quality requires ground-truth labels)"
                    ),
                },
                None,
            )

        baseline = float(serving.metrics.get("roc_auc", 0.0))
        drop = baseline - live.roc_auc
        hit = drop > tolerance
        detail = (
            f"live ROC-AUC {live.roc_auc:.4f} vs offline baseline {baseline:.4f} "
            f"(drop {drop:+.4f}, tolerance {tolerance:.4f}, "
            f"{live.labelled_samples} labelled rows)"
        )
        if not hit:
            return ({"name": "performance", "fired": False, "detail": detail}, None)

        return (
            {"name": "performance", "fired": True, "detail": detail},
            TriggerDecision(
                should_retrain=True,
                trigger=RetrainingTrigger.PERFORMANCE,
                reason=(
                    f"live model quality degraded: ROC-AUC fell from {baseline:.4f} "
                    f"to {live.roc_auc:.4f} ({drop:.4f} > tolerance {tolerance:.4f}) "
                    f"over {live.labelled_samples} labelled predictions"
                ),
                evidence={
                    "baseline_roc_auc": baseline,
                    "live_roc_auc": live.roc_auc,
                    "drop": drop,
                    "labelled_samples": live.labelled_samples,
                },
            ),
        )

    def _in_cooldown(self) -> tuple[bool, str | None]:
        minutes = self.settings.retraining.cooldown_minutes
        if minutes <= 0:
            return False, None
        row = self.db.query_one(
            "SELECT created_at FROM retraining_events WHERE created_at >= ? "
            "AND status != ? ORDER BY created_at DESC LIMIT 1",
            (iso_minutes_ago(minutes), RetrainingStatus.SKIPPED.value),
        )
        return (row is not None), (row["created_at"] if row else None)


# --------------------------------------------------------------------------- #
# Event store
# --------------------------------------------------------------------------- #
class RetrainingEventStore:
    """Persistence for retraining events."""

    def __init__(self, db: Database | None = None) -> None:
        self._db = db

    @property
    def db(self) -> Database:
        if self._db is None:
            self._db = get_database()
        return self._db

    def create(
        self,
        trigger: RetrainingTrigger,
        reason: str,
        model_name: str,
        baseline_version: int | None = None,
        detail: dict[str, Any] | None = None,
    ) -> RetrainingEvent:
        event = RetrainingEvent(
            trigger=trigger,
            reason=reason,
            model_name=model_name,
            baseline_version=baseline_version,
            detail=detail or {},
            status=RetrainingStatus.CREATED,
        )
        self.db.execute(
            "INSERT INTO retraining_events (id, trigger, reason, status, model_name, "
            "baseline_version, candidate_version, decision, detail, created_at, updated_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (
                event.id,
                event.trigger.value,
                event.reason,
                event.status.value,
                event.model_name,
                event.baseline_version,
                None,
                None,
                dumps(event.detail),
                event.created_at,
                event.updated_at,
            ),
        )
        logger.info(
            "retraining.event_created",
            extra={
                "event_id": event.id,
                "trigger": trigger.value,
                "reason": reason,
                "baseline_version": baseline_version,
            },
        )
        return event

    def update(self, event_id: str, **fields: Any) -> None:
        if not fields:
            return
        assignments: list[str] = []
        values: list[Any] = []
        for key, value in fields.items():
            if key == "detail":
                value = dumps(value)
            elif hasattr(value, "value"):
                value = value.value
            assignments.append(f"{key} = ?")
            values.append(value)
        assignments.append("updated_at = ?")
        values.append(utcnow_iso())
        values.append(event_id)
        self.db.execute(
            f"UPDATE retraining_events SET {', '.join(assignments)} WHERE id = ?", values
        )

    def get(self, event_id: str) -> RetrainingEvent | None:
        row = self.db.query_one("SELECT * FROM retraining_events WHERE id = ?", (event_id,))
        return _to_event(row) if row else None

    def recent(self, limit: int = 25) -> list[RetrainingEvent]:
        rows = self.db.query(
            "SELECT * FROM retraining_events ORDER BY created_at DESC LIMIT ?", (limit,)
        )
        return [_to_event(row) for row in rows]


def _to_event(row) -> RetrainingEvent:
    data = dict(row)
    data["detail"] = loads(data.get("detail"), {})
    data["trigger"] = RetrainingTrigger(data["trigger"])
    data["status"] = RetrainingStatus(data["status"])
    return RetrainingEvent(**data)


_STORE: RetrainingEventStore | None = None


def get_event_store() -> RetrainingEventStore:
    global _STORE
    if _STORE is None:
        _STORE = RetrainingEventStore()
    return _STORE


def evaluate_trigger(force: bool = False) -> TriggerDecision:
    return RetrainingTriggerEvaluator().evaluate(force)
