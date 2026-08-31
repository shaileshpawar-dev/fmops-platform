"""Alerting.

Alerts are raised by the drift scanner, the health watchdog, the retraining
trigger, the cost tracker and the deployment manager. Each alert is fanned out
to the configured sinks.

De-duplication: an alert's ``fingerprint`` is derived from its category, title
and the identity fields in its context. An identical fingerprint seen inside
``dedupe_window_seconds`` is suppressed, so a drift scan running every five
minutes does not produce 288 identical pages a day. Suppression is logged at
debug level so it is never silent.

Sink failures never propagate: a webhook being down must not break the drift
scan that raised the alert. Failures are logged as errors and the alert is still
persisted locally.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from app.core.config import Settings, get_settings
from app.core.db import Database, dumps, get_database, loads
from app.core.logging import get_logger
from app.core.utils import fingerprint, jsonable, write_json
from app.monitoring.metrics import record_alert
from app.schemas.common import AlertCategory, Severity
from app.schemas.evaluation import Alert

logger = get_logger(__name__)


class AlertSink(ABC):
    """Somewhere an alert is delivered."""

    name: str = "abstract"

    @abstractmethod
    def emit(self, alert: Alert) -> None: ...


class LogSink(AlertSink):
    """Writes the alert as a structured log line."""

    name = "log"

    _LEVELS = {
        Severity.INFO: logger.info,
        Severity.WARNING: logger.warning,
        Severity.CRITICAL: logger.error,
    }

    def emit(self, alert: Alert) -> None:
        log = self._LEVELS.get(alert.severity, logger.info)
        log(
            "alert",
            extra={
                "alert_id": alert.id,
                "severity": alert.severity.value,
                "category": alert.category.value,
                "title": alert.title,
                "alert_message": alert.message,
                "context": jsonable(alert.context),
            },
        )


class DatabaseSink(AlertSink):
    """Persists the alert so the API and dashboard can show it."""

    name = "database"

    def __init__(self, db: Database | None = None) -> None:
        self._db = db

    @property
    def db(self) -> Database:
        if self._db is None:
            self._db = get_database()
        return self._db

    def emit(self, alert: Alert) -> None:
        self.db.execute(
            "INSERT INTO alerts (id, severity, category, title, message, fingerprint, "
            "context, acknowledged, created_at) VALUES (?,?,?,?,?,?,?,?,?)",
            (
                alert.id,
                alert.severity.value,
                alert.category.value,
                alert.title,
                alert.message,
                alert.fingerprint,
                dumps(alert.context),
                1 if alert.acknowledged else 0,
                alert.created_at,
            ),
        )


class FileSink(AlertSink):
    """Appends the alert to a JSON file under ``artifacts/reports/alerts``."""

    name = "file"

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def emit(self, alert: Alert) -> None:
        directory = self.settings.paths.reports_dir / "alerts"
        directory.mkdir(parents=True, exist_ok=True)
        write_json(directory / f"{alert.id}.json", alert.model_dump(mode="json"))


class WebhookSink(AlertSink):
    """POSTs the alert to a webhook (Slack-compatible payload shape)."""

    name = "webhook"

    def __init__(self, url: str, timeout: float = 5.0) -> None:
        self.url = url
        self.timeout = timeout

    def emit(self, alert: Alert) -> None:
        import httpx

        payload = {
            "text": f"[{alert.severity.value.upper()}] {alert.title}",
            "attachments": [
                {
                    "title": alert.title,
                    "text": alert.message,
                    "fields": [
                        {"title": k, "value": str(v), "short": True}
                        for k, v in list(jsonable(alert.context).items())[:10]
                    ],
                }
            ],
            "alert": alert.model_dump(mode="json"),
        }
        response = httpx.post(self.url, json=payload, timeout=self.timeout)
        response.raise_for_status()


class SNSSink(AlertSink):
    """Publishes the alert to an Amazon SNS topic."""

    name = "sns"

    def __init__(self, topic_arn: str, region: str | None = None) -> None:
        self.topic_arn = topic_arn
        self.region = region
        self._client = None

    @property
    def client(self):
        if self._client is None:
            import boto3

            self._client = boto3.client("sns", region_name=self.region)
        return self._client

    def emit(self, alert: Alert) -> None:
        import json

        self.client.publish(
            TopicArn=self.topic_arn,
            Subject=f"[{alert.severity.value.upper()}] {alert.title}"[:100],
            Message=json.dumps(alert.model_dump(mode="json"), default=str),
            MessageAttributes={
                "severity": {"DataType": "String", "StringValue": alert.severity.value},
                "category": {"DataType": "String", "StringValue": alert.category.value},
            },
        )


class AlertManager:
    """Builds, de-duplicates and fans out alerts."""

    def __init__(
        self,
        sinks: list[AlertSink] | None = None,
        settings: Settings | None = None,
        db: Database | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self._db = db
        self.sinks = sinks if sinks is not None else self._build_sinks()

    @property
    def db(self) -> Database:
        if self._db is None:
            self._db = get_database()
        return self._db

    def _build_sinks(self) -> list[AlertSink]:
        config = self.settings.alerts
        sinks: list[AlertSink] = []
        for name in config.sinks:
            try:
                if name == "log":
                    sinks.append(LogSink())
                elif name == "database":
                    sinks.append(DatabaseSink())
                elif name == "file":
                    sinks.append(FileSink(self.settings))
                elif name == "webhook":
                    if not config.webhook_url:
                        logger.warning(
                            "alerts.webhook_sink_skipped",
                            extra={"reason": "FMOPS_ALERTS__WEBHOOK_URL is not set"},
                        )
                        continue
                    sinks.append(WebhookSink(config.webhook_url))
                elif name == "sns":
                    if not config.sns_topic_arn:
                        logger.warning(
                            "alerts.sns_sink_skipped",
                            extra={"reason": "FMOPS_ALERTS__SNS_TOPIC_ARN is not set"},
                        )
                        continue
                    sinks.append(SNSSink(config.sns_topic_arn, self.settings.aws.region))
            except Exception as exc:
                logger.error(
                    "alerts.sink_construction_failed",
                    extra={"sink": name, "error": str(exc)},
                )
        return sinks

    # -- raising ------------------------------------------------------------- #
    def raise_alert(
        self,
        severity: Severity,
        category: AlertCategory,
        title: str,
        message: str,
        context: dict[str, Any] | None = None,
        dedupe_keys: tuple[str, ...] = (),
    ) -> Alert | None:
        """Emit an alert. Returns None when it was suppressed as a duplicate."""
        if not self.settings.alerts.enabled:
            return None

        context = context or {}
        keys = [str(context.get(k, "")) for k in dedupe_keys]
        alert = Alert(
            severity=severity,
            category=category,
            title=title,
            message=message,
            fingerprint=fingerprint(category.value, title, *keys),
            context=jsonable(context),
        )

        if self._is_duplicate(alert.fingerprint):
            logger.debug(
                "alerts.suppressed_duplicate",
                extra={
                    "fingerprint": alert.fingerprint,
                    "title": title,
                    "window_seconds": self.settings.alerts.dedupe_window_seconds,
                },
            )
            return None

        for sink in self.sinks:
            try:
                sink.emit(alert)
            except Exception as exc:
                # A failing sink must not break the caller that raised the alert.
                logger.error(
                    "alerts.sink_failed",
                    extra={"sink": sink.name, "alert_id": alert.id, "error": str(exc)},
                )
        record_alert(severity.value, category.value)
        return alert

    def _is_duplicate(self, alert_fingerprint: str) -> bool:
        window = self.settings.alerts.dedupe_window_seconds
        if window <= 0:
            return False
        from app.core.utils import iso_minutes_ago

        since = iso_minutes_ago(max(1, round(window / 60)))
        try:
            count = int(
                self.db.scalar(
                    "SELECT COUNT(*) FROM alerts WHERE fingerprint = ? AND created_at >= ?",
                    (alert_fingerprint, since),
                    0,
                )
            )
        except Exception:
            return False
        return count > 0

    # -- convenience wrappers ------------------------------------------------ #
    def info(self, category: AlertCategory, title: str, message: str, **context: Any):
        return self.raise_alert(Severity.INFO, category, title, message, context)

    def warning(self, category: AlertCategory, title: str, message: str, **context: Any):
        return self.raise_alert(Severity.WARNING, category, title, message, context)

    def critical(self, category: AlertCategory, title: str, message: str, **context: Any):
        return self.raise_alert(Severity.CRITICAL, category, title, message, context)

    # -- reads --------------------------------------------------------------- #
    def recent(
        self,
        limit: int = 50,
        severity: Severity | None = None,
        category: AlertCategory | None = None,
        unacknowledged_only: bool = False,
    ) -> list[Alert]:
        sql = "SELECT * FROM alerts"
        clauses: list[str] = []
        params: list[Any] = []
        if severity:
            clauses.append("severity = ?")
            params.append(severity.value)
        if category:
            clauses.append("category = ?")
            params.append(category.value)
        if unacknowledged_only:
            clauses.append("acknowledged = 0")
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        out: list[Alert] = []
        for row in self.db.query(sql, params):
            data = dict(row)
            out.append(
                Alert(
                    id=data["id"],
                    severity=Severity(data["severity"]),
                    category=AlertCategory(data["category"]),
                    title=data["title"],
                    message=data["message"],
                    fingerprint=data["fingerprint"],
                    context=loads(data["context"], {}),
                    acknowledged=bool(data["acknowledged"]),
                    created_at=data["created_at"],
                )
            )
        return out

    def open_count(self) -> int:
        return int(
            self.db.scalar(
                "SELECT COUNT(*) FROM alerts WHERE acknowledged = 0 AND severity != ?",
                (Severity.INFO.value,),
                0,
            )
        )

    def acknowledge(self, alert_id: str) -> bool:
        cursor = self.db.execute(
            "UPDATE alerts SET acknowledged = 1 WHERE id = ?", (alert_id,)
        )
        acknowledged = bool(cursor.rowcount)
        if acknowledged:
            logger.info("alerts.acknowledged", extra={"alert_id": alert_id})
        return acknowledged

    def acknowledge_all(self) -> int:
        cursor = self.db.execute("UPDATE alerts SET acknowledged = 1 WHERE acknowledged = 0")
        return cursor.rowcount if cursor.rowcount and cursor.rowcount > 0 else 0


_MANAGER: AlertManager | None = None


def get_alert_manager() -> AlertManager:
    global _MANAGER
    if _MANAGER is None:
        _MANAGER = AlertManager()
    return _MANAGER


def set_alert_manager(manager: AlertManager | None) -> None:
    global _MANAGER
    _MANAGER = manager
