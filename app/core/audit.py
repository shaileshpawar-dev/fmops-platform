"""Audit log.

Every state-changing action -- stage transitions, deployments, rollbacks,
retraining decisions, configuration overrides -- writes one immutable audit
record.  Audit records are written to the operational database *and* emitted as
structured log lines so they land in CloudWatch/Loki even if the database is
lost.
"""

from __future__ import annotations

from typing import Any, Literal

from app.core.config import get_settings
from app.core.db import Database, dumps, get_database
from app.core.logging import get_context, get_logger
from app.core.utils import utcnow_iso

logger = get_logger("fmops.audit")

Outcome = Literal["success", "failure", "denied"]


class AuditLog:
    """Append-only record of who changed what."""

    def __init__(self, db: Database | None = None) -> None:
        self._db = db

    @property
    def db(self) -> Database:
        if self._db is None:
            self._db = get_database()
        return self._db

    def record(
        self,
        action: str,
        resource_type: str,
        resource_id: str | None = None,
        outcome: Outcome = "success",
        actor: str | None = None,
        detail: dict[str, Any] | None = None,
        source_ip: str | None = None,
    ) -> None:
        settings = get_settings()
        if not settings.security.audit_log_enabled:
            return
        ctx = get_context()
        request_id = ctx.get("request_id")
        actor = actor or ctx.get("actor") or "system"
        payload = detail or {}
        created_at = utcnow_iso()
        try:
            self.db.execute(
                "INSERT INTO audit_log (actor, action, resource_type, resource_id, "
                "outcome, detail, request_id, source_ip, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    actor,
                    action,
                    resource_type,
                    resource_id,
                    outcome,
                    dumps(payload),
                    request_id,
                    source_ip,
                    created_at,
                ),
            )
        except Exception as exc:  # audit must never break the caller
            logger.error(
                "audit.persist_failed",
                extra={"action": action, "error": str(exc)},
            )
        logger.info(
            "audit",
            extra={
                "audit_action": action,
                "resource_type": resource_type,
                "resource_id": resource_id,
                "outcome": outcome,
                "actor": actor,
                "detail": payload,
            },
        )

    def recent(self, limit: int = 100, action: str | None = None) -> list[dict]:
        sql = "SELECT * FROM audit_log"
        params: list[Any] = []
        if action:
            sql += " WHERE action = ?"
            params.append(action)
        sql += " ORDER BY id DESC LIMIT ?"
        params.append(limit)
        from app.core.db import loads

        rows = self.db.query(sql, params)
        out = []
        for row in rows:
            item = dict(row)
            item["detail"] = loads(item.get("detail"), {})
            out.append(item)
        return out


_AUDIT: AuditLog | None = None


def get_audit_log() -> AuditLog:
    global _AUDIT
    if _AUDIT is None:
        _AUDIT = AuditLog()
    return _AUDIT


def audit(
    action: str,
    resource_type: str,
    resource_id: str | None = None,
    outcome: Outcome = "success",
    **detail: Any,
) -> None:
    """Convenience wrapper: ``audit("model.promote", "model", "loan/3", stage=...)``."""
    get_audit_log().record(
        action=action,
        resource_type=resource_type,
        resource_id=resource_id,
        outcome=outcome,
        detail=detail or None,
    )
