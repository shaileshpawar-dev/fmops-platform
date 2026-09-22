"""Gate decisions: why a version was promoted, held or rejected.

The approval gate used to leave its verdict only as side effects -- a status
string, a tag, the reason on a stage transition. That answers "what stage is it
in" but not "why": which checks ran, what each one observed against which
threshold, what the champion scored, and who signed off.

Every decision is now one immutable row: the automated verdict from a training
or retraining run, and every human approval or rejection after it. The rows are
never updated, so a version's history reads top to bottom as it happened.
"""

from __future__ import annotations

from typing import Any, Literal

from app.core.db import Database, dumps, get_database, loads
from app.core.utils import utcnow_iso
from app.schemas.evaluation import ApprovalResult, ModelComparison

Source = Literal["pipeline", "manual"]


class GateDecisionStore:
    def __init__(self, db: Database | None = None) -> None:
        self._db = db

    @property
    def db(self) -> Database:
        return self._db or get_database()

    def record(
        self,
        *,
        model_name: str,
        model_version: int,
        source: Source,
        decision: str,
        reason: str,
        approval: ApprovalResult | None = None,
        comparison: ModelComparison | None = None,
        thresholds: dict[str, Any] | None = None,
        target_stage: str | None = None,
        final_stage: str | None = None,
        actor: str | None = None,
        comment: str | None = None,
    ) -> dict[str, Any]:
        checks = [c.model_dump(mode="json") for c in approval.checks] if approval else []
        cursor = self.db.execute(
            "INSERT INTO gate_decisions (model_name, model_version, source, decision, "
            "target_stage, final_stage, reason, checks, comparison, thresholds, actor, "
            "comment, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                model_name,
                int(model_version),
                source,
                decision,
                target_stage,
                final_stage,
                reason,
                dumps(checks),
                dumps(comparison.model_dump(mode="json")) if comparison else None,
                dumps(thresholds or {}),
                actor,
                comment,
                utcnow_iso(),
            ),
        )
        return self.get(int(cursor.lastrowid or 0))

    def get(self, decision_id: int) -> dict[str, Any]:
        row = self.db.query_one("SELECT * FROM gate_decisions WHERE id = ?", (decision_id,))
        return _to_dict(row)

    def latest(self, model_name: str, model_version: int) -> dict[str, Any] | None:
        row = self.db.query_one(
            "SELECT * FROM gate_decisions WHERE model_name = ? AND model_version = ? "
            "ORDER BY id DESC LIMIT 1",
            (model_name, int(model_version)),
        )
        return _to_dict(row) if row else None

    def latest_automated(self, model_name: str, model_version: int) -> dict[str, Any] | None:
        row = self.db.query_one(
            "SELECT * FROM gate_decisions WHERE model_name = ? AND model_version = ? "
            "AND source = 'pipeline' ORDER BY id DESC LIMIT 1",
            (model_name, int(model_version)),
        )
        return _to_dict(row) if row else None

    def list(
        self, model_name: str | None = None, model_version: int | None = None, limit: int = 100
    ) -> list[dict[str, Any]]:
        sql = "SELECT * FROM gate_decisions"
        clauses: list[str] = []
        params: list[Any] = []
        if model_name:
            clauses.append("model_name = ?")
            params.append(model_name)
        if model_version is not None:
            clauses.append("model_version = ?")
            params.append(int(model_version))
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY id DESC LIMIT ?"
        params.append(int(limit))
        return [_to_dict(r) for r in self.db.query(sql, params)]


def _to_dict(row: Any) -> dict[str, Any]:
    data = dict(row)
    data["checks"] = loads(data.get("checks"), [])
    data["comparison"] = (
        loads(data.get("comparison"), None) if data.get("comparison") else None
    )
    data["thresholds"] = loads(data.get("thresholds"), {})
    return data


_STORE = GateDecisionStore()


def get_gate_decisions() -> GateDecisionStore:
    return _STORE
