"""Persistence for jobs and their logs.

A job is the *execution* of a long-running operation; the thing it produces
(a training run, an AutoML run, a retraining event, a deployment) keeps its own
record and outcome. The split is deliberate: "the job failed" and "the model was
rejected by the gate" are different facts, and a job that succeeded can have
produced a rejected model.

States:

``queued``     waiting for a worker
``running``    claimed by a worker, heartbeating
``succeeded``  the operation ran to completion (whatever its domain outcome)
``failed``     the operation raised, or its worker was lost
``cancelled``  stopped on request, before completion
"""

from __future__ import annotations

import os
import socket
import uuid
from typing import Any

from app.core.db import Database, dumps, get_database, loads
from app.core.utils import utcnow_iso

QUEUED = "queued"
RUNNING = "running"
SUCCEEDED = "succeeded"
FAILED = "failed"
CANCELLED = "cancelled"
TERMINAL = frozenset({SUCCEEDED, FAILED, CANCELLED})

# Identifies this process's claims. The pid alone is not enough inside
# containers, where every replica is pid 1-ish.
WORKER_ID = f"{socket.gethostname()}:{os.getpid()}"


class JobStore:
    def __init__(self, db: Database | None = None) -> None:
        self._db = db

    @property
    def db(self) -> Database:
        return self._db or get_database()

    # -- writes ----------------------------------------------------------------- #
    def create(
        self,
        kind: str,
        payload: dict[str, Any],
        *,
        model_name: str | None = None,
        resource_id: str | None = None,
        requested_by: str | None = None,
        idempotency_key: str | None = None,
        retry_of: str | None = None,
    ) -> dict[str, Any]:
        """Queue a job, or return the live job that already holds the key.

        An idempotency key makes a double-clicked "Start training" produce one
        job, not two. A key whose job failed or was cancelled can be reused --
        otherwise a failure could never be retried under the same key.
        """
        if idempotency_key:
            existing = self.db.query_one(
                "SELECT * FROM jobs WHERE idempotency_key = ? AND status NOT IN (?, ?)",
                (idempotency_key, FAILED, CANCELLED),
            )
            if existing is not None:
                return _to_dict(existing)
            # Release the key held by a failed/cancelled job so it can be reused.
            self.db.execute(
                "UPDATE jobs SET idempotency_key = NULL WHERE idempotency_key = ?",
                (idempotency_key,),
            )
        job_id = f"job-{uuid.uuid4().hex[:16]}"
        self.db.execute(
            "INSERT INTO jobs (id, kind, status, model_name, resource_id, payload, "
            "requested_by, idempotency_key, retry_of, created_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?)",
            (
                job_id,
                kind,
                QUEUED,
                model_name,
                resource_id,
                dumps(payload),
                requested_by,
                idempotency_key,
                retry_of,
                utcnow_iso(),
            ),
        )
        return self.get(job_id)

    def claim_next(self, max_running: int) -> dict[str, Any] | None:
        """Atomically take the oldest queued job, if the global cap allows.

        ``BEGIN IMMEDIATE`` takes SQLite's write lock before reading, so two
        worker processes can never claim the same job, and the running-count
        check and the claim happen as one step.
        """
        now = utcnow_iso()
        with self.db.transaction() as conn:
            running = conn.execute(
                "SELECT COUNT(*) FROM jobs WHERE status = ?", (RUNNING,)
            ).fetchone()[0]
            if running >= max_running:
                return None
            row = conn.execute(
                "SELECT id FROM jobs WHERE status = ? ORDER BY created_at, id LIMIT 1",
                (QUEUED,),
            ).fetchone()
            if row is None:
                return None
            conn.execute(
                "UPDATE jobs SET status = ?, worker = ?, started_at = ?, heartbeat_at = ?, "
                "attempts = attempts + 1 WHERE id = ? AND status = ?",
                (RUNNING, WORKER_ID, now, now, row["id"], QUEUED),
            )
        return self.get(row["id"])

    def finish(
        self,
        job_id: str,
        status: str,
        *,
        result: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> None:
        self.db.execute(
            "UPDATE jobs SET status = ?, result = ?, error = ?, finished_at = ? "
            "WHERE id = ? AND status NOT IN (?, ?, ?)",
            (
                status,
                dumps(result) if result is not None else None,
                error,
                utcnow_iso(),
                job_id,
                SUCCEEDED,
                FAILED,
                CANCELLED,
            ),
        )

    def set_resource(self, job_id: str, resource_id: str) -> None:
        self.db.execute("UPDATE jobs SET resource_id = ? WHERE id = ?", (resource_id, job_id))

    def heartbeat(self, worker: str) -> None:
        self.db.execute(
            "UPDATE jobs SET heartbeat_at = ? WHERE worker = ? AND status = ?",
            (utcnow_iso(), worker, RUNNING),
        )

    def request_cancel(self, job_id: str) -> dict[str, Any]:
        """Cancel a queued job now; flag a running one for its next checkpoint."""
        with self.db.transaction() as conn:
            conn.execute(
                "UPDATE jobs SET status = ?, finished_at = ?, error = ? "
                "WHERE id = ? AND status = ?",
                (CANCELLED, utcnow_iso(), "cancelled before it started", job_id, QUEUED),
            )
            conn.execute(
                "UPDATE jobs SET cancel_requested = 1 WHERE id = ? AND status = ?",
                (job_id, RUNNING),
            )
        return self.get(job_id)

    def cancel_requested(self, job_id: str) -> bool:
        return bool(
            self.db.scalar("SELECT cancel_requested FROM jobs WHERE id = ?", (job_id,), 0)
        )

    def stale(self, older_than_iso: str) -> list[dict[str, Any]]:
        rows = self.db.query(
            "SELECT * FROM jobs WHERE status = ? AND COALESCE(heartbeat_at, started_at) < ?",
            (RUNNING, older_than_iso),
        )
        return [_to_dict(r) for r in rows]

    def overrunning(self, started_before_iso: str) -> list[dict[str, Any]]:
        rows = self.db.query(
            "SELECT * FROM jobs WHERE status = ? AND started_at < ? AND cancel_requested = 0",
            (RUNNING, started_before_iso),
        )
        return [_to_dict(r) for r in rows]

    # -- logs ---------------------------------------------------------------------- #
    def append_log(
        self, job_id: str, level: str, message: str, context: dict[str, Any], cap: int
    ) -> None:
        count = int(
            self.db.scalar("SELECT COUNT(*) FROM job_logs WHERE job_id = ?", (job_id,), 0)
        )
        if count > cap:
            return
        if count == cap:
            level, message, context = "WARNING", f"log capped at {cap} lines", {}
        self.db.execute(
            "INSERT INTO job_logs (job_id, level, message, context, created_at) "
            "VALUES (?,?,?,?,?)",
            (job_id, level, message, dumps(context), utcnow_iso()),
        )

    def logs(self, job_id: str, after_id: int = 0, limit: int = 500) -> list[dict[str, Any]]:
        rows = self.db.query(
            "SELECT id, level, message, context, created_at FROM job_logs "
            "WHERE job_id = ? AND id > ? ORDER BY id LIMIT ?",
            (job_id, int(after_id), int(limit)),
        )
        out = []
        for row in rows:
            item = dict(row)
            item["context"] = loads(item.get("context"), {})
            out.append(item)
        return out

    # -- reads ----------------------------------------------------------------------- #
    def get(self, job_id: str) -> dict[str, Any] | None:
        row = self.db.query_one("SELECT * FROM jobs WHERE id = ?", (job_id,))
        return _to_dict(row) if row else None

    def for_resource(self, resource_id: str) -> dict[str, Any] | None:
        row = self.db.query_one(
            "SELECT * FROM jobs WHERE resource_id = ? ORDER BY created_at DESC LIMIT 1",
            (resource_id,),
        )
        return _to_dict(row) if row else None

    def list(
        self,
        limit: int = 50,
        status: str | None = None,
        kind: str | None = None,
        model_name: str | None = None,
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        for column, value in (("status", status), ("kind", kind), ("model_name", model_name)):
            if value:
                clauses.append(f"{column} = ?")
                params.append(value)
        sql = "SELECT * FROM jobs"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY created_at DESC LIMIT ?"
        params.append(int(limit))
        return [_to_dict(r) for r in self.db.query(sql, params)]

    def counts(self) -> dict[str, int]:
        rows = self.db.query("SELECT status, COUNT(*) AS n FROM jobs GROUP BY status")
        out = dict.fromkeys((QUEUED, RUNNING, SUCCEEDED, FAILED, CANCELLED), 0)
        out.update({row["status"]: int(row["n"]) for row in rows})
        return out


def _to_dict(row: Any) -> dict[str, Any]:
    data = dict(row)
    data["payload"] = loads(data.get("payload"), {})
    data["result"] = loads(data.get("result"), None) if data.get("result") else None
    data["cancel_requested"] = bool(data.get("cancel_requested"))
    return data


_STORE = JobStore()


def get_job_store() -> JobStore:
    return _STORE
