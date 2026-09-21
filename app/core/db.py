"""SQLite-backed operational state store.

The platform keeps its *operational* state (inference logs, deployments, drift
reports, alerts, retraining events, LLM traces, cost records, audit log) in a
small relational store.  SQLite is used because it needs no server, works
identically on a laptop and inside a container with a mounted volume, and is
part of the standard library.

Every table is reached through a repository class in the owning module -- no
module outside this file writes raw SQL against another module's table.  To move
to Postgres/RDS in a real deployment you replace :class:`Database` with a
psycopg-backed equivalent exposing the same three methods; the schema is plain
ANSI-ish SQL.  See ``docs/deployment.md``.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from app.core.logging import get_logger

logger = get_logger(__name__)

SCHEMA_VERSION = 2

SCHEMA_STATEMENTS: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS schema_meta (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL
    )
    """,
    # ---------------- model registry (local backend) ----------------------- #
    """
    CREATE TABLE IF NOT EXISTS model_versions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        version INTEGER NOT NULL,
        stage TEXT NOT NULL,
        status TEXT NOT NULL,
        run_id TEXT,
        artifact_uri TEXT NOT NULL,
        dataset_version TEXT,
        dataset_hash TEXT,
        git_commit TEXT,
        algorithm TEXT,
        params TEXT NOT NULL DEFAULT '{}',
        metrics TEXT NOT NULL DEFAULT '{}',
        tags TEXT NOT NULL DEFAULT '{}',
        description TEXT,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        created_by TEXT,
        UNIQUE (name, version)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS model_stage_transitions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        version INTEGER NOT NULL,
        from_stage TEXT,
        to_stage TEXT NOT NULL,
        reason TEXT,
        actor TEXT,
        created_at TEXT NOT NULL
    )
    """,
    # ---------------- deployments ------------------------------------------ #
    """
    CREATE TABLE IF NOT EXISTS deployments (
        id TEXT PRIMARY KEY,
        endpoint_name TEXT NOT NULL,
        provider TEXT NOT NULL,
        strategy TEXT NOT NULL,
        state TEXT NOT NULL,
        model_name TEXT NOT NULL,
        current_version INTEGER,
        previous_version INTEGER,
        candidate_version INTEGER,
        traffic TEXT NOT NULL DEFAULT '{}',
        shadow_version INTEGER,
        health TEXT NOT NULL DEFAULT 'unknown',
        message TEXT,
        metadata TEXT NOT NULL DEFAULT '{}',
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS deployment_events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        deployment_id TEXT NOT NULL,
        event TEXT NOT NULL,
        detail TEXT NOT NULL DEFAULT '{}',
        created_at TEXT NOT NULL
    )
    """,
    # ---------------- inference ------------------------------------------- #
    """
    CREATE TABLE IF NOT EXISTS inference_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        request_id TEXT NOT NULL,
        model_name TEXT NOT NULL,
        model_version INTEGER,
        deployment_id TEXT,
        variant TEXT,
        shadow INTEGER NOT NULL DEFAULT 0,
        features TEXT NOT NULL,
        prediction INTEGER,
        probability REAL,
        latency_ms REAL NOT NULL,
        status TEXT NOT NULL DEFAULT 'ok',
        error_code TEXT,
        created_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS feedback (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        request_id TEXT NOT NULL,
        actual_label INTEGER NOT NULL,
        source TEXT,
        created_at TEXT NOT NULL
    )
    """,
    # ---------------- drift ------------------------------------------------ #
    """
    CREATE TABLE IF NOT EXISTS drift_reports (
        id TEXT PRIMARY KEY,
        model_name TEXT NOT NULL,
        model_version INTEGER,
        engine TEXT NOT NULL,
        drift_detected INTEGER NOT NULL,
        dataset_drift_score REAL NOT NULL,
        prediction_drift_score REAL,
        concept_drift_score REAL,
        concept_drift_status TEXT NOT NULL DEFAULT 'unavailable',
        drifted_features TEXT NOT NULL DEFAULT '[]',
        n_reference INTEGER NOT NULL,
        n_current INTEGER NOT NULL,
        report TEXT NOT NULL DEFAULT '{}',
        created_at TEXT NOT NULL
    )
    """,
    # ---------------- alerts ----------------------------------------------- #
    """
    CREATE TABLE IF NOT EXISTS alerts (
        id TEXT PRIMARY KEY,
        severity TEXT NOT NULL,
        category TEXT NOT NULL,
        title TEXT NOT NULL,
        message TEXT NOT NULL,
        fingerprint TEXT NOT NULL,
        context TEXT NOT NULL DEFAULT '{}',
        acknowledged INTEGER NOT NULL DEFAULT 0,
        created_at TEXT NOT NULL
    )
    """,
    # ---------------- retraining ------------------------------------------- #
    """
    CREATE TABLE IF NOT EXISTS retraining_events (
        id TEXT PRIMARY KEY,
        trigger TEXT NOT NULL,
        reason TEXT NOT NULL,
        status TEXT NOT NULL,
        model_name TEXT NOT NULL,
        baseline_version INTEGER,
        candidate_version INTEGER,
        decision TEXT,
        detail TEXT NOT NULL DEFAULT '{}',
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )
    """,
    # ---------------- LLMOps ----------------------------------------------- #
    """
    CREATE TABLE IF NOT EXISTS llm_traces (
        id TEXT PRIMARY KEY,
        request_id TEXT NOT NULL,
        provider TEXT NOT NULL,
        model TEXT NOT NULL,
        prompt_name TEXT,
        prompt_version TEXT,
        temperature REAL,
        max_tokens INTEGER,
        input_tokens INTEGER NOT NULL DEFAULT 0,
        output_tokens INTEGER NOT NULL DEFAULT 0,
        total_tokens INTEGER NOT NULL DEFAULT 0,
        latency_ms REAL NOT NULL DEFAULT 0,
        estimated_cost_usd REAL NOT NULL DEFAULT 0,
        status TEXT NOT NULL DEFAULT 'ok',
        error_code TEXT,
        safety_verdict TEXT,
        rendered_prompt TEXT,
        output_text TEXT,
        metadata TEXT NOT NULL DEFAULT '{}',
        git_commit TEXT,
        created_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS llm_evaluations (
        id TEXT PRIMARY KEY,
        suite TEXT NOT NULL,
        dataset TEXT NOT NULL,
        provider TEXT NOT NULL,
        model TEXT NOT NULL,
        prompt_name TEXT NOT NULL,
        prompt_version TEXT NOT NULL,
        n_cases INTEGER NOT NULL,
        aggregate TEXT NOT NULL DEFAULT '{}',
        per_case TEXT NOT NULL DEFAULT '[]',
        total_tokens INTEGER NOT NULL DEFAULT 0,
        total_cost_usd REAL NOT NULL DEFAULT 0,
        created_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS llm_registry (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        version INTEGER NOT NULL,
        stage TEXT NOT NULL,
        provider TEXT NOT NULL,
        model TEXT NOT NULL,
        prompt_name TEXT NOT NULL,
        prompt_version TEXT NOT NULL,
        params TEXT NOT NULL DEFAULT '{}',
        evaluation_id TEXT,
        metrics TEXT NOT NULL DEFAULT '{}',
        git_commit TEXT,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        UNIQUE (name, version)
    )
    """,
    # ---------------- audit ------------------------------------------------ #
    """
    CREATE TABLE IF NOT EXISTS audit_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        actor TEXT,
        action TEXT NOT NULL,
        resource_type TEXT NOT NULL,
        resource_id TEXT,
        outcome TEXT NOT NULL,
        detail TEXT NOT NULL DEFAULT '{}',
        request_id TEXT,
        source_ip TEXT,
        created_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS training_runs (
        id TEXT PRIMARY KEY,
        status TEXT NOT NULL,
        dataset_version TEXT,
        algorithm TEXT,
        tune INTEGER NOT NULL DEFAULT 0,
        promote INTEGER NOT NULL DEFAULT 0,
        target_stage TEXT,
        model_name TEXT,
        model_version INTEGER,
        exit_code INTEGER,
        error TEXT,
        report TEXT NOT NULL DEFAULT '{}',
        requested_by TEXT,
        created_at TEXT NOT NULL,
        started_at TEXT,
        completed_at TEXT,
        updated_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS automl_runs (
        id TEXT PRIMARY KEY,
        status TEXT NOT NULL,
        dataset_version TEXT NOT NULL,
        target_column TEXT NOT NULL,
        problem_type TEXT NOT NULL,
        algorithms TEXT NOT NULL DEFAULT '[]',
        primary_metric TEXT NOT NULL,
        tune INTEGER NOT NULL DEFAULT 0,
        target_stage TEXT,
        max_models INTEGER NOT NULL DEFAULT 3,
        profile TEXT NOT NULL DEFAULT '{}',
        candidates TEXT NOT NULL DEFAULT '[]',
        best_algorithm TEXT,
        best_model_version INTEGER,
        promotion TEXT,
        ranking_rule TEXT,
        error TEXT,
        duration_seconds REAL,
        created_at TEXT NOT NULL,
        started_at TEXT,
        completed_at TEXT,
        updated_at TEXT NOT NULL
    )
    """,
    # ---------------- gate decisions --------------------------------------- #
    # Immutable: one row per automated verdict and per human sign-off.
    """
    CREATE TABLE IF NOT EXISTS gate_decisions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        model_name TEXT NOT NULL,
        model_version INTEGER NOT NULL,
        source TEXT NOT NULL,
        decision TEXT NOT NULL,
        target_stage TEXT,
        final_stage TEXT,
        reason TEXT NOT NULL DEFAULT '',
        checks TEXT NOT NULL DEFAULT '[]',
        comparison TEXT,
        thresholds TEXT NOT NULL DEFAULT '{}',
        actor TEXT,
        comment TEXT,
        created_at TEXT NOT NULL
    )
    """,
    # ---------------- jobs -------------------------------------------------- #
    # The execution of long operations. The operation's own record (training
    # run, AutoML run, retraining event, deployment) is linked by resource_id.
    """
    CREATE TABLE IF NOT EXISTS jobs (
        id TEXT PRIMARY KEY,
        kind TEXT NOT NULL,
        status TEXT NOT NULL,
        model_name TEXT,
        resource_id TEXT,
        payload TEXT NOT NULL DEFAULT '{}',
        result TEXT,
        error TEXT,
        attempts INTEGER NOT NULL DEFAULT 0,
        cancel_requested INTEGER NOT NULL DEFAULT 0,
        requested_by TEXT,
        idempotency_key TEXT,
        retry_of TEXT,
        worker TEXT,
        heartbeat_at TEXT,
        created_at TEXT NOT NULL,
        started_at TEXT,
        finished_at TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS job_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        job_id TEXT NOT NULL,
        level TEXT NOT NULL,
        message TEXT NOT NULL,
        context TEXT NOT NULL DEFAULT '{}',
        created_at TEXT NOT NULL
    )
    """,
    # ---------------- indexes --------------------------------------------- #
    "CREATE INDEX IF NOT EXISTS ix_inference_created ON inference_log (created_at)",
    "CREATE INDEX IF NOT EXISTS ix_inference_request ON inference_log (request_id)",
    "CREATE INDEX IF NOT EXISTS ix_inference_model ON inference_log (model_name, model_version)",
    "CREATE INDEX IF NOT EXISTS ix_feedback_request ON feedback (request_id)",
    "CREATE INDEX IF NOT EXISTS ix_mv_name_stage ON model_versions (name, stage)",
    "CREATE INDEX IF NOT EXISTS ix_drift_created ON drift_reports (created_at)",
    "CREATE INDEX IF NOT EXISTS ix_alerts_created ON alerts (created_at)",
    "CREATE INDEX IF NOT EXISTS ix_alerts_fingerprint ON alerts (fingerprint, created_at)",
    "CREATE INDEX IF NOT EXISTS ix_llm_traces_created ON llm_traces (created_at)",
    "CREATE INDEX IF NOT EXISTS ix_llm_traces_model ON llm_traces (model, created_at)",
    "CREATE INDEX IF NOT EXISTS ix_audit_created ON audit_log (created_at)",
    "CREATE INDEX IF NOT EXISTS ix_training_runs_created ON training_runs (created_at)",
    "CREATE INDEX IF NOT EXISTS ix_automl_runs_created ON automl_runs (created_at)",
    "CREATE INDEX IF NOT EXISTS ix_deploy_endpoint ON deployments (endpoint_name, updated_at)",
    "CREATE INDEX IF NOT EXISTS ix_gate_model ON gate_decisions (model_name, model_version, id)",
    "CREATE INDEX IF NOT EXISTS ix_jobs_status ON jobs (status, created_at)",
    "CREATE INDEX IF NOT EXISTS ix_jobs_resource ON jobs (resource_id)",
    "CREATE UNIQUE INDEX IF NOT EXISTS ix_jobs_idempotency ON jobs (idempotency_key) "
    "WHERE idempotency_key IS NOT NULL",
    "CREATE INDEX IF NOT EXISTS ix_job_logs ON job_logs (job_id, id)",
)


# Columns added after a table first shipped. ``CREATE TABLE IF NOT EXISTS`` never
# alters an existing table, so an upgraded deployment would otherwise keep the
# old shape forever. Each entry is applied only when the column is absent, which
# makes the list safe to run on every start, against any prior version.
COLUMN_MIGRATIONS: tuple[tuple[str, str, str], ...] = (
    # v2: the input contract each model version was trained against.
    ("model_versions", "signature", "TEXT"),
    # v2: AutoML runs produce a named model, with an explicit positive class.
    ("automl_runs", "model_name", "TEXT"),
    ("automl_runs", "positive_label", "TEXT"),
    ("training_runs", "target_column", "TEXT"),
    ("training_runs", "positive_label", "TEXT"),
)


class Database:
    """Thin, thread-safe SQLite wrapper with schema management."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._write_lock = threading.RLock()
        self._initialise()

    # -- connection --------------------------------------------------------- #
    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(
            self.path, timeout=30.0, isolation_level=None, check_same_thread=False
        )
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA busy_timeout=30000")
        return conn

    @property
    def connection(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = self._connect()
            self._local.conn = conn
        return conn

    def _initialise(self) -> None:
        with self._write_lock:
            conn = self.connection
            for statement in SCHEMA_STATEMENTS:
                conn.execute(statement)
            self._migrate_columns(conn)
            conn.execute(
                "INSERT INTO schema_meta (key, value) VALUES ('schema_version', ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (str(SCHEMA_VERSION),),
            )
        logger.debug(
            "database.ready", extra={"db_path": str(self.path), "schema": SCHEMA_VERSION}
        )

    @staticmethod
    def _migrate_columns(conn: sqlite3.Connection) -> None:
        for table, column, ddl in COLUMN_MIGRATIONS:
            existing = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
            if column not in existing:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}")
                logger.info("database.column_added", extra={"table": table, "column": column})

    # -- operations --------------------------------------------------------- #
    def execute(self, sql: str, params: Sequence[Any] = ()) -> sqlite3.Cursor:
        with self._write_lock:
            return self.connection.execute(sql, tuple(params))

    def executemany(self, sql: str, rows: Sequence[Sequence[Any]]) -> sqlite3.Cursor:
        with self._write_lock:
            return self.connection.executemany(sql, [tuple(r) for r in rows])

    def query(self, sql: str, params: Sequence[Any] = ()) -> list[sqlite3.Row]:
        return list(self.connection.execute(sql, tuple(params)).fetchall())

    def query_one(self, sql: str, params: Sequence[Any] = ()) -> sqlite3.Row | None:
        return self.connection.execute(sql, tuple(params)).fetchone()

    def scalar(self, sql: str, params: Sequence[Any] = (), default: Any = None) -> Any:
        row = self.query_one(sql, params)
        if row is None:
            return default
        value = row[0]
        return default if value is None else value

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        with self._write_lock:
            conn = self.connection
            conn.execute("BEGIN IMMEDIATE")
            try:
                yield conn
            except Exception:
                conn.execute("ROLLBACK")
                raise
            else:
                conn.execute("COMMIT")

    def close(self) -> None:
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None


# --------------------------------------------------------------------------- #
# Module-level accessor
# --------------------------------------------------------------------------- #
_DB: Database | None = None
_DB_LOCK = threading.Lock()


def get_database(path: Path | str | None = None) -> Database:
    """Return the process-wide database, creating it on first use."""
    global _DB
    if path is not None:
        return Database(path)
    if _DB is None:
        with _DB_LOCK:
            if _DB is None:
                from app.core.config import get_settings

                settings = get_settings()
                settings.paths.ensure()
                _DB = Database(settings.db_path)
    return _DB


def set_database(db: Database | None) -> None:
    """Override the process-wide database (used by tests)."""
    global _DB
    with _DB_LOCK:
        _DB = db


# --------------------------------------------------------------------------- #
# JSON helpers -- every JSON column goes through these
# --------------------------------------------------------------------------- #
def dumps(value: Any) -> str:
    return json.dumps(value, default=str, ensure_ascii=False)


def loads(raw: str | bytes | None, default: Any = None) -> Any:
    if raw is None or raw == "":
        return {} if default is None else default
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        logger.warning("db.json_decode_failed", extra={"raw_prefix": str(raw)[:80]})
        return {} if default is None else default


def row_to_dict(row: sqlite3.Row | None, json_fields: Sequence[str] = ()) -> dict | None:
    """Convert a Row to a dict, decoding the named JSON columns."""
    if row is None:
        return None
    data = dict(row)
    for field in json_fields:
        if field in data:
            data[field] = loads(data[field], default={})
    return data
