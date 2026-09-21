"""SQLite-backed model registry.

The default local backend. It implements the full stage machine including the
"only one Production version" invariant and the transition history that
:meth:`previous_production` uses as the rollback target.
"""

from __future__ import annotations

from typing import Any

from app.core.audit import audit
from app.core.db import Database, dumps, get_database, loads
from app.core.exceptions import ModelNotFoundError
from app.core.logging import get_logger
from app.core.utils import utcnow_iso
from app.registry.base import ModelRegistry, assert_transition
from app.schemas.common import ModelStage, ModelStatus
from app.schemas.model import ModelVersion, StageTransition

logger = get_logger(__name__)

_JSON_FIELDS = ("params", "metrics", "tags")


class LocalModelRegistry(ModelRegistry):
    """Model registry stored in the platform's operational database."""

    backend = "local"

    def __init__(self, db: Database | None = None) -> None:
        self._db = db

    @property
    def db(self) -> Database:
        if self._db is None:
            self._db = get_database()
        return self._db

    # -- registration -------------------------------------------------------- #
    def register(
        self,
        name: str,
        artifact_uri: str,
        run_id: str | None = None,
        metrics: dict[str, float] | None = None,
        params: dict[str, Any] | None = None,
        dataset_version: str | None = None,
        dataset_hash: str | None = None,
        git_commit: str = "unknown",
        algorithm: str = "",
        tags: dict[str, str] | None = None,
        description: str = "",
        created_by: str | None = None,
        signature: dict[str, Any] | None = None,
    ) -> ModelVersion:
        now = utcnow_iso()
        with self.db.transaction() as conn:
            current_max = conn.execute(
                "SELECT COALESCE(MAX(version), 0) FROM model_versions WHERE name = ?",
                (name,),
            ).fetchone()[0]
            version = int(current_max) + 1
            conn.execute(
                "INSERT INTO model_versions (name, version, stage, status, run_id, "
                "artifact_uri, dataset_version, dataset_hash, git_commit, algorithm, "
                "params, metrics, tags, description, created_at, updated_at, created_by, "
                "signature) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    name,
                    version,
                    ModelStage.DEVELOPMENT.value,
                    ModelStatus.READY.value,
                    run_id,
                    artifact_uri,
                    dataset_version,
                    dataset_hash,
                    git_commit,
                    algorithm,
                    dumps(params or {}),
                    dumps(metrics or {}),
                    dumps(tags or {}),
                    description,
                    now,
                    now,
                    created_by or "system",
                    dumps(signature) if signature else None,
                ),
            )
            conn.execute(
                "INSERT INTO model_stage_transitions (name, version, from_stage, "
                "to_stage, reason, actor, created_at) VALUES (?,?,?,?,?,?,?)",
                (
                    name,
                    version,
                    None,
                    ModelStage.DEVELOPMENT.value,
                    "initial registration",
                    created_by or "system",
                    now,
                ),
            )

        logger.info(
            "registry.registered",
            extra={
                "model": name,
                "version": version,
                "algorithm": algorithm,
                "dataset_version": dataset_version,
                "git_commit": git_commit[:12],
                "backend": self.backend,
            },
        )
        audit(
            "model.register",
            "model",
            f"{name}:{version}",
            algorithm=algorithm,
            dataset_version=dataset_version,
            metrics=metrics or {},
        )
        return self.get(name, version)

    # -- reads --------------------------------------------------------------- #
    def get(self, name: str, version: int) -> ModelVersion:
        row = self.db.query_one(
            "SELECT * FROM model_versions WHERE name = ? AND version = ?",
            (name, int(version)),
        )
        if row is None:
            raise ModelNotFoundError(
                f"model {name} version {version} is not registered",
                model=name,
                version=version,
            )
        return _to_model(row)

    def list_versions(
        self, name: str | None = None, stage: ModelStage | None = None
    ) -> list[ModelVersion]:
        sql = "SELECT * FROM model_versions"
        clauses: list[str] = []
        params: list[Any] = []
        if name:
            clauses.append("name = ?")
            params.append(name)
        if stage:
            clauses.append("stage = ?")
            params.append(stage.value)
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY name ASC, version DESC"
        return [_to_model(row) for row in self.db.query(sql, params)]

    def list_models(self) -> list[dict[str, Any]]:
        rows = self.db.query(
            "SELECT name, COUNT(*) AS versions, MAX(version) AS latest_version, "
            "MIN(created_at) AS created_at, MAX(updated_at) AS updated_at "
            "FROM model_versions GROUP BY name ORDER BY name"
        )
        out = []
        for row in rows:
            item = dict(row)
            production = self.get_latest(item["name"], ModelStage.PRODUCTION)
            staging = self.get_latest(item["name"], ModelStage.STAGING)
            item["production_version"] = production.version if production else None
            item["staging_version"] = staging.version if staging else None
            out.append(item)
        return out

    def get_latest(self, name: str, stage: ModelStage | None = None) -> ModelVersion | None:
        if stage is None:
            row = self.db.query_one(
                "SELECT * FROM model_versions WHERE name = ? " "ORDER BY version DESC LIMIT 1",
                (name,),
            )
        else:
            row = self.db.query_one(
                "SELECT * FROM model_versions WHERE name = ? AND stage = ? "
                "ORDER BY version DESC LIMIT 1",
                (name, stage.value),
            )
        return _to_model(row) if row is not None else None

    def history(self, name: str, version: int | None = None) -> list[StageTransition]:
        sql = "SELECT * FROM model_stage_transitions WHERE name = ?"
        params: list[Any] = [name]
        if version is not None:
            sql += " AND version = ?"
            params.append(int(version))
        sql += " ORDER BY id DESC"
        out: list[StageTransition] = []
        for row in self.db.query(sql, params):
            data = dict(row)
            out.append(
                StageTransition(
                    name=data["name"],
                    version=data["version"],
                    from_stage=ModelStage(data["from_stage"]) if data["from_stage"] else None,
                    to_stage=ModelStage(data["to_stage"]),
                    reason=data["reason"] or "",
                    actor=data["actor"] or "system",
                    created_at=data["created_at"],
                )
            )
        return out

    # -- writes -------------------------------------------------------------- #
    def transition_stage(
        self,
        name: str,
        version: int,
        stage: ModelStage,
        reason: str = "",
        actor: str = "system",
        archive_existing: bool = True,
    ) -> ModelVersion:
        current = self.get(name, version)
        assert_transition(current.stage, stage)
        now = utcnow_iso()

        with self.db.transaction() as conn:
            displaced: int | None = None
            if stage == ModelStage.PRODUCTION and archive_existing:
                # Exactly one Production version per model. The incumbent is
                # archived, and that archival is what previous_production()
                # later reads to find the rollback target.
                for row in conn.execute(
                    "SELECT version FROM model_versions WHERE name = ? AND stage = ? "
                    "AND version != ?",
                    (name, ModelStage.PRODUCTION.value, int(version)),
                ).fetchall():
                    displaced = int(row[0])
                    conn.execute(
                        "UPDATE model_versions SET stage = ?, updated_at = ? "
                        "WHERE name = ? AND version = ?",
                        (ModelStage.ARCHIVED.value, now, name, displaced),
                    )
                    conn.execute(
                        "INSERT INTO model_stage_transitions (name, version, "
                        "from_stage, to_stage, reason, actor, created_at) "
                        "VALUES (?,?,?,?,?,?,?)",
                        (
                            name,
                            displaced,
                            ModelStage.PRODUCTION.value,
                            ModelStage.ARCHIVED.value,
                            f"superseded by version {version}",
                            actor,
                            now,
                        ),
                    )

            conn.execute(
                "UPDATE model_versions SET stage = ?, updated_at = ? "
                "WHERE name = ? AND version = ?",
                (stage.value, now, name, int(version)),
            )
            conn.execute(
                "INSERT INTO model_stage_transitions (name, version, from_stage, "
                "to_stage, reason, actor, created_at) VALUES (?,?,?,?,?,?,?)",
                (
                    name,
                    int(version),
                    current.stage.value,
                    stage.value,
                    reason,
                    actor,
                    now,
                ),
            )

        logger.info(
            "registry.stage_transition",
            extra={
                "model": name,
                "version": version,
                "from_stage": current.stage.value,
                "to_stage": stage.value,
                "reason": reason,
                "displaced_version": displaced,
            },
        )
        audit(
            "model.transition_stage",
            "model",
            f"{name}:{version}",
            from_stage=current.stage.value,
            to_stage=stage.value,
            reason=reason,
            actor=actor,
            displaced_version=displaced,
        )
        return self.get(name, version)

    def set_tags(self, name: str, version: int, tags: dict[str, str]) -> ModelVersion:
        current = self.get(name, version)
        merged = {**current.tags, **{str(k): str(v) for k, v in tags.items()}}
        self.db.execute(
            "UPDATE model_versions SET tags = ?, updated_at = ? "
            "WHERE name = ? AND version = ?",
            (dumps(merged), utcnow_iso(), name, int(version)),
        )
        return self.get(name, version)

    def update_status(self, name: str, version: int, status: str) -> ModelVersion:
        self.get(name, version)  # existence check
        self.db.execute(
            "UPDATE model_versions SET status = ?, updated_at = ? "
            "WHERE name = ? AND version = ?",
            (str(status), utcnow_iso(), name, int(version)),
        )
        logger.info(
            "registry.status_updated",
            extra={"model": name, "version": version, "status": status},
        )
        return self.get(name, version)


def _to_model(row) -> ModelVersion:
    data = dict(row)
    for field in _JSON_FIELDS:
        data[field] = loads(data.get(field), {})
    data.pop("id", None)
    data["signature"] = loads(data.get("signature"), None) if data.get("signature") else None
    data["stage"] = ModelStage(data["stage"])
    data["status"] = ModelStatus(data["status"])
    return ModelVersion(**data)
