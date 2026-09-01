"""Deployment provider interface and the deployment state store.

A *provider* owns the mechanics of putting model versions behind an endpoint and
splitting traffic between them:

* :class:`~app.deployment.local_provider.LocalDeploymentProvider` serves the
  models in-process and routes real requests by weight. Nothing about the
  routing is faked -- a 90/10 canary genuinely sends about 10% of predictions to
  the candidate. What it does not do is provision infrastructure.
* :class:`~app.deployment.sagemaker_provider.SageMakerDeploymentProvider` maps
  the same operations onto SageMaker endpoint configs and production-variant
  weights.

A *strategy* (blue/green, canary, shadow) drives a provider through a sequence
of traffic states and decides whether to continue, finish, or roll back. That
separation is what lets the same canary logic run locally and on SageMaker.
"""

from __future__ import annotations

import builtins
from abc import ABC, abstractmethod
from typing import Any

from app.core.db import Database, dumps, get_database, loads
from app.core.exceptions import DeploymentNotFoundError
from app.core.logging import get_logger
from app.core.utils import new_id, utcnow_iso
from app.schemas.common import DeploymentState, DeploymentStrategy, HealthStatus
from app.schemas.deployment import (
    Deployment,
    DeploymentEvent,
    EndpointHealth,
    TrafficSplit,
)

logger = get_logger(__name__)


class DeploymentProvider(ABC):
    """Puts model versions behind an endpoint and splits traffic."""

    name: str = "abstract"

    @abstractmethod
    def apply(
        self,
        endpoint_name: str,
        model_name: str,
        traffic: TrafficSplit,
        shadow_version: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Make ``traffic`` the live routing for ``endpoint_name``."""

    @abstractmethod
    def status(self, endpoint_name: str) -> dict[str, Any]:
        """Provider-side view of the endpoint."""

    @abstractmethod
    def health_check(self, endpoint_name: str, model_name: str) -> EndpointHealth: ...

    @abstractmethod
    def teardown(self, endpoint_name: str) -> None: ...

    def supports_shadow(self) -> bool:
        return True


class DeploymentStore:
    """Persistence for deployments and their event timeline."""

    _JSON_FIELDS = ("traffic", "metadata")

    def __init__(self, db: Database | None = None) -> None:
        self._db = db

    @property
    def db(self) -> Database:
        if self._db is None:
            self._db = get_database()
        return self._db

    def create(
        self,
        endpoint_name: str,
        provider: str,
        strategy: DeploymentStrategy,
        model_name: str,
        candidate_version: int | None,
        current_version: int | None = None,
        previous_version: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Deployment:
        deployment_id = new_id("dep")
        now = utcnow_iso()
        self.db.execute(
            "INSERT INTO deployments (id, endpoint_name, provider, strategy, state, "
            "model_name, current_version, previous_version, candidate_version, "
            "traffic, shadow_version, health, message, metadata, created_at, updated_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                deployment_id,
                endpoint_name,
                provider,
                strategy.value,
                DeploymentState.PENDING.value,
                model_name,
                current_version,
                previous_version,
                candidate_version,
                dumps({}),
                None,
                HealthStatus.UNKNOWN.value,
                "",
                dumps(metadata or {}),
                now,
                now,
            ),
        )
        self.add_event(
            deployment_id,
            "created",
            {
                "strategy": strategy.value,
                "candidate_version": candidate_version,
                "current_version": current_version,
            },
        )
        return self.get(deployment_id)

    def get(self, deployment_id: str) -> Deployment:
        row = self.db.query_one("SELECT * FROM deployments WHERE id = ?", (deployment_id,))
        if row is None:
            raise DeploymentNotFoundError(
                f"deployment {deployment_id} not found", deployment_id=deployment_id
            )
        return self._to_deployment(row, with_events=True)

    def active(self, endpoint_name: str) -> Deployment | None:
        """The most recent deployment that is live or in progress."""
        row = self.db.query_one(
            "SELECT * FROM deployments WHERE endpoint_name = ? AND state IN (?,?,?) "
            "ORDER BY updated_at DESC, rowid DESC LIMIT 1",
            (
                endpoint_name,
                DeploymentState.LIVE.value,
                DeploymentState.IN_PROGRESS.value,
                DeploymentState.ROLLED_BACK.value,
            ),
        )
        return self._to_deployment(row, with_events=True) if row is not None else None

    def latest(self, endpoint_name: str | None = None) -> Deployment | None:
        if endpoint_name:
            row = self.db.query_one(
                "SELECT * FROM deployments WHERE endpoint_name = ? "
                "ORDER BY updated_at DESC, rowid DESC LIMIT 1",
                (endpoint_name,),
            )
        else:
            row = self.db.query_one(
                "SELECT * FROM deployments ORDER BY updated_at DESC, rowid DESC LIMIT 1"
            )
        return self._to_deployment(row, with_events=True) if row is not None else None

    def list(self, endpoint_name: str | None = None, limit: int = 50) -> list[Deployment]:
        if endpoint_name:
            rows = self.db.query(
                "SELECT * FROM deployments WHERE endpoint_name = ? "
                "ORDER BY created_at DESC LIMIT ?",
                (endpoint_name, limit),
            )
        else:
            rows = self.db.query(
                "SELECT * FROM deployments ORDER BY created_at DESC LIMIT ?", (limit,)
            )
        return [self._to_deployment(row) for row in rows]

    def update(self, deployment_id: str, **fields: Any) -> Deployment:
        if not fields:
            return self.get(deployment_id)
        assignments: list[str] = []
        values: list[Any] = []
        for key, value in fields.items():
            if key in self._JSON_FIELDS:
                value = dumps(value)
            elif hasattr(value, "value"):  # enum
                value = value.value
            assignments.append(f"{key} = ?")
            values.append(value)
        assignments.append("updated_at = ?")
        values.append(utcnow_iso())
        values.append(deployment_id)
        self.db.execute(
            f"UPDATE deployments SET {', '.join(assignments)} WHERE id = ?", values
        )
        return self.get(deployment_id)

    def add_event(
        self, deployment_id: str, event: str, detail: dict[str, Any] | None = None
    ) -> None:
        self.db.execute(
            "INSERT INTO deployment_events (deployment_id, event, detail, created_at) "
            "VALUES (?,?,?,?)",
            (deployment_id, event, dumps(detail or {}), utcnow_iso()),
        )
        logger.info(
            "deployment.event",
            extra={
                "deployment_id": deployment_id,
                "deployment_event": event,
                **(detail or {}),
            },
        )

    # builtins.list, not bare list: this class defines a method named `list`,
    # which shadows the builtin for annotations evaluated in class scope.
    def events(self, deployment_id: str, limit: int = 100) -> builtins.list[DeploymentEvent]:
        rows = self.db.query(
            "SELECT * FROM deployment_events WHERE deployment_id = ? "
            "ORDER BY id ASC LIMIT ?",
            (deployment_id, limit),
        )
        return [
            DeploymentEvent(
                deployment_id=row["deployment_id"],
                event=row["event"],
                detail=loads(row["detail"], {}),
                created_at=row["created_at"],
            )
            for row in rows
        ]

    def _to_deployment(self, row, with_events: bool = False) -> Deployment:
        data = dict(row)
        data["traffic"] = {str(k): float(v) for k, v in loads(data.get("traffic"), {}).items()}
        data["metadata"] = loads(data.get("metadata"), {})
        data["state"] = DeploymentState(data["state"])
        data["strategy"] = DeploymentStrategy(data["strategy"])
        data["health"] = HealthStatus(data.get("health") or "unknown")
        data["message"] = data.get("message") or ""
        deployment = Deployment(**data)
        if with_events:
            deployment.events = self.events(deployment.id)
        return deployment


_STORE: DeploymentStore | None = None


def get_deployment_store() -> DeploymentStore:
    global _STORE
    if _STORE is None:
        _STORE = DeploymentStore()
    return _STORE
