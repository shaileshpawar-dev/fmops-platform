"""Model registry interface and promotion rules.

The registry is the platform's source of truth for *what is deployable*. A model
version carries not just an artifact but the full reproducibility block: dataset
version, dataset content hash, git commit, params, metrics and timestamps.

Stage machine
-------------

    Development -> Validation -> Staging -> Production
         |             |            |           |
         +-------------+------------+-----------+---> Archived

Rules enforced by :func:`assert_transition`:

* Forward moves must follow the chain; you cannot jump Development -> Production.
* Any stage may be Archived.
* Archived is terminal except for reinstatement to Development (rollback of an
  archival mistake).
* Demotion Production -> Staging is allowed: that is what a rollback does.
* Exactly one version may sit in Production per model name; promoting a new one
  archives the incumbent, and the previous production version is recorded so
  rollback has a target.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from app.core.exceptions import InvalidStageTransitionError
from app.core.logging import get_logger
from app.schemas.common import ModelStage
from app.schemas.model import ModelVersion, StageTransition

logger = get_logger(__name__)

# Legal stage moves. Keyed by source stage.
ALLOWED_TRANSITIONS: dict[ModelStage, set[ModelStage]] = {
    ModelStage.DEVELOPMENT: {ModelStage.VALIDATION, ModelStage.ARCHIVED},
    ModelStage.VALIDATION: {
        ModelStage.STAGING,
        ModelStage.DEVELOPMENT,
        ModelStage.ARCHIVED,
    },
    ModelStage.STAGING: {
        ModelStage.PRODUCTION,
        ModelStage.VALIDATION,
        ModelStage.ARCHIVED,
    },
    ModelStage.PRODUCTION: {ModelStage.STAGING, ModelStage.ARCHIVED},
    ModelStage.ARCHIVED: {ModelStage.DEVELOPMENT},
}

# Stages that may serve traffic.
SERVING_STAGES = (ModelStage.PRODUCTION, ModelStage.STAGING)


def assert_transition(current: ModelStage, target: ModelStage) -> None:
    """Raise unless ``current -> target`` is a legal move."""
    if current == target:
        return
    allowed = ALLOWED_TRANSITIONS.get(current, set())
    if target not in allowed:
        raise InvalidStageTransitionError(
            f"cannot move a model from {current.value} to {target.value}; "
            f"legal moves from {current.value} are: "
            f"{', '.join(sorted(s.value for s in allowed)) or 'none'}",
            current_stage=current.value,
            target_stage=target.value,
            allowed=[s.value for s in sorted(allowed, key=lambda x: x.value)],
        )


class ModelRegistry(ABC):
    """Versioned catalogue of deployable models."""

    backend: str = "abstract"

    @abstractmethod
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
    ) -> ModelVersion: ...

    @abstractmethod
    def get(self, name: str, version: int) -> ModelVersion: ...

    @abstractmethod
    def list_versions(
        self, name: str | None = None, stage: ModelStage | None = None
    ) -> list[ModelVersion]: ...

    @abstractmethod
    def list_models(self) -> list[dict[str, Any]]: ...

    @abstractmethod
    def transition_stage(
        self,
        name: str,
        version: int,
        stage: ModelStage,
        reason: str = "",
        actor: str = "system",
        archive_existing: bool = True,
    ) -> ModelVersion: ...

    @abstractmethod
    def get_latest(
        self, name: str, stage: ModelStage | None = None
    ) -> ModelVersion | None: ...

    @abstractmethod
    def set_tags(self, name: str, version: int, tags: dict[str, str]) -> ModelVersion: ...

    @abstractmethod
    def update_status(self, name: str, version: int, status: str) -> ModelVersion: ...

    @abstractmethod
    def history(self, name: str, version: int | None = None) -> list[StageTransition]: ...

    # -- derived helpers shared by every backend ---------------------------- #
    def get_production(self, name: str) -> ModelVersion | None:
        return self.get_latest(name, ModelStage.PRODUCTION)

    def get_serving(self, name: str) -> ModelVersion | None:
        """The version that should serve traffic: Production, else Staging."""
        for stage in SERVING_STAGES:
            found = self.get_latest(name, stage)
            if found is not None:
                return found
        return None

    def previous_production(self, name: str) -> ModelVersion | None:
        """The most recently archived version that was previously in Production.

        This is the rollback target when the current production model fails.
        """
        transitions = [
            t
            for t in self.history(name)
            if t.from_stage == ModelStage.PRODUCTION and t.to_stage != ModelStage.PRODUCTION
        ]
        for transition in sorted(transitions, key=lambda t: t.created_at, reverse=True):
            try:
                return self.get(name, transition.version)
            except Exception as exc:
                logger.debug(
                    "registry.rollback_candidate_unavailable",
                    extra={
                        "model": name,
                        "version": transition.version,
                        "error": str(exc),
                    },
                )
                continue
        return None
