"""MLflow Model Registry backend.

Maps the platform's stage machine onto MLflow 3.x registry primitives. MLflow
removed the built-in ``stage`` field in 3.x in favour of *aliases* and tags, so
this adapter:

* stores the stage as the tag ``fmops.stage`` on the model version,
* maintains an alias per stage (``production``, ``staging``, ...) pointing at
  the current holder, so ``models:/name@production`` resolves correctly for
  anyone loading models through MLflow directly,
* keeps the transition history as ordered ``fmops.transition.<n>`` tags.

The same :func:`~app.registry.base.assert_transition` rules apply, so promotion
behaviour is identical between backends.
"""

from __future__ import annotations

import json
from datetime import UTC
from typing import Any

from app.core.audit import audit
from app.core.config import Settings, get_settings
from app.core.exceptions import (
    DependencyMissingError,
    ModelNotFoundError,
    ProviderUnavailableError,
    RegistryError,
)
from app.core.logging import get_logger
from app.core.utils import jsonable, utcnow_iso
from app.registry.base import ModelRegistry, assert_transition
from app.schemas.common import ModelStage, ModelStatus
from app.schemas.model import ModelVersion, StageTransition

logger = get_logger(__name__)

STAGE_TAG = "fmops.stage"
STATUS_TAG = "fmops.status"
TRANSITION_TAG_PREFIX = "fmops.transition."
_RESERVED_TAGS = (
    STAGE_TAG,
    STATUS_TAG,
    "fmops.dataset_version",
    "fmops.dataset_hash",
    "fmops.git_commit",
    "fmops.algorithm",
    "fmops.params",
    "fmops.created_by",
    "fmops.signature",
)

# MLflow rejects tag values much past 5000 characters. A signature for a wide
# dataset can exceed that; the artifact sidecar always carries the full copy,
# and the serving layer falls back to it.
_MAX_TAG_CHARS = 4900


def _alias(stage: ModelStage) -> str:
    return stage.value.lower()


class MLflowModelRegistry(ModelRegistry):
    """Registry backed by an MLflow tracking/registry server."""

    backend = "mlflow"

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        try:
            import mlflow
            from mlflow.exceptions import RestException
        except ImportError as exc:
            raise DependencyMissingError(
                "mlflow is not installed; set FMOPS_TRACKING__REGISTRY_BACKEND=local",
                backend="mlflow",
            ) from exc
        self._mlflow = mlflow
        self._RestException = RestException
        try:
            mlflow.set_tracking_uri(self.settings.mlflow_tracking_uri)
            mlflow.set_registry_uri(self.settings.mlflow_registry_uri)
            self.client = mlflow.tracking.MlflowClient()
        except Exception as exc:
            raise ProviderUnavailableError(
                f"MLflow registry unreachable: {exc}",
                registry_uri=self.settings.mlflow_registry_uri,
            ) from exc

    # -- helpers ------------------------------------------------------------- #
    def _ensure_registered_model(self, name: str) -> None:
        try:
            self.client.get_registered_model(name)
        except Exception:
            try:
                self.client.create_registered_model(name)
            except Exception as exc:  # pragma: no cover - concurrent create
                if "RESOURCE_ALREADY_EXISTS" not in str(exc):
                    raise RegistryError(
                        f"could not create registered model {name}: {exc}", model=name
                    ) from exc

    def _fetch(self, name: str, version: int):
        try:
            return self.client.get_model_version(name, str(version))
        except Exception as exc:
            raise ModelNotFoundError(
                f"model {name} version {version} is not registered in MLflow",
                model=name,
                version=version,
            ) from exc

    def _to_model(self, mv) -> ModelVersion:
        tags = dict(mv.tags or {})
        params = json.loads(tags.get("fmops.params", "{}") or "{}")
        metrics: dict[str, float] = {}
        if mv.run_id:
            try:
                metrics = dict(self.client.get_run(mv.run_id).data.metrics)
            except Exception:
                metrics = {}
        user_tags = {
            k: v
            for k, v in tags.items()
            if k not in _RESERVED_TAGS and not k.startswith(TRANSITION_TAG_PREFIX)
        }
        return ModelVersion(
            name=mv.name,
            version=int(mv.version),
            stage=ModelStage(tags.get(STAGE_TAG, ModelStage.DEVELOPMENT.value)),
            status=ModelStatus(tags.get(STATUS_TAG, ModelStatus.READY.value)),
            run_id=mv.run_id,
            artifact_uri=mv.source or "",
            dataset_version=tags.get("fmops.dataset_version"),
            dataset_hash=tags.get("fmops.dataset_hash"),
            git_commit=tags.get("fmops.git_commit", "unknown"),
            algorithm=tags.get("fmops.algorithm", ""),
            params=params,
            metrics=metrics,
            tags=user_tags,
            description=mv.description or "",
            created_at=_ms_to_iso(mv.creation_timestamp),
            updated_at=_ms_to_iso(mv.last_updated_timestamp),
            created_by=tags.get("fmops.created_by"),
            signature=_signature_from_tag(tags.get("fmops.signature")),
        )

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
        self._ensure_registered_model(name)
        signature_json = json.dumps(signature) if signature else ""
        all_tags = {
            STAGE_TAG: ModelStage.DEVELOPMENT.value,
            STATUS_TAG: ModelStatus.READY.value,
            "fmops.git_commit": git_commit,
            "fmops.algorithm": algorithm,
            "fmops.params": json.dumps(jsonable(params or {}))[:4900],
            **(
                {"fmops.signature": signature_json}
                if signature_json and len(signature_json) <= _MAX_TAG_CHARS
                else {}
            ),
            f"{TRANSITION_TAG_PREFIX}0": json.dumps(
                {
                    "from": None,
                    "to": ModelStage.DEVELOPMENT.value,
                    "reason": "initial registration",
                    "actor": created_by or "system",
                    "at": utcnow_iso(),
                }
            ),
            **{str(k): str(v) for k, v in (tags or {}).items()},
        }
        if dataset_version:
            all_tags["fmops.dataset_version"] = dataset_version
        if dataset_hash:
            all_tags["fmops.dataset_hash"] = dataset_hash
        if created_by:
            all_tags["fmops.created_by"] = created_by

        try:
            created = self.client.create_model_version(
                name=name,
                source=artifact_uri,
                run_id=run_id,
                tags=all_tags,
                description=description,
            )
        except Exception as exc:
            raise RegistryError(
                f"MLflow rejected the model version: {exc}",
                model=name,
                artifact_uri=artifact_uri,
            ) from exc

        logger.info(
            "registry.registered",
            extra={
                "model": name,
                "version": int(created.version),
                "backend": self.backend,
                "algorithm": algorithm,
            },
        )
        audit(
            "model.register",
            "model",
            f"{name}:{created.version}",
            backend=self.backend,
            algorithm=algorithm,
            dataset_version=dataset_version,
        )
        return self._to_model(created)

    # -- reads --------------------------------------------------------------- #
    def get(self, name: str, version: int) -> ModelVersion:
        return self._to_model(self._fetch(name, version))

    def list_versions(
        self, name: str | None = None, stage: ModelStage | None = None
    ) -> list[ModelVersion]:
        names = [name] if name else [m["name"] for m in self.list_models()]
        out: list[ModelVersion] = []
        for model_name in names:
            try:
                versions = self.client.search_model_versions(f"name='{model_name}'")
            except Exception as exc:
                logger.warning(
                    "registry.search_failed",
                    extra={"model": model_name, "error": str(exc)},
                )
                continue
            for mv in versions:
                item = self._to_model(mv)
                if stage is None or item.stage == stage:
                    out.append(item)
        out.sort(key=lambda m: (m.name, -m.version))
        return out

    def list_models(self) -> list[dict[str, Any]]:
        out = []
        try:
            registered = self.client.search_registered_models()
        except Exception as exc:
            raise ProviderUnavailableError(f"could not list registered models: {exc}") from exc
        for model in registered:
            versions = self.list_versions(model.name)
            production = next((v for v in versions if v.stage == ModelStage.PRODUCTION), None)
            staging = next((v for v in versions if v.stage == ModelStage.STAGING), None)
            out.append(
                {
                    "name": model.name,
                    "versions": len(versions),
                    "latest_version": max((v.version for v in versions), default=None),
                    "production_version": production.version if production else None,
                    "staging_version": staging.version if staging else None,
                    "created_at": _ms_to_iso(model.creation_timestamp),
                    "updated_at": _ms_to_iso(model.last_updated_timestamp),
                }
            )
        return out

    def get_latest(self, name: str, stage: ModelStage | None = None) -> ModelVersion | None:
        versions = self.list_versions(name, stage)
        return versions[0] if versions else None

    def history(self, name: str, version: int | None = None) -> list[StageTransition]:
        versions = (
            [self.get(name, version)] if version is not None else self.list_versions(name)
        )
        out: list[StageTransition] = []
        for item in versions:
            mv = self._fetch(name, item.version)
            for key, raw in sorted((mv.tags or {}).items()):
                if not key.startswith(TRANSITION_TAG_PREFIX):
                    continue
                try:
                    payload = json.loads(raw)
                except ValueError:
                    continue
                out.append(
                    StageTransition(
                        name=name,
                        version=item.version,
                        from_stage=(
                            ModelStage(payload["from"]) if payload.get("from") else None
                        ),
                        to_stage=ModelStage(payload["to"]),
                        reason=payload.get("reason", ""),
                        actor=payload.get("actor", "system"),
                        created_at=payload.get("at", ""),
                    )
                )
        out.sort(key=lambda t: t.created_at, reverse=True)
        return out

    # -- writes -------------------------------------------------------------- #
    def _record_transition(
        self,
        name: str,
        version: int,
        from_stage: ModelStage | None,
        to_stage: ModelStage,
        reason: str,
        actor: str,
    ) -> None:
        mv = self._fetch(name, version)
        index = sum(1 for k in (mv.tags or {}) if k.startswith(TRANSITION_TAG_PREFIX))
        self.client.set_model_version_tag(
            name,
            str(version),
            f"{TRANSITION_TAG_PREFIX}{index}",
            json.dumps(
                {
                    "from": from_stage.value if from_stage else None,
                    "to": to_stage.value,
                    "reason": reason,
                    "actor": actor,
                    "at": utcnow_iso(),
                }
            ),
        )

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

        displaced: int | None = None
        if stage == ModelStage.PRODUCTION and archive_existing:
            for other in self.list_versions(name, ModelStage.PRODUCTION):
                if other.version == version:
                    continue
                displaced = other.version
                self.client.set_model_version_tag(
                    name, str(other.version), STAGE_TAG, ModelStage.ARCHIVED.value
                )
                self._record_transition(
                    name,
                    other.version,
                    ModelStage.PRODUCTION,
                    ModelStage.ARCHIVED,
                    f"superseded by version {version}",
                    actor,
                )

        self.client.set_model_version_tag(name, str(version), STAGE_TAG, stage.value)
        self._record_transition(name, version, current.stage, stage, reason, actor)

        # Keep MLflow aliases in sync so `models:/name@production` resolves.
        try:
            if stage == ModelStage.ARCHIVED:
                for candidate in (ModelStage.PRODUCTION, ModelStage.STAGING):
                    try:
                        aliased = self.client.get_model_version_by_alias(
                            name, _alias(candidate)
                        )
                        if int(aliased.version) == int(version):
                            self.client.delete_registered_model_alias(name, _alias(candidate))
                    except Exception as exc:
                        logger.debug(
                            "registry.alias_absent",
                            extra={"model": name, "error": str(exc)},
                        )
                        continue
            else:
                self.client.set_registered_model_alias(name, _alias(stage), str(version))
        except Exception as exc:
            logger.warning(
                "registry.alias_update_failed",
                extra={"model": name, "version": version, "error": str(exc)},
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
                "backend": self.backend,
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
        for key, value in tags.items():
            self.client.set_model_version_tag(name, str(version), str(key), str(value))
        return self.get(name, version)

    def update_status(self, name: str, version: int, status: str) -> ModelVersion:
        self.client.set_model_version_tag(name, str(version), STATUS_TAG, str(status))
        return self.get(name, version)


def _ms_to_iso(milliseconds: int | None) -> str:
    if not milliseconds:
        return utcnow_iso()
    from datetime import datetime

    return (
        datetime.fromtimestamp(milliseconds / 1000.0, tz=UTC).strftime("%Y-%m-%dT%H:%M:%S.%f")[
            :-3
        ]
        + "Z"
    )


def _signature_from_tag(raw: str | None) -> dict[str, Any] | None:
    if not raw:
        return None
    try:
        value = json.loads(raw)
    except ValueError:
        return None
    return value if isinstance(value, dict) else None
