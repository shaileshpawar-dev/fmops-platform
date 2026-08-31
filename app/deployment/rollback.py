"""Rollback.

The platform always knows three versions per endpoint:

``current_version``    what is serving now
``previous_version``   what was serving before the last successful deployment
``candidate_version``  what is being rolled out (or shadowing)

Rollback restores ``previous_version`` (or an explicitly requested version),
moves the registry stages to match reality, and records why. It is deliberately
usable from three places: manually via the API, automatically by a strategy that
fails its health gate, and automatically by the health watchdog when a live
endpoint degrades.

Choosing the target: an explicit ``to_version`` wins; otherwise the deployment's
recorded ``previous_version``; otherwise the registry's most recently archived
ex-production version. If none of those exist there is nothing safe to roll back
to and we raise rather than guess.
"""

from __future__ import annotations

from app.core.audit import audit
from app.core.config import Settings, get_settings
from app.core.exceptions import RollbackError
from app.core.logging import get_logger
from app.deployment.base import DeploymentProvider, DeploymentStore, get_deployment_store
from app.registry.base import ModelRegistry
from app.registry.factory import get_registry
from app.schemas.common import DeploymentState, HealthStatus, ModelStage
from app.schemas.deployment import Deployment, RollbackResult, TrafficSplit

logger = get_logger(__name__)


class RollbackManager:
    """Restores a previously healthy model version."""

    def __init__(
        self,
        provider: DeploymentProvider,
        store: DeploymentStore | None = None,
        registry: ModelRegistry | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.provider = provider
        self.store = store or get_deployment_store()
        self.registry = registry or get_registry()
        self.settings = settings or get_settings()

    def resolve_target(self, deployment: Deployment, to_version: int | None = None) -> int:
        """Pick the version to roll back to, or explain why none exists."""
        if to_version is not None:
            # Validate it exists before touching traffic.
            self.registry.get(deployment.model_name, to_version)
            return to_version

        if (
            deployment.previous_version is not None
            and deployment.previous_version != deployment.current_version
        ):
            return deployment.previous_version

        archived = self.registry.previous_production(deployment.model_name)
        if archived is not None and archived.version != deployment.current_version:
            return archived.version

        raise RollbackError(
            "no rollback target: this endpoint has no recorded previous version "
            "and the registry has no archived production version to restore",
            endpoint=deployment.endpoint_name,
            current_version=deployment.current_version,
        )

    def rollback(
        self,
        endpoint_name: str | None = None,
        to_version: int | None = None,
        reason: str = "manual rollback",
        actor: str = "system",
        restore_registry_stage: bool = True,
    ) -> RollbackResult:
        endpoint = endpoint_name or self.settings.deployment.endpoint_name
        deployment = self.store.active(endpoint) or self.store.latest(endpoint)
        if deployment is None:
            raise RollbackError(
                f"no deployment exists for endpoint {endpoint!r}", endpoint=endpoint
            )

        target = self.resolve_target(deployment, to_version)
        rolled_from = deployment.current_version

        self.store.update(deployment.id, state=DeploymentState.ROLLING_BACK)
        self.store.add_event(
            deployment.id,
            "rollback.started",
            {"from_version": rolled_from, "to_version": target, "reason": reason},
        )

        split = TrafficSplit.all_to(target)
        try:
            self.provider.apply(endpoint, deployment.model_name, split)
        except Exception as exc:
            self.store.update(
                deployment.id,
                state=DeploymentState.FAILED,
                health=HealthStatus.UNHEALTHY,
                message=f"rollback to version {target} failed: {exc}",
            )
            self.store.add_event(
                deployment.id, "rollback.failed", {"error": str(exc), "to_version": target}
            )
            audit(
                "deployment.rollback",
                "deployment",
                deployment.id,
                outcome="failure",
                endpoint=endpoint,
                to_version=target,
                error=str(exc),
            )
            raise RollbackError(
                f"could not roll back {endpoint} to version {target}: {exc}",
                endpoint=endpoint,
                to_version=target,
            ) from exc

        health = self.provider.health_check(endpoint, deployment.model_name)
        updated = self.store.update(
            deployment.id,
            state=DeploymentState.ROLLED_BACK,
            health=health.status,
            current_version=target,
            previous_version=rolled_from,
            candidate_version=None,
            shadow_version=None,
            traffic=split.as_dict(),
            message=f"rolled back to version {target}: {reason}",
        )
        self.store.add_event(
            deployment.id,
            "rollback.completed",
            {
                "from_version": rolled_from,
                "to_version": target,
                "health": health.status.value,
            },
        )

        if restore_registry_stage:
            self._restore_stages(deployment.model_name, target, rolled_from, reason, actor)

        logger.warning(
            "deployment.rolled_back",
            extra={
                "endpoint": endpoint,
                "from_version": rolled_from,
                "to_version": target,
                "reason": reason,
                "health": health.status.value,
            },
        )
        audit(
            "deployment.rollback",
            "deployment",
            deployment.id,
            endpoint=endpoint,
            from_version=rolled_from,
            to_version=target,
            reason=reason,
            actor=actor,
        )
        return RollbackResult(
            deployment_id=updated.id,
            endpoint_name=endpoint,
            rolled_back_from=rolled_from,
            rolled_back_to=target,
            reason=reason,
            succeeded=health.status == HealthStatus.HEALTHY,
            message=(
                f"traffic restored to version {target}"
                if health.status == HealthStatus.HEALTHY
                else f"traffic restored to version {target} but health is {health.status.value}"
            ),
        )

    def _restore_stages(
        self,
        model_name: str,
        target: int,
        demoted: int | None,
        reason: str,
        actor: str,
    ) -> None:
        """Make the registry agree with what is actually serving.

        The failing version is demoted out of Production and the restored
        version is walked back up to Production, so the registry never claims a
        version is live when it is not.
        """
        try:
            if demoted is not None and demoted != target:
                current = self.registry.get(model_name, demoted)
                if current.stage == ModelStage.PRODUCTION:
                    self.registry.transition_stage(
                        model_name,
                        demoted,
                        ModelStage.STAGING,
                        reason=f"demoted by rollback: {reason}",
                        actor=actor,
                    )
                    self.registry.update_status(model_name, demoted, "rejected")

            restored = self.registry.get(model_name, target)
            if restored.stage != ModelStage.PRODUCTION:
                chain = [
                    ModelStage.DEVELOPMENT,
                    ModelStage.VALIDATION,
                    ModelStage.STAGING,
                    ModelStage.PRODUCTION,
                ]
                if restored.stage == ModelStage.ARCHIVED:
                    self.registry.transition_stage(
                        model_name,
                        target,
                        ModelStage.DEVELOPMENT,
                        reason=f"reinstated by rollback: {reason}",
                        actor=actor,
                    )
                    start = 0
                else:
                    start = chain.index(restored.stage)
                for stage in chain[start + 1 :]:
                    self.registry.transition_stage(
                        model_name,
                        target,
                        stage,
                        reason=f"restored by rollback: {reason}",
                        actor=actor,
                    )
                self.registry.update_status(model_name, target, "approved")
        except Exception as exc:
            # Traffic is already restored; a registry mismatch is serious but
            # must not be reported as a failed rollback.
            logger.error(
                "deployment.rollback_stage_restore_failed",
                extra={
                    "model": model_name,
                    "to_version": target,
                    "error": str(exc),
                    "impact": "traffic is correct but registry stages may be stale",
                },
            )
