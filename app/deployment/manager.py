"""Deployment manager.

The single entry point for putting a model version into service. It:

1. resolves the provider (local or SageMaker) and strategy from configuration
   or from the request,
2. refuses to deploy a version the registry has not approved,
3. records a deployment row and its event timeline,
4. runs the strategy,
5. keeps the model registry's stages in agreement with what is actually serving.

Step 2 is the important one: the manager is the last gate before traffic. A
version that failed its approval checks cannot be deployed by accident, only by
an explicit ``force=True`` that is written to the audit log.
"""

from __future__ import annotations

from typing import Any

from app.core.audit import audit
from app.core.config import Settings, get_settings
from app.core.exceptions import DeploymentError, ModelNotFoundError
from app.core.logging import get_logger
from app.deployment.base import DeploymentProvider, DeploymentStore, get_deployment_store
from app.deployment.blue_green import BlueGreenStrategy
from app.deployment.canary import CanaryStrategy
from app.deployment.direct import DirectStrategy
from app.deployment.local_provider import get_local_provider
from app.deployment.rollback import RollbackManager
from app.deployment.shadow import ShadowStrategy
from app.deployment.strategy import HealthProbe, Strategy, StrategyContext
from app.registry.base import ModelRegistry
from app.registry.context import default_model_name, endpoint_for, signature_of
from app.registry.factory import get_registry
from app.schemas.common import DeploymentState, DeploymentStrategy, ModelStage
from app.schemas.deployment import (
    Deployment,
    DeploymentRequest,
    DeploymentResult,
    EndpointHealth,
    RollbackResult,
)

logger = get_logger(__name__)

STRATEGIES: dict[DeploymentStrategy, type[Strategy]] = {
    DeploymentStrategy.BLUE_GREEN: BlueGreenStrategy,
    DeploymentStrategy.CANARY: CanaryStrategy,
    DeploymentStrategy.SHADOW: ShadowStrategy,
    DeploymentStrategy.DIRECT: DirectStrategy,
}

# Stages whose versions may be deployed. Only the approval gate (the end of a
# training run, or /approve) moves a version into either, so "deployable" means
# "the gate said yes". Validation is deliberately absent: it is reachable
# without the gate, and a successful deploy walks the version to Production.
DEPLOYABLE_STAGES = (ModelStage.STAGING, ModelStage.PRODUCTION)


class DeploymentRefusedError(DeploymentError):
    """The request cannot be deployed as asked; nothing was changed."""

    code = "deployment_refused"
    http_status = 409


def build_provider(settings: Settings | None = None) -> DeploymentProvider:
    """Resolve the deployment provider from configuration.

    SageMaker is used only when explicitly configured *and* AWS is enabled; we
    never present the local provider as if it were SageMaker.
    """
    settings = settings or get_settings()
    if settings.deployment.provider == "sagemaker":
        if not settings.aws.enabled:
            raise DeploymentError(
                "deployment provider is set to 'sagemaker' but AWS is disabled; "
                "set FMOPS_AWS__ENABLED=true and configure a role ARN, or switch "
                "to FMOPS_DEPLOYMENT__PROVIDER=local",
                provider="sagemaker",
            )
        from app.deployment.sagemaker_provider import SageMakerDeploymentProvider

        return SageMakerDeploymentProvider(settings)
    return get_local_provider()


def default_endpoint(settings: Settings) -> str:
    return endpoint_for(default_model_name(settings), settings)


class DeploymentManager:
    """Orchestrates deployments, health checks and rollbacks."""

    def __init__(
        self,
        provider: DeploymentProvider | None = None,
        store: DeploymentStore | None = None,
        registry: ModelRegistry | None = None,
        settings: Settings | None = None,
        probe: HealthProbe | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.provider = provider or build_provider(self.settings)
        self.store = store or get_deployment_store()
        self.registry = registry or get_registry()
        self._probe = probe

    @property
    def probe(self) -> HealthProbe | None:
        if self._probe is None:
            from app.monitoring.metrics import MetricsProbe

            self._probe = MetricsProbe()
        return self._probe

    # -- deploy -------------------------------------------------------------- #
    def validate_request(
        self, request: DeploymentRequest, force: bool = False
    ) -> tuple[str, str, DeploymentStrategy]:
        """Everything that can refuse a deployment, checked without side effects.

        The API calls this before queueing a deployment job, so a request that
        would be refused is refused immediately, not minutes later in a job.
        """
        model_name = request.model_name or default_model_name(self.settings)
        endpoint = endpoint_for(model_name, self.settings)
        if request.endpoint_name and request.endpoint_name != endpoint:
            # An endpoint belongs to one model. Deploying another model onto it
            # would silently change what its callers are scored by.
            raise DeploymentRefusedError(
                f"endpoint {request.endpoint_name!r} does not belong to {model_name}; "
                f"its endpoint is {endpoint!r}",
                model=model_name,
            )
        strategy_kind = request.strategy or DeploymentStrategy(
            self.settings.deployment.strategy
        )
        candidate = self._validate_candidate(model_name, request.model_version, force)
        active = self.store.active(endpoint)
        current_version = active.current_version if active else None
        if strategy_kind in (DeploymentStrategy.CANARY, DeploymentStrategy.SHADOW) and (
            current_version is not None and current_version != candidate.version
        ):
            self._check_contracts_match(model_name, current_version, candidate.version)
        return model_name, endpoint, strategy_kind

    def _check_contracts_match(self, model_name: str, live: int, candidate: int) -> None:
        """Canary and shadow send one request to two versions.

        That only works when both accept the same input. A version with a
        different feature set would fail on its share of traffic (canary) or on
        every mirrored request (shadow) -- which reads as a bad model when the
        real problem is a bad rollout. Such a change needs a full switch.
        """
        live_sig = signature_of(self.registry.get(model_name, live))
        cand_sig = signature_of(self.registry.get(model_name, candidate))
        if live_sig is None or cand_sig is None:
            return
        if set(live_sig.feature_columns) != set(cand_sig.feature_columns):
            added = sorted(set(cand_sig.feature_columns) - set(live_sig.feature_columns))
            removed = sorted(set(live_sig.feature_columns) - set(cand_sig.feature_columns))
            raise DeploymentRefusedError(
                f"v{candidate} and the live v{live} take different inputs "
                f"(added: {', '.join(added) or 'none'}; removed: {', '.join(removed) or 'none'}), "
                "so they cannot share traffic; deploy it with blue_green or direct",
                model=model_name,
            )

    def deploy(
        self,
        request: DeploymentRequest,
        force: bool = False,
        actor: str = "system",
        sleep: Any = None,
        on_created: Any = None,
    ) -> DeploymentResult:
        """Put a version into service.

        ``sleep`` replaces the canary's wait between steps; the job runner
        passes one that wakes to honour cancellation.
        """
        model_name, endpoint, strategy_kind = self.validate_request(request, force)
        candidate = self.registry.get(model_name, request.model_version)
        active = self.store.active(endpoint)
        current_version = active.current_version if active else None

        config = self.settings.deployment
        if request.canary_steps:
            config = config.model_copy(update={"canary_steps": request.canary_steps})

        deployment = self.store.create(
            endpoint_name=endpoint,
            provider=self.provider.name,
            strategy=strategy_kind,
            model_name=model_name,
            candidate_version=candidate.version,
            current_version=current_version,
            previous_version=active.previous_version if active else None,
            metadata={
                "reason": request.reason,
                "actor": actor,
                "forced": force,
                "candidate_stage": candidate.stage.value,
            },
        )

        if on_created is not None:
            on_created(deployment.id)
        if strategy_kind == DeploymentStrategy.CANARY and sleep is not None:
            strategy: Strategy = CanaryStrategy(self.settings, sleep=sleep)
        else:
            strategy = STRATEGIES[strategy_kind](self.settings)
        ctx = StrategyContext(
            deployment=deployment,
            provider=self.provider,
            store=self.store,
            model_name=model_name,
            candidate_version=candidate.version,
            current_version=current_version,
            config=config,
            settings=self.settings,
            probe=self.probe,
            metadata={"deployment_id": deployment.id},
        )

        logger.info(
            "deployment.started",
            extra={
                "deployment_id": deployment.id,
                "endpoint": endpoint,
                "model": model_name,
                "candidate_version": candidate.version,
                "current_version": current_version,
                "strategy": strategy_kind.value,
                "provider": self.provider.name,
            },
        )

        try:
            result = strategy.execute(ctx)
        except DeploymentError:
            audit(
                "deployment.deploy",
                "deployment",
                deployment.id,
                outcome="failure",
                endpoint=endpoint,
                model=model_name,
                version=candidate.version,
                strategy=strategy_kind.value,
            )
            raise

        if result.succeeded and strategy_kind != DeploymentStrategy.SHADOW:
            self._sync_registry_stage(model_name, candidate.version, current_version, actor)

        audit(
            "deployment.deploy",
            "deployment",
            deployment.id,
            outcome="success" if result.succeeded else "failure",
            endpoint=endpoint,
            model=model_name,
            version=candidate.version,
            previous_version=current_version,
            strategy=strategy_kind.value,
            rolled_back=result.rolled_back,
            actor=actor,
        )
        return result

    def _validate_candidate(self, model_name: str, version: int, force: bool):
        try:
            candidate = self.registry.get(model_name, version)
        except ModelNotFoundError:
            raise
        if candidate.stage in DEPLOYABLE_STAGES or force:
            if force and candidate.stage not in DEPLOYABLE_STAGES:
                logger.warning(
                    "deployment.forced_undeployable_stage",
                    extra={
                        "model": model_name,
                        "version": version,
                        "stage": candidate.stage.value,
                        "status": candidate.status.value,
                    },
                )
            return candidate
        raise DeploymentRefusedError(
            f"model {model_name} v{version} is in stage {candidate.stage.value} and "
            f"status {candidate.status.value}; only "
            f"{', '.join(s.value for s in DEPLOYABLE_STAGES)} versions may be "
            "deployed. A version reaches those stages only through the approval gate "
            f"(POST /api/v1/models/{model_name}/versions/{version}/approve).",
            model=model_name,
            version=version,
            stage=candidate.stage.value,
        )

    def _sync_registry_stage(
        self, model_name: str, version: int, previous: int | None, actor: str
    ) -> None:
        """Move the deployed version to Production in the registry."""
        try:
            current = self.registry.get(model_name, version)
            if current.stage == ModelStage.PRODUCTION:
                return
            chain = [
                ModelStage.DEVELOPMENT,
                ModelStage.VALIDATION,
                ModelStage.STAGING,
                ModelStage.PRODUCTION,
            ]
            start = chain.index(current.stage) if current.stage in chain else 0
            for stage in chain[start + 1 :]:
                self.registry.transition_stage(
                    model_name,
                    version,
                    stage,
                    reason="deployed to the live endpoint",
                    actor=actor,
                )
        except Exception as exc:
            logger.error(
                "deployment.registry_sync_failed",
                extra={
                    "model": model_name,
                    "version": version,
                    "error": str(exc),
                    "impact": "the version is serving but its registry stage is stale",
                },
            )

    # -- inspection ---------------------------------------------------------- #
    def status(self, endpoint_name: str | None = None) -> Deployment | None:
        endpoint = endpoint_name or default_endpoint(self.settings)
        return self.store.active(endpoint) or self.store.latest(endpoint)

    def list(self, endpoint_name: str | None = None, limit: int = 50) -> list[Deployment]:
        return self.store.list(endpoint_name, limit)

    def health(self, endpoint_name: str | None = None) -> EndpointHealth:
        endpoint = endpoint_name or default_endpoint(self.settings)
        deployment = self.status(endpoint)
        model_name = deployment.model_name if deployment else default_model_name(self.settings)
        return self.provider.health_check(endpoint, model_name)

    def provider_status(self, endpoint_name: str | None = None) -> dict[str, Any]:
        endpoint = endpoint_name or default_endpoint(self.settings)
        return self.provider.status(endpoint)

    # -- rollback ------------------------------------------------------------ #
    def rollback(
        self,
        endpoint_name: str | None = None,
        to_version: int | None = None,
        reason: str = "manual rollback",
        actor: str = "system",
    ) -> RollbackResult:
        manager = RollbackManager(self.provider, self.store, self.registry, self.settings)
        return manager.rollback(endpoint_name, to_version, reason, actor)

    def terminate(self, endpoint_name: str | None = None) -> None:
        endpoint = endpoint_name or default_endpoint(self.settings)
        self.provider.teardown(endpoint)
        deployment = self.status(endpoint)
        if deployment:
            self.store.update(deployment.id, state=DeploymentState.TERMINATED)
            self.store.add_event(deployment.id, "terminated", {})
        audit("deployment.terminate", "deployment", endpoint)


_MANAGER: DeploymentManager | None = None


def get_deployment_manager() -> DeploymentManager:
    global _MANAGER
    if _MANAGER is None:
        _MANAGER = DeploymentManager()
    return _MANAGER


def set_deployment_manager(manager: DeploymentManager | None) -> None:
    global _MANAGER
    _MANAGER = manager
