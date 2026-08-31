"""Direct deployment.

Cuts 100% of traffic to the candidate in one step with no warm-up stage and no
progressive rollout. Appropriate for development and tests; the production
config uses canary instead. It is kept as a first-class strategy rather than a
special case so the audit trail always names the strategy that was used.
"""

from __future__ import annotations

from app.core.exceptions import DeploymentError
from app.core.logging import get_logger
from app.deployment.strategy import Strategy, StrategyContext
from app.schemas.common import DeploymentState, DeploymentStrategy, HealthStatus
from app.schemas.deployment import DeploymentResult, TrafficSplit

logger = get_logger(__name__)


class DirectStrategy(Strategy):
    kind = DeploymentStrategy.DIRECT

    def execute(self, ctx: StrategyContext) -> DeploymentResult:
        store = ctx.store
        deployment = ctx.deployment
        endpoint = deployment.endpoint_name

        store.update(deployment.id, state=DeploymentState.IN_PROGRESS)
        split = TrafficSplit.all_to(ctx.candidate_version)

        try:
            ctx.provider.apply(endpoint, ctx.model_name, split, metadata=ctx.metadata)
        except Exception as exc:
            store.update(
                deployment.id,
                state=DeploymentState.FAILED,
                health=HealthStatus.UNHEALTHY,
                message=f"direct deployment failed: {exc}",
            )
            store.add_event(deployment.id, "direct.failed", {"error": str(exc)})
            raise DeploymentError(
                f"direct deployment of version {ctx.candidate_version} failed: {exc}",
                endpoint=endpoint,
                candidate_version=ctx.candidate_version,
            ) from exc

        health = ctx.provider.health_check(endpoint, ctx.model_name)
        updated = store.update(
            deployment.id,
            state=(
                DeploymentState.LIVE
                if health.status == HealthStatus.HEALTHY
                else DeploymentState.FAILED
            ),
            health=health.status,
            current_version=ctx.candidate_version,
            previous_version=ctx.current_version,
            candidate_version=None,
            traffic=split.as_dict(),
            message=health.detail or "direct deployment completed",
        )
        store.add_event(
            deployment.id,
            "direct.applied",
            {"version": ctx.candidate_version, "health": health.status.value},
        )
        succeeded = health.status == HealthStatus.HEALTHY
        logger.info(
            "deployment.direct_completed",
            extra={
                "endpoint": endpoint,
                "version": ctx.candidate_version,
                "healthy": succeeded,
            },
        )
        return DeploymentResult(
            deployment=updated,
            succeeded=succeeded,
            strategy=self.kind,
            message=(
                f"version {ctx.candidate_version} is live"
                if succeeded
                else f"deployed but unhealthy: {health.detail}"
            ),
        )
