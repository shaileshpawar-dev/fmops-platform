"""Blue/green deployment.

The candidate ("green") is brought up alongside the incumbent ("blue") and
health-checked while receiving no traffic. Only if it is healthy does traffic
cut over in a single step. If the post-cutover health check fails, traffic
returns to blue immediately.

Trade-off versus canary: the cutover is instant, so a defect that only appears
under real traffic hits 100% of requests for the duration of one health check.
In exchange, there is no long period of split-version traffic, which matters
when two versions writing to the same downstream store would be inconsistent.
"""

from __future__ import annotations

from app.core.exceptions import DeploymentError
from app.core.logging import get_logger
from app.deployment.strategy import Strategy, StrategyContext
from app.schemas.common import DeploymentState, DeploymentStrategy, HealthStatus
from app.schemas.deployment import DeploymentResult, TrafficSplit

logger = get_logger(__name__)


class BlueGreenStrategy(Strategy):
    kind = DeploymentStrategy.BLUE_GREEN

    def execute(self, ctx: StrategyContext) -> DeploymentResult:
        store = ctx.store
        deployment = ctx.deployment
        endpoint = deployment.endpoint_name

        store.update(deployment.id, state=DeploymentState.IN_PROGRESS)
        store.add_event(
            deployment.id,
            "blue_green.staging_green",
            {
                "blue_version": ctx.current_version,
                "green_version": ctx.candidate_version,
            },
        )

        # --- 1. stand up green with zero traffic and probe it -------------- #
        # The provider loads the artifact during apply(), so a green version
        # that cannot be loaded fails here, before any traffic moves.
        if ctx.current_version is not None and ctx.current_version != ctx.candidate_version:
            warmup = TrafficSplit(
                weights={ctx.current_version: 100.0, ctx.candidate_version: 0.0}
            )
        else:
            warmup = TrafficSplit.all_to(ctx.candidate_version)

        try:
            ctx.provider.apply(endpoint, ctx.model_name, warmup, metadata=ctx.metadata)
        except Exception as exc:
            store.update(
                deployment.id,
                state=DeploymentState.FAILED,
                health=HealthStatus.UNHEALTHY,
                message=f"green version failed to come up: {exc}",
            )
            store.add_event(deployment.id, "blue_green.green_failed", {"error": str(exc)})
            raise DeploymentError(
                f"blue/green deployment aborted: the candidate version could not be "
                f"brought up: {exc}",
                endpoint=endpoint,
                candidate_version=ctx.candidate_version,
            ) from exc

        health = ctx.provider.health_check(endpoint, ctx.model_name)
        if health.status != HealthStatus.HEALTHY:
            store.update(
                deployment.id,
                state=DeploymentState.FAILED,
                health=health.status,
                message=f"pre-cutover health check failed: {health.detail}",
            )
            store.add_event(
                deployment.id,
                "blue_green.precutover_health_failed",
                {"checks": health.checks, "detail": health.detail},
            )
            raise DeploymentError(
                "blue/green deployment aborted: the candidate failed its "
                f"pre-cutover health check ({health.detail})",
                endpoint=endpoint,
                checks=health.checks,
            )

        # --- 2. cut over ---------------------------------------------------- #
        cutover = TrafficSplit.all_to(ctx.candidate_version)
        ctx.provider.apply(endpoint, ctx.model_name, cutover, metadata=ctx.metadata)
        store.add_event(
            deployment.id,
            "blue_green.cutover",
            {"from_version": ctx.current_version, "to_version": ctx.candidate_version},
        )

        # --- 3. verify after cutover, roll back if unhealthy ---------------- #
        post = ctx.provider.health_check(endpoint, ctx.model_name)
        if post.status != HealthStatus.HEALTHY:
            if ctx.current_version is not None and ctx.config.auto_rollback:
                ctx.provider.apply(
                    endpoint,
                    ctx.model_name,
                    TrafficSplit.all_to(ctx.current_version),
                    metadata=ctx.metadata,
                )
                store.update(
                    deployment.id,
                    state=DeploymentState.ROLLED_BACK,
                    health=post.status,
                    current_version=ctx.current_version,
                    traffic=TrafficSplit.all_to(ctx.current_version).as_dict(),
                    message="post-cutover health check failed; rolled back to blue",
                )
                store.add_event(
                    deployment.id,
                    "blue_green.rolled_back",
                    {"restored_version": ctx.current_version, "detail": post.detail},
                )
                return DeploymentResult(
                    deployment=store.get(deployment.id),
                    succeeded=False,
                    strategy=self.kind,
                    rolled_back=True,
                    message=(
                        "candidate failed the post-cutover health check; traffic "
                        f"restored to version {ctx.current_version}"
                    ),
                )
            store.update(
                deployment.id,
                state=DeploymentState.FAILED,
                health=post.status,
                message="post-cutover health check failed and no rollback target exists",
            )
            raise DeploymentError(
                "blue/green cutover failed and there is no previous version to "
                "roll back to",
                endpoint=endpoint,
            )

        updated = store.update(
            deployment.id,
            state=DeploymentState.LIVE,
            health=HealthStatus.HEALTHY,
            current_version=ctx.candidate_version,
            previous_version=ctx.current_version,
            candidate_version=None,
            traffic=cutover.as_dict(),
            message="blue/green cutover completed",
        )
        store.add_event(deployment.id, "blue_green.live", {"version": ctx.candidate_version})
        logger.info(
            "deployment.blue_green_completed",
            extra={
                "endpoint": endpoint,
                "from_version": ctx.current_version,
                "to_version": ctx.candidate_version,
            },
        )
        return DeploymentResult(
            deployment=updated,
            succeeded=True,
            strategy=self.kind,
            message=f"version {ctx.candidate_version} is live (blue/green)",
        )
