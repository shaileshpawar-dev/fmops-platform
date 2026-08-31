"""Shadow deployment.

The candidate receives a mirrored copy of live traffic but its predictions are
**never returned to the caller** and never affect the response. This is how you
get production-traffic evidence about a new model with zero blast radius.

What shadow mode can and cannot tell you:

* It **can** show latency under real traffic shapes, error/exception rates on
  real inputs, and how far the candidate's prediction distribution diverges from
  the incumbent's.
* It **cannot** tell you the candidate is more *accurate*, because shadow
  traffic is unlabelled at the moment of scoring. Accuracy only becomes
  measurable once ground truth arrives through the feedback endpoint, and the
  agreement rate reported here is divergence from the incumbent, not correctness.

Shadow scoring runs inside the request path but is fully isolated: any exception
is logged and swallowed so a broken candidate cannot degrade production.
"""

from __future__ import annotations

from app.core.logging import get_logger
from app.deployment.strategy import Strategy, StrategyContext
from app.schemas.common import DeploymentState, DeploymentStrategy, HealthStatus
from app.schemas.deployment import DeploymentResult, TrafficSplit

logger = get_logger(__name__)


class ShadowStrategy(Strategy):
    kind = DeploymentStrategy.SHADOW

    def execute(self, ctx: StrategyContext) -> DeploymentResult:
        store = ctx.store
        deployment = ctx.deployment
        endpoint = deployment.endpoint_name

        if ctx.current_version is None:
            # Shadowing needs something to shadow. Rather than quietly promoting
            # the candidate to primary, we refuse and say why.
            store.update(
                deployment.id,
                state=DeploymentState.FAILED,
                message="shadow deployment requires an existing live version",
            )
            store.add_event(deployment.id, "shadow.no_primary", {})
            return DeploymentResult(
                deployment=store.get(deployment.id),
                succeeded=False,
                strategy=self.kind,
                message=(
                    "shadow deployment requires a live version to mirror; deploy a "
                    "primary version first (blue/green or direct)"
                ),
            )

        store.update(deployment.id, state=DeploymentState.IN_PROGRESS)

        live = TrafficSplit.all_to(ctx.current_version)
        try:
            ctx.provider.apply(
                endpoint,
                ctx.model_name,
                live,
                shadow_version=ctx.candidate_version,
                metadata=ctx.metadata,
            )
        except Exception as exc:
            store.update(
                deployment.id,
                state=DeploymentState.FAILED,
                health=HealthStatus.UNHEALTHY,
                message=f"could not attach shadow version: {exc}",
            )
            store.add_event(deployment.id, "shadow.attach_failed", {"error": str(exc)})
            return DeploymentResult(
                deployment=store.get(deployment.id),
                succeeded=False,
                strategy=self.kind,
                message=f"shadow version {ctx.candidate_version} could not be loaded: {exc}",
            )

        updated = store.update(
            deployment.id,
            state=DeploymentState.LIVE,
            health=HealthStatus.HEALTHY,
            current_version=ctx.current_version,
            shadow_version=ctx.candidate_version,
            candidate_version=ctx.candidate_version,
            traffic=live.as_dict(),
            message=(
                f"version {ctx.candidate_version} is shadowing version "
                f"{ctx.current_version}; its predictions are recorded but not served"
            ),
        )
        store.add_event(
            deployment.id,
            "shadow.attached",
            {
                "shadow_version": ctx.candidate_version,
                "primary_version": ctx.current_version,
                "sample_rate": ctx.config.shadow_sample_rate,
            },
        )
        logger.info(
            "deployment.shadow_attached",
            extra={
                "endpoint": endpoint,
                "primary_version": ctx.current_version,
                "shadow_version": ctx.candidate_version,
            },
        )
        return DeploymentResult(
            deployment=updated,
            succeeded=True,
            strategy=self.kind,
            message=(
                f"version {ctx.candidate_version} now shadows production version "
                f"{ctx.current_version}; predictions are logged, not returned"
            ),
        )
