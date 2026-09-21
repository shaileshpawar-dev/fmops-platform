"""Canary deployment.

Traffic moves to the candidate in configured steps (10% -> 25% -> 50% -> 100%).
After each step the strategy waits for the step window, reads the *observed*
error rate and p95 latency for the candidate version from the metrics probe, and
either advances or aborts.

Two guards matter and are both enforced here:

* **Minimum sample size.** A step that saw fewer than
  ``canary_min_requests_per_step`` requests cannot fail the deployment on error
  rate -- with 3 requests, one error is 33% and means nothing. The step is
  recorded as passed-with-insufficient-data rather than silently treated as
  healthy or as a failure.
* **Rollback on breach.** Exceeding the error-rate or latency budget restores
  100% of traffic to the incumbent immediately, before advancing further.
"""

from __future__ import annotations

import time

from app.core.exceptions import DeploymentError
from app.core.logging import get_logger
from app.deployment.strategy import Strategy, StrategyContext
from app.schemas.common import DeploymentState, DeploymentStrategy, HealthStatus
from app.schemas.deployment import CanaryStepResult, DeploymentResult, TrafficSplit

logger = get_logger(__name__)


class CanaryStrategy(Strategy):
    kind = DeploymentStrategy.CANARY

    def __init__(self, settings=None, sleep=time.sleep) -> None:
        super().__init__(settings)
        # Injected so tests can run the full step sequence instantly.
        self._sleep = sleep

    def execute(self, ctx: StrategyContext) -> DeploymentResult:
        store = ctx.store
        deployment = ctx.deployment
        endpoint = deployment.endpoint_name
        steps = [s for s in (ctx.config.canary_steps or [100]) if 0 < s <= 100]
        if not steps or steps[-1] != 100:
            steps = [*steps, 100]

        store.update(deployment.id, state=DeploymentState.IN_PROGRESS)
        store.add_event(
            deployment.id,
            "canary.started",
            {
                "steps": steps,
                "candidate_version": ctx.candidate_version,
                "baseline_version": ctx.current_version,
            },
        )

        results: list[CanaryStepResult] = []

        for index, percent in enumerate(steps):
            split = self._split(ctx, percent)
            try:
                ctx.provider.apply(endpoint, ctx.model_name, split, metadata=ctx.metadata)
            except Exception as exc:
                store.add_event(
                    deployment.id, "canary.apply_failed", {"step": index, "error": str(exc)}
                )
                return self._abort(
                    ctx,
                    results,
                    f"could not shift traffic to {percent}%: {exc}",
                )

            store.update(deployment.id, traffic=split.as_dict())
            store.add_event(
                deployment.id,
                "canary.step_started",
                {"step": index, "traffic_percent": percent},
            )

            health = ctx.provider.health_check(endpoint, ctx.model_name)
            if health.status != HealthStatus.HEALTHY:
                results.append(
                    CanaryStepResult(
                        step_index=index,
                        traffic_percent=percent,
                        requests_observed=0,
                        error_rate=1.0,
                        latency_p95_ms=0.0,
                        passed=False,
                        reason=f"endpoint health check failed: {health.detail}",
                    )
                )
                return self._abort(ctx, results, "endpoint became unhealthy during canary")

            if percent < 100:
                try:
                    self._sleep(max(0, ctx.config.canary_step_seconds))
                except BaseException as exc:
                    # Cancelled (or interrupted) mid-rollout: never leave traffic
                    # split. Restore the live version, then let the caller see
                    # the interruption.
                    self._interrupt(ctx, percent, exc)
                    raise

            step = self._observe(ctx, index, percent)
            results.append(step)
            store.add_event(
                deployment.id,
                "canary.step_evaluated",
                {
                    "step": index,
                    "traffic_percent": percent,
                    "requests": step.requests_observed,
                    "error_rate": round(step.error_rate, 5),
                    "latency_p95_ms": round(step.latency_p95_ms, 2),
                    "passed": step.passed,
                    "reason": step.reason,
                },
            )
            if not step.passed:
                return self._abort(ctx, results, step.reason)

        final = TrafficSplit.all_to(ctx.candidate_version)
        updated = store.update(
            deployment.id,
            state=DeploymentState.LIVE,
            health=HealthStatus.HEALTHY,
            current_version=ctx.candidate_version,
            previous_version=ctx.current_version,
            candidate_version=None,
            traffic=final.as_dict(),
            message="canary completed at 100% traffic",
        )
        store.add_event(deployment.id, "canary.completed", {"version": ctx.candidate_version})
        logger.info(
            "deployment.canary_completed",
            extra={
                "endpoint": endpoint,
                "version": ctx.candidate_version,
                "steps": len(results),
            },
        )
        return DeploymentResult(
            deployment=updated,
            succeeded=True,
            strategy=self.kind,
            steps=results,
            message=f"version {ctx.candidate_version} is live after {len(results)} canary steps",
        )

    # -- helpers ------------------------------------------------------------- #
    def _split(self, ctx: StrategyContext, percent: int) -> TrafficSplit:
        if percent >= 100 or ctx.current_version is None:
            return TrafficSplit.all_to(ctx.candidate_version)
        if ctx.current_version == ctx.candidate_version:
            return TrafficSplit.all_to(ctx.candidate_version)
        return TrafficSplit(
            weights={
                ctx.current_version: float(100 - percent),
                ctx.candidate_version: float(percent),
            }
        )

    def _observe(self, ctx: StrategyContext, index: int, percent: int) -> CanaryStepResult:
        if ctx.probe is None:
            return CanaryStepResult(
                step_index=index,
                traffic_percent=percent,
                requests_observed=0,
                error_rate=0.0,
                latency_p95_ms=0.0,
                passed=True,
                reason="no metrics probe configured; step accepted without evidence",
            )

        window = max(ctx.config.canary_step_seconds, 60)
        count, error_rate, latency_p95 = ctx.probe.observe(
            ctx.deployment.endpoint_name, ctx.model_name, ctx.candidate_version, window
        )

        if count < ctx.config.canary_min_requests_per_step:
            # Not enough evidence to judge. Advancing on thin data is a known
            # weakness of canary deployments; we record it rather than hide it.
            return CanaryStepResult(
                step_index=index,
                traffic_percent=percent,
                requests_observed=count,
                error_rate=error_rate,
                latency_p95_ms=latency_p95,
                passed=True,
                reason=(
                    f"only {count} requests observed (minimum "
                    f"{ctx.config.canary_min_requests_per_step}); advancing without "
                    "statistically meaningful evidence"
                ),
            )

        if error_rate > ctx.config.canary_max_error_rate:
            return CanaryStepResult(
                step_index=index,
                traffic_percent=percent,
                requests_observed=count,
                error_rate=error_rate,
                latency_p95_ms=latency_p95,
                passed=False,
                reason=(
                    f"error rate {error_rate:.2%} exceeds the budget "
                    f"{ctx.config.canary_max_error_rate:.2%} over {count} requests"
                ),
            )
        if latency_p95 > ctx.config.canary_max_latency_ms:
            return CanaryStepResult(
                step_index=index,
                traffic_percent=percent,
                requests_observed=count,
                error_rate=error_rate,
                latency_p95_ms=latency_p95,
                passed=False,
                reason=(
                    f"p95 latency {latency_p95:.1f}ms exceeds the budget "
                    f"{ctx.config.canary_max_latency_ms:.1f}ms over {count} requests"
                ),
            )
        return CanaryStepResult(
            step_index=index,
            traffic_percent=percent,
            requests_observed=count,
            error_rate=error_rate,
            latency_p95_ms=latency_p95,
            passed=True,
            reason="within error-rate and latency budgets",
        )

    def _interrupt(self, ctx: StrategyContext, percent: int, exc: BaseException) -> None:
        """Put all traffic back on the live version after an interrupted step.

        Unlike a failed step this ignores ``auto_rollback``: a cancellation is
        an operator saying stop, and a half-shifted endpoint is never the state
        they asked for.
        """
        store = ctx.store
        endpoint = ctx.deployment.endpoint_name
        reason = f"rollout interrupted at {percent}% traffic: {exc}"
        if ctx.current_version is None:
            store.update(ctx.deployment.id, state=DeploymentState.FAILED, message=reason)
            store.add_event(ctx.deployment.id, "canary.interrupted", {"reason": reason})
            return
        try:
            ctx.provider.apply(
                endpoint,
                ctx.model_name,
                TrafficSplit.all_to(ctx.current_version),
                metadata=ctx.metadata,
            )
        except Exception as restore_exc:
            store.update(
                ctx.deployment.id,
                state=DeploymentState.FAILED,
                health=HealthStatus.UNHEALTHY,
                message=f"{reason}; restoring v{ctx.current_version} ALSO failed: {restore_exc}",
            )
            return
        store.update(
            ctx.deployment.id,
            state=DeploymentState.ROLLED_BACK,
            traffic=TrafficSplit.all_to(ctx.current_version).as_dict(),
            candidate_version=None,
            message=f"{reason}; all traffic restored to v{ctx.current_version}",
        )
        store.add_event(
            ctx.deployment.id,
            "canary.interrupted",
            {"reason": reason, "restored_version": ctx.current_version},
        )

    def _abort(
        self, ctx: StrategyContext, results: list[CanaryStepResult], reason: str
    ) -> DeploymentResult:
        store = ctx.store
        endpoint = ctx.deployment.endpoint_name

        if ctx.current_version is None or not ctx.config.auto_rollback:
            store.update(
                ctx.deployment.id,
                state=DeploymentState.FAILED,
                health=HealthStatus.UNHEALTHY,
                message=reason,
            )
            store.add_event(ctx.deployment.id, "canary.failed", {"reason": reason})
            raise DeploymentError(
                f"canary deployment failed and could not be rolled back: {reason}",
                endpoint=endpoint,
                candidate_version=ctx.candidate_version,
            )

        restore = TrafficSplit.all_to(ctx.current_version)
        try:
            ctx.provider.apply(endpoint, ctx.model_name, restore, metadata=ctx.metadata)
        except Exception as exc:
            store.update(
                ctx.deployment.id,
                state=DeploymentState.FAILED,
                health=HealthStatus.UNHEALTHY,
                message=f"{reason}; rollback ALSO failed: {exc}",
            )
            raise DeploymentError(
                f"canary failed ({reason}) and the rollback to version "
                f"{ctx.current_version} also failed: {exc}",
                endpoint=endpoint,
            ) from exc

        updated = store.update(
            ctx.deployment.id,
            state=DeploymentState.ROLLED_BACK,
            health=HealthStatus.DEGRADED,
            current_version=ctx.current_version,
            traffic=restore.as_dict(),
            message=f"canary aborted: {reason}",
        )
        store.add_event(
            ctx.deployment.id,
            "canary.rolled_back",
            {"reason": reason, "restored_version": ctx.current_version},
        )
        logger.warning(
            "deployment.canary_rolled_back",
            extra={
                "endpoint": endpoint,
                "candidate_version": ctx.candidate_version,
                "restored_version": ctx.current_version,
                "reason": reason,
            },
        )
        return DeploymentResult(
            deployment=updated,
            succeeded=False,
            strategy=self.kind,
            steps=results,
            rolled_back=True,
            message=f"canary aborted and traffic restored to version {ctx.current_version}: {reason}",
        )
