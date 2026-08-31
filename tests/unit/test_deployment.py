"""Deployment strategy and rollback tests.

These use a stub provider so strategy logic is tested without loading real
models, plus a deterministic health probe so canary promotion decisions are
reproducible instead of depending on generated traffic.
"""

from __future__ import annotations

from typing import Any

import pytest

from app.core.exceptions import DeploymentError, RollbackError
from app.deployment.base import DeploymentProvider, DeploymentStore
from app.deployment.blue_green import BlueGreenStrategy
from app.deployment.canary import CanaryStrategy
from app.deployment.direct import DirectStrategy
from app.deployment.rollback import RollbackManager
from app.deployment.shadow import ShadowStrategy
from app.deployment.strategy import StrategyContext
from app.schemas.common import (
    DeploymentState,
    DeploymentStrategy,
    HealthStatus,
    ModelStage,
)
from app.schemas.deployment import EndpointHealth, TrafficSplit

pytestmark = pytest.mark.unit


class StubProvider(DeploymentProvider):
    """Records what it was asked to do; can be told to fail or be unhealthy."""

    name = "stub"

    def __init__(self, healthy: bool = True, fail_on_apply: bool = False) -> None:
        self.healthy = healthy
        self.fail_on_apply = fail_on_apply
        self.applied: list[dict[str, Any]] = []
        self.torn_down: list[str] = []

    def apply(self, endpoint_name, model_name, traffic, shadow_version=None, metadata=None):
        if self.fail_on_apply:
            raise RuntimeError("provider refused the traffic change")
        self.applied.append(
            {
                "endpoint": endpoint_name,
                "traffic": traffic.as_dict(),
                "shadow_version": shadow_version,
            }
        )
        return {"provider": self.name}

    def status(self, endpoint_name):
        return {"provider": self.name, "applied": len(self.applied)}

    def health_check(self, endpoint_name, model_name):
        return EndpointHealth(
            endpoint_name=endpoint_name,
            status=HealthStatus.HEALTHY if self.healthy else HealthStatus.UNHEALTHY,
            checks={"stub": self.healthy},
            detail="" if self.healthy else "stub provider was told to be unhealthy",
        )

    def teardown(self, endpoint_name):
        self.torn_down.append(endpoint_name)


class StubProbe:
    """Deterministic canary evidence."""

    def __init__(self, count=100, error_rate=0.0, latency_p95=50.0) -> None:
        self.count = count
        self.error_rate = error_rate
        self.latency_p95 = latency_p95
        self.calls = 0

    def observe(self, endpoint_name, model_name, version, window_seconds):
        self.calls += 1
        return self.count, self.error_rate, self.latency_p95


@pytest.fixture
def store(db) -> DeploymentStore:
    return DeploymentStore(db)


def make_context(
    store, provider, settings, current=None, candidate=2, probe=None, config=None
):
    deployment = store.create(
        endpoint_name="test-endpoint",
        provider=provider.name,
        strategy=DeploymentStrategy.DIRECT,
        model_name="m",
        candidate_version=candidate,
        current_version=current,
    )
    return StrategyContext(
        deployment=deployment,
        provider=provider,
        store=store,
        model_name="m",
        candidate_version=candidate,
        current_version=current,
        config=config or settings.deployment,
        settings=settings,
        probe=probe,
    )


# --------------------------------------------------------------------------- #
# Traffic split
# --------------------------------------------------------------------------- #
def test_traffic_must_sum_to_100():
    with pytest.raises(ValueError, match="must sum to 100"):
        TrafficSplit(weights={1: 60.0, 2: 30.0})


def test_negative_weights_are_rejected():
    with pytest.raises(ValueError):
        TrafficSplit(weights={1: 120.0, 2: -20.0})


def test_primary_version_is_the_heaviest():
    split = TrafficSplit(weights={1: 90.0, 2: 10.0})
    assert split.primary_version() == 1
    assert TrafficSplit.all_to(7).weights == {7: 100.0}


# --------------------------------------------------------------------------- #
# Direct
# --------------------------------------------------------------------------- #
def test_direct_sends_all_traffic_to_the_candidate(store, settings):
    provider = StubProvider()
    ctx = make_context(store, provider, settings, current=1, candidate=2)
    result = DirectStrategy(settings).execute(ctx)

    assert result.succeeded
    assert provider.applied[-1]["traffic"] == {"2": 100.0}
    assert result.deployment.state == DeploymentState.LIVE
    assert result.deployment.previous_version == 1


def test_direct_raises_when_the_provider_fails(store, settings):
    provider = StubProvider(fail_on_apply=True)
    ctx = make_context(store, provider, settings, current=1, candidate=2)
    with pytest.raises(DeploymentError):
        DirectStrategy(settings).execute(ctx)
    assert store.get(ctx.deployment.id).state == DeploymentState.FAILED


# --------------------------------------------------------------------------- #
# Blue / green
# --------------------------------------------------------------------------- #
def test_blue_green_warms_up_before_cutting_over(store, settings):
    provider = StubProvider()
    ctx = make_context(store, provider, settings, current=1, candidate=2)
    result = BlueGreenStrategy(settings).execute(ctx)

    assert result.succeeded
    assert len(provider.applied) == 2, "expected a warm-up apply then a cutover"
    # Green is staged at zero traffic first: blue still serves everything.
    assert provider.applied[0]["traffic"] == {"1": 100.0, "2": 0.0}
    assert provider.applied[1]["traffic"] == {"2": 100.0}


def test_blue_green_aborts_when_green_is_unhealthy(store, settings):
    provider = StubProvider(healthy=False)
    ctx = make_context(store, provider, settings, current=1, candidate=2)
    with pytest.raises(DeploymentError, match="pre-cutover health check"):
        BlueGreenStrategy(settings).execute(ctx)
    # Traffic never moved off blue.
    assert provider.applied[-1]["traffic"] == {"1": 100.0, "2": 0.0}


def test_blue_green_with_no_incumbent_deploys_directly(store, settings):
    provider = StubProvider()
    ctx = make_context(store, provider, settings, current=None, candidate=1)
    result = BlueGreenStrategy(settings).execute(ctx)
    assert result.succeeded
    assert provider.applied[0]["traffic"] == {"1": 100.0}


# --------------------------------------------------------------------------- #
# Canary
# --------------------------------------------------------------------------- #
def test_canary_advances_through_every_step(store, settings):
    provider = StubProvider()
    config = settings.deployment.model_copy(
        update={
            "canary_steps": [10, 50, 100],
            "canary_step_seconds": 0,
            "canary_min_requests_per_step": 10,
        }
    )
    probe = StubProbe(count=200, error_rate=0.001, latency_p95=40.0)
    ctx = make_context(
        store, provider, settings, current=1, candidate=2, probe=probe, config=config
    )

    result = CanaryStrategy(settings, sleep=lambda _s: None).execute(ctx)

    assert result.succeeded
    assert len(result.steps) == 3
    assert all(step.passed for step in result.steps)
    splits = [a["traffic"] for a in provider.applied]
    assert splits[0] == {"1": 90.0, "2": 10.0}
    assert splits[1] == {"1": 50.0, "2": 50.0}
    assert splits[2] == {"2": 100.0}


def test_canary_rolls_back_on_high_error_rate(store, settings):
    provider = StubProvider()
    config = settings.deployment.model_copy(
        update={
            "canary_steps": [10, 50, 100],
            "canary_step_seconds": 0,
            "canary_min_requests_per_step": 10,
            "canary_max_error_rate": 0.05,
            "auto_rollback": True,
        }
    )
    probe = StubProbe(count=200, error_rate=0.35, latency_p95=40.0)
    ctx = make_context(
        store, provider, settings, current=1, candidate=2, probe=probe, config=config
    )

    result = CanaryStrategy(settings, sleep=lambda _s: None).execute(ctx)

    assert not result.succeeded
    assert result.rolled_back
    assert provider.applied[-1]["traffic"] == {
        "1": 100.0
    }, "traffic must return to the incumbent"
    assert result.deployment.state == DeploymentState.ROLLED_BACK
    assert "error rate" in result.steps[-1].reason


def test_canary_rolls_back_on_latency_breach(store, settings):
    provider = StubProvider()
    config = settings.deployment.model_copy(
        update={
            "canary_steps": [25, 100],
            "canary_step_seconds": 0,
            "canary_min_requests_per_step": 10,
            "canary_max_latency_ms": 100.0,
            "auto_rollback": True,
        }
    )
    probe = StubProbe(count=200, error_rate=0.0, latency_p95=800.0)
    ctx = make_context(
        store, provider, settings, current=1, candidate=2, probe=probe, config=config
    )

    result = CanaryStrategy(settings, sleep=lambda _s: None).execute(ctx)
    assert result.rolled_back
    assert "latency" in result.steps[-1].reason


def test_canary_flags_insufficient_evidence_rather_than_failing(store, settings):
    """Three requests cannot prove anything; the step must say so explicitly."""
    provider = StubProvider()
    config = settings.deployment.model_copy(
        update={
            "canary_steps": [50, 100],
            "canary_step_seconds": 0,
            "canary_min_requests_per_step": 100,
        }
    )
    probe = StubProbe(count=3, error_rate=0.33, latency_p95=40.0)
    ctx = make_context(
        store, provider, settings, current=1, candidate=2, probe=probe, config=config
    )

    result = CanaryStrategy(settings, sleep=lambda _s: None).execute(ctx)

    assert result.succeeded
    assert all(step.passed for step in result.steps)
    assert "statistically meaningful" in result.steps[0].reason


def test_canary_without_rollback_target_raises(store, settings):
    provider = StubProvider()
    config = settings.deployment.model_copy(
        update={
            "canary_steps": [50, 100],
            "canary_step_seconds": 0,
            "canary_min_requests_per_step": 10,
            "canary_max_error_rate": 0.01,
        }
    )
    probe = StubProbe(count=200, error_rate=0.9)
    ctx = make_context(
        store, provider, settings, current=None, candidate=1, probe=probe, config=config
    )

    with pytest.raises(DeploymentError, match="could not be rolled back"):
        CanaryStrategy(settings, sleep=lambda _s: None).execute(ctx)


# --------------------------------------------------------------------------- #
# Shadow
# --------------------------------------------------------------------------- #
def test_shadow_keeps_the_incumbent_serving(store, settings):
    provider = StubProvider()
    ctx = make_context(store, provider, settings, current=1, candidate=2)
    result = ShadowStrategy(settings).execute(ctx)

    assert result.succeeded
    # All live traffic still goes to v1; v2 only mirrors.
    assert provider.applied[-1]["traffic"] == {"1": 100.0}
    assert provider.applied[-1]["shadow_version"] == 2
    assert result.deployment.shadow_version == 2
    assert result.deployment.current_version == 1


def test_shadow_refuses_without_a_live_version(store, settings):
    provider = StubProvider()
    ctx = make_context(store, provider, settings, current=None, candidate=1)
    result = ShadowStrategy(settings).execute(ctx)

    assert not result.succeeded
    assert "requires a live version" in result.message


# --------------------------------------------------------------------------- #
# Rollback
# --------------------------------------------------------------------------- #
def test_rollback_restores_the_previous_version(store, registry, settings):
    for artifact in ("file:///a", "file:///b"):
        registry.register("m", artifact, metrics={"roc_auc": 0.9})
    for stage in (ModelStage.VALIDATION, ModelStage.STAGING, ModelStage.PRODUCTION):
        registry.transition_stage("m", 1, stage)
    for stage in (ModelStage.VALIDATION, ModelStage.STAGING, ModelStage.PRODUCTION):
        registry.transition_stage("m", 2, stage)

    provider = StubProvider()
    deployment = store.create(
        endpoint_name="test-endpoint",
        provider=provider.name,
        strategy=DeploymentStrategy.BLUE_GREEN,
        model_name="m",
        candidate_version=2,
        current_version=2,
        previous_version=1,
    )
    store.update(deployment.id, state=DeploymentState.LIVE, traffic={"2": 100.0})

    manager = RollbackManager(provider, store, registry, settings)
    result = manager.rollback("test-endpoint", reason="error rate spike")

    assert result.succeeded
    assert result.rolled_back_from == 2
    assert result.rolled_back_to == 1
    assert provider.applied[-1]["traffic"] == {"1": 100.0}
    # The registry must agree with what is actually serving.
    assert registry.get_production("m").version == 1
    assert registry.get("m", 2).stage == ModelStage.STAGING


def test_rollback_to_an_explicit_version(store, registry, settings):
    for _ in range(3):
        registry.register("m", "file:///x", metrics={"roc_auc": 0.9})
    provider = StubProvider()
    deployment = store.create(
        endpoint_name="e",
        provider="stub",
        strategy=DeploymentStrategy.DIRECT,
        model_name="m",
        candidate_version=3,
        current_version=3,
    )
    store.update(deployment.id, state=DeploymentState.LIVE, traffic={"3": 100.0})

    result = RollbackManager(provider, store, registry, settings).rollback("e", to_version=1)
    assert result.rolled_back_to == 1


def test_rollback_without_a_target_raises(store, registry, settings):
    registry.register("m", "file:///only", metrics={"roc_auc": 0.9})
    provider = StubProvider()
    deployment = store.create(
        endpoint_name="e",
        provider="stub",
        strategy=DeploymentStrategy.DIRECT,
        model_name="m",
        candidate_version=1,
        current_version=1,
    )
    store.update(deployment.id, state=DeploymentState.LIVE, traffic={"1": 100.0})

    with pytest.raises(RollbackError, match="no rollback target"):
        RollbackManager(provider, store, registry, settings).rollback("e")


def test_rollback_with_no_deployment_raises(store, registry, settings):
    with pytest.raises(RollbackError, match="no deployment exists"):
        RollbackManager(StubProvider(), store, registry, settings).rollback("nonexistent")


def test_failed_rollback_is_reported_not_swallowed(store, registry, settings):
    registry.register("m", "file:///a", metrics={"roc_auc": 0.9})
    registry.register("m", "file:///b", metrics={"roc_auc": 0.9})
    provider = StubProvider(fail_on_apply=True)
    deployment = store.create(
        endpoint_name="e",
        provider="stub",
        strategy=DeploymentStrategy.DIRECT,
        model_name="m",
        candidate_version=2,
        current_version=2,
        previous_version=1,
    )
    store.update(deployment.id, state=DeploymentState.LIVE, traffic={"2": 100.0})

    with pytest.raises(RollbackError, match="could not roll back"):
        RollbackManager(provider, store, registry, settings).rollback("e")
    assert store.get(deployment.id).state == DeploymentState.FAILED


# --------------------------------------------------------------------------- #
# Store
# --------------------------------------------------------------------------- #
def test_deployment_events_form_a_timeline(store, settings):
    provider = StubProvider()
    ctx = make_context(store, provider, settings, current=1, candidate=2)
    BlueGreenStrategy(settings).execute(ctx)

    events = [e.event for e in store.events(ctx.deployment.id)]
    assert "created" in events
    assert "blue_green.staging_green" in events
    assert "blue_green.cutover" in events
    assert "blue_green.live" in events
