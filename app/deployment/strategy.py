"""Deployment strategy interface.

A strategy drives a :class:`~app.deployment.base.DeploymentProvider` through a
sequence of traffic states, deciding after each one whether to advance, finish,
or abort and roll back. Strategies never touch infrastructure directly; that
keeps blue/green and canary logic identical whether the provider is in-process
or SageMaker.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Protocol

from app.core.config import DeploymentConfig, Settings, get_settings
from app.core.logging import get_logger
from app.deployment.base import DeploymentProvider, DeploymentStore
from app.schemas.common import DeploymentStrategy
from app.schemas.deployment import Deployment, DeploymentResult

logger = get_logger(__name__)


class HealthProbe(Protocol):
    """Observed serving health for a version over a recent window.

    Implemented by :class:`app.monitoring.metrics.MetricsProbe` in production and
    by a deterministic stub in tests, so canary promotion logic can be tested
    without generating real traffic.
    """

    def observe(
        self, endpoint_name: str, model_name: str, version: int, window_seconds: int
    ) -> tuple[int, float, float]:
        """Return (request_count, error_rate, latency_p95_ms)."""


@dataclass
class StrategyContext:
    """Everything a strategy needs to execute one deployment."""

    deployment: Deployment
    provider: DeploymentProvider
    store: DeploymentStore
    model_name: str
    candidate_version: int
    current_version: int | None
    config: DeploymentConfig
    settings: Settings
    probe: HealthProbe | None = None
    metadata: dict[str, Any] | None = None


class Strategy(ABC):
    """Executes one deployment from candidate to live (or rolled back)."""

    kind: DeploymentStrategy

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    @abstractmethod
    def execute(self, ctx: StrategyContext) -> DeploymentResult: ...
