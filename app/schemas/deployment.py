"""Schemas for deployments, traffic routing and rollback."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator

from app.core.utils import utcnow_iso
from app.schemas.common import DeploymentState, DeploymentStrategy, HealthStatus


class TrafficSplit(BaseModel):
    """Percentage of live traffic per model version. Must total 100."""

    weights: dict[int, float] = Field(default_factory=dict)

    @field_validator("weights")
    @classmethod
    def _sums_to_100(cls, value: dict[int, float]) -> dict[int, float]:
        if not value:
            return value
        total = sum(value.values())
        if abs(total - 100.0) > 0.01:
            raise ValueError(f"traffic weights must sum to 100, got {total}")
        if any(w < 0 for w in value.values()):
            raise ValueError("traffic weights must be non-negative")
        return value

    @classmethod
    def all_to(cls, version: int) -> TrafficSplit:
        return cls(weights={version: 100.0})

    def primary_version(self) -> int | None:
        if not self.weights:
            return None
        return max(self.weights.items(), key=lambda kv: kv[1])[0]

    def as_dict(self) -> dict[str, float]:
        return {str(k): float(v) for k, v in self.weights.items()}


class DeploymentRequest(BaseModel):
    model_name: str | None = None
    model_version: int
    strategy: DeploymentStrategy | None = None
    endpoint_name: str | None = None
    canary_steps: list[int] | None = None
    shadow_of: int | None = Field(
        default=None,
        description="For shadow deployments: the live version whose traffic is mirrored.",
    )
    auto_promote: bool = True
    reason: str = ""


class DeploymentEvent(BaseModel):
    deployment_id: str
    event: str
    detail: dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default_factory=utcnow_iso)


class Deployment(BaseModel):
    """A deployment is the live binding of model versions to an endpoint."""

    id: str
    endpoint_name: str
    provider: str
    strategy: DeploymentStrategy
    state: DeploymentState
    model_name: str
    current_version: int | None = None
    previous_version: int | None = None
    candidate_version: int | None = None
    shadow_version: int | None = None
    traffic: dict[str, float] = Field(default_factory=dict)
    health: HealthStatus = HealthStatus.UNKNOWN
    message: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default_factory=utcnow_iso)
    updated_at: str = Field(default_factory=utcnow_iso)
    events: list[DeploymentEvent] = Field(default_factory=list)

    @property
    def is_live(self) -> bool:
        return self.state == DeploymentState.LIVE

    def traffic_split(self) -> TrafficSplit:
        return TrafficSplit(weights={int(k): float(v) for k, v in self.traffic.items()})


class EndpointHealth(BaseModel):
    endpoint_name: str
    status: HealthStatus
    checks: dict[str, bool] = Field(default_factory=dict)
    latency_p95_ms: float = 0.0
    error_rate: float = 0.0
    request_count: int = 0
    detail: str = ""
    checked_at: str = Field(default_factory=utcnow_iso)


class RollbackRequest(BaseModel):
    endpoint_name: str | None = None
    to_version: int | None = Field(
        default=None, description="Defaults to the recorded previous version."
    )
    reason: str = "manual rollback"


class RollbackResult(BaseModel):
    deployment_id: str
    endpoint_name: str
    rolled_back_from: int | None
    rolled_back_to: int
    reason: str
    succeeded: bool
    message: str = ""
    created_at: str = Field(default_factory=utcnow_iso)


class CanaryStepResult(BaseModel):
    step_index: int
    traffic_percent: int
    requests_observed: int
    error_rate: float
    latency_p95_ms: float
    passed: bool
    reason: str = ""


class DeploymentResult(BaseModel):
    """Return value of every deployment strategy."""

    deployment: Deployment
    succeeded: bool
    strategy: DeploymentStrategy
    steps: list[CanaryStepResult] = Field(default_factory=list)
    rolled_back: bool = False
    message: str = ""
