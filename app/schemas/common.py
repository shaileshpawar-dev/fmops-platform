"""Shared enums and small value objects used across schema modules."""

from __future__ import annotations

from enum import Enum


class ModelStage(str, Enum):
    """Lifecycle stages a registered model version moves through.

    The legal transitions are enforced by the registry
    (:data:`app.registry.base.ALLOWED_TRANSITIONS`).
    """

    DEVELOPMENT = "Development"
    VALIDATION = "Validation"
    STAGING = "Staging"
    PRODUCTION = "Production"
    ARCHIVED = "Archived"


class ModelStatus(str, Enum):
    PENDING = "pending"
    READY = "ready"
    APPROVED = "approved"
    REJECTED = "rejected"
    FAILED = "failed"


class DeploymentState(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    LIVE = "live"
    ROLLING_BACK = "rolling_back"
    ROLLED_BACK = "rolled_back"
    FAILED = "failed"
    TERMINATED = "terminated"


class DeploymentStrategy(str, Enum):
    DIRECT = "direct"
    BLUE_GREEN = "blue_green"
    CANARY = "canary"
    SHADOW = "shadow"


class HealthStatus(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class Severity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class AlertCategory(str, Enum):
    DATA_QUALITY = "data_quality"
    DRIFT = "drift"
    PERFORMANCE = "performance"
    LATENCY = "latency"
    ERROR_RATE = "error_rate"
    DEPLOYMENT = "deployment"
    RETRAINING = "retraining"
    COST = "cost"
    SAFETY = "safety"
    RESOURCE = "resource"


class DriftType(str, Enum):
    DATA = "data"
    FEATURE = "feature"
    PREDICTION = "prediction"
    CONCEPT = "concept"


class RetrainingTrigger(str, Enum):
    DRIFT = "drift"
    PERFORMANCE = "performance"
    SCHEDULE = "schedule"
    MANUAL = "manual"
    VOLUME = "volume"


class RetrainingStatus(str, Enum):
    CREATED = "created"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    REJECTED = "rejected"
    SKIPPED = "skipped"


class ValidationSeverity(str, Enum):
    ERROR = "error"
    WARNING = "warning"


class ApprovalDecision(str, Enum):
    APPROVED = "approved"
    REJECTED = "rejected"
    PENDING_MANUAL = "pending_manual"
