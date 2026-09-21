"""Platform exception hierarchy.

Every failure mode the platform can hit has a named exception carrying a stable
machine-readable ``code`` and a ``details`` payload.  The API layer maps these
onto HTTP status codes (see :mod:`app.api.errors`) and the pipeline layer maps
them onto pipeline stage failures.  Nothing in the platform is allowed to
swallow one of these silently -- see ``docs/troubleshooting.md``.
"""

from __future__ import annotations

from typing import Any


class FMOpsError(Exception):
    """Base class for every error raised deliberately by the platform."""

    code: str = "fmops_error"
    http_status: int = 500

    def __init__(self, message: str, **details: Any) -> None:
        super().__init__(message)
        self.message = message
        self.details: dict[str, Any] = details

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code, "message": self.message, "details": self.details}

    def __str__(self) -> str:  # pragma: no cover - trivial
        if self.details:
            return f"{self.message} ({self.details})"
        return self.message


# --------------------------------------------------------------------------- #
# Configuration / environment
# --------------------------------------------------------------------------- #
class ConfigurationError(FMOpsError):
    code = "configuration_error"
    http_status = 500


class ProviderUnavailableError(FMOpsError):
    """A backing provider (AWS, LLM vendor, MLflow server) is not reachable."""

    code = "provider_unavailable"
    http_status = 503


class DependencyMissingError(FMOpsError):
    """An optional dependency is required for the requested backend."""

    code = "dependency_missing"
    http_status = 501


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
class DataError(FMOpsError):
    code = "data_error"
    http_status = 400


class DataValidationError(DataError):
    """Raised when a dataset fails a blocking expectation suite."""

    code = "data_validation_failed"
    http_status = 422


class DatasetNotFoundError(DataError):
    code = "dataset_not_found"
    http_status = 404


# --------------------------------------------------------------------------- #
# Training / evaluation
# --------------------------------------------------------------------------- #
class TrainingError(FMOpsError):
    code = "training_failed"


class EvaluationError(FMOpsError):
    code = "evaluation_failed"


class TuningError(FMOpsError):
    code = "tuning_failed"


# --------------------------------------------------------------------------- #
# Registry / approval
# --------------------------------------------------------------------------- #
class RegistryError(FMOpsError):
    code = "registry_error"


class ModelNotFoundError(RegistryError):
    code = "model_not_found"
    http_status = 404


class InvalidStageTransitionError(RegistryError):
    code = "invalid_stage_transition"
    http_status = 409


class ApprovalRejectedError(FMOpsError):
    """The candidate model did not clear the configured approval gate."""

    code = "approval_rejected"
    http_status = 409


# --------------------------------------------------------------------------- #
# Deployment / serving
# --------------------------------------------------------------------------- #
class DeploymentError(FMOpsError):
    code = "deployment_failed"


class DeploymentNotFoundError(DeploymentError):
    code = "deployment_not_found"
    http_status = 404


class EndpointUnhealthyError(DeploymentError):
    code = "endpoint_unhealthy"
    http_status = 503


class RollbackError(DeploymentError):
    code = "rollback_failed"


class ModelNotLoadedError(FMOpsError):
    """Inference was requested but no model is currently serving."""

    code = "model_not_loaded"
    http_status = 503


class PredictionError(FMOpsError):
    code = "prediction_failed"
    http_status = 400


# --------------------------------------------------------------------------- #
# Monitoring / drift / retraining
# --------------------------------------------------------------------------- #
class InvalidPredictionInputError(PredictionError):
    """The request does not match the serving version's input contract."""

    code = "invalid_prediction_input"
    http_status = 422


class PredictionNotFoundError(PredictionError):
    """Feedback for a request the platform never served."""

    code = "prediction_not_found"
    http_status = 404


class MonitoringError(FMOpsError):
    code = "monitoring_error"


class DriftError(FMOpsError):
    code = "drift_error"


class InsufficientDataError(FMOpsError):
    """Not enough observations to compute a statistic honestly."""

    code = "insufficient_data"
    http_status = 409


class RetrainingError(FMOpsError):
    code = "retraining_failed"


# --------------------------------------------------------------------------- #
# LLMOps
# --------------------------------------------------------------------------- #
class LLMError(FMOpsError):
    code = "llm_error"


class LLMProviderError(LLMError):
    code = "llm_provider_error"
    http_status = 502


class LLMRateLimitError(LLMProviderError):
    code = "llm_rate_limited"
    http_status = 429


class PromptNotFoundError(LLMError):
    code = "prompt_not_found"
    http_status = 404


class SafetyViolationError(LLMError):
    code = "safety_violation"
    http_status = 422


# --------------------------------------------------------------------------- #
# Security
# --------------------------------------------------------------------------- #
class AuthenticationError(FMOpsError):
    code = "authentication_failed"
    http_status = 401


class AuthorizationError(FMOpsError):
    code = "authorization_failed"
    http_status = 403
