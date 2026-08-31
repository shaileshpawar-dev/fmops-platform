"""Registry selection."""

from __future__ import annotations

from app.core.config import Settings, get_settings
from app.core.exceptions import DependencyMissingError, ProviderUnavailableError
from app.core.logging import get_logger
from app.registry.base import ModelRegistry
from app.registry.local import LocalModelRegistry

logger = get_logger(__name__)

_REGISTRY: ModelRegistry | None = None


def build_registry(
    settings: Settings | None = None, allow_fallback: bool = True
) -> ModelRegistry:
    settings = settings or get_settings()
    if settings.tracking.registry_backend == "local":
        return LocalModelRegistry()

    from app.registry.mlflow_registry import MLflowModelRegistry

    try:
        return MLflowModelRegistry(settings)
    except (DependencyMissingError, ProviderUnavailableError) as exc:
        if not allow_fallback:
            raise
        logger.error(
            "registry.mlflow_unavailable_falling_back",
            extra={
                "error": str(exc),
                "fallback": "local",
                "impact": "model versions will NOT appear in the MLflow registry",
            },
        )
        return LocalModelRegistry()


def get_registry() -> ModelRegistry:
    """Process-wide registry instance."""
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = build_registry()
    return _REGISTRY


def set_registry(registry: ModelRegistry | None) -> None:
    """Override the process-wide registry (used by tests)."""
    global _REGISTRY
    _REGISTRY = registry
