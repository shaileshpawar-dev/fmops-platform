"""Tracker selection.

Honest degradation rule: if MLflow is configured but genuinely unreachable, we
fall back to the local tracker *and say so loudly*. A training run that cannot
be tracked is still better than no run, but the operator must know the run will
not appear in MLflow.
"""

from __future__ import annotations

from app.core.config import Settings, get_settings
from app.core.exceptions import DependencyMissingError, ProviderUnavailableError
from app.core.logging import get_logger
from app.tracking.base import ExperimentTracker
from app.tracking.local import LocalFileTracker

logger = get_logger(__name__)


def build_tracker(
    settings: Settings | None = None, allow_fallback: bool = True
) -> ExperimentTracker:
    settings = settings or get_settings()
    if settings.tracking.backend == "local":
        return LocalFileTracker(settings)

    from app.tracking.mlflow_tracker import MLflowTracker

    try:
        return MLflowTracker(settings)
    except (DependencyMissingError, ProviderUnavailableError) as exc:
        if not allow_fallback:
            raise
        logger.error(
            "tracking.mlflow_unavailable_falling_back",
            extra={
                "error": str(exc),
                "tracking_uri": settings.mlflow_tracking_uri,
                "fallback": "local",
                "impact": "runs will NOT appear in MLflow for this session",
            },
        )
        return LocalFileTracker(settings)
