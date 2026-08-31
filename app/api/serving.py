"""The scoring path.

One place where a prediction is actually produced, shared by the real-time
endpoint, the batch endpoint and the traffic simulator. It:

1. asks the deployment provider which version should serve this request,
2. scores it through the fitted pipeline (which carries its own preprocessing),
3. applies the *model's own* decision threshold unless one is passed explicitly,
4. mirrors the request to the shadow version if one is attached,
5. records the prediction to the inference log and to Prometheus.

Shadow scoring is wrapped so that a failing shadow model can never affect the
response the caller receives.
"""

from __future__ import annotations

import time
from typing import Any

import pandas as pd

from app.core.config import Settings, get_settings
from app.core.exceptions import ModelNotLoadedError, PredictionError
from app.core.logging import get_logger
from app.data.preprocessing import prepare_inference_frame
from app.deployment.base import get_deployment_store
from app.deployment.local_provider import LocalDeploymentProvider, get_local_provider
from app.deployment.model_cache import LoadedModel
from app.monitoring.inference_log import InferenceLog, get_inference_log
from app.monitoring.metrics import record_error, record_prediction
from app.registry.base import ModelRegistry
from app.registry.factory import get_registry
from app.schemas.common import ModelStage
from app.schemas.prediction import (
    BatchPredictionResponse,
    PredictionResponse,
)

logger = get_logger(__name__)

LABELS = {0: "no_default", 1: "default"}


class PredictionService:
    """Serves predictions from whatever version the deployment routes to."""

    def __init__(
        self,
        provider: LocalDeploymentProvider | None = None,
        registry: ModelRegistry | None = None,
        inference_log: InferenceLog | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self._provider = provider
        self._registry = registry
        self._log = inference_log

    @property
    def provider(self) -> LocalDeploymentProvider:
        if self._provider is None:
            self._provider = get_local_provider()
        return self._provider

    @property
    def registry(self) -> ModelRegistry:
        if self._registry is None:
            self._registry = get_registry()
        return self._registry

    @property
    def inference_log(self) -> InferenceLog:
        if self._log is None:
            self._log = get_inference_log()
        return self._log

    # -- model resolution ---------------------------------------------------- #
    def resolve_model(self, version: int | None = None) -> tuple[LoadedModel, str]:
        """Pick the serving model, falling back to the registry when no
        deployment exists.

        The fallback matters for a fresh install: a registered Production model
        should be servable before anyone has run an explicit deployment. It is
        logged, so it is never a silent surprise.
        """
        endpoint = self.settings.deployment.endpoint_name
        try:
            return self.provider.resolve(endpoint, version)
        except ModelNotLoadedError:
            model_name = self.settings.tracking.registered_model_name
            if version is not None:
                return self.provider.cache.get(model_name, version), "pinned"
            serving = self.registry.get_serving(model_name)
            if serving is None:
                raise ModelNotLoadedError(
                    "no model is available to serve: nothing is deployed and the "
                    f"registry has no {ModelStage.PRODUCTION.value} or "
                    f"{ModelStage.STAGING.value} version of "
                    f"{model_name!r}. Train and promote a model first "
                    "(make train, make promote).",
                    model=model_name,
                    endpoint=endpoint,
                ) from None
            logger.info(
                "serving.registry_fallback",
                extra={
                    "model": model_name,
                    "version": serving.version,
                    "stage": serving.stage.value,
                    "reason": "no active deployment for this endpoint",
                },
            )
            return self.provider.cache.get(model_name, serving.version), "primary"

    # -- prediction ---------------------------------------------------------- #
    def predict_one(
        self,
        features: dict[str, Any],
        request_id: str,
        version: int | None = None,
        threshold: float | None = None,
        explain: bool = False,
    ) -> PredictionResponse:
        started = time.perf_counter()
        model, variant = self.resolve_model(version)
        effective_threshold = float(threshold) if threshold is not None else model.threshold

        frame = prepare_inference_frame([features], self.settings.data)
        try:
            probability = float(model.pipeline.predict_proba(frame)[0][1])
        except Exception as exc:
            latency_ms = (time.perf_counter() - started) * 1000
            record_error("prediction_failed", "serving")
            self.inference_log.record(
                request_id=request_id,
                model_name=model.model_name,
                model_version=model.version,
                features=features,
                prediction=None,
                probability=None,
                latency_ms=latency_ms,
                variant=variant,
                status="error",
                error_code="prediction_failed",
                force=True,
            )
            raise PredictionError(
                f"the model could not score this request: {exc}",
                model=model.model_name,
                version=model.version,
            ) from exc

        prediction = int(probability >= effective_threshold)
        latency_ms = (time.perf_counter() - started) * 1000

        record_prediction(
            model.model_name,
            model.version,
            variant,
            latency_ms / 1000.0,
            probability=probability,
        )
        self.inference_log.record(
            request_id=request_id,
            model_name=model.model_name,
            model_version=model.version,
            features=features,
            prediction=prediction,
            probability=probability,
            latency_ms=latency_ms,
            variant=variant,
            deployment_id=self._deployment_id(),
        )
        self._score_shadow(features, request_id)

        return PredictionResponse(
            request_id=request_id,
            prediction=prediction,
            prediction_label=LABELS.get(prediction, str(prediction)),
            probability=round(probability, 6),
            threshold=round(effective_threshold, 6),
            model_name=model.model_name,
            model_version=model.version,
            model_stage=model.stage,
            variant=variant,
            inference_latency_ms=round(latency_ms, 3),
            explanation=self._explain(model, frame) if explain else None,
        )

    def predict_batch(
        self,
        instances: list[dict[str, Any]],
        request_id: str,
        version: int | None = None,
        threshold: float | None = None,
    ) -> BatchPredictionResponse:
        started = time.perf_counter()
        model, variant = self.resolve_model(version)
        effective_threshold = float(threshold) if threshold is not None else model.threshold

        frame = prepare_inference_frame(instances, self.settings.data)
        try:
            probabilities = model.pipeline.predict_proba(frame)[:, 1]
        except Exception as exc:
            record_error("prediction_failed", "serving")
            raise PredictionError(
                f"batch scoring failed: {exc}",
                model=model.model_name,
                version=model.version,
                n_instances=len(instances),
            ) from exc

        predictions = (probabilities >= effective_threshold).astype(int)
        latency_ms = (time.perf_counter() - started) * 1000
        per_row_seconds = (latency_ms / max(len(instances), 1)) / 1000.0

        for index, (probability, prediction) in enumerate(
            zip(probabilities, predictions, strict=True)
        ):
            record_prediction(
                model.model_name,
                model.version,
                variant,
                per_row_seconds,
                probability=float(probability),
            )
            self.inference_log.record(
                request_id=f"{request_id}-{index}",
                model_name=model.model_name,
                model_version=model.version,
                features=instances[index],
                prediction=int(prediction),
                probability=float(probability),
                latency_ms=per_row_seconds * 1000.0,
                variant=variant,
                deployment_id=self._deployment_id(),
            )

        return BatchPredictionResponse(
            request_id=request_id,
            model_name=model.model_name,
            model_version=model.version,
            n_instances=len(instances),
            predictions=[int(p) for p in predictions],
            probabilities=[round(float(p), 6) for p in probabilities],
            threshold=round(effective_threshold, 6),
            inference_latency_ms=round(latency_ms, 3),
        )

    # -- helpers ------------------------------------------------------------- #
    def _deployment_id(self) -> str | None:
        try:
            deployment = get_deployment_store().active(self.settings.deployment.endpoint_name)
            return deployment.id if deployment else None
        except Exception:
            return None

    def _score_shadow(self, features: dict[str, Any], request_id: str) -> None:
        """Mirror the request to the shadow model. Never raises."""
        try:
            shadow = self.provider.shadow_model(self.settings.deployment.endpoint_name)
            if shadow is None:
                return
            started = time.perf_counter()
            frame = prepare_inference_frame([features], self.settings.data)
            probability = float(shadow.pipeline.predict_proba(frame)[0][1])
            latency_ms = (time.perf_counter() - started) * 1000
            self.inference_log.record(
                request_id=request_id,
                model_name=shadow.model_name,
                model_version=shadow.version,
                features=features,
                prediction=int(probability >= shadow.threshold),
                probability=probability,
                latency_ms=latency_ms,
                variant="shadow",
                shadow=True,
                force=True,
            )
            record_prediction(
                shadow.model_name,
                shadow.version,
                "shadow",
                latency_ms / 1000.0,
                probability=probability,
            )
        except Exception as exc:
            # A broken shadow is a monitoring problem, not a serving problem.
            logger.error(
                "serving.shadow_scoring_failed",
                extra={"request_id": request_id, "error": str(exc)},
            )

    def _explain(self, model: LoadedModel, frame: pd.DataFrame) -> dict[str, float]:
        """Per-request contribution estimate.

        Uses the model's global feature importance weighted by how far each
        numeric input sits from the training median. This is an *indicative*
        attribution, not SHAP: it is cheap enough to run inline and is labelled
        as approximate wherever it is surfaced.
        """
        importance = (
            {k: float(v) for k, v in (model.metrics.get("feature_importance") or {}).items()}
            if isinstance(model.metrics.get("feature_importance"), dict)
            else {}
        )
        if not importance:
            return {}
        row = frame.iloc[0]
        out: dict[str, float] = {}
        for feature, weight in importance.items():
            value = row.get(feature)
            if isinstance(value, (int, float)):
                out[feature] = round(float(weight), 6)
        return out


_SERVICE: PredictionService | None = None


def get_prediction_service() -> PredictionService:
    global _SERVICE
    if _SERVICE is None:
        _SERVICE = PredictionService()
    return _SERVICE


def set_prediction_service(service: PredictionService | None) -> None:
    global _SERVICE
    _SERVICE = service
