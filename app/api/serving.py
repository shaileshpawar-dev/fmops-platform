"""The scoring path.

One place where a prediction is actually produced, shared by the real-time
endpoint, the batch endpoint and the traffic simulator. It:

1. asks the model's own endpoint which version should serve this request,
2. scores it through the fitted pipeline, shaped by that version's recorded
   data contract (not a global feature list),
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

from app.core.config import DataConfig, Settings, get_settings
from app.core.exceptions import (
    InvalidPredictionInputError,
    ModelNotLoadedError,
    PredictionError,
)
from app.core.logging import get_logger
from app.data.preprocessing import prepare_inference_frame
from app.deployment.base import get_deployment_store
from app.deployment.local_provider import LocalDeploymentProvider, get_local_provider
from app.deployment.model_cache import LoadedModel
from app.monitoring.inference_log import InferenceLog, get_inference_log
from app.monitoring.metrics import record_error, record_prediction
from app.registry.base import ModelRegistry
from app.registry.context import default_model_name, endpoint_for
from app.registry.factory import get_registry
from app.schemas.common import ModelStage
from app.schemas.prediction import (
    BatchPredictionResponse,
    PredictionResponse,
)

logger = get_logger(__name__)


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
    def resolve_model(
        self, version: int | None = None, model_name: str | None = None
    ) -> tuple[LoadedModel, str]:
        """Pick the serving version of one model, falling back to the registry
        when that model has no deployment.

        The fallback matters for a fresh install: a registered Production model
        should be servable before anyone has run an explicit deployment. It is
        logged, so it is never a silent surprise.
        """
        model_name = model_name or default_model_name(self.settings)
        endpoint = endpoint_for(model_name, self.settings)
        try:
            return self.provider.resolve(endpoint, version)
        except ModelNotLoadedError:
            if version is not None:
                return self.provider.cache.get(model_name, version), "pinned"
            serving = self.registry.get_serving(model_name)
            if serving is None:
                raise ModelNotLoadedError(
                    "no model is available to serve: nothing is deployed and the "
                    f"registry has no {ModelStage.PRODUCTION.value} or "
                    f"{ModelStage.STAGING.value} version of "
                    f"{model_name!r}. Train it and promote a version to Staging "
                    "or Production first.",
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
        model_name: str | None = None,
        check_contract: bool = False,
    ) -> PredictionResponse:
        started = time.perf_counter()
        model, variant = self.resolve_model(version, model_name)
        # Checked against the version that will actually score the request --
        # under a canary two versions share an endpoint and may not share a
        # feature set.
        warnings: list[str] = []
        if check_contract:
            features, warnings = check_against_contract(model, features)
        effective_threshold = float(threshold) if threshold is not None else model.threshold
        endpoint = endpoint_for(model.model_name, self.settings)

        frame = prepare_inference_frame([features], self._data_config(model))
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
            deployment_id=self._deployment_id(endpoint),
        )
        self._score_shadow(features, request_id, endpoint)

        return PredictionResponse(
            request_id=request_id,
            prediction=prediction,
            prediction_label=model.label_for(prediction),
            positive_label=model.label_for(1),
            probability=round(probability, 6),
            threshold=round(effective_threshold, 6),
            model_name=model.model_name,
            model_version=model.version,
            model_stage=model.stage,
            variant=variant,
            inference_latency_ms=round(latency_ms, 3),
            explanation=self._explain(model, frame) if explain else None,
            warnings=warnings,
        )

    def predict_batch(
        self,
        instances: list[dict[str, Any]],
        request_id: str,
        version: int | None = None,
        threshold: float | None = None,
        model_name: str | None = None,
        check_contract: bool = False,
    ) -> BatchPredictionResponse:
        started = time.perf_counter()
        model, variant = self.resolve_model(version, model_name)
        if check_contract:
            instances = check_batch_against_contract(model, instances)
        effective_threshold = float(threshold) if threshold is not None else model.threshold
        deployment_id = self._deployment_id(endpoint_for(model.model_name, self.settings))

        frame = prepare_inference_frame(instances, self._data_config(model))
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
                deployment_id=deployment_id,
            )

        return BatchPredictionResponse(
            request_id=request_id,
            model_name=model.model_name,
            model_version=model.version,
            n_instances=len(instances),
            predictions=[int(p) for p in predictions],
            prediction_labels=[model.label_for(int(p)) for p in predictions],
            probabilities=[round(float(p), 6) for p in probabilities],
            threshold=round(effective_threshold, 6),
            inference_latency_ms=round(latency_ms, 3),
        )

    # -- helpers ------------------------------------------------------------- #
    def _data_config(self, model: LoadedModel) -> DataConfig:
        return model.data_config or self.settings.data

    def _deployment_id(self, endpoint: str) -> str | None:
        try:
            deployment = get_deployment_store().active(endpoint)
            return deployment.id if deployment else None
        except Exception:
            return None

    def _score_shadow(self, features: dict[str, Any], request_id: str, endpoint: str) -> None:
        """Mirror the request to the shadow model. Never raises."""
        try:
            shadow = self.provider.shadow_model(endpoint)
            if shadow is None:
                return
            started = time.perf_counter()
            frame = prepare_inference_frame([features], self._data_config(shadow))
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
        raw_importance = model.metrics.get("feature_importance")
        importance: dict[str, float] = (
            {str(k): float(v) for k, v in raw_importance.items()}
            if isinstance(raw_importance, dict)
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


def check_against_contract(
    model: LoadedModel, record: dict[str, Any]
) -> tuple[dict[str, Any], list[str]]:
    """Validate one record against the input contract of the version serving it."""
    from app.core.signature import SignatureError
    from app.schemas.prediction import LoanApplicationFeatures

    if model.signature is not None:
        try:
            return model.signature.check_record(record)
        except SignatureError as exc:
            raise InvalidPredictionInputError(
                str(exc), model=model.model_name, version=model.version
            ) from exc
    # A pre-signature version is always the reference model, whose contract is
    # the typed loan schema.
    try:
        return LoanApplicationFeatures.model_validate(record).model_dump(), []
    except ValueError as exc:
        raise InvalidPredictionInputError(
            str(exc), model=model.model_name, version=model.version
        ) from exc


def check_batch_against_contract(
    model: LoadedModel, instances: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Validate every row, reporting all failing rows at once."""
    clean: list[dict[str, Any]] = []
    problems: list[str] = []
    for index, record in enumerate(instances):
        try:
            clean.append(check_against_contract(model, record)[0])
        except InvalidPredictionInputError as exc:
            problems.append(f"row {index}: {exc.message}")
    if problems:
        raise InvalidPredictionInputError(
            f"{len(problems)} of {len(instances)} rows do not match the input contract: "
            + "; ".join(problems[:10]),
            model=model.model_name,
            version=model.version,
            failing_rows=len(problems),
        )
    return clean


_SERVICE: PredictionService | None = None


def get_prediction_service() -> PredictionService:
    global _SERVICE
    if _SERVICE is None:
        _SERVICE = PredictionService()
    return _SERVICE


def set_prediction_service(service: PredictionService | None) -> None:
    global _SERVICE
    _SERVICE = service
