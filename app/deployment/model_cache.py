"""Loaded-model cache.

Serving needs the actual fitted pipeline in memory. This cache maps
``(model_name, version)`` onto a loaded pipeline plus its sidecar metadata
(decision threshold, feature columns, training metrics), loading lazily from the
artifact store and evicting least-recently-used entries.

Keeping the threshold with the model matters: a model trained at an operating
point of 0.55 that gets served at 0.5 is a silently different model.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sklearn.pipeline import Pipeline

from app.core.config import DataConfig, Settings, get_settings
from app.core.exceptions import ModelNotLoadedError
from app.core.logging import get_logger
from app.core.signature import ModelSignature
from app.core.storage import resolve_uri_to_local
from app.core.utils import read_json, utcnow_iso
from app.registry.base import ModelRegistry
from app.registry.factory import get_registry
from app.schemas.model import ModelVersion

logger = get_logger(__name__)


# Warm-up input for pre-signature versions of the reference model only. Every
# other version warms up with its own signature's example.
_REFERENCE_SAMPLE: dict[str, Any] = {
    "age": 40.0,
    "annual_income": 60000.0,
    "loan_amount": 15000.0,
    "loan_term_months": 36,
    "credit_score": 700.0,
    "debt_to_income": 0.25,
    "employment_years": 5.0,
    "num_credit_lines": 5,
    "num_late_payments_12m": 0,
    "credit_utilization": 0.3,
    "employment_type": "salaried",
    "housing_status": "own",
    "loan_purpose": "auto",
    "region": "north",
}


@dataclass
class LoadedModel:
    """A fitted pipeline plus everything the serving path needs about it."""

    model_name: str
    version: int
    stage: str
    pipeline: Pipeline
    threshold: float = 0.5
    algorithm: str = ""
    artifact_uri: str = ""
    dataset_version: str | None = None
    git_commit: str = "unknown"
    metrics: dict[str, float] = field(default_factory=dict)
    params: dict[str, Any] = field(default_factory=dict)
    feature_columns: list[str] = field(default_factory=list)
    # The input contract this version was trained against, and the data config
    # rebuilt from it. ``signature`` is None only for pre-signature versions of
    # the reference model, whose contract is the configured one.
    signature: ModelSignature | None = None
    data_config: DataConfig | None = None
    loaded_at: str = field(default_factory=utcnow_iso)

    @property
    def key(self) -> str:
        return f"{self.model_name}:{self.version}"

    def label_for(self, prediction: int) -> str:
        if self.signature is not None:
            return self.signature.label_for(prediction)
        return "default" if int(prediction) == 1 else "no_default"


class ModelCache:
    """Thread-safe LRU cache of loaded models."""

    def __init__(
        self,
        max_size: int = 4,
        registry: ModelRegistry | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.max_size = max_size
        self.settings = settings or get_settings()
        self._registry = registry
        self._entries: OrderedDict[str, LoadedModel] = OrderedDict()
        self._lock = threading.RLock()

    @property
    def registry(self) -> ModelRegistry:
        if self._registry is None:
            self._registry = get_registry()
        return self._registry

    # -- loading ------------------------------------------------------------- #
    def get(self, model_name: str, version: int) -> LoadedModel:
        key = f"{model_name}:{version}"
        with self._lock:
            cached = self._entries.get(key)
            if cached is not None:
                self._entries.move_to_end(key)
                return cached

        loaded = self._load(model_name, version)

        with self._lock:
            self._entries[key] = loaded
            self._entries.move_to_end(key)
            while len(self._entries) > self.max_size:
                evicted, _ = self._entries.popitem(last=False)
                logger.info("model_cache.evicted", extra={"model_key": evicted})
        return loaded

    def _load(self, model_name: str, version: int) -> LoadedModel:
        model_version: ModelVersion = self.registry.get(model_name, version)
        local_path = self._materialise(model_version)

        try:
            import joblib

            pipeline = joblib.load(local_path)
        except Exception as exc:
            raise ModelNotLoadedError(
                f"could not deserialise model {model_name} v{version}: {exc}",
                model=model_name,
                version=version,
                path=str(local_path),
            ) from exc

        metadata = read_json(local_path.parent / "metadata.json", default={}) or {}
        threshold = float(
            metadata.get("threshold") or model_version.params.get("threshold") or 0.5
        )
        # The sidecar travels with the artifact and is authoritative; the
        # registry copy covers artifacts whose sidecar was not fetched.
        raw_signature = metadata.get("signature") or model_version.signature
        signature = ModelSignature.model_validate(raw_signature) if raw_signature else None
        from app.registry.context import data_config_for

        data_config = (
            signature.to_data_config(self.settings.data)
            if signature is not None
            else data_config_for(model_version, self.settings)
        )

        loaded = LoadedModel(
            model_name=model_name,
            version=version,
            stage=model_version.stage.value,
            pipeline=pipeline,
            threshold=threshold,
            algorithm=model_version.algorithm,
            artifact_uri=model_version.artifact_uri,
            dataset_version=model_version.dataset_version,
            git_commit=model_version.git_commit,
            metrics=model_version.metrics,
            params=model_version.params,
            feature_columns=metadata.get("feature_columns", []),
            signature=signature,
            data_config=data_config,
        )
        logger.info(
            "model_cache.loaded",
            extra={
                "model": model_name,
                "version": version,
                "stage": loaded.stage,
                "threshold": threshold,
                "artifact_uri": model_version.artifact_uri[:120],
            },
        )
        return loaded

    def _materialise(self, model_version: ModelVersion) -> Path:
        """Get a local path for the artifact, downloading from S3 if needed."""
        uri = model_version.artifact_uri
        destination = (
            self.settings.paths.models_dir
            / "_cache"
            / model_version.name
            / str(model_version.version)
            / "model.joblib"
        )
        try:
            path = resolve_uri_to_local(uri, destination)
        except Exception as exc:
            raise ModelNotLoadedError(
                f"could not fetch artifact for {model_version.key}: {exc}",
                artifact_uri=uri,
            ) from exc
        if not path.is_file():
            raise ModelNotLoadedError(
                f"model artifact missing for {model_version.key} at {path}",
                artifact_uri=uri,
                path=str(path),
            )
        # Pull the sidecar too when it lives beside a remote artifact.
        sidecar = path.parent / "metadata.json"
        if not sidecar.is_file() and "://" in uri and not uri.startswith("file:"):
            try:
                resolve_uri_to_local(uri.rsplit("/", 1)[0] + "/metadata.json", sidecar)
            except Exception as exc:  # metadata is optional
                logger.debug(
                    "model_cache.metadata_unavailable",
                    extra={"model_key": model_version.key, "error": str(exc)},
                )
        return path

    def warm(self, loaded: LoadedModel) -> float:
        """Run one throwaway prediction so the first real request is not slow.

        Deserialising a pipeline is cheap; the first ``predict`` through it is
        not -- scikit-learn and numpy do a lot of lazy setup on first call. On a
        cold container that shows up as a ~2s p99 on the very first request,
        which then poisons latency SLO measurements and canary decisions.
        """
        import time

        from app.data.preprocessing import prepare_inference_frame

        started = time.perf_counter()
        try:
            sample = (
                loaded.signature.example()
                if loaded.signature is not None
                else _REFERENCE_SAMPLE
            )
            frame = prepare_inference_frame([sample], loaded.data_config or self.settings.data)
            loaded.pipeline.predict_proba(frame)
        except Exception as exc:
            # A failed warm-up is not fatal to the process, but for a version
            # with a recorded signature it means the model cannot score its own
            # training medians -- worth an error, not a shrug.
            logger.error(
                "model_cache.warmup_failed",
                extra={"model_key": loaded.key, "error": str(exc)},
            )
            return 0.0
        elapsed_ms = (time.perf_counter() - started) * 1000
        logger.info(
            "model_cache.warmed",
            extra={"model_key": loaded.key, "warmup_ms": round(elapsed_ms, 1)},
        )
        return elapsed_ms

    # -- management ---------------------------------------------------------- #
    def preload(self, model_name: str, versions: list[int]) -> list[int]:
        """Warm the cache. Returns the versions that loaded successfully."""
        loaded: list[int] = []
        for version in versions:
            try:
                self.warm(self.get(model_name, version))
                loaded.append(version)
            except Exception as exc:
                logger.error(
                    "model_cache.preload_failed",
                    extra={"model": model_name, "version": version, "error": str(exc)},
                )
        return loaded

    def invalidate(self, model_name: str | None = None, version: int | None = None) -> int:
        with self._lock:
            if model_name is None:
                count = len(self._entries)
                self._entries.clear()
                return count
            keys = [
                k
                for k in self._entries
                if k.startswith(f"{model_name}:")
                and (version is None or k == f"{model_name}:{version}")
            ]
            for key in keys:
                self._entries.pop(key, None)
            return len(keys)

    def loaded_keys(self) -> list[str]:
        with self._lock:
            return list(self._entries)


_CACHE: ModelCache | None = None
_CACHE_LOCK = threading.Lock()


def get_model_cache() -> ModelCache:
    global _CACHE
    if _CACHE is None:
        with _CACHE_LOCK:
            if _CACHE is None:
                _CACHE = ModelCache()
    return _CACHE


def set_model_cache(cache: ModelCache | None) -> None:
    """Override the process-wide cache (used by tests)."""
    global _CACHE
    _CACHE = cache
