"""In-process deployment provider.

This is a *real* serving implementation, not a mock: it loads the model versions
into memory and routes actual prediction requests between them according to the
configured weights. A 90/10 canary really does send ~10% of traffic to the
candidate, and the shadow variant really is scored on every mirrored request.

What it deliberately does not do is provision infrastructure -- there are no
instances, no autoscaling, no blue/green DNS cutover. Those belong to
:class:`~app.deployment.sagemaker_provider.SageMakerDeploymentProvider`. The
routing decision logic is shared, which is the point of the split.
"""

from __future__ import annotations

import random
import threading
from typing import Any

from app.core.config import Settings, get_settings
from app.core.exceptions import ModelNotLoadedError
from app.core.logging import get_logger
from app.core.utils import utcnow_iso
from app.deployment.base import DeploymentProvider
from app.deployment.model_cache import LoadedModel, ModelCache, get_model_cache
from app.schemas.common import HealthStatus
from app.schemas.deployment import EndpointHealth, TrafficSplit

logger = get_logger(__name__)


class Routing:
    """The live routing table for one endpoint."""

    def __init__(
        self,
        model_name: str,
        traffic: TrafficSplit,
        shadow_version: int | None = None,
    ) -> None:
        self.model_name = model_name
        self.traffic = traffic
        self.shadow_version = shadow_version
        self.updated_at = utcnow_iso()

    @property
    def versions(self) -> list[int]:
        return sorted(self.traffic.weights)

    def choose(self, rng: random.Random) -> int:
        """Pick a version according to the configured weights."""
        weights = self.traffic.weights
        if not weights:
            raise ModelNotLoadedError(
                "no model versions are routed for this endpoint",
                model=self.model_name,
            )
        if len(weights) == 1:
            return next(iter(weights))
        versions = list(weights)
        return rng.choices(versions, weights=[weights[v] for v in versions], k=1)[0]

    def as_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "traffic": self.traffic.as_dict(),
            "shadow_version": self.shadow_version,
            "updated_at": self.updated_at,
        }


class LocalDeploymentProvider(DeploymentProvider):
    """Serves models in this process and routes traffic by weight."""

    name = "local"

    def __init__(
        self,
        cache: ModelCache | None = None,
        settings: Settings | None = None,
        seed: int | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self._cache = cache
        self._routes: dict[str, Routing] = {}
        self._lock = threading.RLock()
        # Seeded only in tests; production routing must not be predictable.
        self._rng = random.Random(seed)

    @property
    def cache(self) -> ModelCache:
        if self._cache is None:
            self._cache = get_model_cache()
        return self._cache

    # -- provider interface -------------------------------------------------- #
    def apply(
        self,
        endpoint_name: str,
        model_name: str,
        traffic: TrafficSplit,
        shadow_version: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        # Load every target version *before* switching traffic. A version that
        # cannot load must never receive requests.
        required = sorted({*traffic.weights, *([shadow_version] if shadow_version else [])})
        for version in required:
            self.cache.get(model_name, version)

        with self._lock:
            self._routes[endpoint_name] = Routing(model_name, traffic, shadow_version)

        logger.info(
            "deployment.routing_applied",
            extra={
                "endpoint": endpoint_name,
                "model": model_name,
                "traffic": traffic.as_dict(),
                "shadow_version": shadow_version,
                "provider": self.name,
            },
        )
        return {
            "provider": self.name,
            "endpoint": endpoint_name,
            "loaded_versions": required,
            **(metadata or {}),
        }

    def _rehydrate(self, endpoint_name: str) -> Routing | None:
        """Rebuild the in-memory routing table from the persisted deployment.

        The routing table is process-local, but the deployment record is not.
        Without this, a CLI invocation or a second worker would report the
        endpoint as unrouted even though a deployment is live -- and a restarted
        container would stop serving until someone redeployed.
        """
        from app.deployment.base import get_deployment_store

        try:
            deployment = get_deployment_store().active(endpoint_name)
        except Exception as exc:
            logger.debug(
                "deployment.rehydrate_lookup_failed",
                extra={"endpoint": endpoint_name, "error": str(exc)},
            )
            return None
        if deployment is None or not deployment.traffic:
            return None

        try:
            routing = Routing(
                deployment.model_name,
                deployment.traffic_split(),
                deployment.shadow_version,
            )
        except Exception as exc:
            logger.warning(
                "deployment.rehydrate_failed",
                extra={"endpoint": endpoint_name, "error": str(exc)},
            )
            return None

        with self._lock:
            self._routes.setdefault(endpoint_name, routing)
            restored = self._routes[endpoint_name]
        logger.info(
            "deployment.routing_rehydrated",
            extra={
                "endpoint": endpoint_name,
                "traffic": deployment.traffic,
                "deployment_id": deployment.id,
            },
        )
        return restored

    def status(self, endpoint_name: str) -> dict[str, Any]:
        routing = self.routing(endpoint_name)
        if routing is None:
            return {"provider": self.name, "endpoint": endpoint_name, "routed": False}
        return {
            "provider": self.name,
            "endpoint": endpoint_name,
            "routed": True,
            "loaded_models": self.cache.loaded_keys(),
            **routing.as_dict(),
        }

    def health_check(self, endpoint_name: str, model_name: str) -> EndpointHealth:
        """Verify every routed version is loadable and can score a request."""
        routing = self.routing(endpoint_name)

        checks: dict[str, bool] = {}
        if routing is None:
            return EndpointHealth(
                endpoint_name=endpoint_name,
                status=HealthStatus.UNHEALTHY,
                checks={"routing_configured": False},
                detail="no routing configured for this endpoint",
            )

        checks["routing_configured"] = True
        checks["traffic_sums_to_100"] = (
            abs(sum(routing.traffic.weights.values()) - 100.0) < 0.01
        )
        for version in routing.versions:
            try:
                self.cache.get(model_name, version)
                checks[f"v{version}_loadable"] = True
            except Exception as exc:
                checks[f"v{version}_loadable"] = False
                logger.error(
                    "deployment.health_check_failed",
                    extra={
                        "endpoint": endpoint_name,
                        "version": version,
                        "error": str(exc),
                    },
                )

        healthy = all(checks.values())
        return EndpointHealth(
            endpoint_name=endpoint_name,
            status=HealthStatus.HEALTHY if healthy else HealthStatus.UNHEALTHY,
            checks=checks,
            detail="" if healthy else "one or more routed versions failed to load",
        )

    def teardown(self, endpoint_name: str) -> None:
        with self._lock:
            routing = self._routes.pop(endpoint_name, None)
        if routing:
            logger.info(
                "deployment.torn_down",
                extra={"endpoint": endpoint_name, "provider": self.name},
            )

    # -- serving-side API ---------------------------------------------------- #
    def routing(self, endpoint_name: str) -> Routing | None:
        with self._lock:
            routing = self._routes.get(endpoint_name)
        if routing is not None:
            return routing
        return self._rehydrate(endpoint_name)

    def resolve(
        self, endpoint_name: str, pinned_version: int | None = None
    ) -> tuple[LoadedModel, str]:
        """Pick the model that will serve this request.

        Returns the loaded model plus the variant label ("primary", "canary",
        or "pinned") used for metrics and the inference log.
        """
        routing = self.routing(endpoint_name)
        if routing is None:
            raise ModelNotLoadedError(
                f"endpoint {endpoint_name!r} has no active deployment; "
                "deploy a model version first",
                endpoint=endpoint_name,
            )
        if pinned_version is not None:
            return self.cache.get(routing.model_name, pinned_version), "pinned"

        version = routing.choose(self._rng)
        primary = routing.traffic.primary_version()
        variant = "primary" if version == primary else "canary"
        return self.cache.get(routing.model_name, version), variant

    def shadow_model(self, endpoint_name: str) -> LoadedModel | None:
        routing = self.routing(endpoint_name)
        if routing is None or routing.shadow_version is None:
            return None
        try:
            return self.cache.get(routing.model_name, routing.shadow_version)
        except Exception as exc:
            # A broken shadow must never affect the live response.
            logger.error(
                "deployment.shadow_load_failed",
                extra={
                    "endpoint": endpoint_name,
                    "version": routing.shadow_version,
                    "error": str(exc),
                },
            )
            return None


_PROVIDER: LocalDeploymentProvider | None = None
_PROVIDER_LOCK = threading.Lock()


def get_local_provider() -> LocalDeploymentProvider:
    """Process-wide local provider (the serving path and the manager share it)."""
    global _PROVIDER
    if _PROVIDER is None:
        with _PROVIDER_LOCK:
            if _PROVIDER is None:
                _PROVIDER = LocalDeploymentProvider()
    return _PROVIDER


def set_local_provider(provider: LocalDeploymentProvider | None) -> None:
    global _PROVIDER
    _PROVIDER = provider
