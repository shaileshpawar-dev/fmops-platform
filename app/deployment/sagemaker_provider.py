"""SageMaker deployment provider.

Implements the same :class:`~app.deployment.base.DeploymentProvider` contract as
the local provider, so blue/green, canary and rollback logic is byte-identical
between them -- only the mechanism differs.

Mapping:

* a model version becomes a **SageMaker Model** plus a **production variant**,
* a traffic split becomes **variant weights** on one endpoint config,
* blue/green becomes a **new endpoint config + UpdateEndpoint** (SageMaker
  performs the instance swap with no downtime),
* canary steps become **UpdateEndpointWeightsAndCapacities** calls, which shift
  traffic without reprovisioning.

Every call here provisions billable resources, so this provider is never active
in tests or in ``make demo``.
"""

from __future__ import annotations

from typing import Any

from app.aws.sagemaker import SageMakerClient
from app.core.config import Settings, get_settings
from app.core.exceptions import ConfigurationError, DeploymentError
from app.core.logging import get_logger
from app.deployment.base import DeploymentProvider
from app.registry.base import ModelRegistry
from app.registry.factory import get_registry
from app.schemas.common import HealthStatus
from app.schemas.deployment import EndpointHealth, TrafficSplit

logger = get_logger(__name__)


def variant_name(model_name: str, version: int) -> str:
    """SageMaker variant names allow alphanumerics and hyphens only."""
    safe = "".join(c if c.isalnum() else "-" for c in model_name)[:40].strip("-")
    return f"{safe}-v{version}"


class SageMakerDeploymentProvider(DeploymentProvider):
    name = "sagemaker"

    def __init__(
        self,
        settings: Settings | None = None,
        client: SageMakerClient | None = None,
        registry: ModelRegistry | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self._client = client
        self._registry = registry

    @property
    def client(self) -> SageMakerClient:
        if self._client is None:
            self._client = SageMakerClient(self.settings)
        return self._client

    @property
    def registry(self) -> ModelRegistry:
        if self._registry is None:
            self._registry = get_registry()
        return self._registry

    # -- provider interface -------------------------------------------------- #
    def apply(
        self,
        endpoint_name: str,
        model_name: str,
        traffic: TrafficSplit,
        shadow_version: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        image_uri = self.settings.aws.sagemaker_training_image
        if not image_uri:
            raise ConfigurationError(
                "FMOPS_AWS__SAGEMAKER_TRAINING_IMAGE must point at the inference "
                "image in ECR before deploying to SageMaker",
                endpoint=endpoint_name,
            )

        # A variant with weight 0 is still provisioned (and billed), so drop
        # zero-weight versions rather than paying for idle instances.
        active = {v: w for v, w in traffic.weights.items() if w > 0}
        if not active:
            raise DeploymentError(
                "traffic split assigns no weight to any version",
                endpoint=endpoint_name,
            )

        variants: list[dict[str, Any]] = []
        for version, weight in sorted(active.items()):
            model_version = self.registry.get(model_name, version)
            sm_model_name = variant_name(model_name, version)
            self.client.create_model(sm_model_name, model_version.artifact_uri, image_uri)
            variants.append(
                {
                    "variant_name": sm_model_name,
                    "model_name": sm_model_name,
                    "weight": weight,
                    "instance_type": self.settings.deployment.instance_type,
                    "instance_count": self.settings.deployment.instance_count,
                }
            )

        if shadow_version is not None:
            # SageMaker has first-class shadow tests via ShadowModeConfig; this
            # provider does not configure them, so we refuse rather than
            # silently serving the shadow version real traffic.
            raise DeploymentError(
                "shadow deployments are not implemented for the SageMaker "
                "provider. Use SageMaker's native shadow tests, or run the "
                "shadow strategy against the local provider.",
                endpoint=endpoint_name,
                shadow_version=shadow_version,
            )

        # If the variant set is unchanged, shifting weights is far cheaper than
        # a full endpoint update -- this is the canary fast path.
        if self._same_variants(endpoint_name, {v["variant_name"] for v in variants}):
            self.client.update_variant_weights(
                endpoint_name,
                {v["variant_name"]: v["weight"] for v in variants},
            )
            logger.info(
                "deployment.sagemaker_weights_updated",
                extra={"endpoint": endpoint_name, "traffic": traffic.as_dict()},
            )
            return {
                "provider": self.name,
                "endpoint": endpoint_name,
                "operation": "update_weights",
                "variants": [v["variant_name"] for v in variants],
            }

        config_name = f"{endpoint_name}-cfg-{'-'.join(str(v) for v in sorted(active))}"[:63]
        self.client.create_endpoint_config(config_name, variants)
        result = self.client.deploy_endpoint(endpoint_name, config_name, wait=True)
        logger.info(
            "deployment.sagemaker_endpoint_updated",
            extra={
                "endpoint": endpoint_name,
                "config": config_name,
                "traffic": traffic.as_dict(),
            },
        )
        return {
            "provider": self.name,
            "endpoint": endpoint_name,
            "operation": "update_endpoint",
            "endpoint_config": config_name,
            **result,
            **(metadata or {}),
        }

    def _same_variants(self, endpoint_name: str, wanted: set[str]) -> bool:
        if not self.client.endpoint_exists(endpoint_name):
            return False
        try:
            description = self.client.describe_endpoint(endpoint_name)
        except DeploymentError:
            return False
        current = {v["VariantName"] for v in description.get("ProductionVariants", [])}
        return current == wanted

    def status(self, endpoint_name: str) -> dict[str, Any]:
        if not self.client.endpoint_exists(endpoint_name):
            return {"provider": self.name, "endpoint": endpoint_name, "exists": False}
        description = self.client.describe_endpoint(endpoint_name)
        return {
            "provider": self.name,
            "endpoint": endpoint_name,
            "exists": True,
            "status": description.get("EndpointStatus"),
            "endpoint_config": description.get("EndpointConfigName"),
            "variants": [
                {
                    "name": v["VariantName"],
                    "weight": v.get("CurrentWeight"),
                    "desired_weight": v.get("DesiredWeight"),
                    "instances": v.get("CurrentInstanceCount"),
                }
                for v in description.get("ProductionVariants", [])
            ],
            "last_modified": str(description.get("LastModifiedTime", "")),
        }

    def health_check(self, endpoint_name: str, model_name: str) -> EndpointHealth:
        checks: dict[str, bool] = {}
        if not self.client.endpoint_exists(endpoint_name):
            return EndpointHealth(
                endpoint_name=endpoint_name,
                status=HealthStatus.UNHEALTHY,
                checks={"endpoint_exists": False},
                detail="the SageMaker endpoint does not exist",
            )

        description = self.client.describe_endpoint(endpoint_name)
        status = description.get("EndpointStatus")
        checks["endpoint_exists"] = True
        checks["in_service"] = status == "InService"

        variants = description.get("ProductionVariants", [])
        checks["has_variants"] = bool(variants)
        checks["all_instances_running"] = all(
            v.get("CurrentInstanceCount", 0) > 0 for v in variants
        )

        healthy = all(checks.values())
        return EndpointHealth(
            endpoint_name=endpoint_name,
            status=HealthStatus.HEALTHY if healthy else HealthStatus.UNHEALTHY,
            checks=checks,
            detail="" if healthy else f"endpoint status is {status}",
        )

    def teardown(self, endpoint_name: str) -> None:
        self.client.delete_endpoint(endpoint_name)
        logger.info("deployment.sagemaker_endpoint_deleted", extra={"endpoint": endpoint_name})

    def supports_shadow(self) -> bool:
        return False
