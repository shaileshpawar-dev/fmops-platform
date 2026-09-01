"""Amazon SageMaker integration.

Three capabilities, each a real API call -- there is no simulated SageMaker
anywhere in this project. When AWS is not configured these raise a clear error
naming what is missing, and the platform's local equivalents are used instead.

1. **Training jobs** -- run the containerised training entrypoint on managed
   infrastructure.
2. **Hyperparameter tuning jobs** -- the AWS backend for
   :class:`~app.training.tuning.Tuner`.
3. **Endpoints** -- the AWS backend for
   :class:`~app.deployment.base.DeploymentProvider`, using production-variant
   weights for canary and a new endpoint-config swap for blue/green.

Cost warning: every method here provisions billable resources. Nothing in this
module runs during tests or during ``make demo``.
"""

from __future__ import annotations

import time
from typing import Any

from app.aws.client import get_client, require_aws_enabled, require_boto3
from app.core.config import Settings, get_settings
from app.core.exceptions import (
    ConfigurationError,
    DeploymentError,
    ProviderUnavailableError,
    TuningError,
)
from app.core.logging import get_logger
from app.core.utils import utcnow_iso
from app.schemas.model import TrialResult, TuningResult

logger = get_logger(__name__)


def _job_name(prefix: str) -> str:
    """SageMaker job names: <=63 chars, alphanumeric and hyphens only."""
    stamp = utcnow_iso().replace(":", "").replace("-", "").replace(".", "")[:15]
    return f"{prefix}-{stamp}"[:63].strip("-")


class SageMakerClient:
    """Wrapper over the SageMaker control plane."""

    def __init__(self, settings: Settings | None = None, client=None) -> None:
        self.settings = settings or get_settings()
        self._client = client

    @property
    def client(self):
        if self._client is None:
            require_boto3()
            self._client = get_client("sagemaker", self.settings.aws.region)
        return self._client

    def _require_config(self) -> None:
        require_aws_enabled(self.settings)
        aws = self.settings.aws
        missing = [
            name
            for name, value in (
                ("FMOPS_AWS__SAGEMAKER_ROLE_ARN", aws.sagemaker_role_arn),
                ("FMOPS_AWS__S3_BUCKET", aws.s3_bucket),
                ("FMOPS_AWS__SAGEMAKER_TRAINING_IMAGE", aws.sagemaker_training_image),
            )
            if not value
        ]
        if missing:
            raise ConfigurationError(
                "SageMaker is not fully configured; missing: " + ", ".join(missing),
                missing=missing,
                hint="terraform/ provisions these; see docs/deployment.md",
            )

    def _s3(self, *parts: str) -> str:
        aws = self.settings.aws
        prefix = "/".join(p.strip("/") for p in parts if p)
        return f"s3://{aws.s3_bucket}/{aws.s3_prefix}/{prefix}"

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #
    def run_training_job(
        self,
        input_s3_uri: str,
        hyperparameters: dict[str, Any] | None = None,
        instance_type: str | None = None,
        wait: bool = True,
        max_runtime_seconds: int = 3600,
    ) -> dict[str, Any]:
        """Submit a training job and (optionally) wait for it."""
        self._require_config()
        aws = self.settings.aws
        job_name = _job_name("fmops-train")

        request: dict[str, Any] = {
            "TrainingJobName": job_name,
            "AlgorithmSpecification": {
                "TrainingImage": aws.sagemaker_training_image,
                "TrainingInputMode": "File",
            },
            "RoleArn": aws.sagemaker_role_arn,
            "InputDataConfig": [
                {
                    "ChannelName": "training",
                    "DataSource": {
                        "S3DataSource": {
                            "S3DataType": "S3Prefix",
                            "S3Uri": input_s3_uri,
                            "S3DataDistributionType": "FullyReplicated",
                        }
                    },
                    "ContentType": "text/csv",
                }
            ],
            "OutputDataConfig": {"S3OutputPath": self._s3("training-output")},
            "ResourceConfig": {
                "InstanceType": instance_type or aws.sagemaker_instance_type,
                "InstanceCount": 1,
                "VolumeSizeInGB": 30,
            },
            "StoppingCondition": {"MaxRuntimeInSeconds": max_runtime_seconds},
            "HyperParameters": {str(k): str(v) for k, v in (hyperparameters or {}).items()},
            "Tags": self._tags(),
            "EnableManagedSpotTraining": False,
        }

        logger.info(
            "sagemaker.training_job_submitting",
            extra={
                "job_name": job_name,
                "instance_type": request["ResourceConfig"]["InstanceType"],
            },
        )
        try:
            self.client.create_training_job(**request)
        except Exception as exc:
            raise ProviderUnavailableError(
                f"SageMaker rejected the training job: {exc}", job_name=job_name
            ) from exc

        if not wait:
            return {"job_name": job_name, "status": "InProgress"}
        return self._wait_for_training(job_name, max_runtime_seconds)

    def _wait_for_training(self, job_name: str, timeout: int) -> dict[str, Any]:
        deadline = time.time() + timeout
        while time.time() < deadline:
            description = self.client.describe_training_job(TrainingJobName=job_name)
            status = description["TrainingJobStatus"]
            if status in ("Completed", "Failed", "Stopped"):
                logger.info(
                    "sagemaker.training_job_finished",
                    extra={"job_name": job_name, "status": status},
                )
                if status != "Completed":
                    raise ProviderUnavailableError(
                        f"SageMaker training job {job_name} ended as {status}: "
                        f"{description.get('FailureReason', 'no reason given')}",
                        job_name=job_name,
                        status=status,
                    )
                return {
                    "job_name": job_name,
                    "status": status,
                    "model_artifacts": description["ModelArtifacts"]["S3ModelArtifacts"],
                    "metrics": {
                        m["MetricName"]: m["Value"]
                        for m in description.get("FinalMetricDataList", [])
                    },
                }
            time.sleep(30)
        raise ProviderUnavailableError(
            f"SageMaker training job {job_name} did not finish within {timeout}s",
            job_name=job_name,
        )

    # ------------------------------------------------------------------ #
    # Hyperparameter tuning
    # ------------------------------------------------------------------ #
    def run_tuning_job(
        self,
        algorithm: str,
        search_space: dict[str, list[Any]],
        metric: str = "roc_auc",
        direction: str = "maximize",
        max_trials: int = 20,
        max_parallel: int = 2,
        input_s3_uri: str | None = None,
        wait: bool = True,
    ) -> TuningResult:
        """Submit a hyperparameter tuning job and return the platform's result type."""
        self._require_config()
        aws = self.settings.aws
        job_name = _job_name("fmops-hpo")
        ranges = self._parameter_ranges(search_space)

        request: dict[str, Any] = {
            "HyperParameterTuningJobName": job_name,
            "HyperParameterTuningJobConfig": {
                "Strategy": "Bayesian",
                "HyperParameterTuningJobObjective": {
                    "Type": "Maximize" if direction == "maximize" else "Minimize",
                    "MetricName": metric,
                },
                "ResourceLimits": {
                    "MaxNumberOfTrainingJobs": max_trials,
                    "MaxParallelTrainingJobs": max_parallel,
                },
                "ParameterRanges": ranges,
            },
            "TrainingJobDefinition": {
                "AlgorithmSpecification": {
                    "TrainingImage": aws.sagemaker_training_image,
                    "TrainingInputMode": "File",
                    "MetricDefinitions": [{"Name": metric, "Regex": f"{metric}=([0-9\\.]+)"}],
                },
                "RoleArn": aws.sagemaker_role_arn,
                "InputDataConfig": [
                    {
                        "ChannelName": "training",
                        "DataSource": {
                            "S3DataSource": {
                                "S3DataType": "S3Prefix",
                                "S3Uri": input_s3_uri or self._s3("datasets"),
                                "S3DataDistributionType": "FullyReplicated",
                            }
                        },
                        "ContentType": "text/csv",
                    }
                ],
                "OutputDataConfig": {"S3OutputPath": self._s3("tuning-output")},
                "ResourceConfig": {
                    "InstanceType": aws.sagemaker_instance_type,
                    "InstanceCount": 1,
                    "VolumeSizeInGB": 30,
                },
                "StoppingCondition": {"MaxRuntimeInSeconds": 3600},
                "StaticHyperParameters": {"algorithm": algorithm},
            },
            "Tags": self._tags(),
        }

        logger.info(
            "sagemaker.tuning_job_submitting",
            extra={"job_name": job_name, "max_trials": max_trials, "metric": metric},
        )
        try:
            self.client.create_hyper_parameter_tuning_job(**request)
        except Exception as exc:
            raise TuningError(
                f"SageMaker rejected the tuning job: {exc}", job_name=job_name
            ) from exc

        if not wait:
            return TuningResult(
                backend="sagemaker",
                metric=metric,
                direction=direction,
                n_trials=0,
                best_params={},
                best_score=0.0,
                trials=[],
            )
        return self._wait_for_tuning(job_name, metric, direction)

    def _wait_for_tuning(
        self, job_name: str, metric: str, direction: str, timeout: int = 7200
    ) -> TuningResult:
        started = time.time()
        while time.time() - started < timeout:
            description = self.client.describe_hyper_parameter_tuning_job(
                HyperParameterTuningJobName=job_name
            )
            status = description["HyperParameterTuningJobStatus"]
            if status in ("Completed", "Failed", "Stopped"):
                if status != "Completed":
                    raise TuningError(
                        f"SageMaker tuning job {job_name} ended as {status}",
                        job_name=job_name,
                    )
                best = description.get("BestTrainingJob", {})
                trials = self._collect_trials(job_name, metric)
                return TuningResult(
                    backend="sagemaker",
                    metric=metric,
                    direction=direction,
                    n_trials=len(trials),
                    best_params={
                        k: _coerce(v) for k, v in best.get("TunedHyperParameters", {}).items()
                    },
                    best_score=float(
                        best.get("FinalHyperParameterTuningJobObjectiveMetric", {}).get(
                            "Value", 0.0
                        )
                    ),
                    trials=trials,
                    duration_seconds=round(time.time() - started, 2),
                )
            time.sleep(60)
        raise TuningError(
            f"SageMaker tuning job {job_name} did not finish within {timeout}s",
            job_name=job_name,
        )

    def _collect_trials(self, job_name: str, metric: str) -> list[TrialResult]:
        trials: list[TrialResult] = []
        paginator = self.client.get_paginator(
            "list_training_jobs_for_hyper_parameter_tuning_job"
        )
        for index, page in enumerate(paginator.paginate(HyperParameterTuningJobName=job_name)):
            for job in page.get("TrainingJobSummaries", []):
                objective = job.get("FinalHyperParameterTuningJobObjectiveMetric", {})
                trials.append(
                    TrialResult(
                        trial_id=index,
                        params={
                            k: _coerce(v)
                            for k, v in job.get("TunedHyperParameters", {}).items()
                        },
                        score=float(objective.get("Value", 0.0)),
                        metric=metric,
                        status=(
                            "completed"
                            if job.get("TrainingJobStatus") == "Completed"
                            else "failed"
                        ),
                    )
                )
        return trials

    @staticmethod
    def _parameter_ranges(search_space: dict[str, list[Any]]) -> dict[str, list[dict]]:
        """Translate the platform's search space into SageMaker parameter ranges."""
        integer: list[dict] = []
        continuous: list[dict] = []
        categorical: list[dict] = []
        for name, values in search_space.items():
            if all(isinstance(v, bool) for v in values) or any(
                isinstance(v, str) for v in values
            ):
                categorical.append({"Name": name, "Values": [str(v) for v in values]})
            elif all(isinstance(v, int) for v in values):
                integer.append(
                    {
                        "Name": name,
                        "MinValue": str(min(values)),
                        "MaxValue": str(max(values)),
                    }
                )
            else:
                continuous.append(
                    {
                        "Name": name,
                        "MinValue": str(min(values)),
                        "MaxValue": str(max(values)),
                        "ScalingType": (
                            "Logarithmic"
                            if min(values) > 0 and max(values) / min(values) >= 100
                            else "Auto"
                        ),
                    }
                )
        return {
            "IntegerParameterRanges": integer,
            "ContinuousParameterRanges": continuous,
            "CategoricalParameterRanges": categorical,
        }

    # ------------------------------------------------------------------ #
    # Endpoints
    # ------------------------------------------------------------------ #
    def create_model(self, model_name: str, model_data_url: str, image_uri: str) -> str:
        self._require_config()
        try:
            self.client.create_model(
                ModelName=model_name,
                PrimaryContainer={
                    "Image": image_uri,
                    "ModelDataUrl": model_data_url,
                    "Environment": {"FMOPS_ENV": self.settings.environment},
                },
                ExecutionRoleArn=self.settings.aws.sagemaker_role_arn,
                Tags=self._tags(),
            )
        except Exception as exc:
            if "Cannot create already existing model" not in str(exc):
                raise DeploymentError(
                    f"could not create SageMaker model {model_name}: {exc}",
                    model_name=model_name,
                ) from exc
        return model_name

    def create_endpoint_config(self, config_name: str, variants: list[dict[str, Any]]) -> str:
        """Create an endpoint config with weighted production variants.

        Variant weights are how canary traffic splitting is done on SageMaker --
        the same percentages the local provider applies in-process.
        """
        self._require_config()
        production_variants = [
            {
                "VariantName": v["variant_name"],
                "ModelName": v["model_name"],
                "InitialInstanceCount": v.get(
                    "instance_count", self.settings.deployment.instance_count
                ),
                "InstanceType": v.get("instance_type", self.settings.deployment.instance_type),
                "InitialVariantWeight": float(v.get("weight", 1.0)),
            }
            for v in variants
        ]
        try:
            self.client.create_endpoint_config(
                EndpointConfigName=config_name,
                ProductionVariants=production_variants,
                Tags=self._tags(),
            )
        except Exception as exc:
            raise DeploymentError(
                f"could not create endpoint config {config_name}: {exc}",
                config_name=config_name,
            ) from exc
        return config_name

    def deploy_endpoint(
        self, endpoint_name: str, config_name: str, wait: bool = True
    ) -> dict[str, Any]:
        """Create or update an endpoint. Updates are a zero-downtime swap."""
        self._require_config()
        exists = self.endpoint_exists(endpoint_name)
        try:
            if exists:
                self.client.update_endpoint(
                    EndpointName=endpoint_name, EndpointConfigName=config_name
                )
            else:
                self.client.create_endpoint(
                    EndpointName=endpoint_name,
                    EndpointConfigName=config_name,
                    Tags=self._tags(),
                )
        except Exception as exc:
            raise DeploymentError(
                f"could not {'update' if exists else 'create'} endpoint "
                f"{endpoint_name}: {exc}",
                endpoint_name=endpoint_name,
            ) from exc

        logger.info(
            "sagemaker.endpoint_deploying",
            extra={
                "endpoint": endpoint_name,
                "config": config_name,
                "operation": "update" if exists else "create",
            },
        )
        if wait:
            return self._wait_for_endpoint(endpoint_name)
        return {"endpoint_name": endpoint_name, "status": "Updating"}

    def update_variant_weights(self, endpoint_name: str, weights: dict[str, float]) -> None:
        """Shift traffic between existing variants without redeploying."""
        self._require_config()
        try:
            self.client.update_endpoint_weights_and_capacities(
                EndpointName=endpoint_name,
                DesiredWeightsAndCapacities=[
                    {"VariantName": name, "DesiredWeight": float(weight)}
                    for name, weight in weights.items()
                ],
            )
        except Exception as exc:
            raise DeploymentError(
                f"could not update variant weights on {endpoint_name}: {exc}",
                endpoint_name=endpoint_name,
                weights=weights,
            ) from exc

    def endpoint_exists(self, endpoint_name: str) -> bool:
        try:
            self.client.describe_endpoint(EndpointName=endpoint_name)
            return True
        except Exception:
            return False

    def describe_endpoint(self, endpoint_name: str) -> dict[str, Any]:
        try:
            return self.client.describe_endpoint(EndpointName=endpoint_name)
        except Exception as exc:
            raise DeploymentError(
                f"could not describe endpoint {endpoint_name}: {exc}",
                endpoint_name=endpoint_name,
            ) from exc

    def _wait_for_endpoint(self, endpoint_name: str, timeout: int = 1800) -> dict[str, Any]:
        deadline = time.time() + timeout
        while time.time() < deadline:
            description = self.describe_endpoint(endpoint_name)
            status = description["EndpointStatus"]
            if status == "InService":
                return {
                    "endpoint_name": endpoint_name,
                    "status": status,
                    "variants": [
                        {
                            "name": v["VariantName"],
                            "weight": v.get("CurrentWeight", 0),
                            "instances": v.get("CurrentInstanceCount", 0),
                        }
                        for v in description.get("ProductionVariants", [])
                    ],
                }
            if status in ("Failed", "OutOfService"):
                raise DeploymentError(
                    f"endpoint {endpoint_name} entered {status}: "
                    f"{description.get('FailureReason', 'no reason given')}",
                    endpoint_name=endpoint_name,
                    status=status,
                )
            time.sleep(20)
        raise DeploymentError(
            f"endpoint {endpoint_name} was not InService within {timeout}s",
            endpoint_name=endpoint_name,
        )

    def delete_endpoint(self, endpoint_name: str) -> None:
        try:
            self.client.delete_endpoint(EndpointName=endpoint_name)
        except Exception as exc:
            logger.warning(
                "sagemaker.delete_endpoint_failed",
                extra={"endpoint": endpoint_name, "error": str(exc)},
            )

    def _tags(self) -> list[dict[str, str]]:
        return [
            {"Key": "Project", "Value": "fmops-platform"},
            {"Key": "Environment", "Value": self.settings.environment},
            {"Key": "ManagedBy", "Value": "fmops"},
        ]


def _coerce(value: Any) -> Any:
    """SageMaker returns every hyperparameter as a string."""
    if not isinstance(value, str):
        return value
    text = value.strip().strip('"')
    for caster in (int, float):
        try:
            return caster(text)
        except ValueError:
            continue
    return text
