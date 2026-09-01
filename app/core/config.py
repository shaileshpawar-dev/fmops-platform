"""Configuration for the FMOps platform.

Resolution order (lowest precedence first):

1. Defaults declared on the pydantic models below.
2. ``configs/<environment>.yaml`` -- environment picked from ``FMOPS_ENV``.
3. ``.env`` file in the project root.
4. Process environment variables, prefixed ``FMOPS_`` and nested with ``__``
   (e.g. ``FMOPS_DRIFT__THRESHOLD=0.25``).

Secrets are never declared with a usable default and are never written to YAML.
They only ever arrive via environment variables.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, SecretStr, model_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"

Environment = Literal["development", "staging", "production", "test"]


class ConfigFileError(Exception):
    """Raised for a malformed configuration file."""


# --------------------------------------------------------------------------- #
# Sections
# --------------------------------------------------------------------------- #
class PathsConfig(BaseModel):
    """Filesystem locations. Relative paths resolve against PROJECT_ROOT."""

    data_dir: Path = Path("data")
    raw_dir: Path = Path("data/raw")
    processed_dir: Path = Path("data/processed")
    sample_dir: Path = Path("data/sample")
    reference_dir: Path = Path("data/reference")
    artifacts_dir: Path = Path("artifacts")
    models_dir: Path = Path("artifacts/models")
    reports_dir: Path = Path("artifacts/reports")
    state_dir: Path = Path("artifacts/state")

    def resolve(self, root: Path = PROJECT_ROOT) -> PathsConfig:
        def _abs(p: Path) -> Path:
            return p if p.is_absolute() else (root / p)

        return PathsConfig(**{k: _abs(Path(v)) for k, v in self.model_dump().items()})

    def ensure(self) -> None:
        for value in self.model_dump().values():
            Path(value).mkdir(parents=True, exist_ok=True)


class DataConfig(BaseModel):
    """Dataset shape and generation parameters for the reference use case."""

    dataset_name: str = "loan_default"
    target_column: str = "default"
    id_column: str = "application_id"
    timestamp_column: str = "application_date"
    numeric_features: list[str] = Field(
        default_factory=lambda: [
            "age",
            "annual_income",
            "loan_amount",
            "loan_term_months",
            "credit_score",
            "debt_to_income",
            "employment_years",
            "num_credit_lines",
            "num_late_payments_12m",
            "credit_utilization",
        ]
    )
    categorical_features: list[str] = Field(
        default_factory=lambda: [
            "employment_type",
            "housing_status",
            "loan_purpose",
            "region",
        ]
    )
    train_rows: int = 8000
    test_size: float = 0.2
    validation_size: float = 0.15
    random_seed: int = 42

    @property
    def feature_columns(self) -> list[str]:
        return [*self.numeric_features, *self.categorical_features]


class ValidationConfig(BaseModel):
    """Thresholds for the data-validation gate."""

    engine: Literal["native", "great_expectations"] = "native"
    max_missing_fraction: float = 0.05
    max_duplicate_fraction: float = 0.01
    max_outlier_fraction: float = 0.05
    outlier_z_threshold: float = 4.0
    min_rows: int = 500
    min_class_fraction: float = 0.02
    max_class_fraction: float = 0.98
    fail_on_warning: bool = False


class TrainingConfig(BaseModel):
    algorithm: Literal[
        "hist_gradient_boosting",
        "random_forest",
        "logistic_regression",
        "xgboost",
        "lightgbm",
    ] = "hist_gradient_boosting"
    hyperparameters: dict[str, Any] = Field(default_factory=dict)
    cv_folds: int = 3
    class_weight_balanced: bool = True
    max_train_seconds: int = 1800


# Defaults for list/dict fields whose element type is a Literal union. An
# inline `default_factory=lambda: [...]` infers list[str], which is wider than
# the field, so name them once with the type spelled out.
AlertSinkName = Literal["log", "database", "file", "webhook", "sns"]
RetrainingTriggerName = Literal["drift", "performance", "schedule", "manual", "volume"]

_DEFAULT_SEARCH_SPACE: dict[str, list[Any]] = {
    "n_estimators": [100, 200, 300],
    "max_depth": [5, 10, 20],
    "learning_rate": [0.01, 0.05, 0.1],
}
_DEFAULT_ALERT_SINKS: list[AlertSinkName] = ["log", "database"]
_DEFAULT_RETRAINING_TRIGGERS: list[RetrainingTriggerName] = ["drift", "performance", "manual"]


class TuningConfig(BaseModel):
    enabled: bool = True
    backend: Literal["local_random", "local_grid", "sagemaker"] = "local_random"
    max_trials: int = 12
    metric: str = "roc_auc"
    direction: Literal["maximize", "minimize"] = "maximize"
    n_jobs: int = 1
    search_space: dict[str, list[Any]] = Field(
        default_factory=lambda: _DEFAULT_SEARCH_SPACE.copy()
    )


class ApprovalConfig(BaseModel):
    """The gate between 'training finished' and 'eligible for production'."""

    enabled: bool = True
    min_f1: float = 0.60
    min_roc_auc: float = 0.70
    min_precision: float = 0.0
    min_recall: float = 0.0
    max_inference_latency_ms: float = 250.0
    require_clean_validation: bool = True
    max_drift_score: float = 0.30
    comparison_metric: str = "roc_auc"
    min_improvement: float = 0.005
    require_manual_approval: bool = False


class DeploymentConfig(BaseModel):
    provider: Literal["local", "sagemaker"] = "local"
    strategy: Literal["blue_green", "canary", "shadow", "direct"] = "blue_green"
    endpoint_name: str = "fmops-loan-default"
    canary_steps: list[int] = Field(default_factory=lambda: [10, 25, 50, 100])
    canary_step_seconds: int = 30
    canary_min_requests_per_step: int = 20
    canary_max_error_rate: float = 0.05
    canary_max_latency_ms: float = 400.0
    shadow_sample_rate: float = 1.0
    auto_rollback: bool = True
    health_check_failures_before_rollback: int = 3
    instance_type: str = "ml.m5.large"
    instance_count: int = 1


class DriftConfig(BaseModel):
    enabled: bool = True
    engine: Literal["native", "evidently"] = "native"
    threshold: float = 0.20
    psi_threshold: float = 0.20
    ks_p_value: float = 0.05
    chi2_p_value: float = 0.05
    min_samples: int = 200
    reference_window: int = 5000
    detection_window: int = 1000
    numeric_bins: int = 10
    dataset_drift_share: float = 0.30


class MonitoringConfig(BaseModel):
    metrics_enabled: bool = True
    metrics_namespace: str = "fmops"
    cloudwatch_enabled: bool = False
    cloudwatch_namespace: str = "FMOps"
    resource_sample_seconds: int = 15
    # Counting open file descriptors enumerates every OS handle (~1.8s on
    # Windows). Useful when hunting a descriptor leak, far too slow for a
    # request path, so it is opt-in and only the background sampler uses it.
    sample_open_files: bool = False
    latency_slo_ms: float = 250.0
    error_rate_slo: float = 0.02
    log_predictions: bool = True
    prediction_log_sample_rate: float = 1.0


class AlertConfig(BaseModel):
    enabled: bool = True
    sinks: list[AlertSinkName] = Field(default_factory=lambda: _DEFAULT_ALERT_SINKS.copy())
    webhook_url: str | None = None
    sns_topic_arn: str | None = None
    dedupe_window_seconds: int = 300


class RetrainingConfig(BaseModel):
    enabled: bool = True
    triggers: list[RetrainingTriggerName] = Field(
        default_factory=lambda: _DEFAULT_RETRAINING_TRIGGERS.copy()
    )
    min_new_samples: int = 500
    performance_drop_tolerance: float = 0.05
    schedule_cron: str = "0 3 * * 1"
    cooldown_minutes: int = 60
    auto_deploy_if_better: bool = True


class LLMConfig(BaseModel):
    provider: Literal["mock", "bedrock", "gemini", "openai_compatible", "anthropic"] = "mock"
    model: str = "mock-small"
    temperature: float = 0.0
    max_tokens: int = 512
    timeout_seconds: float = 60.0
    max_retries: int = 2
    prompt_dir: Path = Path("app/llmops/prompts/library")
    default_prompt: str = "support_summarizer"
    default_prompt_version: str | None = None
    safety_enabled: bool = True
    safety_block_on_violation: bool = True
    eval_dataset_dir: Path = Path("data/sample/llm")
    pricing: dict[str, dict[str, float]] = Field(
        default_factory=lambda: {
            "mock-small": {"input": 0.0, "output": 0.0},
            "anthropic.claude-3-5-haiku-20241022-v1:0": {"input": 0.80, "output": 4.00},
            "anthropic.claude-sonnet-4-20250514-v1:0": {"input": 3.00, "output": 15.00},
            "claude-haiku-4-5-20251001": {"input": 1.00, "output": 5.00},
            "gemini-2.0-flash": {"input": 0.10, "output": 0.40},
            "gpt-4o-mini": {"input": 0.15, "output": 0.60},
        }
    )
    daily_cost_budget_usd: float = 25.0
    monthly_cost_budget_usd: float = 500.0


class AWSConfig(BaseModel):
    enabled: bool = False
    region: str = "us-east-1"
    s3_bucket: str | None = None
    s3_prefix: str = "fmops"
    ecr_repository: str | None = None
    sagemaker_role_arn: str | None = None
    sagemaker_endpoint_name: str | None = None
    sagemaker_training_image: str | None = None
    sagemaker_instance_type: str = "ml.m5.large"
    bedrock_region: str | None = None


class TrackingConfig(BaseModel):
    backend: Literal["mlflow", "local"] = "mlflow"
    tracking_uri: str | None = None
    registry_uri: str | None = None
    experiment_name: str = "fmops-loan-default"
    registry_backend: Literal["local", "mlflow"] = "local"
    registered_model_name: str = "loan_default_classifier"


class SecurityConfig(BaseModel):
    auth_backend: Literal["none", "api_key"] = "none"
    api_keys: list[str] = Field(default_factory=list)
    api_key_header: str = "X-API-Key"
    require_auth_for_writes: bool = True
    audit_log_enabled: bool = True
    cors_allow_origins: list[str] = Field(default_factory=lambda: ["*"])
    max_request_bytes: int = 2_000_000
    max_batch_rows: int = 1000


class ServerConfig(BaseModel):
    host: str = "0.0.0.0"  # noqa: S104 - container binding, see docs/deployment.md
    port: int = 8000
    workers: int = 1
    root_path: str = ""
    docs_enabled: bool = True


# --------------------------------------------------------------------------- #
# YAML settings source
# --------------------------------------------------------------------------- #
class YamlConfigSource(PydanticBaseSettingsSource):
    """Loads ``configs/<FMOPS_ENV>.yaml`` as a pydantic-settings source."""

    def __init__(self, settings_cls: type[BaseSettings], path: Path) -> None:
        super().__init__(settings_cls)
        self._path = path
        self._data: dict[str, Any] | None = None

    def _load(self) -> dict[str, Any]:
        if self._data is None:
            if self._path.is_file():
                loaded = yaml.safe_load(self._path.read_text(encoding="utf-8")) or {}
                if not isinstance(loaded, dict):
                    raise ConfigFileError(
                        f"{self._path} must contain a YAML mapping at the top level"
                    )
                self._data = loaded
            else:
                self._data = {}
        return self._data

    def get_field_value(self, field: Any, field_name: str) -> tuple[Any, str, bool]:
        return self._load().get(field_name), field_name, False

    def __call__(self) -> dict[str, Any]:
        return self._load()


# --------------------------------------------------------------------------- #
# Root settings
# --------------------------------------------------------------------------- #
class Settings(BaseSettings):
    """Root configuration object. Obtain via :func:`get_settings`."""

    model_config = SettingsConfigDict(
        env_prefix="FMOPS_",
        env_nested_delimiter="__",
        env_file=os.environ.get("FMOPS_ENV_FILE", str(PROJECT_ROOT / ".env")),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    environment: Environment = "development"
    service_name: str = "fmops-platform"
    version: str = "1.0.0"
    log_level: str = "INFO"
    log_format: Literal["json", "console"] = "json"
    git_commit: str = "unknown"
    database_url: str | None = None

    paths: PathsConfig = Field(default_factory=PathsConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    tuning: TuningConfig = Field(default_factory=TuningConfig)
    approval: ApprovalConfig = Field(default_factory=ApprovalConfig)
    deployment: DeploymentConfig = Field(default_factory=DeploymentConfig)
    drift: DriftConfig = Field(default_factory=DriftConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    alerts: AlertConfig = Field(default_factory=AlertConfig)
    retraining: RetrainingConfig = Field(default_factory=RetrainingConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    aws: AWSConfig = Field(default_factory=AWSConfig)
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)

    # --- secrets: env-only, never persisted ------------------------------- #
    aws_access_key_id: SecretStr | None = None
    aws_secret_access_key: SecretStr | None = None
    anthropic_api_key: SecretStr | None = None
    google_api_key: SecretStr | None = None
    openai_api_key: SecretStr | None = None
    openai_base_url: str | None = None

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        env_name = os.environ.get("FMOPS_ENV", "development")
        yaml_path = Path(
            os.environ.get("FMOPS_CONFIG_FILE", str(CONFIG_DIR / f"{env_name}.yaml"))
        )
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            YamlConfigSource(settings_cls, yaml_path),
            file_secret_settings,
        )

    @model_validator(mode="after")
    def _resolve_paths(self) -> Settings:
        object.__setattr__(self, "paths", self.paths.resolve())
        return self

    # --- derived helpers --------------------------------------------------- #
    @property
    def is_production(self) -> bool:
        return self.environment == "production"

    @property
    def db_path(self) -> Path:
        if self.database_url:
            return Path(self.database_url)
        return self.paths.state_dir / "fmops.db"

    @property
    def mlflow_tracking_uri(self) -> str:
        """Tracking store URI.

        Defaults to a local SQLite database rather than the ``file:`` store:
        MLflow 3.x put the filesystem backend into maintenance mode, and a
        SQL backend is also the shape used in production (RDS/Postgres), so
        local and deployed behaviour stay the same.
        """
        if self.tracking.tracking_uri:
            return self.tracking.tracking_uri
        mlruns = self.paths.artifacts_dir / "mlruns"
        mlruns.mkdir(parents=True, exist_ok=True)
        db_file = (mlruns / "mlflow.db").resolve()
        return f"sqlite:///{db_file.as_posix()}"

    @property
    def mlflow_artifact_root(self) -> str:
        """Where MLflow writes run artifacts (S3 when AWS is enabled)."""
        if self.aws.enabled and self.aws.s3_bucket:
            return f"s3://{self.aws.s3_bucket}/{self.aws.s3_prefix}/mlflow-artifacts"
        root = (self.paths.artifacts_dir / "mlruns" / "artifacts").resolve()
        root.mkdir(parents=True, exist_ok=True)
        return root.as_uri()

    @property
    def mlflow_registry_uri(self) -> str:
        return self.tracking.registry_uri or self.mlflow_tracking_uri

    def resolved_prompt_dir(self) -> Path:
        p = Path(self.llm.prompt_dir)
        return p if p.is_absolute() else PROJECT_ROOT / p

    def resolved_llm_eval_dir(self) -> Path:
        p = Path(self.llm.eval_dataset_dir)
        return p if p.is_absolute() else PROJECT_ROOT / p

    def redacted(self) -> dict[str, Any]:
        """Config dump safe to log or return over the API."""
        data = self.model_dump(mode="json")
        for key in (
            "aws_access_key_id",
            "aws_secret_access_key",
            "anthropic_api_key",
            "google_api_key",
            "openai_api_key",
        ):
            if data.get(key) is not None:
                data[key] = "***redacted***"
        if data.get("security", {}).get("api_keys"):
            data["security"]["api_keys"] = ["***redacted***"] * len(
                data["security"]["api_keys"]
            )
        if data.get("alerts", {}).get("webhook_url"):
            data["alerts"]["webhook_url"] = "***redacted***"
        return data


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide cached settings instance."""
    return Settings()


def reload_settings() -> Settings:
    """Clear the cache and re-read configuration (used by tests and the CLI)."""
    get_settings.cache_clear()
    return get_settings()
