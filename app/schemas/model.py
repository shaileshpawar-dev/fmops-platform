"""Schemas for training runs, model versions and the registry."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from app.core.utils import utcnow_iso
from app.schemas.common import ModelStage, ModelStatus


class TrainingRequest(BaseModel):
    """Everything needed to reproduce a training run."""

    dataset_path: str | None = None
    dataset_version: str | None = None
    algorithm: str | None = None
    hyperparameters: dict[str, Any] = Field(default_factory=dict)
    tune: bool | None = None
    register_model: bool = True
    experiment_name: str | None = None
    run_name: str | None = None
    tags: dict[str, str] = Field(default_factory=dict)


class Metrics(BaseModel):
    """Classification metrics tracked for every model version.

    ``inference_latency_p95_ms`` is measured during evaluation on a fixed batch
    so it is comparable across versions; it is not a production SLO reading.
    """

    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    roc_auc: float = 0.0
    pr_auc: float = 0.0
    log_loss: float = 0.0
    brier_score: float = 0.0
    inference_latency_p50_ms: float = 0.0
    inference_latency_p95_ms: float = 0.0
    n_samples: int = 0
    positive_rate: float = 0.0

    def as_dict(self) -> dict[str, float]:
        return {k: float(v) for k, v in self.model_dump().items()}

    def get(self, name: str, default: float = 0.0) -> float:
        return float(self.model_dump().get(name, default))


class ConfusionMatrix(BaseModel):
    true_negative: int = 0
    false_positive: int = 0
    false_negative: int = 0
    true_positive: int = 0

    def as_dict(self) -> dict[str, int]:
        return self.model_dump()


class EvaluationResult(BaseModel):
    """The complete evaluation of one model against one dataset."""

    model_name: str
    model_version: int | None = None
    dataset_version: str | None = None
    split: str = "test"
    metrics: Metrics = Field(default_factory=Metrics)
    confusion_matrix: ConfusionMatrix = Field(default_factory=ConfusionMatrix)
    threshold: float = 0.5
    per_class: dict[str, dict[str, float]] = Field(default_factory=dict)
    feature_importance: dict[str, float] = Field(default_factory=dict)
    calibration: dict[str, list[float]] = Field(default_factory=dict)
    created_at: str = Field(default_factory=utcnow_iso)


class TrialResult(BaseModel):
    """One hyperparameter-search trial."""

    trial_id: int
    params: dict[str, Any]
    score: float
    metric: str
    duration_seconds: float = 0.0
    status: str = "completed"
    error: str | None = None


class TuningResult(BaseModel):
    backend: str
    metric: str
    direction: str
    n_trials: int
    best_params: dict[str, Any] = Field(default_factory=dict)
    best_score: float = 0.0
    trials: list[TrialResult] = Field(default_factory=list)
    duration_seconds: float = 0.0


class ModelVersion(BaseModel):
    """A registered model version -- the unit the platform deploys."""

    name: str
    version: int
    stage: ModelStage = ModelStage.DEVELOPMENT
    status: ModelStatus = ModelStatus.PENDING
    run_id: str | None = None
    artifact_uri: str
    dataset_version: str | None = None
    dataset_hash: str | None = None
    git_commit: str = "unknown"
    algorithm: str = ""
    params: dict[str, Any] = Field(default_factory=dict)
    metrics: dict[str, float] = Field(default_factory=dict)
    tags: dict[str, str] = Field(default_factory=dict)
    description: str = ""
    created_at: str = Field(default_factory=utcnow_iso)
    updated_at: str = Field(default_factory=utcnow_iso)
    created_by: str | None = None

    @property
    def key(self) -> str:
        return f"{self.name}:{self.version}"


class StageTransition(BaseModel):
    name: str
    version: int
    from_stage: ModelStage | None = None
    to_stage: ModelStage
    reason: str = ""
    actor: str = "system"
    created_at: str = Field(default_factory=utcnow_iso)


class TrainingRunResult(BaseModel):
    """Everything the training pipeline produces for one run."""

    run_id: str
    experiment_name: str
    model_name: str
    algorithm: str
    params: dict[str, Any] = Field(default_factory=dict)
    dataset_version: str | None = None
    dataset_hash: str | None = None
    git_commit: str = "unknown"
    model_path: str = ""
    artifact_uri: str = ""
    evaluation: EvaluationResult | None = None
    tuning: TuningResult | None = None
    validation_passed: bool = True
    registered_version: int | None = None
    duration_seconds: float = 0.0
    environment: dict[str, str] = Field(default_factory=dict)
    created_at: str = Field(default_factory=utcnow_iso)
