"""Request/response schemas for the inference API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator

from app.core.utils import utcnow_iso


class LoanApplicationFeatures(BaseModel):
    """One scoring request for the reference loan-default model.

    Field constraints mirror the expectations enforced by the data-validation
    suite, so an input that would have been rejected at training time is
    rejected at inference time too.
    """

    model_config = {"extra": "forbid"}

    age: float = Field(..., ge=18, le=100)
    annual_income: float = Field(..., ge=0, le=5_000_000)
    loan_amount: float = Field(..., ge=100, le=2_000_000)
    loan_term_months: int = Field(..., ge=6, le=480)
    credit_score: float = Field(..., ge=300, le=850)
    debt_to_income: float = Field(..., ge=0, le=3.0)
    employment_years: float = Field(..., ge=0, le=60)
    num_credit_lines: int = Field(..., ge=0, le=60)
    num_late_payments_12m: int = Field(..., ge=0, le=40)
    credit_utilization: float = Field(..., ge=0, le=2.0)
    employment_type: str
    housing_status: str
    loan_purpose: str
    region: str

    @field_validator(
        "employment_type", "housing_status", "loan_purpose", "region", mode="before"
    )
    @classmethod
    def _strip(cls, value: Any) -> Any:
        return value.strip() if isinstance(value, str) else value


class PredictionRequest(BaseModel):
    features: LoanApplicationFeatures
    model_version: int | None = Field(
        default=None,
        description="Pin a specific version; defaults to the live production routing.",
    )
    threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    explain: bool = False


class ModelPredictionRequest(BaseModel):
    """Score one record with any registered model.

    ``features`` is checked against the serving version's recorded signature,
    not against a static schema: the contract is whatever the model was
    trained on. See ``GET /api/v1/models/{name}/signature``.
    """

    model_config = {"extra": "forbid"}

    features: dict[str, Any]
    model_version: int | None = Field(
        default=None, description="Pin a version; defaults to the model's live routing."
    )
    threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    explain: bool = False


class ModelBatchPredictionRequest(BaseModel):
    model_config = {"extra": "forbid"}

    instances: list[dict[str, Any]]
    model_version: int | None = None
    threshold: float | None = Field(default=None, ge=0.0, le=1.0)

    @field_validator("instances")
    @classmethod
    def _not_empty(cls, value: list) -> list:
        if not value:
            raise ValueError("instances must not be empty")
        return value


class BatchPredictionRequest(BaseModel):
    instances: list[LoanApplicationFeatures]
    model_version: int | None = None
    threshold: float | None = None

    @field_validator("instances")
    @classmethod
    def _not_empty(cls, value: list) -> list:
        if not value:
            raise ValueError("instances must not be empty")
        return value


class PredictionResponse(BaseModel):
    request_id: str
    prediction: int
    prediction_label: str
    positive_label: str = Field(
        default="", description="What `probability` is the probability of."
    )
    probability: float
    threshold: float
    model_name: str
    model_version: int
    model_stage: str
    variant: str = "primary"
    inference_latency_ms: float
    explanation: dict[str, float] | None = None
    warnings: list[str] = Field(
        default_factory=list,
        description="Accepted but notable: values outside the training range, unseen "
        "categories, imputed nulls.",
    )
    created_at: str = Field(default_factory=utcnow_iso)


class BatchPredictionResponse(BaseModel):
    request_id: str
    model_name: str
    model_version: int
    n_instances: int
    predictions: list[int]
    prediction_labels: list[str] = Field(default_factory=list)
    probabilities: list[float]
    threshold: float
    inference_latency_ms: float
    created_at: str = Field(default_factory=utcnow_iso)


class FeedbackRequest(BaseModel):
    """Ground truth arriving after the fact.

    Labels are what make concept-drift and live-performance measurement possible
    at all; without them the platform can only measure input and output
    distributions.
    """

    request_id: str
    # 0/1, or the model's own class label ("yes", "churned", ...). Resolved
    # against the signature of the model that served the request.
    actual_label: int | str
    source: str = "manual"

    @field_validator("actual_label")
    @classmethod
    def _label_shape(cls, value: int | str) -> int | str:
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, int) and value not in (0, 1):
            raise ValueError("a numeric label must be 0 or 1")
        if isinstance(value, str) and not value.strip():
            raise ValueError("label must not be empty")
        return value


class FeedbackResponse(BaseModel):
    request_id: str
    recorded: bool
    labelled_total: int
    model_name: str | None = None
    model_version: int | None = None
    actual_label: int | None = None
    replaced_previous: bool = False


class ModelInfoResponse(BaseModel):
    model_name: str
    model_version: int
    model_stage: str
    algorithm: str
    artifact_uri: str
    dataset_version: str | None = None
    git_commit: str = "unknown"
    metrics: dict[str, float] = Field(default_factory=dict)
    params: dict[str, Any] = Field(default_factory=dict)
    loaded_at: str | None = None
    feature_names: list[str] = Field(default_factory=list)
