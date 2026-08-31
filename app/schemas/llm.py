"""Schemas for the LLMOps layer: prompts, invocations, traces, evaluation, cost."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from app.core.utils import new_id, utcnow_iso


# --------------------------------------------------------------------------- #
# Prompts
# --------------------------------------------------------------------------- #
class PromptVersion(BaseModel):
    """An immutable, versioned prompt template.

    Prompts are code: they live in the repo under ``prompts/library``, carry a
    semantic version, and are hashed so a trace can prove exactly which text
    produced an output.
    """

    name: str
    version: str
    template: str
    system: str | None = None
    description: str = ""
    variables: list[str] = Field(default_factory=list)
    model: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    tags: list[str] = Field(default_factory=list)
    content_hash: str = ""
    git_commit: str = "unknown"
    created_at: str = Field(default_factory=utcnow_iso)

    @property
    def key(self) -> str:
        return f"{self.name}@{self.version}"


class PromptRenderResult(BaseModel):
    prompt: PromptVersion
    rendered: str
    system: str | None = None
    variables_used: dict[str, Any] = Field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Invocation
# --------------------------------------------------------------------------- #
class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class LLMRequest(BaseModel):
    """Provider-agnostic generation request."""

    messages: list[Message] = Field(default_factory=list)
    system: str | None = None
    model: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    stop_sequences: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_text(cls, text: str, system: str | None = None, **kw: Any) -> LLMRequest:
        return cls(messages=[Message(role="user", content=text)], system=system, **kw)

    def prompt_text(self) -> str:
        return "\n\n".join(f"{m.role}: {m.content}" for m in self.messages)


class TokenUsage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    estimated: bool = Field(
        default=False,
        description="True when the provider did not return usage and counts were approximated.",
    )

    def __add__(self, other: TokenUsage) -> TokenUsage:
        return TokenUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
            estimated=self.estimated or other.estimated,
        )


class LLMResponse(BaseModel):
    """Provider-agnostic generation response."""

    text: str
    model: str
    provider: str
    usage: TokenUsage = Field(default_factory=TokenUsage)
    latency_ms: float = 0.0
    finish_reason: str = "stop"
    raw: dict[str, Any] = Field(default_factory=dict)
    request_id: str = Field(default_factory=lambda: new_id("llm"))


class LLMGenerateRequest(BaseModel):
    """API-level request: either a raw prompt or a named prompt version."""

    prompt_name: str | None = None
    prompt_version: str | None = None
    variables: dict[str, Any] = Field(default_factory=dict)
    text: str | None = None
    system: str | None = None
    model: str | None = None
    provider: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    run_safety_check: bool = True


class SafetyFinding(BaseModel):
    check: str
    category: str
    severity: Literal["info", "low", "medium", "high"]
    triggered: bool
    evidence: str = ""
    detail: str = ""


class SafetyVerdict(BaseModel):
    """Result of the heuristic safety screen.

    This is a *heuristic* screen (pattern and keyword based), not a guarantee.
    See ``docs/llmops.md`` for the documented limitations.
    """

    passed: bool = True
    blocked: bool = False
    risk_score: float = 0.0
    findings: list[SafetyFinding] = Field(default_factory=list)
    checked: list[str] = Field(default_factory=list)
    detail: str = ""

    @property
    def triggered_findings(self) -> list[SafetyFinding]:
        return [f for f in self.findings if f.triggered]


class LLMTrace(BaseModel):
    """One end-to-end LLM interaction, persisted for audit and cost accounting."""

    id: str = Field(default_factory=lambda: new_id("trace"))
    request_id: str
    provider: str
    model: str
    prompt_name: str | None = None
    prompt_version: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    usage: TokenUsage = Field(default_factory=TokenUsage)
    latency_ms: float = 0.0
    estimated_cost_usd: float = 0.0
    status: str = "ok"
    error_code: str | None = None
    safety_verdict: str | None = None
    rendered_prompt: str | None = None
    output_text: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    git_commit: str = "unknown"
    created_at: str = Field(default_factory=utcnow_iso)


class LLMGenerateResponse(BaseModel):
    request_id: str
    text: str
    provider: str
    model: str
    prompt_name: str | None = None
    prompt_version: str | None = None
    usage: TokenUsage
    latency_ms: float
    estimated_cost_usd: float
    safety: SafetyVerdict | None = None
    trace_id: str


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
class LLMEvalCase(BaseModel):
    """One case in an evaluation dataset."""

    id: str
    input: dict[str, Any] = Field(default_factory=dict)
    reference: str | None = None
    context: str | None = None
    expected_keywords: list[str] = Field(default_factory=list)
    forbidden_keywords: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)


class LLMEvalDataset(BaseModel):
    name: str
    description: str = ""
    prompt_name: str | None = None
    cases: list[LLMEvalCase] = Field(default_factory=list)


class CaseScore(BaseModel):
    case_id: str
    output: str = ""
    scores: dict[str, float] = Field(default_factory=dict)
    passed: bool = True
    latency_ms: float = 0.0
    usage: TokenUsage = Field(default_factory=TokenUsage)
    cost_usd: float = 0.0
    safety: SafetyVerdict | None = None
    error: str | None = None


class LLMEvaluationResult(BaseModel):
    """Aggregate evaluation of one (provider, model, prompt version) triple."""

    id: str = Field(default_factory=lambda: new_id("lleval"))
    suite: str
    dataset: str
    provider: str
    model: str
    prompt_name: str
    prompt_version: str
    n_cases: int = 0
    aggregate: dict[str, float] = Field(default_factory=dict)
    per_case: list[CaseScore] = Field(default_factory=list)
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    created_at: str = Field(default_factory=utcnow_iso)

    @property
    def overall_score(self) -> float:
        return float(self.aggregate.get("overall", 0.0))


class LLMComparison(BaseModel):
    """A/B comparison between two evaluation runs (models or prompt versions)."""

    dimension: Literal["model", "prompt"]
    variant_a: str
    variant_b: str
    metric: str
    score_a: float
    score_b: float
    delta: float
    winner: str
    cost_a_usd: float = 0.0
    cost_b_usd: float = 0.0
    latency_a_ms: float = 0.0
    latency_b_ms: float = 0.0
    per_metric: dict[str, dict[str, float]] = Field(default_factory=dict)
    created_at: str = Field(default_factory=utcnow_iso)


# --------------------------------------------------------------------------- #
# Cost
# --------------------------------------------------------------------------- #
class CostRecord(BaseModel):
    model: str
    provider: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    input_cost_usd: float
    output_cost_usd: float
    total_cost_usd: float
    priced: bool = Field(
        default=True,
        description="False when no price table entry existed and cost is reported as 0.",
    )


class CostBucket(BaseModel):
    period: str
    requests: int = 0
    successful_requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    cost_per_request: float = 0.0
    cost_per_successful_response: float = 0.0


class CostSummary(BaseModel):
    daily: list[CostBucket] = Field(default_factory=list)
    weekly: list[CostBucket] = Field(default_factory=list)
    monthly: list[CostBucket] = Field(default_factory=list)
    by_model: dict[str, CostBucket] = Field(default_factory=dict)
    today_cost_usd: float = 0.0
    month_cost_usd: float = 0.0
    daily_budget_usd: float = 0.0
    monthly_budget_usd: float = 0.0
    daily_budget_used_pct: float = 0.0
    monthly_budget_used_pct: float = 0.0
    created_at: str = Field(default_factory=utcnow_iso)


# --------------------------------------------------------------------------- #
# LLM registry
# --------------------------------------------------------------------------- #
class LLMConfiguration(BaseModel):
    """A registered, versioned LLM 'model' = provider + model + prompt version.

    This is the LLMOps analogue of a registered ML model version: the exact,
    reproducible artifact that gets promoted to production.
    """

    name: str
    version: int
    stage: str = "Development"
    provider: str
    model: str
    prompt_name: str
    prompt_version: str
    params: dict[str, Any] = Field(default_factory=dict)
    evaluation_id: str | None = None
    metrics: dict[str, float] = Field(default_factory=dict)
    git_commit: str = "unknown"
    created_at: str = Field(default_factory=utcnow_iso)
    updated_at: str = Field(default_factory=utcnow_iso)
