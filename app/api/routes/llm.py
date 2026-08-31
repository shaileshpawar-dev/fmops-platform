"""LLMOps endpoints."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Body, Query, Request

from app.core.config import get_settings
from app.core.logging import get_logger, new_request_id
from app.llmops.client import get_llm_client
from app.llmops.cost import get_cost_tracker
from app.llmops.evaluation.runner import (
    EvaluationRunner,
    compare,
    discover_datasets,
    get_evaluation,
    recent_evaluations,
)
from app.llmops.prompts.registry import get_prompt_registry
from app.llmops.providers.factory import provider_status
from app.llmops.safety.checks import get_safety_screen
from app.llmops.token_tracking import get_trace_store
from app.schemas.llm import (
    CostSummary,
    LLMComparison,
    LLMEvaluationResult,
    LLMGenerateRequest,
    LLMGenerateResponse,
    LLMTrace,
    PromptVersion,
    SafetyVerdict,
)

logger = get_logger(__name__)
router = APIRouter(prefix="/api/v1/llm", tags=["llmops"])


# --------------------------------------------------------------------------- #
# Invocation
# --------------------------------------------------------------------------- #
@router.post("/generate", response_model=LLMGenerateResponse, summary="Invoke an LLM")
def generate(payload: LLMGenerateRequest, request: Request) -> LLMGenerateResponse:
    """Generate a completion through the governed path.

    Every call is prompt-versioned, safety-screened, traced, and cost-accounted.
    With no API key configured the deterministic mock provider is used in
    non-production environments -- and the response's ``provider`` field says so.
    """
    request_id = getattr(request.state, "request_id", None) or new_request_id()
    return get_llm_client().generate(payload, request_id=request_id)


@router.get("/providers", summary="Provider availability")
def providers() -> dict[str, Any]:
    """Which providers can be used here, and why the others cannot."""
    settings = get_settings()
    return {
        "configured": settings.llm.provider,
        "configured_model": settings.llm.model,
        "providers": provider_status(settings),
    }


# --------------------------------------------------------------------------- #
# Prompts
# --------------------------------------------------------------------------- #
@router.get("/prompts", summary="List prompts and their versions")
def list_prompts() -> dict[str, Any]:
    registry = get_prompt_registry()
    out: dict[str, Any] = {}
    for name in registry.list_names():
        versions = registry.list_versions(name)
        out[name] = {
            "latest": versions[-1].version,
            "versions": [
                {
                    "version": v.version,
                    "description": v.description,
                    "variables": v.variables,
                    "tags": v.tags,
                    "content_hash": v.content_hash[:12],
                }
                for v in versions
            ],
        }
    return {"directory": str(registry.directory), "prompts": out}


@router.get(
    "/prompts/{name}", response_model=list[PromptVersion], summary="Prompt version history"
)
def get_prompt_versions(name: str) -> list[PromptVersion]:
    return get_prompt_registry().list_versions(name)


@router.get(
    "/prompts/{name}/{version}", response_model=PromptVersion, summary="Get one prompt version"
)
def get_prompt(name: str, version: str) -> PromptVersion:
    return get_prompt_registry().get(name, version)


@router.get("/prompts/{name}/diff/{version_a}/{version_b}", summary="Diff two prompt versions")
def diff_prompts(name: str, version_a: str, version_b: str) -> dict[str, Any]:
    return get_prompt_registry().diff(name, version_a, version_b)


@router.post("/prompts/{name}/render", summary="Render a prompt without invoking a model")
def render_prompt(
    name: str,
    variables: dict[str, Any] = Body(default_factory=dict),
    version: str | None = Query(default=None),
) -> dict[str, Any]:
    """Preview exactly what would be sent to the model. Costs nothing."""
    result = get_prompt_registry().render(name, variables, version)
    return {
        "prompt": result.prompt.key,
        "content_hash": result.prompt.content_hash[:12],
        "system": result.system,
        "rendered": result.rendered,
        "variables_used": result.variables_used,
    }


# --------------------------------------------------------------------------- #
# Traces and tokens
# --------------------------------------------------------------------------- #
@router.get("/traces", response_model=list[LLMTrace], summary="Recent LLM traces")
def traces(
    limit: int = Query(default=50, le=500),
    model: str | None = None,
    prompt_name: str | None = None,
    status: str | None = None,
) -> list[LLMTrace]:
    return get_trace_store().recent(limit, model, prompt_name, status)


@router.get("/traces/{trace_id}", summary="Get one trace")
def get_trace(trace_id: str) -> dict[str, Any]:
    trace = get_trace_store().get(trace_id)
    if trace is None:
        return {"trace_id": trace_id, "found": False}
    return {"found": True, "trace": trace.model_dump(mode="json")}


@router.get("/tokens", summary="Token usage totals")
def tokens(days: int = Query(default=30, ge=1, le=365)) -> dict[str, Any]:
    store = get_trace_store()
    return {
        "totals": store.token_totals(days),
        "by_prompt_version": store.by_prompt_version(days),
    }


# --------------------------------------------------------------------------- #
# Cost
# --------------------------------------------------------------------------- #
@router.get("/cost", response_model=CostSummary, summary="Cost summary")
def cost() -> CostSummary:
    """Daily / weekly / monthly spend, per-model breakdown and budget usage.

    Figures are estimates derived from token counts and the configured price
    table; they will not match a vendor invoice exactly.
    """
    return get_cost_tracker().summary()


@router.post("/cost/check-budgets", summary="Evaluate budgets and raise alerts")
def check_budgets() -> dict[str, Any]:
    breaches = get_cost_tracker().check_budgets()
    return {"breaches": breaches, "healthy": not breaches}


# --------------------------------------------------------------------------- #
# Safety
# --------------------------------------------------------------------------- #
@router.post("/safety/screen", response_model=SafetyVerdict, summary="Screen text")
def screen(
    text: str = Body(..., embed=True),
    output: str = Body(default="", embed=True),
    context: str = Body(default="", embed=True),
) -> SafetyVerdict:
    """Run the heuristic safety screen without invoking a model.

    This is a pattern-based screen, not a content-safety classifier. It will
    miss paraphrased and encoded attacks. See docs/llmops.md for the full
    statement of limitations.
    """
    return get_safety_screen().screen(text, output, context)


@router.get("/safety/checks", summary="Which safety checks are active")
def safety_checks() -> dict[str, Any]:
    screen_instance = get_safety_screen()
    return {
        "enabled": get_settings().llm.safety_enabled,
        "block_on_violation": get_settings().llm.safety_block_on_violation,
        "limitations": (
            "Heuristic pattern matching only. No model, no measured recall. "
            "Cannot detect hallucination (only stylistic indicators). Production "
            "systems need a dedicated moderation model in addition."
        ),
        "checks": [
            {
                "name": c.name,
                "category": c.category,
                "severity": c.severity,
                "applies_to": [
                    d
                    for d, on in (("input", c.checks_input), ("output", c.checks_output))
                    if on
                ],
            }
            for c in screen_instance.checks
        ],
    }


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
@router.get("/evaluations", summary="Recent evaluation runs")
def list_evaluations(limit: int = Query(default=20, le=100)) -> dict[str, Any]:
    return {
        "datasets": sorted(discover_datasets().keys()),
        "evaluations": recent_evaluations(limit),
    }


@router.get("/evaluations/{evaluation_id}", summary="Get one evaluation run")
def get_evaluation_result(evaluation_id: str) -> dict[str, Any]:
    result = get_evaluation(evaluation_id)
    if result is None:
        return {"evaluation_id": evaluation_id, "found": False}
    return {"found": True, "evaluation": result.model_dump(mode="json")}


@router.post(
    "/evaluations/run", response_model=LLMEvaluationResult, summary="Run an evaluation suite"
)
def run_evaluation(
    dataset: str = Query(default="support_triage"),
    prompt_name: str | None = None,
    prompt_version: str | None = None,
    model: str | None = None,
    provider: str | None = None,
    suite: str = Query(default="api"),
) -> LLMEvaluationResult:
    """Score a prompt version against an evaluation dataset."""
    return EvaluationRunner().run(
        dataset,
        prompt_name=prompt_name,
        prompt_version=prompt_version,
        model=model,
        provider=provider,
        suite=suite,
    )


@router.post("/evaluations/compare", response_model=LLMComparison, summary="A/B two runs")
def compare_evaluations(
    evaluation_a: str = Body(..., embed=True),
    evaluation_b: str = Body(..., embed=True),
    metric: str = Body(default="overall", embed=True),
) -> LLMComparison:
    """Compare two evaluation runs (model vs model, or prompt vs prompt).

    Refuses to compare runs from different providers -- mock-provider scores
    measure the harness, not model quality.
    """
    a = get_evaluation(evaluation_a)
    b = get_evaluation(evaluation_b)
    if a is None or b is None:
        from app.core.exceptions import FMOpsError

        raise FMOpsError(
            "one or both evaluation ids were not found",
            evaluation_a=evaluation_a,
            evaluation_b=evaluation_b,
        )
    return compare(a, b, metric)
