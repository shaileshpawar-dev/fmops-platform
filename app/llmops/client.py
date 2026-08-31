"""The LLM invocation path.

One place where an LLM call happens, so every call is uniformly:

    prompt version -> render -> safety screen (input)
        -> provider invoke -> safety screen (output)
        -> token accounting -> cost -> trace -> metrics

Nothing bypasses this. That is what makes the LLM layer as governable as the ML
layer: there is exactly one chokepoint where policy applies and one place where
evidence is recorded.
"""

from __future__ import annotations

import time
from typing import Any

from app.core.config import Settings, get_settings
from app.core.exceptions import LLMProviderError, SafetyViolationError
from app.core.logging import get_logger, new_request_id
from app.llmops.cost import CostCalculator
from app.llmops.prompts.registry import PromptRegistry, get_prompt_registry
from app.llmops.providers.base import LLMProvider
from app.llmops.providers.factory import build_provider
from app.llmops.safety.checks import SafetyScreen, get_safety_screen
from app.llmops.token_tracking import TraceStore, get_trace_store
from app.monitoring.metrics import record_llm_call
from app.schemas.llm import (
    LLMGenerateRequest,
    LLMGenerateResponse,
    LLMRequest,
    LLMResponse,
    Message,
    SafetyVerdict,
    TokenUsage,
)

logger = get_logger(__name__)


class LLMClient:
    """Governed LLM invocation."""

    def __init__(
        self,
        provider: LLMProvider | None = None,
        prompts: PromptRegistry | None = None,
        safety: SafetyScreen | None = None,
        traces: TraceStore | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self._provider = provider
        self._prompts = prompts
        self._safety = safety
        self._traces = traces
        self.cost = CostCalculator(self.settings)

    @property
    def provider(self) -> LLMProvider:
        if self._provider is None:
            self._provider = build_provider(settings=self.settings)
        return self._provider

    @property
    def prompts(self) -> PromptRegistry:
        if self._prompts is None:
            self._prompts = get_prompt_registry()
        return self._prompts

    @property
    def safety(self) -> SafetyScreen:
        if self._safety is None:
            self._safety = get_safety_screen()
        return self._safety

    @property
    def traces(self) -> TraceStore:
        if self._traces is None:
            self._traces = get_trace_store()
        return self._traces

    # -- main entry point ----------------------------------------------------- #
    def generate(
        self, request: LLMGenerateRequest, request_id: str | None = None
    ) -> LLMGenerateResponse:
        request_id = request_id or new_request_id()
        settings = self.settings
        provider = (
            build_provider(request.provider, settings) if request.provider else self.provider
        )

        # ---- 1. resolve the prompt ---------------------------------------- #
        prompt_name: str | None = None
        prompt_version: str | None = None
        system = request.system
        if request.prompt_name:
            rendered = self.prompts.render(
                request.prompt_name, request.variables, request.prompt_version
            )
            user_text = rendered.rendered
            system = system or rendered.system
            prompt_name = rendered.prompt.name
            prompt_version = rendered.prompt.version
            model = request.model or rendered.prompt.model or provider.default_model()
            temperature = (
                request.temperature
                if request.temperature is not None
                else rendered.prompt.temperature
            )
            max_tokens = request.max_tokens or rendered.prompt.max_tokens
        elif request.text:
            user_text = request.text
            model = request.model or provider.default_model()
            temperature = request.temperature
            max_tokens = request.max_tokens
        else:
            raise LLMProviderError(
                "either prompt_name (with variables) or text must be supplied",
                provider=provider.name,
            )

        temperature = temperature if temperature is not None else settings.llm.temperature
        max_tokens = max_tokens or settings.llm.max_tokens

        # ---- 2. screen the input ------------------------------------------ #
        input_verdict: SafetyVerdict | None = None
        if settings.llm.safety_enabled and request.run_safety_check:
            input_verdict = self.safety.screen_input(user_text)
            if input_verdict.blocked:
                self._record_blocked(
                    request_id,
                    provider,
                    model,
                    prompt_name,
                    prompt_version,
                    temperature,
                    max_tokens,
                    user_text,
                    input_verdict,
                )
                raise SafetyViolationError(
                    "the request was blocked by the safety screen: "
                    + "; ".join(f.detail for f in input_verdict.triggered_findings),
                    request_id=request_id,
                    risk_score=input_verdict.risk_score,
                    findings=[f.check for f in input_verdict.triggered_findings],
                )

        # ---- 3. invoke ------------------------------------------------------ #
        llm_request = LLMRequest(
            messages=[Message(role="user", content=user_text)],
            system=system,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        started = time.perf_counter()
        try:
            response: LLMResponse = provider.generate(llm_request)
        except Exception as exc:
            latency_ms = (time.perf_counter() - started) * 1000
            code = getattr(exc, "code", "llm_provider_error")
            self.traces.record(
                request_id=request_id,
                provider=provider.name,
                model=model,
                usage=TokenUsage(),
                latency_ms=latency_ms,
                estimated_cost_usd=0.0,
                prompt_name=prompt_name,
                prompt_version=prompt_version,
                temperature=temperature,
                max_tokens=max_tokens,
                status="error",
                error_code=code,
                rendered_prompt=user_text,
                metadata={"error": str(exc)},
            )
            record_llm_call(
                provider.name,
                model,
                prompt_name or "adhoc",
                prompt_version or "none",
                0,
                0,
                0.0,
                latency_ms / 1000.0,
                outcome="error",
            )
            raise

        # ---- 4. screen the output -------------------------------------------- #
        output_verdict: SafetyVerdict | None = None
        if settings.llm.safety_enabled and request.run_safety_check:
            context = " ".join(str(v) for v in (request.variables or {}).values())
            output_verdict = self.safety.screen_output(response.text, context)

        verdict = _merge_verdicts(input_verdict, output_verdict)

        # ---- 5. cost + trace + metrics ---------------------------------------- #
        cost_record = self.cost.compute(response.model, provider.name, response.usage)
        trace = self.traces.record(
            request_id=request_id,
            provider=provider.name,
            model=response.model,
            usage=response.usage,
            latency_ms=response.latency_ms,
            estimated_cost_usd=cost_record.total_cost_usd,
            prompt_name=prompt_name,
            prompt_version=prompt_version,
            temperature=temperature,
            max_tokens=max_tokens,
            status="ok",
            safety_verdict=_verdict_label(verdict),
            rendered_prompt=user_text,
            output_text=response.text,
            metadata={
                "priced": cost_record.priced,
                "finish_reason": response.finish_reason,
                "risk_score": verdict.risk_score if verdict else 0.0,
            },
        )
        record_llm_call(
            provider.name,
            response.model,
            prompt_name or "adhoc",
            prompt_version or "none",
            response.usage.input_tokens,
            response.usage.output_tokens,
            cost_record.total_cost_usd,
            response.latency_ms / 1000.0,
        )

        return LLMGenerateResponse(
            request_id=request_id,
            text=response.text,
            provider=provider.name,
            model=response.model,
            prompt_name=prompt_name,
            prompt_version=prompt_version,
            usage=response.usage,
            latency_ms=round(response.latency_ms, 3),
            estimated_cost_usd=cost_record.total_cost_usd,
            safety=verdict,
            trace_id=trace.id,
        )

    def _record_blocked(
        self,
        request_id: str,
        provider: LLMProvider,
        model: str,
        prompt_name: str | None,
        prompt_version: str | None,
        temperature: float | None,
        max_tokens: int | None,
        text: str,
        verdict: SafetyVerdict,
    ) -> None:
        self.traces.record(
            request_id=request_id,
            provider=provider.name,
            model=model,
            usage=TokenUsage(),
            latency_ms=0.0,
            estimated_cost_usd=0.0,
            prompt_name=prompt_name,
            prompt_version=prompt_version,
            temperature=temperature,
            max_tokens=max_tokens,
            status="blocked",
            error_code="safety_violation",
            safety_verdict="blocked",
            rendered_prompt=text,
            metadata={
                "findings": [f.check for f in verdict.triggered_findings],
                "risk_score": verdict.risk_score,
            },
        )
        record_llm_call(
            provider.name,
            model,
            prompt_name or "adhoc",
            prompt_version or "none",
            0,
            0,
            0.0,
            0.0,
            outcome="blocked",
        )

    # -- convenience ---------------------------------------------------------- #
    def complete(self, text: str, **kwargs: Any) -> str:
        """Simple text-in/text-out helper, still fully traced."""
        return self.generate(LLMGenerateRequest(text=text, **kwargs)).text


def _merge_verdicts(a: SafetyVerdict | None, b: SafetyVerdict | None) -> SafetyVerdict | None:
    if a is None:
        return b
    if b is None:
        return a
    return SafetyVerdict(
        passed=a.passed and b.passed,
        blocked=a.blocked or b.blocked,
        risk_score=min(1.0, round(max(a.risk_score, b.risk_score), 4)),
        findings=[*a.findings, *b.findings],
        checked=[*a.checked, *b.checked],
        detail=a.detail,
    )


def _verdict_label(verdict: SafetyVerdict | None) -> str | None:
    if verdict is None:
        return None
    if verdict.blocked:
        return "blocked"
    if not verdict.passed:
        return "flagged"
    return "passed"


_CLIENT: LLMClient | None = None


def get_llm_client() -> LLMClient:
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = LLMClient()
    return _CLIENT


def set_llm_client(client: LLMClient | None) -> None:
    global _CLIENT
    _CLIENT = client
