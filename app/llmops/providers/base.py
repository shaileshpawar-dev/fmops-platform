"""LLM provider abstraction.

Every provider maps its vendor-specific request/response shape onto
:class:`~app.schemas.llm.LLMRequest` / :class:`~app.schemas.llm.LLMResponse`, so
the rest of the LLMOps layer -- prompts, tracing, evaluation, safety, cost --
never imports a vendor SDK.

Providers shipped:

* :class:`~app.llmops.providers.mock.MockProvider` -- deterministic, offline,
  needs no credentials. Everything works without an API key.
* Bedrock, Anthropic, Gemini, and any OpenAI-compatible endpoint.

Token accounting: real providers return usage; when one does not, the response
is marked ``usage.estimated=True`` and counts come from an approximation. Cost
figures derived from estimated tokens are therefore themselves estimates, and
the platform labels them as such rather than presenting them as billing truth.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod

from app.core.config import LLMConfig, Settings, get_settings
from app.core.exceptions import LLMProviderError, LLMRateLimitError
from app.core.logging import get_logger
from app.schemas.llm import LLMRequest, LLMResponse, TokenUsage

logger = get_logger(__name__)

# Rough token approximation used only when a provider returns no usage block.
# English prose averages ~4 characters per token for common BPE vocabularies.
CHARS_PER_TOKEN = 4.0


def estimate_tokens(text: str) -> int:
    """Approximate token count. Only used when real usage is unavailable."""
    if not text:
        return 0
    return max(1, int(len(text) / CHARS_PER_TOKEN))


class LLMProvider(ABC):
    """Generates completions from a provider-agnostic request."""

    name: str = "abstract"
    requires_credentials: bool = True

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.config: LLMConfig = self.settings.llm

    @abstractmethod
    def _generate(self, request: LLMRequest) -> LLMResponse:
        """Vendor call. Implementations must not retry; the base class does."""

    @abstractmethod
    def available(self) -> tuple[bool, str]:
        """Whether this provider can be used, and why not if it cannot."""

    def default_model(self) -> str:
        return self.config.model

    def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate with retries and consistent error mapping.

        Retries only transient failures (rate limits, timeouts). A malformed
        request is not retried -- retrying it just wastes budget.
        """
        ok, detail = self.available()
        if not ok:
            raise LLMProviderError(
                f"provider {self.name!r} is not available: {detail}",
                provider=self.name,
            )

        attempts = max(1, self.config.max_retries + 1)
        last_error: Exception | None = None

        for attempt in range(attempts):
            started = time.perf_counter()
            try:
                response = self._generate(request)
                if not response.latency_ms:
                    response.latency_ms = (time.perf_counter() - started) * 1000
                return response
            except LLMRateLimitError as exc:
                last_error = exc
                if attempt < attempts - 1:
                    backoff = 2.0**attempt
                    logger.warning(
                        "llm.rate_limited_retrying",
                        extra={
                            "provider": self.name,
                            "attempt": attempt + 1,
                            "backoff_seconds": backoff,
                        },
                    )
                    time.sleep(backoff)
                    continue
                raise
            except LLMProviderError as exc:
                last_error = exc
                if attempt < attempts - 1 and _is_transient(exc):
                    logger.warning(
                        "llm.transient_error_retrying",
                        extra={
                            "provider": self.name,
                            "attempt": attempt + 1,
                            "error": exc.message,
                        },
                    )
                    time.sleep(1.0 * (attempt + 1))
                    continue
                raise
            except Exception as exc:
                last_error = exc
                raise LLMProviderError(
                    f"{self.name} generation failed: {exc}",
                    provider=self.name,
                    model=request.model or self.default_model(),
                ) from exc

        raise LLMProviderError(
            f"{self.name} generation failed after {attempts} attempts: {last_error}",
            provider=self.name,
        )

    # -- helpers for subclasses --------------------------------------------- #
    def _usage_from_text(self, prompt: str, completion: str) -> TokenUsage:
        input_tokens = estimate_tokens(prompt)
        output_tokens = estimate_tokens(completion)
        return TokenUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            estimated=True,
        )

    @staticmethod
    def _usage(input_tokens: int, output_tokens: int) -> TokenUsage:
        return TokenUsage(
            input_tokens=int(input_tokens),
            output_tokens=int(output_tokens),
            total_tokens=int(input_tokens) + int(output_tokens),
            estimated=False,
        )

    def __repr__(self) -> str:  # pragma: no cover
        return f"<{type(self).__name__} model={self.default_model()}>"


def _is_transient(exc: Exception) -> bool:
    text = str(exc).lower()
    return any(
        marker in text
        for marker in (
            "timeout",
            "timed out",
            "connection",
            "temporarily",
            "503",
            "502",
            "504",
            "overloaded",
            "throttl",
        )
    )
