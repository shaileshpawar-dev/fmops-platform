"""Provider selection.

Fallback policy, stated explicitly because silent fallbacks are how teams end up
shipping mock output to production:

* The configured provider is used when it is available.
* If it is unavailable (missing SDK, missing key) in a **non-production**
  environment, the mock provider is used and an ERROR is logged naming exactly
  what is missing. Development should keep working without credentials.
* In **production**, an unavailable provider raises instead. Returning mock text
  to real users is never the right failure mode.
"""

from __future__ import annotations

from app.core.config import Settings, get_settings
from app.core.exceptions import LLMProviderError
from app.core.logging import get_logger
from app.llmops.providers.base import LLMProvider
from app.llmops.providers.mock import MockProvider

logger = get_logger(__name__)


def _construct(name: str, settings: Settings) -> LLMProvider:
    if name == "mock":
        return MockProvider(settings)
    if name == "bedrock":
        from app.llmops.providers.bedrock import BedrockProvider

        return BedrockProvider(settings)
    if name == "anthropic":
        from app.llmops.providers.anthropic_provider import AnthropicProvider

        return AnthropicProvider(settings)
    if name == "gemini":
        from app.llmops.providers.gemini import GeminiProvider

        return GeminiProvider(settings)
    if name == "openai_compatible":
        from app.llmops.providers.openai_compatible import OpenAICompatibleProvider

        return OpenAICompatibleProvider(settings)
    raise LLMProviderError(f"unknown LLM provider {name!r}", provider=name)


def build_provider(
    name: str | None = None,
    settings: Settings | None = None,
    allow_fallback: bool = True,
) -> LLMProvider:
    settings = settings or get_settings()
    requested = name or settings.llm.provider

    provider = _construct(requested, settings)
    ok, detail = provider.available()
    if ok:
        return provider

    if settings.is_production or not allow_fallback:
        raise LLMProviderError(
            f"LLM provider {requested!r} is not available: {detail}",
            provider=requested,
            environment=settings.environment,
        )

    logger.error(
        "llm.provider_unavailable_using_mock",
        extra={
            "requested_provider": requested,
            "reason": detail,
            "fallback": "mock",
            "impact": (
                "responses are deterministic placeholders, NOT model output; "
                "evaluation scores from this session measure the harness only"
            ),
        },
    )
    return MockProvider(settings)


def provider_status(settings: Settings | None = None) -> dict[str, dict[str, object]]:
    """Availability of every provider, for the API and the dashboard."""
    settings = settings or get_settings()
    status: dict[str, dict[str, object]] = {}
    for name in ("mock", "bedrock", "anthropic", "gemini", "openai_compatible"):
        try:
            provider = _construct(name, settings)
            ok, detail = provider.available()
            status[name] = {
                "available": ok,
                "detail": detail,
                "default_model": provider.default_model(),
                "configured": name == settings.llm.provider,
            }
        except Exception as exc:
            status[name] = {
                "available": False,
                "detail": str(exc),
                "configured": name == settings.llm.provider,
            }
    return status
