"""Anthropic API provider.

Uses the Messages API. Credentials come only from ``ANTHROPIC_API_KEY`` in the
environment; the key is never read from a config file and never logged.
"""

from __future__ import annotations

from app.core.exceptions import DependencyMissingError, LLMProviderError, LLMRateLimitError
from app.core.logging import get_logger
from app.llmops.providers.base import LLMProvider
from app.schemas.llm import LLMRequest, LLMResponse

logger = get_logger(__name__)

DEFAULT_MODEL = "claude-haiku-4-5-20251001"


class AnthropicProvider(LLMProvider):
    name = "anthropic"

    def __init__(self, settings=None, client=None) -> None:
        super().__init__(settings)
        self._client = client

    def _api_key(self) -> str | None:
        secret = self.settings.anthropic_api_key
        return secret.get_secret_value() if secret else None

    @property
    def client(self):
        if self._client is None:
            try:
                import anthropic
            except ImportError as exc:
                raise DependencyMissingError(
                    "the anthropic package is required; install the [llm] extra",
                    provider=self.name,
                ) from exc
            self._client = anthropic.Anthropic(
                api_key=self._api_key(), timeout=self.config.timeout_seconds
            )
        return self._client

    def default_model(self) -> str:
        model = self.config.model
        if not model or model.startswith("mock"):
            return DEFAULT_MODEL
        return model

    def available(self) -> tuple[bool, str]:
        try:
            import anthropic  # noqa: F401
        except ImportError:
            return False, "the anthropic package is not installed (pip install -e '.[llm]')"
        if not self._api_key():
            return False, "ANTHROPIC_API_KEY is not set"
        return True, "ready"

    def _generate(self, request: LLMRequest) -> LLMResponse:
        model = request.model or self.default_model()
        messages = [
            {"role": m.role, "content": m.content}
            for m in request.messages
            if m.role in ("user", "assistant")
        ]
        if not messages:
            raise LLMProviderError(
                "anthropic requires at least one user message", provider=self.name
            )

        kwargs = {
            "model": model,
            "messages": messages,
            "max_tokens": request.max_tokens or self.config.max_tokens,
            "temperature": (
                request.temperature
                if request.temperature is not None
                else self.config.temperature
            ),
        }
        if request.system:
            kwargs["system"] = request.system
        if request.stop_sequences:
            kwargs["stop_sequences"] = request.stop_sequences

        try:
            message = self.client.messages.create(**kwargs)
        except Exception as exc:
            name = type(exc).__name__
            if "RateLimit" in name:
                raise LLMRateLimitError(
                    f"Anthropic rate limit: {exc}", provider=self.name, model=model
                ) from exc
            raise LLMProviderError(
                f"Anthropic request failed: {exc}", provider=self.name, model=model
            ) from exc

        text = "".join(
            block.text for block in message.content if getattr(block, "type", "") == "text"
        )
        return LLMResponse(
            text=text,
            model=model,
            provider=self.name,
            usage=self._usage(message.usage.input_tokens, message.usage.output_tokens),
            finish_reason=message.stop_reason or "stop",
            raw={"id": message.id, "stop_reason": message.stop_reason},
        )
