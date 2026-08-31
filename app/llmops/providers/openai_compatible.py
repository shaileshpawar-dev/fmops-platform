"""OpenAI-compatible provider.

Speaks the ``/v1/chat/completions`` contract, which is implemented by OpenAI
itself and by most self-hosted servers (vLLM, Ollama, LM Studio, Together,
Groq, LiteLLM...). Point ``FMOPS_OPENAI_BASE_URL`` at any of them.

Implemented with httpx rather than the openai SDK so the platform can talk to a
local server with no vendor package installed at all.
"""

from __future__ import annotations

from app.core.exceptions import LLMProviderError, LLMRateLimitError
from app.core.logging import get_logger
from app.llmops.providers.base import LLMProvider
from app.schemas.llm import LLMRequest, LLMResponse

logger = get_logger(__name__)

DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_BASE_URL = "https://api.openai.com/v1"


class OpenAICompatibleProvider(LLMProvider):
    name = "openai_compatible"

    def __init__(self, settings=None, base_url: str | None = None) -> None:
        super().__init__(settings)
        self._base_url = base_url

    @property
    def base_url(self) -> str:
        return (self._base_url or self.settings.openai_base_url or DEFAULT_BASE_URL).rstrip(
            "/"
        )

    def _api_key(self) -> str | None:
        secret = self.settings.openai_api_key
        return secret.get_secret_value() if secret else None

    def default_model(self) -> str:
        model = self.config.model
        if not model or model.startswith("mock"):
            return DEFAULT_MODEL
        return model

    def available(self) -> tuple[bool, str]:
        # A self-hosted server on localhost normally needs no key, so an absent
        # key is only fatal when talking to a remote endpoint.
        is_local = any(host in self.base_url for host in ("localhost", "127.0.0.1", "0.0.0.0"))
        if not self._api_key() and not is_local:
            return False, (
                "OPENAI_API_KEY is not set and the base URL is not a local server "
                f"({self.base_url})"
            )
        return True, "ready"

    def _generate(self, request: LLMRequest) -> LLMResponse:
        import httpx

        model = request.model or self.default_model()
        messages = []
        if request.system:
            messages.append({"role": "system", "content": request.system})
        messages.extend({"role": m.role, "content": m.content} for m in request.messages)
        if not messages:
            raise LLMProviderError(
                "openai-compatible providers require at least one message",
                provider=self.name,
            )

        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": request.max_tokens or self.config.max_tokens,
            "temperature": (
                request.temperature
                if request.temperature is not None
                else self.config.temperature
            ),
        }
        if request.stop_sequences:
            payload["stop"] = request.stop_sequences

        headers = {"Content-Type": "application/json"}
        key = self._api_key()
        if key:
            headers["Authorization"] = f"Bearer {key}"

        try:
            response = httpx.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers=headers,
                timeout=self.config.timeout_seconds,
            )
        except httpx.HTTPError as exc:
            raise LLMProviderError(
                f"could not reach {self.base_url}: {exc}",
                provider=self.name,
                model=model,
            ) from exc

        if response.status_code == 429:
            raise LLMRateLimitError(
                "rate limited by the upstream endpoint",
                provider=self.name,
                model=model,
            )
        if response.status_code >= 400:
            raise LLMProviderError(
                f"upstream returned {response.status_code}: {response.text[:300]}",
                provider=self.name,
                model=model,
                status_code=response.status_code,
            )

        body = response.json()
        choices = body.get("choices") or []
        if not choices:
            raise LLMProviderError(
                "upstream returned no choices", provider=self.name, model=model
            )
        text = choices[0].get("message", {}).get("content") or ""
        usage_block = body.get("usage") or {}

        if usage_block:
            usage = self._usage(
                usage_block.get("prompt_tokens", 0),
                usage_block.get("completion_tokens", 0),
            )
        else:
            # Some self-hosted servers omit usage entirely.
            usage = self._usage_from_text(request.prompt_text(), text)

        return LLMResponse(
            text=text,
            model=body.get("model", model),
            provider=self.name,
            usage=usage,
            finish_reason=choices[0].get("finish_reason", "stop"),
            raw={"id": body.get("id")},
        )
