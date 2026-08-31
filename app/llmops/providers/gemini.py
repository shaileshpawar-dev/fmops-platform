"""Google Gemini provider (google-genai SDK).

Credentials come only from ``GOOGLE_API_KEY``. Gemini returns a usage metadata
block, so token counts are measured rather than estimated.
"""

from __future__ import annotations

from app.core.exceptions import DependencyMissingError, LLMProviderError, LLMRateLimitError
from app.core.logging import get_logger
from app.llmops.providers.base import LLMProvider
from app.schemas.llm import LLMRequest, LLMResponse

logger = get_logger(__name__)

DEFAULT_MODEL = "gemini-2.0-flash"


class GeminiProvider(LLMProvider):
    name = "gemini"

    def __init__(self, settings=None, client=None) -> None:
        super().__init__(settings)
        self._client = client

    def _api_key(self) -> str | None:
        secret = self.settings.google_api_key
        return secret.get_secret_value() if secret else None

    @property
    def client(self):
        if self._client is None:
            try:
                from google import genai
            except ImportError as exc:
                raise DependencyMissingError(
                    "google-genai is required; install the [llm] extra",
                    provider=self.name,
                ) from exc
            self._client = genai.Client(api_key=self._api_key())
        return self._client

    def default_model(self) -> str:
        model = self.config.model
        if not model or model.startswith("mock"):
            return DEFAULT_MODEL
        return model

    def available(self) -> tuple[bool, str]:
        try:
            from google import genai  # noqa: F401
        except ImportError:
            return False, "google-genai is not installed (pip install -e '.[llm]')"
        if not self._api_key():
            return False, "GOOGLE_API_KEY is not set"
        return True, "ready"

    def _generate(self, request: LLMRequest) -> LLMResponse:
        from google.genai import types

        model = request.model or self.default_model()
        contents = [
            types.Content(
                role="model" if m.role == "assistant" else "user",
                parts=[types.Part.from_text(text=m.content)],
            )
            for m in request.messages
            if m.role in ("user", "assistant")
        ]
        if not contents:
            raise LLMProviderError(
                "gemini requires at least one user message", provider=self.name
            )

        config = types.GenerateContentConfig(
            temperature=(
                request.temperature
                if request.temperature is not None
                else self.config.temperature
            ),
            max_output_tokens=request.max_tokens or self.config.max_tokens,
            system_instruction=request.system or None,
            stop_sequences=request.stop_sequences or None,
        )

        try:
            response = self.client.models.generate_content(
                model=model, contents=contents, config=config
            )
        except Exception as exc:
            text = str(exc).lower()
            if "429" in text or "quota" in text or "rate" in text:
                raise LLMRateLimitError(
                    f"Gemini rate limit: {exc}", provider=self.name, model=model
                ) from exc
            raise LLMProviderError(
                f"Gemini request failed: {exc}", provider=self.name, model=model
            ) from exc

        usage_meta = getattr(response, "usage_metadata", None)
        if usage_meta is not None:
            usage = self._usage(
                getattr(usage_meta, "prompt_token_count", 0) or 0,
                getattr(usage_meta, "candidates_token_count", 0) or 0,
            )
        else:
            usage = self._usage_from_text(request.prompt_text(), response.text or "")

        return LLMResponse(
            text=response.text or "",
            model=model,
            provider=self.name,
            usage=usage,
            finish_reason="stop",
            raw={},
        )
