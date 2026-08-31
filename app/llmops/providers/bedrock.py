"""Amazon Bedrock provider.

Uses the Bedrock Converse API, which gives one message shape across the model
families Bedrock hosts (Anthropic, Meta, Mistral, Amazon) and returns a real
usage block -- so token counts and therefore costs are measured, not estimated.

Requires the ``[aws]`` extra and normal AWS credential resolution (instance
role, SSO, environment). No key is ever read from configuration files.
"""

from __future__ import annotations

from app.core.exceptions import DependencyMissingError, LLMProviderError, LLMRateLimitError
from app.core.logging import get_logger
from app.llmops.providers.base import LLMProvider
from app.schemas.llm import LLMRequest, LLMResponse

logger = get_logger(__name__)


class BedrockProvider(LLMProvider):
    name = "bedrock"

    def __init__(self, settings=None, client=None) -> None:
        super().__init__(settings)
        self._client = client

    @property
    def region(self) -> str:
        return self.settings.aws.bedrock_region or self.settings.aws.region

    @property
    def client(self):
        if self._client is None:
            try:
                import boto3
            except ImportError as exc:
                raise DependencyMissingError(
                    "boto3 is required for Bedrock; install the [aws] extra",
                    provider=self.name,
                ) from exc
            self._client = boto3.client("bedrock-runtime", region_name=self.region)
        return self._client

    def available(self) -> tuple[bool, str]:
        try:
            import boto3  # noqa: F401
        except ImportError:
            return False, "boto3 is not installed (pip install -e '.[aws]')"
        if not self.settings.aws.enabled:
            return False, "AWS is disabled (set FMOPS_AWS__ENABLED=true)"
        try:
            import botocore.session

            credentials = botocore.session.get_session().get_credentials()
            if credentials is None:
                return False, "no AWS credentials could be resolved"
        except Exception as exc:  # pragma: no cover
            return False, f"AWS credential lookup failed: {exc}"
        return True, "ready"

    def _generate(self, request: LLMRequest) -> LLMResponse:
        model_id = request.model or self.default_model()
        messages = [
            {"role": m.role, "content": [{"text": m.content}]}
            for m in request.messages
            if m.role in ("user", "assistant")
        ]
        if not messages:
            raise LLMProviderError(
                "bedrock requires at least one user message", provider=self.name
            )

        kwargs = {
            "modelId": model_id,
            "messages": messages,
            "inferenceConfig": {
                "maxTokens": request.max_tokens or self.config.max_tokens,
                "temperature": (
                    request.temperature
                    if request.temperature is not None
                    else self.config.temperature
                ),
            },
        }
        if request.system:
            kwargs["system"] = [{"text": request.system}]
        if request.stop_sequences:
            kwargs["inferenceConfig"]["stopSequences"] = request.stop_sequences

        try:
            response = self.client.converse(**kwargs)
        except Exception as exc:
            name = type(exc).__name__
            if "Throttling" in name or "TooManyRequests" in name:
                raise LLMRateLimitError(
                    f"Bedrock throttled the request: {exc}",
                    provider=self.name,
                    model=model_id,
                ) from exc
            raise LLMProviderError(
                f"Bedrock converse failed: {exc}", provider=self.name, model=model_id
            ) from exc

        content = response.get("output", {}).get("message", {}).get("content", [])
        text = "".join(block.get("text", "") for block in content)
        usage_block = response.get("usage", {})

        return LLMResponse(
            text=text,
            model=model_id,
            provider=self.name,
            usage=self._usage(
                usage_block.get("inputTokens", 0), usage_block.get("outputTokens", 0)
            ),
            latency_ms=float(response.get("metrics", {}).get("latencyMs", 0) or 0),
            finish_reason=response.get("stopReason", "stop"),
            raw={"stopReason": response.get("stopReason")},
        )
