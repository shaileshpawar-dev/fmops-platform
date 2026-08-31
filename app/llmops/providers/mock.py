"""Deterministic offline LLM provider.

This is what makes "the project works with no API key" true rather than
aspirational. It is *not* a language model: it is a deterministic responder that
produces plausible, structurally correct output for the prompt templates this
project ships, so the whole LLMOps pipeline -- prompt versioning, tracing,
evaluation, safety screening, token and cost accounting -- can be exercised and
tested end to end, offline and reproducibly.

What it is honest about:

* Output quality is **not** representative of a real model. Evaluation scores
  produced against the mock provider measure the *harness*, not model quality.
  Every evaluation result records the provider, so mock-derived scores are never
  mistaken for real ones.
* Its pricing entry is 0.00, so cost dashboards show zero spend rather than
  fictional spend.
* Latency is simulated with a small deterministic delay so latency plumbing has
  something to measure.
"""

from __future__ import annotations

import hashlib
import random
import re
import time

from app.core.logging import get_logger
from app.llmops.providers.base import LLMProvider
from app.schemas.llm import LLMRequest, LLMResponse

logger = get_logger(__name__)

_SENTIMENTS = ("positive", "neutral", "negative")
_CATEGORIES = (
    "billing",
    "technical_issue",
    "account_access",
    "feature_request",
    "cancellation",
)
_PRIORITIES = ("low", "medium", "high", "urgent")


class MockProvider(LLMProvider):
    """Deterministic responder. Same input always yields the same output."""

    name = "mock"
    requires_credentials = False

    def available(self) -> tuple[bool, str]:
        return True, "mock provider is always available"

    def default_model(self) -> str:
        return self.config.model or "mock-small"

    def _generate(self, request: LLMRequest) -> LLMResponse:
        prompt = request.prompt_text()
        system = request.system or ""
        full = f"{system}\n{prompt}".strip()

        # Seed from the prompt so the response is stable across runs -- that is
        # what lets prompt A/B comparisons in tests be meaningful.
        seed = int(hashlib.sha256(full.encode("utf-8")).hexdigest()[:12], 16)
        rng = random.Random(seed)

        # Deterministic pseudo-latency so latency metrics have real numbers.
        latency_ms = 40 + (seed % 120)
        time.sleep(min(latency_ms, 60) / 1000.0)

        text = self._respond(full, rng)
        usage = self._usage_from_text(full, text)

        return LLMResponse(
            text=text,
            model=request.model or self.default_model(),
            provider=self.name,
            usage=usage,
            latency_ms=float(latency_ms),
            finish_reason="stop",
            raw={"mock": True, "seed": seed},
        )

    # -- response shaping ---------------------------------------------------- #
    def _respond(self, prompt: str, rng: random.Random) -> str:
        lowered = prompt.lower()

        # Refuse the things a real safety-aligned model would refuse. This lets
        # the safety suite exercise both the pass and the refusal path offline.
        if self._looks_unsafe(lowered):
            return (
                "I can't help with that request. If you have a legitimate "
                "account or billing question I am happy to assist instead."
            )

        if "json" in lowered or "fields:" in lowered:
            return self._json_response(prompt, rng)
        if "classify" in lowered or "category" in lowered:
            return rng.choice(_CATEGORIES)
        if "summar" in lowered:
            return self._summary(prompt, rng)
        if "extract" in lowered:
            return self._json_response(prompt, rng)
        if any(q in lowered for q in ("what", "how", "why", "when", "?")):
            return self._answer(prompt, rng)
        return (
            "Acknowledged. This deterministic mock provider returns structured "
            "placeholder output so the LLMOps pipeline can be exercised offline."
        )

    #: Instruction lines to drop, so the "summary" echoes the supplied content
    #: rather than the template's own directives.
    _INSTRUCTION_PREFIXES = (
        "do not",
        "only use",
        "summarise",
        "summarize",
        "extract",
        "state ",
        "write ",
        "- ",
    )

    def _summary(self, prompt: str, rng: random.Random) -> str:
        # Prefer the supplied content over the instruction text: templates put
        # the variable last, after a "Ticket:"-style marker, so split there when
        # one is present and fall back to the whole prompt otherwise.
        body = prompt
        for marker in ("Ticket:", "ticket:", "Context:", "Input:"):
            if marker in prompt:
                body = prompt.rsplit(marker, 1)[1]
                break

        sentences = [
            s.strip()
            for s in re.split(r"[.\n]", body)
            if len(s.strip()) > 25
            and not s.strip().lower().startswith(self._INSTRUCTION_PREFIXES)
        ]
        # Echo real content so faithfulness scoring has something genuine to
        # grade rather than boilerplate.
        picked = sentences[:3] if sentences else []
        if not picked:
            return "The customer reported an issue and is awaiting a resolution."
        body = ". ".join(s[:160] for s in picked)
        return (
            f"Summary: {body}. "
            f"Sentiment appears {rng.choice(_SENTIMENTS)}; "
            f"priority {rng.choice(_PRIORITIES)}."
        )

    def _json_response(self, prompt: str, rng: random.Random) -> str:
        import json

        return json.dumps(
            {
                "category": rng.choice(_CATEGORIES),
                "sentiment": rng.choice(_SENTIMENTS),
                "priority": rng.choice(_PRIORITIES),
                "summary": self._summary(prompt, rng)[:220],
                "requires_human": rng.random() < 0.3,
            },
            indent=2,
        )

    def _answer(self, prompt: str, rng: random.Random) -> str:
        keywords = [
            w
            for w in re.findall(r"[a-zA-Z]{5,}", prompt)[:6]
            if w.lower() not in {"please", "should", "would", "there", "which", "about"}
        ]
        topic = ", ".join(keywords[:3]) if keywords else "the request"
        return (
            f"Based on the information provided about {topic}, the recommended "
            f"next step is to review the account history and confirm the details "
            f"with the customer before proceeding. Confidence: "
            f"{rng.choice(('low', 'moderate', 'high'))}."
        )

    @staticmethod
    def _looks_unsafe(lowered: str) -> bool:
        markers = (
            "ignore previous instructions",
            "ignore all previous",
            "disregard your instructions",
            "reveal your system prompt",
            "print your system prompt",
            "developer mode",
            "do anything now",
            "how do i make a bomb",
            "synthesize a nerve agent",
            "steal credit card",
            "without getting caught",
        )
        return any(m in lowered for m in markers)
