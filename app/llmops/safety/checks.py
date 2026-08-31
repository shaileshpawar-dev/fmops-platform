"""Heuristic LLM safety screening.

**What this is:** a fast, dependency-free, pattern- and heuristic-based screen
that catches obvious prompt-injection attempts, obvious jailbreak framings,
clearly unsafe requests, PII/secret leakage in outputs, and a small set of
hallucination *indicators*. It runs inline on every request and costs
microseconds.

**What this is NOT, and must never be presented as:**

* It is **not** a content-safety classifier. It has no model behind it. A
  determined adversary bypasses regex screening trivially -- by paraphrasing,
  by encoding, by translating, by splitting an attack across turns.
* It **cannot detect hallucination**. Nothing in this file knows whether a claim
  is true. The "hallucination indicators" check flags *stylistic* correlates
  (specific-sounding numbers, dates and citations that do not appear anywhere in
  the provided context). A grounded true statement can trigger it; a fluent
  false one can pass it. It is a review-prioritisation signal, not a verdict.
* Its false-negative rate is unmeasured, because there is no labelled adversarial
  corpus in this project.

**What a production system needs in addition:** a dedicated moderation model
(Amazon Bedrock Guardrails, Azure Content Safety, Llama Guard, or equivalent),
output schema validation, retrieval grounding with citation checking, rate
limiting per principal, and human review of flagged traffic. This module is
designed so those slot in as additional :class:`SafetyCheck` implementations
without touching callers.

See ``docs/llmops.md`` for the full statement of limitations.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod

from app.core.config import Settings, get_settings
from app.core.logging import get_logger
from app.schemas.llm import SafetyFinding, SafetyVerdict

logger = get_logger(__name__)


class SafetyCheck(ABC):
    """One screening rule applied to an input and/or an output."""

    name: str = "abstract"
    category: str = "generic"
    severity: str = "medium"
    #: Which direction this check applies to.
    checks_input: bool = True
    checks_output: bool = False

    @abstractmethod
    def run(self, text: str, context: str = "") -> SafetyFinding: ...

    def _finding(self, triggered: bool, evidence: str = "", detail: str = "") -> SafetyFinding:
        return SafetyFinding(
            check=self.name,
            category=self.category,
            severity=self.severity,  # type: ignore[arg-type]
            triggered=triggered,
            evidence=evidence[:200],
            detail=detail,
        )


class PromptInjectionCheck(SafetyCheck):
    """Instruction-override attempts in user-supplied text."""

    name = "prompt_injection"
    category = "injection"
    severity = "high"

    PATTERNS = (
        r"ignore\s+(all\s+)?(the\s+)?(previous|prior|above|earlier)\s+(instructions?|prompts?|rules?)",
        r"disregard\s+(all\s+)?(your|the|previous|prior)\s+(instructions?|rules?|guidelines?)",
        r"forget\s+(everything|all|your)\s+(you|instructions?|rules?)",
        r"(reveal|show|print|output|repeat)\s+(me\s+)?(your|the)\s+(system\s+)?(prompt|instructions?)",
        r"what\s+(are|were)\s+your\s+(original\s+)?instructions",
        r"you\s+are\s+no\s+longer\s+(a|an|bound)",
        r"new\s+(instructions?|system\s+prompt)\s*:",
        r"</?(system|instruction)s?>",
        r"###\s*(system|instruction)",
    )

    def run(self, text: str, context: str = "") -> SafetyFinding:
        lowered = text.lower()
        for pattern in self.PATTERNS:
            match = re.search(pattern, lowered)
            if match:
                return self._finding(
                    True,
                    evidence=match.group(0),
                    detail=(
                        "the input contains an instruction-override pattern; treat "
                        "the content as untrusted data, not as instructions"
                    ),
                )
        return self._finding(False, detail="no instruction-override pattern found")


class JailbreakCheck(SafetyCheck):
    """Well-known jailbreak framings."""

    name = "jailbreak"
    category = "jailbreak"
    severity = "high"

    PATTERNS = (
        r"\bdan\s+mode\b",
        r"\bdo\s+anything\s+now\b",
        r"developer\s+mode\s+(enabled|on)",
        r"pretend\s+(you\s+are|to\s+be)\s+(an?\s+)?(unrestricted|unfiltered|uncensored)",
        r"(without|no)\s+(any\s+)?(restrictions?|filters?|guidelines?|ethics)",
        r"hypothetically,?\s+if\s+you\s+(had\s+no|were\s+not)",
        r"for\s+(educational|research)\s+purposes\s+only,?\s+(explain|describe)\s+how\s+to\s+(make|build|synthesize)",
        r"you\s+must\s+comply\s+(with\s+)?(all|any)",
        r"simulate\s+(an?\s+)?(ai|assistant)\s+(without|with\s+no)",
    )

    def run(self, text: str, context: str = "") -> SafetyFinding:
        lowered = text.lower()
        for pattern in self.PATTERNS:
            match = re.search(pattern, lowered)
            if match:
                return self._finding(
                    True,
                    evidence=match.group(0),
                    detail="the input uses a recognised jailbreak framing",
                )
        return self._finding(False, detail="no known jailbreak framing found")


class UnsafeRequestCheck(SafetyCheck):
    """Requests for clearly harmful assistance."""

    name = "unsafe_request"
    category = "harmful_content"
    severity = "high"

    PATTERNS = (
        r"how\s+to\s+(make|build|construct|synthesi[sz]e)\s+(a\s+)?(bomb|explosive|nerve\s+agent|bioweapon)",
        r"(synthesi[sz]e|manufacture)\s+(methamphetamine|fentanyl|sarin|ricin)",
        r"how\s+to\s+(hack|break)\s+into\s+(someone|somebody|a\s+person)",
        r"steal\s+(credit\s+card|bank\s+account|identity|someone'?s\s+(money|identity))",
        r"(kill|poison|harm)\s+(someone|somebody|a\s+person|my\s+\w+)\s+(and|without)\s+(get|avoid)",
        r"without\s+(getting\s+caught|being\s+detected|leaving\s+(a\s+)?trace)",
        r"(launder|laundering)\s+money",
    )

    def run(self, text: str, context: str = "") -> SafetyFinding:
        lowered = text.lower()
        for pattern in self.PATTERNS:
            match = re.search(pattern, lowered)
            if match:
                return self._finding(
                    True,
                    evidence=match.group(0),
                    detail="the input requests assistance with clearly harmful activity",
                )
        return self._finding(False, detail="no clearly harmful request detected")


class SensitiveDataLeakCheck(SafetyCheck):
    """Credentials and personal identifiers appearing in text.

    Applied to model *output* primarily: an assistant echoing a card number or
    an API key back to a user is a leak regardless of how it got there.
    """

    name = "sensitive_data_leak"
    category = "data_leakage"
    severity = "high"
    checks_output = True

    PATTERNS: tuple[tuple[str, str], ...] = (
        (r"\b(?:\d[ -]*?){13,19}\b", "possible payment card number"),
        (r"\b\d{3}-\d{2}-\d{4}\b", "possible US social security number"),
        (r"\b(sk|pk)-[A-Za-z0-9]{20,}\b", "possible API key"),
        (r"\bAKIA[0-9A-Z]{16}\b", "AWS access key id"),
        (r"\bghp_[A-Za-z0-9]{30,}\b", "GitHub personal access token"),
        (
            r"(?i)\b(api[_-]?key|secret|password|passwd|token)\s*[:=]\s*\S{8,}",
            "credential assignment",
        ),
        (r"-----BEGIN\s+(RSA\s+|EC\s+|OPENSSH\s+)?PRIVATE KEY-----", "private key material"),
    )

    def run(self, text: str, context: str = "") -> SafetyFinding:
        for pattern, description in self.PATTERNS:
            match = re.search(pattern, text)
            if match:
                candidate = match.group(0)
                if description == "possible payment card number" and not _luhn(candidate):
                    continue
                # Never echo the matched secret itself into logs or findings.
                return self._finding(
                    True,
                    evidence=f"[redacted {len(candidate)} chars]",
                    detail=f"{description} detected in the text",
                )
        return self._finding(False, detail="no sensitive-data pattern found")


class HallucinationIndicatorCheck(SafetyCheck):
    """Stylistic correlates of unsupported claims. NOT a factuality check.

    Flags specific-sounding numbers, dates, percentages and citation-like
    constructs in the output that do not appear in the supplied context. When no
    context is supplied there is nothing to ground against, and the check
    abstains rather than guessing.
    """

    name = "hallucination_indicators"
    category = "grounding"
    severity = "low"
    checks_input = False
    checks_output = True

    SPECIFICS = (
        r"\b\d{1,3}(?:\.\d+)?%\b",
        r"\b(19|20)\d{2}\b",
        r"\$\s?\d[\d,]*(?:\.\d{2})?\b",
        r"\b(?:according to|as stated in|per)\s+[A-Z][\w\s]{3,30}\b",
    )
    OVERCONFIDENCE = (
        r"\b(definitely|certainly|without\s+a\s+doubt|guaranteed|always|never)\b",
    )

    def run(self, text: str, context: str = "") -> SafetyFinding:
        if not context.strip():
            return self._finding(
                False,
                detail=(
                    "abstained: no grounding context was supplied, so unsupported "
                    "claims cannot be identified"
                ),
            )

        unsupported: list[str] = []
        lowered_context = context.lower()
        for pattern in self.SPECIFICS:
            for match in re.findall(pattern, text):
                token = match if isinstance(match, str) else match[0]
                if token and token.lower() not in lowered_context:
                    unsupported.append(token)

        overconfident = any(re.search(p, text.lower()) for p in self.OVERCONFIDENCE)

        if unsupported:
            return self._finding(
                True,
                evidence=", ".join(sorted(set(unsupported))[:5]),
                detail=(
                    f"{len(set(unsupported))} specific value(s) in the output do not "
                    "appear in the provided context. This is a review signal, not "
                    "proof of a false statement."
                ),
            )
        if overconfident:
            return self._finding(
                True,
                evidence="absolute language",
                detail=(
                    "the output uses absolute language; worth reviewing for "
                    "overstated certainty"
                ),
            )
        return self._finding(False, detail="no unsupported specifics detected")


def _luhn(candidate: str) -> bool:
    """Luhn checksum, to cut the false-positive rate on long digit strings."""
    digits = [int(c) for c in re.sub(r"\D", "", candidate)]
    if not 13 <= len(digits) <= 19:
        return False
    total = 0
    parity = len(digits) % 2
    for index, digit in enumerate(digits):
        if index % 2 == parity:
            digit *= 2
            if digit > 9:
                digit -= 9
        total += digit
    return total % 10 == 0


DEFAULT_CHECKS: tuple[type[SafetyCheck], ...] = (
    PromptInjectionCheck,
    JailbreakCheck,
    UnsafeRequestCheck,
    SensitiveDataLeakCheck,
    HallucinationIndicatorCheck,
)

_SEVERITY_WEIGHT = {"info": 0.0, "low": 0.15, "medium": 0.4, "high": 1.0}


class SafetyScreen:
    """Runs the configured checks over an input and/or an output."""

    def __init__(
        self,
        checks: list[SafetyCheck] | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.checks = checks if checks is not None else [c() for c in DEFAULT_CHECKS]

    def screen_input(self, text: str) -> SafetyVerdict:
        return self._run([c for c in self.checks if c.checks_input], text, "")

    def screen_output(self, text: str, context: str = "") -> SafetyVerdict:
        return self._run([c for c in self.checks if c.checks_output], text, context)

    def screen(
        self, input_text: str, output_text: str = "", context: str = ""
    ) -> SafetyVerdict:
        findings: list[SafetyFinding] = []
        names: list[str] = []
        for check in self.checks:
            if check.checks_input and input_text:
                findings.append(check.run(input_text))
                names.append(f"{check.name}:input")
            if check.checks_output and output_text:
                findings.append(check.run(output_text, context))
                names.append(f"{check.name}:output")
        return self._verdict(findings, names)

    def _run(self, checks: list[SafetyCheck], text: str, context: str) -> SafetyVerdict:
        findings = [check.run(text, context) for check in checks]
        return self._verdict(findings, [c.name for c in checks])

    def _verdict(self, findings: list[SafetyFinding], names: list[str]) -> SafetyVerdict:
        triggered = [f for f in findings if f.triggered]
        risk = min(1.0, sum(_SEVERITY_WEIGHT.get(f.severity, 0.4) for f in triggered))
        blocking = [f for f in triggered if f.severity == "high"]
        blocked = bool(blocking) and self.settings.llm.safety_block_on_violation

        verdict = SafetyVerdict(
            passed=not triggered,
            blocked=blocked,
            risk_score=round(risk, 4),
            findings=findings,
            checked=names,
            detail=(
                "heuristic screen only -- pattern based, no model; "
                "see docs/llmops.md for limitations"
            ),
        )
        if triggered:
            from app.monitoring.metrics import get_metrics

            metrics = get_metrics()
            for finding in triggered:
                metrics.llm_safety_findings_total.labels(
                    check=finding.check, severity=finding.severity
                ).inc()
            logger.warning(
                "llm.safety_findings",
                extra={
                    "triggered": [f.check for f in triggered],
                    "risk_score": verdict.risk_score,
                    "blocked": blocked,
                },
            )
        return verdict


_SCREEN: SafetyScreen | None = None


def get_safety_screen() -> SafetyScreen:
    global _SCREEN
    if _SCREEN is None:
        _SCREEN = SafetyScreen()
    return _SCREEN
