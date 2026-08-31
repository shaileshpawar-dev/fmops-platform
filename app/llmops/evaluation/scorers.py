"""LLM output scorers.

How LLM evaluation differs from ML evaluation, and why this file looks the way
it does: a classifier has one correct answer per row, so accuracy is exact. A
generated paragraph has many acceptable forms, so every score here is a
*proxy* with known blind spots. Each scorer therefore documents what it actually
measures.

Three families are implemented:

**Reference-free structural scorers** (``format_validity``, ``latency``,
``token_efficiency``) -- objective and reliable.

**Reference-based lexical scorers** (``correctness``, ``relevance``,
``keyword_coverage``) -- token-overlap based. They reward saying the same words,
not being right. A correct paraphrase scores poorly; a fluent restatement of the
question scores well. Cheap, deterministic, offline.

**Grounding scorer** (``faithfulness``) -- what fraction of the output's content
words appear in the supplied context. Measures *support*, not truth.

An LLM-as-judge scorer is deliberately **not** enabled by default: it costs
money per evaluation, is non-deterministic, and cannot be run offline, which
would break this project's "works with no API key" property. The interface below
accommodates one -- see :class:`Scorer` -- and ``docs/llmops.md`` explains when
to add it.
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod

from app.core.logging import get_logger
from app.schemas.llm import LLMEvalCase

logger = get_logger(__name__)

_STOPWORDS = frozenset(
    [
        "a",
        "an",
        "the",
        "and",
        "or",
        "but",
        "if",
        "then",
        "than",
        "that",
        "this",
        "these",
        "those",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "to",
        "of",
        "in",
        "on",
        "at",
        "by",
        "for",
        "with",
        "about",
        "into",
        "over",
        "after",
        "under",
        "again",
        "further",
        "once",
        "here",
        "there",
        "when",
        "where",
        "why",
        "how",
        "all",
        "any",
        "both",
        "each",
        "few",
        "more",
        "most",
        "other",
        "some",
        "such",
        "no",
        "nor",
        "not",
        "only",
        "own",
        "same",
        "so",
        "too",
        "very",
        "can",
        "will",
        "just",
        "should",
        "now",
        "it",
        "its",
        "as",
        "from",
    ]
)


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", (text or "").lower())


def content_words(text: str) -> set[str]:
    return {w for w in tokenize(text) if w not in _STOPWORDS and len(w) > 2}


class Scorer(ABC):
    """Scores one output. Returns a value in [0, 1]; higher is better."""

    name: str = "abstract"
    #: True when the scorer needs a reference answer to work.
    needs_reference: bool = False
    #: True when the scorer needs grounding context.
    needs_context: bool = False

    @abstractmethod
    def score(self, output: str, case: LLMEvalCase) -> float: ...


class CorrectnessScorer(Scorer):
    """Token-overlap F1 against the reference answer.

    Measures lexical agreement, not semantic correctness. Reported alongside
    keyword coverage precisely because neither alone is trustworthy.
    """

    name = "correctness"
    needs_reference = True

    def score(self, output: str, case: LLMEvalCase) -> float:
        if not case.reference:
            return 0.0
        predicted = content_words(output)
        expected = content_words(case.reference)
        if not expected:
            return 0.0
        if not predicted:
            return 0.0
        overlap = len(predicted & expected)
        if overlap == 0:
            return 0.0
        precision = overlap / len(predicted)
        recall = overlap / len(expected)
        return round(2 * precision * recall / (precision + recall), 6)


class RelevanceScorer(Scorer):
    """How much of the output relates to the input.

    Penalises off-topic drift and empty responses. Does not detect an answer
    that is on-topic but wrong.
    """

    name = "relevance"

    def score(self, output: str, case: LLMEvalCase) -> float:
        if not output.strip():
            return 0.0
        source = " ".join(str(v) for v in case.input.values())
        source_words = content_words(source)
        output_words = content_words(output)
        if not source_words or not output_words:
            return 0.0
        overlap = len(source_words & output_words)
        # Recall against the input, capped: an answer need not echo everything.
        return round(min(1.0, overlap / max(1, min(len(source_words), 12))), 6)


class FaithfulnessScorer(Scorer):
    """Fraction of the output's content words supported by the context.

    This measures *groundedness*, not truth: an output that copies the context
    verbatim scores 1.0 even if the context itself is wrong. It is the strongest
    signal available without a judge model or a knowledge base.
    """

    name = "faithfulness"
    needs_context = True

    def score(self, output: str, case: LLMEvalCase) -> float:
        context = case.context or " ".join(str(v) for v in case.input.values())
        if not context.strip():
            return 0.0
        supported = content_words(context)
        produced = content_words(output)
        if not produced:
            return 0.0
        # Numbers are the classic hallucination vector; weight them heavily.
        numbers = set(re.findall(r"\d[\d,.]*", output))
        context_numbers = set(re.findall(r"\d[\d,.]*", context))
        unsupported_numbers = numbers - context_numbers

        word_support = len(produced & supported) / len(produced)
        penalty = 0.15 * len(unsupported_numbers)
        return round(max(0.0, min(1.0, word_support - penalty)), 6)


class KeywordCoverageScorer(Scorer):
    """Fraction of required keywords present, minus forbidden ones.

    The most reliable reference-based scorer here, because the test author
    states exactly what must and must not appear.
    """

    name = "keyword_coverage"

    def score(self, output: str, case: LLMEvalCase) -> float:
        lowered = (output or "").lower()
        if not case.expected_keywords and not case.forbidden_keywords:
            return 1.0
        hits = sum(1 for k in case.expected_keywords if k.lower() in lowered)
        expected_score = hits / len(case.expected_keywords) if case.expected_keywords else 1.0
        violations = sum(1 for k in case.forbidden_keywords if k.lower() in lowered)
        penalty = (
            violations / max(len(case.forbidden_keywords), 1)
            if case.forbidden_keywords
            else 0.0
        )
        return round(max(0.0, expected_score - penalty), 6)


class FormatValidityScorer(Scorer):
    """Does the output honour the requested output contract?

    Objective and reliable. When a case is tagged ``json`` the output must parse
    as JSON; when it is tagged ``max_sentences:N`` the length bound is checked.
    """

    name = "format_validity"

    def score(self, output: str, case: LLMEvalCase) -> float:
        checks: list[float] = []
        text = (output or "").strip()

        if "json" in case.tags:
            cleaned = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
            try:
                json.loads(cleaned)
                checks.append(1.0)
            except (ValueError, TypeError):
                checks.append(0.0)

        for tag in case.tags:
            if tag.startswith("max_sentences:"):
                try:
                    limit = int(tag.split(":", 1)[1])
                except ValueError:
                    continue
                sentences = [s for s in re.split(r"[.!?]+", text) if s.strip()]
                checks.append(1.0 if len(sentences) <= limit else 0.0)

        if not text:
            return 0.0
        return round(sum(checks) / len(checks), 6) if checks else 1.0


DEFAULT_SCORERS: tuple[type[Scorer], ...] = (
    CorrectnessScorer,
    RelevanceScorer,
    FaithfulnessScorer,
    KeywordCoverageScorer,
    FormatValidityScorer,
)

# Weights for the composite "overall" score. Keyword coverage and format
# validity are weighted highest because they are the least noisy.
DEFAULT_WEIGHTS: dict[str, float] = {
    "keyword_coverage": 0.30,
    "format_validity": 0.25,
    "faithfulness": 0.20,
    "correctness": 0.15,
    "relevance": 0.10,
}


def build_scorers(names: list[str] | None = None) -> list[Scorer]:
    available = {cls.name: cls for cls in DEFAULT_SCORERS}
    if names is None:
        return [cls() for cls in DEFAULT_SCORERS]
    out: list[Scorer] = []
    for name in names:
        cls = available.get(name)
        if cls is None:
            logger.warning("llm_eval.unknown_scorer", extra={"scorer": name})
            continue
        out.append(cls())
    return out


def composite_score(
    scores: dict[str, float], weights: dict[str, float] | None = None
) -> float:
    """Weighted mean over the scorers that actually ran."""
    weights = weights or DEFAULT_WEIGHTS
    applicable = {k: v for k, v in scores.items() if k in weights}
    if not applicable:
        return 0.0
    total_weight = sum(weights[k] for k in applicable)
    if total_weight <= 0:
        return 0.0
    return round(sum(applicable[k] * weights[k] for k in applicable) / total_weight, 6)
