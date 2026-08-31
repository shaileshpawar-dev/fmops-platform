"""LLM evaluation runner.

Runs an evaluation dataset against a (provider, model, prompt version) triple,
scores every case, aggregates, persists, and supports A/B comparison along two
axes:

* **model A vs model B** at a fixed prompt version, and
* **prompt version A vs B** at a fixed model.

Both are the same operation with a different variable held constant, which is
why one comparison function serves both.

Every stored result records the provider. That matters: results produced against
the mock provider measure the *harness*, not model quality, and must never be
compared against results from a real provider. :func:`compare` refuses such a
comparison rather than producing a meaningless delta.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import yaml

from app.core.config import Settings, get_settings
from app.core.db import Database, dumps, get_database, loads
from app.core.exceptions import FMOpsError
from app.core.logging import get_logger
from app.core.utils import write_json
from app.llmops.client import LLMClient, get_llm_client
from app.llmops.evaluation.scorers import Scorer, build_scorers, composite_score
from app.monitoring.metrics import get_metrics
from app.schemas.llm import (
    CaseScore,
    LLMComparison,
    LLMEvalCase,
    LLMEvalDataset,
    LLMEvaluationResult,
    LLMGenerateRequest,
)

logger = get_logger(__name__)


def load_dataset(path: Path | str) -> LLMEvalDataset:
    """Load an evaluation dataset from YAML."""
    source = Path(path)
    if not source.is_file():
        raise FMOpsError(f"evaluation dataset not found: {source}", path=str(source))
    payload = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    cases = [LLMEvalCase(**case) for case in payload.get("cases", [])]
    if not cases:
        raise FMOpsError(f"evaluation dataset {source} contains no cases")
    return LLMEvalDataset(
        name=payload.get("name", source.stem),
        description=payload.get("description", ""),
        prompt_name=payload.get("prompt_name"),
        cases=cases,
    )


def discover_datasets(settings: Settings | None = None) -> dict[str, Path]:
    settings = settings or get_settings()
    directory = settings.resolved_llm_eval_dir()
    if not directory.is_dir():
        return {}
    return {p.stem: p for p in sorted(directory.glob("*.yaml"))}


class EvaluationRunner:
    """Executes an evaluation suite and persists the result."""

    def __init__(
        self,
        client: LLMClient | None = None,
        scorers: list[Scorer] | None = None,
        settings: Settings | None = None,
        db: Database | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self._client = client
        self.scorers = scorers if scorers is not None else build_scorers()
        self._db = db

    @property
    def client(self) -> LLMClient:
        if self._client is None:
            self._client = get_llm_client()
        return self._client

    @property
    def db(self) -> Database:
        if self._db is None:
            self._db = get_database()
        return self._db

    def run(
        self,
        dataset: LLMEvalDataset | str | Path,
        prompt_name: str | None = None,
        prompt_version: str | None = None,
        model: str | None = None,
        provider: str | None = None,
        suite: str = "default",
        persist: bool = True,
    ) -> LLMEvaluationResult:
        if not isinstance(dataset, LLMEvalDataset):
            candidates = discover_datasets(self.settings)
            path = (
                candidates.get(str(dataset)) if str(dataset) in candidates else Path(dataset)
            )
            dataset = load_dataset(path)

        resolved_prompt = (
            prompt_name or dataset.prompt_name or self.settings.llm.default_prompt
        )
        if prompt_version is None:
            from app.llmops.prompts.registry import get_prompt_registry

            prompt_version = get_prompt_registry().latest_version(resolved_prompt)

        started = time.perf_counter()
        per_case: list[CaseScore] = []
        resolved_provider = provider or self.settings.llm.provider
        resolved_model = model or self.settings.llm.model

        logger.info(
            "llm_eval.started",
            extra={
                "suite": suite,
                "dataset": dataset.name,
                "cases": len(dataset.cases),
                "prompt": f"{resolved_prompt}@{prompt_version}",
                "provider": resolved_provider,
                "model": resolved_model,
            },
        )

        for case in dataset.cases:
            per_case.append(
                self._run_case(case, resolved_prompt, prompt_version, resolved_model, provider)
            )

        succeeded = [c for c in per_case if c.error is None]
        aggregate = self._aggregate(per_case, succeeded)

        # The provider/model actually used, as reported by the traces, which may
        # differ from what was requested if a fallback happened.
        if succeeded:
            resolved_provider = self._observed_provider(resolved_provider)

        result = LLMEvaluationResult(
            suite=suite,
            dataset=dataset.name,
            provider=resolved_provider,
            model=resolved_model,
            prompt_name=resolved_prompt,
            prompt_version=prompt_version or "unknown",
            n_cases=len(per_case),
            aggregate=aggregate,
            per_case=per_case,
            total_tokens=sum(c.usage.total_tokens for c in per_case),
            total_cost_usd=round(sum(c.cost_usd for c in per_case), 8),
        )

        self._export_metrics(result)
        if persist:
            self._persist(result)

        logger.info(
            "llm_eval.completed",
            extra={
                "suite": suite,
                "dataset": dataset.name,
                "overall": result.overall_score,
                "cases": len(per_case),
                "failed": len(per_case) - len(succeeded),
                "tokens": result.total_tokens,
                "cost_usd": result.total_cost_usd,
                "duration_s": round(time.perf_counter() - started, 2),
            },
        )
        return result

    def _run_case(
        self,
        case: LLMEvalCase,
        prompt_name: str,
        prompt_version: str | None,
        model: str | None,
        provider: str | None,
    ) -> CaseScore:
        try:
            response = self.client.generate(
                LLMGenerateRequest(
                    prompt_name=prompt_name,
                    prompt_version=prompt_version,
                    variables=case.input,
                    model=model,
                    provider=provider,
                )
            )
        except Exception as exc:
            # One failing case must not abort the suite; it is scored zero and
            # the error is recorded so the aggregate is interpretable.
            logger.warning(
                "llm_eval.case_failed",
                extra={"case_id": case.id, "error": str(exc)},
            )
            return CaseScore(case_id=case.id, passed=False, error=str(exc)[:400])

        scores: dict[str, float] = {}
        for scorer in self.scorers:
            if scorer.needs_reference and not case.reference:
                continue
            if scorer.needs_context and not (case.context or case.input):
                continue
            try:
                scores[scorer.name] = scorer.score(response.text, case)
            except Exception as exc:
                logger.warning(
                    "llm_eval.scorer_failed",
                    extra={"scorer": scorer.name, "case_id": case.id, "error": str(exc)},
                )

        overall = composite_score(scores)
        scores["overall"] = overall
        return CaseScore(
            case_id=case.id,
            output=response.text[:2000],
            scores=scores,
            passed=overall >= 0.5,
            latency_ms=response.latency_ms,
            usage=response.usage,
            cost_usd=response.estimated_cost_usd,
            safety=response.safety,
        )

    def _aggregate(
        self, per_case: list[CaseScore], succeeded: list[CaseScore]
    ) -> dict[str, float]:
        if not succeeded:
            return {"overall": 0.0, "success_rate": 0.0, "n_failed": float(len(per_case))}

        metric_names = {name for case in succeeded for name in case.scores}
        aggregate: dict[str, float] = {}
        for name in metric_names:
            values = [c.scores[name] for c in succeeded if name in c.scores]
            aggregate[name] = round(sum(values) / len(values), 6)

        latencies = [c.latency_ms for c in succeeded]
        aggregate.update(
            success_rate=round(len(succeeded) / len(per_case), 6),
            pass_rate=round(sum(1 for c in succeeded if c.passed) / len(succeeded), 6),
            n_failed=float(len(per_case) - len(succeeded)),
            avg_latency_ms=round(sum(latencies) / len(latencies), 3),
            max_latency_ms=round(max(latencies), 3),
            avg_tokens=round(sum(c.usage.total_tokens for c in succeeded) / len(succeeded), 2),
            avg_cost_usd=round(sum(c.cost_usd for c in succeeded) / len(succeeded), 8),
            safety_flagged=float(
                sum(1 for c in succeeded if c.safety and not c.safety.passed)
            ),
        )
        return aggregate

    def _observed_provider(self, requested: str) -> str:
        row = self.db.query_one(
            "SELECT provider FROM llm_traces ORDER BY created_at DESC LIMIT 1"
        )
        return row["provider"] if row else requested

    def _export_metrics(self, result: LLMEvaluationResult) -> None:
        metrics = get_metrics()
        for name, value in result.aggregate.items():
            metrics.llm_eval_score.labels(
                suite=result.suite,
                model=result.model,
                prompt_version=result.prompt_version,
                metric=name,
            ).set(float(value))

    def _persist(self, result: LLMEvaluationResult) -> None:
        try:
            self.db.execute(
                "INSERT INTO llm_evaluations (id, suite, dataset, provider, model, "
                "prompt_name, prompt_version, n_cases, aggregate, per_case, "
                "total_tokens, total_cost_usd, created_at) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    result.id,
                    result.suite,
                    result.dataset,
                    result.provider,
                    result.model,
                    result.prompt_name,
                    result.prompt_version,
                    result.n_cases,
                    dumps(result.aggregate),
                    dumps([c.model_dump(mode="json") for c in result.per_case]),
                    result.total_tokens,
                    result.total_cost_usd,
                    result.created_at,
                ),
            )
        except Exception as exc:
            logger.error("llm_eval.persist_failed", extra={"error": str(exc)})

        try:
            path = self.settings.paths.reports_dir / "llm_eval" / f"{result.id}.json"
            write_json(path, result.model_dump(mode="json"))
        except Exception as exc:
            logger.warning("llm_eval.report_write_failed", extra={"error": str(exc)})


# --------------------------------------------------------------------------- #
# Comparison
# --------------------------------------------------------------------------- #
def compare(
    a: LLMEvaluationResult,
    b: LLMEvaluationResult,
    metric: str = "overall",
) -> LLMComparison:
    """A/B two evaluation runs along whichever axis differs."""
    if a.provider != b.provider:
        raise FMOpsError(
            "refusing to compare evaluation runs from different providers "
            f"({a.provider} vs {b.provider}): scores from the deterministic mock "
            "provider measure the evaluation harness, not model quality, and are "
            "not comparable with real model output",
            provider_a=a.provider,
            provider_b=b.provider,
        )

    if a.model != b.model:
        dimension = "model"
        variant_a, variant_b = a.model, b.model
    else:
        dimension = "prompt"
        variant_a = f"{a.prompt_name}@{a.prompt_version}"
        variant_b = f"{b.prompt_name}@{b.prompt_version}"

    score_a = float(a.aggregate.get(metric, 0.0))
    score_b = float(b.aggregate.get(metric, 0.0))
    delta = score_b - score_a

    per_metric: dict[str, dict[str, float]] = {}
    for name in sorted(set(a.aggregate) | set(b.aggregate)):
        value_a = float(a.aggregate.get(name, 0.0))
        value_b = float(b.aggregate.get(name, 0.0))
        per_metric[name] = {
            "a": value_a,
            "b": value_b,
            "delta": round(value_b - value_a, 6),
        }

    if abs(delta) < 1e-9:
        winner = "tie"
    else:
        winner = variant_b if delta > 0 else variant_a

    comparison = LLMComparison(
        dimension=dimension,
        variant_a=variant_a,
        variant_b=variant_b,
        metric=metric,
        score_a=round(score_a, 6),
        score_b=round(score_b, 6),
        delta=round(delta, 6),
        winner=winner,
        cost_a_usd=a.total_cost_usd,
        cost_b_usd=b.total_cost_usd,
        latency_a_ms=float(a.aggregate.get("avg_latency_ms", 0.0)),
        latency_b_ms=float(b.aggregate.get("avg_latency_ms", 0.0)),
        per_metric=per_metric,
    )
    logger.info(
        "llm_eval.comparison",
        extra={
            "dimension": dimension,
            "variant_a": variant_a,
            "variant_b": variant_b,
            "metric": metric,
            "delta": comparison.delta,
            "winner": winner,
        },
    )
    return comparison


def recent_evaluations(limit: int = 20, db: Database | None = None) -> list[dict[str, Any]]:
    database = db or get_database()
    rows = database.query(
        "SELECT id, suite, dataset, provider, model, prompt_name, prompt_version, "
        "n_cases, aggregate, total_tokens, total_cost_usd, created_at "
        "FROM llm_evaluations ORDER BY created_at DESC LIMIT ?",
        (limit,),
    )
    out = []
    for row in rows:
        item = dict(row)
        item["aggregate"] = loads(item.get("aggregate"), {})
        out.append(item)
    return out


def get_evaluation(
    evaluation_id: str, db: Database | None = None
) -> LLMEvaluationResult | None:
    database = db or get_database()
    row = database.query_one("SELECT * FROM llm_evaluations WHERE id = ?", (evaluation_id,))
    if row is None:
        return None
    data = dict(row)
    return LLMEvaluationResult(
        id=data["id"],
        suite=data["suite"],
        dataset=data["dataset"],
        provider=data["provider"],
        model=data["model"],
        prompt_name=data["prompt_name"],
        prompt_version=data["prompt_version"],
        n_cases=data["n_cases"],
        aggregate=loads(data["aggregate"], {}),
        per_case=[CaseScore(**c) for c in loads(data["per_case"], [])],
        total_tokens=data["total_tokens"],
        total_cost_usd=data["total_cost_usd"],
        created_at=data["created_at"],
    )
