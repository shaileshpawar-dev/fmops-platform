"""LLM trace store and token accounting.

Every invocation -- successful or failed -- becomes one persisted
:class:`~app.schemas.llm.LLMTrace` carrying the prompt identity, model,
parameters, token counts, latency, estimated cost, safety verdict and git
commit. That record is what makes the LLM layer auditable in the same way the
ML layer is: given an output, you can recover exactly which prompt version and
model produced it.

Failed calls are recorded too. A provider outage that produces zero traces would
make the cost dashboard look healthy while the feature is completely broken.
"""

from __future__ import annotations

from typing import Any

from app.core.config import Settings, get_settings
from app.core.db import Database, dumps, get_database, loads
from app.core.logging import get_logger
from app.core.utils import git_commit, iso_days_ago, truncate, utcnow_iso
from app.schemas.llm import LLMTrace, TokenUsage

logger = get_logger(__name__)

# Prompts and completions can be large; cap what is persisted so the trace store
# stays a debugging aid rather than a data lake.
MAX_STORED_CHARS = 4000


class TraceStore:
    """Repository over ``llm_traces``."""

    def __init__(self, db: Database | None = None, settings: Settings | None = None) -> None:
        self._db = db
        self.settings = settings or get_settings()

    @property
    def db(self) -> Database:
        if self._db is None:
            self._db = get_database()
        return self._db

    def record(
        self,
        request_id: str,
        provider: str,
        model: str,
        usage: TokenUsage,
        latency_ms: float,
        estimated_cost_usd: float,
        prompt_name: str | None = None,
        prompt_version: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        status: str = "ok",
        error_code: str | None = None,
        safety_verdict: str | None = None,
        rendered_prompt: str | None = None,
        output_text: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> LLMTrace:
        trace = LLMTrace(
            request_id=request_id,
            provider=provider,
            model=model,
            prompt_name=prompt_name,
            prompt_version=prompt_version,
            temperature=temperature,
            max_tokens=max_tokens,
            usage=usage,
            latency_ms=round(float(latency_ms), 3),
            estimated_cost_usd=round(float(estimated_cost_usd), 8),
            status=status,
            error_code=error_code,
            safety_verdict=safety_verdict,
            rendered_prompt=truncate(rendered_prompt or "", MAX_STORED_CHARS) or None,
            output_text=truncate(output_text or "", MAX_STORED_CHARS) or None,
            metadata={**(metadata or {}), "tokens_estimated": usage.estimated},
            git_commit=git_commit(),
            created_at=utcnow_iso(),
        )

        try:
            self.db.execute(
                "INSERT INTO llm_traces (id, request_id, provider, model, prompt_name, "
                "prompt_version, temperature, max_tokens, input_tokens, output_tokens, "
                "total_tokens, latency_ms, estimated_cost_usd, status, error_code, "
                "safety_verdict, rendered_prompt, output_text, metadata, git_commit, "
                "created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    trace.id,
                    trace.request_id,
                    trace.provider,
                    trace.model,
                    trace.prompt_name,
                    trace.prompt_version,
                    trace.temperature,
                    trace.max_tokens,
                    trace.usage.input_tokens,
                    trace.usage.output_tokens,
                    trace.usage.total_tokens,
                    trace.latency_ms,
                    trace.estimated_cost_usd,
                    trace.status,
                    trace.error_code,
                    trace.safety_verdict,
                    trace.rendered_prompt,
                    trace.output_text,
                    dumps(trace.metadata),
                    trace.git_commit,
                    trace.created_at,
                ),
            )
        except Exception as exc:
            # Losing a trace must not fail the caller's request.
            logger.error(
                "llm.trace_persist_failed",
                extra={"request_id": request_id, "error": str(exc)},
            )

        logger.info(
            "llm.call",
            extra={
                "trace_id": trace.id,
                "provider": provider,
                "model": model,
                "prompt": f"{prompt_name}@{prompt_version}" if prompt_name else None,
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
                "total_tokens": usage.total_tokens,
                "tokens_estimated": usage.estimated,
                "latency_ms": trace.latency_ms,
                "estimated_cost_usd": trace.estimated_cost_usd,
                "status": status,
                "safety_verdict": safety_verdict,
            },
        )
        return trace

    # -- reads ---------------------------------------------------------------- #
    def recent(
        self,
        limit: int = 50,
        model: str | None = None,
        prompt_name: str | None = None,
        status: str | None = None,
    ) -> list[LLMTrace]:
        sql = "SELECT * FROM llm_traces"
        clauses: list[str] = []
        params: list[Any] = []
        if model:
            clauses.append("model = ?")
            params.append(model)
        if prompt_name:
            clauses.append("prompt_name = ?")
            params.append(prompt_name)
        if status:
            clauses.append("status = ?")
            params.append(status)
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        return [_to_trace(dict(row)) for row in self.db.query(sql, params)]

    def get(self, trace_id: str) -> LLMTrace | None:
        row = self.db.query_one("SELECT * FROM llm_traces WHERE id = ?", (trace_id,))
        return _to_trace(dict(row)) if row else None

    def token_totals(self, days: int = 30) -> dict[str, Any]:
        row = self.db.query_one(
            "SELECT COUNT(*) AS calls, "
            "SUM(CASE WHEN status = 'ok' THEN 1 ELSE 0 END) AS successful, "
            "COALESCE(SUM(input_tokens), 0) AS input_tokens, "
            "COALESCE(SUM(output_tokens), 0) AS output_tokens, "
            "COALESCE(SUM(total_tokens), 0) AS total_tokens, "
            "COALESCE(AVG(latency_ms), 0) AS avg_latency_ms, "
            "COALESCE(SUM(estimated_cost_usd), 0) AS cost "
            "FROM llm_traces WHERE created_at >= ?",
            (iso_days_ago(days),),
        )
        data = dict(row) if row else {}
        calls = int(data.get("calls") or 0)
        successful = int(data.get("successful") or 0)
        return {
            "window_days": days,
            "calls": calls,
            "successful_calls": successful,
            "failed_calls": calls - successful,
            "error_rate": round((calls - successful) / calls, 5) if calls else 0.0,
            "input_tokens": int(data.get("input_tokens") or 0),
            "output_tokens": int(data.get("output_tokens") or 0),
            "total_tokens": int(data.get("total_tokens") or 0),
            "avg_latency_ms": round(float(data.get("avg_latency_ms") or 0.0), 2),
            "estimated_cost_usd": round(float(data.get("cost") or 0.0), 6),
            "avg_tokens_per_call": (
                round(int(data.get("total_tokens") or 0) / calls, 1) if calls else 0.0
            ),
        }

    def by_prompt_version(self, days: int = 30) -> list[dict[str, Any]]:
        """Usage broken down by prompt version -- shows rollout progress."""
        rows = self.db.query(
            "SELECT prompt_name, prompt_version, COUNT(*) AS calls, "
            "COALESCE(SUM(total_tokens), 0) AS total_tokens, "
            "COALESCE(AVG(latency_ms), 0) AS avg_latency_ms, "
            "COALESCE(SUM(estimated_cost_usd), 0) AS cost "
            "FROM llm_traces WHERE created_at >= ? AND prompt_name IS NOT NULL "
            "GROUP BY prompt_name, prompt_version ORDER BY calls DESC",
            (iso_days_ago(days),),
        )
        return [
            {
                "prompt_name": row["prompt_name"],
                "prompt_version": row["prompt_version"],
                "calls": int(row["calls"]),
                "total_tokens": int(row["total_tokens"]),
                "avg_latency_ms": round(float(row["avg_latency_ms"]), 2),
                "estimated_cost_usd": round(float(row["cost"]), 6),
            }
            for row in rows
        ]


def _to_trace(data: dict[str, Any]) -> LLMTrace:
    usage = TokenUsage(
        input_tokens=int(data.pop("input_tokens", 0) or 0),
        output_tokens=int(data.pop("output_tokens", 0) or 0),
        total_tokens=int(data.pop("total_tokens", 0) or 0),
    )
    metadata = loads(data.pop("metadata", None), {})
    usage.estimated = bool(metadata.get("tokens_estimated", False))
    return LLMTrace(usage=usage, metadata=metadata, **data)


_STORE: TraceStore | None = None


def get_trace_store() -> TraceStore:
    global _STORE
    if _STORE is None:
        _STORE = TraceStore()
    return _STORE
