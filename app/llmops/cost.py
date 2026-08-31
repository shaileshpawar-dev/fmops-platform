"""Token cost accounting.

Cost is computed from token counts and a configurable price table
(``FMOPS_LLM__PRICING``, USD per **million** tokens, separate input and output
rates). Two honesty rules are enforced:

* A model with no price-table entry is recorded with ``priced=False`` and a cost
  of 0.00, and a warning is logged naming the model. Silently pricing an unknown
  model at zero is how spend disappears from dashboards.
* When token counts were *estimated* (a provider that returned no usage block),
  the derived cost is an estimate of an estimate. That is surfaced through the
  trace's ``usage.estimated`` flag rather than hidden.

These figures are for engineering visibility -- they are not a billing system
and will not match a vendor invoice exactly.
"""

from __future__ import annotations

from typing import Any

from app.core.config import Settings, get_settings
from app.core.db import Database, get_database
from app.core.logging import get_logger
from app.core.utils import iso_days_ago
from app.schemas.llm import CostBucket, CostRecord, CostSummary, TokenUsage

logger = get_logger(__name__)

TOKENS_PER_PRICE_UNIT = 1_000_000

_WARNED_MODELS: set[str] = set()


class CostCalculator:
    """Turns token usage into money."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def price_for(self, model: str) -> tuple[float, float, bool]:
        """(input_rate, output_rate, priced) per million tokens."""
        pricing = self.settings.llm.pricing
        entry = pricing.get(model)
        if entry is None:
            # Try a prefix match: vendors version model ids heavily.
            for key, value in pricing.items():
                if model.startswith(key) or key in model:
                    entry = value
                    break
        if entry is None:
            if model not in _WARNED_MODELS:
                _WARNED_MODELS.add(model)
                logger.warning(
                    "cost.model_not_priced",
                    extra={
                        "model": model,
                        "impact": "spend for this model is recorded as 0.00 USD",
                        "fix": "add it to FMOPS_LLM__PRICING",
                    },
                )
            return 0.0, 0.0, False
        return float(entry.get("input", 0.0)), float(entry.get("output", 0.0)), True

    def compute(self, model: str, provider: str, usage: TokenUsage) -> CostRecord:
        input_rate, output_rate, priced = self.price_for(model)
        input_cost = (usage.input_tokens / TOKENS_PER_PRICE_UNIT) * input_rate
        output_cost = (usage.output_tokens / TOKENS_PER_PRICE_UNIT) * output_rate
        return CostRecord(
            model=model,
            provider=provider,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            total_tokens=usage.total_tokens,
            input_cost_usd=round(input_cost, 8),
            output_cost_usd=round(output_cost, 8),
            total_cost_usd=round(input_cost + output_cost, 8),
            priced=priced,
        )


class CostTracker:
    """Aggregates spend from the persisted LLM traces."""

    def __init__(self, settings: Settings | None = None, db: Database | None = None) -> None:
        self.settings = settings or get_settings()
        self._db = db
        self.calculator = CostCalculator(self.settings)

    @property
    def db(self) -> Database:
        if self._db is None:
            self._db = get_database()
        return self._db

    # -- aggregates ----------------------------------------------------------- #
    def _bucket(self, expression: str, days: int, limit: int) -> list[CostBucket]:
        rows = self.db.query(
            f"SELECT {expression} AS period, COUNT(*) AS requests, "
            "SUM(CASE WHEN status = 'ok' THEN 1 ELSE 0 END) AS successful, "
            "SUM(input_tokens) AS input_tokens, SUM(output_tokens) AS output_tokens, "
            "SUM(total_tokens) AS total_tokens, SUM(estimated_cost_usd) AS cost "
            "FROM llm_traces WHERE created_at >= ? "
            "GROUP BY period ORDER BY period DESC LIMIT ?",
            (iso_days_ago(days), limit),
        )
        return [_to_bucket(dict(row)) for row in rows]

    def daily(self, days: int = 30) -> list[CostBucket]:
        return self._bucket("substr(created_at, 1, 10)", days, days)

    def weekly(self, weeks: int = 12) -> list[CostBucket]:
        # SQLite has no ISO-week function; %W is close enough for a trend line.
        return self._bucket("strftime('%Y-W%W', created_at)", weeks * 7, weeks)

    def monthly(self, months: int = 12) -> list[CostBucket]:
        return self._bucket("substr(created_at, 1, 7)", months * 31, months)

    def by_model(self, days: int = 30) -> dict[str, CostBucket]:
        rows = self.db.query(
            "SELECT model AS period, COUNT(*) AS requests, "
            "SUM(CASE WHEN status = 'ok' THEN 1 ELSE 0 END) AS successful, "
            "SUM(input_tokens) AS input_tokens, SUM(output_tokens) AS output_tokens, "
            "SUM(total_tokens) AS total_tokens, SUM(estimated_cost_usd) AS cost "
            "FROM llm_traces WHERE created_at >= ? GROUP BY model ORDER BY cost DESC",
            (iso_days_ago(days),),
        )
        return {row["period"]: _to_bucket(dict(row)) for row in rows}

    def total_since(self, iso_timestamp: str) -> float:
        return float(
            self.db.scalar(
                "SELECT COALESCE(SUM(estimated_cost_usd), 0) FROM llm_traces "
                "WHERE created_at >= ?",
                (iso_timestamp,),
                0.0,
            )
        )

    def today_cost(self) -> float:
        from app.core.utils import utcnow

        return self.total_since(utcnow().strftime("%Y-%m-%dT00:00:00.000Z"))

    def month_cost(self) -> float:
        from app.core.utils import utcnow

        return self.total_since(utcnow().strftime("%Y-%m-01T00:00:00.000Z"))

    def summary(self) -> CostSummary:
        today = self.today_cost()
        month = self.month_cost()
        daily_budget = self.settings.llm.daily_cost_budget_usd
        monthly_budget = self.settings.llm.monthly_cost_budget_usd
        return CostSummary(
            daily=self.daily(),
            weekly=self.weekly(),
            monthly=self.monthly(),
            by_model=self.by_model(),
            today_cost_usd=round(today, 6),
            month_cost_usd=round(month, 6),
            daily_budget_usd=daily_budget,
            monthly_budget_usd=monthly_budget,
            daily_budget_used_pct=(
                round(100.0 * today / daily_budget, 2) if daily_budget else 0.0
            ),
            monthly_budget_used_pct=(
                round(100.0 * month / monthly_budget, 2) if monthly_budget else 0.0
            ),
        )

    # -- budget enforcement ---------------------------------------------------- #
    def check_budgets(self) -> list[dict[str, Any]]:
        """Raise alerts when spend crosses 80% or 100% of a budget."""
        from app.monitoring.alerts import get_alert_manager
        from app.schemas.common import AlertCategory, Severity

        breaches: list[dict[str, Any]] = []
        alerts = get_alert_manager()

        for scope, spend, budget in (
            ("daily", self.today_cost(), self.settings.llm.daily_cost_budget_usd),
            ("monthly", self.month_cost(), self.settings.llm.monthly_cost_budget_usd),
        ):
            if budget <= 0:
                continue
            used = 100.0 * spend / budget
            if used < 80.0:
                continue
            severity = Severity.CRITICAL if used >= 100.0 else Severity.WARNING
            breaches.append(
                {"scope": scope, "spend": spend, "budget": budget, "used_pct": used}
            )
            alerts.raise_alert(
                severity,
                AlertCategory.COST,
                f"LLM {scope} budget at {used:.0f}%",
                (
                    f"Estimated {scope} LLM spend is ${spend:.2f} against a budget of "
                    f"${budget:.2f} ({used:.1f}% used)."
                ),
                context={"scope": scope, "spend_usd": spend, "budget_usd": budget},
                dedupe_keys=("scope",),
            )
        return breaches


def _to_bucket(row: dict[str, Any]) -> CostBucket:
    requests = int(row.get("requests") or 0)
    successful = int(row.get("successful") or 0)
    cost = float(row.get("cost") or 0.0)
    return CostBucket(
        period=str(row.get("period") or ""),
        requests=requests,
        successful_requests=successful,
        input_tokens=int(row.get("input_tokens") or 0),
        output_tokens=int(row.get("output_tokens") or 0),
        total_tokens=int(row.get("total_tokens") or 0),
        total_cost_usd=round(cost, 6),
        cost_per_request=round(cost / requests, 8) if requests else 0.0,
        cost_per_successful_response=round(cost / successful, 8) if successful else 0.0,
    )


_TRACKER: CostTracker | None = None


def get_cost_tracker() -> CostTracker:
    global _TRACKER
    if _TRACKER is None:
        _TRACKER = CostTracker()
    return _TRACKER
