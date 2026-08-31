"""Retraining pipeline entrypoint.

Invoked on a schedule (GitHub Actions cron / EventBridge) and by the drift alert
path. Evaluates the triggers, and only if one fires runs the full retraining
cycle.

Exit codes:

    0  nothing to do, or a candidate was promoted
    2  a candidate was trained and correctly rejected (worse than production, or
       failed the approval gate) -- a successful run with a negative outcome,
       which should not page anyone
    1  the run itself failed
"""

from __future__ import annotations

import argparse
import sys
from typing import Any

from app.core.logging import configure_from_settings, get_logger
from app.core.utils import write_json
from app.retraining.pipeline import RetrainingPipeline
from app.retraining.trigger import evaluate_trigger
from app.schemas.common import RetrainingStatus

logger = get_logger("fmops.pipeline.retraining")


def run(
    force: bool = False,
    check_only: bool = False,
    deploy: bool | None = None,
    report_path: str | None = None,
) -> tuple[int, dict[str, Any]]:
    decision = evaluate_trigger(force=force)
    print(decision.render_text())

    if check_only or not decision.should_retrain:
        report = {
            "pipeline": "retraining",
            "should_retrain": decision.should_retrain,
            "trigger": decision.trigger.value if decision.trigger else None,
            "reason": decision.reason,
            "checks": decision.checks,
            "status": "skipped",
        }
        if report_path:
            write_json(report_path, report)
        return 0, report

    try:
        result = RetrainingPipeline().run(decision=decision, deploy=deploy)
    except Exception as exc:
        logger.error("pipeline.retraining_failed", exc_info=exc)
        report = {"pipeline": "retraining", "status": "failed", "error": str(exc)}
        if report_path:
            write_json(report_path, report)
        return 1, report

    report = result.model_dump(mode="json")
    report["pipeline"] = "retraining"
    if report_path:
        write_json(report_path, report)

    print(f"\nstatus    : {result.status.value}")
    print(f"candidate : v{result.candidate_version}")
    print(f"production: v{result.production_version}")
    print(f"deployed  : {result.deployed}")
    print(f"message   : {result.message}")

    if result.status == RetrainingStatus.FAILED:
        return 1, report
    if result.status == RetrainingStatus.REJECTED:
        # The gate did its job: a worse model was kept out of production.
        return 2, report
    return 0, report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="FMOps retraining pipeline")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--deploy", dest="deploy", action="store_true", default=None)
    group.add_argument("--no-deploy", dest="deploy", action="store_false")
    parser.add_argument("--report")
    args = parser.parse_args(argv)

    configure_from_settings(force=True)
    code, _ = run(args.force, args.check_only, args.deploy, args.report)
    return code


if __name__ == "__main__":
    sys.exit(main())
