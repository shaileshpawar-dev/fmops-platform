"""Deployment pipeline entrypoint.

Used by the CD workflow. Runs the pre-deployment gate, deploys with the
configured strategy, verifies health with a real smoke request, and rolls back
automatically if that smoke test fails.

Exit codes: 0 deployed and healthy, 2 deployment rejected or rolled back,
1 error.
"""

from __future__ import annotations

import argparse
import sys
from typing import Any

from app.core.config import get_settings
from app.core.exceptions import FMOpsError
from app.core.logging import configure_from_settings, get_logger
from app.core.utils import write_json
from app.deployment.manager import get_deployment_manager
from app.registry.factory import get_registry
from app.schemas.common import DeploymentStrategy, HealthStatus
from app.schemas.deployment import DeploymentRequest

logger = get_logger("fmops.pipeline.deployment")


def run(
    version: int | None = None,
    strategy: str | None = None,
    force: bool = False,
    smoke_test: bool = True,
    report_path: str | None = None,
) -> tuple[int, dict[str, Any]]:
    settings = get_settings()
    registry = get_registry()
    manager = get_deployment_manager()
    name = settings.tracking.registered_model_name

    target = version
    if target is None:
        candidate = registry.get_latest(name, None)
        if candidate is None:
            return 1, {"status": "failed", "error": "no model versions registered"}
        target = candidate.version

    try:
        result = manager.deploy(
            DeploymentRequest(
                model_version=target,
                strategy=DeploymentStrategy(strategy) if strategy else None,
                reason="deployment pipeline",
            ),
            force=force,
            actor="pipeline",
        )
    except FMOpsError as exc:
        logger.error("pipeline.deployment_failed", extra={"error": exc.message})
        report: dict[str, Any] = {
            "status": "failed",
            "error": exc.message,
            "model_version": target,
        }
        if report_path:
            write_json(report_path, report)
        return 1, report

    report = {
        "pipeline": "deployment",
        "model_version": target,
        "strategy": result.strategy.value,
        "succeeded": result.succeeded,
        "rolled_back": result.rolled_back,
        "message": result.message,
        "state": result.deployment.state.value,
        "traffic": result.deployment.traffic,
        "steps": [s.model_dump(mode="json") for s in result.steps],
    }

    if result.succeeded and smoke_test:
        passed, detail = _smoke_test()
        report["smoke_test"] = {"passed": passed, "detail": detail}
        if not passed:
            logger.error("pipeline.smoke_test_failed", extra={"detail": detail})
            rollback = manager.rollback(
                reason=f"smoke test failed after deployment: {detail}",
                actor="pipeline",
            )
            report["rollback"] = rollback.model_dump(mode="json")
            report["succeeded"] = False
            if report_path:
                write_json(report_path, report)
            return 2, report

    health = manager.health()
    report["health"] = health.status.value
    if report_path:
        write_json(report_path, report)

    print(f"deployment: {result.message}")
    print(f"health    : {health.status.value} {health.checks}")

    if result.succeeded and health.status != HealthStatus.UNHEALTHY:
        return 0, report
    return 2, report


def _smoke_test() -> tuple[bool, str]:
    """Score one known-good request through the live serving path."""
    from app.api.serving import get_prediction_service
    from app.core.logging import new_request_id

    sample = {
        "age": 41.0,
        "annual_income": 72000.0,
        "loan_amount": 21000.0,
        "loan_term_months": 48,
        "credit_score": 705.0,
        "debt_to_income": 0.24,
        "employment_years": 8.0,
        "num_credit_lines": 7,
        "num_late_payments_12m": 0,
        "credit_utilization": 0.28,
        "employment_type": "salaried",
        "housing_status": "mortgage",
        "loan_purpose": "home_improvement",
        "region": "west",
    }
    try:
        response = get_prediction_service().predict_one(
            features=sample, request_id=new_request_id()
        )
    except Exception as exc:
        return False, f"smoke request raised: {exc}"

    if not 0.0 <= response.probability <= 1.0:
        return False, f"probability out of range: {response.probability}"
    if response.prediction not in (0, 1):
        return False, f"prediction was not binary: {response.prediction}"
    return True, (
        f"scored v{response.model_version} p={response.probability:.4f} "
        f"in {response.inference_latency_ms:.1f}ms"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="FMOps deployment pipeline")
    parser.add_argument("--version", type=int)
    parser.add_argument("--strategy", choices=["blue_green", "canary", "shadow", "direct"])
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-smoke-test", action="store_true")
    parser.add_argument("--report")
    args = parser.parse_args(argv)

    configure_from_settings(force=True)
    code, _ = run(
        version=args.version,
        strategy=args.strategy,
        force=args.force,
        smoke_test=not args.no_smoke_test,
        report_path=args.report,
    )
    return code


if __name__ == "__main__":
    sys.exit(main())
