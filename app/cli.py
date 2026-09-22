"""FMOps command line interface.

Every pipeline stage is reachable from here, which is what lets CI, the
Makefile, cron and a developer's terminal all drive the platform the same way::

    fmops data generate
    fmops data validate
    fmops train --tune
    fmops promote --version 3
    fmops deploy --version 3 --strategy canary
    fmops simulate --rows 500 --drift severe
    fmops drift scan
    fmops retrain --force
    fmops rollback
    fmops llm generate --prompt support_summarizer --var ticket_text="..."
    fmops llm eval --dataset support_triage --compare 1.0.0 1.1.0
    fmops status

Implemented with argparse so the CLI has no dependency of its own.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from app.core.config import get_settings, reload_settings
from app.core.logging import configure_from_settings, get_logger

logger = get_logger("fmops.cli")


def _out(payload: Any, as_json: bool = False) -> None:
    if as_json:
        from app.core.utils import jsonable

        print(json.dumps(jsonable(payload), indent=2))
    else:
        print(payload)


def _rule(title: str) -> None:
    print(f"\n=== {title} ".ljust(72, "=") + "\n")


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def cmd_data_generate(args: argparse.Namespace) -> int:
    from app.data.generator import bootstrap_sample_data
    from app.data.versioning import get_dataset_registry

    paths = bootstrap_sample_data(force=args.force)
    _rule("datasets generated")
    for name, path in paths.items():
        print(f"  {name:18s} {path}")

    registry = get_dataset_registry()
    for key in ("train", "train_v2"):
        record = registry.register(paths[key], description=f"{key} dataset")
        print(
            f"  registered {record.version}  ({record.n_rows} rows, "
            f"hash {record.content_hash[:12]})"
        )
    return 0


def cmd_data_validate(args: argparse.Namespace) -> int:
    from app.data.validation import validate_dataframe
    from app.data.versioning import get_dataset_registry

    registry = get_dataset_registry()
    frame, version = registry.resolve(version=args.version, path=args.path)
    report = validate_dataframe(frame, dataset_version=version.version if version else None)
    _rule("data validation")
    print(report.render_text())
    if args.json:
        _out(report.model_dump(mode="json"), as_json=True)
    return 0 if report.passed else 1


def cmd_data_versions(args: argparse.Namespace) -> int:
    from app.data.versioning import get_dataset_registry

    registry = get_dataset_registry()
    _rule("dataset versions")
    for record in registry.list_versions():
        print(
            f"  {record.version:20s} {record.n_rows:>7d} rows  "
            f"hash={record.content_hash[:12]}  dvc={record.dvc_tracked}  "
            f"{record.description}"
        )
    print(f"\n  DVC: {registry.dvc.status()}")
    return 0


# --------------------------------------------------------------------------- #
# training / promotion
# --------------------------------------------------------------------------- #
def cmd_train(args: argparse.Namespace) -> int:
    from app.schemas.model import TrainingRequest
    from app.training.train import train_model

    request = TrainingRequest(
        dataset_version=args.dataset_version,
        dataset_path=args.dataset_path,
        algorithm=args.algorithm,
        tune=args.tune,
        run_name=args.run_name,
    )
    result = train_model(request)
    _rule("training complete")
    metrics = result.evaluation.metrics if result.evaluation else None
    print(f"  run id           : {result.run_id}")
    print(f"  model version    : {result.registered_version}")
    print(f"  algorithm        : {result.algorithm}")
    print(f"  dataset          : {result.dataset_version}")
    print(f"  git commit       : {result.git_commit[:12]}")
    print(f"  duration         : {result.duration_seconds}s")
    if metrics and result.evaluation is not None:
        print(f"  threshold        : {result.evaluation.threshold:.4f}")
        print(
            f"  metrics          : f1={metrics.f1:.4f} roc_auc={metrics.roc_auc:.4f} "
            f"precision={metrics.precision:.4f} recall={metrics.recall:.4f}"
        )
        print(f"  latency p95      : {metrics.inference_latency_p95_ms:.2f} ms")
    if result.tuning:
        print(
            f"  tuning           : {result.tuning.backend}, "
            f"{result.tuning.n_trials} trials, best "
            f"{result.tuning.best_score:.5f} {result.tuning.best_params}"
        )
    if args.json:
        _out(result.model_dump(mode="json"), as_json=True)
    return 0


def cmd_promote(args: argparse.Namespace) -> int:
    from app.registry.factory import get_registry
    from app.schemas.common import ModelStage
    from app.schemas.model import EvaluationResult, Metrics, TrainingRunResult
    from app.training.registry import register_and_promote

    settings = get_settings()
    registry = get_registry()
    name = settings.tracking.registered_model_name
    latest = registry.get_latest(name)
    version = args.version or (latest.version if latest else None)
    if version is None:
        print("no model versions are registered; run 'fmops train' first")
        return 1

    model_version = registry.get(name, version)
    run = TrainingRunResult(
        run_id=model_version.run_id or "",
        experiment_name=settings.tracking.experiment_name,
        model_name=name,
        algorithm=model_version.algorithm,
        registered_version=version,
        validation_passed=True,
        evaluation=EvaluationResult(
            model_name=name,
            model_version=version,
            metrics=Metrics.model_validate(
                {k: v for k, v in model_version.metrics.items() if k in Metrics.model_fields}
            ),
        ),
    )
    outcome = register_and_promote(
        run,
        registry,
        target_stage=ModelStage(args.stage),
        compare=not args.no_compare,
    )
    _rule(f"approval gate: {name} v{version}")
    print(outcome.render_text())
    if args.json:
        _out(
            {
                "promoted": outcome.promoted,
                "final_stage": outcome.final_stage.value,
                "approval": outcome.approval.model_dump(mode="json"),
                "comparison": (
                    outcome.comparison.model_dump(mode="json") if outcome.comparison else None
                ),
            },
            as_json=True,
        )
    return 0 if outcome.promoted or outcome.approval.approved else 2


def cmd_models(args: argparse.Namespace) -> int:
    from app.registry.factory import get_registry

    registry = get_registry()
    settings = get_settings()
    versions = registry.list_versions(settings.tracking.registered_model_name)
    _rule(f"registry ({registry.backend})")
    if not versions:
        print("  no models registered")
        return 0
    print(f"  {'VER':>4}  {'STAGE':<12} {'STATUS':<10} {'ROC-AUC':>8} {'F1':>8}  DATASET")
    for v in versions:
        print(
            f"  {v.version:>4}  {v.stage.value:<12} {v.status.value:<10} "
            f"{v.metrics.get('roc_auc', 0):>8.4f} {v.metrics.get('f1', 0):>8.4f}  "
            f"{v.dataset_version or '-'}"
        )
    production = registry.get_production(settings.tracking.registered_model_name)
    previous = registry.previous_production(settings.tracking.registered_model_name)
    print(f"\n  production      : v{production.version if production else '-'}")
    print(f"  rollback target : v{previous.version if previous else '-'}")
    return 0


# --------------------------------------------------------------------------- #
# deployment
# --------------------------------------------------------------------------- #
def cmd_deploy(args: argparse.Namespace) -> int:
    from app.deployment.manager import get_deployment_manager
    from app.registry.factory import get_registry
    from app.schemas.common import DeploymentStrategy
    from app.schemas.deployment import DeploymentRequest

    settings = get_settings()
    registry = get_registry()
    name = settings.tracking.registered_model_name
    version = args.version
    if version is None:
        serving = registry.get_serving(name) or registry.get_latest(name)
        if serving is None:
            print("no model versions available to deploy")
            return 1
        version = serving.version

    manager = get_deployment_manager()
    result = manager.deploy(
        DeploymentRequest(
            model_version=version,
            strategy=DeploymentStrategy(args.strategy) if args.strategy else None,
            reason=args.reason,
        ),
        force=args.force,
        actor="cli",
    )
    _rule(f"deployment: {result.strategy.value}")
    print(f"  succeeded    : {result.succeeded}")
    print(f"  message      : {result.message}")
    print(f"  state        : {result.deployment.state.value}")
    print(f"  active       : v{result.deployment.current_version}")
    print(f"  previous     : v{result.deployment.previous_version}")
    print(f"  traffic      : {result.deployment.traffic}")
    for step in result.steps:
        mark = "PASS" if step.passed else "FAIL"
        print(
            f"    [{mark}] step {step.step_index} @{step.traffic_percent}%: "
            f"{step.requests_observed} reqs, err={step.error_rate:.2%}, "
            f"p95={step.latency_p95_ms:.1f}ms -- {step.reason}"
        )
    return 0 if result.succeeded else 2


def cmd_rollback(args: argparse.Namespace) -> int:
    from app.deployment.manager import get_deployment_manager

    result = get_deployment_manager().rollback(
        to_version=args.to_version, reason=args.reason, actor="cli"
    )
    _rule("rollback")
    print(f"  from version : v{result.rolled_back_from}")
    print(f"  to version   : v{result.rolled_back_to}")
    print(f"  succeeded    : {result.succeeded}")
    print(f"  reason       : {result.reason}")
    print(f"  message      : {result.message}")
    return 0 if result.succeeded else 2


def cmd_deployments(args: argparse.Namespace) -> int:
    from app.deployment.manager import get_deployment_manager

    manager = get_deployment_manager()
    _rule("deployments")
    for d in manager.list(limit=args.limit):
        print(
            f"  {d.created_at[:19]}  {d.strategy.value:<11} {d.state.value:<12} "
            f"v{d.current_version}  {d.message[:60]}"
        )
    health = manager.health()
    print(f"\n  endpoint health : {health.status.value}  {health.checks}")
    return 0


# --------------------------------------------------------------------------- #
# traffic / drift / retraining
# --------------------------------------------------------------------------- #
def cmd_simulate(args: argparse.Namespace) -> int:
    from scripts.simulate_traffic import simulate

    stats = simulate(
        rows=args.rows,
        drift=args.drift,
        drift_strength=args.strength,
        label_fraction=args.label_fraction,
        seed=args.seed,
    )
    _rule("traffic simulation")
    for key, value in stats.items():
        print(f"  {key:22s} {value}")
    return 0


def cmd_drift_scan(args: argparse.Namespace) -> int:
    from app.monitoring.service import get_monitoring_service

    report = get_monitoring_service().run_drift_scan()
    _rule("drift scan")
    print(report.render_text())
    if args.json:
        _out(report.model_dump(mode="json"), as_json=True)
    return 2 if report.drift_detected else 0


def cmd_drift_list(args: argparse.Namespace) -> int:
    from app.monitoring.drift import recent_drift_reports

    _rule("drift history")
    for r in recent_drift_reports(limit=args.limit):
        flag = "DRIFT" if r["drift_detected"] else "ok   "
        print(
            f"  {r['created_at'][:19]}  {flag}  score={r['dataset_drift_score']:.4f}  "
            f"features={len(r['drifted_features'])}"
        )
    return 0


def cmd_retrain(args: argparse.Namespace) -> int:
    from app.retraining.pipeline import RetrainingPipeline
    from app.retraining.trigger import evaluate_trigger

    if args.check_only:
        decision = evaluate_trigger(force=args.force)
        _rule("retraining trigger")
        print(decision.render_text())
        return 0 if not decision.should_retrain else 2

    result = RetrainingPipeline().run(force=args.force, deploy=args.deploy)
    _rule("retraining")
    print(f"  triggered        : {result.triggered}")
    print(f"  trigger          : {result.trigger.value if result.trigger else '-'}")
    print(f"  status           : {result.status.value}")
    print(f"  reason           : {result.reason}")
    print(f"  candidate        : v{result.candidate_version}")
    print(f"  production        : v{result.production_version}")
    print(f"  deployed         : {result.deployed}")
    if result.comparison:
        c = result.comparison
        print(
            f"  comparison       : {c.metric} candidate={c.candidate_score:.4f} "
            f"baseline={c.baseline_score:.4f} delta={c.improvement:+.4f} "
            f"(min {c.min_improvement}) -> {c.decision}"
        )
    print(f"  message          : {result.message}")
    if args.json:
        _out(result.model_dump(mode="json"), as_json=True)
    return 0


# --------------------------------------------------------------------------- #
# llmops
# --------------------------------------------------------------------------- #
def cmd_llm_generate(args: argparse.Namespace) -> int:
    from app.llmops.client import get_llm_client
    from app.schemas.llm import LLMGenerateRequest

    variables: dict[str, Any] = {}
    for item in args.var or []:
        if "=" not in item:
            print(f"invalid --var {item!r}; expected key=value")
            return 1
        key, value = item.split("=", 1)
        variables[key] = value

    response = get_llm_client().generate(
        LLMGenerateRequest(
            prompt_name=args.prompt,
            prompt_version=args.prompt_version,
            variables=variables,
            text=args.text,
            model=args.model,
            provider=args.provider,
        )
    )
    _rule("llm response")
    print(response.text)
    print(
        f"\n  provider={response.provider} model={response.model} "
        f"prompt={response.prompt_name}@{response.prompt_version}"
    )
    print(
        f"  tokens: in={response.usage.input_tokens} out={response.usage.output_tokens} "
        f"total={response.usage.total_tokens} (estimated={response.usage.estimated})"
    )
    print(
        f"  latency={response.latency_ms:.1f}ms cost=${response.estimated_cost_usd:.6f} "
        f"trace={response.trace_id}"
    )
    if response.safety:
        print(
            f"  safety: passed={response.safety.passed} "
            f"risk={response.safety.risk_score} "
            f"triggered={[f.check for f in response.safety.triggered_findings]}"
        )
    return 0


def cmd_llm_prompts(args: argparse.Namespace) -> int:
    from app.llmops.prompts.registry import get_prompt_registry

    registry = get_prompt_registry()
    _rule("prompt library")
    for name in registry.list_names():
        versions = registry.list_versions(name)
        print(f"  {name}")
        for v in versions:
            marker = " (latest)" if v.version == versions[-1].version else ""
            print(
                f"      {v.version:<8} hash={v.content_hash[:10]} "
                f"vars={v.variables} tags={v.tags}{marker}"
            )
    return 0


def cmd_llm_eval(args: argparse.Namespace) -> int:
    from app.llmops.evaluation.runner import EvaluationRunner, compare

    runner = EvaluationRunner()
    if args.compare:
        version_a, version_b = args.compare
        result_a = runner.run(args.dataset, prompt_version=version_a, suite="cli-compare")
        result_b = runner.run(args.dataset, prompt_version=version_b, suite="cli-compare")
        comparison = compare(result_a, result_b, args.metric)
        _rule(f"prompt A/B: {comparison.variant_a} vs {comparison.variant_b}")
        print(f"  metric   : {comparison.metric}")
        print(f"  A ({comparison.variant_a}) : {comparison.score_a:.4f}")
        print(f"  B ({comparison.variant_b}) : {comparison.score_b:.4f}")
        print(f"  delta    : {comparison.delta:+.4f}")
        print(f"  winner   : {comparison.winner}")
        print("\n  per metric:")
        for name, values in comparison.per_metric.items():
            print(
                f"    {name:<20} A={values['a']:.4f}  B={values['b']:.4f}  "
                f"delta={values['delta']:+.4f}"
            )
        print(f"\n  cost: A=${comparison.cost_a_usd:.6f} B=${comparison.cost_b_usd:.6f}")
        print(
            f"  latency: A={comparison.latency_a_ms:.1f}ms B={comparison.latency_b_ms:.1f}ms"
        )
        return 0

    result = runner.run(
        args.dataset,
        prompt_version=args.prompt_version,
        model=args.model,
        provider=args.provider,
        suite=args.suite,
    )
    _rule(f"llm evaluation: {result.dataset}")
    print(f"  id           : {result.id}")
    print(f"  provider     : {result.provider}  model: {result.model}")
    print(f"  prompt       : {result.prompt_name}@{result.prompt_version}")
    print(f"  cases        : {result.n_cases}")
    print(f"  tokens       : {result.total_tokens}  cost: ${result.total_cost_usd:.6f}")
    print("\n  scores:")
    for name, value in sorted(result.aggregate.items()):
        print(f"    {name:<22} {value:.4f}")
    if result.provider == "mock":
        print(
            "\n  NOTE: the mock provider is deterministic and offline. These "
            "scores exercise the evaluation harness; they do not measure the "
            "quality of any real model."
        )
    return 0


def cmd_llm_cost(args: argparse.Namespace) -> int:
    from app.llmops.cost import get_cost_tracker
    from app.llmops.token_tracking import get_trace_store

    tracker = get_cost_tracker()
    summary = tracker.summary()
    totals = get_trace_store().token_totals(30)
    _rule("llm cost and tokens (30 days)")
    print(f"  calls            : {totals['calls']} ({totals['failed_calls']} failed)")
    print(
        f"  tokens           : {totals['total_tokens']} "
        f"(in {totals['input_tokens']} / out {totals['output_tokens']})"
    )
    print(f"  avg latency      : {totals['avg_latency_ms']} ms")
    print(
        f"  cost today       : ${summary.today_cost_usd:.6f} "
        f"({summary.daily_budget_used_pct:.1f}% of ${summary.daily_budget_usd})"
    )
    print(
        f"  cost this month  : ${summary.month_cost_usd:.6f} "
        f"({summary.monthly_budget_used_pct:.1f}% of ${summary.monthly_budget_usd})"
    )
    if summary.by_model:
        print("\n  by model:")
        for model, bucket in summary.by_model.items():
            print(
                f"    {model:<45} {bucket.requests:>5} calls  "
                f"{bucket.total_tokens:>8} tok  ${bucket.total_cost_usd:.6f}"
            )
    return 0


def cmd_llm_safety(args: argparse.Namespace) -> int:
    from app.llmops.safety.checks import get_safety_screen

    verdict = get_safety_screen().screen(args.text, args.output or "", args.context or "")
    _rule("safety screen")
    print(f"  passed     : {verdict.passed}")
    print(f"  blocked    : {verdict.blocked}")
    print(f"  risk score : {verdict.risk_score}")
    for finding in verdict.findings:
        mark = "HIT " if finding.triggered else "    "
        print(f"  [{mark}] {finding.check:<26} {finding.detail}")
    print(f"\n  {verdict.detail}")
    return 0


# --------------------------------------------------------------------------- #
# status / serve
# --------------------------------------------------------------------------- #
def cmd_status(args: argparse.Namespace) -> int:
    from app.api.routes.dashboard import dashboard_data

    data = dashboard_data()
    if args.json:
        _out(data, as_json=True)
        return 0

    service = data["service"]
    _rule("FMOps platform status")
    print(
        f"  environment : {service['environment']}  v{service['version']}  "
        f"commit {service['git_commit']}"
    )
    print(
        f"  backends    : deploy={service['deployment_provider']} "
        f"registry={service['registry_backend']} llm={service['llm_provider']} "
        f"aws={'on' if service['aws_enabled'] else 'off'}"
    )

    model = data["model"]
    print("\n  MODEL")
    if model.get("available"):
        print(
            f"    production v{model['current_version']} ({model['algorithm']}), "
            f"previous v{model['previous_version']}"
        )
        print(
            f"    roc_auc={model['metrics'].get('roc_auc', 0):.4f} "
            f"f1={model['metrics'].get('f1', 0):.4f}"
        )
        print(f"    dataset {model['dataset_version']}, {model['total_versions']} versions")
    else:
        print("    no production model")

    deployment = data["deployment"]
    print("\n  DEPLOYMENT")
    if deployment.get("available"):
        print(
            f"    {deployment['endpoint']}: {deployment['state']} / "
            f"{deployment['health']} via {deployment['strategy']}"
        )
        print(f"    traffic {deployment['traffic']}")
    else:
        print("    none")

    drift = data["drift"]
    print("\n  DRIFT")
    if drift.get("available"):
        print(
            f"    score {drift['dataset_drift_score']:.4f} "
            f"(threshold {drift['threshold']}) -> "
            f"{'DETECTED' if drift['drift_detected'] else 'stable'}"
        )
        print(f"    drifted features: {drift['drifted_features']}")
        print(f"    concept drift: {drift['concept_drift_status']}")
    else:
        print("    no scan yet")

    system = data["system"]
    print("\n  SYSTEM")
    if "requests" not in system:
        print(f"    unavailable: {system.get('error', 'no data')}")
    else:
        print(
            f"    {system['requests']} requests/60min  err={system['error_rate']:.2%}  "
            f"p95={system['latency_p95_ms']:.1f}ms  cpu={system['cpu_percent']}%  "
            f"mem={system['memory_percent']}%"
        )

    llm = data["llm"]
    print("\n  LLM")
    print(
        f"    {llm['provider']}/{llm['model']}  {llm['calls']} calls  "
        f"{llm['total_tokens']} tokens  ${llm['today_cost_usd']:.6f} today"
    )

    retraining = data["retraining"]
    print("\n  RETRAINING")
    print(f"    would trigger: {retraining['would_trigger']} -- {retraining['reason']}")

    print(f"\n  OPEN ALERTS : {data['alerts'].get('open_count', 0)}")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "app.api.main:app",
        host=args.host or settings.server.host,
        port=args.port or settings.server.port,
        workers=args.workers or settings.server.workers,
        reload=args.reload,
        log_config=None,
    )
    return 0


def cmd_config(args: argparse.Namespace) -> int:
    _out(get_settings().redacted(), as_json=True)
    return 0


def cmd_aws_status(args: argparse.Namespace) -> int:
    from app.aws.client import aws_status

    _rule("aws status")
    for key, value in aws_status().items():
        print(f"  {key:22s} {value}")
    return 0


# --------------------------------------------------------------------------- #
# parser
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fmops", description="Foundation Model Operations Platform"
    )
    parser.add_argument("--env", help="override FMOPS_ENV for this invocation")
    parser.add_argument("--json", action="store_true", help="emit JSON where supported")
    sub = parser.add_subparsers(dest="command", required=True)

    # data
    data = sub.add_parser("data", help="dataset generation, validation, versions")
    data_sub = data.add_subparsers(dest="subcommand", required=True)
    gen = data_sub.add_parser("generate", help="generate the sample datasets")
    gen.add_argument("--force", action="store_true", help="overwrite existing files")
    gen.set_defaults(func=cmd_data_generate)
    val = data_sub.add_parser("validate", help="run the validation suite")
    val.add_argument("--version", help="registered dataset version")
    val.add_argument("--path", help="path to a CSV file")
    val.set_defaults(func=cmd_data_validate)
    ver = data_sub.add_parser("versions", help="list dataset versions")
    ver.set_defaults(func=cmd_data_versions)

    # train
    train = sub.add_parser("train", help="run the training pipeline")
    train.add_argument("--dataset-version")
    train.add_argument("--dataset-path")
    train.add_argument("--algorithm")
    train.add_argument("--run-name")
    tune_group = train.add_mutually_exclusive_group()
    tune_group.add_argument("--tune", dest="tune", action="store_true", default=None)
    tune_group.add_argument("--no-tune", dest="tune", action="store_false")
    train.set_defaults(func=cmd_train)

    # promote
    promote = sub.add_parser("promote", help="run the approval gate and promote")
    promote.add_argument("--version", type=int)
    promote.add_argument("--stage", default="Production")
    promote.add_argument(
        "--no-compare",
        action="store_true",
        help="skip the champion/challenger comparison",
    )
    promote.set_defaults(func=cmd_promote)

    models = sub.add_parser("models", help="list registered model versions")
    models.set_defaults(func=cmd_models)

    # deploy
    deploy = sub.add_parser("deploy", help="deploy a model version")
    deploy.add_argument("--version", type=int)
    deploy.add_argument("--strategy", choices=["blue_green", "canary", "shadow", "direct"])
    deploy.add_argument("--reason", default="cli deployment")
    deploy.add_argument(
        "--force", action="store_true", help="deploy despite the approval gate"
    )
    deploy.set_defaults(func=cmd_deploy)

    rollback = sub.add_parser("rollback", help="roll back to the previous version")
    rollback.add_argument("--to-version", type=int)
    rollback.add_argument("--reason", default="cli rollback")
    rollback.set_defaults(func=cmd_rollback)

    deployments = sub.add_parser("deployments", help="deployment history")
    deployments.add_argument("--limit", type=int, default=10)
    deployments.set_defaults(func=cmd_deployments)

    # simulate
    simulate = sub.add_parser("simulate", help="send synthetic traffic to the model")
    simulate.add_argument("--rows", type=int, default=500)
    simulate.add_argument(
        "--drift",
        default="none",
        choices=["none", "covariate", "categorical", "concept", "severe"],
    )
    simulate.add_argument("--strength", type=float, default=1.0)
    simulate.add_argument(
        "--label-fraction",
        type=float,
        default=0.0,
        help="fraction of requests to submit ground-truth labels for",
    )
    simulate.add_argument("--seed", type=int, default=None)
    simulate.set_defaults(func=cmd_simulate)

    # drift
    drift = sub.add_parser("drift", help="drift detection")
    drift_sub = drift.add_subparsers(dest="subcommand", required=True)
    scan = drift_sub.add_parser("scan", help="run a drift scan")
    scan.set_defaults(func=cmd_drift_scan)
    dlist = drift_sub.add_parser("list", help="drift history")
    dlist.add_argument("--limit", type=int, default=10)
    dlist.set_defaults(func=cmd_drift_list)

    # retrain
    retrain = sub.add_parser("retrain", help="run the retraining pipeline")
    retrain.add_argument("--force", action="store_true", help="ignore triggers")
    retrain.add_argument(
        "--check-only", action="store_true", help="evaluate triggers without training"
    )
    deploy_group = retrain.add_mutually_exclusive_group()
    deploy_group.add_argument("--deploy", dest="deploy", action="store_true", default=None)
    deploy_group.add_argument("--no-deploy", dest="deploy", action="store_false")
    retrain.set_defaults(func=cmd_retrain)

    # llm
    llm = sub.add_parser("llm", help="LLMOps commands")
    llm_sub = llm.add_subparsers(dest="subcommand", required=True)

    gen_llm = llm_sub.add_parser("generate", help="invoke the configured LLM")
    gen_llm.add_argument("--prompt", help="prompt name from the library")
    gen_llm.add_argument("--prompt-version")
    gen_llm.add_argument("--var", action="append", help="key=value prompt variable")
    gen_llm.add_argument("--text", help="raw prompt text instead of a named prompt")
    gen_llm.add_argument("--model")
    gen_llm.add_argument("--provider")
    gen_llm.set_defaults(func=cmd_llm_generate)

    prompts = llm_sub.add_parser("prompts", help="list prompt versions")
    prompts.set_defaults(func=cmd_llm_prompts)

    ev = llm_sub.add_parser("eval", help="run an evaluation suite")
    ev.add_argument("--dataset", default="support_triage")
    ev.add_argument("--prompt-version")
    ev.add_argument("--model")
    ev.add_argument("--provider")
    ev.add_argument("--suite", default="cli")
    ev.add_argument("--metric", default="overall")
    ev.add_argument(
        "--compare",
        nargs=2,
        metavar=("VERSION_A", "VERSION_B"),
        help="A/B two prompt versions",
    )
    ev.set_defaults(func=cmd_llm_eval)

    cost = llm_sub.add_parser("cost", help="token and cost summary")
    cost.set_defaults(func=cmd_llm_cost)

    safety = llm_sub.add_parser("safety", help="run the safety screen on text")
    safety.add_argument("text")
    safety.add_argument("--output", help="model output to screen as well")
    safety.add_argument("--context", help="grounding context")
    safety.set_defaults(func=cmd_llm_safety)

    # meta
    status = sub.add_parser("status", help="platform status summary")
    status.set_defaults(func=cmd_status)

    serve = sub.add_parser("serve", help="run the API server")
    serve.add_argument("--host")
    serve.add_argument("--port", type=int)
    serve.add_argument("--workers", type=int)
    serve.add_argument("--reload", action="store_true")
    serve.set_defaults(func=cmd_serve)

    config = sub.add_parser("config", help="print the effective configuration")
    config.set_defaults(func=cmd_config)

    aws = sub.add_parser("aws", help="AWS integration commands")
    aws_sub = aws.add_subparsers(dest="subcommand", required=True)
    aws_status = aws_sub.add_parser("status", help="check AWS configuration")
    aws_status.set_defaults(func=cmd_aws_status)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.env:
        import os

        os.environ["FMOPS_ENV"] = args.env
        reload_settings()

    configure_from_settings(force=True)

    try:
        return int(args.func(args) or 0)
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130
    except Exception as exc:
        from app.core.exceptions import FMOpsError

        if isinstance(exc, FMOpsError):
            print(f"\nERROR [{exc.code}] {exc.message}", file=sys.stderr)
            if exc.details:
                from app.core.utils import jsonable

                print(json.dumps(jsonable(exc.details), indent=2)[:4000], file=sys.stderr)
            return 1
        logger.error("cli.unhandled", exc_info=exc)
        print(f"\nERROR {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
