"""End-to-end lifecycle tests.

These are the acceptance tests for the platform's promises:

* a valid dataset trains, registers and promotes,
* an invalid dataset stops the pipeline before any model is trained,
* drift is detected and fires the retraining trigger,
* a retrained model that is worse is REJECTED and production is untouched,
* a retrained model that is better is promoted and deployed,
* a degraded production endpoint can be rolled back.

They are slower than the unit suite because they really train models, so they
are marked ``pipeline`` and run in their own CI lane.
"""

from __future__ import annotations

from contextlib import contextmanager

import pytest

from app.core.exceptions import DataValidationError
from app.schemas.common import (
    DeploymentStrategy,
    ModelStage,
    RetrainingStatus,
)
from app.schemas.deployment import DeploymentRequest
from app.schemas.model import TrainingRequest

pytestmark = [pytest.mark.pipeline, pytest.mark.slow]


@pytest.fixture
def platform(clean_singletons, settings, registry, valid_frame, tmp_path):
    """A wired-up platform with an isolated dataset registry."""
    import app.data.versioning as versioning
    from app.data.versioning import DatasetRegistry
    from app.deployment.manager import DeploymentManager

    datasets = DatasetRegistry(settings)
    datasets.manifest_path = tmp_path / "versions.json"
    versioning._REGISTRY = datasets

    path = tmp_path / "train.csv"
    valid_frame.to_csv(path, index=False)
    record = datasets.register(path, use_dvc=False, description="baseline")

    manager = DeploymentManager(registry=registry, settings=settings)
    try:
        yield {
            "datasets": datasets,
            "dataset_version": record.version,
            "registry": registry,
            "manager": manager,
            "settings": settings,
            "tmp_path": tmp_path,
        }
    finally:
        versioning._REGISTRY = None


def _train(platform, **kwargs):
    from app.training.train import train_model

    payload = {"dataset_version": platform["dataset_version"], "tune": False}
    payload.update(kwargs)
    return train_model(TrainingRequest(**payload), registry=platform["registry"])


@contextmanager
def override(settings, section: str, **fields):
    """Temporarily override a settings section.

    ``settings`` is session-scoped, so a test that mutates it without restoring
    the ORIGINAL object silently reconfigures every later test.
    """
    original = getattr(settings, section)
    object.__setattr__(settings, section, original.model_copy(update=fields))
    try:
        yield settings
    finally:
        object.__setattr__(settings, section, original)


def _promote(platform, run, compare=True):
    from app.training.registry import register_and_promote

    return register_and_promote(
        run,
        platform["registry"],
        target_stage=ModelStage.PRODUCTION,
        compare=compare,
    )


# --------------------------------------------------------------------------- #
# Happy path
# --------------------------------------------------------------------------- #
def test_valid_dataset_trains_registers_and_promotes(platform):
    run = _train(platform)

    assert run.registered_version == 1
    assert run.validation_passed
    assert run.evaluation is not None
    assert run.evaluation.metrics.roc_auc > 0.7
    assert run.dataset_version == platform["dataset_version"]
    assert run.dataset_hash, "the run must record the exact bytes it trained on"

    outcome = _promote(platform, run)
    assert outcome.promoted
    assert outcome.final_stage == ModelStage.PRODUCTION
    assert (
        platform["registry"]
        .get_production(platform["settings"].tracking.registered_model_name)
        .version
        == 1
    )


def test_training_run_is_fully_reproducible_on_paper(platform):
    """Every field needed to re-create the run must be recorded."""
    run = _train(platform)
    assert run.git_commit
    assert run.dataset_version
    assert run.dataset_hash
    assert run.params, "the effective hyperparameters must be recorded"
    assert "max_iter" in run.params, "estimator defaults must be resolved, not implicit"
    assert run.environment["python"]
    assert run.environment["scikit_learn"]
    assert run.model_path
    assert run.artifact_uri


def test_deployment_makes_the_model_servable(platform, sample_features):
    from app.api.serving import PredictionService
    from app.core.logging import new_request_id

    run = _train(platform)
    _promote(platform, run)
    result = platform["manager"].deploy(
        DeploymentRequest(
            model_version=run.registered_version, strategy=DeploymentStrategy.BLUE_GREEN
        )
    )
    assert result.succeeded

    service = PredictionService(settings=platform["settings"], registry=platform["registry"])
    response = service.predict_one(sample_features, new_request_id())
    assert response.model_version == run.registered_version
    assert 0.0 <= response.probability <= 1.0


# --------------------------------------------------------------------------- #
# Failure paths
# --------------------------------------------------------------------------- #
def test_invalid_dataset_stops_the_pipeline_before_training(platform, invalid_frame):
    """No model may be produced from data that failed validation."""
    bad_path = platform["tmp_path"] / "bad.csv"
    invalid_frame.to_csv(bad_path, index=False)
    record = platform["datasets"].register(bad_path, use_dvc=False, description="corrupt")

    before = len(platform["registry"].list_versions())
    with pytest.raises(DataValidationError) as excinfo:
        _train(platform, dataset_version=record.version)

    assert excinfo.value.details["failed"]
    after = len(platform["registry"].list_versions())
    assert after == before, "a model was registered from invalid data"


def test_training_pipeline_entrypoint_reports_validation_failure(platform, invalid_frame):
    from pipelines.training_pipeline import run as run_pipeline

    bad_path = platform["tmp_path"] / "bad2.csv"
    invalid_frame.to_csv(bad_path, index=False)
    record = platform["datasets"].register(bad_path, use_dvc=False)

    code, report = run_pipeline(dataset_version=record.version, tune=False)
    assert code == 1
    assert report["stage"] == "data_validation"
    assert report["failed_expectations"]


def test_pipeline_exit_code_2_means_rejected_not_broken(platform):
    """CI must distinguish 'the gate said no' from 'the pipeline crashed'."""
    from pipelines.training_pipeline import run as run_pipeline

    with override(platform["settings"], "approval", min_roc_auc=0.999):
        code, report = run_pipeline(tune=False, promote=True)
    assert code == 2
    assert report["approval_decision"] == "rejected"
    assert report["promoted"] is False


# --------------------------------------------------------------------------- #
# Drift -> retraining
# --------------------------------------------------------------------------- #
def _send_traffic(platform, frame, label_fraction=0.0):
    """Score rows through the real serving path so the inference log fills."""
    from app.api.serving import PredictionService
    from app.core.logging import new_request_id
    from app.monitoring.inference_log import get_inference_log

    service = PredictionService(settings=platform["settings"], registry=platform["registry"])
    log = get_inference_log()
    settings = platform["settings"]
    scored = 0

    for index, row in frame.iterrows():
        features = {}
        for column in settings.data.feature_columns:
            value = row[column]
            item = getattr(value, "item", None)
            features[column] = item() if callable(item) else value

        request_id = new_request_id()
        service.predict_one(features, request_id)
        scored += 1
        if label_fraction and index % max(1, int(1 / label_fraction)) == 0:
            log.record_feedback(request_id, int(row[settings.data.target_column]))
    return scored


def test_drift_is_detected_from_production_traffic(platform, drifted_frame):
    from app.monitoring.service import MonitoringService

    run = _train(platform)
    _promote(platform, run)
    platform["manager"].deploy(
        DeploymentRequest(
            model_version=run.registered_version, strategy=DeploymentStrategy.DIRECT
        )
    )

    _send_traffic(platform, drifted_frame.head(400))

    report = MonitoringService(platform["settings"], platform["registry"]).run_drift_scan()
    assert report.drift_detected
    assert report.drifted_features
    assert report.n_current >= 200


def test_clean_traffic_does_not_trigger_drift(platform, valid_frame):
    from app.monitoring.service import MonitoringService

    run = _train(platform)
    _promote(platform, run)
    platform["manager"].deploy(
        DeploymentRequest(
            model_version=run.registered_version, strategy=DeploymentStrategy.DIRECT
        )
    )

    _send_traffic(platform, valid_frame.head(400))

    report = MonitoringService(platform["settings"], platform["registry"]).run_drift_scan()
    assert not report.drift_detected, (
        f"clean traffic produced a false positive: score={report.dataset_drift_score} "
        f"features={report.drifted_features}"
    )


def test_drift_fires_the_retraining_trigger(platform, drifted_frame):
    """Drift plus labelled production rows: there is something new to learn from."""
    from app.monitoring.service import MonitoringService
    from app.retraining.trigger import evaluate_trigger

    run = _train(platform)
    _promote(platform, run)
    platform["manager"].deploy(
        DeploymentRequest(
            model_version=run.registered_version, strategy=DeploymentStrategy.DIRECT
        )
    )
    _send_traffic(platform, drifted_frame.head(400), label_fraction=0.5)
    MonitoringService(platform["settings"], platform["registry"]).run_drift_scan()

    decision = evaluate_trigger()
    assert decision.should_retrain
    assert decision.trigger.value == "drift"
    assert decision.evidence["drifted_features"]


def test_drift_without_new_labels_does_not_retrain(platform, drifted_frame):
    """Drift alone is detected and reported, but does not start a retrain.

    Retraining on exactly the data the serving version was trained on cannot
    correct drift. The platform used to paper over that by appending generated
    rows; now it says what is missing instead.
    """
    from app.monitoring.service import MonitoringService
    from app.retraining.trigger import evaluate_trigger

    run = _train(platform)
    _promote(platform, run)
    platform["manager"].deploy(
        DeploymentRequest(
            model_version=run.registered_version, strategy=DeploymentStrategy.DIRECT
        )
    )
    _send_traffic(platform, drifted_frame.head(400))
    MonitoringService(platform["settings"], platform["registry"]).run_drift_scan()

    decision = evaluate_trigger()
    assert not decision.should_retrain
    assert "no new labelled data" in decision.reason
    assert decision.evidence["blocked_by"] == "no_new_labelled_data"
    drift_check = next(c for c in decision.checks if c["name"] == "drift")
    assert drift_check["fired"], "the drift itself must still be detected and shown"


def test_no_trigger_without_evidence(platform):
    from app.retraining.trigger import evaluate_trigger

    run = _train(platform)
    _promote(platform, run)
    decision = evaluate_trigger()
    assert not decision.should_retrain
    assert "no retraining trigger fired" in decision.reason


def test_cooldown_suppresses_repeat_triggers(platform, drifted_frame):
    from app.monitoring.service import MonitoringService
    from app.retraining.trigger import RetrainingTriggerEvaluator, get_event_store
    from app.schemas.common import RetrainingTrigger

    run = _train(platform)
    _promote(platform, run)
    platform["manager"].deploy(
        DeploymentRequest(
            model_version=run.registered_version, strategy=DeploymentStrategy.DIRECT
        )
    )
    _send_traffic(platform, drifted_frame.head(400))
    MonitoringService(platform["settings"], platform["registry"]).run_drift_scan()

    # A retraining event just happened; a cooldown must suppress the next trigger.
    get_event_store().create(
        RetrainingTrigger.DRIFT, "previous run", "loan_default_classifier"
    )
    with override(platform["settings"], "retraining", cooldown_minutes=120) as settings:
        decision = RetrainingTriggerEvaluator(settings).evaluate()
    assert not decision.should_retrain
    assert decision.suppressed_by_cooldown


# --------------------------------------------------------------------------- #
# Retraining decisions -- the core guarantee
# --------------------------------------------------------------------------- #
def test_worse_candidate_is_rejected_and_production_is_untouched(platform):
    """The single most important behaviour in the platform.

    A retrained model that does not beat production must never replace it.
    """
    from app.retraining.pipeline import RetrainingPipeline

    run = _train(platform)
    _promote(platform, run)
    registry = platform["registry"]
    name = platform["settings"].tracking.registered_model_name
    production_before = registry.get_production(name)

    # Force the comparison to be unwinnable: demand a huge improvement.
    with override(platform["settings"], "approval", min_improvement=0.5) as settings:
        pipeline = RetrainingPipeline(
            settings=settings, registry=registry, deployment_manager=platform["manager"]
        )
        result = pipeline.run(force=True, collect_production_data=False)

    assert result.status == RetrainingStatus.REJECTED
    assert result.deployed is False
    assert result.comparison is not None
    assert not result.comparison.candidate_is_better

    production_after = registry.get_production(name)
    assert (
        production_after.version == production_before.version
    ), "a rejected candidate replaced the production model"
    candidate = registry.get(name, result.candidate_version)
    assert candidate.status.value == "rejected"
    assert candidate.stage == ModelStage.DEVELOPMENT


def test_better_candidate_is_promoted_and_deployed(platform, monkeypatch):
    """Given a comparison the candidate wins, it is promoted and goes live.

    This isolates the promotion path from run-to-run training noise. It used to
    do so by overwriting the incumbent's recorded metric -- which no longer
    decides anything: both models are now re-scored on a shared holdout, so a
    forged number cannot make a candidate win. The comparison is stubbed here
    instead, and is itself tested against real models in tests/e2e and
    tests/unit/test_holdout.py.
    """
    import app.retraining.pipeline as pipeline_module
    from app.retraining.pipeline import RetrainingPipeline
    from app.training.approval import compare_to_production

    run = _train(platform)
    _promote(platform, run)
    registry = platform["registry"]
    name = platform["settings"].tracking.registered_model_name

    def candidate_wins(candidate, incumbent, settings=None):
        return compare_to_production(
            candidate.metrics,
            {"roc_auc": 0.55, "f1": 0.40},
            candidate_version=candidate.version,
            baseline_version=incumbent.version if incumbent else None,
            settings=settings,
        )

    monkeypatch.setattr(pipeline_module, "compare_on_shared_holdout", candidate_wins)

    pipeline = RetrainingPipeline(
        settings=platform["settings"],
        registry=registry,
        deployment_manager=platform["manager"],
    )
    result = pipeline.run(force=True, deploy=True, collect_production_data=False)

    assert result.status == RetrainingStatus.SUCCEEDED
    assert result.comparison.candidate_is_better
    assert result.candidate_version != run.registered_version
    assert result.deployed
    assert registry.get_production(name).version == result.candidate_version


def test_retraining_pipeline_entrypoint_exit_codes(platform):
    from pipelines.retraining_pipeline import run as run_pipeline

    run = _train(platform)
    _promote(platform, run)

    # No trigger fired -> exit 0, nothing done.
    code, report = run_pipeline(check_only=True)
    assert code == 0
    assert report["status"] == "skipped"


# --------------------------------------------------------------------------- #
# Rollback
# --------------------------------------------------------------------------- #
def test_rollback_restores_the_previous_model_end_to_end(platform):
    from app.api.serving import PredictionService

    registry = platform["registry"]
    manager = platform["manager"]
    name = platform["settings"].tracking.registered_model_name

    first = _train(platform)
    _promote(platform, first)
    manager.deploy(
        DeploymentRequest(
            model_version=first.registered_version, strategy=DeploymentStrategy.BLUE_GREEN
        )
    )

    second = _train(platform, run_name="second")
    # Only a Production version takes live traffic.
    for stage in (ModelStage.VALIDATION, ModelStage.STAGING, ModelStage.PRODUCTION):
        registry.transition_stage(name, second.registered_version, stage)
    manager.deploy(
        DeploymentRequest(
            model_version=second.registered_version, strategy=DeploymentStrategy.BLUE_GREEN
        )
    )
    assert registry.get_production(name).version == second.registered_version

    result = manager.rollback(reason="latency regression after rollout")

    assert result.succeeded
    assert result.rolled_back_to == first.registered_version
    assert registry.get_production(name).version == first.registered_version

    # And traffic really goes to the restored version.
    service = PredictionService(settings=platform["settings"], registry=registry)
    model, _variant = service.resolve_model()
    assert model.version == first.registered_version


def test_deployment_pipeline_smoke_test_and_health(platform):
    from pipelines.deployment_pipeline import run as run_pipeline

    run = _train(platform)
    _promote(platform, run)

    code, report = run_pipeline(
        version=run.registered_version, strategy="direct", smoke_test=True
    )
    assert code == 0
    assert report["succeeded"]
    assert report["smoke_test"]["passed"]
    assert report["health"] == "healthy"


# --------------------------------------------------------------------------- #
# Shadow
# --------------------------------------------------------------------------- #
def test_shadow_records_predictions_without_serving_them(platform, sample_features):
    from app.api.serving import PredictionService
    from app.core.logging import new_request_id
    from app.monitoring.inference_log import get_inference_log

    registry = platform["registry"]
    manager = platform["manager"]
    name = platform["settings"].tracking.registered_model_name

    first = _train(platform)
    _promote(platform, first)
    manager.deploy(
        DeploymentRequest(
            model_version=first.registered_version, strategy=DeploymentStrategy.DIRECT
        )
    )

    second = _train(platform, run_name="shadow-candidate")
    for stage in (ModelStage.VALIDATION, ModelStage.STAGING):
        registry.transition_stage(name, second.registered_version, stage)
    result = manager.deploy(
        DeploymentRequest(
            model_version=second.registered_version, strategy=DeploymentStrategy.SHADOW
        )
    )
    assert result.succeeded
    assert result.deployment.shadow_version == second.registered_version

    service = PredictionService(settings=platform["settings"], registry=registry)
    response = service.predict_one(sample_features, new_request_id())

    # The caller is served by the primary, never the shadow.
    assert response.model_version == first.registered_version
    assert response.variant == "primary"

    # But the shadow prediction was recorded.
    shadow_rows = [
        r
        for r in get_inference_log().recent(include_shadow=True, limit=50)
        if r["shadow"] == 1
    ]
    assert shadow_rows
    assert shadow_rows[0]["model_version"] == second.registered_version
