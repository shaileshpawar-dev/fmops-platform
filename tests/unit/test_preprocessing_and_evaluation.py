"""Feature engineering, evaluation metrics and the registry."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.core.exceptions import DependencyMissingError, ModelNotFoundError, TrainingError
from app.data.preprocessing import (
    DERIVED_FEATURES,
    build_feature_pipeline,
    engineer_features,
    prepare_inference_frame,
    split_features_target,
)
from app.schemas.common import ModelStage
from app.training.evaluate import (
    compute_metrics,
    evaluate_model,
    measure_inference_latency,
    metrics_to_flat_dict,
    select_threshold,
)
from app.training.model_factory import (
    available_algorithms,
    build_estimator,
    normalise_params,
)

pytestmark = pytest.mark.unit


# --------------------------------------------------------------------------- #
# Feature engineering
# --------------------------------------------------------------------------- #
def test_derived_features_are_added(valid_frame):
    engineered = engineer_features(valid_frame)
    for feature in DERIVED_FEATURES:
        assert feature in engineered.columns


def test_engineering_does_not_mutate_the_input(valid_frame):
    before = list(valid_frame.columns)
    engineer_features(valid_frame)
    assert list(valid_frame.columns) == before


def test_ratios_are_correct():
    frame = pd.DataFrame(
        {
            "annual_income": [100000.0],
            "loan_amount": [25000.0],
            "loan_term_months": [50],
            "credit_score": [700.0],
            "num_late_payments_12m": [0],
            "credit_utilization": [0.2],
            "num_credit_lines": [5],
            "employment_years": [10.0],
            "age": [40.0],
        }
    )
    out = engineer_features(frame)
    assert out["loan_to_income"].iloc[0] == pytest.approx(0.25)
    assert out["monthly_payment"].iloc[0] == pytest.approx(500.0)
    assert out["income_per_credit_line"].iloc[0] == pytest.approx(20000.0)
    assert out["tenure_ratio"].iloc[0] == pytest.approx(0.25)


def test_division_by_zero_becomes_nan_not_inf():
    """The imputer handles NaN; inf would poison the scaler."""
    frame = pd.DataFrame(
        {
            "annual_income": [0.0],
            "loan_amount": [10000.0],
            "loan_term_months": [0],
            "credit_score": [700.0],
            "num_late_payments_12m": [0],
            "credit_utilization": [0.2],
            "num_credit_lines": [0],
            "employment_years": [1.0],
            "age": [0.0],
        }
    )
    out = engineer_features(frame)
    for feature in DERIVED_FEATURES:
        values = out[feature].to_numpy(dtype=float)
        assert not np.isinf(values).any(), f"{feature} produced an infinity"


def test_credit_score_band_is_ordinal():
    frame = pd.DataFrame(
        {
            "credit_score": [500.0, 620.0, 700.0, 770.0, 830.0],
            "annual_income": [50000.0] * 5,
            "loan_amount": [10000.0] * 5,
            "loan_term_months": [36] * 5,
            "num_late_payments_12m": [0] * 5,
            "credit_utilization": [0.3] * 5,
            "num_credit_lines": [4] * 5,
            "employment_years": [5.0] * 5,
            "age": [40.0] * 5,
        }
    )
    bands = engineer_features(frame)["credit_score_band"].tolist()
    assert bands == [0.0, 1.0, 2.0, 3.0, 4.0]


def test_split_drops_identifier_and_timestamp(valid_frame, settings):
    features, target = split_features_target(valid_frame, settings.data)
    assert settings.data.id_column not in features.columns
    assert settings.data.timestamp_column not in features.columns
    assert settings.data.target_column not in features.columns
    assert set(target.unique()) <= {0, 1}


def test_split_requires_the_target(valid_frame, settings):
    with pytest.raises(KeyError):
        split_features_target(valid_frame.drop(columns=["default"]), settings.data)


def test_pipeline_output_is_numeric_and_finite(valid_frame, settings):
    features, target = split_features_target(valid_frame, settings.data)
    pipeline = build_feature_pipeline(settings.data)
    transformed = pipeline.fit_transform(features, target)
    assert transformed.shape[0] == len(features)
    assert np.isfinite(transformed).all()


def test_unknown_categories_do_not_break_inference(valid_frame, settings):
    """A category unseen at training time must encode to zeros, not raise."""
    features, target = split_features_target(valid_frame, settings.data)
    pipeline = build_feature_pipeline(settings.data)
    pipeline.fit(features, target)

    unseen = features.head(3).copy()
    unseen["employment_type"] = "brand_new_category"
    transformed = pipeline.transform(unseen)
    assert transformed.shape[0] == 3
    assert np.isfinite(transformed).all()


def test_prepare_inference_frame_fills_missing_columns(settings, sample_features):
    partial = dict(sample_features)
    del partial["credit_score"]
    frame = prepare_inference_frame([partial], settings.data)
    assert list(frame.columns) == settings.data.feature_columns
    assert frame["credit_score"].isna().all()


def test_prepare_inference_frame_drops_unexpected_columns(settings, sample_features):
    payload = {**sample_features, "not_a_feature": 1}
    frame = prepare_inference_frame([payload], settings.data)
    assert "not_a_feature" not in frame.columns


# --------------------------------------------------------------------------- #
# Model factory
# --------------------------------------------------------------------------- #
def test_supported_estimators_build(settings):
    for algorithm in ("hist_gradient_boosting", "random_forest", "logistic_regression"):
        assert build_estimator(algorithm, {}) is not None


def test_unknown_algorithm_raises_with_the_supported_list():
    with pytest.raises(TrainingError, match="supported"):
        build_estimator("neural_quantum_forest", {})


def test_generic_search_keys_map_onto_estimator_params():
    params = normalise_params("hist_gradient_boosting", {"n_estimators": 200})
    assert params == {"max_iter": 200}
    # learning_rate is meaningless for a random forest and must be dropped.
    assert "learning_rate" not in normalise_params("random_forest", {"learning_rate": 0.1})


def test_optional_backends_report_availability_honestly():
    status = available_algorithms()
    assert status["hist_gradient_boosting"] is True
    assert set(status) >= {"xgboost", "lightgbm"}


def test_missing_optional_backend_raises_a_clear_error():
    import importlib.util

    if importlib.util.find_spec("xgboost") is not None:
        pytest.skip("xgboost is installed in this environment")
    with pytest.raises(DependencyMissingError, match="boosting"):
        build_estimator("xgboost", {})


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
def test_metrics_on_a_perfect_classifier():
    y_true = np.array([0, 0, 1, 1])
    probabilities = np.array([0.01, 0.02, 0.98, 0.99])
    metrics, matrix = compute_metrics(y_true, probabilities, threshold=0.5)
    assert metrics.accuracy == 1.0
    assert metrics.f1 == 1.0
    assert metrics.roc_auc == 1.0
    assert matrix.false_positive == 0
    assert matrix.false_negative == 0


def test_metrics_on_a_useless_classifier():
    y_true = np.array([0, 1, 0, 1, 0, 1])
    probabilities = np.full(6, 0.5)
    metrics, _ = compute_metrics(y_true, probabilities, threshold=0.5)
    assert metrics.roc_auc == pytest.approx(0.5)


def test_single_class_split_does_not_crash():
    metrics, _ = compute_metrics(np.zeros(10, dtype=int), np.full(10, 0.3), 0.5)
    assert metrics.roc_auc == 0.0
    assert metrics.accuracy == 1.0


def test_threshold_selection_beats_the_default_on_imbalanced_data():
    rng = np.random.default_rng(0)
    y_true = (rng.uniform(size=2000) < 0.15).astype(int)
    probabilities = np.clip(y_true * 0.35 + rng.normal(0.25, 0.14, 2000), 0.001, 0.999)

    chosen = select_threshold(y_true, probabilities)
    from sklearn.metrics import f1_score

    f1_default = f1_score(y_true, (probabilities >= 0.5).astype(int), zero_division=0)
    f1_chosen = f1_score(y_true, (probabilities >= chosen).astype(int), zero_division=0)
    assert f1_chosen >= f1_default


def test_threshold_selection_handles_single_class():
    assert select_threshold(np.zeros(50, dtype=int), np.full(50, 0.4)) == 0.5


def test_evaluate_model_returns_a_full_result(trained_pipeline, settings):
    pipeline, x_test, y_test = trained_pipeline
    result = evaluate_model(pipeline, x_test, y_test, "m", threshold=0.5)

    assert 0.0 <= result.metrics.roc_auc <= 1.0
    assert result.metrics.n_samples == len(y_test)
    assert result.metrics.inference_latency_p95_ms > 0
    assert result.confusion_matrix.as_dict()
    assert result.calibration["predicted"]
    # HistGradientBoosting has no feature_importances_, so this exercises the
    # permutation fallback over raw columns.
    assert result.feature_importance
    assert set(result.feature_importance) <= set(x_test.columns)


def test_latency_measurement_is_per_row(trained_pipeline):
    pipeline, x_test, _ = trained_pipeline
    p50, p95 = measure_inference_latency(pipeline, x_test, n_samples=20)
    assert p50 > 0
    assert p95 >= p50


def test_flat_metrics_include_the_confusion_matrix(trained_pipeline):
    pipeline, x_test, y_test = trained_pipeline
    result = evaluate_model(
        pipeline, x_test, y_test, "m", measure_latency=False, compute_importance=False
    )
    flat = metrics_to_flat_dict(result)
    assert "roc_auc" in flat
    assert "cm_true_positive" in flat
    assert "threshold" in flat
    assert all(isinstance(v, float) for v in flat.values())


def test_trained_model_learns_the_signal(trained_pipeline):
    """Sanity floor: the dataset must be learnable or every gate test is vacuous."""
    pipeline, x_test, y_test = trained_pipeline
    result = evaluate_model(
        pipeline, x_test, y_test, "m", measure_latency=False, compute_importance=False
    )
    assert result.metrics.roc_auc > 0.75


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
def test_versions_increment_per_model(registry):
    first = registry.register("m", "file:///a")
    second = registry.register("m", "file:///b")
    other = registry.register("other", "file:///c")
    assert (first.version, second.version, other.version) == (1, 2, 1)


def test_registered_version_starts_in_development(registry):
    version = registry.register("m", "file:///a")
    assert version.stage == ModelStage.DEVELOPMENT
    assert version.status.value == "ready"


def test_reproducibility_block_is_persisted(registry):
    version = registry.register(
        "m",
        "file:///a",
        run_id="run-123",
        metrics={"roc_auc": 0.9},
        params={"max_iter": 100},
        dataset_version="v1-abc",
        dataset_hash="deadbeef",
        git_commit="abc123def456",
        algorithm="hist_gradient_boosting",
        tags={"owner": "ml-platform"},
    )
    fetched = registry.get("m", version.version)
    assert fetched.run_id == "run-123"
    assert fetched.dataset_version == "v1-abc"
    assert fetched.dataset_hash == "deadbeef"
    assert fetched.git_commit == "abc123def456"
    assert fetched.params["max_iter"] == 100
    assert fetched.metrics["roc_auc"] == 0.9
    assert fetched.tags["owner"] == "ml-platform"


def test_unknown_version_raises(registry):
    with pytest.raises(ModelNotFoundError):
        registry.get("m", 999)


def test_get_serving_prefers_production_over_staging(registry):
    staging = registry.register("m", "file:///a")
    registry.transition_stage("m", staging.version, ModelStage.VALIDATION)
    registry.transition_stage("m", staging.version, ModelStage.STAGING)
    assert registry.get_serving("m").version == staging.version

    production = registry.register("m", "file:///b")
    for stage in (ModelStage.VALIDATION, ModelStage.STAGING, ModelStage.PRODUCTION):
        registry.transition_stage("m", production.version, stage)
    assert registry.get_serving("m").version == production.version


def test_history_records_every_transition(registry):
    version = registry.register("m", "file:///a")
    registry.transition_stage(
        "m", version.version, ModelStage.VALIDATION, reason="gate passed"
    )
    history = registry.history("m", version.version)
    assert len(history) == 2  # initial registration + one transition
    assert history[0].reason == "gate passed"


def test_tags_merge_rather_than_replace(registry):
    version = registry.register("m", "file:///a", tags={"a": "1"})
    registry.set_tags("m", version.version, {"b": "2"})
    tags = registry.get("m", version.version).tags
    assert tags == {"a": "1", "b": "2"}


def test_list_models_summarises_stages(registry):
    version = registry.register("m", "file:///a")
    for stage in (ModelStage.VALIDATION, ModelStage.STAGING, ModelStage.PRODUCTION):
        registry.transition_stage("m", version.version, stage)
    summary = {m["name"]: m for m in registry.list_models()}["m"]
    assert summary["production_version"] == version.version
    assert summary["versions"] == 1
