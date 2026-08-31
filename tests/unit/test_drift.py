"""Drift detection tests.

The most important assertions here are the *negative* ones: that identical
distributions do not register as drift, and that concept drift is reported as
unavailable when no labels exist rather than being silently approximated.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.core.exceptions import InsufficientDataError
from app.data.generator import GenerationSpec, build_production_dataset, generate_dataset
from app.monitoring.drift import (
    DriftDetector,
    NativeDriftEngine,
    categorical_js_distance,
    categorical_psi,
    jensen_shannon_distance,
    numeric_js_distance,
    population_stability_index,
)

pytestmark = pytest.mark.unit


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #
def test_psi_is_zero_for_identical_samples():
    rng = np.random.default_rng(0)
    sample = rng.normal(0, 1, 5000)
    assert population_stability_index(sample, sample) == pytest.approx(0.0, abs=1e-9)


def test_psi_grows_with_the_size_of_the_shift():
    rng = np.random.default_rng(0)
    reference = rng.normal(0, 1, 5000)
    small = rng.normal(0.2, 1, 5000)
    large = rng.normal(2.0, 1, 5000)

    psi_small = population_stability_index(reference, small)
    psi_large = population_stability_index(reference, large)
    assert psi_small < psi_large
    assert psi_large > 0.25, "a 2-sigma shift must exceed the 'significant' band"


def test_js_distance_is_bounded_and_symmetric():
    p = np.array([0.7, 0.2, 0.1])
    q = np.array([0.1, 0.3, 0.6])
    forward = jensen_shannon_distance(p, q)
    backward = jensen_shannon_distance(q, p)

    assert forward == pytest.approx(backward)
    assert 0.0 <= forward <= 1.0
    assert jensen_shannon_distance(p, p) == pytest.approx(0.0, abs=1e-9)


def test_js_distance_of_disjoint_distributions_approaches_one():
    p = np.array([1.0, 0.0])
    q = np.array([0.0, 1.0])
    assert jensen_shannon_distance(p, q) == pytest.approx(1.0, abs=1e-3)


def test_numeric_js_distance_detects_a_shift():
    rng = np.random.default_rng(1)
    reference = rng.normal(50, 10, 4000)
    same = rng.normal(50, 10, 4000)
    shifted = rng.normal(80, 10, 4000)

    assert numeric_js_distance(reference, same) < 0.1
    assert numeric_js_distance(reference, shifted) > 0.4


def test_categorical_psi_flags_a_changed_mix():
    reference = pd.Series(["a"] * 700 + ["b"] * 200 + ["c"] * 100)
    same = pd.Series(["a"] * 700 + ["b"] * 200 + ["c"] * 100)
    shifted = pd.Series(["a"] * 200 + ["b"] * 200 + ["c"] * 600)

    assert categorical_psi(reference, same) == pytest.approx(0.0, abs=1e-6)
    assert categorical_psi(reference, shifted) > 0.25
    assert categorical_js_distance(reference, shifted) > 0.2


def test_new_category_produces_a_large_distance():
    reference = pd.Series(["a"] * 500 + ["b"] * 500)
    with_new = pd.Series(["a"] * 400 + ["b"] * 400 + ["zzz_new"] * 200)
    assert categorical_js_distance(reference, with_new) > 0.1


def test_statistics_survive_empty_input():
    assert population_stability_index(np.array([]), np.array([1.0])) == 0.0
    assert numeric_js_distance(np.array([]), np.array([1.0])) == 0.0


# --------------------------------------------------------------------------- #
# Engine
# --------------------------------------------------------------------------- #
def test_identical_windows_show_no_drift(settings, valid_frame):
    engine = NativeDriftEngine()
    results = engine.analyse(
        valid_frame,
        valid_frame.copy(),
        settings.data.numeric_features,
        settings.data.categorical_features,
        settings.drift,
    )
    assert results, "the engine must actually evaluate features"
    assert not any(r.drifted for r in results)
    assert max(r.score for r in results) < 0.05


def test_severe_drift_is_detected_across_many_features(settings, valid_frame, drifted_frame):
    engine = NativeDriftEngine()
    results = engine.analyse(
        valid_frame,
        drifted_frame,
        settings.data.numeric_features,
        settings.data.categorical_features,
        settings.drift,
    )
    drifted = [r.feature for r in results if r.drifted]
    assert len(drifted) >= 4, f"expected several drifted features, got {drifted}"
    # The generator shifts income and age directly, so those must be caught.
    assert "annual_income" in drifted
    assert "age" in drifted


def test_engine_reports_both_the_statistic_and_a_normalised_score(
    settings, valid_frame, drifted_frame
):
    engine = NativeDriftEngine()
    results = engine.analyse(
        valid_frame,
        drifted_frame,
        settings.data.numeric_features,
        settings.data.categorical_features,
        settings.drift,
    )
    for result in results:
        assert 0.0 <= result.score <= 1.0, "score must be normalised for comparability"
        assert result.statistic >= 0.0
        assert result.test in ("psi+ks", "psi+chi2")


# --------------------------------------------------------------------------- #
# Detector
# --------------------------------------------------------------------------- #
def test_detector_flags_drift_and_persists(
    clean_singletons, settings, valid_frame, drifted_frame
):
    detector = DriftDetector(settings)
    report = detector.detect(
        reference=valid_frame,
        current=drifted_frame,
        model_name="loan_default_classifier",
        model_version=1,
    )
    assert report.drift_detected
    assert report.drifted_features
    assert report.n_reference == len(valid_frame)
    assert report.n_current == len(drifted_frame)

    from app.monitoring.drift import recent_drift_reports

    stored = recent_drift_reports("loan_default_classifier", limit=5)
    assert stored and stored[0]["id"] == report.id


def test_detector_reports_stable_for_undrifted_traffic(
    clean_singletons, settings, valid_frame
):
    fresh = generate_dataset(GenerationSpec(n_rows=800, seed=99))
    report = DriftDetector(settings).detect(
        reference=valid_frame,
        current=fresh,
        model_name="loan_default_classifier",
        persist=False,
    )
    assert not report.drift_detected
    assert report.dataset_drift_score < settings.drift.threshold


def test_concept_drift_is_unavailable_without_labels(
    clean_singletons, settings, valid_frame, drifted_frame
):
    """The central honesty guarantee: no labels, no concept-drift number."""
    report = DriftDetector(settings).detect(
        reference=valid_frame,
        current=drifted_frame,
        model_name="loan_default_classifier",
        labelled=None,
        baseline_metric=0.88,
        persist=False,
    )
    assert report.concept_drift_status == "unavailable"
    assert report.concept_drift_score is None
    assert "cannot be computed from" in report.concept_drift_detail
    assert "proxy" in report.concept_drift_detail.lower()


def test_concept_drift_is_measured_when_labels_exist(clean_singletons, settings, valid_frame):
    rng = np.random.default_rng(5)
    n = 300
    labels = rng.integers(0, 2, n)
    # Probabilities that are almost uncorrelated with truth => AUC near 0.5,
    # i.e. a large degradation against a 0.88 baseline.
    labelled = pd.DataFrame({"actual_label": labels, "probability": rng.uniform(0, 1, n)})
    report = DriftDetector(settings).detect(
        reference=valid_frame,
        current=valid_frame.head(400),
        model_name="loan_default_classifier",
        labelled=labelled,
        baseline_metric=0.88,
        persist=False,
    )
    assert report.concept_drift_status == "measured"
    assert report.concept_drift_score is not None
    assert report.concept_drift_score > 0.2
    assert "live ROC-AUC" in report.concept_drift_detail


def test_concept_drift_needs_enough_labels(clean_singletons, settings, valid_frame):
    labelled = pd.DataFrame({"actual_label": [0, 1, 0], "probability": [0.1, 0.9, 0.2]})
    report = DriftDetector(settings).detect(
        reference=valid_frame,
        current=valid_frame.head(400),
        model_name="loan_default_classifier",
        labelled=labelled,
        baseline_metric=0.88,
        persist=False,
    )
    assert report.concept_drift_status == "insufficient_labels"
    assert report.concept_drift_score is None


def test_concept_drift_mode_leaves_inputs_unchanged(settings, valid_frame):
    """Concept drift must be invisible to input-distribution monitoring.

    This is the property that justifies reporting concept drift separately: the
    feature distributions are identical, so a data-drift scan sees nothing.
    """
    concept = build_production_dataset(n_rows=1500, drift="concept", seed=44)
    baseline = build_production_dataset(n_rows=1500, drift="none", seed=44)

    engine = NativeDriftEngine()
    results = engine.analyse(
        baseline,
        concept,
        settings.data.numeric_features,
        settings.data.categorical_features,
        settings.drift,
    )
    assert not any(r.drifted for r in results), (
        "concept drift changed an input distribution, which defeats the point "
        "of the fixture"
    )


def test_insufficient_current_data_raises(clean_singletons, settings, valid_frame):
    with pytest.raises(InsufficientDataError):
        DriftDetector(settings).detect(
            reference=valid_frame,
            current=valid_frame.head(5),
            model_name="loan_default_classifier",
            persist=False,
        )


def test_empty_reference_raises(clean_singletons, settings, drifted_frame):
    with pytest.raises(InsufficientDataError):
        DriftDetector(settings).detect(
            reference=pd.DataFrame(),
            current=drifted_frame,
            model_name="loan_default_classifier",
            persist=False,
        )


def test_prediction_drift_is_measured_when_probabilities_exist(
    clean_singletons, settings, valid_frame
):
    reference = valid_frame.copy()
    reference["_probability"] = np.random.default_rng(1).beta(2, 8, len(reference))
    current = valid_frame.head(600).copy()
    current["_probability"] = np.random.default_rng(2).beta(8, 2, len(current))

    report = DriftDetector(settings).detect(
        reference=reference,
        current=current,
        model_name="loan_default_classifier",
        persist=False,
    )
    assert report.prediction_drift_score is not None
    assert report.prediction_drift_score > 0.3
    assert report.prediction_drift_detected


def test_drift_report_render_text(clean_singletons, settings, valid_frame, drifted_frame):
    report = DriftDetector(settings).detect(
        reference=valid_frame,
        current=drifted_frame,
        model_name="loan_default_classifier",
        persist=False,
    )
    text = report.render_text()
    assert "DRIFT DETECTED" in text
    assert "concept drift" in text
