"""Profiling, target recommendation, problem inference and candidate ranking.

These are the parts of AutoML that make judgements, so they are tested against
data shaped to provoke each judgement -- including the ones where the right
answer is "I cannot tell, ask the user".
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.automl.profiler import infer_problem_type, profile_frame
from app.automl.recommend import (
    default_selection,
    primary_metric_for,
    recommend_candidates,
    supported_problem_types,
)
from app.automl.runner import Candidate, rank_candidates

RNG = np.random.default_rng(0)


def _binary_frame(n: int = 400) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "customer_id": range(n),
            "age": RNG.integers(20, 70, n),
            "income": RNG.normal(50000, 12000, n).round(2),
            "segment": RNG.choice(["a", "b", "c"], n),
            "churn": RNG.choice([0, 1], n, p=[0.7, 0.3]),
        }
    )


# --------------------------------------------------------------------------- #
# Problem inference
# --------------------------------------------------------------------------- #
def test_two_distinct_values_is_binary_classification():
    problem, confidence, reasons = infer_problem_type(pd.Series([0, 1, 1, 0, 1] * 40))
    assert problem == "binary_classification"
    assert confidence == "high"
    assert reasons


def test_few_repeated_categories_is_multiclass():
    problem, _, _ = infer_problem_type(pd.Series(RNG.choice(list("abcd"), 300)))
    assert problem == "multiclass_classification"


def test_continuous_numeric_is_regression():
    problem, _, _ = infer_problem_type(pd.Series(RNG.normal(100, 15, 500)))
    assert problem == "regression"


def test_a_constant_column_is_not_a_problem_at_all():
    problem, confidence, _ = infer_problem_type(pd.Series([7] * 200))
    assert problem == "unknown"
    assert confidence == "low"


def test_an_empty_column_is_unknown():
    problem, _, _ = infer_problem_type(pd.Series([np.nan] * 50))
    assert problem == "unknown"


def test_high_cardinality_text_is_not_a_label():
    problem, _, _ = infer_problem_type(pd.Series([f"free text {i}" for i in range(300)]))
    assert problem == "unknown"


# --------------------------------------------------------------------------- #
# Target recommendation
# --------------------------------------------------------------------------- #
def test_a_named_binary_column_is_recommended_with_reasons():
    profile = profile_frame(_binary_frame())
    target = profile.suggested_target
    assert target is not None
    assert target.column == "churn"
    assert target.problem_type == "binary_classification"
    assert target.reasons, "a recommendation without a reason is not reviewable"
    assert set(target.class_balance) == {"0", "1"}


def test_a_dataset_with_no_plausible_target_recommends_nothing():
    """Silence beats a confident guess.

    Every column here is an identifier or free text. Returning None is what
    forces the caller to choose rather than letting training start on noise.
    """
    frame = pd.DataFrame({"row_id": range(200), "note": [f"n{i}" for i in range(200)]})
    assert profile_frame(frame).suggested_target is None


def test_an_override_is_honoured_even_when_the_heuristics_disagree():
    profile = profile_frame(_binary_frame(), target_override="income")
    target = profile.suggested_target
    assert target is not None
    assert target.column == "income"
    assert target.problem_type == "regression"
    assert any("manual" in r or "confirmed" in r for r in target.reasons)


def test_an_override_naming_a_missing_column_is_rejected():
    with pytest.raises(KeyError):
        profile_frame(_binary_frame(), target_override="not_a_column")


# --------------------------------------------------------------------------- #
# Column roles
# --------------------------------------------------------------------------- #
def test_identifier_columns_are_excluded_with_a_stated_reason():
    profile = profile_frame(_binary_frame())
    by_name = {c.name: c for c in profile.columns}
    assert by_name["customer_id"].role == "excluded"
    assert "identifier" in (by_name["customer_id"].exclusion_reason or "")


def test_continuous_measurements_are_never_mistaken_for_identifiers():
    """Income is near-unique. It is still the most useful feature in the frame.

    Treating near-uniqueness alone as an identifier signal silently deletes the
    best columns in most real datasets, so this pins the distinction.
    """
    profile = profile_frame(_binary_frame())
    by_name = {c.name: c for c in profile.columns}
    assert by_name["income"].likely_identifier is False
    assert by_name["income"].role == "feature"


def test_date_like_string_columns_are_excluded_rather_than_one_hot_encoded():
    frame = _binary_frame(200)
    frame["opened_at"] = pd.date_range("2024-01-01", periods=200).astype(str)
    profile = profile_frame(frame)
    by_name = {c.name: c for c in profile.columns}
    assert by_name["opened_at"].kind == "datetime"
    assert by_name["opened_at"].role == "excluded"


def test_text_columns_are_recognised_whatever_dtype_pandas_gives_them():
    """pandas 2 stores strings as ``object``; pandas 3 stores them as ``str``.

    Testing for ``object`` alone made every date column a high-cardinality
    one-hot, and every string key a feature, on the pandas the image ships.
    """
    frame = _binary_frame(200)
    frame["opened_at"] = pd.array(
        pd.date_range("2024-01-01", periods=200).astype(str), dtype="string"
    )
    frame["account_ref"] = pd.array([f"ACC-{i:05d}" for i in range(200)], dtype="string")
    by_name = {c.name: c for c in profile_frame(frame).columns}
    assert by_name["opened_at"].kind == "datetime"
    assert by_name["opened_at"].role == "excluded"
    assert by_name["account_ref"].likely_identifier is True
    assert by_name["account_ref"].role == "excluded"


def test_constant_columns_are_excluded_and_warned_about():
    frame = _binary_frame(200)
    frame["always_one"] = 1
    profile = profile_frame(frame)
    by_name = {c.name: c for c in profile.columns}
    assert by_name["always_one"].role == "excluded"
    assert any(w["type"] == "constant_feature" for w in profile.warnings)


# --------------------------------------------------------------------------- #
# Warnings
# --------------------------------------------------------------------------- #
def test_class_imbalance_is_reported():
    frame = pd.DataFrame({"f": RNG.normal(size=1000), "fraud": [1] * 30 + [0] * 970})
    profile = profile_frame(frame, target_override="fraud")
    assert any(w["type"] == "class_imbalance" for w in profile.warnings)


def test_leakage_is_phrased_as_a_risk_never_as_a_finding():
    """A heuristic cannot prove leakage, so it must not claim to."""
    frame = _binary_frame(200)
    frame["final_outcome_amount"] = RNG.normal(size=200)
    profile = profile_frame(frame, target_override="churn")
    leak = [w for w in profile.warnings if w["type"] == "potential_leakage"]
    assert leak
    assert all("potential" in w["type"] for w in leak)
    assert all("confirm" in w["detail"] or "suggests" in w["detail"] for w in leak)


def test_a_dataset_too_small_to_train_is_flagged_as_an_error():
    frame = pd.DataFrame({"f": [1.0, 2.0, 3.0], "y": [0, 1, 0]})
    profile = profile_frame(frame, target_override="y")
    rows = [w for w in profile.warnings if w["type"] == "insufficient_rows"]
    assert rows and rows[0]["severity"] == "error"


# --------------------------------------------------------------------------- #
# Candidate recommendation
# --------------------------------------------------------------------------- #
def test_candidates_are_recommended_for_binary_classification_with_reasons():
    profile = profile_frame(_binary_frame(1000))
    recs = recommend_candidates(profile, "binary_classification")
    available = [r for r in recs if r.available]
    assert available
    assert all(r.reasons for r in available)
    assert any(r.tier == "recommended" for r in available)
    assert any(r.tier == "baseline" for r in available)


def test_unsupported_problem_types_offer_no_usable_candidate():
    """The stack fits binary classification. It must say so rather than try."""
    profile = profile_frame(_binary_frame(500), target_override="income")
    for problem in ("regression", "multiclass_classification"):
        recs = recommend_candidates(profile, problem)
        assert recs
        assert all(r.tier == "unsuitable" and not r.available for r in recs)
    assert supported_problem_types() == ["binary_classification"]


def test_only_installed_algorithms_are_offered():
    from app.training.model_factory import available_algorithms

    installed = {k for k, v in available_algorithms().items() if v}
    recs = recommend_candidates(profile_frame(_binary_frame(1000)), "binary_classification")
    assert {r.algorithm for r in recs if r.available} <= installed


def test_default_selection_is_capped():
    recs = recommend_candidates(profile_frame(_binary_frame(1000)), "binary_classification")
    assert len(default_selection(recs, max_models=2)) <= 2


def test_imbalanced_targets_surface_recall_alongside_roc_auc():
    frame = pd.DataFrame({"f": RNG.normal(size=1000), "fraud": [1] * 40 + [0] * 960})
    profile = profile_frame(frame, target_override="fraud")
    metric, secondary, note = primary_metric_for("binary_classification", profile)
    assert metric == "roc_auc"
    assert "recall" in secondary
    assert note


# --------------------------------------------------------------------------- #
# Ranking
# --------------------------------------------------------------------------- #
def test_ranking_orders_by_primary_metric_and_excludes_failures():
    candidates = [
        Candidate(
            "a", status="completed", metrics={"roc_auc": 0.80, "f1": 0.5}, duration_seconds=1
        ),
        Candidate(
            "b", status="completed", metrics={"roc_auc": 0.91, "f1": 0.6}, duration_seconds=9
        ),
        Candidate("c", status="failed", error="boom"),
    ]
    ranked = rank_candidates(candidates, "roc_auc")
    assert [c.algorithm for c in ranked] == ["b", "a"]
    assert ranked[0].rank == 1
    assert all(c.status == "completed" for c in ranked)


def test_a_failed_candidate_can_never_win():
    candidates = [Candidate("only", status="failed", error="boom", metrics={"roc_auc": 0.99})]
    assert rank_candidates(candidates, "roc_auc") == []


def test_ties_break_deterministically():
    """Two runs over the same results must produce the same leaderboard."""

    def build():
        return [
            Candidate(
                "zeta",
                status="completed",
                metrics={"roc_auc": 0.9, "f1": 0.5},
                duration_seconds=5,
            ),
            Candidate(
                "alpha",
                status="completed",
                metrics={"roc_auc": 0.9, "f1": 0.5},
                duration_seconds=5,
            ),
        ]

    first = [c.algorithm for c in rank_candidates(build(), "roc_auc")]
    second = [c.algorithm for c in rank_candidates(build(), "roc_auc")]
    assert first == second == ["alpha", "zeta"]


def test_faster_candidate_wins_a_metric_tie():
    candidates = [
        Candidate(
            "slow",
            status="completed",
            metrics={"roc_auc": 0.9, "f1": 0.6},
            duration_seconds=30,
        ),
        Candidate(
            "fast", status="completed", metrics={"roc_auc": 0.9, "f1": 0.6}, duration_seconds=2
        ),
    ]
    assert rank_candidates(candidates, "roc_auc")[0].algorithm == "fast"


# --------------------------------------------------------------------------- #
# Per-run settings
# --------------------------------------------------------------------------- #
def test_run_settings_never_list_a_column_the_splitter_drops(settings):
    """The feature lists and the drop set must agree.

    They are produced by different functions, and when they disagree the
    ColumnTransformer asks for a column that is no longer in the frame and
    every candidate dies at fit time.
    """
    from app.data.contract import contract_for_target

    frame = _binary_frame(300)
    frame["opened_at"] = pd.date_range("2024-01-01", periods=300).astype(str)
    profile = profile_frame(frame, target_override="churn")

    run_settings = contract_for_target(frame, "churn", profile=profile)
    data = run_settings.data
    listed = set(data.numeric_features) | set(data.categorical_features)
    dropped = {data.target_column, data.id_column, data.timestamp_column}

    assert not (listed & dropped)
    assert data.target_column == "churn"


def test_run_settings_do_not_mutate_the_global_configuration(settings):
    from app.data.contract import contract_for_target

    before = settings.data.target_column
    before_features = list(settings.data.feature_columns)
    contract_for_target(_binary_frame(200), "churn", settings=settings)
    assert settings.data.target_column == before
    assert settings.data.feature_columns == before_features
