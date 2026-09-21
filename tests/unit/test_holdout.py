"""Champion vs challenger is decided on rows neither model trained on."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _frame(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    y = (rng.random(n) < 1 / (1 + np.exp(-(1.5 * x1 - x2)))).astype(int)
    return pd.DataFrame(
        {"row_id": [f"{seed}-{i}" for i in range(n)], "x1": x1, "x2": x2, "y": y}
    )


@pytest.fixture
def two_versions(settings, registry, tmp_path, monkeypatch):
    """An incumbent trained on A and a candidate trained on A + B, registered."""
    import app.data.versioning as versioning
    from app.data.contract import contract_for_target
    from app.schemas.model import TrainingRequest
    from app.training.train import train_model

    datasets = versioning.DatasetRegistry(settings)
    datasets.manifest_path = tmp_path / "versions.json"
    monkeypatch.setattr(versioning, "_REGISTRY", datasets)

    a = _frame(800, 1)
    ab = pd.concat([a, _frame(800, 2)], ignore_index=True)
    versions = []
    for frame, stem in ((a, "a"), (ab, "ab")):
        record = datasets.register_frame(frame, dataset_name="toy", filename=f"{stem}.csv")
        run_settings = contract_for_target(frame, "y", settings=settings)
        result = train_model(
            TrainingRequest(
                model_name="toy_model",
                dataset_version=record.version,
                algorithm="logistic_regression",
                tune=False,
            ),
            settings=run_settings,
            registry=registry,
        )
        versions.append(registry.get("toy_model", result.registered_version))
    return versions


def test_the_shared_holdout_excludes_every_row_the_incumbent_trained_on(
    settings, two_versions
):
    from app.training.holdout import _row_keys, shared_holdout, split_for

    incumbent, candidate = two_versions
    x, y = shared_holdout(candidate, incumbent, settings)
    trained_on = set(_row_keys(split_for(incumbent, settings)[0], list(x.columns)))
    assert len(y) > 0
    assert not set(_row_keys(x, list(x.columns))) & trained_on
    # ...and it is a subset of the candidate's own test split.
    candidate_test = split_for(candidate, settings)[1]
    assert set(x.index) <= set(candidate_test.index)


def test_the_comparison_scores_both_models_on_the_same_rows(settings, two_versions):
    from app.training.holdout import compare_on_shared_holdout

    incumbent, candidate = two_versions
    comparison = compare_on_shared_holdout(candidate, incumbent, settings)
    assert comparison.basis == "shared_holdout"
    assert comparison.holdout_rows and comparison.holdout_rows >= 50
    # The scores are fresh measurements, not the recorded metrics.
    assert comparison.baseline_score != pytest.approx(incumbent.metrics["roc_auc"], abs=1e-12)


def test_a_forged_recorded_metric_cannot_decide_the_comparison(
    settings, registry, two_versions
):
    """Editing the incumbent's recorded score used to flip the verdict."""
    from app.training.holdout import compare_on_shared_holdout

    incumbent, candidate = two_versions
    honest = compare_on_shared_holdout(candidate, incumbent, settings)
    registry.db.execute(
        "UPDATE model_versions SET metrics = ? WHERE name = ? AND version = ?",
        ('{"roc_auc": 0.01, "f1": 0.01}', incumbent.name, incumbent.version),
    )
    forged = compare_on_shared_holdout(
        candidate, registry.get(incumbent.name, incumbent.version), settings
    )
    assert forged.baseline_score == pytest.approx(honest.baseline_score)
    assert forged.candidate_is_better == honest.candidate_is_better


def test_too_few_shared_rows_falls_back_and_says_so(settings, two_versions, monkeypatch):
    import app.training.holdout as holdout

    incumbent, candidate = two_versions
    monkeypatch.setattr(holdout, "MIN_SHARED_ROWS", 10**6)
    comparison = holdout.compare_on_shared_holdout(candidate, incumbent, settings)
    assert comparison.basis == "recorded_metrics"
    assert "recorded test metric" in comparison.reason


def test_no_incumbent_is_its_own_basis(settings, two_versions):
    from app.training.holdout import compare_on_shared_holdout

    _, candidate = two_versions
    comparison = compare_on_shared_holdout(candidate, None, settings)
    assert comparison.basis == "no_incumbent"
    assert comparison.candidate_is_better
