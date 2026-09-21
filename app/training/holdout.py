"""Champion vs challenger on the same rows.

A comparison of two models is only meaningful when both are scored on the same
data. The recorded metric of each version comes from *its own* test split, so
comparing recorded metrics compares two models on two different samples -- and
a retrained candidate, whose split includes new production rows, can lose to
the incumbent purely because its exam was harder.

This module scores both on one shared holdout:

1. Rebuild the candidate's test split exactly (the split is deterministic in
   the dataset version, the data contract and the configured seed).
2. Drop every row the incumbent was *trained* on -- found by rebuilding the
   incumbent's own split and matching rows by content -- because an incumbent
   scored on its training rows looks better than it is.
3. Score both models on what remains.

When too few rows survive (or only one class), there is no fair shared exam;
the comparison falls back to the recorded metrics and says so.
"""

from __future__ import annotations

import hashlib
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

from app.core.config import Settings, get_settings
from app.core.logging import get_logger
from app.schemas.model import ModelVersion

logger = get_logger(__name__)

MIN_SHARED_ROWS = 50


def split_for(version: ModelVersion, settings: Settings) -> tuple[pd.DataFrame, ...] | None:
    """(X_train, X_test, y_train, y_test) exactly as the version was trained.

    Mirrors the split in :func:`app.training.train._train_within_run`: the fit
    set is ``1 - test_size`` of the data; the validation split is carved from it,
    so the training rows are the fit set minus nothing the test set contains.
    """
    from app.data.preprocessing import split_features_target
    from app.data.versioning import get_dataset_registry
    from app.registry.context import data_config_for

    if not version.dataset_version:
        return None
    cfg = data_config_for(version, settings)
    frame = get_dataset_registry().load(version.dataset_version)
    features, target = split_features_target(frame, cfg)
    x_fit, x_test, y_fit, y_test = train_test_split(
        features,
        target,
        test_size=cfg.test_size,
        random_state=cfg.random_seed,
        stratify=target,
    )
    return x_fit, x_test, y_fit, y_test


def _row_keys(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    """A content key per row over the given columns, independent of index and order."""
    subset = frame.reindex(columns=sorted(columns)).astype(str)
    return subset.apply(
        lambda row: hashlib.sha256("\x1f".join(row.values).encode()).hexdigest(), axis=1
    )


def shared_holdout(
    candidate: ModelVersion, incumbent: ModelVersion, settings: Settings
) -> tuple[pd.DataFrame, pd.Series] | None:
    """The candidate's test rows that the incumbent never trained on."""
    candidate_split = split_for(candidate, settings)
    incumbent_split = split_for(incumbent, settings)
    if candidate_split is None or incumbent_split is None:
        return None
    _, x_test, _, y_test = candidate_split
    x_inc_train = incumbent_split[0]
    columns = sorted(set(x_test.columns) & set(x_inc_train.columns))
    seen = set(_row_keys(x_inc_train, columns))
    fresh = ~_row_keys(x_test, columns).isin(seen)
    return x_test[fresh.values], y_test[fresh.values]


def score(version: ModelVersion, x: pd.DataFrame, y: pd.Series) -> dict[str, float]:
    """Metrics of one registered version on the given rows, at its own threshold."""
    from app.data.preprocessing import prepare_inference_frame
    from app.deployment.model_cache import get_model_cache
    from app.training.evaluate import evaluate_model

    loaded = get_model_cache().get(version.name, version.version)
    frame = prepare_inference_frame(x, loaded.data_config)
    result = evaluate_model(
        loaded.pipeline,
        frame,
        y,
        version.name,
        split="shared_holdout",
        threshold=loaded.threshold,
        measure_latency=False,
        compute_importance=False,
    )
    return result.metrics.as_dict()


def compare_on_shared_holdout(
    candidate: ModelVersion,
    incumbent: ModelVersion | None,
    settings: Settings | None = None,
) -> Any:
    """Champion vs challenger, on the fairest data available -- and which it was."""
    from app.training.approval import compare_to_production

    settings = settings or get_settings()
    if incumbent is None or incumbent.version == candidate.version:
        comparison = compare_to_production(
            candidate.metrics, None, candidate_version=candidate.version, settings=settings
        )
        return comparison.model_copy(update={"basis": "no_incumbent"})

    fallback_reason = ""
    try:
        holdout = shared_holdout(candidate, incumbent, settings)
        if holdout is not None:
            x, y = holdout
            if len(y) >= MIN_SHARED_ROWS and y.nunique() == 2:
                comparison = compare_to_production(
                    score(candidate, x, y),
                    score(incumbent, x, y),
                    candidate_version=candidate.version,
                    baseline_version=incumbent.version,
                    settings=settings,
                )
                return comparison.model_copy(
                    update={
                        "basis": "shared_holdout",
                        "holdout_rows": len(y),
                        "reason": (
                            f"{comparison.reason} -- both scored on the same {len(y)} held-out "
                            "rows, none of which either model was trained on"
                        ),
                    }
                )
            fallback_reason = f"only {len(y)} held-out rows neither model trained on" + (
                "" if y.nunique() == 2 else ", with a single class"
            )
        else:
            fallback_reason = "a training dataset is no longer available"
    except Exception as exc:  # the comparison must still be made, on what exists
        logger.warning(
            "approval.shared_holdout_failed",
            extra={"candidate": candidate.key, "incumbent": incumbent.key, "error": str(exc)},
        )
        fallback_reason = f"the shared holdout could not be built ({exc})"

    comparison = compare_to_production(
        candidate.metrics,
        incumbent.metrics,
        candidate_version=candidate.version,
        baseline_version=incumbent.version,
        settings=settings,
    )
    return comparison.model_copy(
        update={
            "basis": "recorded_metrics",
            "reason": (
                f"{comparison.reason} -- compared on each version's own recorded test "
                f"metric, because {fallback_reason}"
            ),
        }
    )
