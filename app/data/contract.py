"""From "this dataset, this target" to the data contract a model trains under.

Training runs and AutoML runs both start from a registered dataset and a
chosen target column, and both need the same answer: which columns are
features (and of which kind), which are identifiers or timestamps to drop, what
the target's two classes are, which one is positive, and whether the dataset is
the declared reference dataset or a user's. That answer lives here, once.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from app.core.config import Settings, get_settings
from app.core.exceptions import FMOpsError
from app.core.signature import (
    SignatureError,
    resolve_class_labels,
    stringify_label,
    validate_model_name,
)


class InvalidTrainingTargetError(FMOpsError):
    """The dataset and target cannot train a binary classifier."""

    code = "invalid_training_target"
    http_status = 422


def explicit_labels(series: pd.Series, positive: str | None) -> list[str] | None:
    """``[negative, positive]`` when the caller named the positive class."""
    if positive is None:
        return None
    classes = sorted({stringify_label(v) for v in series.dropna().unique()})
    if len(classes) != 2 or positive not in classes:
        raise InvalidTrainingTargetError(
            f"positive class {positive!r} is not one of the target's classes {classes}"
        )
    return [next(c for c in classes if c != positive), positive]


def contract_for_target(
    frame: pd.DataFrame,
    target: str,
    positive_label: str | None = None,
    settings: Settings | None = None,
    profile: Any | None = None,
) -> Settings:
    """A copy of the settings whose ``DataConfig`` describes this training problem.

    The global settings object is never mutated -- doing so would leak one
    run's target into every other request in the process.
    """
    from app.automl.profiler import profile_frame
    from app.data.validation import dataset_contract

    settings = settings or get_settings()
    if target not in frame.columns:
        raise InvalidTrainingTargetError(
            f"target column {target!r} is not in the dataset",
            available_columns=[str(c) for c in frame.columns][:50],
        )
    try:
        labels, _ = resolve_class_labels(
            frame[target], explicit_labels(frame[target], positive_label)
        )
    except SignatureError as exc:
        raise InvalidTrainingTargetError(str(exc), target=target) from exc

    profile = profile or profile_frame(frame, target_override=target)
    reference = (
        dataset_contract(frame, settings).contract == "loan_reference"
        and target == settings.data.target_column
    )
    base = settings.data
    by_name = {c.name: c for c in profile.columns}
    identifiers = [c.name for c in profile.columns if c.likely_identifier and c.name != target]
    datetimes = [c.name for c in profile.columns if c.kind == "datetime" and c.name != target]
    id_column = identifiers[0] if identifiers else base.id_column
    timestamp_column = datetimes[0] if datetimes else base.timestamp_column

    # The dropped columns and the feature lists come from one decision: a
    # dropped column still listed as a feature makes every fit fail.
    dropped = {target, id_column, timestamp_column}
    features = [
        c.name for c in profile.columns if c.role == "feature" and c.name not in dropped
    ]
    if not features:
        raise InvalidTrainingTargetError(
            "no usable feature columns remain after profiling", target=target
        )
    numeric = [f for f in features if by_name[f].kind == "numeric"]
    categorical = [f for f in features if by_name[f].kind in ("categorical", "boolean")]

    return settings.model_copy(
        update={
            "data": base.model_copy(
                update={
                    "contract": "loan_reference" if reference else "inferred",
                    "class_labels": None if labels == ["0", "1"] else labels,
                    "dataset_name": base.dataset_name if reference else "uploaded",
                    "target_column": target,
                    "numeric_features": numeric,
                    "categorical_features": categorical,
                    "id_column": id_column,
                    "timestamp_column": timestamp_column,
                }
            )
        },
        deep=True,
    )


def resolve_model_name(
    frame: pd.DataFrame,
    target: str,
    requested: str | None,
    positive_label: str | None = None,
    settings: Settings | None = None,
) -> str:
    """Validate the requested model name -- or derive one -- before any work starts.

    Fails fast on a malformed name, a target that is not binary, or a target
    that does not match the model's existing lineage, so none of those is
    discovered minutes into a background run.
    """
    from app.data.validation import dataset_contract
    from app.registry.context import check_lineage_compatible, suggest_model_name
    from app.registry.factory import get_registry

    settings = settings or get_settings()
    if target not in frame.columns:
        raise InvalidTrainingTargetError(f"target column {target!r} is not in the dataset")
    reference = dataset_contract(frame, settings).contract == "loan_reference"
    name = requested or suggest_model_name(target, reference, settings)
    try:
        validate_model_name(name)
        labels, _ = resolve_class_labels(
            frame[target], explicit_labels(frame[target], positive_label)
        )
    except SignatureError as exc:
        raise InvalidTrainingTargetError(str(exc), target=target) from exc
    check_lineage_compatible(get_registry(), name, target, labels)
    return name
