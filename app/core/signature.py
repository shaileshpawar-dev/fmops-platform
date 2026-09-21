"""Model signatures: the input contract a model version was trained against.

A registered model is only useful if the platform can answer, without loading
it, three questions: what does it expect as input, what does it predict, and
what did its training data look like. The signature records exactly that, once,
at training time:

* the task (binary classification -- the only task this platform trains),
* the target column and its two classes, in the dataset's own labels,
* every input feature with its kind, observed dtype, missingness and either its
  numeric range or its categories.

It is written into the artifact's sidecar (so the artifact is self-describing)
and onto the registry record (so the console and the serving layer can read it
without deserialising the model). Everything downstream -- the prediction
contract, the drift feature lists, the retraining data contract, the input form
in the console -- is derived from it rather than from global configuration.

Nothing here is inferred at serving time. A request is checked against what the
model actually saw, not against what a request "looks like".
"""

from __future__ import annotations

import math
import re
from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from app.core.config import DataConfig

# Lower-case, starts with a letter, 3-64 characters. Model names end up in
# endpoint names, file paths and URLs, so the alphabet is deliberately small.
MODEL_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_-]{2,63}$")

# Categories beyond this many are not listed individually. The signature then
# records the count only, and serving cannot tell "unseen" from "not listed" --
# so it does not claim to.
MAX_LISTED_CATEGORIES = 50

# Pairs where the "positive" class is unambiguous from the words themselves.
_AFFIRMATIVE = {"1", "true", "yes", "y", "t", "positive", "pos"}
_NEGATIVE = {"0", "false", "no", "n", "f", "negative", "neg"}

SignatureError = ValueError


class FeatureSpec(BaseModel):
    """One input column as the model saw it during training."""

    name: str
    kind: Literal["numeric", "categorical"]
    dtype: str
    missing_fraction: float = 0.0
    minimum: float | None = None
    maximum: float | None = None
    median: float | None = None
    categories: list[str] | None = None
    n_categories: int | None = None

    @property
    def categories_complete(self) -> bool:
        """True when every training category is listed, so 'unseen' is knowable."""
        return (
            self.categories is not None
            and self.n_categories is not None
            and self.n_categories <= MAX_LISTED_CATEGORIES
        )

    def default_value(self) -> Any:
        """A representative value: the training median, or the commonest category."""
        if self.kind == "numeric":
            return self.median
        return self.categories[0] if self.categories else None


class ModelSignature(BaseModel):
    """The contract a model version was trained and must be served against."""

    schema_version: int = 1
    task: Literal["binary_classification"] = "binary_classification"
    contract: Literal["loan_reference", "inferred"] = "inferred"
    dataset_name: str
    target: str
    # [negative, positive], exactly as the values appear in the target column.
    # This is what training encodes against, so it must never be renamed.
    class_labels: list[str] = Field(min_length=2, max_length=2)
    # Optional presentation names for the same two classes (the reference
    # dataset stores 0/1 but means "no_default"/"default").
    display_labels: list[str] | None = None
    positive_label_rule: str = ""
    id_column: str | None = None
    timestamp_column: str | None = None
    features: list[FeatureSpec]

    # -- views ---------------------------------------------------------------- #
    @property
    def numeric_features(self) -> list[str]:
        return [f.name for f in self.features if f.kind == "numeric"]

    @property
    def categorical_features(self) -> list[str]:
        return [f.name for f in self.features if f.kind == "categorical"]

    @property
    def feature_columns(self) -> list[str]:
        return [f.name for f in self.features]

    @property
    def labels(self) -> list[str]:
        """What to call the two classes when showing a prediction."""
        return list(self.display_labels or self.class_labels)

    @property
    def positive_label(self) -> str:
        return self.labels[1]

    def feature(self, name: str) -> FeatureSpec | None:
        return next((f for f in self.features if f.name == name), None)

    def label_for(self, prediction: int) -> str:
        return self.labels[1] if int(prediction) == 1 else self.labels[0]

    def example(self) -> dict[str, Any]:
        """A valid request built from training medians and commonest categories."""
        return {f.name: f.default_value() for f in self.features}

    def to_data_config(self, base: DataConfig) -> DataConfig:
        """The ``DataConfig`` the training pipeline used, rebuilt from the record.

        Serving, drift and retraining all speak ``DataConfig``; rebuilding it
        from the signature is what lets them work on a model the global
        configuration has never heard of.
        """
        numeric_target = self.class_labels == ["0", "1"]
        return base.model_copy(
            update={
                "contract": self.contract,
                "class_labels": None if numeric_target else list(self.class_labels),
                "dataset_name": self.dataset_name,
                "target_column": self.target,
                "id_column": self.id_column or base.id_column,
                "timestamp_column": self.timestamp_column or base.timestamp_column,
                "numeric_features": self.numeric_features,
                "categorical_features": self.categorical_features,
            },
            deep=True,
        )

    # -- serving contract ------------------------------------------------------ #
    def check_record(self, record: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
        """Validate one prediction request against what the model was trained on.

        Returns the cleaned record and a list of non-fatal observations. Raises
        :class:`SignatureError` with every problem at once, so a caller fixing a
        payload does not have to discover the errors one round trip at a time.

        Rules:

        * Unknown fields are rejected. A misspelled feature name that silently
          became "missing" is the classic invisible serving bug.
        * Every feature must be present as a key. ``null`` is allowed and is
          imputed by the pipeline exactly as missing training values were; the
          response says which ones were.
        * Numeric features must be finite numbers; categorical features strings
          (numbers are accepted and stringified, since CSV columns of codes are
          often numeric on disk).
        * A value outside the training range, or a category never seen in
          training, is *accepted* and reported. It is a drift signal, not an
          invalid request.
        """
        errors: list[str] = []
        notes: list[str] = []
        known = set(self.feature_columns)
        passthrough = {c for c in (self.id_column, self.timestamp_column) if c}

        unknown = sorted(set(record) - known - passthrough)
        if unknown:
            errors.append(f"unknown field(s): {', '.join(unknown)}")
        missing = [f for f in self.feature_columns if f not in record]
        if missing:
            errors.append(f"missing field(s): {', '.join(missing)}")

        clean: dict[str, Any] = {}
        imputed: list[str] = []
        for spec in self.features:
            if spec.name not in record:
                continue
            value = record[spec.name]
            if value is None:
                clean[spec.name] = None
                imputed.append(spec.name)
                continue
            if spec.kind == "numeric":
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    errors.append(
                        f"{spec.name}: expected a number, got {type(value).__name__}"
                    )
                    continue
                if not math.isfinite(float(value)):
                    errors.append(f"{spec.name}: must be a finite number")
                    continue
                clean[spec.name] = float(value)
                if (
                    spec.minimum is not None
                    and spec.maximum is not None
                    and not (spec.minimum <= float(value) <= spec.maximum)
                ):
                    notes.append(
                        f"{spec.name}={value:g} is outside the training range "
                        f"[{spec.minimum:g}, {spec.maximum:g}]"
                    )
            else:
                if isinstance(value, (bool, int, float)):
                    value = stringify_label(value)
                if not isinstance(value, str):
                    errors.append(
                        f"{spec.name}: expected a string, got {type(value).__name__}"
                    )
                    continue
                value = value.strip()
                clean[spec.name] = value
                if spec.categories_complete and value not in (spec.categories or []):
                    notes.append(f"{spec.name}={value!r} was not seen in training")

        if errors:
            raise SignatureError("; ".join(errors))
        if imputed:
            notes.append(f"imputed from training data: {', '.join(imputed)}")
        return clean, notes

    # -- construction ------------------------------------------------------------ #
    @classmethod
    def from_training_frame(
        cls, frame: pd.DataFrame, data_config: DataConfig
    ) -> ModelSignature:
        """Record the contract from the exact frame a model is about to fit on."""
        target = data_config.target_column
        if target not in frame.columns:
            raise SignatureError(f"target column {target!r} is not in the dataset")
        labels, rule = resolve_class_labels(frame[target], data_config.class_labels)

        features: list[FeatureSpec] = []
        for name in data_config.numeric_features:
            if name in frame.columns:
                features.append(_numeric_spec(name, frame[name]))
        for name in data_config.categorical_features:
            if name in frame.columns:
                features.append(_categorical_spec(name, frame[name]))
        if not features:
            raise SignatureError("no feature columns remain; nothing to train on")

        return cls(
            contract=data_config.contract,
            dataset_name=data_config.dataset_name,
            target=target,
            class_labels=labels,
            positive_label_rule=rule,
            id_column=data_config.id_column if data_config.id_column in frame else None,
            timestamp_column=(
                data_config.timestamp_column if data_config.timestamp_column in frame else None
            ),
            features=features,
        )


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def validate_model_name(name: str) -> str:
    """Return ``name`` if it is a legal model name, else raise with the rule."""
    if not isinstance(name, str) or not MODEL_NAME_PATTERN.match(name):
        raise SignatureError(
            "model names must be 3-64 characters of lower-case letters, digits, '_' "
            "or '-', starting with a letter"
        )
    return name


def resolve_class_labels(
    series: pd.Series, explicit: list[str] | None = None
) -> tuple[list[str], str]:
    """Decide ``[negative, positive]`` for a two-class target, and say how.

    An explicit choice always wins. Otherwise 0/1 and true/false style pairs
    are unambiguous; for anything else the rarer class is taken as positive,
    because the class a team trains a detector for is almost always the rare
    event -- and the rule is recorded so nobody has to guess later.
    """
    values = [stringify_label(v) for v in series.dropna().unique()]
    distinct = sorted(set(values))
    if len(distinct) != 2:
        raise SignatureError(
            f"target must have exactly two classes for binary classification; "
            f"found {len(distinct)}: {', '.join(distinct[:6])}"
        )

    if explicit:
        explicit = [str(v) for v in explicit]
        if sorted(explicit) != distinct:
            raise SignatureError(
                f"positive/negative labels {explicit} do not match the target's "
                f"classes {distinct}"
            )
        return explicit, "chosen explicitly"

    lowered = {v.lower(): v for v in distinct}
    for pos_word in _AFFIRMATIVE:
        if pos_word in lowered:
            other = next(v for v in distinct if v != lowered[pos_word])
            if other.lower() in _NEGATIVE:
                return [other, lowered[pos_word]], f"'{lowered[pos_word]}' reads as positive"

    counts = series.dropna().map(stringify_label).value_counts()
    positive = str(counts.idxmin())
    negative = next(v for v in distinct if v != positive)
    return [negative, positive], f"'{positive}' is the rarer class"


def encode_target(series: pd.Series, class_labels: list[str] | None) -> pd.Series:
    """Map a target column onto 0/1 using the recorded ``[negative, positive]``."""
    if not class_labels:
        return pd.to_numeric(series, errors="raise").astype(int)
    mapping = {class_labels[0]: 0, class_labels[1]: 1}
    encoded = series.map(lambda v: mapping.get(stringify_label(v)))
    if encoded.isna().any():
        unexpected = sorted({stringify_label(v) for v in series[encoded.isna()].unique()})
        raise SignatureError(
            f"target has values outside {class_labels}: {', '.join(unexpected[:6])}"
        )
    return encoded.astype(int)


def stringify_label(value: Any) -> str:
    """The one text form of a label or category, used at training and serving.

    '1.0' and 1 must be the same class; '01' and '1' must not be merged; and a
    numpy bool from a pandas column must read the same as a JSON ``true``.
    """
    if isinstance(value, (bool, np.bool_)):
        return "true" if bool(value) else "false"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _numeric_spec(name: str, series: pd.Series) -> FeatureSpec:
    numeric = pd.to_numeric(series, errors="coerce")
    clean = numeric.dropna()
    return FeatureSpec(
        name=name,
        kind="numeric",
        dtype=str(series.dtype),
        missing_fraction=round(float(numeric.isna().mean()), 6),
        minimum=float(clean.min()) if not clean.empty else None,
        maximum=float(clean.max()) if not clean.empty else None,
        median=float(clean.median()) if not clean.empty else None,
    )


def _categorical_spec(name: str, series: pd.Series) -> FeatureSpec:
    as_text = series.dropna().map(stringify_label).str.strip()
    counts = as_text.value_counts()
    return FeatureSpec(
        name=name,
        kind="categorical",
        dtype=str(series.dtype),
        missing_fraction=round(float(series.isna().mean()), 6),
        categories=[str(c) for c in counts.index[:MAX_LISTED_CATEGORIES]],
        n_categories=int(counts.size),
    )


REFERENCE_DISPLAY_LABELS = ["no_default", "default"]


def build_signature(frame: pd.DataFrame, data_config: DataConfig) -> ModelSignature:
    """The signature for a frame about to be trained on.

    The reference dataset stores its target as 0/1; it gets its declared class
    names for display only, leaving the encoding labels untouched.
    """
    signature = ModelSignature.from_training_frame(frame, data_config)
    if data_config.contract == "loan_reference" and signature.class_labels == ["0", "1"]:
        signature.display_labels = list(REFERENCE_DISPLAY_LABELS)
        signature.positive_label_rule = "declared by the reference use case"
    return signature
