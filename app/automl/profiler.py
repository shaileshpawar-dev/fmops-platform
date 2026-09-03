"""Dataset profiling, target recommendation and problem-type inference.

Every judgement in this module is a **recommendation** derived from observable
properties of the data, and every one carries the reason that produced it. None
of it is certainty: a column called ``default`` holding two values is a strong
candidate target, not a fact about the user's intent. The caller confirms or
overrides, and the API refuses to train until a target is settled.

The heuristics are deliberately boring and deterministic -- no model picks the
model. That is what makes the result explainable and repeatable, and it is why
the reasons below are phrased as evidence rather than as conclusions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import pandas as pd
from pandas.api import types as ptypes

ProblemType = Literal[
    "binary_classification", "multiclass_classification", "regression", "unknown"
]
Confidence = Literal["high", "medium", "low"]

# Names that commonly denote an outcome. Presence is evidence, never proof: a
# column called "class" might be a product class, and a real target might be
# called something this list has never seen.
TARGET_NAME_HINTS = (
    "target",
    "label",
    "y",
    "outcome",
    "default",
    "churn",
    "fraud",
    "class",
    "is_fraud",
    "converted",
    "response",
    "status",
    "result",
    "survived",
)
# Names that suggest a row identifier rather than a feature.
ID_NAME_HINTS = ("id", "uuid", "guid", "key", "number", "no", "code", "ref", "index")
# Names that suggest information recorded *after* the outcome. A feature that
# could only be known once the answer is known is a leakage risk.
LEAKAGE_NAME_HINTS = (
    "outcome",
    "result",
    "repaid",
    "recovered",
    "settled",
    "chargeoff",
    "charge_off",
    "collection",
    "closed_at",
    "resolved",
    "final",
    "post_",
    "_after",
    "actual_",
)

MIN_ROWS_FOR_TRAINING = 50
HIGH_CARDINALITY_RATIO = 0.5
IMBALANCE_WARN_RATIO = 0.20


@dataclass
class ColumnProfile:
    name: str
    dtype: str
    kind: Literal["numeric", "categorical", "boolean", "datetime", "empty"]
    n_unique: int
    unique_ratio: float
    missing: int
    missing_pct: float
    is_constant: bool
    likely_identifier: bool
    role: Literal["feature", "target", "excluded"] = "feature"
    exclusion_reason: str | None = None
    example: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "dtype": self.dtype,
            "kind": self.kind,
            "n_unique": self.n_unique,
            "unique_ratio": round(self.unique_ratio, 4),
            "missing": self.missing,
            "missing_pct": round(self.missing_pct, 2),
            "is_constant": self.is_constant,
            "likely_identifier": self.likely_identifier,
            "role": self.role,
            "exclusion_reason": self.exclusion_reason,
            "example": self.example,
        }


@dataclass
class TargetSuggestion:
    column: str
    confidence: Confidence
    reasons: list[str]
    problem_type: ProblemType
    n_classes: int | None = None
    class_balance: dict[str, float] = field(default_factory=dict)
    score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "column": self.column,
            "confidence": self.confidence,
            "reasons": self.reasons,
            "problem_type": self.problem_type,
            "n_classes": self.n_classes,
            "class_balance": self.class_balance,
        }


@dataclass
class DatasetProfile:
    n_rows: int
    n_columns: int
    missing_cells: int
    duplicate_rows: int
    columns: list[ColumnProfile]
    suggested_target: TargetSuggestion | None
    alternative_targets: list[TargetSuggestion]
    warnings: list[dict[str, str]]

    @property
    def numeric_columns(self) -> list[str]:
        return [c.name for c in self.columns if c.kind == "numeric"]

    @property
    def categorical_columns(self) -> list[str]:
        return [c.name for c in self.columns if c.kind in ("categorical", "boolean")]

    @property
    def identifier_columns(self) -> list[str]:
        return [c.name for c in self.columns if c.likely_identifier]

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_rows": self.n_rows,
            "n_columns": self.n_columns,
            "missing_cells": self.missing_cells,
            "duplicate_rows": self.duplicate_rows,
            "n_numeric": len(self.numeric_columns),
            "n_categorical": len(self.categorical_columns),
            "identifier_columns": self.identifier_columns,
            "columns": [c.to_dict() for c in self.columns],
            "suggested_target": (
                self.suggested_target.to_dict() if self.suggested_target else None
            ),
            "alternative_targets": [t.to_dict() for t in self.alternative_targets],
            "warnings": self.warnings,
        }


# --------------------------------------------------------------------------- #
# Column-level profiling
# --------------------------------------------------------------------------- #
def _looks_like_dates(series: pd.Series, sample: int = 200) -> bool:
    """Do the string values parse as dates?

    A CSV date column arrives as plain object dtype. Left as a category it gets
    one-hot encoded into thousands of columns, which is both useless and slow,
    so it is worth spending one parse on a sample to find out.
    """
    values = series.dropna().astype(str).head(sample)
    if values.empty or len(values) < 5:
        return False
    # Digits and separators only; this should not fire on ordinary free text.
    if not values.str.match(r"^[\d\-/:\.\sTZ+]{6,}$").all():
        return False
    parsed = pd.to_datetime(values, errors="coerce", format="mixed")
    return bool(parsed.notna().mean() > 0.9)


def _classify_kind(series: pd.Series) -> str:
    if series.isna().all():
        return "empty"
    if ptypes.is_datetime64_any_dtype(series):
        return "datetime"
    if ptypes.is_bool_dtype(series):
        return "boolean"
    if ptypes.is_numeric_dtype(series):
        return "numeric"
    if series.dtype == object and _looks_like_dates(series):
        return "datetime"
    return "categorical"


def _looks_like_identifier(name: str, series: pd.Series, n_rows: int) -> bool:
    """Nearly-unique *and* shaped like a key, or named like one.

    Near-uniqueness alone is not enough, and treating it as enough is how a
    profiler quietly throws away the best features in the dataset: income,
    price and any other continuous measurement are naturally almost unique.
    A float column is therefore never an identifier on uniqueness alone --
    only integers (which is what keys are) and free-text strings qualify.
    """
    if n_rows == 0:
        return False
    unique_ratio = series.nunique(dropna=True) / n_rows
    lowered = name.lower()
    named_like_id = any(
        lowered == hint or lowered.endswith("_" + hint) or lowered.startswith(hint + "_")
        for hint in ID_NAME_HINTS
    )
    if named_like_id and unique_ratio > HIGH_CARDINALITY_RATIO:
        return True
    if unique_ratio <= 0.95:
        return False
    if ptypes.is_bool_dtype(series) or ptypes.is_float_dtype(series):
        return False
    # An almost-unique integer or string column is a key in all but name.
    return bool(ptypes.is_integer_dtype(series) or series.dtype == object)


def profile_columns(frame: pd.DataFrame) -> list[ColumnProfile]:
    n_rows = len(frame)
    profiles: list[ColumnProfile] = []
    for name in frame.columns:
        series = frame[name]
        n_unique = int(series.nunique(dropna=True))
        missing = int(series.isna().sum())
        example = None
        if series.notna().any():
            example = str(series.dropna().iloc[0])[:80]
        profiles.append(
            ColumnProfile(
                name=str(name),
                dtype=str(series.dtype),
                kind=_classify_kind(series),  # type: ignore[arg-type]
                n_unique=n_unique,
                unique_ratio=(n_unique / n_rows) if n_rows else 0.0,
                missing=missing,
                missing_pct=(missing / n_rows * 100) if n_rows else 0.0,
                is_constant=n_unique <= 1,
                likely_identifier=_looks_like_identifier(str(name), series, n_rows),
                example=example,
            )
        )
    return profiles


# --------------------------------------------------------------------------- #
# Problem type
# --------------------------------------------------------------------------- #
def infer_problem_type(series: pd.Series) -> tuple[ProblemType, Confidence, list[str]]:
    """Infer what kind of problem this target implies.

    Uses dtype, distinct count and the ratio of distinct values to rows
    together, because none of them decides it alone: an integer column with two
    values is a binary label, and an integer column with eight thousand values
    is a measurement.
    """
    clean = series.dropna()
    if clean.empty:
        return "unknown", "low", ["the column is entirely missing"]

    n_unique = int(clean.nunique())
    n_rows = len(clean)
    ratio = n_unique / n_rows if n_rows else 0.0
    reasons: list[str] = []

    if n_unique <= 1:
        return (
            "unknown",
            "low",
            [f"only {n_unique} distinct value; a model cannot learn from a constant"],
        )

    if n_unique == 2:
        reasons.append(
            f"exactly 2 distinct values ({', '.join(map(str, sorted(clean.unique())[:2]))})"
        )
        return "binary_classification", "high", reasons

    if ptypes.is_numeric_dtype(clean) and not ptypes.is_bool_dtype(clean):
        is_integral = (
            bool((clean == clean.astype("int64", errors="ignore")).all())
            if ptypes.is_integer_dtype(clean)
            else False
        )
        if is_integral and n_unique <= 20 and ratio < 0.05:
            reasons.append(f"{n_unique} distinct integer values over {n_rows} rows")
            return "multiclass_classification", "medium", reasons
        reasons.append(
            f"continuous numeric with {n_unique} distinct values ({ratio:.1%} of rows)"
        )
        return "regression", "high" if ratio > 0.2 else "medium", reasons

    if n_unique <= 50:
        reasons.append(f"categorical with {n_unique} distinct values")
        return "multiclass_classification", "high" if n_unique <= 20 else "medium", reasons

    reasons.append(f"{n_unique} distinct categorical values -- too many to be a label")
    return "unknown", "low", reasons


# --------------------------------------------------------------------------- #
# Target suggestion
# --------------------------------------------------------------------------- #
def _score_target(
    profile: ColumnProfile, series: pd.Series, position: int, n_cols: int
) -> TargetSuggestion | None:
    """Score one column as a possible target, or reject it outright."""
    if (
        profile.kind in ("datetime", "empty")
        or profile.is_constant
        or profile.likely_identifier
    ):
        return None

    problem, problem_conf, problem_reasons = infer_problem_type(series)
    if problem == "unknown":
        return None

    score = 0.0
    reasons: list[str] = []
    lowered = profile.name.lower()

    if lowered in TARGET_NAME_HINTS:
        score += 4.0
        reasons.append(f"column name '{profile.name}' is a common outcome name")
    elif any(h in lowered for h in TARGET_NAME_HINTS):
        score += 2.0
        reasons.append("column name contains an outcome-like term")

    if problem == "binary_classification":
        score += 3.0
        reasons.append("two distinct values, the shape of a binary label")
    elif problem == "multiclass_classification":
        score += 1.0

    # Targets are usually last. Weak evidence, so it only breaks ties.
    if position >= n_cols - 2:
        score += 0.5
        reasons.append("positioned at the end of the dataset, where labels usually sit")

    if profile.missing_pct > 0:
        score -= 1.0
        reasons.append(f"{profile.missing_pct:.1f}% missing, unusual for a label")

    reasons.extend(problem_reasons)

    clean = series.dropna()
    balance: dict[str, float] = {}
    n_classes = None
    if problem in ("binary_classification", "multiclass_classification"):
        counts = clean.value_counts(normalize=True)
        balance = {str(k): round(float(v), 4) for k, v in counts.head(10).items()}
        n_classes = int(clean.nunique())

    if score >= 5.0:
        confidence: Confidence = "high"
    elif score >= 2.5:
        confidence = "medium"
    else:
        confidence = "low"
    if problem_conf == "low":
        confidence = "low"

    return TargetSuggestion(
        column=profile.name,
        confidence=confidence,
        reasons=reasons,
        problem_type=problem,
        n_classes=n_classes,
        class_balance=balance,
        score=score,
    )


# --------------------------------------------------------------------------- #
# Warnings
# --------------------------------------------------------------------------- #
def _build_warnings(
    frame: pd.DataFrame, columns: list[ColumnProfile], target: TargetSuggestion | None
) -> list[dict[str, str]]:
    warnings: list[dict[str, str]] = []
    n_rows = len(frame)

    if n_rows < MIN_ROWS_FOR_TRAINING:
        warnings.append(
            {
                "type": "insufficient_rows",
                "severity": "error",
                "column": "",
                "detail": f"{n_rows} rows is too few to train and evaluate meaningfully",
            }
        )

    for col in columns:
        if col.is_constant and col.role != "target":
            warnings.append(
                {
                    "type": "constant_feature",
                    "severity": "warning",
                    "column": col.name,
                    "detail": "a single distinct value carries no signal",
                }
            )
        if (
            col.kind == "categorical"
            and col.unique_ratio > HIGH_CARDINALITY_RATIO
            and not col.likely_identifier
        ):
            warnings.append(
                {
                    "type": "high_cardinality",
                    "severity": "warning",
                    "column": col.name,
                    "detail": f"{col.n_unique} distinct values across {n_rows} rows",
                }
            )
        if col.missing_pct > 20:
            warnings.append(
                {
                    "type": "missing_values",
                    "severity": "warning",
                    "column": col.name,
                    "detail": f"{col.missing_pct:.1f}% missing",
                }
            )

    # Leakage is a *risk*, never a finding. Both signals below can fire on a
    # perfectly innocent column, so the wording stays hypothetical.
    if target is not None:
        for col in columns:
            if col.name == target.column:
                continue
            lowered = col.name.lower()
            if any(h in lowered for h in LEAKAGE_NAME_HINTS):
                warnings.append(
                    {
                        "type": "potential_leakage",
                        "severity": "warning",
                        "column": col.name,
                        "detail": "name suggests information recorded after the outcome; "
                        "confirm it is available at prediction time",
                    }
                )

        if target.class_balance:
            minority = min(target.class_balance.values()) if target.class_balance else 1.0
            if minority < IMBALANCE_WARN_RATIO:
                warnings.append(
                    {
                        "type": "class_imbalance",
                        "severity": "warning",
                        "column": target.column,
                        "detail": f"minority class is {minority:.1%} of rows; ROC-AUC alone will "
                        "flatter a model here, so weigh F1 and recall too",
                    }
                )
        if target.n_classes == 1:
            warnings.append(
                {
                    "type": "single_class",
                    "severity": "error",
                    "column": target.column,
                    "detail": "only one class present; there is nothing to separate",
                }
            )
    return warnings


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def profile_frame(frame: pd.DataFrame, target_override: str | None = None) -> DatasetProfile:
    """Profile a dataframe and recommend a target.

    ``target_override`` pins the target to a named column and re-derives the
    problem type from it, which is what the confirmation step sends back.
    """
    n_rows = len(frame)
    columns = profile_columns(frame)
    by_name = {c.name: c for c in columns}

    candidates: list[TargetSuggestion] = []
    for position, col in enumerate(columns):
        suggestion = _score_target(col, frame[col.name], position, len(columns))
        if suggestion is not None:
            candidates.append(suggestion)
    candidates.sort(key=lambda s: s.score, reverse=True)

    chosen: TargetSuggestion | None
    if target_override:
        if target_override not in by_name:
            raise KeyError(f"column {target_override!r} is not in the dataset")
        chosen = next((c for c in candidates if c.column == target_override), None)
        if chosen is None:
            # The user picked a column the heuristics rejected. Honour it and
            # describe it, rather than overriding the person who knows the data.
            problem, conf, reasons = infer_problem_type(frame[target_override])
            clean = frame[target_override].dropna()
            counts = (
                clean.value_counts(normalize=True)
                if problem != "regression"
                else pd.Series(dtype=float)
            )
            chosen = TargetSuggestion(
                column=target_override,
                confidence=conf,
                reasons=["selected manually", *reasons],
                problem_type=problem,
                n_classes=int(clean.nunique()) if problem != "regression" else None,
                class_balance={str(k): round(float(v), 4) for k, v in counts.head(10).items()},
            )
        else:
            chosen.reasons = ["confirmed by the user", *chosen.reasons]
    else:
        chosen = candidates[0] if candidates else None
        # Two comparably strong candidates is not confidence, it is ambiguity.
        if (
            chosen
            and len(candidates) > 1
            and (candidates[0].score - candidates[1].score) < 1.0
        ):
            chosen.confidence = "low" if chosen.confidence != "high" else "medium"
            chosen.reasons.append(
                f"'{candidates[1].column}' scores comparably, so this needs confirmation"
            )

    if chosen is not None and chosen.column in by_name:
        by_name[chosen.column].role = "target"
    for col in columns:
        if col.role == "target":
            continue
        if col.likely_identifier:
            col.role, col.exclusion_reason = (
                "excluded",
                "likely identifier (near-unique values)",
            )
        elif col.is_constant:
            col.role, col.exclusion_reason = "excluded", "constant value"
        elif col.kind in ("datetime", "empty"):
            col.role, col.exclusion_reason = "excluded", f"{col.kind} column"

    return DatasetProfile(
        n_rows=n_rows,
        n_columns=len(frame.columns),
        missing_cells=int(frame.isna().sum().sum()),
        duplicate_rows=int(frame.duplicated().sum()),
        columns=columns,
        suggested_target=chosen,
        alternative_targets=[c for c in candidates if not chosen or c.column != chosen.column][
            :4
        ],
        warnings=_build_warnings(frame, columns, chosen),
    )
