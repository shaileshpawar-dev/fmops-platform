"""Candidate model recommendation.

Rules over a dataset profile, not a model choosing models. Every
recommendation carries the observations that produced it, so a user can
disagree with the reasoning rather than with an opaque score.

Only algorithms the existing model factory can actually build are offered, and
only those whose optional backend is installed in this environment. There is no
"suitability score" invented here -- suitability is expressed as a tier
(recommended / baseline / not suitable) with reasons, because a number would
imply a precision these heuristics do not have.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from app.automl.profiler import DatasetProfile, ProblemType

Tier = Literal["recommended", "baseline", "optional", "unsuitable"]

# What each supported algorithm is good for, in plain terms. These are
# properties of the estimator, not of any particular dataset.
_ALGORITHM_NOTES: dict[str, dict[str, Any]] = {
    "hist_gradient_boosting": {
        "label": "HistGradientBoosting",
        "strengths": [
            "strong default for tabular data",
            "handles mixed feature types",
            "native missing-value handling",
        ],
        "prefers_rows": 500,
    },
    "random_forest": {
        "label": "Random Forest",
        "strengths": [
            "robust tabular baseline",
            "little tuning needed",
            "tolerant of irrelevant features",
        ],
        "prefers_rows": 200,
    },
    "xgboost": {
        "label": "XGBoost",
        "strengths": ["competitive gradient boosting", "responds well to tuning"],
        "prefers_rows": 1000,
    },
    "lightgbm": {
        "label": "LightGBM",
        "strengths": ["fast on wide or large tabular data", "efficient with categoricals"],
        "prefers_rows": 1000,
    },
    "logistic_regression": {
        "label": "Logistic Regression",
        "strengths": [
            "fast, interpretable reference point",
            "coefficients are directly inspectable",
        ],
        "prefers_rows": 0,
    },
}


@dataclass
class CandidateRecommendation:
    algorithm: str
    label: str
    tier: Tier
    reasons: list[str]
    available: bool
    unavailable_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "algorithm": self.algorithm,
            "label": self.label,
            "tier": self.tier,
            "reasons": self.reasons,
            "available": self.available,
            "unavailable_reason": self.unavailable_reason,
        }


def supported_problem_types() -> list[ProblemType]:
    """What the training stack can actually fit today.

    The model factory builds classifiers, ``split_features_target`` casts the
    label to int, and evaluation computes ROC-AUC from ``predict_proba[:, 1]``.
    Binary classification is therefore the whole of it. Regression and
    multiclass are refused with a reason rather than attempted and silently
    mangled.
    """
    return ["binary_classification"]


def recommend_candidates(
    profile: DatasetProfile, problem_type: ProblemType, max_models: int = 3
) -> list[CandidateRecommendation]:
    """Rank the available algorithms for this dataset.

    Ordering is deterministic: tier first, then the fixed preference order
    below. Two runs over the same profile produce the same list.
    """
    from app.training.model_factory import available_algorithms

    availability = available_algorithms()
    n_rows = profile.n_rows
    n_features = len([c for c in profile.columns if c.role == "feature"])
    n_categorical = len(
        [
            c
            for c in profile.columns
            if c.role == "feature" and c.kind in ("categorical", "boolean")
        ]
    )

    shared: list[str] = [f"{n_rows:,} rows and {n_features} usable features"]
    if n_categorical:
        shared.append(f"{n_categorical} categorical feature(s)")

    # Fixed preference order; the tier decides placement, this decides ties.
    order = [
        "hist_gradient_boosting",
        "random_forest",
        "xgboost",
        "lightgbm",
        "logistic_regression",
    ]
    out: list[CandidateRecommendation] = []

    for algo in order:
        note = _ALGORITHM_NOTES[algo]
        installed = availability.get(algo, False)
        reasons: list[str] = []

        if problem_type != "binary_classification":
            out.append(
                CandidateRecommendation(
                    algorithm=algo,
                    label=note["label"],
                    tier="unsuitable",
                    reasons=[
                        f"the training stack fits binary classification only; this target is {problem_type}"
                    ],
                    available=False,
                    unavailable_reason="problem type not supported by the training pipeline",
                )
            )
            continue

        if not installed:
            out.append(
                CandidateRecommendation(
                    algorithm=algo,
                    label=note["label"],
                    tier="unsuitable",
                    reasons=[f"{note['label']} is not installed in this environment"],
                    available=False,
                    unavailable_reason="optional dependency not installed",
                )
            )
            continue

        tier: Tier
        if algo == "logistic_regression":
            tier = "baseline"
            reasons.append(
                "included as a reference point: if a boosted model cannot beat it, "
                "the extra complexity is not earning anything"
            )
        elif n_rows < note["prefers_rows"]:
            tier = "optional"
            reasons.append(
                f"typically wants more than {note['prefers_rows']:,} rows; "
                f"this dataset has {n_rows:,}"
            )
        else:
            tier = "recommended"
            reasons.extend(note["strengths"][:2])

        reasons.extend(shared)
        out.append(
            CandidateRecommendation(
                algorithm=algo, label=note["label"], tier=tier, reasons=reasons, available=True
            )
        )

    rank = {"recommended": 0, "baseline": 1, "optional": 2, "unsuitable": 3}
    out.sort(key=lambda c: (rank[c.tier], order.index(c.algorithm)))
    return out


def default_selection(
    recommendations: list[CandidateRecommendation], max_models: int = 3
) -> list[str]:
    """Which candidates to pre-select.

    Recommended first, then the interpretable baseline, capped at ``max_models``
    so opening the page cannot queue an unbounded amount of training.
    """
    picked = [c.algorithm for c in recommendations if c.available and c.tier == "recommended"]
    if len(picked) < max_models:
        picked += [
            c.algorithm for c in recommendations if c.available and c.tier == "baseline"
        ]
    return picked[:max_models]


def primary_metric_for(
    problem_type: ProblemType, profile: DatasetProfile
) -> tuple[str, list[str], str]:
    """Default primary metric, the secondaries worth watching, and why.

    Only metrics the existing evaluator already computes are offered.
    """
    target = profile.suggested_target
    imbalanced = False
    if target and target.class_balance:
        imbalanced = min(target.class_balance.values()) < 0.20

    if problem_type == "binary_classification":
        if imbalanced:
            return (
                "roc_auc",
                ["f1", "recall", "precision", "pr_auc"],
                "ROC-AUC ranks well under imbalance, but on a skewed target it can look strong "
                "while the minority class is being missed -- F1 and recall are shown alongside it.",
            )
        return (
            "roc_auc",
            ["f1", "precision", "recall"],
            "ROC-AUC is threshold-independent, so it compares candidates without fixing an "
            "operating point first.",
        )
    return ("roc_auc", ["f1"], "")
