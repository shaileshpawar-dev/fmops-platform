"""Estimator factory.

Adding an algorithm means adding one entry here; nothing else in the pipeline
changes. XGBoost and LightGBM are optional -- requesting one without the
``[boosting]`` extra installed raises a clear
:class:`~app.core.exceptions.DependencyMissingError` rather than silently
substituting a different model, which would make the registry lie about what
was trained.
"""

from __future__ import annotations

from typing import Any

from sklearn.base import BaseEstimator
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from app.core.exceptions import DependencyMissingError, TrainingError
from app.core.logging import get_logger

logger = get_logger(__name__)

# Default hyperparameters per algorithm. Overridden by config, then by tuning.
DEFAULT_PARAMS: dict[str, dict[str, Any]] = {
    "hist_gradient_boosting": {
        "max_iter": 200,
        "learning_rate": 0.06,
        "max_depth": 8,
        "min_samples_leaf": 25,
        "l2_regularization": 1.0,
        "early_stopping": True,
        "validation_fraction": 0.12,
        "n_iter_no_change": 15,
        "random_state": 42,
    },
    "random_forest": {
        "n_estimators": 300,
        "max_depth": 14,
        "min_samples_leaf": 5,
        "max_features": "sqrt",
        "n_jobs": -1,
        "random_state": 42,
    },
    "logistic_regression": {
        "C": 1.0,
        "max_iter": 2000,
        "solver": "lbfgs",
        "random_state": 42,
    },
    "xgboost": {
        "n_estimators": 300,
        "max_depth": 6,
        "learning_rate": 0.06,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "reg_lambda": 1.0,
        "eval_metric": "logloss",
        "tree_method": "hist",
        "random_state": 42,
    },
    "lightgbm": {
        "n_estimators": 300,
        "max_depth": -1,
        "num_leaves": 31,
        "learning_rate": 0.06,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "random_state": 42,
        "verbose": -1,
    },
}

# Search-space keys are algorithm-agnostic (n_estimators, max_depth,
# learning_rate); this maps them onto each estimator's real parameter names so
# one configured search space works for every algorithm.
PARAM_ALIASES: dict[str, dict[str, str]] = {
    "hist_gradient_boosting": {"n_estimators": "max_iter"},
    "random_forest": {"learning_rate": "__drop__"},
    "logistic_regression": {
        "n_estimators": "__drop__",
        "max_depth": "__drop__",
        "learning_rate": "__drop__",
    },
}

SUPPORTED = tuple(DEFAULT_PARAMS)


def normalise_params(algorithm: str, params: dict[str, Any]) -> dict[str, Any]:
    """Translate generic search-space keys into estimator-specific ones."""
    aliases = PARAM_ALIASES.get(algorithm, {})
    out: dict[str, Any] = {}
    for key, value in params.items():
        target = aliases.get(key, key)
        if target == "__drop__":
            continue
        out[target] = value
    return out


def build_estimator(
    algorithm: str,
    params: dict[str, Any] | None = None,
    class_weight_balanced: bool = True,
) -> BaseEstimator:
    """Instantiate an estimator with defaults merged under ``params``."""
    if algorithm not in DEFAULT_PARAMS:
        raise TrainingError(
            f"unsupported algorithm {algorithm!r}; supported: {', '.join(SUPPORTED)}",
            algorithm=algorithm,
        )

    merged = {**DEFAULT_PARAMS[algorithm], **normalise_params(algorithm, params or {})}

    if algorithm == "hist_gradient_boosting":
        if class_weight_balanced:
            merged["class_weight"] = "balanced"
        return HistGradientBoostingClassifier(**merged)

    if algorithm == "random_forest":
        if class_weight_balanced:
            merged["class_weight"] = "balanced_subsample"
        return RandomForestClassifier(**merged)

    if algorithm == "logistic_regression":
        if class_weight_balanced:
            merged["class_weight"] = "balanced"
        return LogisticRegression(**merged)

    if algorithm == "xgboost":
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise DependencyMissingError(
                "xgboost is not installed; install the [boosting] extra "
                "(pip install -e '.[boosting]') or choose another algorithm",
                algorithm=algorithm,
            ) from exc
        return XGBClassifier(**merged)

    if algorithm == "lightgbm":
        try:
            from lightgbm import LGBMClassifier
        except ImportError as exc:
            raise DependencyMissingError(
                "lightgbm is not installed; install the [boosting] extra "
                "(pip install -e '.[boosting]') or choose another algorithm",
                algorithm=algorithm,
            ) from exc
        if class_weight_balanced:
            merged["class_weight"] = "balanced"
        return LGBMClassifier(**merged)

    raise TrainingError(f"unhandled algorithm {algorithm!r}", algorithm=algorithm)


def available_algorithms() -> dict[str, bool]:
    """Which algorithms can actually be instantiated in this environment."""
    status: dict[str, bool] = {}
    for name in SUPPORTED:
        if name == "xgboost":
            status[name] = _importable("xgboost")
        elif name == "lightgbm":
            status[name] = _importable("lightgbm")
        else:
            status[name] = True
    return status


def _importable(module: str) -> bool:
    import importlib.util

    return importlib.util.find_spec(module) is not None
