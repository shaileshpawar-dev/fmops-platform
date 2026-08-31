"""Model evaluation.

Produces the :class:`~app.schemas.model.EvaluationResult` that every downstream
decision reads: the approval gate, champion/challenger comparison and the
registry all consume these numbers.

Two details that matter for a class-imbalanced problem:

* **Threshold selection.** The default 0.5 cut-off is a poor operating point at
  an 18% positive rate. :func:`select_threshold` picks the threshold that
  maximises F1 on the validation split; it is chosen on *validation* data and
  then applied unchanged to test data, so the reported test metrics are not
  optimistically biased.
* **Latency is measured, not guessed.** :func:`measure_inference_latency` times
  single-row predictions through the full fitted pipeline, which is what the
  approval gate's latency ceiling is checked against.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from app.core.exceptions import EvaluationError
from app.core.logging import get_logger
from app.core.utils import percentile, safe_float
from app.schemas.model import ConfusionMatrix, EvaluationResult, Metrics

logger = get_logger(__name__)


def predict_proba(model: Pipeline, features: pd.DataFrame) -> np.ndarray:
    """Positive-class probabilities, whatever the estimator exposes."""
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(features)
        return np.asarray(proba)[:, 1]
    if hasattr(model, "decision_function"):
        scores = np.asarray(model.decision_function(features))
        return 1.0 / (1.0 + np.exp(-scores))
    raise EvaluationError(
        "estimator exposes neither predict_proba nor decision_function",
        estimator=type(model).__name__,
    )


def select_threshold(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    objective: str = "f1",
    grid: int = 99,
) -> float:
    """Pick the probability cut-off that maximises ``objective``.

    Must be called on validation data, never on the test split.
    """
    y_true = np.asarray(y_true).astype(int)
    if len(np.unique(y_true)) < 2:
        logger.warning("evaluate.threshold_single_class")
        return 0.5

    candidates = np.linspace(0.01, 0.99, grid)
    best_threshold, best_score = 0.5, -1.0
    for threshold in candidates:
        predicted = (probabilities >= threshold).astype(int)
        if objective == "f1":
            score = f1_score(y_true, predicted, zero_division=0)
        elif objective == "youden":
            tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()
            sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
            specificity = tn / (tn + fp) if (tn + fp) else 0.0
            score = sensitivity + specificity - 1.0
        else:
            raise EvaluationError(f"unknown threshold objective {objective!r}")
        if score > best_score:
            best_threshold, best_score = float(threshold), float(score)
    logger.info(
        "evaluate.threshold_selected",
        extra={
            "threshold": round(best_threshold, 4),
            "objective": objective,
            "score": round(best_score, 4),
        },
    )
    return best_threshold


def measure_inference_latency(
    model: Pipeline,
    features: pd.DataFrame,
    n_samples: int = 100,
    warmup: int = 5,
) -> tuple[float, float]:
    """Time single-row inference through the fitted pipeline.

    Returns (p50_ms, p95_ms). Single-row timing is the honest measure for a
    real-time endpoint -- batch throughput would flatter the numbers.
    """
    if features.empty:
        return 0.0, 0.0
    sample = features.iloc[: min(n_samples, len(features))]

    for i in range(min(warmup, len(sample))):
        model.predict(sample.iloc[[i]])

    timings: list[float] = []
    for i in range(len(sample)):
        row = sample.iloc[[i]]
        start = time.perf_counter()
        model.predict(row)
        timings.append((time.perf_counter() - start) * 1000.0)

    return percentile(timings, 50), percentile(timings, 95)


def compute_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float = 0.5,
) -> tuple[Metrics, ConfusionMatrix]:
    """Full metric set at a given operating threshold."""
    y_true = np.asarray(y_true).astype(int)
    probabilities = np.asarray(probabilities, dtype=float)
    predicted = (probabilities >= threshold).astype(int)

    single_class = len(np.unique(y_true)) < 2
    if single_class:
        logger.warning(
            "evaluate.single_class_split",
            extra={"n": len(y_true), "detail": "ROC-AUC and PR-AUC are undefined"},
        )

    metrics = Metrics(
        accuracy=safe_float(accuracy_score(y_true, predicted)),
        precision=safe_float(precision_score(y_true, predicted, zero_division=0)),
        recall=safe_float(recall_score(y_true, predicted, zero_division=0)),
        f1=safe_float(f1_score(y_true, predicted, zero_division=0)),
        roc_auc=0.0 if single_class else safe_float(roc_auc_score(y_true, probabilities)),
        pr_auc=(
            0.0 if single_class else safe_float(average_precision_score(y_true, probabilities))
        ),
        log_loss=safe_float(
            log_loss(y_true, np.clip(probabilities, 1e-7, 1 - 1e-7), labels=[0, 1])
        ),
        brier_score=safe_float(brier_score_loss(y_true, probabilities)),
        n_samples=len(y_true),
        positive_rate=safe_float(y_true.mean()),
    )

    tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()
    matrix = ConfusionMatrix(
        true_negative=int(tn),
        false_positive=int(fp),
        false_negative=int(fn),
        true_positive=int(tp),
    )
    return metrics, matrix


def extract_feature_importance(
    model: Pipeline,
    feature_names: list[str],
    top_n: int = 25,
    features: pd.DataFrame | None = None,
    target: np.ndarray | None = None,
) -> dict[str, float]:
    """Best-effort importance, in decreasing order of preference.

    1. ``feature_importances_`` (tree ensembles) or ``|coef_|`` (linear models),
       both of which describe the *encoded* feature space.
    2. Permutation importance over the raw input columns, for estimators that
       expose neither -- notably ``HistGradientBoostingClassifier``. Computed on
       a capped sample because it costs one refit-free scoring pass per column.

    Returns {} when nothing can be computed, rather than fabricating numbers.
    """
    estimator = model.named_steps.get("estimator") if hasattr(model, "named_steps") else model
    values: np.ndarray | None = None

    if hasattr(estimator, "feature_importances_"):
        values = np.asarray(estimator.feature_importances_, dtype=float)
    elif hasattr(estimator, "coef_"):
        values = np.abs(np.asarray(estimator.coef_, dtype=float)).ravel()

    if (values is None or not len(values)) and features is not None and target is not None:
        return _permutation_importance(model, features, target, top_n)

    if values is None or not len(values):
        return {}
    if feature_names and len(feature_names) != len(values):
        logger.debug(
            "evaluate.importance_length_mismatch",
            extra={"names": len(feature_names), "values": len(values)},
        )
        feature_names = [f"f{i}" for i in range(len(values))]
    names = feature_names or [f"f{i}" for i in range(len(values))]

    total = float(values.sum()) or 1.0
    pairs = sorted(
        ((n, float(v) / total) for n, v in zip(names, values, strict=False)),
        key=lambda kv: kv[1],
        reverse=True,
    )
    return {name: round(score, 6) for name, score in pairs[:top_n]}


def _permutation_importance(
    model: Pipeline,
    features: pd.DataFrame,
    target: np.ndarray,
    top_n: int,
    max_rows: int = 1500,
    n_repeats: int = 3,
    seed: int = 42,
) -> dict[str, float]:
    """Drop in ROC-AUC when each raw input column is shuffled.

    Reported over the *raw* columns rather than the encoded ones, which is what
    an operator actually wants to reason about. Negative drops are clipped to
    zero: a column whose shuffling improves the score is not informative.
    """
    y_true = np.asarray(target).astype(int)
    if len(np.unique(y_true)) < 2:
        return {}

    rng = np.random.default_rng(seed)
    if len(features) > max_rows:
        index = rng.choice(len(features), size=max_rows, replace=False)
        sample = features.iloc[index].reset_index(drop=True)
        y_sample = y_true[index]
    else:
        sample = features.reset_index(drop=True)
        y_sample = y_true

    try:
        baseline = roc_auc_score(y_sample, predict_proba(model, sample))
    except Exception as exc:
        logger.warning("evaluate.permutation_baseline_failed", extra={"error": str(exc)})
        return {}

    drops: dict[str, float] = {}
    for column in sample.columns:
        deltas: list[float] = []
        original = sample[column].to_numpy(copy=True)
        for _ in range(n_repeats):
            shuffled = sample.copy()
            shuffled[column] = rng.permutation(original)
            try:
                score = roc_auc_score(y_sample, predict_proba(model, shuffled))
            except Exception as exc:
                logger.debug(
                    "evaluate.permutation_pass_failed",
                    extra={"feature": column, "error": str(exc)},
                )
                continue
            deltas.append(float(baseline - score))
        if deltas:
            drops[column] = max(0.0, float(np.mean(deltas)))

    total = sum(drops.values())
    if total <= 0:
        return {}
    ranked = sorted(drops.items(), key=lambda kv: kv[1], reverse=True)
    return {name: round(value / total, 6) for name, value in ranked[:top_n]}


def calibration_curve_points(
    y_true: np.ndarray, probabilities: np.ndarray, bins: int = 10
) -> dict[str, list[float]]:
    """Reliability-diagram points: mean predicted vs observed rate per bin."""
    y_true = np.asarray(y_true).astype(int)
    probabilities = np.asarray(probabilities, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    predicted_means: list[float] = []
    observed_means: list[float] = []
    counts: list[float] = []
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (probabilities >= lo) & (
            probabilities < hi if i < bins - 1 else probabilities <= hi
        )
        if not mask.any():
            continue
        predicted_means.append(float(probabilities[mask].mean()))
        observed_means.append(float(y_true[mask].mean()))
        counts.append(float(mask.sum()))
    return {"predicted": predicted_means, "observed": observed_means, "count": counts}


def evaluate_model(
    model: Pipeline,
    features: pd.DataFrame,
    target: pd.Series | np.ndarray,
    model_name: str,
    model_version: int | None = None,
    dataset_version: str | None = None,
    split: str = "test",
    threshold: float = 0.5,
    feature_names: list[str] | None = None,
    measure_latency: bool = True,
    compute_importance: bool = True,
) -> EvaluationResult:
    """Evaluate a fitted pipeline and return the full result object."""
    if len(features) == 0:
        raise EvaluationError("cannot evaluate on an empty dataset", split=split)

    y_true = np.asarray(target).astype(int)
    try:
        probabilities = predict_proba(model, features)
    except Exception as exc:
        raise EvaluationError(
            f"inference failed during evaluation: {exc}",
            split=split,
            model=model_name,
        ) from exc

    metrics, matrix = compute_metrics(y_true, probabilities, threshold)

    if measure_latency:
        p50, p95 = measure_inference_latency(model, features)
        metrics.inference_latency_p50_ms = round(p50, 4)
        metrics.inference_latency_p95_ms = round(p95, 4)

    result = EvaluationResult(
        model_name=model_name,
        model_version=model_version,
        dataset_version=dataset_version,
        split=split,
        metrics=metrics,
        confusion_matrix=matrix,
        threshold=float(threshold),
        per_class=_per_class(y_true, probabilities, threshold),
        feature_importance=(
            extract_feature_importance(
                model, feature_names or [], features=features, target=y_true
            )
            if compute_importance
            else {}
        ),
        calibration=calibration_curve_points(y_true, probabilities),
    )
    logger.info(
        "evaluate.completed",
        extra={
            "model": model_name,
            "split": split,
            "n": len(y_true),
            "threshold": round(threshold, 4),
            "f1": round(metrics.f1, 4),
            "roc_auc": round(metrics.roc_auc, 4),
            "precision": round(metrics.precision, 4),
            "recall": round(metrics.recall, 4),
            "latency_p95_ms": metrics.inference_latency_p95_ms,
        },
    )
    return result


def _per_class(
    y_true: np.ndarray, probabilities: np.ndarray, threshold: float
) -> dict[str, dict[str, float]]:
    predicted = (probabilities >= threshold).astype(int)
    out: dict[str, dict[str, float]] = {}
    for label in (0, 1):
        out[str(label)] = {
            "precision": safe_float(
                precision_score(y_true, predicted, pos_label=label, zero_division=0)
            ),
            "recall": safe_float(
                recall_score(y_true, predicted, pos_label=label, zero_division=0)
            ),
            "f1": safe_float(f1_score(y_true, predicted, pos_label=label, zero_division=0)),
            "support": float((y_true == label).sum()),
        }
    return out


def metrics_to_flat_dict(result: EvaluationResult, prefix: str = "") -> dict[str, float]:
    """Flatten for MLflow / the registry."""
    flat: dict[str, Any] = {f"{prefix}{k}": v for k, v in result.metrics.as_dict().items()}
    for key, value in result.confusion_matrix.as_dict().items():
        flat[f"{prefix}cm_{key}"] = float(value)
    flat[f"{prefix}threshold"] = float(result.threshold)
    return flat
