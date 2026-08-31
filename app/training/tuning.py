"""Hyperparameter optimisation.

One :class:`Tuner` interface, three backends:

``local_grid``
    Exhaustive grid search. Deterministic, good for small spaces and for tests.
``local_random``
    Random search over the same declared space, capped at ``max_trials``. The
    default -- random search dominates grid search at equal budget for all but
    the smallest spaces.
``sagemaker``
    Submits an Amazon SageMaker Hyperparameter Tuning Job
    (:mod:`app.aws.sagemaker`). Used when running the pipeline inside AWS.

All three evaluate candidates with stratified cross-validation on the training
split only -- the test split is never touched during search, which is what keeps
the reported test metrics trustworthy.
"""

from __future__ import annotations

import itertools
import time
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

from app.core.config import Settings, TuningConfig, get_settings
from app.core.exceptions import TuningError
from app.core.logging import get_logger
from app.data.preprocessing import build_feature_pipeline
from app.schemas.model import TrialResult, TuningResult
from app.training.model_factory import build_estimator, normalise_params

logger = get_logger(__name__)

# Maps our metric names onto sklearn scorer names.
SCORERS: dict[str, str] = {
    "roc_auc": "roc_auc",
    "pr_auc": "average_precision",
    "f1": "f1",
    "precision": "precision",
    "recall": "recall",
    "accuracy": "accuracy",
    "neg_log_loss": "neg_log_loss",
}


class Tuner(ABC):
    """Searches a hyperparameter space and returns the best configuration."""

    backend: str = "abstract"

    @abstractmethod
    def search(
        self,
        features: pd.DataFrame,
        target: pd.Series,
        algorithm: str,
        search_space: dict[str, list[Any]],
    ) -> TuningResult: ...


class LocalSearchTuner(Tuner):
    """In-process grid or random search with stratified cross-validation."""

    def __init__(
        self,
        config: TuningConfig | None = None,
        settings: Settings | None = None,
        mode: str = "random",
    ) -> None:
        self.settings = settings or get_settings()
        self.config = config or self.settings.tuning
        self.mode = mode
        self.backend = f"local_{mode}"

    def _candidates(self, search_space: dict[str, list[Any]]) -> list[dict[str, Any]]:
        if not search_space:
            return [{}]
        keys = sorted(search_space)
        grid = [
            dict(zip(keys, values, strict=True))
            for values in itertools.product(*(search_space[k] for k in keys))
        ]
        if self.mode == "grid":
            if len(grid) > self.config.max_trials:
                logger.warning(
                    "tuning.grid_truncated",
                    extra={"total": len(grid), "max_trials": self.config.max_trials},
                )
            return grid[: self.config.max_trials]

        rng = np.random.default_rng(self.settings.data.random_seed)
        if len(grid) <= self.config.max_trials:
            return grid
        chosen = rng.choice(len(grid), size=self.config.max_trials, replace=False)
        return [grid[int(i)] for i in sorted(chosen)]

    def search(
        self,
        features: pd.DataFrame,
        target: pd.Series,
        algorithm: str,
        search_space: dict[str, list[Any]],
    ) -> TuningResult:
        started = time.perf_counter()
        scorer = SCORERS.get(self.config.metric)
        if scorer is None:
            raise TuningError(
                f"unsupported tuning metric {self.config.metric!r}; "
                f"supported: {', '.join(sorted(SCORERS))}",
                metric=self.config.metric,
            )

        candidates = self._candidates(search_space)
        folds = max(2, min(self.settings.training.cv_folds, int(target.value_counts().min())))
        splitter = StratifiedKFold(
            n_splits=folds, shuffle=True, random_state=self.settings.data.random_seed
        )

        trials: list[TrialResult] = []
        for index, params in enumerate(candidates):
            trial_start = time.perf_counter()
            try:
                pipeline = Pipeline(
                    steps=[
                        *build_feature_pipeline(self.settings.data).steps,
                        (
                            "estimator",
                            build_estimator(
                                algorithm,
                                params,
                                self.settings.training.class_weight_balanced,
                            ),
                        ),
                    ]
                )
                scores = cross_val_score(
                    pipeline,
                    features,
                    target,
                    cv=splitter,
                    scoring=scorer,
                    n_jobs=self.config.n_jobs,
                    error_score="raise",
                )
                score = float(np.mean(scores))
                trials.append(
                    TrialResult(
                        trial_id=index,
                        params=normalise_params(algorithm, params),
                        score=score,
                        metric=self.config.metric,
                        duration_seconds=round(time.perf_counter() - trial_start, 3),
                    )
                )
                logger.info(
                    "tuning.trial",
                    extra={
                        "trial": index,
                        "params": params,
                        "score": round(score, 5),
                        "metric": self.config.metric,
                    },
                )
            except Exception as exc:
                # One bad configuration must not sink the whole search, but it
                # is recorded as a failed trial rather than silently skipped.
                logger.warning(
                    "tuning.trial_failed",
                    extra={"trial": index, "params": params, "error": str(exc)},
                )
                trials.append(
                    TrialResult(
                        trial_id=index,
                        params=params,
                        score=(
                            float("-inf")
                            if self.config.direction == "maximize"
                            else float("inf")
                        ),
                        metric=self.config.metric,
                        duration_seconds=round(time.perf_counter() - trial_start, 3),
                        status="failed",
                        error=str(exc)[:500],
                    )
                )

        completed = [t for t in trials if t.status == "completed"]
        if not completed:
            raise TuningError(
                "every hyperparameter trial failed; see the logs for the " "per-trial errors",
                algorithm=algorithm,
                n_trials=len(trials),
                first_error=trials[0].error if trials else None,
            )

        best = (
            max(completed, key=lambda t: t.score)
            if self.config.direction == "maximize"
            else min(completed, key=lambda t: t.score)
        )
        result = TuningResult(
            backend=self.backend,
            metric=self.config.metric,
            direction=self.config.direction,
            n_trials=len(trials),
            best_params=best.params,
            best_score=best.score,
            trials=trials,
            duration_seconds=round(time.perf_counter() - started, 3),
        )
        logger.info(
            "tuning.completed",
            extra={
                "backend": self.backend,
                "trials": len(trials),
                "failed": len(trials) - len(completed),
                "best_score": round(best.score, 5),
                "best_params": best.params,
                "duration_s": result.duration_seconds,
            },
        )
        return result


class SageMakerTuner(Tuner):
    """Delegates the search to a SageMaker Hyperparameter Tuning Job.

    Requires AWS to be configured (role ARN, training image, S3 bucket). It does
    not simulate anything: without those settings it raises, so a run can never
    claim SageMaker tuning happened when it did not.
    """

    backend = "sagemaker"

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def search(
        self,
        features: pd.DataFrame,
        target: pd.Series,
        algorithm: str,
        search_space: dict[str, list[Any]],
    ) -> TuningResult:
        from app.aws.sagemaker import SageMakerClient

        client = SageMakerClient(self.settings)
        return client.run_tuning_job(
            algorithm=algorithm,
            search_space=search_space,
            metric=self.settings.tuning.metric,
            direction=self.settings.tuning.direction,
            max_trials=self.settings.tuning.max_trials,
        )


def build_tuner(settings: Settings | None = None) -> Tuner:
    settings = settings or get_settings()
    backend = settings.tuning.backend
    if backend == "sagemaker":
        return SageMakerTuner(settings)
    if backend == "local_grid":
        return LocalSearchTuner(settings.tuning, settings, mode="grid")
    return LocalSearchTuner(settings.tuning, settings, mode="random")


def tune(
    features: pd.DataFrame,
    target: pd.Series,
    algorithm: str | None = None,
    search_space: dict[str, list[Any]] | None = None,
    settings: Settings | None = None,
) -> TuningResult:
    """Run the configured tuner over the configured search space."""
    settings = settings or get_settings()
    tuner = build_tuner(settings)
    return tuner.search(
        features,
        target,
        algorithm or settings.training.algorithm,
        search_space or settings.tuning.search_space,
    )
