"""The training pipeline.

Stages, in order, each logged and each producing artifacts:

1. **Load**            -- resolve a dataset version (never an anonymous file).
2. **Validate**        -- blocking gate; invalid data stops the run here.
3. **Split**           -- train / validation / test, stratified and seeded.
4. **Preprocess**      -- feature engineering + encoding, inside the pipeline.
5. **Tune**            -- optional hyperparameter search on the training split.
6. **Train**           -- fit the final pipeline on train (+validation) data.
7. **Evaluate**        -- threshold chosen on validation, metrics on test.
8. **Register**        -- persist the artifact and the reproducibility block.

Everything needed to reproduce the run -- dataset version, dataset content hash,
git commit, library versions, resolved parameters, chosen threshold -- is logged
to the tracker and stored on the model version.
"""

from __future__ import annotations

import platform
import sys
import time
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from app.core.config import Settings, get_settings
from app.core.exceptions import TrainingError
from app.core.logging import StageTimer, get_logger, log_context
from app.core.signature import (
    ModelSignature,
    SignatureError,
    build_signature,
    validate_model_name,
)
from app.core.storage import build_artifact_store
from app.core.utils import git_commit, jsonable, new_id, utcnow_iso, write_json
from app.data.preprocessing import (
    build_feature_pipeline,
    feature_names,
    split_features_target,
)
from app.data.validation import validate_or_raise
from app.data.versioning import build_profile, get_dataset_registry
from app.registry.base import ModelRegistry
from app.registry.context import check_lineage_compatible
from app.registry.factory import get_registry
from app.schemas.model import (
    EvaluationResult,
    TrainingRequest,
    TrainingRunResult,
    TuningResult,
)
from app.tracking.base import ExperimentTracker
from app.tracking.factory import build_tracker
from app.training.evaluate import (
    evaluate_model,
    metrics_to_flat_dict,
    predict_proba,
    select_threshold,
)
from app.training.model_factory import (
    DEFAULT_PARAMS,
    build_estimator,
    normalise_params,
)
from app.training.tuning import tune

logger = get_logger(__name__)


def environment_snapshot() -> dict[str, str]:
    """Library and interpreter versions, recorded with every run."""
    import numpy
    import sklearn

    snapshot = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": numpy.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
    }
    for optional in ("mlflow", "xgboost", "lightgbm"):
        try:
            module = __import__(optional)
            snapshot[optional] = getattr(module, "__version__", "unknown")
        except ImportError:
            continue
    return snapshot


def build_pipeline(algorithm: str, params: dict[str, Any], settings: Settings) -> Pipeline:
    """Feature pipeline + estimator as one serialisable object.

    Serialising the *whole* pipeline is what guarantees the serving path applies
    exactly the transformations the training path did.
    """
    feature_pipeline = build_feature_pipeline(settings.data)
    return Pipeline(
        steps=[
            *feature_pipeline.steps,
            (
                "estimator",
                build_estimator(algorithm, params, settings.training.class_weight_balanced),
            ),
        ]
    )


def train_model(
    request: TrainingRequest | None = None,
    settings: Settings | None = None,
    tracker: ExperimentTracker | None = None,
    registry: ModelRegistry | None = None,
) -> TrainingRunResult:
    """Run the full training pipeline once and return its result."""
    settings = settings or get_settings()
    request = request or TrainingRequest()
    settings.paths.ensure()

    algorithm = request.algorithm or settings.training.algorithm
    experiment_name = request.experiment_name or settings.tracking.experiment_name
    model_name = request.model_name or settings.tracking.registered_model_name
    try:
        validate_model_name(model_name)
    except SignatureError as exc:
        raise TrainingError(str(exc), model=model_name) from exc
    should_tune = settings.tuning.enabled if request.tune is None else request.tune
    commit = git_commit()
    started = time.perf_counter()

    tracker = tracker or build_tracker(settings)
    registry = registry or get_registry()
    dataset_registry = get_dataset_registry()

    run_name = request.run_name or f"{algorithm}-{utcnow_iso()[:19].replace(':', '')}"
    correlation_id = new_id("train")

    with log_context(pipeline_run=correlation_id, model=model_name):
        # ---------------- 1. load ------------------------------------------ #
        with StageTimer(logger, "load_data"):
            frame, dataset_version = dataset_registry.resolve(
                version=request.dataset_version, path=request.dataset_path
            )
            version_id = dataset_version.version if dataset_version else None
            dataset_hash = dataset_version.content_hash if dataset_version else None

        # ---------------- 1b. signature ------------------------------------ #
        # Recorded before validation because the target's two classes have to
        # be known to validate it at all. From here on the run uses the data
        # contract rebuilt from the signature, so what is recorded is exactly
        # what is trained.
        try:
            signature = build_signature(frame, settings.data)
        except SignatureError as exc:
            raise TrainingError(
                f"dataset {version_id or 'unregistered'} cannot train a binary "
                f"classifier: {exc}",
                dataset_version=version_id,
            ) from exc
        settings = settings.model_copy(
            update={"data": signature.to_data_config(settings.data)}, deep=True
        )
        # A name keeps one prediction problem for life; see check_lineage_compatible.
        check_lineage_compatible(
            registry, model_name, signature.target, signature.class_labels, signature.task
        )

        # ---------------- 2. validate -------------------------------------- #
        with StageTimer(logger, "validate_data", dataset_version=version_id):
            # Raises DataValidationError, which stops the pipeline. The report
            # is attached to the exception so the caller can surface it.
            validation_report = validate_or_raise(
                frame, settings.data.dataset_name, version_id, settings
            )

        with (
            tracker.run(
                experiment_name,
                run_name=run_name,
                tags={
                    "fmops.algorithm": algorithm,
                    "fmops.dataset_version": version_id or "unregistered",
                    "fmops.pipeline_run": correlation_id,
                    **request.tags,
                },
            ) as run_info,
            log_context(run_id=run_info.run_id),
        ):
            result = _train_within_run(
                frame=frame,
                algorithm=algorithm,
                should_tune=should_tune,
                request=request,
                settings=settings,
                tracker=tracker,
                registry=registry,
                run_info=run_info,
                experiment_name=experiment_name,
                model_name=model_name,
                dataset_version=version_id,
                dataset_hash=dataset_hash,
                commit=commit,
                validation_passed=validation_report.passed,
                signature=signature,
            )
            result.duration_seconds = round(time.perf_counter() - started, 3)
            return result


def _train_within_run(
    *,
    frame: pd.DataFrame,
    algorithm: str,
    should_tune: bool,
    request: TrainingRequest,
    settings: Settings,
    tracker: ExperimentTracker,
    registry: ModelRegistry,
    run_info,
    experiment_name: str,
    model_name: str,
    dataset_version: str | None,
    dataset_hash: str | None,
    commit: str,
    validation_passed: bool,
    signature: ModelSignature,
) -> TrainingRunResult:
    # ---------------- 3. split --------------------------------------------- #
    with StageTimer(logger, "split_data"):
        features, target = split_features_target(frame, settings.data)
        if target.nunique() < 2:
            raise TrainingError(
                "training data contains a single class; the model cannot learn "
                "a decision boundary",
                classes=sorted(target.unique().tolist()),
            )
        X_fit, X_test, y_fit, y_test = train_test_split(
            features,
            target,
            test_size=settings.data.test_size,
            random_state=settings.data.random_seed,
            stratify=target,
        )
        validation_share = settings.data.validation_size / (1.0 - settings.data.test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_fit,
            y_fit,
            test_size=validation_share,
            random_state=settings.data.random_seed,
            stratify=y_fit,
        )
        logger.info(
            "split.completed",
            extra={
                "train": len(X_train),
                "validation": len(X_val),
                "test": len(X_test),
                "positive_rate_train": round(float(y_train.mean()), 4),
            },
        )

    # ---------------- 4/5. tune -------------------------------------------- #
    tuning_result: TuningResult | None = None
    # Start from the algorithm's declared defaults so the recorded params fully
    # determine the estimator. Recording only the overrides would mean a run
    # could not be reproduced without also knowing which library version's
    # defaults were in force at the time.
    params: dict[str, Any] = {
        **DEFAULT_PARAMS.get(algorithm, {}),
        **normalise_params(algorithm, settings.training.hyperparameters),
        **normalise_params(algorithm, request.hyperparameters),
    }
    if should_tune:
        with StageTimer(logger, "hyperparameter_tuning", algorithm=algorithm):
            tuning_result = tune(
                X_train, y_train, algorithm, settings.tuning.search_space, settings
            )
            params = {**params, **normalise_params(algorithm, tuning_result.best_params)}
            tracker.log_metrics(
                {
                    "tuning_best_score": tuning_result.best_score,
                    "tuning_trials": float(tuning_result.n_trials),
                }
            )
            tracker.log_dict(tuning_result.model_dump(mode="json"), "tuning_result.json")
    else:
        logger.info("tuning.skipped", extra={"reason": "disabled by configuration"})

    tracker.log_params(
        {
            "algorithm": algorithm,
            "dataset_version": dataset_version or "unregistered",
            "dataset_hash": (dataset_hash or "")[:16],
            "git_commit": commit[:12],
            "tuned": should_tune,
            "n_train": len(X_train),
            "n_validation": len(X_val),
            "n_test": len(X_test),
            "test_size": settings.data.test_size,
            "random_seed": settings.data.random_seed,
            **{f"hp_{k}": v for k, v in params.items()},
        }
    )

    # ---------------- 6. train --------------------------------------------- #
    with StageTimer(logger, "train_model", algorithm=algorithm):
        pipeline = build_pipeline(algorithm, params, settings)
        try:
            pipeline.fit(X_train, y_train)
        except Exception as exc:
            raise TrainingError(
                f"model fitting failed: {exc}",
                algorithm=algorithm,
                params=jsonable(params),
            ) from exc

    # ---------------- 7. evaluate ------------------------------------------ #
    with StageTimer(logger, "evaluate_model"):
        # Threshold is selected on validation data only, then frozen.
        validation_probabilities = predict_proba(pipeline, X_val)
        threshold = select_threshold(y_val.to_numpy(), validation_probabilities)

        names = feature_names(pipeline)
        validation_eval = evaluate_model(
            pipeline,
            X_val,
            y_val,
            model_name,
            dataset_version=dataset_version,
            split="validation",
            threshold=threshold,
            feature_names=names,
            measure_latency=False,
            compute_importance=False,
        )
        test_eval = evaluate_model(
            pipeline,
            X_test,
            y_test,
            model_name,
            dataset_version=dataset_version,
            split="test",
            threshold=threshold,
            feature_names=names,
            measure_latency=True,
        )
        tracker.log_metrics(metrics_to_flat_dict(test_eval))
        tracker.log_metrics(metrics_to_flat_dict(validation_eval, prefix="val_"))
        tracker.log_dict(test_eval.model_dump(mode="json"), "evaluation_test.json")
        tracker.log_dict(validation_eval.model_dump(mode="json"), "evaluation_validation.json")

    # ---------------- 8. persist + register -------------------------------- #
    with StageTimer(logger, "persist_model"):
        artifact_uri, model_path = _persist_model(
            pipeline=pipeline,
            settings=settings,
            model_name=model_name,
            run_id=run_info.run_id,
            evaluation=test_eval,
            threshold=threshold,
            algorithm=algorithm,
            params=params,
            dataset_version=dataset_version,
            dataset_hash=dataset_hash,
            commit=commit,
            feature_columns=list(features.columns),
            signature=signature,
        )
        tracker.log_artifact(model_path.parent)

        # Reference profile for drift detection: the training distribution the
        # production window will later be compared against.
        profile = build_profile(frame, settings.data.dataset_name, dataset_version, settings)
        profile_path = (
            settings.paths.reports_dir / f"{model_name}-{run_info.run_id}-profile.json"
        )
        write_json(profile_path, profile.model_dump(mode="json"))
        tracker.log_artifact(profile_path)

    registered_version: int | None = None
    if request.register_model:
        with StageTimer(logger, "register_model"):
            model_version = registry.register(
                name=model_name,
                artifact_uri=artifact_uri,
                run_id=run_info.run_id,
                metrics=test_eval.metrics.as_dict(),
                params={"algorithm": algorithm, "threshold": threshold, **params},
                dataset_version=dataset_version,
                dataset_hash=dataset_hash,
                git_commit=commit,
                algorithm=algorithm,
                tags={
                    "experiment": experiment_name,
                    "tuned": str(should_tune).lower(),
                    "threshold": f"{threshold:.4f}",
                },
                signature=signature.model_dump(mode="json"),
                description=(
                    f"{algorithm} trained on {dataset_version or 'unregistered data'} "
                    f"({len(X_train)} rows)"
                ),
            )
            registered_version = model_version.version
            test_eval.model_version = registered_version
            tracker.set_tags({"fmops.model_version": str(registered_version)})

    tracker.log_model_metadata(
        {
            "model_name": model_name,
            "model_version": registered_version,
            "algorithm": algorithm,
            "params": jsonable(params),
            "threshold": threshold,
            "dataset_version": dataset_version,
            "dataset_hash": dataset_hash,
            "git_commit": commit,
            "environment": environment_snapshot(),
            "feature_columns": list(features.columns),
            "n_engineered_features": len(names),
        }
    )

    return TrainingRunResult(
        run_id=run_info.run_id,
        experiment_name=experiment_name,
        model_name=model_name,
        algorithm=algorithm,
        params=jsonable(params),
        dataset_version=dataset_version,
        dataset_hash=dataset_hash,
        git_commit=commit,
        model_path=str(model_path),
        artifact_uri=artifact_uri,
        evaluation=test_eval,
        tuning=tuning_result,
        validation_passed=validation_passed,
        registered_version=registered_version,
        environment=environment_snapshot(),
    )


def _persist_model(
    *,
    pipeline: Pipeline,
    settings: Settings,
    model_name: str,
    run_id: str,
    evaluation: EvaluationResult,
    threshold: float,
    algorithm: str,
    params: dict[str, Any],
    dataset_version: str | None,
    dataset_hash: str | None,
    commit: str,
    feature_columns: list[str],
    signature: ModelSignature,
) -> tuple[str, Path]:
    """Write the model + its sidecar metadata, then upload to the artifact store.

    The sidecar is what lets the serving layer load a model without the registry
    being reachable: it carries the threshold and the expected feature columns.
    """
    local_dir = settings.paths.models_dir / model_name / run_id
    local_dir.mkdir(parents=True, exist_ok=True)
    model_path = local_dir / "model.joblib"
    joblib.dump(pipeline, model_path, compress=3)

    metadata = {
        "model_name": model_name,
        "run_id": run_id,
        "algorithm": algorithm,
        "params": jsonable(params),
        "threshold": threshold,
        "metrics": evaluation.metrics.as_dict(),
        "dataset_version": dataset_version,
        "dataset_hash": dataset_hash,
        "git_commit": commit,
        "feature_columns": feature_columns,
        "signature": signature.model_dump(mode="json"),
        "environment": environment_snapshot(),
        "created_at": utcnow_iso(),
    }
    write_json(local_dir / "metadata.json", metadata)

    store = build_artifact_store(settings)
    key_prefix = f"models/{model_name}/{run_id}"
    artifact_uri = store.put_file(f"{key_prefix}/model.joblib", model_path)
    store.put_file(f"{key_prefix}/metadata.json", local_dir / "metadata.json")

    logger.info(
        "model.persisted",
        extra={
            "path": str(model_path),
            "artifact_uri": artifact_uri,
            "size_kb": round(model_path.stat().st_size / 1024, 1),
        },
    )
    return artifact_uri, model_path


def load_model(path_or_uri: str | Path) -> Pipeline:
    """Load a serialised pipeline from a local path or an artifact URI."""
    from app.core.storage import resolve_uri_to_local

    text = str(path_or_uri)
    if "://" in text:
        settings = get_settings()
        destination = settings.paths.models_dir / "_cache" / Path(text).name
        destination.parent.mkdir(parents=True, exist_ok=True)
        local = resolve_uri_to_local(text, destination)
    else:
        local = Path(text)
    if not local.is_file():
        raise TrainingError(f"model artifact not found: {local}", path=str(local))
    return joblib.load(local)
