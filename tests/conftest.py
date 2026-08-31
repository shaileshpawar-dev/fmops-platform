"""Shared pytest fixtures.

Isolation strategy: every test session runs against a temporary artifacts/data
tree and a fresh SQLite database, so tests never see or mutate a developer's
real registry. ``FMOPS_ENV=test`` selects ``configs/test.yaml`` (small datasets,
no tuning, permissive gates) which keeps the suite fast.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from collections.abc import Iterator
from pathlib import Path

import pytest

# Must be set before app.core.config is imported anywhere.
os.environ.setdefault("FMOPS_ENV", "test")
os.environ.setdefault("FMOPS_LOG_LEVEL", "WARNING")
os.environ.setdefault("FMOPS_LOG_FORMAT", "console")
os.environ.setdefault("FMOPS_ENV_FILE", "")


@pytest.fixture(scope="session")
def workspace() -> Iterator[Path]:
    """A throwaway artifacts/data tree for the whole session."""
    root = Path(tempfile.mkdtemp(prefix="fmops-test-"))
    (root / "data" / "raw").mkdir(parents=True, exist_ok=True)
    (root / "data" / "processed").mkdir(parents=True, exist_ok=True)
    (root / "data" / "sample").mkdir(parents=True, exist_ok=True)
    (root / "data" / "reference").mkdir(parents=True, exist_ok=True)
    (root / "artifacts").mkdir(parents=True, exist_ok=True)

    os.environ["FMOPS_PATHS__DATA_DIR"] = str(root / "data")
    os.environ["FMOPS_PATHS__RAW_DIR"] = str(root / "data" / "raw")
    os.environ["FMOPS_PATHS__PROCESSED_DIR"] = str(root / "data" / "processed")
    os.environ["FMOPS_PATHS__SAMPLE_DIR"] = str(root / "data" / "sample")
    os.environ["FMOPS_PATHS__REFERENCE_DIR"] = str(root / "data" / "reference")
    os.environ["FMOPS_PATHS__ARTIFACTS_DIR"] = str(root / "artifacts")
    os.environ["FMOPS_PATHS__MODELS_DIR"] = str(root / "artifacts" / "models")
    os.environ["FMOPS_PATHS__REPORTS_DIR"] = str(root / "artifacts" / "reports")
    os.environ["FMOPS_PATHS__STATE_DIR"] = str(root / "artifacts" / "state")

    yield root

    shutil.rmtree(root, ignore_errors=True)


@pytest.fixture(scope="session")
def settings(workspace: Path):
    from app.core.config import reload_settings

    resolved = reload_settings()
    resolved.paths.ensure()
    return resolved


@pytest.fixture
def db(settings, tmp_path):
    """A fresh database per test, installed as the process-wide instance."""
    from app.core.db import Database, set_database

    database = Database(tmp_path / "test.db")
    set_database(database)
    yield database
    set_database(None)
    database.close()


@pytest.fixture
def clean_singletons(db):
    """Reset every module-level singleton so tests do not leak state."""
    import app.api.serving as serving
    import app.deployment.base as deployment_base
    import app.deployment.local_provider as local_provider
    import app.deployment.manager as manager_module
    import app.deployment.model_cache as model_cache
    import app.llmops.client as llm_client
    import app.llmops.cost as llm_cost
    import app.llmops.prompts.registry as prompt_registry
    import app.llmops.token_tracking as token_tracking
    import app.monitoring.alerts as alerts
    import app.monitoring.inference_log as inference_log
    import app.monitoring.service as monitoring_service
    import app.registry.factory as registry_factory
    import app.retraining.trigger as retraining_trigger
    from app.monitoring.metrics import reset_metrics

    modules = [
        (deployment_base, "_STORE"),
        (local_provider, "_PROVIDER"),
        (manager_module, "_MANAGER"),
        (model_cache, "_CACHE"),
        (registry_factory, "_REGISTRY"),
        (alerts, "_MANAGER"),
        (inference_log, "_LOG"),
        (monitoring_service, "_SERVICE"),
        (retraining_trigger, "_STORE"),
        (serving, "_SERVICE"),
        (llm_client, "_CLIENT"),
        (llm_cost, "_TRACKER"),
        (token_tracking, "_STORE"),
        (prompt_registry, "_REGISTRY"),
    ]
    for module, attribute in modules:
        setattr(module, attribute, None)

    reset_metrics()
    yield
    for module, attribute in modules:
        setattr(module, attribute, None)


@pytest.fixture
def registry(clean_singletons, db):
    from app.registry.local import LocalModelRegistry

    return LocalModelRegistry(db)


# --------------------------------------------------------------------------- #
# Data fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="session")
def valid_frame(settings):
    from app.data.generator import GenerationSpec, generate_dataset

    return generate_dataset(GenerationSpec(n_rows=1200, seed=7))


@pytest.fixture(scope="session")
def invalid_frame(settings):
    from app.data.generator import build_invalid_dataset

    return build_invalid_dataset(n_rows=600, seed=11)


@pytest.fixture(scope="session")
def drifted_frame(settings):
    from app.data.generator import build_production_dataset

    return build_production_dataset(n_rows=800, drift="severe", seed=33)


@pytest.fixture
def registered_dataset(settings, valid_frame, tmp_path):
    """A dataset file registered in a per-test dataset registry."""
    from app.data.versioning import DatasetRegistry

    path = tmp_path / "train.csv"
    valid_frame.to_csv(path, index=False)

    dataset_registry = DatasetRegistry(settings)
    dataset_registry.manifest_path = tmp_path / "versions.json"
    record = dataset_registry.register(path, use_dvc=False, description="test dataset")
    return dataset_registry, record


@pytest.fixture
def trained_pipeline(settings, valid_frame):
    """A fitted pipeline plus its held-out split. Session-cheap at 1200 rows."""
    from sklearn.model_selection import train_test_split

    from app.data.preprocessing import split_features_target
    from app.training.train import build_pipeline

    features, target = split_features_target(valid_frame, settings.data)
    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=0.25, random_state=7, stratify=target
    )
    pipeline = build_pipeline("hist_gradient_boosting", {"max_iter": 60}, settings)
    pipeline.fit(x_train, y_train)
    return pipeline, x_test, y_test


# --------------------------------------------------------------------------- #
# API
# --------------------------------------------------------------------------- #
@pytest.fixture
def api_client(clean_singletons, settings):
    from fastapi.testclient import TestClient

    from app.api.main import create_app

    with TestClient(create_app(settings)) as client:
        yield client


@pytest.fixture
def sample_features() -> dict:
    return {
        "age": 38.0,
        "annual_income": 68000.0,
        "loan_amount": 19500.0,
        "loan_term_months": 36,
        "credit_score": 688.0,
        "debt_to_income": 0.29,
        "employment_years": 6.5,
        "num_credit_lines": 6,
        "num_late_payments_12m": 1,
        "credit_utilization": 0.41,
        "employment_type": "salaried",
        "housing_status": "mortgage",
        "loan_purpose": "auto",
        "region": "west",
    }
