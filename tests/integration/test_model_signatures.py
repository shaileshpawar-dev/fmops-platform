"""Tests for ModelSignature handling, external conversion, and dashboard resilience."""

import json
from unittest.mock import MagicMock
import pytest
from pydantic import ValidationError

from app.core.signature import FeatureSpec, ModelSignature
from app.registry.context import signature_of
from app.schemas.model import ModelStage, ModelStatus, ModelVersion
from app.api.routes.dashboard import _models_section


def test_valid_internally_serialized_model_signature():
    """Test 1: A valid internally serialized ModelSignature parses and validates correctly."""
    internal = {
        "dataset_name": "credit_default",
        "target": "default",
        "class_labels": ["0", "1"],
        "display_labels": ["no_default", "default"],
        "features": [
            {
                "name": "income",
                "kind": "numeric",
                "dtype": "float64",
                "minimum": 0.0,
                "maximum": 100000.0,
                "median": 45000.0,
            },
            {
                "name": "home_ownership",
                "kind": "categorical",
                "dtype": "object",
                "categories": ["RENT", "OWN", "MORTGAGE"],
                "n_categories": 3,
            },
        ],
    }
    sig = ModelSignature.model_validate(internal)
    assert sig.dataset_name == "credit_default"
    assert sig.target == "default"
    assert sig.class_labels == ["0", "1"]
    assert len(sig.features) == 2
    assert sig.numeric_features == ["income"]
    assert sig.categorical_features == ["home_ownership"]

    version = ModelVersion(
        name="test_internal_model",
        version=1,
        artifact_uri="file:///test/path",
        signature=internal,
    )
    validated = signature_of(version)
    assert validated is not None
    assert validated.target == "default"


def test_valid_external_inputs_outputs_signature():
    """Test 2: An uploaded model with a valid external inputs/outputs signature converts seamlessly."""
    external = {
        "dataset_name": "credit_scoring",
        "class_labels": ["good", "bad"],
        "inputs": [
            {"name": "f1", "type": "double"},
            {"name": "f2", "type": "integer"},
            {"name": "f3", "type": "string"},
            {"name": "f4", "type": "boolean"},
        ],
        "outputs": [{"name": "risk_level", "type": "string"}],
    }
    sig = ModelSignature.model_validate(external)
    assert sig.dataset_name == "credit_scoring"
    assert sig.target == "risk_level"
    assert sig.class_labels == ["good", "bad"]
    assert len(sig.features) == 4
    assert sig.numeric_features == ["f1", "f2"]
    assert sig.categorical_features == ["f3", "f4"]


def test_legacy_record_with_incomplete_signature():
    """Test 3: A legacy record with an incomplete signature does not crash and returns None."""
    legacy = {
        "inputs": [
            {"name": "f1", "type": "double"},
            {"name": "f2", "type": "double"},
        ]
    }
    version = ModelVersion(
        name="legacy_model",
        version=1,
        artifact_uri="file:///test/legacy",
        signature=legacy,
    )
    sig = signature_of(version)
    assert sig is None


def test_dashboard_loading_with_invalid_and_multiple_valid_models(monkeypatch):
    """Test 4 & 5: Dashboard loading when one model has an invalid signature and multiple valid models exist."""
    valid_sig = {
        "dataset_name": "lending",
        "target": "loan_status",
        "class_labels": ["0", "1"],
        "features": [{"name": "amount", "kind": "numeric", "dtype": "float64"}],
    }
    malformed_sig = {
        "inputs": [{"name": "x", "type": "double"}]  # missing dataset_name, target, class_labels
    }

    v_valid_1 = ModelVersion(
        name="valid_model_1",
        version=1,
        stage=ModelStage.PRODUCTION,
        status=ModelStatus.READY,
        artifact_uri="file:///test/1",
        signature=valid_sig,
        metrics={"roc_auc": 0.88},
    )
    v_valid_2 = ModelVersion(
        name="valid_model_2",
        version=1,
        stage=ModelStage.STAGING,
        status=ModelStatus.READY,
        artifact_uri="file:///test/2",
        signature=valid_sig,
        metrics={"roc_auc": 0.92},
    )
    v_malformed = ModelVersion(
        name="malformed_model",
        version=1,
        stage=ModelStage.DEVELOPMENT,
        status=ModelStatus.READY,
        artifact_uri="file:///test/3",
        signature=malformed_sig,
    )

    models_list = [
        {"name": "valid_model_1", "versions": 1, "updated_at": "2026-10-10T12:00:00Z"},
        {"name": "malformed_model", "versions": 1, "updated_at": "2026-10-10T12:00:00Z"},
        {"name": "valid_model_2", "versions": 1, "updated_at": "2026-10-10T12:00:00Z"},
    ]

    mock_registry = MagicMock()
    mock_registry.list_models.return_value = models_list

    def get_serving_mock(name):
        if name == "valid_model_1":
            return v_valid_1
        return None

    def get_latest_mock(name):
        if name == "valid_model_2":
            return v_valid_2
        if name == "malformed_model":
            return v_malformed
        return v_valid_1

    mock_registry.get_serving.side_effect = get_serving_mock
    mock_registry.get_latest.side_effect = get_latest_mock

    monkeypatch.setattr("app.registry.factory.get_registry", lambda: mock_registry)

    result = _models_section()
    assert result["available"] is True
    assert len(result["models"]) == 3

    names = [m["name"] for m in result["models"]]
    assert names == ["valid_model_1", "malformed_model", "valid_model_2"]

    # Valid models have target populated
    m1 = next(m for m in result["models"] if m["name"] == "valid_model_1")
    assert m1["target"] == "loan_status"
    assert m1["roc_auc"] == 0.88

    # Malformed model still appears, but with target=None
    m_bad = next(m for m in result["models"] if m["name"] == "malformed_model")
    assert m_bad["target"] is None


def test_api_upload_signature_handling(api_client):
    """Test 6: Pretrained model upload with external signature format."""
    import joblib
    import io
    from sklearn.linear_model import LogisticRegression
    import numpy as np

    model = LogisticRegression()
    model.fit(np.array([[1.0], [2.0], [3.0], [4.0]]), np.array([0, 0, 1, 1]))
    buf = io.BytesIO()
    joblib.dump(model, buf)
    artifact_bytes = buf.getvalue()

    external_sig = json.dumps({
        "dataset_name": "api_test_data",
        "target": "target_col",
        "class_labels": ["0", "1"],
        "inputs": [{"name": "feature_x", "type": "double"}]
    })

    res = api_client.post(
        "/api/v1/models/test_upload_external_sig/upload",
        content=artifact_bytes,
        headers={
            "Content-Type": "application/octet-stream",
            "X-File-Name": "model.joblib",
            "X-Model-Signature": external_sig,
        },
    )
    assert res.status_code == 200
    data = res.json()
    assert data["name"] == "test_upload_external_sig"
    assert data["signature"] is not None
    assert data["signature"]["target"] == "target_col"
    assert data["signature"]["features"][0]["name"] == "feature_x"
