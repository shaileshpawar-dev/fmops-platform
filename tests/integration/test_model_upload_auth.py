"""Regression tests for pretrained model upload authorization and security controls.

Tests cover production mode (allow_model_upload=False with model_upload_token),
header validation, token leak prevention, non-ASCII input safety, and evaluation metrics isolation.
"""

from __future__ import annotations

import json
import joblib
import pytest
from sklearn.linear_model import LogisticRegression

pytestmark = [pytest.mark.integration]


@pytest.fixture
def dummy_model_bytes(tmp_path):
    """Serialize a small scikit-learn classifier for upload tests."""
    model = LogisticRegression()
    path = tmp_path / "small_model.joblib"
    joblib.dump(model, path)
    return path.read_bytes()


def test_upload_without_token_rejected_in_production(api_client, settings, monkeypatch, dummy_model_bytes):
    """1. Upload without a token is rejected when uploads require a token."""
    monkeypatch.setattr(settings.security, "allow_model_upload", False)
    monkeypatch.setattr(settings.security, "model_upload_token", "demo_secret_token_123")

    response = api_client.post(
        "/api/v1/models/auth_test_model/upload",
        content=dummy_model_bytes,
        headers={"X-File-Name": "small_model.joblib"},
    )
    assert response.status_code == 403
    assert "Pretrained model upload is disabled in production" in response.json()["error"]["message"]


def test_upload_with_incorrect_token_rejected(api_client, settings, monkeypatch, dummy_model_bytes):
    """2. Incorrect token is rejected with 403."""
    monkeypatch.setattr(settings.security, "allow_model_upload", False)
    monkeypatch.setattr(settings.security, "model_upload_token", "demo_secret_token_123")

    response = api_client.post(
        "/api/v1/models/auth_test_model/upload",
        content=dummy_model_bytes,
        headers={
            "X-File-Name": "small_model.joblib",
            "X-Upload-Token": "completely_wrong_token",
        },
    )
    assert response.status_code == 403
    assert "Pretrained model upload is disabled in production" in response.json()["error"]["message"]


def test_upload_with_correct_token_octet_stream(api_client, settings, monkeypatch, dummy_model_bytes):
    """3. Correct X-Upload-Token permits an authorized octet-stream upload."""
    monkeypatch.setattr(settings.security, "allow_model_upload", False)
    monkeypatch.setattr(settings.security, "model_upload_token", "demo_secret_token_123")

    sig = {
        "inputs": [{"name": "x1", "type": "double"}],
        "outputs": [{"name": "target", "type": "integer"}],
    }

    response = api_client.post(
        "/api/v1/models/auth_octet_model/upload",
        content=dummy_model_bytes,
        headers={
            "X-File-Name": "small_model.joblib",
            "X-Upload-Token": "demo_secret_token_123",
            "X-Model-Signature": json.dumps(sig),
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "auth_octet_model"
    assert data["algorithm"] == "uploaded"
    assert data["version"] == 1


def test_upload_with_correct_token_multipart(api_client, settings, monkeypatch, dummy_model_bytes):
    """4. Correct token permits an authorized multipart upload."""
    monkeypatch.setattr(settings.security, "allow_model_upload", False)
    monkeypatch.setattr(settings.security, "model_upload_token", "demo_secret_token_123")

    response = api_client.post(
        "/api/v1/models/auth_multipart_model/upload",
        files={"file": ("small_model.joblib", dummy_model_bytes, "application/octet-stream")},
        headers={"X-Upload-Token": "demo_secret_token_123"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "auth_multipart_model"
    assert data["algorithm"] == "uploaded"


def test_upload_succeeds_without_token_when_allow_model_upload_true(api_client, settings, monkeypatch, dummy_model_bytes):
    """5. Upload succeeds without a token only when allow_model_upload permits it."""
    monkeypatch.setattr(settings.security, "allow_model_upload", True)
    monkeypatch.setattr(settings.security, "model_upload_token", "demo_secret_token_123")

    response = api_client.post(
        "/api/v1/models/open_upload_model/upload",
        content=dummy_model_bytes,
        headers={"X-File-Name": "small_model.joblib"},
    )
    assert response.status_code == 200
    assert response.json()["name"] == "open_upload_model"


def test_success_response_does_not_echo_token(api_client, settings, monkeypatch, dummy_model_bytes):
    """6. No response echoes the supplied token."""
    secret = "very_sensitive_custom_token_987654"
    monkeypatch.setattr(settings.security, "allow_model_upload", False)
    monkeypatch.setattr(settings.security, "model_upload_token", secret)

    response = api_client.post(
        "/api/v1/models/echo_check_model/upload",
        content=dummy_model_bytes,
        headers={
            "X-File-Name": "small_model.joblib",
            "X-Upload-Token": secret,
        },
    )
    assert response.status_code == 200
    assert secret not in response.text
    for hname, hval in response.headers.items():
        assert secret not in hval


def test_error_response_does_not_echo_token(api_client, settings, monkeypatch, dummy_model_bytes):
    """7. Error responses do not echo the supplied token."""
    supplied_token = "attempted_attacker_token_xyz999"
    monkeypatch.setattr(settings.security, "allow_model_upload", False)
    monkeypatch.setattr(settings.security, "model_upload_token", "server_configured_token_888")

    response = api_client.post(
        "/api/v1/models/error_echo_model/upload",
        content=dummy_model_bytes,
        headers={
            "X-File-Name": "small_model.joblib",
            "X-Upload-Token": supplied_token,
        },
    )
    assert response.status_code == 403
    assert supplied_token not in response.text
    assert "server_configured_token_888" not in response.text
    for hname, hval in response.headers.items():
        assert supplied_token not in hval
        assert "server_configured_token_888" not in hval


def test_invalid_and_non_ascii_token_does_not_produce_http_500(api_client, settings, monkeypatch, dummy_model_bytes):
    """8. Invalid token handling (including non-ASCII / unicode) does not produce HTTP 500."""
    monkeypatch.setattr(settings.security, "allow_model_upload", False)
    monkeypatch.setattr(settings.security, "model_upload_token", "standard_token_123")

    # Non-ASCII unicode token that would cause TypeError in naive compare_digest
    unicode_token_bytes = "unicode_token_\u2603_\u2728".encode("utf-8")
    response = api_client.post(
        "/api/v1/models/unicode_token_model/upload",
        content=dummy_model_bytes,
        headers={
            "X-File-Name": "small_model.joblib",
            "X-Upload-Token": unicode_token_bytes,
        },
    )
    assert response.status_code == 403
    assert response.status_code != 500


def test_uploaded_model_metrics_unavailable_until_verified(api_client, settings, monkeypatch, dummy_model_bytes):
    """9. Uploaded model metrics remain unavailable until verified evaluation takes place."""
    monkeypatch.setattr(settings.security, "allow_model_upload", True)

    client_claimed_metrics = {"f1": 0.99, "roc_auc": 0.99, "accuracy": 0.99}
    response = api_client.post(
        "/api/v1/models/governed_upload_model/upload",
        content=dummy_model_bytes,
        headers={
            "X-File-Name": "small_model.joblib",
            "X-Model-Metrics": json.dumps(client_claimed_metrics),
        },
    )
    assert response.status_code == 200
    model_data = response.json()

    # The actual evaluation metrics on the registered model must be empty
    assert model_data["metrics"] == {}
    # Client claimed metrics must be quarantined inside params
    assert model_data["params"]["client_metrics"] == client_claimed_metrics

    # Previewing gate must show evaluation metrics as unavailable / not passed
    preview = api_client.post(f"/api/v1/models/{model_data['name']}/versions/{model_data['version']}/evaluate-gate").json()
    assert preview["approval"]["decision"] == "rejected"


def test_config_endpoint_redacts_model_upload_token(api_client, settings, monkeypatch):
    """Meta-check: Effective config endpoint must redact model_upload_token."""
    secret = "secret_never_leak_token_456"
    monkeypatch.setattr(settings.security, "model_upload_token", secret)

    response = api_client.get("/api/v1/config")
    assert response.status_code == 200
    config = response.json()
    assert config["security"]["model_upload_token"] == "***redacted***"
    assert secret not in response.text
