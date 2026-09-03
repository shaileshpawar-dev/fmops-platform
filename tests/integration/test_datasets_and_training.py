"""Dataset upload/validation/preview and API-initiated training runs.

These exercise the real endpoints against the real app: a CSV goes in, the
existing validation engine judges it, and the existing training pipeline
produces a real model. Nothing here stubs the pipeline -- if the gate logic
changed, these would notice.
"""

from __future__ import annotations

import io

import pytest

CSV_HEADERS = {"Content-Type": "text/csv"}


def _csv_bytes(frame) -> bytes:
    buf = io.StringIO()
    frame.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")


# --------------------------------------------------------------------------- #
# Upload
# --------------------------------------------------------------------------- #
def test_upload_registers_a_version_and_validates_it(api_client, valid_frame):
    response = api_client.post(
        "/api/v1/datasets/upload?filename=demo.csv&description=integration",
        content=_csv_bytes(valid_frame),
        headers=CSV_HEADERS,
    )
    assert response.status_code == 201

    body = response.json()
    assert body["rows"] == len(valid_frame)
    assert body["columns"] == len(valid_frame.columns)
    assert body["version"]

    validation = body["validation"]
    assert validation is not None
    assert validation["expectations"] > 0
    # Counts must reconcile: a summary that does not add up is worse than none.
    assert validation["succeeded"] + validation["failed"] == validation["expectations"]


def test_uploading_identical_bytes_reuses_the_same_version(api_client, valid_frame):
    """Versions are content, not events."""
    payload = _csv_bytes(valid_frame)
    first = api_client.post(
        "/api/v1/datasets/upload?filename=same.csv", content=payload, headers=CSV_HEADERS
    )
    second = api_client.post(
        "/api/v1/datasets/upload?filename=same.csv", content=payload, headers=CSV_HEADERS
    )
    assert first.status_code == second.status_code == 201
    assert first.json()["version"] == second.json()["version"]


@pytest.mark.parametrize(
    ("name", "content", "reason"),
    [
        ("evil.exe", b"MZ\x90\x00", "executable extension"),
        ("archive.zip", b"PK\x03\x04", "archive extension"),
        ("empty.csv", b"   \n", "empty body"),
        ("broken.csv", b'a,b\n"unterminated,1\n', "malformed csv"),
    ],
)
def test_upload_rejects_unsafe_or_unusable_files(api_client, name, content, reason):
    response = api_client.post(
        f"/api/v1/datasets/upload?filename={name}", content=content, headers=CSV_HEADERS
    )
    assert response.status_code == 422, reason
    assert response.json()["error"]["code"] in {
        "dataset_upload_rejected",
        "request_validation_failed",
    }


def test_upload_rejects_a_file_over_the_size_limit(api_client):
    from app.api.routes.datasets import MAX_UPLOAD_BYTES

    oversized = b"a,b\n" + b"1,2\n" * ((MAX_UPLOAD_BYTES // 4) + 10)
    response = api_client.post(
        "/api/v1/datasets/upload?filename=big.csv", content=oversized, headers=CSV_HEADERS
    )
    assert response.status_code == 422
    assert "limit" in response.json()["error"]["message"].lower()


def test_a_traversal_filename_cannot_escape_the_data_directory(api_client, valid_frame):
    """The filename is neutralised, not trusted.

    The upload is accepted -- the bytes are fine -- but the name must not be
    able to choose a path. Asserted on the sanitiser directly as well, because
    a 201 alone would not prove where the file landed.
    """
    from app.api.routes.datasets import _safe_stem

    for hostile in ("../../../etc/passwd.csv", r"C:\windows\system32\evil.csv", "....//x.csv"):
        stem = _safe_stem(hostile)
        assert "/" not in stem and "\\" not in stem and ".." not in stem

    response = api_client.post(
        "/api/v1/datasets/upload?filename=../../../etc/passwd.csv",
        content=_csv_bytes(valid_frame),
        headers=CSV_HEADERS,
    )
    assert response.status_code == 201


# --------------------------------------------------------------------------- #
# Validation and preview
# --------------------------------------------------------------------------- #
def test_validation_endpoint_uses_the_real_engine(api_client, valid_frame):
    upload = api_client.post(
        "/api/v1/datasets/upload?filename=v.csv&validate=false",
        content=_csv_bytes(valid_frame),
        headers=CSV_HEADERS,
    )
    version = upload.json()["version"]
    assert upload.json()["validation"] is None  # not asked for

    report = api_client.get(f"/api/v1/datasets/{version}/validation")
    assert report.status_code == 200
    body = report.json()
    assert body["dataset_version"] == version
    assert body["expectations"] > 0
    assert body["engine"]


def test_preview_is_bounded(api_client, valid_frame):
    upload = api_client.post(
        "/api/v1/datasets/upload?filename=p.csv",
        content=_csv_bytes(valid_frame),
        headers=CSV_HEADERS,
    )
    version = upload.json()["version"]

    preview = api_client.get(f"/api/v1/datasets/{version}/preview?rows=5")
    assert preview.status_code == 200
    body = preview.json()
    assert len(body["sample"]) == 5
    assert len(body["column_profile"]) == len(valid_frame.columns)
    assert body["rows"] == len(valid_frame)  # full size still reported

    # The cap is enforced by the schema, not by trimming silently.
    assert api_client.get(f"/api/v1/datasets/{version}/preview?rows=10000").status_code == 422


def test_unknown_dataset_version_is_404(api_client):
    assert api_client.get("/api/v1/datasets/no-such-version/preview").status_code == 404
    assert api_client.get("/api/v1/datasets/no-such-version/validation").status_code == 404


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def test_training_rejects_unknown_algorithm_and_dataset(api_client):
    bad_algo = api_client.post("/api/v1/training/runs", json={"algorithm": "not-an-algorithm"})
    assert bad_algo.status_code == 422
    assert bad_algo.json()["error"]["code"] == "unknown_algorithm"

    bad_data = api_client.post("/api/v1/training/runs", json={"dataset_version": "nope"})
    assert bad_data.status_code == 404


def test_training_run_is_accepted_and_recorded(api_client, valid_frame):
    """202 with a pollable id, and the run reaches a terminal state.

    The TestClient runs background tasks before the response context closes, so
    by the time the poll happens the pipeline has actually executed.
    """
    upload = api_client.post(
        "/api/v1/datasets/upload?filename=train.csv",
        content=_csv_bytes(valid_frame),
        headers=CSV_HEADERS,
    )
    version = upload.json()["version"]

    started = api_client.post(
        "/api/v1/training/runs",
        json={"dataset_version": version, "algorithm": "logistic_regression"},
    )
    assert started.status_code == 202
    body = started.json()
    run_id = body["run_id"]
    assert body["status"] == "queued"
    assert started.headers["Location"].endswith(run_id)

    run = api_client.get(f"/api/v1/training/runs/{run_id}")
    assert run.status_code == 200
    record = run.json()
    assert record["status"] in {"completed", "rejected", "failed"}
    assert record["dataset_version"] == version
    assert record["algorithm"] == "logistic_regression"

    if record["status"] == "completed":
        assert record["metrics"]["roc_auc"] > 0.5
        assert record["duration_seconds"] is not None
        assert record["model_version"] is not None

    listing = api_client.get("/api/v1/training/runs")
    assert listing.status_code == 200
    assert any(r["run_id"] == run_id for r in listing.json()["runs"])


def test_unknown_training_run_is_404(api_client):
    response = api_client.get("/api/v1/training/runs/train-does-not-exist")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "training_run_not_found"


def test_training_request_does_not_accept_arbitrary_parameters(api_client):
    """The request surface is a closed set.

    A free-form params passthrough would be a way to reach code the API never
    meant to expose, so extra fields must be refused rather than ignored.
    """
    response = api_client.post(
        "/api/v1/training/runs",
        json={"algorithm": "logistic_regression", "params": {"__import__": "os"}},
    )
    assert response.status_code == 422


def test_orphaned_runs_are_reconciled_not_left_running(api_client):
    """A run the process was executing when it died must not look alive."""
    from app.training.jobs import RUNNING, get_training_run_store

    store = get_training_run_store()
    run = store.create(
        dataset_version=None,
        algorithm="logistic_regression",
        tune=False,
        promote=False,
        target_stage="Staging",
    )
    store.update(run.id, status=RUNNING)

    assert store.reconcile_orphans() >= 1
    recovered = store.get(run.id)
    assert recovered is not None
    assert recovered.status == "failed"
    assert "restart" in (recovered.error or "")
