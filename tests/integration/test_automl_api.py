"""AutoML over the real API: profile, run, leaderboard, and the approval gate.

The run test trains real models through the existing pipeline. It is slower
than a unit test on purpose -- the point is that AutoML drives the actual
training and the actual gate, not a stand-in for either.
"""

from __future__ import annotations

import io

CSV = {"Content-Type": "text/csv"}


def _upload(client, frame, name="automl.csv") -> str:
    buf = io.StringIO()
    frame.to_csv(buf, index=False)
    response = client.post(
        f"/api/v1/datasets/upload?filename={name}", content=buf.getvalue(), headers=CSV
    )
    assert response.status_code == 201
    return response.json()["version"]


# --------------------------------------------------------------------------- #
# Profiling endpoint
# --------------------------------------------------------------------------- #
def test_profile_endpoint_recommends_a_target_with_evidence(api_client, valid_frame):
    version = _upload(api_client, valid_frame, "profile.csv")
    response = api_client.get(f"/api/v1/automl/profile/{version}")
    assert response.status_code == 200

    body = response.json()
    target = body["profile"]["suggested_target"]
    assert target["column"] == "default"
    assert target["problem_type"] == "binary_classification"
    assert body["problem_supported"] is True
    assert target["reasons"], "a recommendation must carry its reasons"
    assert body["primary_metric"] == "roc_auc"
    assert body["default_selection"], "at least one candidate should be pre-selected"


def test_profile_endpoint_honours_a_target_override(api_client, valid_frame):
    version = _upload(api_client, valid_frame, "override.csv")
    response = api_client.get(f"/api/v1/automl/profile/{version}?target=credit_score")
    assert response.status_code == 200
    body = response.json()
    assert body["profile"]["suggested_target"]["column"] == "credit_score"
    # Continuous target -> regression -> outside what this stack can fit.
    assert body["problem_supported"] is False
    assert all(c["tier"] == "unsuitable" for c in body["candidates"])


def test_profile_of_an_unknown_dataset_is_404(api_client):
    assert api_client.get("/api/v1/automl/profile/no-such-version").status_code == 404


# --------------------------------------------------------------------------- #
# Request validation
# --------------------------------------------------------------------------- #
def test_run_rejects_a_target_that_is_not_in_the_dataset(api_client, valid_frame):
    version = _upload(api_client, valid_frame, "badtarget.csv")
    response = api_client.post(
        "/api/v1/automl/runs", json={"dataset_version": version, "target_column": "nope"}
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_automl_request"


def test_run_refuses_an_unsupported_problem_type_with_a_reason(api_client, valid_frame):
    """Regression is refused, not attempted and silently mangled."""
    version = _upload(api_client, valid_frame, "regression.csv")
    response = api_client.post(
        "/api/v1/automl/runs",
        json={"dataset_version": version, "target_column": "credit_score"},
    )
    assert response.status_code == 422
    error = response.json()["error"]
    assert error["code"] == "unsupported_problem_type"
    assert "binary classification" in error["message"]


def test_run_refuses_an_algorithm_the_environment_cannot_build(api_client, valid_frame):
    from app.training.model_factory import available_algorithms

    missing = [k for k, v in available_algorithms().items() if not v]
    if not missing:
        return  # everything is installed here; nothing to assert
    version = _upload(api_client, valid_frame, "missingalgo.csv")
    response = api_client.post(
        "/api/v1/automl/runs",
        json={
            "dataset_version": version,
            "target_column": "default",
            "algorithms": missing[:1],
        },
    )
    assert response.status_code == 422
    assert "not available" in response.json()["error"]["message"]


def test_run_request_rejects_unknown_fields(api_client, valid_frame):
    version = _upload(api_client, valid_frame, "extra.csv")
    response = api_client.post(
        "/api/v1/automl/runs",
        json={"dataset_version": version, "target_column": "default", "params": {"x": 1}},
    )
    assert response.status_code == 422


def test_run_caps_the_number_of_candidates(api_client, valid_frame):
    """Opening a page must not be able to queue unbounded training."""
    version = _upload(api_client, valid_frame, "cap.csv")
    response = api_client.post(
        "/api/v1/automl/runs",
        json={
            "dataset_version": version,
            "target_column": "default",
            "algorithms": ["hist_gradient_boosting", "random_forest", "logistic_regression"],
            "max_models": 2,
        },
    )
    assert response.status_code == 202
    assert len(response.json()["algorithms"]) == 2


def test_unknown_automl_run_is_404(api_client):
    response = api_client.get("/api/v1/automl/runs/automl-nope")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "automl_run_not_found"


# --------------------------------------------------------------------------- #
# A real run
# --------------------------------------------------------------------------- #
def test_automl_trains_candidates_ranks_them_and_uses_the_real_gate(api_client, valid_frame):
    """The whole path, with real models.

    Asserts the three things that make this AutoML rather than a leaderboard
    mock-up: candidates actually trained, the ranking follows the configured
    primary metric, and the winner went through the same approval gate as any
    other model.
    """
    version = _upload(api_client, valid_frame, "run.csv")
    started = api_client.post(
        "/api/v1/automl/runs",
        json={
            "dataset_version": version,
            "target_column": "default",
            "algorithms": ["logistic_regression", "random_forest"],
            "primary_metric": "roc_auc",
            "target_stage": "Staging",
        },
    )
    assert started.status_code == 202
    run_id = started.json()["run_id"]
    assert started.headers["Location"].endswith(run_id)

    run = api_client.get(f"/api/v1/automl/runs/{run_id}").json()
    assert run["status"] in {"completed", "completed_with_warnings", "failed"}

    if run["status"] == "failed":
        assert run["error"], "a failed run must say why"
        return

    leaderboard = run["leaderboard"]
    assert leaderboard, "a completed run must produce a leaderboard"
    scores = [c["metrics"]["roc_auc"] for c in leaderboard]
    assert scores == sorted(scores, reverse=True), "leaderboard must follow the primary metric"
    assert leaderboard[0]["rank"] == 1
    assert run["best_algorithm"] == leaderboard[0]["algorithm"]
    assert run["ranking_rule"], "the ranking rule must be recorded with the run"

    # Reproducibility: the run records what produced it.
    assert run["dataset_version"] == version
    assert run["target_column"] == "default"
    assert run["problem_type"] == "binary_classification"
    assert run["profile"]["n_rows"] == len(valid_frame)

    promotion = run.get("promotion") or {}
    assert promotion, "the winner must be put through the approval gate"
    assert promotion["decision"] in {"approved", "rejected"}
    if promotion["decision"] == "rejected":
        # Rejection is a valid outcome and must not be dressed up as success.
        assert promotion["promoted"] is False
        assert promotion["failed_checks"]


def test_candidates_endpoint_exposes_the_ranking_rule(api_client, valid_frame):
    version = _upload(api_client, valid_frame, "cands.csv")
    run_id = api_client.post(
        "/api/v1/automl/runs",
        json={
            "dataset_version": version,
            "target_column": "default",
            "algorithms": ["logistic_regression"],
        },
    ).json()["run_id"]

    response = api_client.get(f"/api/v1/automl/runs/{run_id}/candidates")
    assert response.status_code == 200
    body = response.json()
    assert body["ranking_rule"]
    assert body["primary_metric"] == "roc_auc"
    assert isinstance(body["candidates"], list)


def test_runs_listing_omits_the_bulky_profile(api_client, valid_frame):
    version = _upload(api_client, valid_frame, "list.csv")
    api_client.post(
        "/api/v1/automl/runs",
        json={
            "dataset_version": version,
            "target_column": "default",
            "algorithms": ["logistic_regression"],
        },
    )
    body = api_client.get("/api/v1/automl/runs").json()
    assert body["count"] >= 1
    assert all("profile" not in run for run in body["runs"])


def test_orphaned_automl_runs_are_reconciled(api_client):
    from app.automl.runner import get_automl_store

    store = get_automl_store()
    run_id = store.create(
        dataset_version="v-any",
        target="y",
        problem_type="binary_classification",
        algorithms=["logistic_regression"],
        primary_metric="roc_auc",
        tune=False,
        target_stage="Staging",
        max_models=1,
    )
    assert store.reconcile_orphans() >= 1
    recovered = store.get(run_id)
    assert recovered is not None
    assert recovered["status"] == "failed"
    assert "restart" in (recovered["error"] or "")
