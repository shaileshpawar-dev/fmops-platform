"""Acceptance: a user's own dataset, driven through the whole lifecycle over HTTP.

This is the product claim, tested as a user would exercise it. The dataset is a
customer-churn table the platform has never seen -- not the bundled loan data --
with a named target ("yes"/"no"), an identifier column and mixed feature types.
Every step goes through the public API and asserts on persisted state:

     1 upload             8 gate preview           15 feedback (own labels)
     2 validate           9 gate decision record   16 retraining trigger
     3 profile           10 approval               17 retraining job
     4 train (job)       11 deploy (job)           18 candidate vs production
     5 job completion    12 predict + contract     19 promote or reject
     6 metrics           13 monitoring             20 production version
     7 registry          14 drift                  21 rollback
                                                      + lineage, both ways

Where the outcome is a real decision (did the retrained candidate win?), the
test accepts either honest answer and asserts that the platform's state is
consistent with the one it gave. It never forces a result.
"""

from __future__ import annotations

import io

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.pipeline

MODEL = "churn_model"
CSV = {"Content-Type": "text/csv"}


def churn_frame(
    n: int, seed: int, spend_shift: float = 0.0, changed_relationship: bool = False
) -> pd.DataFrame:
    """Synthetic churn data for the test -- generated here, in the test, only.

    ``changed_relationship`` flips what drives churn (long-tenure customers on
    the premium plan now leave), which is the situation retraining exists for.
    """
    rng = np.random.default_rng(seed)
    tenure = rng.integers(1, 72, n)
    spend = rng.normal(60 + spend_shift, 20, n).round(2)
    plan = rng.choice(["basic", "pro", "team"], n, p=[0.5, 0.35, 0.15])
    tickets = rng.poisson(1.2, n)
    if changed_relationship:
        logit = -2.6 + 0.05 * tenure + (plan == "team") * 1.8 + 0.35 * tickets
    else:
        logit = (
            0.4
            - 0.05 * tenure
            + 0.03 * (spend - 60)
            + (plan == "basic") * 1.1
            + 0.35 * tickets
        )
    churned = np.where(rng.random(n) < 1 / (1 + np.exp(-logit)), "yes", "no")
    return pd.DataFrame(
        {
            "customer_id": [f"s{seed}-{i:05d}" for i in range(n)],
            "tenure_months": tenure,
            "monthly_spend": spend,
            "plan": plan,
            "support_tickets": tickets,
            "churned": churned,
        }
    )


@pytest.fixture
def isolated_datasets(settings, tmp_path, monkeypatch):
    """A dataset registry of this test's own, so nothing leaks between tests."""
    import app.data.versioning as versioning

    registry = versioning.DatasetRegistry(settings)
    registry.manifest_path = tmp_path / "versions.json"
    monkeypatch.setattr(versioning, "_REGISTRY", registry)
    return registry


def _ok(response, *codes):
    assert response.status_code in (codes or (200,)), (response.status_code, response.text)
    return response.json()


def _job(client, job_id):
    job = _ok(client.get(f"/api/v1/jobs/{job_id}"))
    assert job["status"] == "succeeded", job
    return job


def _score(client, frame):
    """Score rows through the public endpoint; returns [(request_id, true label)]."""
    served = []
    for record in frame.to_dict(orient="records"):
        truth = record.pop("churned")
        body = _ok(client.post(f"/api/v1/models/{MODEL}/predict", json={"features": record}))
        served.append((body["request_id"], truth))
    return served


def test_a_users_own_model_through_the_whole_lifecycle(
    api_client, isolated_datasets, settings
):
    client = api_client

    # ---- 1. upload ------------------------------------------------------------ #
    upload = _ok(
        client.post(
            "/api/v1/datasets/upload?filename=churn.csv",
            content=churn_frame(1500, seed=1).to_csv(index=False),
            headers=CSV,
        ),
        201,
    )
    dataset = upload["version"]
    assert upload["dataset_name"] == "churn", "an upload must be named after its file"
    assert upload["validation"]["passed"], upload["validation"]

    # ---- 2. validate against the chosen target -------------------------------- #
    validation = _ok(client.get(f"/api/v1/datasets/{dataset}/validation?target=churned"))
    assert validation["passed"], validation["failures"]
    bad = _ok(client.get(f"/api/v1/datasets/{dataset}/validation?target=plan"))
    assert not bad["passed"], "a three-class column must fail the binary-target check"

    # ---- 3. profile ------------------------------------------------------------- #
    profile = _ok(client.get(f"/api/v1/automl/profile/{dataset}?target=churned"))
    assert profile["problem_type"] == "binary_classification"
    assert profile["problem_supported"]

    # ---- 4/5. train as a job, and follow it to completion ----------------------- #
    started = _ok(
        client.post(
            "/api/v1/automl/runs",
            json={
                "dataset_version": dataset,
                "target_column": "churned",
                "model_name": MODEL,
                "algorithms": ["logistic_regression", "random_forest"],
                "target_stage": "Staging",
            },
        ),
        202,
    )
    assert started["model_name"] == MODEL
    job = _job(client, started["job_id"])
    assert job["resource_id"] == started["run_id"]
    logs = _ok(client.get(f"/api/v1/jobs/{started['job_id']}/logs"))
    assert logs["lines"], "a finished job must have captured its logs"
    run = _ok(client.get(f"/api/v1/automl/runs/{started['run_id']}"))
    assert run["status"] in ("completed", "completed_with_warnings"), run
    v1 = run["best_model_version"]

    # ---- 6. metrics -------------------------------------------------------------- #
    ranked = [c for c in run["candidates"] if c["status"] == "completed"]
    assert ranked and all("roc_auc" in c["metrics"] for c in ranked)

    # ---- 7. registry: its own name, its own contract ------------------------------- #
    models = {m["name"]: m for m in _ok(client.get("/api/v1/models"))["models"]}
    assert MODEL in models
    assert models[MODEL]["target"] == "churned"
    assert models[MODEL]["endpoint"] == "fmops-churn-model"
    reference = settings.tracking.registered_model_name
    assert reference not in models or models[reference]["target"] != "churned"
    signature = _ok(client.get(f"/api/v1/models/{MODEL}/signature?version={v1}"))
    features = [f["name"] for f in signature["signature"]["features"]]
    assert "customer_id" not in features, "an identifier must not become a feature"
    assert signature["signature"]["class_labels"] == ["no", "yes"]

    # ---- 8. gate preview ------------------------------------------------------------ #
    preview = _ok(client.post(f"/api/v1/models/{MODEL}/versions/{v1}/evaluate-gate"))
    assert preview["approval"]["checks"]

    # ---- 9. the gate's decision is a record ---------------------------------------- #
    decisions = _ok(client.get(f"/api/v1/models/{MODEL}/decisions?version={v1}"))["decisions"]
    assert decisions and decisions[-1]["source"] == "pipeline"
    assert decisions[-1]["checks"], "the decision must say which checks ran"

    # ---- 10. approval -- and the gate cannot be walked around ----------------------- #
    bypass = client.post(
        f"/api/v1/models/{MODEL}/versions/{v1}/stage", json={"stage": "Production"}
    )
    assert bypass.status_code == 409
    approved = _ok(
        client.post(
            f"/api/v1/models/{MODEL}/versions/{v1}/approve",
            json={"target_stage": "Production", "comment": "acceptance run"},
        )
    )
    assert approved["model_version"]["stage"] == "Production"
    assert approved["decision"]["source"] == "manual"

    # ---- 11. deploy as a job ----------------------------------------------------------- #
    deploy = _ok(
        client.post(
            "/api/v1/deployments",
            json={"model_name": MODEL, "model_version": v1, "strategy": "blue_green"},
        ),
        202,
    )
    assert deploy["endpoint"] == "fmops-churn-model"
    _job(client, deploy["job"]["id"])
    live = _ok(client.get(f"/api/v1/deployments/current?model={MODEL}"))
    assert live["deployment"]["state"] == "live"
    assert live["versions"]["current"] == v1

    # ---- 12. predict, against the recorded contract ------------------------------------ #
    example = signature["example"]["features"]
    scored = _ok(client.post(f"/api/v1/models/{MODEL}/predict", json={"features": example}))
    assert scored["model_name"] == MODEL and scored["model_version"] == v1
    assert scored["prediction_label"] in ("no", "yes")
    assert scored["positive_label"] == "yes"
    assert 0.0 <= scored["probability"] <= 1.0 and scored["request_id"]
    missing = dict(example)
    missing.pop("plan")
    refused = client.post(f"/api/v1/models/{MODEL}/predict", json={"features": missing})
    assert refused.status_code == 422
    assert "missing field(s): plan" in refused.json()["error"]["message"]
    unusual = _ok(
        client.post(
            f"/api/v1/models/{MODEL}/predict",
            json={"features": {**example, "plan": "enterprise"}},
        )
    )
    assert any("not seen in training" in w for w in unusual["warnings"])

    # ---- 13. monitoring -------------------------------------------------------------------- #
    baseline_traffic = _score(client, churn_frame(250, seed=2).drop(columns=["customer_id"]))
    summary = _ok(client.get(f"/api/v1/monitoring/summary?model={MODEL}"))
    assert summary["model_name"] == MODEL
    assert summary["service"]["request_count"] >= 250

    # ---- 14. drift ------------------------------------------------------------------------- #
    drifted_traffic = _score(
        client, churn_frame(250, seed=3, spend_shift=45).drop(columns=["customer_id"])
    )
    drift = _ok(client.post(f"/api/v1/drift/scan?model={MODEL}"))
    assert drift["drift_detected"]
    assert "monthly_spend" in drift["drifted_features"]
    assert drift["concept_drift_status"] != "measured", "no labels yet: no concept drift"

    # ---- 15. feedback, in the model's own labels ------------------------------------------ #
    for request_id, truth in baseline_traffic[:100] + drifted_traffic[:150]:
        _ok(
            client.post(
                "/api/v1/feedback", json={"request_id": request_id, "actual_label": truth}
            )
        )
    unknown = client.post(
        "/api/v1/feedback", json={"request_id": "nope", "actual_label": "yes"}
    )
    assert unknown.status_code == 404
    wrong = client.post(
        "/api/v1/feedback",
        json={"request_id": baseline_traffic[0][0], "actual_label": "maybe"},
    )
    assert wrong.status_code == 400

    # ---- 16. the retraining trigger ---------------------------------------------------------- #
    trigger = _ok(client.get(f"/api/v1/retraining/trigger/evaluate?model={MODEL}"))
    assert trigger["should_retrain"], trigger
    assert trigger["trigger"] == "drift"

    # ---- 17. retraining runs as a job, on real data only ---------------------------------------- #
    retrain = _ok(client.post("/api/v1/retraining/run", json={"model_name": MODEL}), 202)
    retrain_job = _job(client, retrain["job"]["id"])
    decision = retrain_job["result"]
    event = _ok(client.get(f"/api/v1/retraining/{decision['event_id']}"))["event"]
    sources = event["detail"]["data_sources"]
    assert sources["labelled_production_rows"] == 250
    assert sources["base_dataset"] == dataset
    # Every labelled production row must reach the training set. (A missing id
    # once made them all "duplicates" of each other and kept one of 250.)
    assert sources["duplicates_dropped"] == 0
    assert sources["new_rows"] == 250
    assert sources["rows"] == sources["base_rows"] + 250
    candidate = decision["candidate_version"]
    assert candidate and candidate != v1

    # ---- 18. candidate vs production is a recorded decision --------------------------------- #
    verdicts = _ok(client.get(f"/api/v1/models/{MODEL}/decisions?version={candidate}"))
    assert verdicts["decisions"], "the retraining verdict must be recorded"
    assert decision["comparison"]["baseline_version"] == v1
    # A fair exam: both models scored on the same rows, none of which either
    # trained on -- not each on its own test split.
    assert decision["comparison"]["basis"] == "shared_holdout", decision["comparison"]
    assert decision["comparison"]["holdout_rows"] >= 50

    # ---- 19. promoted or rejected -- and state agrees with the answer ------------------------- #
    now = _ok(client.get(f"/api/v1/deployments/current?model={MODEL}"))["versions"]["current"]
    if decision["deployed"]:
        assert now == candidate
    else:
        assert now == v1, "a candidate that was not deployed must not be serving"

        # ---- 20. a losing candidate has no back door into production ------------------------- #
        # The Production gate compares it with the live version again, and refuses.
        refused = client.post(
            f"/api/v1/models/{MODEL}/versions/{candidate}/approve",
            json={"target_stage": "Production", "comment": "acceptance: try to promote"},
        )
        assert refused.status_code == 409, refused.text
        # Staging only needs the thresholds -- and a Staging version cannot take
        # live traffic, so it cannot reach Production by being deployed either.
        staged = client.post(
            f"/api/v1/models/{MODEL}/versions/{candidate}/approve",
            json={"target_stage": "Staging", "comment": "acceptance: watch it in shadow"},
        )
        assert staged.status_code == 200, staged.text
        live_attempt = client.post(
            "/api/v1/deployments",
            json={"model_name": MODEL, "model_version": candidate, "strategy": "direct"},
        )
        assert live_attempt.status_code == 409, live_attempt.text
        assert live_attempt.json()["error"]["code"] == "deployment_refused"

        # It may be shadowed: scored on mirrored live traffic, serving nothing.
        shadow = _ok(
            client.post(
                "/api/v1/deployments",
                json={"model_name": MODEL, "model_version": candidate, "strategy": "shadow"},
            ),
            202,
        )
        assert _job(client, shadow["job"]["id"])["status"] == "succeeded"
        mirrored = _ok(client.get(f"/api/v1/deployments/current?model={MODEL}"))["versions"]
        assert mirrored["current"] == v1 and mirrored["shadow"] == candidate
        served = _ok(
            client.post(f"/api/v1/models/{MODEL}/predict", json={"features": example})
        )
        assert served["model_version"] == v1, "a shadow must never answer the caller"
        assert (
            _ok(client.get(f"/api/v1/models/{MODEL}/versions/{candidate}"))["stage"]
            == "Staging"
        )

    # ---- 21. rollback (when the candidate went live) ------------------------------------------ #
    # The rejected branch has nothing live to roll back from; rollback after an
    # automatic promotion is exercised end to end by the next test.
    if decision["deployed"]:
        serving = _ok(client.get(f"/api/v1/deployments/current?model={MODEL}"))
        assert serving["versions"]["previous"] == v1
        rolled = _ok(
            client.post(
                "/api/v1/deployments/rollback",
                json={"model_name": MODEL, "reason": "acceptance: roll back"},
            )
        )
        assert rolled["succeeded"] and rolled["rolled_back_to"] == v1
        back = _ok(client.post(f"/api/v1/models/{MODEL}/predict", json={"features": example}))
        assert back["model_version"] == v1, "after rollback the old version must serve"

    # ---- lineage: from the production model back to its data -------------------------------------- #
    lineage = _ok(client.get(f"/api/v1/models/{MODEL}/versions/{v1}/lineage"))
    assert lineage["dataset"]["version"] == dataset
    assert lineage["dataset"]["hash_matches_model"]
    assert lineage["training"]["kind"] == "automl"
    assert lineage["training"]["job"]["status"] == "succeeded"
    assert [d["source"] for d in lineage["gate_decisions"]][:2] == ["pipeline", "manual"]
    assert lineage["deployments"]
    assert lineage["serving"]["predictions"] >= 500 and lineage["serving"]["labelled"] == 250
    assert lineage["drift"] and lineage["drift"][0]["drift_detected"]
    assert lineage["retraining"]["triggered_from"]

    candidate_lineage = _ok(client.get(f"/api/v1/models/{MODEL}/versions/{candidate}/lineage"))
    assert candidate_lineage["retraining"]["produced_by"]["id"] == decision["event_id"]
    retrain_set = candidate_lineage["dataset"]["version"]
    assert retrain_set != dataset

    data_lineage = _ok(client.get(f"/api/v1/datasets/{dataset}/lineage"))
    assert {(m["name"], m["version"]) for m in data_lineage["models_trained"]} >= {(MODEL, v1)}
    assert retrain_set in [
        d["version"] for d in data_lineage["next_versions"]
    ], "the retraining set must be recorded as derived from the dataset it extended"

    # ---- the reference model was never touched --------------------------------------------------- #
    assert _ok(client.get(f"/api/v1/models/{reference}/versions")) == []


def test_a_dataset_version_is_immutable(api_client, isolated_datasets):
    """Two uploads with one filename, or two retraining sets, never share bytes."""
    first = _ok(
        api_client.post(
            "/api/v1/datasets/upload?filename=same.csv",
            content=churn_frame(300, seed=10).to_csv(index=False),
            headers=CSV,
        ),
        201,
    )
    second = _ok(
        api_client.post(
            "/api/v1/datasets/upload?filename=same.csv",
            content=churn_frame(300, seed=11).to_csv(index=False),
            headers=CSV,
        ),
        201,
    )
    assert first["version"] != second["version"]
    one = isolated_datasets.load(first["version"])
    assert one["customer_id"].iloc[0].startswith("s10-"), "the first version's bytes changed"
    raw = pd.read_csv(io.StringIO(churn_frame(300, seed=10).to_csv(index=False)))
    assert one.shape == raw.shape


def test_retraining_promotes_a_candidate_that_learned_a_changed_relationship(
    api_client, isolated_datasets
):
    """The other honest outcome: new data, a better model, an automatic rollout.

    The relationship between the features and churn changes. The user uploads
    fresh labelled data showing it and retrains with it. The candidate must win
    on a holdout neither model trained on, go live without a human (auto-deploy
    is on in this profile), and still be reversible.
    """
    client = api_client
    base = _ok(
        client.post(
            "/api/v1/datasets/upload?filename=churn.csv",
            content=churn_frame(1500, seed=21).to_csv(index=False),
            headers=CSV,
        ),
        201,
    )["version"]
    started = _ok(
        client.post(
            "/api/v1/automl/runs",
            json={
                "dataset_version": base,
                "target_column": "churned",
                "model_name": MODEL,
                "algorithms": ["logistic_regression"],
            },
        ),
        202,
    )
    _job(client, started["job_id"])
    v1 = _ok(client.get(f"/api/v1/automl/runs/{started['run_id']}"))["best_model_version"]
    _ok(
        client.post(
            f"/api/v1/models/{MODEL}/versions/{v1}/approve",
            json={"target_stage": "Production"},
        )
    )
    first = _ok(
        client.post(
            "/api/v1/deployments",
            json={"model_name": MODEL, "model_version": v1, "strategy": "direct"},
        ),
        202,
    )
    _job(client, first["job"]["id"])

    fresh = _ok(
        client.post(
            "/api/v1/datasets/upload?filename=churn-q3.csv",
            content=churn_frame(3000, seed=22, changed_relationship=True).to_csv(index=False),
            headers=CSV,
        ),
        201,
    )["version"]
    retrain = _ok(
        client.post(
            "/api/v1/retraining/run", json={"model_name": MODEL, "dataset_version": fresh}
        ),
        202,
    )
    decision = _job(client, retrain["job"]["id"])["result"]
    assert decision["trigger"] == "manual", "supplying new data is an operator's decision"
    assert decision["comparison"]["basis"] == "shared_holdout"
    assert decision["comparison"]["candidate_is_better"], decision["comparison"]
    assert decision["status"] == "succeeded" and decision["deployed"], decision["message"]
    candidate = decision["candidate_version"]

    event = _ok(client.get(f"/api/v1/retraining/{decision['event_id']}"))["event"]
    assert event["detail"]["data_sources"]["new_dataset"] == fresh
    assert event["detail"]["data_sources"]["new_dataset_rows"] == 3000

    live = _ok(client.get(f"/api/v1/deployments/current?model={MODEL}"))["versions"]
    assert live["current"] == candidate and live["previous"] == v1
    verdict = _ok(client.get(f"/api/v1/models/{MODEL}/decisions?version={candidate}"))
    assert verdict["decisions"][0]["decision"] == "approved"

    rolled = _ok(client.post("/api/v1/deployments/rollback", json={"model_name": MODEL}))
    assert rolled["rolled_back_to"] == v1
    assert (
        _ok(client.get(f"/api/v1/deployments/current?model={MODEL}"))["versions"]["current"]
        == v1
    )


def test_automation_scans_retrains_and_does_not_loop(api_client, isolated_datasets):
    """The scheduled pass, run on demand: drift -> labels -> retraining, once.

    * with fresh traffic it queues a drift scan; with none, it does not repeat one
    * drift without labels does not start a retrain (nothing new to learn from)
    * once labels arrive it queues exactly one retraining job
    * straight after, the cooldown-free test profile still does not retrain
      again, because that job consumed the labels it would have learned from
    """
    client = api_client
    base = _ok(
        client.post(
            "/api/v1/datasets/upload?filename=churn.csv",
            content=churn_frame(1200, seed=31).to_csv(index=False),
            headers=CSV,
        ),
        201,
    )["version"]
    started = _ok(
        client.post(
            "/api/v1/automl/runs",
            json={
                "dataset_version": base,
                "target_column": "churned",
                "model_name": MODEL,
                "algorithms": ["logistic_regression"],
            },
        ),
        202,
    )
    _job(client, started["job_id"])
    v1 = _ok(client.get(f"/api/v1/automl/runs/{started['run_id']}"))["best_model_version"]
    _ok(
        client.post(
            f"/api/v1/models/{MODEL}/versions/{v1}/approve",
            json={"target_stage": "Production"},
        )
    )
    _job(
        client,
        _ok(
            client.post(
                "/api/v1/deployments",
                json={"model_name": MODEL, "model_version": v1, "strategy": "direct"},
            ),
            202,
        )["job"]["id"],
    )

    assert _ok(client.post("/api/v1/automation/run"))["drift_scans"] == [], "no traffic yet"

    served = _score(
        client, churn_frame(300, seed=32, spend_shift=45).drop(columns=["customer_id"])
    )
    first = _ok(client.post("/api/v1/automation/run"))
    assert len(first["drift_scans"]) == 1
    scan = _job(client, first["drift_scans"][0])["result"]
    assert scan["drift_detected"]
    assert first["retraining"] == [], "drift without labels must not retrain"

    again = _ok(client.post("/api/v1/automation/run"))
    assert again["drift_scans"] == [], "the same traffic must not be scanned twice"

    for request_id, truth in served:
        _ok(
            client.post(
                "/api/v1/feedback", json={"request_id": request_id, "actual_label": truth}
            )
        )
    labelled = _ok(client.post("/api/v1/automation/run"))
    assert len(labelled["retraining"]) == 1, labelled
    retrain = _job(client, labelled["retraining"][0])
    assert retrain["result"]["trigger"] in ("drift", "volume")

    after = _ok(client.post("/api/v1/automation/run"))
    assert after["retraining"] == [], "retraining must not loop on the labels it just used"
