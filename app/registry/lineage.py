"""Lineage: the recorded history of one model version, and of one dataset.

Every entity the lifecycle produces is already persisted -- dataset versions,
training and AutoML runs, jobs, gate decisions, stage transitions, deployments,
predictions, drift reports, retraining events. What was missing was the join:
"for this production model, show me the data it learned from, the run that
trained it, why it was approved, where it has served, and what happened next".

Nothing here is inferred. Each section is read from its own table by the keys
that link them (dataset version, tracking run id, model name + version,
deployment id), and a section with no rows is returned empty, not guessed at.
"""

from __future__ import annotations

from typing import Any

from app.core.db import get_database, loads
from app.registry.context import endpoint_for, signature_of
from app.registry.factory import get_registry


def version_lineage(name: str, version: int) -> dict[str, Any]:
    registry = get_registry()
    record = registry.get(name, version)
    db = get_database()
    signature = signature_of(record)

    return {
        "model": {
            "name": name,
            "version": version,
            "stage": record.stage.value,
            "status": record.status.value,
            "algorithm": record.algorithm,
            "created_at": record.created_at,
            "created_by": record.created_by,
            "git_commit": record.git_commit,
            "artifact_uri": record.artifact_uri,
            "endpoint": endpoint_for(name),
            "target": signature.target if signature else None,
            "classes": signature.labels if signature else None,
            "features": signature.feature_columns if signature else None,
        },
        "dataset": _dataset(record.dataset_version, record.dataset_hash),
        "training": _training(db, name, version, record.run_id),
        "evaluation": {
            "metrics": record.metrics,
            "threshold": record.params.get("threshold"),
            "params": {k: v for k, v in record.params.items() if k != "threshold"},
        },
        "gate_decisions": [
            _row(r, ("checks", "comparison", "thresholds"))
            for r in db.query(
                "SELECT * FROM gate_decisions WHERE model_name = ? AND model_version = ? "
                "ORDER BY id",
                (name, version),
            )
        ],
        "stage_history": [
            dict(r)
            for r in db.query(
                "SELECT from_stage, to_stage, reason, actor, created_at "
                "FROM model_stage_transitions WHERE name = ? AND version = ? ORDER BY id",
                (name, version),
            )
        ],
        "deployments": _deployments(db, name, version),
        "serving": _serving(db, name, version),
        "drift": [
            {
                "id": r["id"],
                "drift_detected": bool(r["drift_detected"]),
                "dataset_drift_score": r["dataset_drift_score"],
                "prediction_drift_score": r["prediction_drift_score"],
                "concept_drift_status": r["concept_drift_status"],
                "drifted_features": loads(r["drifted_features"], []),
                "n_current": r["n_current"],
                "created_at": r["created_at"],
            }
            for r in db.query(
                "SELECT * FROM drift_reports WHERE model_name = ? AND model_version = ? "
                "ORDER BY created_at DESC LIMIT 20",
                (name, version),
            )
        ],
        "retraining": {
            "produced_by": _event(
                db.query_one(
                    "SELECT * FROM retraining_events WHERE model_name = ? "
                    "AND candidate_version = ? ORDER BY created_at DESC LIMIT 1",
                    (name, version),
                )
            ),
            "triggered_from": [
                _event(r)
                for r in db.query(
                    "SELECT * FROM retraining_events WHERE model_name = ? "
                    "AND baseline_version = ? ORDER BY created_at DESC LIMIT 20",
                    (name, version),
                )
            ],
        },
    }


def dataset_lineage(dataset_version: str) -> dict[str, Any]:
    """Which models learned from a dataset, and which datasets it fed into."""
    from app.data.versioning import get_dataset_registry

    datasets = get_dataset_registry()
    record = datasets.get(dataset_version)
    db = get_database()
    models = [
        {
            "name": r["name"],
            "version": r["version"],
            "stage": r["stage"],
            "status": r["status"],
            "algorithm": r["algorithm"],
            "created_at": r["created_at"],
        }
        for r in db.query(
            "SELECT name, version, stage, status, algorithm, created_at FROM model_versions "
            "WHERE dataset_version = ? ORDER BY created_at",
            (dataset_version,),
        )
    ]
    runs = [
        {
            "kind": "training",
            "id": r["id"],
            "status": r["status"],
            "model_name": r["model_name"],
        }
        for r in db.query(
            "SELECT id, status, model_name FROM training_runs WHERE dataset_version = ? "
            "ORDER BY created_at",
            (dataset_version,),
        )
    ] + [
        {"kind": "automl", "id": r["id"], "status": r["status"], "model_name": r["model_name"]}
        for r in db.query(
            "SELECT id, status, model_name FROM automl_runs WHERE dataset_version = ? "
            "ORDER BY created_at",
            (dataset_version,),
        )
    ]
    derived = [
        {"version": v.version, "description": v.description, "created_at": v.created_at}
        for v in datasets.list_versions()
        if v.parent_version == dataset_version
    ]
    return {
        "dataset": record.model_dump(mode="json", exclude={"stats"}),
        "models_trained": models,
        "runs": runs,
        "next_versions": derived,
    }


# --------------------------------------------------------------------------- #
# Sections
# --------------------------------------------------------------------------- #
def _dataset(version: str | None, content_hash: str | None) -> dict[str, Any] | None:
    if not version:
        return None
    from app.data.versioning import get_dataset_registry

    try:
        record = get_dataset_registry().get(version)
    except Exception:
        return {"version": version, "available": False, "content_hash": content_hash}
    return {
        "version": record.version,
        "available": True,
        "dataset_name": record.dataset_name,
        "rows": record.n_rows,
        "columns": record.n_columns,
        "content_hash": record.content_hash,
        "hash_matches_model": content_hash is None or content_hash == record.content_hash,
        "parent_version": record.parent_version,
        "description": record.description,
        "created_at": record.created_at,
    }


def _training(db: Any, name: str, version: int, run_id: str | None) -> dict[str, Any] | None:
    """The run that produced this version, and the job that executed it."""
    found: dict[str, Any] | None = None
    row = db.query_one(
        "SELECT * FROM training_runs WHERE model_name = ? AND model_version = ? "
        "ORDER BY created_at DESC LIMIT 1",
        (name, version),
    )
    if row is not None:
        found = {
            "kind": "training",
            "id": row["id"],
            "status": row["status"],
            "algorithm": row["algorithm"],
            "tune": bool(row["tune"]),
            "started_at": row["started_at"],
            "completed_at": row["completed_at"],
        }
    elif run_id:
        automl = db.query_one(
            "SELECT * FROM automl_runs WHERE model_name = ? AND candidates LIKE ? "
            "ORDER BY created_at DESC LIMIT 1",
            (name, f"%{run_id}%"),
        )
        if automl is not None:
            candidates = loads(automl["candidates"], [])
            me = next((c for c in candidates if c.get("run_id") == run_id), {})
            found = {
                "kind": "automl",
                "id": automl["id"],
                "status": automl["status"],
                "algorithm": me.get("algorithm"),
                "rank": me.get("rank"),
                "winner": automl["best_model_version"] == version,
                "candidates": len(candidates),
                "started_at": automl["started_at"],
                "completed_at": automl["completed_at"],
            }
    if found is None:
        return {"kind": "unknown", "tracking_run_id": run_id} if run_id else None
    found["tracking_run_id"] = run_id
    job = db.query_one(
        "SELECT id, status, attempts, requested_by, created_at, finished_at FROM jobs "
        "WHERE resource_id = ? ORDER BY created_at DESC LIMIT 1",
        (found["id"],),
    )
    found["job"] = dict(job) if job else None
    return found


def _deployments(db: Any, name: str, version: int) -> list[dict[str, Any]]:
    rows = db.query(
        "SELECT * FROM deployments WHERE model_name = ? AND "
        "(current_version = ? OR candidate_version = ? OR previous_version = ? "
        "OR shadow_version = ?) ORDER BY created_at DESC LIMIT 20",
        (name, version, version, version, version),
    )
    out = []
    for r in rows:
        role = (
            "serving"
            if r["current_version"] == version
            else (
                "candidate"
                if r["candidate_version"] == version
                else "shadow" if r["shadow_version"] == version else "previous"
            )
        )
        out.append(
            {
                "id": r["id"],
                "endpoint": r["endpoint_name"],
                "strategy": r["strategy"],
                "state": r["state"],
                "role": role,
                "message": r["message"],
                "created_at": r["created_at"],
                "updated_at": r["updated_at"],
            }
        )
    return out


def _serving(db: Any, name: str, version: int) -> dict[str, Any]:
    row = db.query_one(
        "SELECT COUNT(*) AS n, SUM(CASE WHEN status != 'ok' THEN 1 ELSE 0 END) AS errors, "
        "MIN(created_at) AS first, MAX(created_at) AS last FROM inference_log "
        "WHERE model_name = ? AND model_version = ? AND shadow = 0",
        (name, version),
    )
    labelled = db.scalar(
        "SELECT COUNT(*) FROM inference_log i JOIN feedback f ON f.request_id = i.request_id "
        "WHERE i.model_name = ? AND i.model_version = ? AND i.shadow = 0",
        (name, version),
        0,
    )
    return {
        "predictions": int(row["n"] or 0),
        "errors": int(row["errors"] or 0),
        "labelled": int(labelled or 0),
        "first_prediction_at": row["first"],
        "last_prediction_at": row["last"],
    }


def _event(row: Any) -> dict[str, Any] | None:
    if row is None:
        return None
    detail = loads(row["detail"], {})
    return {
        "id": row["id"],
        "trigger": row["trigger"],
        "reason": row["reason"],
        "status": row["status"],
        "decision": row["decision"],
        "baseline_version": row["baseline_version"],
        "candidate_version": row["candidate_version"],
        "data_sources": detail.get("data_sources"),
        "created_at": row["created_at"],
    }


def _row(row: Any, json_fields: tuple[str, ...]) -> dict[str, Any]:
    data = dict(row)
    for key in json_fields:
        if data.get(key) is not None:
            data[key] = loads(data[key], None)
    return data
