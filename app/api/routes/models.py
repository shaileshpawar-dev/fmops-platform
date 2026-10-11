"""Model registry, approval and serving endpoints.

A model is a name with a lineage of versions. Everything here is scoped to one
model: its versions, its input contract (signature), its gate decisions, its
approvals, and its prediction endpoint.

Promotion into Staging or Production goes through exactly two doors: the
automated gate at the end of a training run, and ``/approve``, which re-runs
that gate and records who signed off. The raw stage endpoint can archive or
demote, but it cannot promote -- otherwise a version the gate rejected could be
walked into service one legal hop at a time.
"""

from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter, Body, Query, Request
from pydantic import BaseModel, Field

from app.api.security import request_actor
from app.api.serving import get_prediction_service
from app.core.audit import audit
from app.core.config import get_settings
from app.core.exceptions import FMOpsError, PredictionError
from app.core.logging import get_logger, new_request_id
from app.registry.context import endpoint_for, is_reference_model, signature_of
from app.registry.factory import get_registry
from app.schemas.common import ApprovalDecision, ModelStage, ModelStatus
from app.schemas.model import ModelVersion, StageTransition
from app.schemas.prediction import (
    BatchPredictionResponse,
    ModelBatchPredictionRequest,
    ModelPredictionRequest,
    PredictionResponse,
)
from app.training.approval import evaluate_approval, promote_if_eligible
from app.training.decisions import get_gate_decisions
from app.training.holdout import compare_on_shared_holdout
from app.training.model_factory import available_algorithms

logger = get_logger(__name__)
router = APIRouter(prefix="/api/v1/models", tags=["models"])


class GateRefusedError(FMOpsError):
    """The approval gate does not allow this promotion."""

    code = "gate_refused"
    http_status = 409


class ApproveRequest(BaseModel):
    model_config = {"extra": "forbid"}

    target_stage: Literal["Staging", "Production"] = "Staging"
    comment: str = Field(default="", max_length=500)


class RejectRequest(BaseModel):
    model_config = {"extra": "forbid"}

    comment: str = Field(..., min_length=3, max_length=500)


# --------------------------------------------------------------------------- #
# Registry reads
# --------------------------------------------------------------------------- #
@router.get("", summary="List registered models")
def list_models() -> dict[str, Any]:
    registry = get_registry()
    models = []
    for item in registry.list_models():
        name = item["name"]
        try:
            serving = registry.get_serving(name)
            # A name keeps one target for life, so any version's signature describes
            # the model -- including one that is not serving yet.
            described = serving or registry.get_latest(name)
            signature = signature_of(described) if described else None
            models.append(
                {
                    **item,
                    "endpoint": endpoint_for(name),
                    "serving_version": serving.version if serving else None,
                    "serving_stage": serving.stage.value if serving else None,
                    "target": signature.target if signature else None,
                    "positive_label": signature.positive_label if signature else None,
                    "n_features": len(signature.features) if signature else None,
                }
            )
        except Exception as exc:
            logger.warning("models.list_entry_failed", extra={"model": name, "error": str(exc)})
            models.append(
                {
                    **item,
                    "endpoint": endpoint_for(name),
                    "serving_version": None,
                    "serving_stage": None,
                    "target": None,
                    "positive_label": None,
                    "n_features": None,
                }
            )
    return {"backend": registry.backend, "models": models}


@router.get("/algorithms", summary="Algorithms available in this environment")
def algorithms() -> dict[str, Any]:
    """Which estimators can actually be instantiated here.

    Optional backends (xgboost, lightgbm) report ``false`` when their extra is
    not installed, rather than failing only at training time.
    """
    return {"algorithms": available_algorithms()}


@router.get("/decisions", summary="Gate decisions across every model")
def all_decisions(limit: int = Query(default=50, ge=1, le=500)) -> dict[str, Any]:
    """Newest first. Registered before the per-model routes, and a single
    segment, so it cannot be mistaken for a model name's sub-resource."""
    rows = get_gate_decisions().list(limit=limit)
    return {"count": len(rows), "decisions": rows}


@router.get("/pending", summary="Versions waiting for a human decision")
def pending_approvals() -> dict[str, Any]:
    """Versions that passed every automated check in an environment that
    requires sign-off, across every model."""
    from app.core.db import get_database

    rows = get_database().query(
        "SELECT name, version, stage, algorithm, metrics, created_at FROM model_versions "
        "WHERE status = 'pending' AND stage IN ('Development', 'Validation') "
        "ORDER BY created_at DESC LIMIT 100"
    )
    from app.core.db import loads

    items = [{**dict(r), "metrics": loads(r["metrics"], {})} for r in rows]
    return {"count": len(items), "versions": items}


@router.get("/{name}/versions", response_model=list[ModelVersion], summary="List versions")
def list_versions(
    name: str, stage: ModelStage | None = Query(default=None)
) -> list[ModelVersion]:
    return get_registry().list_versions(name, stage)


@router.get("/{name}/versions/{version}", response_model=ModelVersion, summary="Get a version")
def get_version(name: str, version: int) -> ModelVersion:
    return get_registry().get(name, version)


@router.get("/{name}/production", summary="Current production version")
def production(name: str) -> dict[str, Any]:
    registry = get_registry()
    current = registry.get_production(name)
    previous = registry.previous_production(name)
    return {
        "model_name": name,
        "production": current.model_dump(mode="json") if current else None,
        "previous_production": previous.model_dump(mode="json") if previous else None,
        "rollback_target": previous.version if previous else None,
    }


@router.get(
    "/{name}/history", response_model=list[StageTransition], summary="Stage transition history"
)
def history(name: str, version: int | None = None) -> list[StageTransition]:
    return get_registry().history(name, version)


@router.get("/{name}/signature", summary="Input contract of the serving (or a pinned) version")
def signature(name: str, version: int | None = None) -> dict[str, Any]:
    """What a prediction request for this model must contain.

    Returns the recorded signature of the requested version -- by default the
    one that serves traffic -- plus an example request built from the training
    medians and commonest categories, which is always valid.
    """
    registry = get_registry()
    target = registry.get(name, version) if version is not None else registry.get_serving(name)
    if target is None:
        target = registry.get_latest(name)
    if target is None:
        from app.core.exceptions import ModelNotFoundError

        raise ModelNotFoundError(f"model {name} has no registered versions", model=name)
    sig = signature_of(target)
    return {
        "model_name": name,
        "version": target.version,
        "stage": target.stage.value,
        "endpoint": endpoint_for(name),
        "predict_path": f"/api/v1/models/{name}/predict",
        "signature": sig.model_dump(mode="json") if sig else None,
        "example": {"features": sig.example()} if sig else None,
        "detail": (
            None
            if sig
            else (
                "this version predates recorded signatures; it is the reference model and "
                "takes the typed request documented at POST /api/v1/predict"
                if is_reference_model(name)
                else "this legacy model version has no recorded input signature; please retrain or register a new version with a valid signature"
            )
        ),
    }


@router.get(
    "/{name}/versions/{version}/lineage", summary="Everything recorded about a version"
)
def lineage(name: str, version: int) -> dict[str, Any]:
    """The version's full recorded history, from its training data to today.

    Dataset (with a check that its content hash still matches), the run and job
    that trained it, its evaluation, every gate decision and stage transition,
    every deployment it took part in, its serving volume, its drift reports,
    and the retraining events it produced or came from.
    """
    from app.registry.lineage import version_lineage

    return version_lineage(name, version)


@router.get("/{name}/decisions", summary="Gate decisions for a model")
def decisions(name: str, version: int | None = None, limit: int = 100) -> dict[str, Any]:
    """Every automated verdict and human sign-off, newest first."""
    rows = get_gate_decisions().list(name, version, limit=min(limit, 500))
    return {"model_name": name, "count": len(rows), "decisions": rows}


# --------------------------------------------------------------------------- #
# Serving
# --------------------------------------------------------------------------- #
@router.post("/{name}/predict", response_model=PredictionResponse, summary="Score one record")
def predict(
    name: str, payload: ModelPredictionRequest, request: Request
) -> PredictionResponse:
    """Score one record with the version this model's endpoint routes to.

    The record is checked against the recorded signature of the version that
    actually serves it: unknown or missing fields are rejected (422), and values
    the model never saw in training are accepted but reported in ``warnings``.
    """
    return get_prediction_service().predict_one(
        features=payload.features,
        request_id=getattr(request.state, "request_id", None) or new_request_id(),
        version=payload.model_version,
        threshold=payload.threshold,
        explain=payload.explain,
        model_name=name,
        check_contract=True,
    )


@router.post(
    "/{name}/predict/batch",
    response_model=BatchPredictionResponse,
    summary="Score many records",
)
def predict_batch(
    name: str, payload: ModelBatchPredictionRequest, request: Request
) -> BatchPredictionResponse:
    limit = get_settings().security.max_batch_rows
    if len(payload.instances) > limit:
        raise PredictionError(
            f"batch of {len(payload.instances)} exceeds the configured maximum of {limit} rows",
            n_instances=len(payload.instances),
            max_batch_rows=limit,
        )
    return get_prediction_service().predict_batch(
        instances=payload.instances,
        request_id=getattr(request.state, "request_id", None) or new_request_id(),
        version=payload.model_version,
        threshold=payload.threshold,
        model_name=name,
        check_contract=True,
    )


# --------------------------------------------------------------------------- #
# Approval
# --------------------------------------------------------------------------- #
@router.post("/{name}/versions/{version}/approve", summary="Approve a version for promotion")
def approve(
    name: str, version: int, payload: ApproveRequest, request: Request
) -> dict[str, Any]:
    """Human sign-off, bounded by the automated gate.

    The gate is re-run on the version's recorded metrics under the thresholds
    in force *now*. Approval is refused when any automated check fails, and --
    for Production -- when the version does not beat the current production
    model by the configured margin. What a human adds is the decision, not an
    override.
    """
    actor = request_actor(request)
    settings = get_settings()
    registry = get_registry()
    candidate = registry.get(name, version)
    target = ModelStage(payload.target_stage)
    if candidate.stage == ModelStage.ARCHIVED:
        raise GateRefusedError(
            f"{name} v{version} is archived; restore it to Development first", model=name
        )
    if candidate.stage == target:
        raise GateRefusedError(f"{name} v{version} is already in {target.value}", model=name)

    approval = evaluate_approval(
        candidate.metrics, name, version, validation_passed=True, settings=settings
    )
    comparison = compare_on_shared_holdout(candidate, registry.get_production(name), settings)
    failed = [c.name for c in approval.failed_checks]
    refusal: str | None = None
    if failed:
        refusal = f"automated gate checks fail: {', '.join(failed)}"
    elif target == ModelStage.PRODUCTION and not comparison.candidate_is_better:
        refusal = comparison.reason

    if refusal:
        get_gate_decisions().record(
            model_name=name,
            model_version=version,
            source="manual",
            decision=ApprovalDecision.REJECTED.value,
            reason=f"approval refused: {refusal}",
            approval=approval,
            comparison=comparison,
            thresholds=settings.approval.model_dump(mode="json"),
            target_stage=target.value,
            final_stage=candidate.stage.value,
            actor=actor,
            comment=payload.comment or None,
        )
        audit("model.approve", "model", f"{name}:{version}", outcome="denied", reason=refusal)
        raise GateRefusedError(f"cannot approve {name} v{version}: {refusal}", model=name)

    signed = approval.model_copy(
        update={
            "decision": ApprovalDecision.APPROVED,
            "reason": f"approved by {actor}"
            + (f": {payload.comment}" if payload.comment else ""),
        }
    )
    updated = promote_if_eligible(
        name,
        version,
        signed,
        comparison if target == ModelStage.PRODUCTION else None,
        registry,
        target_stage=target,
        actor=actor,
    )
    record = get_gate_decisions().record(
        model_name=name,
        model_version=version,
        source="manual",
        decision=ApprovalDecision.APPROVED.value,
        reason=signed.reason,
        approval=approval,
        comparison=comparison,
        thresholds=settings.approval.model_dump(mode="json"),
        target_stage=target.value,
        final_stage=updated.stage.value,
        actor=actor,
        comment=payload.comment or None,
    )
    audit(
        "model.approve",
        "model",
        f"{name}:{version}",
        target_stage=target.value,
        final_stage=updated.stage.value,
    )
    return {"model_version": updated.model_dump(mode="json"), "decision": record}


@router.post("/{name}/versions/{version}/reject", summary="Reject a version")
def reject(
    name: str, version: int, payload: RejectRequest, request: Request
) -> dict[str, Any]:
    """Record a human rejection. The version stays registered, for the record.

    Only a version that is not serving can be rejected; a serving version is
    taken out of service with a rollback, which leaves a deployment trail.
    """
    actor = request_actor(request)
    registry = get_registry()
    candidate = registry.get(name, version)
    if candidate.stage in (ModelStage.STAGING, ModelStage.PRODUCTION):
        raise GateRefusedError(
            f"{name} v{version} is in {candidate.stage.value}; roll back or archive it instead",
            model=name,
        )
    registry.update_status(name, version, ModelStatus.REJECTED.value)
    record = get_gate_decisions().record(
        model_name=name,
        model_version=version,
        source="manual",
        decision=ApprovalDecision.REJECTED.value,
        reason=f"rejected by {actor}: {payload.comment}",
        final_stage=candidate.stage.value,
        actor=actor,
        comment=payload.comment,
    )
    audit("model.reject", "model", f"{name}:{version}", comment=payload.comment)
    return {
        "model_version": registry.get(name, version).model_dump(mode="json"),
        "decision": record,
    }


@router.post(
    "/{name}/versions/{version}/stage",
    response_model=ModelVersion,
    summary="Archive, demote or restore a version",
)
def transition(
    name: str,
    version: int,
    request: Request,
    stage: ModelStage = Body(..., embed=True),
    reason: str = Body(default="manual transition", embed=True),
) -> ModelVersion:
    """Move a version through the stage machine -- in every direction but up.

    Promotion into Staging or Production is refused here and must go through
    ``/approve``, which runs the gate. Everything else (archive, demote,
    restore, Development -> Validation) is allowed, and illegal moves are still
    rejected by the stage machine with the list of legal targets.
    """
    registry = get_registry()
    current = registry.get(name, version)
    promoting = stage in (ModelStage.STAGING, ModelStage.PRODUCTION) and not (
        current.stage == ModelStage.PRODUCTION and stage == ModelStage.STAGING
    )
    if promoting:
        raise GateRefusedError(
            f"promotion to {stage.value} goes through the approval gate: POST "
            f"/api/v1/models/{name}/versions/{version}/approve",
            model=name,
            requested_stage=stage.value,
        )
    return registry.transition_stage(
        name, version, stage, reason=reason, actor=request_actor(request)
    )


@router.post(
    "/{name}/versions/{version}/evaluate-gate",
    summary="Preview the approval gate against a version",
)
def evaluate_gate(name: str, version: int) -> dict[str, Any]:
    """Re-run the approval gate using the version's recorded metrics.

    A preview: it records nothing and changes nothing. It shows exactly which
    thresholds a version would clear under the *current* configuration, which
    may differ from the thresholds in force when it was trained.
    """
    registry = get_registry()
    model_version = registry.get(name, version)
    settings = get_settings()
    result = evaluate_approval(
        model_version.metrics,
        name,
        version,
        validation_passed=True,
        settings=settings,
    )
    comparison = compare_on_shared_holdout(
        model_version, registry.get_production(name), settings
    )
    return {
        "approval": result.model_dump(mode="json"),
        "comparison": comparison.model_dump(mode="json"),
        "thresholds": settings.approval.model_dump(mode="json"),
        "report": result.render_text(),
    }


@router.post(
    "/{name}/versions/{version}/tags", response_model=ModelVersion, summary="Set tags"
)
def set_tags(name: str, version: int, tags: dict[str, str] = Body(...)) -> ModelVersion:
    return get_registry().set_tags(name, version, tags)


from fastapi import HTTPException
import uuid
import os
import shutil

MAX_MODEL_UPLOAD_BYTES = 50 * 1024 * 1024  # 50 MB

@router.post("/{name}/upload", response_model=ModelVersion, summary="Upload a pre-trained model")
async def upload_model(name: str, request: Request) -> ModelVersion:
    """Upload a pre-trained model artifact via raw binary streaming or multipart form."""
    import secrets
    from app.core.config import get_settings
    from app.core.signature import SignatureError, validate_model_name

    settings = get_settings()
    if not settings.security.allow_model_upload:
        token = settings.security.model_upload_token
        auth_header = (
            request.headers.get("X-Upload-Token")
            or request.headers.get("Authorization", "")
        ).replace("Bearer ", "").strip()
        authorized = False
        if token and auth_header:
            try:
                authorized = secrets.compare_digest(
                    auth_header.encode("utf-8"),
                    token.encode("utf-8"),
                )
            except Exception:
                authorized = False
        if not authorized:
            raise HTTPException(
                403,
                "Pretrained model upload is disabled in production for deserialization safety. "
                "Enable it with FMOPS_SECURITY__ALLOW_MODEL_UPLOAD=true or provide a valid X-Upload-Token header.",
            )

    try:
        validate_model_name(name)
    except SignatureError as e:
        raise HTTPException(400, str(e))

    content_type = request.headers.get("content-type", "").lower()
    is_multipart = content_type.startswith("multipart/form-data")

    filename = request.headers.get("X-File-Name")
    sig_str = request.headers.get("X-Model-Signature")
    metrics_str = request.headers.get("X-Model-Metrics")

    uploaded_file = None
    if is_multipart:
        form = await request.form()
        file_field = form.get("file")
        if file_field is not None and hasattr(file_field, "filename"):
            uploaded_file = file_field
            filename = filename or file_field.filename
        sig_str = sig_str or form.get("signature") or form.get("X-Model-Signature")
        metrics_str = metrics_str or form.get("metrics") or form.get("X-Model-Metrics")

    if not filename:
        raise HTTPException(400, "No X-File-Name header or multipart file provided")
    
    ext = filename.split(".")[-1].lower()
    if ext not in ("pkl", "joblib"):
        raise HTTPException(400, f"Unsupported file extension: .{ext}. Only .pkl and .joblib are supported.")
    
    settings = get_settings()
    upload_id = str(uuid.uuid4())
    upload_dir = settings.paths.models_dir / "uploads" / upload_id
    upload_dir.mkdir(parents=True, exist_ok=True)
    temp_path = upload_dir / f"model.{ext}"
    
    def cleanup():
        shutil.rmtree(upload_dir, ignore_errors=True)

    size = 0
    try:
        with temp_path.open("wb") as f:
            if is_multipart and uploaded_file is not None:
                while True:
                    chunk = await uploaded_file.read(64 * 1024)
                    if not chunk:
                        break
                    size += len(chunk)
                    if size > MAX_MODEL_UPLOAD_BYTES:
                        cleanup()
                        raise HTTPException(
                            413,
                            f"File exceeds the {MAX_MODEL_UPLOAD_BYTES // (1024 * 1024)} MB upload limit",
                        )
                    f.write(chunk)
            else:
                async for chunk in request.stream():
                    size += len(chunk)
                    if size > MAX_MODEL_UPLOAD_BYTES:
                        cleanup()
                        raise HTTPException(
                            413,
                            f"File exceeds the {MAX_MODEL_UPLOAD_BYTES // (1024 * 1024)} MB upload limit",
                        )
                    f.write(chunk)
    except HTTPException:
        cleanup()
        raise
    except Exception as e:
        cleanup()
        raise HTTPException(500, f"Error saving upload: {e}")

    # Validate size
    if size == 0:
        cleanup()
        raise HTTPException(400, "Uploaded file is empty")
        
    # Check if we can load it to verify it's a valid model
    try:
        import joblib
        model = joblib.load(str(temp_path))
        if not hasattr(model, "predict"):
            cleanup()
            raise HTTPException(400, "Uploaded artifact is not an estimator (missing predict method).")
    except HTTPException:
        cleanup()
        raise
    except Exception as e:
        cleanup()
        raise HTTPException(400, f"Could not deserialize uploaded artifact: {e}")

    import json
    params = {}
    if metrics_str and str(metrics_str).strip():
        try:
            params["client_metrics"] = json.loads(str(metrics_str).strip())
        except Exception:
            cleanup()
            raise HTTPException(400, "Invalid JSON in X-Model-Metrics header")

    from app.core.signature import ModelSignature

    signature = None
    if sig_str and str(sig_str).strip():
        try:
            signature_json = json.loads(str(sig_str).strip())
            if not isinstance(signature_json, dict):
                raise ValueError("Signature must be a JSON object")
            signature_dict = dict(signature_json)
            if "dataset_name" not in signature_dict:
                signature_dict["dataset_name"] = name
            if "target" not in signature_dict:
                signature_dict["target"] = "target"
            if "class_labels" not in signature_dict:
                classes = getattr(model, "classes_", [0, 1])
                signature_dict["class_labels"] = (
                    [str(c) for c in classes] if len(classes) == 2 else ["0", "1"]
                )
            parsed_sig = ModelSignature.model_validate(signature_dict)
            signature = parsed_sig.model_dump(mode="json")
        except Exception as e:
            cleanup()
            raise HTTPException(400, f"Invalid model signature: {e}")
    elif hasattr(model, "feature_names_in_") and model.feature_names_in_ is not None:
        try:
            classes = getattr(model, "classes_", [0, 1])
            class_labels = [str(c) for c in classes] if len(classes) == 2 else ["0", "1"]
            feats = [{"name": str(fn), "kind": "numeric", "dtype": "float64"} for fn in model.feature_names_in_]
            parsed_sig = ModelSignature.model_validate({
                "dataset_name": name,
                "target": "target",
                "class_labels": class_labels,
                "features": feats,
            })
            signature = parsed_sig.model_dump(mode="json")
        except Exception:
            signature = None

    # Write metadata sidecar beside the artifact
    try:
        sidecar = upload_dir / "metadata.json"
        metadata = {
            "model_name": name,
            "algorithm": "uploaded",
            "signature": signature,
            "feature_columns": [f["name"] for f in signature.get("features", [])] if signature else [],
        }
        sidecar.write_text(json.dumps(metadata, indent=2))
    except Exception:
        pass

    # Register
    try:
        actor = request_actor(request)
        version = get_registry().register(
            name=name,
            artifact_uri=temp_path.absolute().as_uri(),
            algorithm="uploaded",
            description=f"Uploaded by {actor}",
            created_by=actor,
            metrics=None,
            params=params,
            signature=signature,
        )
    except Exception as e:
        cleanup()
        raise HTTPException(500, f"Failed to register uploaded model: {e}")

    return version

