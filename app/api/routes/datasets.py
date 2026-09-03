"""Dataset upload, validation and preview.

Read-only dataset listing already lives in :mod:`app.api.routes.experiments`
(``GET /api/v1/datasets`` and ``GET /api/v1/datasets/{version}``) and is left
exactly where it is. This module adds only what was missing: getting a CSV
into the platform, validating it, and looking at it without downloading the
whole file.

Storage, versioning and validation are all existing machinery --
:class:`app.data.versioning.DatasetRegistry` and
:func:`app.data.validation.validate_dataframe`. Nothing here re-implements a
validation rule or a hashing scheme.
"""

from __future__ import annotations

import io
from typing import Annotated, Any

import pandas as pd
from fastapi import APIRouter, Query, Request

from app.core.audit import audit
from app.core.config import get_settings
from app.core.exceptions import DatasetNotFoundError, FMOpsError
from app.core.logging import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/api/v1/datasets", tags=["datasets"])

# A CSV large enough to train on is still small; anything past this is either a
# mistake or an attempt to exhaust the container's memory. Read as bytes before
# pandas ever sees it, so a hostile file cannot expand during parsing.
MAX_UPLOAD_BYTES = 25 * 1024 * 1024
MAX_PREVIEW_ROWS = 50
ALLOWED_SUFFIXES = {".csv"}


class DatasetUploadError(FMOpsError):
    """The uploaded file is not something this platform can accept."""

    code = "dataset_upload_rejected"
    http_status = 422


def _suffix_of(filename: str) -> str:
    """Lowercased extension of a filename, or "" when it has none."""
    tail = filename.replace("\\", "/").rsplit("/", 1)[-1]
    return "." + tail.rsplit(".", 1)[-1].lower() if "." in tail else ""


def _safe_stem(filename: str | None) -> str:
    """Reduce a client-supplied filename to a bare, safe stem.

    Only the final path component is considered and only ``[A-Za-z0-9._-]``
    survives, so ``../../etc/passwd`` and ``C:\\windows\\x.csv`` both collapse
    to an inert name. The result is never used to choose a directory -- the
    dataset registry decides where bytes land -- but a filename that reaches a
    log line or a manifest should still be inert.
    """
    raw = (filename or "upload.csv").replace("\\", "/").rsplit("/", 1)[-1]
    cleaned = "".join(c for c in raw if c.isalnum() or c in "._-").lstrip(".")
    stem = cleaned.rsplit(".", 1)[0] if "." in cleaned else cleaned
    return (stem or "upload")[:64]


@router.post(
    "/upload",
    status_code=201,
    summary="Upload a CSV and register it as a dataset version",
)
async def upload_dataset(
    request: Request,
    filename: Annotated[
        str | None, Query(description="Original filename, for the record.")
    ] = None,
    dataset_name: Annotated[str | None, Query(max_length=128)] = None,
    description: Annotated[str, Query(max_length=500)] = "",
    validate: Annotated[bool, Query(description="Run validation after registering.")] = True,
) -> dict[str, Any]:
    """Register an uploaded CSV as an immutable, content-addressed version.

    The CSV is sent as the raw request body (``Content-Type: text/csv``) with
    metadata in the query string, rather than as a multipart form. Multipart
    would pull ``python-multipart`` into the runtime image for a single
    single-file endpoint, and the environment already carries a conflicting
    ``multipart`` package that FastAPI refuses to use. A raw body needs no
    parser at all and is trivial to send from a browser:

    ``fetch(url, {method: "POST", body: file, headers: {"Content-Type": "text/csv"}})``

    Uploading identical bytes twice returns the existing version rather than
    creating a duplicate: versions are content, not events.

    The body is parsed with pandas and nothing else. It is never executed, never
    passed to a shell, and never used to build a filesystem path.
    """
    if filename and _suffix_of(filename) not in ALLOWED_SUFFIXES:
        raise DatasetUploadError(
            f"unsupported file type {_suffix_of(filename) or '(none)'}; this endpoint accepts CSV only",
            filename=_safe_stem(filename),
            allowed=sorted(ALLOWED_SUFFIXES),
        )

    # Read incrementally and abort the moment the cap is passed, so an oversized
    # or endless body is never fully buffered.
    chunks: list[bytes] = []
    total = 0
    async for chunk in request.stream():
        total += len(chunk)
        if total > MAX_UPLOAD_BYTES:
            raise DatasetUploadError(
                f"file exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB upload limit",
                limit_bytes=MAX_UPLOAD_BYTES,
            )
        chunks.append(chunk)
    raw = b"".join(chunks)

    if not raw.strip():
        raise DatasetUploadError("the uploaded file is empty")

    try:
        frame = pd.read_csv(io.BytesIO(raw))
    except UnicodeDecodeError as exc:
        raise DatasetUploadError("the file is not valid UTF-8 text", detail=str(exc)) from exc
    except Exception as exc:  # pandas raises a wide range for malformed CSV
        raise DatasetUploadError(f"the file could not be parsed as CSV: {exc}") from exc

    if frame.empty or not len(frame.columns):
        raise DatasetUploadError("the CSV parsed to zero rows or zero columns")

    settings = get_settings()
    name = (dataset_name or settings.data.dataset_name).strip() or settings.data.dataset_name

    from app.data.versioning import get_dataset_registry

    version = get_dataset_registry().register_frame(
        frame,
        dataset_name=name,
        filename=f"{_safe_stem(filename)}.csv",
        description=description[:500],
    )

    payload: dict[str, Any] = {
        "version": version.version,
        "dataset_name": version.dataset_name,
        "rows": len(frame),
        "columns": len(frame.columns),
        "column_names": [str(c) for c in frame.columns],
        "created_at": getattr(version, "created_at", None),
        "validation": None,
    }

    if validate:
        payload["validation"] = _validation_summary(frame, version.version)

    audit(
        "dataset.uploaded",
        "dataset",
        version.version,
        dataset_name=name,
        rows=len(frame),
        columns=len(frame.columns),
        bytes=len(raw),
        validated=validate,
        validation_passed=(payload["validation"] or {}).get("passed"),
    )
    logger.info(
        "dataset.uploaded",
        extra={
            "version": version.version,
            "dataset": name,
            "rows": len(frame),
            "columns": len(frame.columns),
            "bytes": len(raw),
        },
    )
    return payload


@router.get("/{version}/validation", summary="Validate one dataset version")
def dataset_validation(version: str) -> dict[str, Any]:
    """Run the platform's validation engine over a registered version.

    Uses the same engine and the same expectations the training pipeline gates
    on, so a dataset that passes here is one the pipeline will accept.
    """
    from app.data.versioning import get_dataset_registry

    registry = get_dataset_registry()
    registry.get(version)  # raises DatasetNotFoundError if unknown
    frame = registry.load(version)
    return _validation_summary(frame, version)


@router.get("/{version}/preview", summary="Preview rows and column profile")
def dataset_preview(
    version: str,
    rows: Annotated[int, Query(ge=1, le=MAX_PREVIEW_ROWS)] = 20,
) -> dict[str, Any]:
    """Return a bounded sample plus a per-column profile.

    Capped at ``MAX_PREVIEW_ROWS``: a preview exists to show the shape of the
    data, and shipping an entire training set to a browser is neither useful
    nor safe.
    """
    from app.data.versioning import get_dataset_registry

    registry = get_dataset_registry()
    registry.get(version)
    frame = registry.load(version)
    head = frame.head(rows)

    columns = []
    for col in frame.columns:
        series = frame[col]
        missing = int(series.isna().sum())
        example = series.dropna().iloc[0] if series.notna().any() else None
        columns.append(
            {
                "name": str(col),
                "dtype": str(series.dtype),
                "missing": missing,
                "missing_pct": round(missing / max(len(frame), 1) * 100, 2),
                "unique": int(series.nunique(dropna=True)),
                "example": None if example is None else str(example)[:80],
            }
        )

    return {
        "version": version,
        "rows": len(frame),
        "columns": len(frame.columns),
        "preview_rows": len(head),
        "column_profile": columns,
        "sample": head.astype(object).where(pd.notna(head), None).to_dict(orient="records"),
    }


def _validation_summary(frame: pd.DataFrame, version: str | None) -> dict[str, Any]:
    """Shape the existing validation report for the API.

    The engine owns every rule; this only selects fields and truncates the
    failure list so one badly broken dataset cannot return a megabyte of JSON.
    """
    from app.data.validation import validate_dataframe

    report = validate_dataframe(frame, dataset_version=version)
    results = list(report.results)
    failed = [r for r in results if not r.success]
    return {
        "passed": bool(report.passed),
        "dataset_name": report.dataset_name,
        "dataset_version": report.dataset_version or version,
        "engine": report.engine,
        "rows": report.n_rows,
        "columns": report.n_columns,
        "expectations": len(results),
        "succeeded": len(results) - len(failed),
        "failed": len(failed),
        "warnings": sum(1 for r in failed if r.severity == "warning"),
        "errors": sum(1 for r in failed if r.severity != "warning"),
        "failures": [
            {
                "expectation": r.expectation,
                "column": r.column,
                "severity": r.severity,
                "message": str(r.message)[:240],
                "observed": str(r.observed)[:120] if r.observed is not None else None,
                "expected": str(r.expected)[:120] if r.expected is not None else None,
            }
            for r in failed[:50]
        ],
        "truncated": len(failed) > 50,
        "created_at": report.created_at,
    }


__all__ = ["DatasetNotFoundError", "router"]
