"""Error handling.

Every :class:`~app.core.exceptions.FMOpsError` becomes a JSON response carrying
its stable ``code``, a human message, and the structured ``details`` the
exception was raised with -- so a client sees *which* validation expectations
failed or *which* gate check rejected a model, not just "500".

Unexpected exceptions are logged with a traceback and correlation id, and the
client gets the correlation id but not the internals.
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.core.exceptions import FMOpsError
from app.core.logging import get_context, get_logger
from app.core.utils import jsonable
from app.monitoring.metrics import record_error

logger = get_logger(__name__)


def error_body(
    code: str, message: str, details: dict[str, Any] | None = None
) -> dict[str, Any]:
    body: dict[str, Any] = {"error": {"code": code, "message": message}}
    request_id = get_context().get("request_id")
    if request_id:
        body["error"]["request_id"] = request_id
    if details:
        body["error"]["details"] = jsonable(details)
    return body


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(FMOpsError)
    async def _fmops_error(request: Request, exc: FMOpsError) -> JSONResponse:
        level = logger.warning if exc.http_status < 500 else logger.error
        level(
            "api.error",
            extra={
                "error_code": exc.code,
                "path": request.url.path,
                "method": request.method,
                "status": exc.http_status,
                "detail": jsonable(exc.details),
            },
        )
        record_error(exc.code, "api")
        return JSONResponse(
            status_code=exc.http_status,
            content=error_body(exc.code, exc.message, exc.details),
        )

    @app.exception_handler(RequestValidationError)
    async def _validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        record_error("request_validation_failed", "api")
        logger.info(
            "api.request_invalid",
            extra={"path": request.url.path, "errors": len(exc.errors())},
        )
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=error_body(
                "request_validation_failed",
                "the request body did not match the expected schema",
                {"violations": jsonable(exc.errors())[:20]},
            ),
        )

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
        # Log the full traceback server-side; return only a correlation id.
        logger.error(
            "api.unhandled_exception",
            exc_info=exc,
            extra={"path": request.url.path, "method": request.method},
        )
        record_error("internal_error", "api")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=error_body(
                "internal_error",
                "an unexpected error occurred; the incident has been logged",
            ),
        )
