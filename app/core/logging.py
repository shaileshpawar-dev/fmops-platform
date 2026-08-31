"""Structured logging.

Everything the platform logs goes through :func:`get_logger`.  Two formatters
are available:

* ``json``    -- one JSON object per line, for CloudWatch / Loki / any collector.
* ``console`` -- human readable, for local development.

Correlation identifiers (request id, run id, model version, deployment id) live
in :mod:`contextvars` so they are attached automatically to every record emitted
inside a request or a pipeline stage, without threading them through call
signatures.
"""

from __future__ import annotations

import contextvars
import json
import logging
import logging.config
import os
import sys
import time
import traceback
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

# Default is None rather than {} so the sentinel is never shared or mutated;
# every accessor materialises a fresh dict.
_LOG_CONTEXT: contextvars.ContextVar[dict[str, Any] | None] = contextvars.ContextVar(
    "fmops_log_context", default=None
)

_CONFIGURED = False

# Attributes present on every stdlib LogRecord; anything else is treated as an
# extra field and merged into the structured payload.
_RESERVED = frozenset(
    {
        "name",
        "msg",
        "args",
        "levelname",
        "levelno",
        "pathname",
        "filename",
        "module",
        "exc_info",
        "exc_text",
        "stack_info",
        "lineno",
        "funcName",
        "created",
        "msecs",
        "relativeCreated",
        "thread",
        "threadName",
        "processName",
        "process",
        "taskName",
        "message",
        "asctime",
    }
)


def new_request_id() -> str:
    return uuid.uuid4().hex


def get_context() -> dict[str, Any]:
    """Return a copy of the current logging context."""
    return dict(_LOG_CONTEXT.get() or {})


def bind_context(**fields: Any) -> contextvars.Token:
    """Attach fields to every subsequent log record in this context."""
    current = _LOG_CONTEXT.get() or {}
    merged = {**current, **{k: v for k, v in fields.items() if v is not None}}
    return _LOG_CONTEXT.set(merged)


def reset_context(token: contextvars.Token) -> None:
    _LOG_CONTEXT.reset(token)


def clear_context() -> None:
    _LOG_CONTEXT.set(None)


@contextmanager
def log_context(**fields: Any) -> Iterator[dict[str, Any]]:
    """Scope a set of correlation fields to a block."""
    token = bind_context(**fields)
    try:
        yield get_context()
    finally:
        reset_context(token)


class JsonFormatter(logging.Formatter):
    """Render a LogRecord as a single-line JSON object."""

    def __init__(self, service: str, environment: str, version: str) -> None:
        super().__init__()
        self.service = service
        self.environment = environment
        self.version = version

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created))
            + f".{int(record.msecs):03d}Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "service": self.service,
            "environment": self.environment,
            "version": self.version,
        }
        payload.update(_LOG_CONTEXT.get() or {})
        for key, value in record.__dict__.items():
            if key not in _RESERVED and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            exc_type, exc_value, exc_tb = record.exc_info
            payload["error"] = {
                "type": getattr(exc_type, "__name__", str(exc_type)),
                "message": str(exc_value),
                "traceback": "".join(traceback.format_exception(exc_type, exc_value, exc_tb))[
                    -8000:
                ],
            }
        try:
            return json.dumps(payload, default=_json_default, ensure_ascii=False)
        except (TypeError, ValueError):  # pragma: no cover - defensive
            return json.dumps(
                {
                    "timestamp": payload["timestamp"],
                    "level": payload["level"],
                    "logger": payload["logger"],
                    "message": payload["message"],
                    "serialization_error": True,
                }
            )


def _json_default(obj: Any) -> str:
    return str(obj)


class ConsoleFormatter(logging.Formatter):
    """Compact human-readable formatter with the context appended."""

    _COLORS = {
        "DEBUG": "\033[36m",
        "INFO": "\033[32m",
        "WARNING": "\033[33m",
        "ERROR": "\033[31m",
        "CRITICAL": "\033[35m",
    }
    _RESET = "\033[0m"

    def __init__(self, use_color: bool = True) -> None:
        super().__init__(datefmt="%H:%M:%S")
        self.use_color = use_color

    def format(self, record: logging.LogRecord) -> str:
        ctx = {**(_LOG_CONTEXT.get() or {})}
        for key, value in record.__dict__.items():
            if key not in _RESERVED and not key.startswith("_"):
                ctx[key] = value
        suffix = " ".join(f"{k}={v}" for k, v in ctx.items())
        level = record.levelname
        if self.use_color:
            color = self._COLORS.get(level, "")
            level = f"{color}{level:<8}{self._RESET}"
        else:
            level = f"{level:<8}"
        line = (
            f"{self.formatTime(record, self.datefmt)} {level} "
            f"{record.name}: {record.getMessage()}"
        )
        if suffix:
            line = f"{line}  | {suffix}"
        if record.exc_info:
            line = f"{line}\n{self.formatException(record.exc_info)}"
        return line


def configure_logging(
    level: str = "INFO",
    fmt: str = "json",
    service: str = "fmops-platform",
    environment: str = "development",
    version: str = "1.0.0",
    force: bool = False,
) -> None:
    """Install the root handler. Idempotent unless ``force`` is set."""
    global _CONFIGURED
    if _CONFIGURED and not force:
        return

    handler = logging.StreamHandler(stream=sys.stdout)
    if fmt == "json":
        handler.setFormatter(JsonFormatter(service, environment, version))
    else:
        handler.setFormatter(
            ConsoleFormatter(use_color=sys.stdout.isatty() and os.name != "nt")
        )

    root = logging.getLogger()
    for existing in list(root.handlers):
        root.removeHandler(existing)
    root.addHandler(handler)
    root.setLevel(level.upper())

    # Third-party loggers that are noisy at INFO.
    for noisy in (
        "botocore",
        "boto3",
        "urllib3",
        "s3transfer",
        "matplotlib",
        "mlflow.utils",
        "mlflow.tracking",
        "git",
        "alembic",
        "watchfiles",
    ):
        logging.getLogger(noisy).setLevel(max(logging.WARNING, root.level))

    _CONFIGURED = True


def configure_from_settings(force: bool = False) -> None:
    """Configure logging from the active :class:`~app.core.config.Settings`."""
    from app.core.config import get_settings

    s = get_settings()
    configure_logging(
        level=s.log_level,
        fmt=s.log_format,
        service=s.service_name,
        environment=s.environment,
        version=s.version,
        force=force,
    )


def get_logger(name: str) -> logging.Logger:
    """Return a logger, configuring the root handler on first use."""
    if not _CONFIGURED:
        try:
            configure_from_settings()
        except Exception:  # pragma: no cover - config errors must not hide logs
            configure_logging()
    return logging.getLogger(name)


class StageTimer:
    """Times a pipeline stage and logs start/finish/failure with duration."""

    def __init__(self, logger: logging.Logger, stage: str, **fields: Any) -> None:
        self.logger = logger
        self.stage = stage
        self.fields = fields
        self.start = 0.0
        self.duration_ms = 0.0

    def __enter__(self) -> StageTimer:
        self.start = time.perf_counter()
        self.logger.info("stage.start", extra={"stage": self.stage, **self.fields})
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.duration_ms = (time.perf_counter() - self.start) * 1000
        if exc_type is None:
            self.logger.info(
                "stage.finish",
                extra={
                    "stage": self.stage,
                    "duration_ms": round(self.duration_ms, 2),
                    **self.fields,
                },
            )
        else:
            self.logger.error(
                "stage.failed",
                exc_info=(exc_type, exc, tb),
                extra={
                    "stage": self.stage,
                    "duration_ms": round(self.duration_ms, 2),
                    **self.fields,
                },
            )
        return False  # never suppress
