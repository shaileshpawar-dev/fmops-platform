"""FastAPI application.

Wires together middleware (correlation ids, metrics, auth), the exception
handlers, and every route module. Startup warms the model cache and restores the
routing table so a restarted container serves the same version it was serving
before, without waiting for a manual deployment.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.errors import error_body, register_exception_handlers
from app.api.routes import (
    automl,
    datasets,
    deployments,
    experiments,
    health,
    llm,
    models,
    monitoring,
    predictions,
    retraining,
    training,
)
from app.api.security import AuthMiddlewareState
from app.core.config import Settings, get_settings
from app.core.db import get_database
from app.core.exceptions import AuthenticationError
from app.core.logging import (
    bind_context,
    clear_context,
    configure_from_settings,
    get_logger,
    new_request_id,
)
from app.core.utils import git_commit
from app.monitoring.metrics import Timer, record_http

logger = get_logger(__name__)

DESCRIPTION = """
**Foundation Model Operations Platform (FMOps)** -- one control plane for the
full lifecycle of both traditional ML models and LLM-backed features.

* **MLOps** -- data validation, versioned datasets, training, hyperparameter
  search, evaluation, a model registry with enforced stage transitions,
  approval gates, blue/green + canary + shadow deployment, and rollback.
* **Monitoring** -- Prometheus metrics, latency/error SLOs, data & prediction
  drift, and live model quality computed only from labelled traffic.
* **LLMOps** -- provider-agnostic invocation, versioned prompts, tracing,
  evaluation, a heuristic safety screen, and token/cost accounting.

Local mode runs entirely on the filesystem, SQLite and a local MLflow store.
AWS (S3, SageMaker, CloudWatch, Bedrock) is an optional production backend --
never a simulation.
"""


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings: Settings = get_settings()
    configure_from_settings(force=True)
    settings.paths.ensure()

    logger.info(
        "api.starting",
        extra={
            "environment": settings.environment,
            "version": settings.version,
            "auth_backend": settings.security.auth_backend,
            "deployment_provider": settings.deployment.provider,
            "registry_backend": settings.tracking.registry_backend,
            "aws_enabled": settings.aws.enabled,
        },
    )

    get_database()  # create/migrate the schema before serving

    # Background training tasks do not survive a restart. Any run still
    # marked in flight belongs to a process that is gone, so retire it rather
    # than showing progress that nothing is making.
    from app.training.jobs import get_training_run_store

    get_training_run_store().reconcile_orphans()

    from app.automl.runner import get_automl_store

    get_automl_store().reconcile_orphans()

    _restore_serving_state(settings)
    _warm_models(settings)

    if settings.monitoring.metrics_enabled:
        from app.monitoring.resource_monitor import get_resource_monitor

        get_resource_monitor().start()

    yield

    if settings.monitoring.metrics_enabled:
        from app.monitoring.resource_monitor import get_resource_monitor

        get_resource_monitor().stop()
    logger.info("api.stopped")


def _warm_models(settings: Settings) -> None:
    """Run a throwaway prediction through every routed version.

    Keeps first-request latency off the SLO. Never fatal.
    """
    try:
        from app.deployment.local_provider import get_local_provider
        from app.deployment.model_cache import get_model_cache

        if settings.deployment.provider != "local":
            return
        routing = get_local_provider().routing(settings.deployment.endpoint_name)
        if routing is None:
            return
        cache = get_model_cache()
        for version in routing.versions:
            cache.warm(cache.get(routing.model_name, version))
    except Exception as exc:
        logger.warning("api.model_warmup_failed", extra={"error": str(exc)})


def _restore_serving_state(settings: Settings) -> None:
    """Re-apply the last known routing so a restart keeps serving.

    Without this, a container restart would leave the endpoint unrouted until
    someone re-deployed, even though the registry and the deployment record both
    know exactly what should be live.
    """
    try:
        from app.deployment.base import get_deployment_store
        from app.deployment.local_provider import get_local_provider
        from app.registry.factory import get_registry
        from app.schemas.deployment import TrafficSplit

        if settings.deployment.provider != "local":
            return

        store = get_deployment_store()
        deployment = store.active(settings.deployment.endpoint_name)
        provider = get_local_provider()

        if deployment is not None and deployment.traffic:
            provider.apply(
                deployment.endpoint_name,
                deployment.model_name,
                deployment.traffic_split(),
                shadow_version=deployment.shadow_version,
            )
            logger.info(
                "api.routing_restored",
                extra={
                    "endpoint": deployment.endpoint_name,
                    "traffic": deployment.traffic,
                    "shadow_version": deployment.shadow_version,
                },
            )
            return

        model_name = settings.tracking.registered_model_name
        serving = get_registry().get_serving(model_name)
        if serving is not None:
            provider.apply(
                settings.deployment.endpoint_name,
                model_name,
                TrafficSplit.all_to(serving.version),
            )
            logger.info(
                "api.routing_bootstrapped_from_registry",
                extra={
                    "model": model_name,
                    "version": serving.version,
                    "stage": serving.stage.value,
                },
            )
        else:
            logger.warning(
                "api.no_model_available",
                extra={
                    "model": model_name,
                    "impact": "/api/v1/predict will return 503 until a model is promoted",
                    "fix": "run 'make demo' or 'make train && make promote'",
                },
            )
    except Exception as exc:
        # Startup must not crash because a model could not be warmed; readiness
        # will report the endpoint as not ready instead.
        logger.error(
            "api.routing_restore_failed",
            extra={"error": str(exc), "impact": "endpoint starts unrouted"},
        )


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    if settings.git_commit == "unknown":
        object.__setattr__(settings, "git_commit", git_commit())

    app = FastAPI(
        title="FMOps Platform",
        description=DESCRIPTION,
        version=settings.version,
        lifespan=lifespan,
        root_path=settings.server.root_path,
        docs_url="/docs" if settings.server.docs_enabled else None,
        redoc_url="/redoc" if settings.server.docs_enabled else None,
        openapi_url="/openapi.json" if settings.server.docs_enabled else None,
    )

    if settings.security.cors_allow_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.security.cors_allow_origins,
            allow_credentials=False,
            allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
            allow_headers=["*", settings.security.api_key_header],
        )

    auth_state = AuthMiddlewareState(settings)
    app.state.auth = auth_state
    app.state.settings = settings

    @app.middleware("http")
    async def _observability(request: Request, call_next):
        request_id = request.headers.get("X-Request-ID") or new_request_id()
        request.state.request_id = request_id
        token = bind_context(request_id=request_id)

        try:
            principal = auth_state.authenticate(request)
        except AuthenticationError as exc:
            logger.warning(
                "api.unauthenticated",
                extra={"path": request.url.path, "method": request.method},
            )
            clear_context()
            return JSONResponse(
                status_code=exc.http_status,
                content=error_body(exc.code, exc.message, exc.details),
                headers={"X-Request-ID": request_id},
            )

        request.state.principal = principal
        bind_context(actor=principal.subject)

        try:
            with Timer() as timer:
                response = await call_next(request)
        except Exception:
            record_http(request.method, _route_label(request), 500, 0.0)
            raise
        finally:
            from app.core.logging import reset_context

            reset_context(token)

        response.headers["X-Request-ID"] = request_id
        record_http(request.method, _route_label(request), response.status_code, timer.elapsed)
        if request.url.path not in ("/metrics", "/health/live"):
            logger.info(
                "api.request",
                extra={
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "status": response.status_code,
                    "duration_ms": round(timer.elapsed_ms, 2),
                    "actor": principal.subject,
                },
            )
        return response

    register_exception_handlers(app)

    app.include_router(health.router)
    app.include_router(monitoring.router)
    app.include_router(predictions.router)
    app.include_router(models.router)
    app.include_router(deployments.router)
    # datasets before experiments: experiments owns GET /api/v1/datasets/{version},
    # which is a single-segment catch-all under the same prefix and would
    # otherwise swallow literal paths like /api/v1/datasets/limits. Nothing in
    # experiments is shadowed in return -- datasets declares no catch-all.
    app.include_router(datasets.router)
    app.include_router(experiments.router)
    app.include_router(retraining.router)
    app.include_router(automl.router)
    app.include_router(training.router)
    app.include_router(llm.router)

    # The console is a handful of static assets rather than one giant inlined
    # file. Mounted read-only from inside the image; the auth middleware
    # already treats /static as public, and nothing here is generated per
    # request, so this adds no work to any API path.
    from fastapi.staticfiles import StaticFiles

    static_dir = Path(__file__).resolve().parent / "static"
    if static_dir.is_dir():
        app.mount("/static", StaticFiles(directory=static_dir), name="static")
    else:  # pragma: no cover - only when the image is built wrong
        logger.error("api.static_missing", extra={"path": str(static_dir)})

    from app.api.routes import dashboard

    app.include_router(dashboard.router)

    @app.get("/", tags=["meta"], summary="Service index")
    def index() -> dict[str, Any]:
        return {
            "service": settings.service_name,
            "version": settings.version,
            "environment": settings.environment,
            "docs": "/docs" if settings.server.docs_enabled else None,
            "dashboard": "/dashboard",
            "endpoints": {
                "health": "/health",
                "predict": "/api/v1/predict",
                "model": "/api/v1/model",
                "models": "/api/v1/models",
                "deployments": "/api/v1/deployments",
                "experiments": "/api/v1/experiments",
                "drift": "/api/v1/drift",
                "monitoring": "/api/v1/monitoring/summary",
                "retraining": "/api/v1/retraining",
                "llm": "/api/v1/llm/generate",
                "metrics": "/metrics",
            },
        }

    @app.get("/api/v1/config", tags=["meta"], summary="Effective configuration (redacted)")
    def effective_config() -> dict[str, Any]:
        """The resolved configuration, with every secret redacted."""
        return get_settings().redacted()

    return app


def _route_label(request: Request) -> str:
    """Templated route path, so metric cardinality stays bounded.

    Using the raw path would create one time series per model id or deployment
    id, which is the classic way to melt a Prometheus instance.
    """
    route = request.scope.get("route")
    return getattr(route, "path", request.url.path)


app = create_app()
