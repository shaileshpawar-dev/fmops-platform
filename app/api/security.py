"""API authentication.

An interface with two implementations:

``none``
    Development default. Every request is an anonymous principal. The app logs a
    warning at startup if this is active in a non-development environment, so a
    misconfigured production deployment is loud rather than silent.

``api_key``
    Shared-secret keys supplied via ``FMOPS_SECURITY__API_KEYS`` (never in YAML,
    never in git). Compared with :func:`secrets.compare_digest` to avoid timing
    leaks.

This is deliberately a *placeholder interface* for real identity: production
would plug in OIDC/JWT verification or an API gateway authorizer by adding one
:class:`AuthBackend`. The rest of the API only depends on the abstraction --
see ``docs/deployment.md`` for the intended production posture.
"""

from __future__ import annotations

import secrets
from abc import ABC, abstractmethod
from dataclasses import dataclass

from fastapi import Request

from app.core.config import Settings, get_settings
from app.core.exceptions import AuthenticationError
from app.core.logging import get_logger

logger = get_logger(__name__)

# Endpoints reachable without credentials even when auth is enabled: liveness,
# readiness and the Prometheus scrape (which is protected at the network layer).
PUBLIC_PATHS = frozenset(
    {
        "/health",
        "/health/live",
        "/health/ready",
        "/metrics",
        "/docs",
        "/openapi.json",
        "/redoc",
    }
)

WRITE_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})


@dataclass(frozen=True)
class Principal:
    """Who is making the request."""

    subject: str
    authenticated: bool = False
    scopes: tuple[str, ...] = ()

    @property
    def is_anonymous(self) -> bool:
        return not self.authenticated


ANONYMOUS = Principal(subject="anonymous", authenticated=False)


class AuthBackend(ABC):
    name: str = "abstract"

    @abstractmethod
    def authenticate(self, request: Request) -> Principal: ...


class NoAuthBackend(AuthBackend):
    """Everything is anonymous. Development only."""

    name = "none"

    def authenticate(self, request: Request) -> Principal:
        return ANONYMOUS


class ApiKeyBackend(AuthBackend):
    """Shared-secret API keys presented in a header."""

    name = "api_key"

    def __init__(self, keys: list[str], header: str = "X-API-Key") -> None:
        self.header = header
        self._keys = [k for k in keys if k]
        if not self._keys:
            logger.error(
                "auth.no_api_keys_configured",
                extra={
                    "impact": "every authenticated request will be rejected",
                    "fix": "set FMOPS_SECURITY__API_KEYS",
                },
            )

    def authenticate(self, request: Request) -> Principal:
        presented = request.headers.get(self.header)
        if not presented:
            raise AuthenticationError(
                f"missing {self.header} header",
                header=self.header,
            )
        for known in self._keys:
            if secrets.compare_digest(presented, known):
                # Never log the key itself, only a short non-reversible tag.
                return Principal(
                    subject=f"key:{_key_tag(known)}",
                    authenticated=True,
                    scopes=("read", "write"),
                )
        raise AuthenticationError("invalid API key", header=self.header)


def _key_tag(key: str) -> str:
    from app.core.utils import hash_text

    return hash_text(key)[:8]


def build_auth_backend(settings: Settings | None = None) -> AuthBackend:
    settings = settings or get_settings()
    if settings.security.auth_backend == "api_key":
        return ApiKeyBackend(settings.security.api_keys, settings.security.api_key_header)
    if settings.is_production:
        logger.error(
            "auth.disabled_in_production",
            extra={
                "auth_backend": "none",
                "impact": "the API accepts unauthenticated writes",
                "fix": "set FMOPS_SECURITY__AUTH_BACKEND=api_key and supply keys",
            },
        )
    return NoAuthBackend()


class AuthMiddlewareState:
    """Holds the backend and decides which requests need credentials."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.backend = build_auth_backend(self.settings)

    def requires_auth(self, method: str, path: str) -> bool:
        if self.backend.name == "none":
            return False
        if path in PUBLIC_PATHS or path.startswith("/static"):
            return False
        if self.settings.security.require_auth_for_writes:
            return method.upper() in WRITE_METHODS
        return True

    def authenticate(self, request: Request) -> Principal:
        if not self.requires_auth(request.method, request.url.path):
            return ANONYMOUS
        return self.backend.authenticate(request)
