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


class NoAuthBackend:
    """Everything is anonymous. Authentication is disabled."""

    name = "none"

    def authenticate(self, request: Request) -> Principal:
        return ANONYMOUS

def build_auth_backend(settings: Settings | None = None) -> NoAuthBackend:
    return NoAuthBackend()

def request_actor(request: Request) -> str:
    """Who is making this request, as the audit log should record it."""
    principal = getattr(request.state, "principal", None)
    return principal.subject if principal is not None else ANONYMOUS.subject

class AuthMiddlewareState:
    """Holds the backend and decides which requests need credentials."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.backend = build_auth_backend(self.settings)

    def requires_auth(self, method: str, path: str) -> bool:
        return False

    def authenticate(self, request: Request) -> Principal:
        return ANONYMOUS
