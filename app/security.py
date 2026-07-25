from __future__ import annotations

from urllib.parse import urlparse

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.types import ASGIApp

STATE_CHANGING_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})


def _origin_from_referer(referer: str) -> str:
    parsed = urlparse(referer)
    if not parsed.scheme or not parsed.netloc:
        return ""
    return f"{parsed.scheme}://{parsed.netloc}"


def is_origin_allowed(
    method: str,
    origin_header: str | None,
    referer_header: str | None,
    allowed_origins: frozenset[str],
) -> bool:
    """Decides whether a request may proceed based on Origin/Referer.

    Only state-changing methods are checked. Requests with neither header
    (curl, same-origin form GET) are allowed through, matching the threat
    model of blocking cross-origin browser-driven POSTs, not scripted clients.
    """
    if method.upper() not in STATE_CHANGING_METHODS:
        return True
    if not origin_header and not referer_header:
        return True
    candidate = origin_header or _origin_from_referer(referer_header or "")
    if not candidate:
        return False
    return candidate in allowed_origins


class OriginGuardMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: ASGIApp, allowed_origins: frozenset[str]) -> None:
        super().__init__(app)
        self.allowed_origins = allowed_origins

    async def dispatch(self, request: Request, call_next):
        allowed = is_origin_allowed(
            request.method,
            request.headers.get("origin"),
            request.headers.get("referer"),
            self.allowed_origins,
        )
        if not allowed:
            return JSONResponse({"detail": "Cross-origin request rejected"}, status_code=403)
        return await call_next(request)


LOOPBACK_HOSTS = frozenset({"127.0.0.1", "::1", "localhost", "testclient"})


def is_loopback_host(client_host: str | None) -> bool:
    return client_host in LOOPBACK_HOSTS


class LoopbackGuardMiddleware(BaseHTTPMiddleware):
    """When AUTH_MODE=off, rejects any request whose peer isn't loopback --
    the guardrail from the approved spec that makes it impossible to expose
    the single-user desktop app to the network by accident (Jupyter/
    code-server pattern). No-op entirely when AUTH_MODE=multi."""

    def __init__(self, app: ASGIApp, auth_mode: str) -> None:
        super().__init__(app)
        self.auth_mode = auth_mode

    async def dispatch(self, request: Request, call_next):
        if self.auth_mode != "off":
            return await call_next(request)
        client = request.client
        if client is not None and not is_loopback_host(client.host):
            return JSONResponse(
                {
                    "detail": (
                        "Upflow está en modo single-user. "
                        "Activá AUTH_MODE=multi para acceso remoto."
                    )
                },
                status_code=403,
            )
        return await call_next(request)
