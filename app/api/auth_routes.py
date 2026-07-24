# app/api/auth_routes.py
from __future__ import annotations

import logging
import time
from collections import defaultdict

from fastapi import APIRouter, Depends, HTTPException, Request, Response

from app.api.auth_deps import get_current_user, get_user_store, off_mode_user
from app.config import Settings, get_settings
from app.schemas import ChangePasswordRequest, LoginRequest, MeResponse, QuotaStatusResponse, SetupRequest
from app.services.auth.identity import AuthenticatedUser, LocalPasswordProvider, authenticated_user_from_record
from app.services.auth.passwords import generate_salt, hash_password, verify_password
from app.services.auth.permissions import Role
from app.services.auth.quotas import QuotaService, QuotaStatus
from app.services.auth.sessions import SESSION_COOKIE_NAME, SESSION_TTL_SECONDS, create_session_cookie_value
from app.services.auth.user_store import UserStore

router = APIRouter(prefix="/api/v1/auth", tags=["auth"])
logger = logging.getLogger(__name__)

LOGIN_RATE_LIMIT_ATTEMPTS = 5
LOGIN_RATE_LIMIT_WINDOW_SECONDS = 60
_login_attempts: dict[str, list[float]] = defaultdict(list)


def get_identity_provider(request: Request) -> LocalPasswordProvider:
    return request.app.state.identity_provider


def get_quota_service(request: Request) -> QuotaService:
    return request.app.state.quota_service


def _client_ip(request: Request) -> str:
    return request.client.host if request.client is not None else "unknown"


def _check_login_rate_limit(client_ip: str) -> None:
    now = time.monotonic()
    attempts = [t for t in _login_attempts[client_ip] if now - t < LOGIN_RATE_LIMIT_WINDOW_SECONDS]
    _login_attempts[client_ip] = attempts
    if len(attempts) >= LOGIN_RATE_LIMIT_ATTEMPTS:
        raise HTTPException(status_code=429, detail="Demasiados intentos de login. Probá de nuevo en un minuto.")


def _record_login_attempt(client_ip: str) -> None:
    _login_attempts[client_ip].append(time.monotonic())


def _quota_status_to_response(status: QuotaStatus) -> QuotaStatusResponse:
    return QuotaStatusResponse(
        max_concurrent=status.max_concurrent, max_queued=status.max_queued,
        max_jobs_per_day=status.max_jobs_per_day, max_gpu_seconds_per_day=status.max_gpu_seconds_per_day,
        used_jobs_today=status.used_jobs_today, used_gpu_seconds_today=status.used_gpu_seconds_today,
    )


def _me_response(user: AuthenticatedUser, settings: Settings, quotas: QuotaService) -> MeResponse:
    return MeResponse(
        user_id=user.id, username=user.username, role=user.role.value,
        permissions=sorted(p.value for p in user.permissions),
        must_change_password=user.must_change_password, auth_mode=settings.auth_mode,
        quota=_quota_status_to_response(quotas.status_for(user)),
    )


@router.post("/login")
async def login(
    request: Request,
    payload: LoginRequest,
    response: Response,
    settings: Settings = Depends(get_settings),
    identity_provider: LocalPasswordProvider = Depends(get_identity_provider),
    user_store: UserStore = Depends(get_user_store),
) -> dict[str, bool]:
    client_ip = _client_ip(request)
    _check_login_rate_limit(client_ip)
    _record_login_attempt(client_ip)
    identity = identity_provider.authenticate(payload.username, payload.password)
    if identity is None:
        raise HTTPException(status_code=401, detail="Usuario o contraseña incorrectos")
    user = user_store.get(identity.user_id)
    if user is None:
        raise HTTPException(status_code=401, detail="Usuario o contraseña incorrectos")
    cookie_value = create_session_cookie_value(user.id, user.session_ver, settings.auth_secret or "")
    response.set_cookie(
        SESSION_COOKIE_NAME, cookie_value, max_age=SESSION_TTL_SECONDS,
        httponly=True, samesite="lax", secure=False,
    )
    return {"ok": True}


@router.post("/logout")
async def logout(response: Response) -> dict[str, bool]:
    response.delete_cookie(SESSION_COOKIE_NAME)
    return {"ok": True}


@router.post("/logout-all")
async def logout_all(
    response: Response,
    current_user: AuthenticatedUser = Depends(get_current_user),
    user_store: UserStore = Depends(get_user_store),
) -> dict[str, bool]:
    if current_user.id is not None:
        user_store.bump_session_ver(current_user.id)
    response.delete_cookie(SESSION_COOKIE_NAME)
    return {"ok": True}


@router.get("/me", response_model=MeResponse)
async def me(
    request: Request,
    response: Response,
    settings: Settings = Depends(get_settings),
    user_store: UserStore = Depends(get_user_store),
    quotas: QuotaService = Depends(get_quota_service),
) -> MeResponse:
    if settings.auth_mode == "off":
        return _me_response(off_mode_user(), settings, quotas)
    cookie_value = request.cookies.get(SESSION_COOKIE_NAME)
    user = None
    if cookie_value is not None:
        from app.services.auth.sessions import verify_session_cookie

        payload = verify_session_cookie(cookie_value, settings.auth_secret or "")
        if payload is not None:
            candidate = user_store.get(payload.user_id)
            if candidate is not None and not candidate.disabled and candidate.session_ver == payload.session_ver:
                user = candidate
    if user is None:
        setup_required = user_store.is_empty()
        raise HTTPException(
            status_code=401, detail="setup_required" if setup_required else "not_authenticated"
        )
    response.set_cookie(
        SESSION_COOKIE_NAME, create_session_cookie_value(user.id, user.session_ver, settings.auth_secret or ""),
        max_age=SESSION_TTL_SECONDS, httponly=True, samesite="lax", secure=False,
    )
    return _me_response(authenticated_user_from_record(user), settings, quotas)


@router.post("/change-password")
async def change_password(
    payload: ChangePasswordRequest,
    current_user: AuthenticatedUser = Depends(get_current_user),
    user_store: UserStore = Depends(get_user_store),
) -> dict[str, bool]:
    if current_user.id is None:
        raise HTTPException(status_code=400, detail="No disponible en modo single-user")
    user = user_store.get(current_user.id)
    if user is None or not verify_password(payload.current_password, user.password_hash, user.salt):
        raise HTTPException(status_code=401, detail="Contraseña actual incorrecta")
    salt = generate_salt()
    password_hash = hash_password(payload.new_password, salt)
    user_store.set_password(current_user.id, password_hash, salt, must_change_password=False)
    return {"ok": True}


@router.post("/setup")
async def setup(
    payload: SetupRequest,
    settings: Settings = Depends(get_settings),
    user_store: UserStore = Depends(get_user_store),
) -> dict[str, bool]:
    if settings.auth_mode != "multi":
        raise HTTPException(status_code=403, detail="Setup solo disponible en AUTH_MODE=multi")
    if not user_store.is_empty():
        raise HTTPException(status_code=403, detail="Setup ya fue completado")
    salt = generate_salt()
    password_hash = hash_password(payload.password, salt)
    user_store.create(
        username=payload.username, password_hash=password_hash, salt=salt,
        role=Role.admin, must_change_password=False,
    )
    return {"ok": True}
