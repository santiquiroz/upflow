from __future__ import annotations

from fastapi import Depends, HTTPException, Request, Response

from app.config import Settings, get_settings
from app.services.auth.identity import AuthenticatedUser, authenticated_user_from_record
from app.services.auth.permissions import ROLE_PERMISSIONS, Permission, Role
from app.services.auth.sessions import (
    SESSION_COOKIE_NAME,
    SESSION_TTL_SECONDS,
    create_session_cookie_value,
    verify_session_cookie,
)
from app.services.auth.user_store import User, UserStore

OFF_MODE_USERNAME = "local"


def get_user_store(request: Request) -> UserStore:
    return request.app.state.user_store


def off_mode_user() -> AuthenticatedUser:
    return AuthenticatedUser(
        id=None, username=OFF_MODE_USERNAME, role=Role.admin,
        permissions=ROLE_PERMISSIONS[Role.admin], must_change_password=False, quota_overrides={},
    )


def resolve_session_user(request: Request, secret: str | None, user_store: UserStore) -> User | None:
    cookie_value = request.cookies.get(SESSION_COOKIE_NAME)
    if cookie_value is None or not secret:
        return None
    payload = verify_session_cookie(cookie_value, secret)
    if payload is None:
        return None
    user = user_store.get(payload.user_id)
    if user is None or user.disabled or user.session_ver != payload.session_ver:
        return None
    return user


def _renew_session_cookie(response: Response, user: User, secret: str) -> None:
    value = create_session_cookie_value(user.id, user.session_ver, secret)
    response.set_cookie(
        SESSION_COOKIE_NAME, value, max_age=SESSION_TTL_SECONDS,
        httponly=True, samesite="lax", secure=False,
    )


async def get_current_user(
    request: Request,
    response: Response,
    settings: Settings = Depends(get_settings),
    user_store: UserStore = Depends(get_user_store),
) -> AuthenticatedUser:
    if settings.auth_mode == "off":
        resolved = off_mode_user()
    else:
        user = resolve_session_user(request, settings.auth_secret, user_store)
        if user is None:
            raise HTTPException(status_code=401, detail="No autenticado")
        _renew_session_cookie(response, user, settings.auth_secret or "")
        resolved = authenticated_user_from_record(user)
    request.state.current_user = resolved
    return resolved


def require(permission: Permission):
    async def _dependency(current_user: AuthenticatedUser = Depends(get_current_user)) -> AuthenticatedUser:
        if current_user.must_change_password:
            raise HTTPException(status_code=403, detail="password_change_required")
        if permission not in current_user.permissions:
            raise HTTPException(status_code=403, detail="No tenés permiso para esta acción")
        return current_user
    return _dependency


def current_user_from_request(request: Request | None) -> AuthenticatedUser | None:
    if request is None:
        return None
    return getattr(request.state, "current_user", None)
