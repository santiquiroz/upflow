from __future__ import annotations

import types
from pathlib import Path

import pytest
from fastapi import HTTPException

from app.api.auth_deps import current_user_from_request, get_current_user, require
from app.config import Settings
from app.services.auth.identity import AuthenticatedUser
from app.services.auth.passwords import generate_salt, hash_password
from app.services.auth.permissions import Permission, ROLE_PERMISSIONS, Role
from app.services.auth.sessions import SESSION_COOKIE_NAME, create_session_cookie_value
from app.services.auth.user_store import UserStore


def make_settings(tmp_path: Path, **overrides: object) -> Settings:
    kwargs: dict[str, object] = {"RUNTIME_DIR": str(tmp_path / "runtime")}
    kwargs.update(overrides)
    return Settings(_env_file=None, **kwargs)


class FakeRequest:
    def __init__(self, cookies: dict[str, str] | None = None) -> None:
        self.cookies = cookies or {}
        self.state = types.SimpleNamespace()


class FakeResponse:
    def __init__(self) -> None:
        self.cookies_set: list[tuple[str, str]] = []

    def set_cookie(self, key: str, value: str, **kwargs: object) -> None:
        self.cookies_set.append((key, value))


async def test_get_current_user_returns_pseudo_admin_when_off(tmp_path: Path) -> None:
    settings = make_settings(tmp_path, AUTH_MODE="off")
    user_store = UserStore(settings)

    user = await get_current_user(FakeRequest(), FakeResponse(), settings, user_store)

    assert user.id is None
    assert user.role == Role.admin
    assert Permission.users_manage in user.permissions


async def test_get_current_user_raises_401_without_cookie_when_multi(tmp_path: Path) -> None:
    settings = make_settings(tmp_path, AUTH_MODE="multi", AUTH_SECRET="s" * 32)
    user_store = UserStore(settings)

    with pytest.raises(HTTPException) as exc_info:
        await get_current_user(FakeRequest(), FakeResponse(), settings, user_store)

    assert exc_info.value.status_code == 401


async def test_get_current_user_accepts_valid_cookie_when_multi(tmp_path: Path) -> None:
    settings = make_settings(tmp_path, AUTH_MODE="multi", AUTH_SECRET="s" * 32)
    user_store = UserStore(settings)
    salt = generate_salt()
    stored = user_store.create(username="alice", password_hash=hash_password("pw", salt), salt=salt, role=Role.user)
    cookie = create_session_cookie_value(stored.id, stored.session_ver, settings.auth_secret)

    user = await get_current_user(FakeRequest({SESSION_COOKIE_NAME: cookie}), FakeResponse(), settings, user_store)

    assert user.id == stored.id
    assert user.role == Role.user


async def test_get_current_user_rejects_cookie_after_logout_all(tmp_path: Path) -> None:
    settings = make_settings(tmp_path, AUTH_MODE="multi", AUTH_SECRET="s" * 32)
    user_store = UserStore(settings)
    salt = generate_salt()
    stored = user_store.create(username="bob", password_hash=hash_password("pw", salt), salt=salt, role=Role.user)
    cookie = create_session_cookie_value(stored.id, stored.session_ver, settings.auth_secret)
    user_store.bump_session_ver(stored.id)

    with pytest.raises(HTTPException) as exc_info:
        await get_current_user(FakeRequest({SESSION_COOKIE_NAME: cookie}), FakeResponse(), settings, user_store)

    assert exc_info.value.status_code == 401


async def test_get_current_user_rejects_disabled_user(tmp_path: Path) -> None:
    settings = make_settings(tmp_path, AUTH_MODE="multi", AUTH_SECRET="s" * 32)
    user_store = UserStore(settings)
    salt = generate_salt()
    stored = user_store.create(username="carol", password_hash=hash_password("pw", salt), salt=salt, role=Role.user)
    user_store.set_disabled(stored.id, True)
    cookie = create_session_cookie_value(stored.id, stored.session_ver, settings.auth_secret)

    with pytest.raises(HTTPException) as exc_info:
        await get_current_user(FakeRequest({SESSION_COOKIE_NAME: cookie}), FakeResponse(), settings, user_store)

    assert exc_info.value.status_code == 401


async def test_get_current_user_stashes_resolved_user_on_request_state(tmp_path: Path) -> None:
    settings = make_settings(tmp_path, AUTH_MODE="off")
    user_store = UserStore(settings)
    request = FakeRequest()

    user = await get_current_user(request, FakeResponse(), settings, user_store)

    assert request.state.current_user is user


def test_current_user_from_request_returns_none_when_request_is_none() -> None:
    assert current_user_from_request(None) is None


def test_current_user_from_request_reads_request_state() -> None:
    request = FakeRequest()
    principal = AuthenticatedUser(
        id="u1", username="alice", role=Role.user, permissions=ROLE_PERMISSIONS[Role.user],
        must_change_password=False, quota_overrides={},
    )
    request.state.current_user = principal

    assert current_user_from_request(request) is principal


async def test_require_raises_403_when_permission_missing(tmp_path: Path) -> None:
    principal = AuthenticatedUser(
        id="u1", username="alice", role=Role.user, permissions=ROLE_PERMISSIONS[Role.user],
        must_change_password=False, quota_overrides={},
    )
    dependency = require(Permission.users_manage)

    with pytest.raises(HTTPException) as exc_info:
        await dependency(current_user=principal)

    assert exc_info.value.status_code == 403


async def test_require_raises_403_when_must_change_password(tmp_path: Path) -> None:
    principal = AuthenticatedUser(
        id="u1", username="alice", role=Role.admin, permissions=ROLE_PERMISSIONS[Role.admin],
        must_change_password=True, quota_overrides={},
    )
    dependency = require(Permission.users_manage)

    with pytest.raises(HTTPException) as exc_info:
        await dependency(current_user=principal)

    assert exc_info.value.status_code == 403
    assert exc_info.value.detail == "password_change_required"


async def test_require_returns_user_when_permission_present(tmp_path: Path) -> None:
    principal = AuthenticatedUser(
        id="u1", username="alice", role=Role.admin, permissions=ROLE_PERMISSIONS[Role.admin],
        must_change_password=False, quota_overrides={},
    )
    dependency = require(Permission.users_manage)

    result = await dependency(current_user=principal)

    assert result is principal
