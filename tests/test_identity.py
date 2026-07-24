from __future__ import annotations

from pathlib import Path

from app.config import Settings
from app.services.auth.identity import (
    LocalPasswordProvider,
    authenticated_user_from_record,
)
from app.services.auth.passwords import generate_salt, hash_password
from app.services.auth.permissions import ROLE_PERMISSIONS, Permission, Role
from app.services.auth.user_store import UserStore


def make_settings(tmp_path: Path) -> Settings:
    return Settings(_env_file=None, RUNTIME_DIR=str(tmp_path / "runtime"))


def make_store_with_user(tmp_path: Path, *, username="alice", password="hunter22", role=Role.user, disabled=False):
    store = UserStore(make_settings(tmp_path))
    salt = generate_salt()
    user = store.create(username=username, password_hash=hash_password(password, salt), salt=salt, role=role)
    if disabled:
        user = store.set_disabled(user.id, True)
    return store, user


def test_authenticate_returns_identity_for_correct_credentials(tmp_path: Path) -> None:
    store, user = make_store_with_user(tmp_path)
    provider = LocalPasswordProvider(store)

    identity = provider.authenticate("alice", "hunter22")

    assert identity is not None
    assert identity.user_id == user.id
    assert identity.username == "alice"
    assert identity.role == Role.user


def test_authenticate_returns_none_for_wrong_password(tmp_path: Path) -> None:
    store, _user = make_store_with_user(tmp_path)
    provider = LocalPasswordProvider(store)

    assert provider.authenticate("alice", "wrong-password") is None


def test_authenticate_returns_none_for_unknown_username(tmp_path: Path) -> None:
    store, _user = make_store_with_user(tmp_path)
    provider = LocalPasswordProvider(store)

    assert provider.authenticate("nobody", "hunter22") is None


def test_authenticate_returns_none_for_disabled_user(tmp_path: Path) -> None:
    store, _user = make_store_with_user(tmp_path, disabled=True)
    provider = LocalPasswordProvider(store)

    assert provider.authenticate("alice", "hunter22") is None


def test_authenticated_user_from_record_copies_role_permissions(tmp_path: Path) -> None:
    _store, user = make_store_with_user(tmp_path, role=Role.admin)

    principal = authenticated_user_from_record(user)

    assert principal.id == user.id
    assert principal.role == Role.admin
    assert principal.permissions == ROLE_PERMISSIONS[Role.admin]
    assert Permission.users_manage in principal.permissions


def test_authenticated_user_from_record_copies_quota_overrides(tmp_path: Path) -> None:
    store, user = make_store_with_user(tmp_path)
    user = store.set_quota_overrides(user.id, {"max_concurrent": 2})

    principal = authenticated_user_from_record(user)

    assert principal.quota_overrides == {"max_concurrent": 2}
