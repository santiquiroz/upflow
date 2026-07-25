from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from app.services.auth.passwords import verify_password
from app.services.auth.permissions import ROLE_PERMISSIONS, Permission, Role
from app.services.auth.user_store import User, UserStore


@dataclass(frozen=True, slots=True)
class UserIdentity:
    user_id: str
    username: str
    role: Role
    external_subject: str | None


class IdentityProvider(Protocol):
    def authenticate(self, username: str, password: str) -> UserIdentity | None: ...


class LocalPasswordProvider:
    def __init__(self, user_store: UserStore) -> None:
        self._user_store = user_store

    def authenticate(self, username: str, password: str) -> UserIdentity | None:
        user = self._user_store.get_by_username(username)
        if user is None or user.disabled:
            return None
        if not verify_password(password, user.password_hash, user.salt):
            return None
        return UserIdentity(
            user_id=user.id, username=user.username, role=user.role,
            external_subject=user.external_subject,
        )


@dataclass(frozen=True, slots=True)
class AuthenticatedUser:
    """Request-scoped principal. `id` is `None` only for the AUTH_MODE=off
    pseudo-admin -- callers use that to skip quota/ownership bookkeeping
    entirely (see app/api/auth_deps.py and app/services/auth/quotas.py)."""

    id: str | None
    username: str
    role: Role
    permissions: frozenset[Permission]
    must_change_password: bool
    quota_overrides: dict[str, int]


def authenticated_user_from_record(user: User) -> AuthenticatedUser:
    return AuthenticatedUser(
        id=user.id, username=user.username, role=user.role,
        permissions=ROLE_PERMISSIONS[user.role], must_change_password=user.must_change_password,
        quota_overrides=user.quota_overrides,
    )
