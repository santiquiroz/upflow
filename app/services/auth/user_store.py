from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Any
from uuid import uuid4

from app.config import Settings
from app.models import utc_now
from app.services.auth.permissions import Role
from app.services.json_store import backup_corrupt_file, write_json_atomically

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class User:
    id: str
    username: str
    password_hash: str
    salt: str
    role: Role
    disabled: bool = False
    must_change_password: bool = False
    session_ver: int = 0
    external_subject: str | None = None
    quota_overrides: dict[str, int] = field(default_factory=dict)
    created_at: datetime = field(default_factory=utc_now)


def _user_to_json_dict(user: User) -> dict[str, Any]:
    return {
        "id": user.id,
        "username": user.username,
        "password_hash": user.password_hash,
        "salt": user.salt,
        "role": user.role.value,
        "disabled": user.disabled,
        "must_change_password": user.must_change_password,
        "session_ver": user.session_ver,
        "external_subject": user.external_subject,
        "quota_overrides": user.quota_overrides,
        "created_at": user.created_at.isoformat(),
    }


def _user_from_json_dict(data: dict[str, Any]) -> User:
    return User(
        id=data["id"],
        username=data["username"],
        password_hash=data["password_hash"],
        salt=data["salt"],
        role=Role(data["role"]),
        disabled=data.get("disabled", False),
        must_change_password=data.get("must_change_password", False),
        session_ver=data.get("session_ver", 0),
        external_subject=data.get("external_subject"),
        quota_overrides=data.get("quota_overrides", {}),
        created_at=datetime.fromisoformat(data["created_at"]),
    )


class UserStore:
    def __init__(self, settings: Settings) -> None:
        self._path = settings.users_file_path
        self._lock = threading.Lock()
        self._users: dict[str, User] = self._load()

    def list(self) -> list[User]:
        with self._lock:
            return list(self._users.values())

    def get(self, user_id: str) -> User | None:
        with self._lock:
            return self._users.get(user_id)

    def get_by_username(self, username: str) -> User | None:
        with self._lock:
            return next((u for u in self._users.values() if u.username == username), None)

    def is_empty(self) -> bool:
        with self._lock:
            return not self._users

    def create(
        self, *, username: str, password_hash: str, salt: str, role: Role,
        must_change_password: bool = True,
    ) -> User:
        with self._lock:
            if any(u.username == username for u in self._users.values()):
                raise ValueError(f"Username already exists: {username!r}")
            user = User(
                id=uuid4().hex, username=username, password_hash=password_hash, salt=salt,
                role=role, must_change_password=must_change_password,
            )
            self._users[user.id] = user
            self._persist()
            return user

    def set_password(self, user_id: str, password_hash: str, salt: str, *, must_change_password: bool) -> User:
        return self._update(
            user_id,
            lambda u: replace(u, password_hash=password_hash, salt=salt, must_change_password=must_change_password),
        )

    def set_role(self, user_id: str, role: Role) -> User:
        return self._update(user_id, lambda u: replace(u, role=role))

    def set_disabled(self, user_id: str, disabled: bool) -> User:
        return self._update(user_id, lambda u: replace(u, disabled=disabled))

    def set_quota_overrides(self, user_id: str, overrides: dict[str, int]) -> User:
        return self._update(user_id, lambda u: replace(u, quota_overrides=overrides))

    def bump_session_ver(self, user_id: str) -> User:
        return self._update(user_id, lambda u: replace(u, session_ver=u.session_ver + 1))

    def _update(self, user_id: str, mutate: Any) -> User:
        with self._lock:
            user = self._require_user(user_id)
            updated = mutate(user)
            self._users[user_id] = updated
            self._persist()
            return updated

    def _require_user(self, user_id: str) -> User:
        user = self._users.get(user_id)
        if user is None:
            raise ValueError(f"Unknown user id: {user_id!r}")
        return user

    def _load(self) -> dict[str, User]:
        if not self._path.exists():
            return {}
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
            return {item["id"]: _user_from_json_dict(item) for item in raw}
        except (json.JSONDecodeError, KeyError, ValueError, TypeError) as exc:
            backup_corrupt_file(self._path, exc, logger)
            return {}

    def _persist(self) -> None:
        payload = [_user_to_json_dict(user) for user in self._users.values()]
        write_json_atomically(self._path, payload)
