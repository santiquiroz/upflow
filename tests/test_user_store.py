from __future__ import annotations

from pathlib import Path

import pytest

from app.config import Settings
from app.services.auth.permissions import Role
from app.services.auth.user_store import UserStore


def make_settings(tmp_path: Path, **overrides: object) -> Settings:
    kwargs: dict[str, object] = {"RUNTIME_DIR": str(tmp_path / "runtime")}
    kwargs.update(overrides)
    return Settings(_env_file=None, **kwargs)


def test_new_store_is_empty(tmp_path: Path) -> None:
    store = UserStore(make_settings(tmp_path))
    assert store.is_empty() is True
    assert store.list() == []


def test_create_persists_user_and_returns_it(tmp_path: Path) -> None:
    store = UserStore(make_settings(tmp_path))

    user = store.create(username="alice", password_hash="hash", salt="salt", role=Role.admin)

    assert user.username == "alice"
    assert user.role == Role.admin
    assert user.must_change_password is True
    assert store.get(user.id) is user or store.get(user.id).id == user.id


def test_create_rejects_duplicate_username(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    store = UserStore(settings)
    store.create(username="alice", password_hash="h", salt="s", role=Role.user)

    with pytest.raises(ValueError, match="already exists"):
        store.create(username="alice", password_hash="h2", salt="s2", role=Role.user)


def test_create_first_admin_creates_admin_when_store_is_empty(tmp_path: Path) -> None:
    store = UserStore(make_settings(tmp_path))

    user = store.create_first_admin(username="admin", password_hash="hash", salt="salt")

    assert user.username == "admin"
    assert user.role == Role.admin
    assert user.must_change_password is False
    assert store.get(user.id).username == "admin"


def test_create_first_admin_rejects_when_store_already_has_users(tmp_path: Path) -> None:
    store = UserStore(make_settings(tmp_path))
    store.create(username="alice", password_hash="h", salt="s", role=Role.user)

    with pytest.raises(ValueError, match="already been completed"):
        store.create_first_admin(username="admin", password_hash="hash", salt="salt")


def test_get_by_username_finds_existing_user(tmp_path: Path) -> None:
    store = UserStore(make_settings(tmp_path))
    store.create(username="bob", password_hash="h", salt="s", role=Role.user)

    found = store.get_by_username("bob")

    assert found is not None
    assert found.username == "bob"


def test_get_by_username_returns_none_when_missing(tmp_path: Path) -> None:
    store = UserStore(make_settings(tmp_path))
    assert store.get_by_username("nobody") is None


def test_persists_across_instances(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    store = UserStore(settings)
    user = store.create(username="carol", password_hash="h", salt="s", role=Role.user)

    reloaded = UserStore(settings)

    assert reloaded.get(user.id) is not None
    assert reloaded.get(user.id).username == "carol"


def test_set_password_updates_hash_and_flag(tmp_path: Path) -> None:
    store = UserStore(make_settings(tmp_path))
    user = store.create(username="dave", password_hash="old", salt="old-salt", role=Role.user)

    updated = store.set_password(user.id, "new-hash", "new-salt", must_change_password=False)

    assert updated.password_hash == "new-hash"
    assert updated.salt == "new-salt"
    assert updated.must_change_password is False


def test_set_role_updates_role(tmp_path: Path) -> None:
    store = UserStore(make_settings(tmp_path))
    user = store.create(username="erin", password_hash="h", salt="s", role=Role.user)

    updated = store.set_role(user.id, Role.admin)

    assert updated.role == Role.admin


def test_set_disabled_flags_user(tmp_path: Path) -> None:
    store = UserStore(make_settings(tmp_path))
    user = store.create(username="frank", password_hash="h", salt="s", role=Role.user)

    updated = store.set_disabled(user.id, True)

    assert updated.disabled is True


def test_set_quota_overrides_stores_dict(tmp_path: Path) -> None:
    store = UserStore(make_settings(tmp_path))
    user = store.create(username="gina", password_hash="h", salt="s", role=Role.user)

    updated = store.set_quota_overrides(user.id, {"max_concurrent": 3})

    assert updated.quota_overrides == {"max_concurrent": 3}


def test_bump_session_ver_increments(tmp_path: Path) -> None:
    store = UserStore(make_settings(tmp_path))
    user = store.create(username="hank", password_hash="h", salt="s", role=Role.user)
    assert user.session_ver == 0

    updated = store.bump_session_ver(user.id)

    assert updated.session_ver == 1


def test_unknown_user_id_raises_value_error(tmp_path: Path) -> None:
    store = UserStore(make_settings(tmp_path))
    with pytest.raises(ValueError, match="Unknown user id"):
        store.set_disabled("does-not-exist", True)


def test_corrupt_users_file_is_backed_up_and_reset(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    settings.users_file_path.parent.mkdir(parents=True, exist_ok=True)
    settings.users_file_path.write_text("not valid json", encoding="utf-8")

    store = UserStore(settings)

    assert store.list() == []
    backups = list(settings.users_file_path.parent.glob("users.json.corrupt-*"))
    assert len(backups) == 1
