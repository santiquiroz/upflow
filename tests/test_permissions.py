from __future__ import annotations

from app.services.auth.permissions import ROLE_PERMISSIONS, Permission, Role, role_has_permission


def test_admin_has_every_permission() -> None:
    assert ROLE_PERMISSIONS[Role.admin] == frozenset(Permission)


def test_user_role_has_only_own_scoped_permissions() -> None:
    user_perms = ROLE_PERMISSIONS[Role.user]
    assert Permission.jobs_create in user_perms
    assert Permission.jobs_read_own in user_perms
    assert Permission.jobs_cancel_own in user_perms
    assert Permission.jobs_read_all not in user_perms
    assert Permission.jobs_cancel_any not in user_perms
    assert Permission.users_manage not in user_perms
    assert Permission.models_install not in user_perms
    assert Permission.models_delete not in user_perms


def test_role_has_permission_matches_table() -> None:
    assert role_has_permission(Role.admin, Permission.users_manage) is True
    assert role_has_permission(Role.user, Permission.users_manage) is False


def test_permission_values_match_spec_strings() -> None:
    expected = {
        "jobs:create", "jobs:read_own", "jobs:cancel_own", "jobs:read_all", "jobs:cancel_any",
        "models:install", "models:delete", "users:manage", "settings:read", "settings:write",
        "devices:read", "queue:read_anonymized",
    }
    assert {p.value for p in Permission} == expected
