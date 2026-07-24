from __future__ import annotations

from enum import Enum


class Role(str, Enum):
    admin = "admin"
    user = "user"


class Permission(str, Enum):
    jobs_create = "jobs:create"
    jobs_read_own = "jobs:read_own"
    jobs_cancel_own = "jobs:cancel_own"
    jobs_read_all = "jobs:read_all"
    jobs_cancel_any = "jobs:cancel_any"
    models_install = "models:install"
    models_delete = "models:delete"
    users_manage = "users:manage"
    settings_read = "settings:read"
    settings_write = "settings:write"
    devices_read = "devices:read"
    queue_read_anonymized = "queue:read_anonymized"


USER_PERMISSIONS = frozenset({
    Permission.jobs_create,
    Permission.jobs_read_own,
    Permission.jobs_cancel_own,
    Permission.settings_read,
    Permission.devices_read,
    Permission.queue_read_anonymized,
})

ROLE_PERMISSIONS: dict[Role, frozenset[Permission]] = {
    Role.user: USER_PERMISSIONS,
    Role.admin: frozenset(Permission),
}


def role_has_permission(role: Role, permission: Permission) -> bool:
    return permission in ROLE_PERMISSIONS[role]
