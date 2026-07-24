# Multi-usuario y autenticación (subproyecto C) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add opt-in multi-user auth (sessions, roles, permissions, per-user quotas, job ownership, user management UI) to Upflow, with `AUTH_MODE=off` (default) preserving today's single-user behavior byte-for-byte.

**Architecture:** A new `app/services/auth/` module (permissions, passwords, sessions, user_store, identity, quotas) sits beside the existing job managers. `AUTH_MODE=off` resolves every request to a pseudo-admin with no cookie/session involved at all (existing code paths untouched); `AUTH_MODE=multi` requires a signed HttpOnly cookie, checked via FastAPI dependencies (`get_current_user`, `require(permission)`). Job ownership (`owner_id`) is added to all 4 job dataclasses and stamped by the route layer, never inferred deeper. Quota admission runs once per `create_job` call, before enqueue. New auth/users routers mirror the existing flat `app/api/*.py` + `app/schemas.py` conventions exactly. Frontend gets an `AuthProvider` (`GET /auth/me` on mount) gating routes and nav items by permission.

**Tech Stack:** FastAPI + pydantic-settings (existing), stdlib-only crypto (`hashlib.scrypt`, `hmac`, `secrets` — zero new Python dependencies), React + TanStack Query + `fetch` (existing frontend stack, no axios/MSW).

## Global Constraints

- `AUTH_MODE=off` (default) MUST leave the existing backend suite (1207 tests, `.venv\Scripts\python.exe -m pytest tests/ -q`) and frontend suite (447 tests, `cd frontend && npx vitest run && npx tsc --noEmit`) passing with **zero modifications to existing test files**. This is the hard non-regression gate for this plan — see the "Backward-compatibility mechanics" note below.
- Zero new pip/npm dependencies. Passwords: `hashlib.scrypt` (N=2^14, r=8, p=1, 16-byte urandom salt per user). Sessions: HMAC-SHA256 over a JSON payload, stdlib `hmac`/`hashlib`/`base64`/`json`.
- New JSON storage (`users.json`, `usage.json`) follows the existing `ModelRegistry` atomic-write pattern (`app/services/model_registry.py`): write to a temp file in the *same directory*, then `Path.replace()`. `ModelRegistry` itself is **not modified** by this plan (it's stable and fully tested); the atomic-write/corrupt-backup mechanics are extracted into a new shared `app/services/json_store.py` used only by new code.
- Commit messages, roles/permissions/quota copy, and error strings shown to users are in Spanish where the existing app already uses Spanish comments/strings for user-facing text (see the exact error strings quoted in each task below — use them verbatim, they come from the approved spec).
- Repo branch prefix `feature/` is enforced by a pre-commit hook — branch as `feature/multiuser-auth`.
- Backend tests: `.venv\Scripts\python.exe -m pytest tests/ -q` (the system Python lacks `optimum`). Frontend: `cd frontend && npx vitest run && npx tsc --noEmit`.
- SDD ledger: add your own `## Plan: Multi-usuario y autenticación` section to `.superpowers/sdd/progress.md` — do not touch other plans' entries there.

### Backward-compatibility mechanics (read before Tasks 20-24)

Sixteen existing test files (e.g. `tests/test_job_status_routes.py`, `tests/test_job_cancel.py`, `tests/test_generation_api.py`) import route handler functions from `app/api/routes.py` and call them **directly as plain coroutines** with today's exact keyword arguments (bypassing FastAPI's dependency-injection machinery entirely) — e.g. `await get_job(job_id="missing", jobs=jobs)`. Because of this:

- **Never add a new parameter whose default is `Depends(...)`** to any of the 12 existing per-kind job handlers (`get_job`, `cancel_job`, `download_job`, and the video/audio/generation equivalents) or to `create_generation_job`. A direct Python call without FastAPI's request cycle running binds that parameter to the raw `fastapi.params.Depends` marker object, not a resolved value, and touching it crashes with `AttributeError`, breaking all 16 files.
- Instead: (1) put the permission/login gate on the **route decorator** via `dependencies=[Depends(require(Permission.X))]` — this never touches the function signature, so direct calls never trigger it at all (matching today's behavior exactly); (2) add a plain optional `request: Request | None = None` parameter (a *normal* Python default, safe for direct calls) for the **data-level** owner filter, and read the already-resolved principal from `request.state.current_user` (stashed there by `get_current_user` as a side effect during real dispatch) via a small `current_user_from_request(request)` helper that returns `None` when `request is None` — filtering is skipped whenever it returns `None`, exactly reproducing today's unfiltered behavior for every direct-call test.
- `create_job`, `create_video_job`, `create_audio_job` **already** take a required `request: Request` today (existing tests already pass `request=None` explicitly for these) — no signature change needed there, just start reading `request.state.current_user` the same way.
- `JobManager.create_job` / `VideoJobManager.create_job` / `AudioJobManager.create_job` / `GenerationJobManager.create_job` are plain async methods, not FastAPI routes — adding `owner: AuthenticatedUser | None = None` there is a completely ordinary, safe Python default (no Depends involved), verified against every direct manager-level test call in the suite.

---

## File Structure

**New backend files:**

| File | Responsibility |
|---|---|
| `app/services/json_store.py` | Atomic JSON/text write (mkstemp+replace) + corrupt-file backup — generic helpers extracted from `ModelRegistry`'s pattern, used only by new auth storage. |
| `app/services/auth/__init__.py` | Empty package marker. |
| `app/services/auth/permissions.py` | `Role`, `Permission` enums + `ROLE_PERMISSIONS` table. |
| `app/services/auth/passwords.py` | `hash_password` / `verify_password` / `generate_salt` (scrypt). |
| `app/services/auth/sessions.py` | Signed session-cookie value creation/verification (HMAC-SHA256). |
| `app/services/auth/user_store.py` | `User` dataclass + `UserStore` (users.json, atomic write, corrupt backup). |
| `app/services/auth/identity.py` | `UserIdentity`, `IdentityProvider` protocol, `LocalPasswordProvider`, `AuthenticatedUser` (request-scoped principal), `authenticated_user_from_record`. |
| `app/services/auth/quotas.py` | `RoleQuota`, `DEFAULT_ROLE_QUOTAS`, `QuotaStatus`, `QuotaService` (usage.json, admission checks, lazy daily reset). |
| `app/api/auth_deps.py` | `get_current_user`, `require(permission)`, `current_user_from_request`, state getters. |
| `app/api/auth_routes.py` | `/api/v1/auth/*`: login, logout, logout-all, me, change-password, setup. |
| `app/api/users_routes.py` | `/api/v1/users/*`: list/create/patch users, `GET /users/{id}/jobs`. |

**Modified backend files:** `app/config.py` (AUTH_MODE/AUTH_SECRET + paths + `ensure_auth_secret`), `app/exceptions.py` (+`QuotaExceededError`), `app/security.py` (+`LoopbackGuardMiddleware`), `app/models.py` (+`owner_id` on all 4 job dataclasses), `app/schemas.py` (+auth/user schemas, +`ownerId` on the 4 job response models, +list-response wrappers), `app/services/job_manager.py` / `video_job_manager.py` / `audio_job_manager.py` / `generation_job_manager.py` (+owner param, +quota admission/usage recording), `app/api/routes.py` (permission gates + owner filtering + 4 new list endpoints), `app/main.py` (wiring), `.env.example` (document `AUTH_MODE`/`AUTH_SECRET`).

**New frontend files:** `frontend/src/services/auth.ts`, `frontend/src/services/users.ts`, `frontend/src/hooks/useAuth.tsx`, `frontend/src/hooks/useUsers.ts`, `frontend/src/pages/LoginPage.tsx`, `frontend/src/pages/SetupPage.tsx`, `frontend/src/pages/UsersPage.tsx`, `frontend/src/components/Header.tsx`, `frontend/src/components/ForcedPasswordChangeModal.tsx`, plus matching `*.test.tsx`/`*.test.ts` files.

**Modified frontend files:** `frontend/src/lib/apiTypes.ts` (+auth/user types, +`ownerId`), `frontend/src/lib/api.ts` (+`apiPatchJson`, +list-jobs fetchers), `frontend/src/services/audio.ts` / `frontend/src/services/generation.ts` (+list fetchers), `frontend/src/App.tsx` (auth gate + `/users` route), `frontend/src/main.tsx` (`AuthProvider`), `frontend/src/lib/navigation.ts` (+`requiredPermission`, +Users entry), `frontend/src/components/AppShell.tsx` (nav filter + `<Header/>`), `frontend/src/hooks/useJobQueue.ts` + `frontend/src/components/JobQueue.tsx` (owner column + "view all" toggle).

---

## Task 1: `json_store.py` — shared atomic JSON/text write helpers

**Files:**
- Create: `app/services/json_store.py`
- Test: `tests/test_json_store.py`

**Interfaces:**
- Produces: `write_text_atomically(path: Path, text: str) -> None`, `write_json_atomically(path: Path, payload: Any) -> None`, `backup_corrupt_file(path: Path, exc: Exception, logger: logging.Logger) -> Path` — consumed by Tasks 5, 6, 9.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_json_store.py
from __future__ import annotations

import json
import logging
from pathlib import Path

from app.services.json_store import backup_corrupt_file, write_json_atomically, write_text_atomically


def test_write_text_atomically_creates_file_with_content(tmp_path: Path) -> None:
    target = tmp_path / "sub" / "file.txt"

    write_text_atomically(target, "hello")

    assert target.read_text(encoding="utf-8") == "hello"


def test_write_text_atomically_leaves_no_tmp_file_on_success(tmp_path: Path) -> None:
    target = tmp_path / "file.txt"

    write_text_atomically(target, "hello")

    leftovers = list(tmp_path.glob(".*.tmp"))
    assert leftovers == []


def test_write_json_atomically_round_trips_payload(tmp_path: Path) -> None:
    target = tmp_path / "data.json"

    write_json_atomically(target, {"a": 1, "b": [1, 2, 3]})

    assert json.loads(target.read_text(encoding="utf-8")) == {"a": 1, "b": [1, 2, 3]}


def test_write_json_atomically_uses_temp_file_and_replace(tmp_path: Path, monkeypatch) -> None:
    target = tmp_path / "data.json"
    replace_sources: list[Path] = []
    original_replace = Path.replace

    def spy_replace(self, dest):
        replace_sources.append(self)
        return original_replace(self, dest)

    monkeypatch.setattr(Path, "replace", spy_replace)

    write_json_atomically(target, {"a": 1})

    assert replace_sources[0].suffix == ".tmp"
    assert replace_sources[0].parent == target.parent


def test_backup_corrupt_file_renames_with_timestamp_suffix(tmp_path: Path) -> None:
    target = tmp_path / "data.json"
    target.write_text("not json", encoding="utf-8")
    logger = logging.getLogger("test_json_store")

    backup_path = backup_corrupt_file(target, ValueError("bad"), logger)

    assert not target.exists()
    assert backup_path.exists()
    assert backup_path.name.startswith("data.json.corrupt-")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv\Scripts\python.exe -m pytest tests/test_json_store.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.services.json_store'`

- [ ] **Step 3: Write the implementation**

```python
# app/services/json_store.py
from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path
from typing import Any

from app.models import utc_now


def write_text_atomically(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # mkstemp in the same directory (not the OS temp dir) so Path.replace is
    # an atomic rename on the same filesystem, never a cross-device copy.
    descriptor, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    tmp_path = Path(tmp_name)
    try:
        with open(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
        tmp_path.replace(path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def write_json_atomically(path: Path, payload: Any) -> None:
    write_text_atomically(path, json.dumps(payload, indent=2))


def backup_corrupt_file(path: Path, exc: Exception, logger: logging.Logger) -> Path:
    timestamp = utc_now().strftime("%Y%m%dT%H%M%S%f")
    backup_path = path.with_name(f"{path.name}.corrupt-{timestamp}")
    path.replace(backup_path)
    logger.warning("Corrupt JSON file at %s (%s); backed up to %s", path, exc, backup_path)
    return backup_path
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv\Scripts\python.exe -m pytest tests/test_json_store.py -q`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add app/services/json_store.py tests/test_json_store.py
git commit -m "feat: add atomic JSON/text write helpers for auth storage"
```

---

## Task 2: `permissions.py` — Role, Permission, ROLE_PERMISSIONS

**Files:**
- Create: `app/services/auth/__init__.py` (empty)
- Create: `app/services/auth/permissions.py`
- Test: `tests/test_permissions.py`

**Interfaces:**
- Produces: `Role` (str enum: `admin`, `user`), `Permission` (str enum, 12 values), `ROLE_PERMISSIONS: dict[Role, frozenset[Permission]]`, `role_has_permission(role, permission) -> bool` — consumed by Tasks 6-14.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_permissions.py
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv\Scripts\python.exe -m pytest tests/test_permissions.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.services.auth'`

- [ ] **Step 3: Write the implementation**

```python
# app/services/auth/__init__.py
```

```python
# app/services/auth/permissions.py
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv\Scripts\python.exe -m pytest tests/test_permissions.py -q`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add app/services/auth/__init__.py app/services/auth/permissions.py tests/test_permissions.py
git commit -m "feat: add Role/Permission tables for multi-user auth"
```

---

## Task 3: `passwords.py` — scrypt hash/verify

**Files:**
- Create: `app/services/auth/passwords.py`
- Test: `tests/test_passwords.py`

**Interfaces:**
- Produces: `generate_salt() -> str` (hex), `hash_password(password: str, salt: str) -> str` (hex), `verify_password(password: str, password_hash: str, salt: str) -> bool` — consumed by Tasks 6, 7, 13, 14.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_passwords.py
from __future__ import annotations

from app.services.auth.passwords import generate_salt, hash_password, verify_password


def test_generate_salt_returns_distinct_values() -> None:
    assert generate_salt() != generate_salt()


def test_hash_password_is_deterministic_for_same_salt() -> None:
    salt = generate_salt()
    assert hash_password("hunter2", salt) == hash_password("hunter2", salt)


def test_hash_password_differs_across_salts() -> None:
    assert hash_password("hunter2", generate_salt()) != hash_password("hunter2", generate_salt())


def test_verify_password_accepts_correct_password() -> None:
    salt = generate_salt()
    password_hash = hash_password("correct horse", salt)
    assert verify_password("correct horse", password_hash, salt) is True


def test_verify_password_rejects_wrong_password() -> None:
    salt = generate_salt()
    password_hash = hash_password("correct horse", salt)
    assert verify_password("wrong password", password_hash, salt) is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv\Scripts\python.exe -m pytest tests/test_passwords.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.services.auth.passwords'`

- [ ] **Step 3: Write the implementation**

```python
# app/services/auth/passwords.py
from __future__ import annotations

import hashlib
import hmac
import secrets

SCRYPT_N = 2**14
SCRYPT_R = 8
SCRYPT_P = 1
SALT_BYTES = 16
KEY_LENGTH = 32


def generate_salt() -> str:
    return secrets.token_hex(SALT_BYTES)


def hash_password(password: str, salt: str) -> str:
    derived = hashlib.scrypt(
        password.encode("utf-8"), salt=bytes.fromhex(salt),
        n=SCRYPT_N, r=SCRYPT_R, p=SCRYPT_P, dklen=KEY_LENGTH,
    )
    return derived.hex()


def verify_password(password: str, password_hash: str, salt: str) -> bool:
    candidate = hash_password(password, salt)
    return hmac.compare_digest(candidate, password_hash)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv\Scripts\python.exe -m pytest tests/test_passwords.py -q`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add app/services/auth/passwords.py tests/test_passwords.py
git commit -m "feat: add scrypt password hashing for local auth"
```

---

## Task 4: `sessions.py` — signed session cookie value

**Files:**
- Create: `app/services/auth/sessions.py`
- Test: `tests/test_sessions.py`

**Interfaces:**
- Produces: `SESSION_COOKIE_NAME = "upflow_session"`, `SESSION_TTL_SECONDS = 30*24*3600`, `SessionPayload` (dataclass: `user_id: str`, `session_ver: int`, `expires_at: datetime`), `create_session_cookie_value(user_id, session_ver, secret, *, now=None) -> str`, `verify_session_cookie(value, secret, *, now=None) -> SessionPayload | None` — consumed by Tasks 12, 13.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_sessions.py
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from app.services.auth.sessions import (
    SESSION_TTL_SECONDS,
    create_session_cookie_value,
    verify_session_cookie,
)

SECRET = "test-secret-32-bytes-of-entropy!"


def test_verify_accepts_freshly_created_cookie() -> None:
    value = create_session_cookie_value("user-1", 0, SECRET)

    payload = verify_session_cookie(value, SECRET)

    assert payload is not None
    assert payload.user_id == "user-1"
    assert payload.session_ver == 0


def test_verify_rejects_tampered_payload() -> None:
    value = create_session_cookie_value("user-1", 0, SECRET)
    payload_b64, signature = value.rsplit(".", 1)
    tampered = f"{payload_b64}x.{signature}"

    assert verify_session_cookie(tampered, SECRET) is None


def test_verify_rejects_tampered_signature() -> None:
    value = create_session_cookie_value("user-1", 0, SECRET)
    payload_b64, signature = value.rsplit(".", 1)
    tampered_signature = ("0" if signature[0] != "0" else "1") + signature[1:]

    assert verify_session_cookie(f"{payload_b64}.{tampered_signature}", SECRET) is None


def test_verify_rejects_wrong_secret() -> None:
    value = create_session_cookie_value("user-1", 0, SECRET)

    assert verify_session_cookie(value, "a-completely-different-secret") is None


def test_verify_rejects_expired_cookie() -> None:
    created_at = datetime(2026, 1, 1, tzinfo=timezone.utc)
    value = create_session_cookie_value("user-1", 0, SECRET, now=created_at)
    past_expiry = created_at + timedelta(seconds=SESSION_TTL_SECONDS + 1)

    assert verify_session_cookie(value, SECRET, now=past_expiry) is None


def test_verify_accepts_cookie_just_before_expiry() -> None:
    created_at = datetime(2026, 1, 1, tzinfo=timezone.utc)
    value = create_session_cookie_value("user-1", 0, SECRET, now=created_at)
    just_before_expiry = created_at + timedelta(seconds=SESSION_TTL_SECONDS - 1)

    assert verify_session_cookie(value, SECRET, now=just_before_expiry) is not None


def test_verify_rejects_malformed_value() -> None:
    assert verify_session_cookie("not-a-valid-token", SECRET) is None
    assert verify_session_cookie("", SECRET) is None


def test_session_ver_round_trips() -> None:
    value = create_session_cookie_value("user-1", 7, SECRET)

    payload = verify_session_cookie(value, SECRET)

    assert payload is not None
    assert payload.session_ver == 7
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv\Scripts\python.exe -m pytest tests/test_sessions.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.services.auth.sessions'`

- [ ] **Step 3: Write the implementation**

```python
# app/services/auth/sessions.py
from __future__ import annotations

import base64
import hashlib
import hmac
import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

SESSION_COOKIE_NAME = "upflow_session"
SESSION_TTL_SECONDS = 30 * 24 * 3600


@dataclass(frozen=True, slots=True)
class SessionPayload:
    user_id: str
    session_ver: int
    expires_at: datetime


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _b64encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _b64decode(data: str) -> bytes:
    padding = "=" * (-len(data) % 4)
    return base64.urlsafe_b64decode(data + padding)


def _sign(payload_b64: str, secret: str) -> str:
    return hmac.new(secret.encode("utf-8"), payload_b64.encode("ascii"), hashlib.sha256).hexdigest()


def create_session_cookie_value(
    user_id: str, session_ver: int, secret: str, *, now: datetime | None = None
) -> str:
    expires_at = (now or _utc_now()) + timedelta(seconds=SESSION_TTL_SECONDS)
    payload = {"uid": user_id, "sv": session_ver, "exp": expires_at.isoformat()}
    payload_b64 = _b64encode(json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8"))
    signature = _sign(payload_b64, secret)
    return f"{payload_b64}.{signature}"


def verify_session_cookie(
    value: str, secret: str, *, now: datetime | None = None
) -> SessionPayload | None:
    try:
        payload_b64, signature = value.rsplit(".", 1)
    except ValueError:
        return None
    expected_signature = _sign(payload_b64, secret)
    if not hmac.compare_digest(signature, expected_signature):
        return None
    try:
        payload = json.loads(_b64decode(payload_b64))
        session_payload = SessionPayload(
            user_id=payload["uid"],
            session_ver=int(payload["sv"]),
            expires_at=datetime.fromisoformat(payload["exp"]),
        )
    except (ValueError, KeyError, TypeError, json.JSONDecodeError):
        return None
    if session_payload.expires_at <= (now or _utc_now()):
        return None
    return session_payload
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv\Scripts\python.exe -m pytest tests/test_sessions.py -q`
Expected: PASS (8 tests)

- [ ] **Step 5: Commit**

```bash
git add app/services/auth/sessions.py tests/test_sessions.py
git commit -m "feat: add HMAC-signed session cookie encode/decode"
```

---

## Task 5: `config.py` — AUTH_MODE/AUTH_SECRET settings + `ensure_auth_secret`

**Files:**
- Modify: `app/config.py`
- Test: `tests/test_config_auth.py`

**Interfaces:**
- Produces: `Settings.auth_mode: str` (`"off"|"multi"`, default `"off"`), `Settings.auth_secret: str | None`, `Settings.auth_path` / `Settings.users_file_path` / `Settings.usage_file_path` (properties), `ENV_FILE_PATH: Path`, `ensure_auth_secret(settings: Settings) -> str` — consumed by Tasks 6, 9, 12, 13, 15.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_config_auth.py
from __future__ import annotations

from pathlib import Path

import pytest

from app.config import Settings, ensure_auth_secret


def make_settings(tmp_path: Path, **overrides: object) -> Settings:
    kwargs: dict[str, object] = {"RUNTIME_DIR": str(tmp_path / "runtime")}
    kwargs.update(overrides)
    return Settings(_env_file=None, **kwargs)


def test_auth_mode_defaults_to_off(tmp_path: Path) -> None:
    assert make_settings(tmp_path).auth_mode == "off"


def test_auth_mode_accepts_multi(tmp_path: Path) -> None:
    assert make_settings(tmp_path, AUTH_MODE="multi").auth_mode == "multi"


def test_auth_mode_rejects_invalid_value(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="AUTH_MODE"):
        make_settings(tmp_path, AUTH_MODE="bogus")


def test_users_file_path_is_under_runtime_auth_dir(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    assert settings.users_file_path == settings.runtime_path / "auth" / "users.json"


def test_usage_file_path_is_under_runtime_auth_dir(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    assert settings.usage_file_path == settings.runtime_path / "auth" / "usage.json"


def test_ensure_auth_secret_generates_and_persists_when_missing(tmp_path: Path, monkeypatch) -> None:
    import app.config as config_module

    env_path = tmp_path / ".env"
    monkeypatch.setattr(config_module, "ENV_FILE_PATH", env_path)
    settings = make_settings(tmp_path)
    assert settings.auth_secret is None

    secret = ensure_auth_secret(settings)

    assert len(secret) == 64  # 32 bytes hex-encoded
    assert settings.auth_secret == secret
    assert f"AUTH_SECRET={secret}" in env_path.read_text(encoding="utf-8")


def test_ensure_auth_secret_is_idempotent(tmp_path: Path, monkeypatch) -> None:
    import app.config as config_module

    env_path = tmp_path / ".env"
    monkeypatch.setattr(config_module, "ENV_FILE_PATH", env_path)
    settings = make_settings(tmp_path)

    first = ensure_auth_secret(settings)
    second = ensure_auth_secret(settings)

    assert first == second
    assert env_path.read_text(encoding="utf-8").count("AUTH_SECRET=") == 1


def test_ensure_auth_secret_appends_without_clobbering_existing_content(tmp_path: Path, monkeypatch) -> None:
    import app.config as config_module

    env_path = tmp_path / ".env"
    env_path.write_text("APP_PORT=8090", encoding="utf-8")
    monkeypatch.setattr(config_module, "ENV_FILE_PATH", env_path)
    settings = make_settings(tmp_path)

    ensure_auth_secret(settings)

    content = env_path.read_text(encoding="utf-8")
    assert "APP_PORT=8090" in content
    assert "AUTH_SECRET=" in content
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv\Scripts\python.exe -m pytest tests/test_config_auth.py -q`
Expected: FAIL with `AttributeError: 'Settings' object has no attribute 'auth_mode'` (and `ImportError` for `ensure_auth_secret`)

- [ ] **Step 3: Write the implementation**

Add near the top of `app/config.py`, right after `PROJECT_ROOT = Path(__file__).resolve().parent.parent` (line 69):

```python
AUTH_MODE_OFF = "off"
AUTH_MODE_MULTI = "multi"
AUTH_MODES = frozenset({AUTH_MODE_OFF, AUTH_MODE_MULTI})

ENV_FILE_PATH = PROJECT_ROOT / ".env"
```

Add `import secrets` to the top-level imports (alongside `import os`), and add:

```python
from app.services.json_store import write_text_atomically
```

Add these fields inside `class Settings(BaseSettings):`, near the end of the field list (after `capability_fix_timeout_seconds`):

```python
    auth_mode: str = Field(default=AUTH_MODE_OFF, alias="AUTH_MODE")
    auth_secret: str | None = Field(default=None, alias="AUTH_SECRET")
    auth_dir: str = Field(default="auth", alias="AUTH_DIR")
```

Add this validator alongside the other `@field_validator` methods:

```python
    @field_validator("auth_mode")
    @classmethod
    def _validate_auth_mode(cls, value: str) -> str:
        if value not in AUTH_MODES:
            raise ValueError(f"AUTH_MODE must be one of {sorted(AUTH_MODES)}")
        return value
```

Add these properties alongside `models_path` (after it):

```python
    @property
    def auth_path(self) -> Path:
        return self.runtime_path / self.auth_dir

    @property
    def users_file_path(self) -> Path:
        return self.auth_path / "users.json"

    @property
    def usage_file_path(self) -> Path:
        return self.auth_path / "usage.json"
```

Add this function right before `get_settings` (near the bottom of the file):

```python
def _append_env_var(env_path: Path, key: str, value: str) -> None:
    existing = env_path.read_text(encoding="utf-8") if env_path.exists() else ""
    if existing and not existing.endswith("\n"):
        existing += "\n"
    write_text_atomically(env_path, f"{existing}{key}={value}\n")


def ensure_auth_secret(settings: Settings) -> str:
    if settings.auth_secret:
        return settings.auth_secret
    secret = secrets.token_hex(32)
    _append_env_var(ENV_FILE_PATH, "AUTH_SECRET", secret)
    settings.auth_secret = secret
    return secret
```

Add to `.env.example`, in a new section after "Seguridad y cola":

```
# --- Multi-usuario y autenticacion (subproyecto C) ---
# off (default) = escritorio single-user, sin login, comportamiento identico a hoy.
# multi = requiere login; AUTH_SECRET se autogenera y se persiste aqui al primer arranque.
AUTH_MODE=off
# AUTH_SECRET se completa solo -- no lo pongas a mano salvo que migres una instalacion existente.
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv\Scripts\python.exe -m pytest tests/test_config_auth.py -q`
Expected: PASS (8 tests)

- [ ] **Step 5: Run the full existing backend suite to confirm no regression from the config.py edit**

Run: `.venv\Scripts\python.exe -m pytest tests/ -q`
Expected: PASS, same count as before this task plus the 8 new tests (no existing test touches `AUTH_MODE`/`AUTH_SECRET`, so none should be affected)

- [ ] **Step 6: Commit**

```bash
git add app/config.py .env.example tests/test_config_auth.py
git commit -m "feat: add AUTH_MODE/AUTH_SECRET settings and first-run secret persistence"
```

---

## Task 6: `user_store.py` — User dataclass + UserStore

**Files:**
- Create: `app/services/auth/user_store.py`
- Test: `tests/test_user_store.py`

**Interfaces:**
- Consumes: `write_json_atomically`, `backup_corrupt_file` (Task 1); `Role` (Task 2); `Settings.users_file_path` (Task 5).
- Produces: `User` (dataclass), `UserStore` with `list`, `get`, `get_by_username`, `is_empty`, `create`, `set_password`, `set_role`, `set_disabled`, `set_quota_overrides`, `bump_session_ver` — consumed by Tasks 7, 12, 13, 14.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_user_store.py
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv\Scripts\python.exe -m pytest tests/test_user_store.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.services.auth.user_store'`

- [ ] **Step 3: Write the implementation**

```python
# app/services/auth/user_store.py
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv\Scripts\python.exe -m pytest tests/test_user_store.py -q`
Expected: PASS (13 tests)

- [ ] **Step 5: Commit**

```bash
git add app/services/auth/user_store.py tests/test_user_store.py
git commit -m "feat: add UserStore (users.json, atomic write + corrupt backup)"
```

---

## Task 7: `identity.py` — IdentityProvider, LocalPasswordProvider, AuthenticatedUser

**Files:**
- Create: `app/services/auth/identity.py`
- Test: `tests/test_identity.py`

**Interfaces:**
- Consumes: `verify_password` (Task 3); `Permission`, `Role`, `ROLE_PERMISSIONS` (Task 2); `User`, `UserStore` (Task 6).
- Produces: `UserIdentity` (dataclass: `user_id`, `username`, `role`, `external_subject`), `IdentityProvider` (Protocol), `LocalPasswordProvider`, `AuthenticatedUser` (dataclass: `id: str | None`, `username`, `role`, `permissions: frozenset[Permission]`, `must_change_password: bool`, `quota_overrides: dict[str, int]`), `authenticated_user_from_record(user: User) -> AuthenticatedUser` — consumed by Task 9, Tasks 12-14, and by `app/services/job_manager.py` et al. (Tasks 16-19).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_identity.py
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv\Scripts\python.exe -m pytest tests/test_identity.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.services.auth.identity'`

- [ ] **Step 3: Write the implementation**

```python
# app/services/auth/identity.py
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv\Scripts\python.exe -m pytest tests/test_identity.py -q`
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add app/services/auth/identity.py tests/test_identity.py
git commit -m "feat: add IdentityProvider/LocalPasswordProvider and AuthenticatedUser principal"
```

---

## Task 8: `exceptions.py` — QuotaExceededError

**Files:**
- Modify: `app/exceptions.py`

**Interfaces:**
- Produces: `QuotaExceededError(Exception)` — consumed by Task 9 (raised) and Tasks 16-24 (caught, mapped to HTTP 429).

- [ ] **Step 1: Add the exception (no separate test file — covered by Task 9's tests and the route-level 429 tests in Tasks 20-23)**

```python
# app/exceptions.py — add at the end of the file
class QuotaExceededError(Exception):
    """Raised when a user's concurrency/queue/daily quota would be exceeded by a new job."""
```

- [ ] **Step 2: Run the full existing backend suite to confirm no regression**

Run: `.venv\Scripts\python.exe -m pytest tests/ -q`
Expected: PASS, same count as before (adding an unused-so-far exception class changes nothing)

- [ ] **Step 3: Commit**

```bash
git add app/exceptions.py
git commit -m "feat: add QuotaExceededError exception type"
```

---

## Task 9: `quotas.py` — RoleQuota, QuotaService

**Files:**
- Create: `app/services/auth/quotas.py`
- Test: `tests/test_quotas.py`

**Interfaces:**
- Consumes: `AuthenticatedUser` (Task 7); `Role` (Task 2); `Settings.usage_file_path` (Task 5); `write_json_atomically`, `backup_corrupt_file` (Task 1); `QuotaExceededError` (Task 8); `app.models.JobStatus`, `app.models.utc_now`.
- Produces: `RoleQuota` (dataclass), `DEFAULT_ROLE_QUOTAS: dict[Role, RoleQuota]`, `QuotaStatus` (dataclass), `QuotaService` with `attach_managers(*managers)`, `check_admission(user: AuthenticatedUser) -> None`, `record_usage(user_id: str | None, gpu_seconds: float) -> None`, `status_for(user: AuthenticatedUser) -> QuotaStatus` — consumed by Tasks 13, 14, 15, 16-19.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_quotas.py
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.config import Settings
from app.exceptions import QuotaExceededError
from app.models import JobStatus
from app.services.auth.identity import AuthenticatedUser
from app.services.auth.permissions import ROLE_PERMISSIONS, Role
from app.services.auth.quotas import QuotaService


def make_settings(tmp_path: Path) -> Settings:
    return Settings(_env_file=None, RUNTIME_DIR=str(tmp_path / "runtime"))


def make_user(user_id: str = "u1", role: Role = Role.user, overrides: dict | None = None) -> AuthenticatedUser:
    return AuthenticatedUser(
        id=user_id, username="alice", role=role, permissions=ROLE_PERMISSIONS[role],
        must_change_password=False, quota_overrides=overrides or {},
    )


class FakeJob:
    def __init__(self, owner_id: str | None, status: JobStatus) -> None:
        self.owner_id = owner_id
        self.status = status


class FakeManager:
    def __init__(self, jobs: dict[str, FakeJob]) -> None:
        self.jobs = jobs


def test_check_admission_passes_for_user_with_no_jobs(tmp_path: Path) -> None:
    service = QuotaService(make_settings(tmp_path))
    service.attach_managers(FakeManager({}))

    service.check_admission(make_user())  # should not raise


def test_check_admission_raises_when_concurrency_limit_reached(tmp_path: Path) -> None:
    service = QuotaService(make_settings(tmp_path))
    service.attach_managers(FakeManager({"j1": FakeJob("u1", JobStatus.running)}))

    with pytest.raises(QuotaExceededError, match="corriendo"):
        service.check_admission(make_user())  # user default max_concurrent=1


def test_check_admission_raises_when_queue_limit_reached(tmp_path: Path) -> None:
    service = QuotaService(make_settings(tmp_path))
    queued_jobs = {f"j{i}": FakeJob("u1", JobStatus.queued) for i in range(5)}
    service.attach_managers(FakeManager(queued_jobs))

    with pytest.raises(QuotaExceededError, match="cola"):
        service.check_admission(make_user())  # user default max_queued=5


def test_check_admission_ignores_other_users_jobs(tmp_path: Path) -> None:
    service = QuotaService(make_settings(tmp_path))
    service.attach_managers(FakeManager({"j1": FakeJob("someone-else", JobStatus.running)}))

    service.check_admission(make_user())  # should not raise


def test_check_admission_skips_off_mode_pseudo_user(tmp_path: Path) -> None:
    service = QuotaService(make_settings(tmp_path))
    running = {f"j{i}": FakeJob(None, JobStatus.running) for i in range(10)}
    service.attach_managers(FakeManager(running))

    off_mode_user = AuthenticatedUser(
        id=None, username="local", role=Role.admin, permissions=ROLE_PERMISSIONS[Role.admin],
        must_change_password=False, quota_overrides={},
    )
    service.check_admission(off_mode_user)  # should not raise: id is None


def test_check_admission_admin_role_is_unlimited(tmp_path: Path) -> None:
    service = QuotaService(make_settings(tmp_path))
    running = {f"j{i}": FakeJob("admin-1", JobStatus.running) for i in range(50)}
    service.attach_managers(FakeManager(running))

    service.check_admission(make_user("admin-1", role=Role.admin))  # should not raise


def test_check_admission_respects_quota_override(tmp_path: Path) -> None:
    service = QuotaService(make_settings(tmp_path))
    service.attach_managers(FakeManager({"j1": FakeJob("u1", JobStatus.running)}))

    service.check_admission(make_user(overrides={"max_concurrent": 5}))  # should not raise


def test_record_usage_then_check_admission_raises_at_daily_job_limit(tmp_path: Path) -> None:
    service = QuotaService(make_settings(tmp_path))
    service.attach_managers(FakeManager({}))
    user = make_user(overrides={"max_jobs_per_day": 2, "max_concurrent": 0, "max_queued": 0})

    service.record_usage(user.id, gpu_seconds=1.0)
    service.record_usage(user.id, gpu_seconds=1.0)

    with pytest.raises(QuotaExceededError, match="diario"):
        service.check_admission(user)


def test_record_usage_accumulates_gpu_seconds_and_raises_at_limit(tmp_path: Path) -> None:
    service = QuotaService(make_settings(tmp_path))
    service.attach_managers(FakeManager({}))
    user = make_user(overrides={"max_gpu_seconds_per_day": 10, "max_concurrent": 0, "max_queued": 0})

    service.record_usage(user.id, gpu_seconds=6.0)
    service.record_usage(user.id, gpu_seconds=5.0)

    with pytest.raises(QuotaExceededError, match="GPU"):
        service.check_admission(user)


def test_record_usage_none_user_id_is_a_no_op(tmp_path: Path) -> None:
    service = QuotaService(make_settings(tmp_path))
    service.record_usage(None, gpu_seconds=100.0)  # should not raise, nothing persisted
    assert not service._path.exists()


def test_status_for_reports_used_and_max_values(tmp_path: Path) -> None:
    service = QuotaService(make_settings(tmp_path))
    user = make_user()
    service.record_usage(user.id, gpu_seconds=42.0)

    status = service.status_for(user)

    assert status.used_jobs_today == 1
    assert status.used_gpu_seconds_today == 42.0
    assert status.max_concurrent == 1
    assert status.max_jobs_per_day == 50


def test_usage_persists_across_instances(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    service = QuotaService(settings)
    service.record_usage("u1", gpu_seconds=10.0)

    reloaded = QuotaService(settings)
    status = reloaded.status_for(make_user())

    assert status.used_gpu_seconds_today == 10.0


def test_usage_resets_lazily_on_new_day(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    service = QuotaService(settings)
    service.record_usage("u1", gpu_seconds=10.0)
    # Simulate a day rollover by rewriting yesterday's date directly on disk.
    payload = json.loads(settings.usage_file_path.read_text(encoding="utf-8"))
    payload["u1"]["date"] = "2000-01-01"
    settings.usage_file_path.write_text(json.dumps(payload), encoding="utf-8")

    reloaded = QuotaService(settings)
    status = reloaded.status_for(make_user())

    assert status.used_jobs_today == 0
    assert status.used_gpu_seconds_today == 0.0


def test_corrupt_usage_file_is_backed_up_and_reset(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    settings.usage_file_path.parent.mkdir(parents=True, exist_ok=True)
    settings.usage_file_path.write_text("not json", encoding="utf-8")

    service = QuotaService(settings)

    assert service.status_for(make_user()).used_jobs_today == 0
    backups = list(settings.usage_file_path.parent.glob("usage.json.corrupt-*"))
    assert len(backups) == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv\Scripts\python.exe -m pytest tests/test_quotas.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.services.auth.quotas'`

- [ ] **Step 3: Write the implementation**

```python
# app/services/auth/quotas.py
from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.config import Settings
from app.exceptions import QuotaExceededError
from app.models import JobStatus, utc_now
from app.services.auth.identity import AuthenticatedUser
from app.services.auth.permissions import Role
from app.services.json_store import backup_corrupt_file, write_json_atomically

logger = logging.getLogger(__name__)

OVERRIDE_KEYS = ("max_concurrent", "max_queued", "max_jobs_per_day", "max_gpu_seconds_per_day")


@dataclass(frozen=True, slots=True)
class RoleQuota:
    max_concurrent: int
    max_queued: int
    max_jobs_per_day: int
    max_gpu_seconds_per_day: int


DEFAULT_ROLE_QUOTAS: dict[Role, RoleQuota] = {
    Role.user: RoleQuota(max_concurrent=1, max_queued=5, max_jobs_per_day=50, max_gpu_seconds_per_day=3600),
    Role.admin: RoleQuota(max_concurrent=0, max_queued=0, max_jobs_per_day=0, max_gpu_seconds_per_day=0),
}


@dataclass(slots=True)
class UsageRecord:
    date: str
    jobs: int = 0
    gpu_seconds: float = 0.0


@dataclass(frozen=True, slots=True)
class QuotaStatus:
    max_concurrent: int
    max_queued: int
    max_jobs_per_day: int
    max_gpu_seconds_per_day: int
    used_jobs_today: int
    used_gpu_seconds_today: float


def _today() -> str:
    return utc_now().date().isoformat()


def _effective_quota(role: Role, overrides: dict[str, int]) -> RoleQuota:
    base = DEFAULT_ROLE_QUOTAS[role]
    values = {key: overrides.get(key, getattr(base, key)) for key in OVERRIDE_KEYS}
    return RoleQuota(**values)


class QuotaService:
    def __init__(self, settings: Settings) -> None:
        self._path = settings.usage_file_path
        self._lock = threading.Lock()
        self._usage: dict[str, UsageRecord] = self._load()
        self._managers: tuple[Any, ...] = ()

    def attach_managers(self, *managers: Any) -> None:
        self._managers = managers

    def check_admission(self, user: AuthenticatedUser) -> None:
        if user.id is None:
            return
        quota = _effective_quota(user.role, user.quota_overrides)
        self._check_concurrency(user.id, quota)
        self._check_queue(user.id, quota)
        self._check_daily(user.id, quota)

    def _check_concurrency(self, user_id: str, quota: RoleQuota) -> None:
        if not quota.max_concurrent:
            return
        running = self._count_owned(user_id, JobStatus.running)
        if running >= quota.max_concurrent:
            raise QuotaExceededError(
                f"Tenés {running} job(s) corriendo y tu límite es {quota.max_concurrent}."
            )

    def _check_queue(self, user_id: str, quota: RoleQuota) -> None:
        if not quota.max_queued:
            return
        queued = self._count_owned(user_id, JobStatus.queued)
        if queued >= quota.max_queued:
            raise QuotaExceededError(
                f"Tenés {queued} job(s) en cola y tu límite es {quota.max_queued}."
            )

    def _check_daily(self, user_id: str, quota: RoleQuota) -> None:
        usage = self._current_usage(user_id)
        if quota.max_jobs_per_day and usage.jobs >= quota.max_jobs_per_day:
            raise QuotaExceededError(
                f"Límite diario alcanzado: {quota.max_jobs_per_day} jobs. Se resetea a medianoche."
            )
        if quota.max_gpu_seconds_per_day and usage.gpu_seconds >= quota.max_gpu_seconds_per_day:
            raise QuotaExceededError(
                f"Límite diario de GPU alcanzado: {quota.max_gpu_seconds_per_day}s. Se resetea a medianoche."
            )

    def _count_owned(self, user_id: str, status: JobStatus) -> int:
        return sum(
            1
            for manager in self._managers
            for job in manager.jobs.values()
            if job.owner_id == user_id and job.status == status
        )

    def record_usage(self, user_id: str | None, gpu_seconds: float) -> None:
        if user_id is None:
            return
        with self._lock:
            usage = self._current_usage(user_id)
            self._usage[user_id] = UsageRecord(
                date=usage.date, jobs=usage.jobs + 1, gpu_seconds=usage.gpu_seconds + gpu_seconds
            )
            self._persist()

    def status_for(self, user: AuthenticatedUser) -> QuotaStatus:
        quota = _effective_quota(user.role, user.quota_overrides)
        usage = self._current_usage(user.id) if user.id is not None else UsageRecord(date=_today())
        return QuotaStatus(
            max_concurrent=quota.max_concurrent, max_queued=quota.max_queued,
            max_jobs_per_day=quota.max_jobs_per_day, max_gpu_seconds_per_day=quota.max_gpu_seconds_per_day,
            used_jobs_today=usage.jobs, used_gpu_seconds_today=usage.gpu_seconds,
        )

    def _current_usage(self, user_id: str) -> UsageRecord:
        today = _today()
        record = self._usage.get(user_id)
        if record is None or record.date != today:
            return UsageRecord(date=today)
        return record

    def _load(self) -> dict[str, UsageRecord]:
        if not self._path.exists():
            return {}
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
            return {
                user_id: UsageRecord(date=item["date"], jobs=item["jobs"], gpu_seconds=item["gpu_seconds"])
                for user_id, item in raw.items()
            }
        except (json.JSONDecodeError, KeyError, ValueError, TypeError) as exc:
            backup_corrupt_file(self._path, exc, logger)
            return {}

    def _persist(self) -> None:
        payload = {
            user_id: {"date": usage.date, "jobs": usage.jobs, "gpu_seconds": usage.gpu_seconds}
            for user_id, usage in self._usage.items()
        }
        write_json_atomically(self._path, payload)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv\Scripts\python.exe -m pytest tests/test_quotas.py -q`
Expected: PASS (14 tests)

- [ ] **Step 5: Commit**

```bash
git add app/services/auth/quotas.py tests/test_quotas.py
git commit -m "feat: add QuotaService (concurrency/queue/daily admission checks)"
```

---

## Task 10: `security.py` — LoopbackGuardMiddleware

**Files:**
- Modify: `app/security.py`
- Test: `tests/test_loopback_guard.py`

**Interfaces:**
- Produces: `LOOPBACK_HOSTS: frozenset[str]`, `is_loopback_host(client_host: str | None) -> bool`, `LoopbackGuardMiddleware` — wired into `app/main.py` in Task 15.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_loopback_guard.py
from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.security import LoopbackGuardMiddleware, is_loopback_host


def test_is_loopback_host_accepts_known_local_hosts() -> None:
    assert is_loopback_host("127.0.0.1") is True
    assert is_loopback_host("::1") is True
    assert is_loopback_host("localhost") is True


def test_is_loopback_host_accepts_testclient_sentinel() -> None:
    # Starlette's TestClient defaults request.client.host to "testclient" when
    # the test doesn't override `client=`. Nearly all of this repo's existing
    # 1207 tests instantiate `TestClient(app)` this way, and AUTH_MODE=off must
    # leave every one of them passing unchanged -- so the guard treats this
    # sentinel as local. No real network peer can ever present this literal
    # Host value.
    assert is_loopback_host("testclient") is True


def test_is_loopback_host_rejects_remote_ip() -> None:
    assert is_loopback_host("203.0.113.5") is False


def _make_app() -> FastAPI:
    app = FastAPI()

    @app.get("/ping")
    async def ping() -> dict[str, bool]:
        return {"ok": True}

    return app


def test_middleware_allows_default_testclient_when_off() -> None:
    app = _make_app()
    app.add_middleware(LoopbackGuardMiddleware, auth_mode="off")

    with TestClient(app) as client:
        response = client.get("/ping")

    assert response.status_code == 200


def test_middleware_allows_loopback_client_when_off() -> None:
    app = _make_app()
    app.add_middleware(LoopbackGuardMiddleware, auth_mode="off")

    with TestClient(app, client=("127.0.0.1", 12345)) as client:
        response = client.get("/ping")

    assert response.status_code == 200


def test_middleware_rejects_remote_client_when_off() -> None:
    app = _make_app()
    app.add_middleware(LoopbackGuardMiddleware, auth_mode="off")

    with TestClient(app, client=("203.0.113.5", 12345)) as client:
        response = client.get("/ping")

    assert response.status_code == 403
    assert "AUTH_MODE=multi" in response.json()["detail"]


def test_middleware_allows_remote_client_when_multi() -> None:
    app = _make_app()
    app.add_middleware(LoopbackGuardMiddleware, auth_mode="multi")

    with TestClient(app, client=("203.0.113.5", 12345)) as client:
        response = client.get("/ping")

    assert response.status_code == 200
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv\Scripts\python.exe -m pytest tests/test_loopback_guard.py -q`
Expected: FAIL with `ImportError: cannot import name 'LoopbackGuardMiddleware' from 'app.security'`

- [ ] **Step 3: Write the implementation**

Append to `app/security.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv\Scripts\python.exe -m pytest tests/test_loopback_guard.py -q`
Expected: PASS (8 tests)

- [ ] **Step 5: Commit**

```bash
git add app/security.py tests/test_loopback_guard.py
git commit -m "feat: add LoopbackGuardMiddleware for AUTH_MODE=off remote-access guardrail"
```

---

## Task 11: `schemas.py` — auth/user schemas + `ownerId` on job responses

**Files:**
- Modify: `app/schemas.py`

**Interfaces:**
- Produces: `LoginRequest`, `ChangePasswordRequest`, `SetupRequest`, `QuotaStatusResponse`, `MeResponse`, `UserSummaryResponse`, `UsersListResponse`, `CreateUserRequest`, `CreateUserResponse`, `UpdateUserRequest`, `UpdateUserResponse`, `OwnedJobSummaryResponse`, `UserJobsResponse`, `JobsListResponse`, `VideoJobsListResponse`, `AudioJobsListResponse`, `GenerationJobsListResponse` — consumed by Tasks 13, 14, 20-23. Also adds `owner_id: str | None` (alias `ownerId`) to `JobResponse`, `VideoJobResponse`, `AudioJobResponse`, `GenerationJobResponse` — consumed by Tasks 16-23.

- [ ] **Step 1: No standalone test file** — these are plain Pydantic models; their round-trip behavior is exercised by Tasks 13/14/20-23's API tests. Add a quick smoke check inline in this task instead.

```python
# tests/test_schemas_auth.py
from __future__ import annotations

from app.models import JobStatus
from app.schemas import JobResponse, MeResponse, QuotaStatusResponse


def test_job_response_serializes_owner_id_as_camel_case() -> None:
    response = JobResponse(
        job_id="j1", status=JobStatus.queued, original_filename="a.png", model_name="m", scale=4,
        output_format="png", created_at="2026-01-01T00:00:00+00:00", owner_id="user-1",
    )
    assert response.model_dump(by_alias=True)["ownerId"] == "user-1"


def test_job_response_owner_id_defaults_to_none() -> None:
    response = JobResponse(
        job_id="j1", status=JobStatus.queued, original_filename="a.png", model_name="m", scale=4,
        output_format="png", created_at="2026-01-01T00:00:00+00:00",
    )
    assert response.model_dump(by_alias=True)["ownerId"] is None


def test_me_response_serializes_camel_case() -> None:
    quota = QuotaStatusResponse(
        max_concurrent=1, max_queued=5, max_jobs_per_day=50, max_gpu_seconds_per_day=3600,
        used_jobs_today=0, used_gpu_seconds_today=0.0,
    )
    response = MeResponse(
        user_id="u1", username="alice", role="user", permissions=["jobs:create"],
        must_change_password=False, auth_mode="multi", quota=quota,
    )
    dumped = response.model_dump(by_alias=True)
    assert dumped["userId"] == "u1"
    assert dumped["mustChangePassword"] is False
    assert dumped["authMode"] == "multi"
    assert dumped["quota"]["maxConcurrent"] == 1
```

Run: `.venv\Scripts\python.exe -m pytest tests/test_schemas_auth.py -q` → FAIL (`ImportError`) before Step 2, PASS after.

- [ ] **Step 2: Add `owner_id` to the 4 existing job response models**

In `app/schemas.py`, add one field to each of `JobResponse`, `VideoJobResponse`, `AudioJobResponse`, `GenerationJobResponse` (right after `error: str | None = None` in each — pick any consistent spot):

```python
    owner_id: str | None = Field(default=None, serialization_alias="ownerId")
```

- [ ] **Step 3: Add the new auth/user schemas** — append to `app/schemas.py`:

```python
class LoginRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    username: str
    password: str


class ChangePasswordRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    current_password: str = Field(alias="currentPassword")
    new_password: str = Field(alias="newPassword", min_length=8)


class SetupRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    username: str = Field(min_length=3)
    password: str = Field(min_length=8)


class QuotaStatusResponse(BaseModel):
    max_concurrent: int = Field(serialization_alias="maxConcurrent")
    max_queued: int = Field(serialization_alias="maxQueued")
    max_jobs_per_day: int = Field(serialization_alias="maxJobsPerDay")
    max_gpu_seconds_per_day: int = Field(serialization_alias="maxGpuSecondsPerDay")
    used_jobs_today: int = Field(serialization_alias="usedJobsToday")
    used_gpu_seconds_today: float = Field(serialization_alias="usedGpuSecondsToday")


class MeResponse(BaseModel):
    user_id: str | None = Field(serialization_alias="userId")
    username: str
    role: str
    permissions: list[str]
    must_change_password: bool = Field(serialization_alias="mustChangePassword")
    auth_mode: str = Field(serialization_alias="authMode")
    quota: QuotaStatusResponse


class UserSummaryResponse(BaseModel):
    id: str
    username: str
    role: str
    disabled: bool
    must_change_password: bool = Field(serialization_alias="mustChangePassword")
    quota_overrides: dict[str, int] = Field(default_factory=dict, serialization_alias="quotaOverrides")
    created_at: datetime = Field(serialization_alias="createdAt")
    used_jobs_today: int = Field(serialization_alias="usedJobsToday")
    used_gpu_seconds_today: float = Field(serialization_alias="usedGpuSecondsToday")


class UsersListResponse(BaseModel):
    users: list[UserSummaryResponse]


class CreateUserRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    username: str = Field(min_length=3)
    role: str = Field(default="user")


class CreateUserResponse(BaseModel):
    user: UserSummaryResponse
    temporary_password: str = Field(serialization_alias="temporaryPassword")


class UpdateUserRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    role: str | None = None
    disabled: bool | None = None
    quota_overrides: dict[str, int] | None = Field(default=None, alias="quotaOverrides")
    reset_password: bool = Field(default=False, alias="resetPassword")


class UpdateUserResponse(BaseModel):
    user: UserSummaryResponse
    temporary_password: str | None = Field(default=None, serialization_alias="temporaryPassword")


class OwnedJobSummaryResponse(BaseModel):
    id: str
    kind: str
    status: JobStatus
    original_filename: str | None = Field(default=None, serialization_alias="originalFilename")
    created_at: datetime = Field(serialization_alias="createdAt")
    finished_at: datetime | None = Field(default=None, serialization_alias="finishedAt")


class UserJobsResponse(BaseModel):
    jobs: list[OwnedJobSummaryResponse]


class JobsListResponse(BaseModel):
    jobs: list[JobResponse]


class VideoJobsListResponse(BaseModel):
    jobs: list[VideoJobResponse]


class AudioJobsListResponse(BaseModel):
    jobs: list[AudioJobResponse]


class GenerationJobsListResponse(BaseModel):
    jobs: list[GenerationJobResponse]
```

Note: `AudioJobResponse` in the current file has no `owner_id`/`error` neighbor named exactly `error` at a convenient spot — add the field right after its `error: str | None = None` line same as the others; `GenerationJobResponse` likewise has `error: str | None = None` — add it there too.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv\Scripts\python.exe -m pytest tests/test_schemas_auth.py -q`
Expected: PASS (3 tests)

- [ ] **Step 5: Run the full existing backend suite to confirm no regression**

Run: `.venv\Scripts\python.exe -m pytest tests/ -q`
Expected: PASS, same count plus 3 new — adding an optional field with a `None` default never changes existing serialization assertions (existing tests don't assert on the FULL set of response keys, only specific ones — verified by reading `test_models_api.py`/`test_job_status_routes.py` patterns, which all read specific attributes off the parsed response, never do exact-dict-equality against the whole payload)

- [ ] **Step 6: Commit**

```bash
git add app/schemas.py tests/test_schemas_auth.py
git commit -m "feat: add auth/user schemas and ownerId field on job responses"
```

---

## Task 12: `auth_deps.py` — get_current_user, require(permission)

**Files:**
- Create: `app/api/auth_deps.py`
- Test: `tests/test_auth_deps.py`

**Interfaces:**
- Consumes: `AuthenticatedUser`, `authenticated_user_from_record` (Task 7); `Role`, `Permission`, `ROLE_PERMISSIONS` (Task 2); `SESSION_COOKIE_NAME`, `SESSION_TTL_SECONDS`, `create_session_cookie_value`, `verify_session_cookie` (Task 4); `UserStore` (Task 6); `Settings.auth_mode`/`auth_secret` (Task 5).
- Produces: `get_user_store(request) -> UserStore`, `off_mode_user() -> AuthenticatedUser`, `get_current_user(request, response, settings, user_store) -> AuthenticatedUser`, `require(permission) -> Callable`, `current_user_from_request(request: Request | None) -> AuthenticatedUser | None` — consumed by Tasks 13, 14, and `app/api/routes.py` (Tasks 20-23).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_auth_deps.py
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv\Scripts\python.exe -m pytest tests/test_auth_deps.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.api.auth_deps'`

- [ ] **Step 3: Write the implementation**

```python
# app/api/auth_deps.py
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


def _resolve_session_user(request: Request, secret: str | None, user_store: UserStore) -> User | None:
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
        user = _resolve_session_user(request, settings.auth_secret, user_store)
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv\Scripts\python.exe -m pytest tests/test_auth_deps.py -q`
Expected: PASS (11 tests)

- [ ] **Step 5: Commit**

```bash
git add app/api/auth_deps.py tests/test_auth_deps.py
git commit -m "feat: add get_current_user/require FastAPI auth dependencies"
```

---

## Task 13: `auth_routes.py` — login/logout/logout-all/me/change-password/setup

**Files:**
- Create: `app/api/auth_routes.py`
- Test: `tests/test_auth_api.py`

**Interfaces:**
- Consumes: `get_current_user`, `get_user_store`, `off_mode_user` (Task 12); `LocalPasswordProvider` (Task 7); `generate_salt`, `hash_password`, `verify_password` (Task 3); `create_session_cookie_value` (Task 4); `LoginRequest`, `ChangePasswordRequest`, `SetupRequest`, `MeResponse`, `QuotaStatusResponse` (Task 11); `QuotaService` (Task 9); `Role` (Task 2).
- Produces: `router` (`APIRouter(prefix="/api/v1/auth")`) with `POST /login`, `POST /logout`, `POST /logout-all`, `GET /me`, `POST /change-password`, `POST /setup` — included into `app.main` in Task 15.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_auth_api.py
from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services.auth.identity import LocalPasswordProvider
from app.services.auth.passwords import generate_salt, hash_password
from app.services.auth.permissions import Role
from app.services.auth.quotas import QuotaService
from app.services.auth.sessions import SESSION_COOKIE_NAME
from app.services.auth.user_store import UserStore


@pytest.fixture
def multi_mode_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("RUNTIME_DIR", str(tmp_path / "runtime"))
    monkeypatch.setenv("AUTH_MODE", "multi")
    monkeypatch.setenv("AUTH_SECRET", "s" * 32)
    from app.config import get_settings
    get_settings.cache_clear()
    with TestClient(app, client=("127.0.0.1", 12345)) as client:
        yield client
    get_settings.cache_clear()


def _create_admin(client: TestClient, username: str = "admin", password: str = "adminpass1") -> None:
    response = client.post("/api/v1/auth/setup", json={"username": username, "password": password})
    assert response.status_code == 200, response.text


def test_setup_creates_first_admin(multi_mode_client: TestClient) -> None:
    response = multi_mode_client.post(
        "/api/v1/auth/setup", json={"username": "admin", "password": "adminpass1"}
    )
    assert response.status_code == 200


def test_setup_second_time_is_forbidden(multi_mode_client: TestClient) -> None:
    _create_admin(multi_mode_client)

    response = multi_mode_client.post(
        "/api/v1/auth/setup", json={"username": "someone-else", "password": "anotherpass1"}
    )

    assert response.status_code == 403


def test_login_succeeds_with_correct_credentials_and_sets_cookie(multi_mode_client: TestClient) -> None:
    _create_admin(multi_mode_client)

    response = multi_mode_client.post(
        "/api/v1/auth/login", json={"username": "admin", "password": "adminpass1"}
    )

    assert response.status_code == 200
    assert SESSION_COOKIE_NAME in response.cookies


def test_login_fails_with_wrong_password(multi_mode_client: TestClient) -> None:
    _create_admin(multi_mode_client)

    response = multi_mode_client.post(
        "/api/v1/auth/login", json={"username": "admin", "password": "wrong"}
    )

    assert response.status_code == 401
    assert "usuario" in response.json()["detail"].lower() or "contraseña" in response.json()["detail"].lower()


def test_login_rate_limits_after_five_failed_attempts(multi_mode_client: TestClient) -> None:
    _create_admin(multi_mode_client)

    for _ in range(5):
        multi_mode_client.post("/api/v1/auth/login", json={"username": "admin", "password": "wrong"})

    response = multi_mode_client.post(
        "/api/v1/auth/login", json={"username": "admin", "password": "wrong"}
    )

    assert response.status_code == 429


def test_me_returns_authenticated_user_after_login(multi_mode_client: TestClient) -> None:
    _create_admin(multi_mode_client)
    multi_mode_client.post("/api/v1/auth/login", json={"username": "admin", "password": "adminpass1"})

    response = multi_mode_client.get("/api/v1/auth/me")

    assert response.status_code == 200
    body = response.json()
    assert body["username"] == "admin"
    assert body["role"] == "admin"
    assert body["authMode"] == "multi"
    assert "users:manage" in body["permissions"]


def test_me_without_session_returns_401_not_authenticated(multi_mode_client: TestClient) -> None:
    _create_admin(multi_mode_client)

    response = multi_mode_client.get("/api/v1/auth/me")

    assert response.status_code == 401
    assert response.json()["detail"] == "not_authenticated"


def test_me_before_setup_returns_401_setup_required(multi_mode_client: TestClient) -> None:
    response = multi_mode_client.get("/api/v1/auth/me")

    assert response.status_code == 401
    assert response.json()["detail"] == "setup_required"


def test_logout_clears_session(multi_mode_client: TestClient) -> None:
    _create_admin(multi_mode_client)
    multi_mode_client.post("/api/v1/auth/login", json={"username": "admin", "password": "adminpass1"})

    multi_mode_client.post("/api/v1/auth/logout")
    response = multi_mode_client.get("/api/v1/auth/me")

    assert response.status_code == 401


def test_logout_all_invalidates_other_sessions(multi_mode_client: TestClient) -> None:
    _create_admin(multi_mode_client)
    multi_mode_client.post("/api/v1/auth/login", json={"username": "admin", "password": "adminpass1"})
    old_cookie = multi_mode_client.cookies.get(SESSION_COOKIE_NAME)

    multi_mode_client.post("/api/v1/auth/logout-all")

    multi_mode_client.cookies.set(SESSION_COOKIE_NAME, old_cookie)
    response = multi_mode_client.get("/api/v1/auth/me")
    assert response.status_code == 401


def test_change_password_updates_and_clears_must_change_password(multi_mode_client: TestClient) -> None:
    _create_admin(multi_mode_client)
    multi_mode_client.post("/api/v1/auth/login", json={"username": "admin", "password": "adminpass1"})

    response = multi_mode_client.post(
        "/api/v1/auth/change-password",
        json={"currentPassword": "adminpass1", "newPassword": "newpassword1"},
    )
    assert response.status_code == 200

    relogin = multi_mode_client.post(
        "/api/v1/auth/login", json={"username": "admin", "password": "newpassword1"}
    )
    assert relogin.status_code == 200


def test_change_password_rejects_wrong_current_password(multi_mode_client: TestClient) -> None:
    _create_admin(multi_mode_client)
    multi_mode_client.post("/api/v1/auth/login", json={"username": "admin", "password": "adminpass1"})

    response = multi_mode_client.post(
        "/api/v1/auth/change-password",
        json={"currentPassword": "wrong", "newPassword": "newpassword1"},
    )

    assert response.status_code == 401


def test_me_reports_off_mode_pseudo_admin(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RUNTIME_DIR", str(tmp_path / "runtime"))
    monkeypatch.delenv("AUTH_MODE", raising=False)
    from app.config import get_settings
    get_settings.cache_clear()
    try:
        with TestClient(app) as client:
            response = client.get("/api/v1/auth/me")
            assert response.status_code == 200
            body = response.json()
            assert body["authMode"] == "off"
            assert body["userId"] is None
            assert body["role"] == "admin"
    finally:
        get_settings.cache_clear()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv\Scripts\python.exe -m pytest tests/test_auth_api.py -q`
Expected: FAIL — `app.main` doesn't wire `user_store`/`quota_service`/`identity_provider` onto `app.state` yet (Task 15), and `app/api/auth_routes.py` doesn't exist yet. Treat this task's tests as green only once Task 15 also lands; write both together if running strictly sequentially causes import errors (`auth_routes.py` itself has no external-wiring dependency at import time, only at request time via `request.app.state`).

- [ ] **Step 3: Write the implementation**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv\Scripts\python.exe -m pytest tests/test_auth_api.py -q`
Expected: PASS (14 tests) — **only after Task 15's `main.py` wiring also lands**, since these tests exercise the full app via `TestClient(app)`.

- [ ] **Step 5: Commit**

```bash
git add app/api/auth_routes.py tests/test_auth_api.py
git commit -m "feat: add /api/v1/auth/* router (login, logout, me, setup, change-password)"
```

---

## Task 14: `users_routes.py` — users CRUD + GET /users/{id}/jobs

**Files:**
- Create: `app/api/users_routes.py`
- Test: `tests/test_users_api.py`

**Interfaces:**
- Consumes: `require`, `get_user_store` (Task 12); `authenticated_user_from_record` (Task 7); `generate_salt`, `hash_password` (Task 3); `Role`, `Permission` (Task 2); `QuotaService` (Task 9); `CreateUserRequest`, `CreateUserResponse`, `UpdateUserRequest`, `UpdateUserResponse`, `UsersListResponse`, `UserSummaryResponse`, `OwnedJobSummaryResponse`, `UserJobsResponse` (Task 11).
- Produces: `router` (`APIRouter(prefix="/api/v1/users")`) with `GET ""`, `POST ""`, `PATCH "/{user_id}"`, `GET "/{user_id}/jobs"` — included into `app.main` in Task 15.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_users_api.py
from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import get_settings
from app.main import app


@pytest.fixture
def admin_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("RUNTIME_DIR", str(tmp_path / "runtime"))
    monkeypatch.setenv("AUTH_MODE", "multi")
    monkeypatch.setenv("AUTH_SECRET", "s" * 32)
    get_settings.cache_clear()
    with TestClient(app, client=("127.0.0.1", 12345)) as client:
        client.post("/api/v1/auth/setup", json={"username": "admin", "password": "adminpass1"})
        client.post("/api/v1/auth/login", json={"username": "admin", "password": "adminpass1"})
        yield client
    get_settings.cache_clear()


def test_create_user_returns_temporary_password(admin_client: TestClient) -> None:
    response = admin_client.post("/api/v1/users", json={"username": "bob", "role": "user"})

    assert response.status_code == 201
    body = response.json()
    assert body["user"]["username"] == "bob"
    assert body["user"]["mustChangePassword"] is True
    assert len(body["temporaryPassword"]) > 0


def test_create_user_rejects_duplicate_username(admin_client: TestClient) -> None:
    admin_client.post("/api/v1/users", json={"username": "bob", "role": "user"})

    response = admin_client.post("/api/v1/users", json={"username": "bob", "role": "user"})

    assert response.status_code == 409


def test_list_users_includes_admin_and_created_users(admin_client: TestClient) -> None:
    admin_client.post("/api/v1/users", json={"username": "bob", "role": "user"})

    response = admin_client.get("/api/v1/users")

    assert response.status_code == 200
    usernames = {u["username"] for u in response.json()["users"]}
    assert usernames == {"admin", "bob"}


def test_update_user_role_and_disabled(admin_client: TestClient) -> None:
    created = admin_client.post("/api/v1/users", json={"username": "bob", "role": "user"}).json()
    user_id = created["user"]["id"]

    response = admin_client.patch(f"/api/v1/users/{user_id}", json={"role": "admin", "disabled": True})

    assert response.status_code == 200
    body = response.json()["user"]
    assert body["role"] == "admin"
    assert body["disabled"] is True


def test_update_user_reset_password_returns_temporary_password(admin_client: TestClient) -> None:
    created = admin_client.post("/api/v1/users", json={"username": "bob", "role": "user"}).json()
    user_id = created["user"]["id"]

    response = admin_client.patch(f"/api/v1/users/{user_id}", json={"resetPassword": True})

    assert response.status_code == 200
    body = response.json()
    assert body["temporaryPassword"] is not None
    assert body["user"]["mustChangePassword"] is True


def test_update_unknown_user_returns_404(admin_client: TestClient) -> None:
    response = admin_client.patch("/api/v1/users/does-not-exist", json={"disabled": True})
    assert response.status_code == 404


def test_users_endpoints_require_users_manage_permission(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RUNTIME_DIR", str(tmp_path / "runtime"))
    monkeypatch.setenv("AUTH_MODE", "multi")
    monkeypatch.setenv("AUTH_SECRET", "s" * 32)
    get_settings.cache_clear()
    try:
        with TestClient(app, client=("127.0.0.1", 12345)) as client:
            client.post("/api/v1/auth/setup", json={"username": "admin", "password": "adminpass1"})
            client.post("/api/v1/auth/login", json={"username": "admin", "password": "adminpass1"})
            created = client.post("/api/v1/users", json={"username": "bob", "role": "user"}).json()
            temp_password = created["temporaryPassword"]
            client.post("/api/v1/auth/logout")

            client.post("/api/v1/auth/login", json={"username": "bob", "password": temp_password})
            response = client.get("/api/v1/users")

            assert response.status_code == 403
    finally:
        get_settings.cache_clear()


def test_get_user_jobs_returns_empty_list_for_new_user(admin_client: TestClient) -> None:
    created = admin_client.post("/api/v1/users", json={"username": "bob", "role": "user"}).json()
    user_id = created["user"]["id"]

    response = admin_client.get(f"/api/v1/users/{user_id}/jobs")

    assert response.status_code == 200
    assert response.json()["jobs"] == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv\Scripts\python.exe -m pytest tests/test_users_api.py -q`
Expected: FAIL — `app/api/users_routes.py` doesn't exist and isn't wired yet (finishes green together with Task 15)

- [ ] **Step 3: Write the implementation**

```python
# app/api/users_routes.py
from __future__ import annotations

import secrets

from fastapi import APIRouter, Depends, HTTPException, Request

from app.api.auth_deps import get_user_store, require
from app.schemas import (
    CreateUserRequest,
    CreateUserResponse,
    OwnedJobSummaryResponse,
    UpdateUserRequest,
    UpdateUserResponse,
    UserJobsResponse,
    UserSummaryResponse,
    UsersListResponse,
)
from app.services.auth.identity import AuthenticatedUser, authenticated_user_from_record
from app.services.auth.passwords import generate_salt, hash_password
from app.services.auth.permissions import Permission, Role
from app.services.auth.quotas import QuotaService
from app.services.auth.user_store import User, UserStore

router = APIRouter(prefix="/api/v1/users", tags=["users"])


def get_quota_service(request: Request) -> QuotaService:
    return request.app.state.quota_service


def _user_summary(user: User, quotas: QuotaService) -> UserSummaryResponse:
    status = quotas.status_for(authenticated_user_from_record(user))
    return UserSummaryResponse(
        id=user.id, username=user.username, role=user.role.value, disabled=user.disabled,
        must_change_password=user.must_change_password, quota_overrides=user.quota_overrides,
        created_at=user.created_at, used_jobs_today=status.used_jobs_today,
        used_gpu_seconds_today=status.used_gpu_seconds_today,
    )


@router.get("", response_model=UsersListResponse)
async def list_users(
    user_store: UserStore = Depends(get_user_store),
    quotas: QuotaService = Depends(get_quota_service),
    _current_user: AuthenticatedUser = Depends(require(Permission.users_manage)),
) -> UsersListResponse:
    return UsersListResponse(users=[_user_summary(u, quotas) for u in user_store.list()])


@router.post("", response_model=CreateUserResponse, status_code=201)
async def create_user(
    payload: CreateUserRequest,
    user_store: UserStore = Depends(get_user_store),
    quotas: QuotaService = Depends(get_quota_service),
    _current_user: AuthenticatedUser = Depends(require(Permission.users_manage)),
) -> CreateUserResponse:
    try:
        role = Role(payload.role)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Unknown role: {payload.role!r}") from exc
    temporary_password = secrets.token_urlsafe(12)
    salt = generate_salt()
    password_hash = hash_password(temporary_password, salt)
    try:
        user = user_store.create(
            username=payload.username, password_hash=password_hash, salt=salt,
            role=role, must_change_password=True,
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return CreateUserResponse(user=_user_summary(user, quotas), temporary_password=temporary_password)


def _apply_user_update(user_store: UserStore, user_id: str, payload: UpdateUserRequest) -> User:
    user = user_store.get(user_id)
    if user is None:
        raise ValueError(f"Unknown user id: {user_id!r}")
    if payload.role is not None:
        user = user_store.set_role(user_id, Role(payload.role))
    if payload.disabled is not None:
        user = user_store.set_disabled(user_id, payload.disabled)
    if payload.quota_overrides is not None:
        user = user_store.set_quota_overrides(user_id, payload.quota_overrides)
    return user


@router.patch("/{user_id}", response_model=UpdateUserResponse)
async def update_user(
    user_id: str,
    payload: UpdateUserRequest,
    user_store: UserStore = Depends(get_user_store),
    quotas: QuotaService = Depends(get_quota_service),
    _current_user: AuthenticatedUser = Depends(require(Permission.users_manage)),
) -> UpdateUserResponse:
    try:
        user = _apply_user_update(user_store, user_id, payload)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    temporary_password = None
    if payload.reset_password:
        temporary_password = secrets.token_urlsafe(12)
        salt = generate_salt()
        password_hash = hash_password(temporary_password, salt)
        user = user_store.set_password(user_id, password_hash, salt, must_change_password=True)
    return UpdateUserResponse(user=_user_summary(user, quotas), temporary_password=temporary_password)


@router.get("/{user_id}/jobs", response_model=UserJobsResponse)
async def get_user_jobs(
    user_id: str,
    request: Request,
    _current_user: AuthenticatedUser = Depends(require(Permission.jobs_read_all)),
) -> UserJobsResponse:
    managers = (
        ("image", request.app.state.job_manager),
        ("video", request.app.state.video_job_manager),
        ("audio", request.app.state.audio_job_manager),
        ("generation", request.app.state.generation_job_manager),
    )
    jobs = [
        OwnedJobSummaryResponse(
            id=job.id, kind=kind, status=job.status,
            original_filename=getattr(job, "original_filename", None),
            created_at=job.created_at, finished_at=job.finished_at,
        )
        for kind, manager in managers
        for job in manager.jobs.values()
        if job.owner_id == user_id
    ]
    return UserJobsResponse(jobs=jobs)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv\Scripts\python.exe -m pytest tests/test_users_api.py -q`
Expected: PASS (8 tests) — **only after Task 15 lands**

- [ ] **Step 5: Commit**

```bash
git add app/api/users_routes.py tests/test_users_api.py
git commit -m "feat: add /api/v1/users/* router (CRUD + per-user job listing)"
```

---

## Task 15: `main.py` — wire auth services, middleware, routers

**Files:**
- Modify: `app/main.py`

**Interfaces:**
- Consumes: everything from Tasks 7-14.
- Produces: `app.state.user_store`, `app.state.identity_provider`, `app.state.quota_service` populated at startup; `auth_router`/`users_router` mounted; `LoopbackGuardMiddleware` active. This is the task that makes Tasks 13/14's tests actually pass end-to-end.

- [ ] **Step 1: No new standalone test file** — this task's correctness is verified by re-running Tasks 13/14's test files plus the full suite.

- [ ] **Step 2: Add imports** to `app/main.py`, alongside the existing `from app.api.*` / `from app.services.*` imports:

```python
from app.api.auth_routes import router as auth_router
from app.api.users_routes import router as users_router
from app.config import ensure_auth_secret
from app.security import LoopbackGuardMiddleware
from app.services.auth.identity import LocalPasswordProvider
from app.services.auth.quotas import QuotaService
from app.services.auth.user_store import UserStore
```

- [ ] **Step 3: Construct auth services in `lifespan`**, right after `settings = get_settings()` (line 49):

```python
    if settings.auth_mode == "multi":
        ensure_auth_secret(settings)
    user_store = UserStore(settings)
    identity_provider = LocalPasswordProvider(user_store)
    quota_service = QuotaService(settings)
```

- [ ] **Step 4: Attach the 4 job managers to `quota_service`** right after `audio_job_manager = AudioJobManager(...)` is constructed (immediately before the existing `retention_sweeper = RetentionSweeper(...)` line):

```python
    quota_service.attach_managers(job_manager, video_job_manager, audio_job_manager, generation_job_manager)
```

- [ ] **Step 5: Pass `quota_service` into each of the 4 managers' constructors** — modify the existing construction calls (this is the interface Tasks 16-19 rely on):

```python
    job_manager = JobManager(
        settings, engine, device_semaphores, onnx_engine=onnx_engine, registry=model_registry,
        devices=devices_service, device_router=device_router, quota_service=quota_service,
    )
```
```python
    video_job_manager = VideoJobManager(
        settings, video_upscaler, media_tools, device_semaphores, registry=model_registry,
        devices=devices_service, device_router=device_router, quota_service=quota_service,
    )
```
```python
    audio_job_manager = AudioJobManager(
        settings, audio_pipeline, device_semaphores, devices=devices_service, quota_service=quota_service,
    )
```
```python
    generation_job_manager = GenerationJobManager(
        settings, generation_engine, device_semaphores, registry=model_registry, upscale_engine=engine,
        onnx_upscale_engine=onnx_engine, devices=devices_service, quota_service=quota_service,
    )
```

Note: `generation_job_manager` is constructed *before* `job_manager`/`video_job_manager`/`audio_job_manager` in the current file (see line 69 vs 82/104/114) — keep that existing order, just add the new kwarg to each call in place. `quota_service` must exist (Step 3) before any of these 4 constructor calls, which it already does since Step 3 runs immediately after `settings = get_settings()`.

- [ ] **Step 6: Register the new state on `app.state`**, alongside the existing assignments (after `app.state.generation_installer = generation_installer`):

```python
    app.state.user_store = user_store
    app.state.identity_provider = identity_provider
    app.state.quota_service = quota_service
```

- [ ] **Step 7: Add the middleware and routers** at module scope, after the existing `app.add_middleware(OriginGuardMiddleware, ...)` / `app.include_router(capability_router)` lines:

```python
app.add_middleware(LoopbackGuardMiddleware, auth_mode=settings.auth_mode)
app.include_router(auth_router)
app.include_router(users_router)
```

(Keep `configure_web_routes(app)` as the last call, since its catch-all route must stay registered after every other router.)

- [ ] **Step 8: Run Tasks 13/14's test files, then the full suite**

Run: `.venv\Scripts\python.exe -m pytest tests/test_auth_api.py tests/test_users_api.py -q`
Expected: PASS (22 tests total)

Run: `.venv\Scripts\python.exe -m pytest tests/ -q`
Expected: PASS, full existing suite (1207) + all new tests so far, zero regressions — this is the first point where `AUTH_MODE=off` default behavior for the WHOLE app (not just isolated units) can be verified end-to-end; treat any failure here as blocking before continuing to Task 16.

- [ ] **Step 9: Commit**

```bash
git add app/main.py
git commit -m "feat: wire auth services, LoopbackGuardMiddleware and auth/users routers into main.py"
```

---

## Task 16: `job_manager.py` (image) — owner_id + quota admission

**Files:**
- Modify: `app/models.py` (add `owner_id` to `UpscaleJob`)
- Modify: `app/services/job_manager.py`
- Test: `tests/test_job_manager_ownership.py`

**Interfaces:**
- Consumes: `AuthenticatedUser` (Task 7); `QuotaService` (Task 9).
- Produces: `UpscaleJob.owner_id: str | None = None`; `JobManager.__init__(..., quota_service: QuotaService | None = None)`; `JobManager.create_job(..., owner: AuthenticatedUser | None = None)` — consumed by Task 20 (route layer) and Task 15 (already wired).

- [ ] **Step 1: Add `owner_id` to `UpscaleJob`** in `app/models.py` (right after `metadata: dict[str, Any] = field(default_factory=dict)` in the `UpscaleJob` dataclass):

```python
    owner_id: str | None = None
```

- [ ] **Step 2: Write the failing tests**

```python
# tests/test_job_manager_ownership.py
from __future__ import annotations

from pathlib import Path

import pytest

from app.config import Settings
from app.exceptions import QuotaExceededError
from app.models import JobStatus
from app.services.auth.identity import AuthenticatedUser
from app.services.auth.permissions import ROLE_PERMISSIONS, Role
from app.services.auth.quotas import QuotaService
from app.services.device_semaphores import DeviceSemaphores
from app.services.engines.base import UpscaleEngine
from app.services.job_manager import JobManager


class FakeEngine(UpscaleEngine):
    def available(self) -> bool:
        return True

    async def run(self, job):
        return job.source_path


def make_settings(tmp_path: Path) -> Settings:
    return Settings(_env_file=None, RUNTIME_DIR=str(tmp_path / "runtime"))


def make_user(user_id: str = "u1", role: Role = Role.user, overrides: dict | None = None) -> AuthenticatedUser:
    return AuthenticatedUser(
        id=user_id, username="alice", role=role, permissions=ROLE_PERMISSIONS[role],
        must_change_password=False, quota_overrides=overrides or {},
    )


def make_image_file(tmp_path: Path) -> Path:
    from PIL import Image

    path = tmp_path / "source.png"
    Image.new("RGB", (8, 8)).save(path)
    return path


async def test_create_job_without_owner_leaves_owner_id_none(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    manager = JobManager(settings, FakeEngine(), DeviceSemaphores(settings))

    job = await manager.create_job(
        source_path=make_image_file(tmp_path), original_filename="a.png",
        model_name="realesrgan-x4plus", scale=4, output_format="png",
    )

    assert job.owner_id is None


async def test_create_job_with_owner_stamps_owner_id(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    manager = JobManager(settings, FakeEngine(), DeviceSemaphores(settings))

    job = await manager.create_job(
        source_path=make_image_file(tmp_path), original_filename="a.png",
        model_name="realesrgan-x4plus", scale=4, output_format="png", owner=make_user("u1"),
    )

    assert job.owner_id == "u1"


async def test_create_job_calls_quota_check_admission_when_configured(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    quota_service = QuotaService(settings)
    manager = JobManager(settings, FakeEngine(), DeviceSemaphores(settings), quota_service=quota_service)
    quota_service.attach_managers(manager)
    manager.jobs["existing"] = type(
        "FakeJob", (), {"owner_id": "u1", "status": JobStatus.running}
    )()

    with pytest.raises(QuotaExceededError):
        await manager.create_job(
            source_path=make_image_file(tmp_path), original_filename="a.png",
            model_name="realesrgan-x4plus", scale=4, output_format="png", owner=make_user("u1"),
        )


async def test_create_job_skips_quota_check_when_owner_is_none(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    quota_service = QuotaService(settings)
    manager = JobManager(settings, FakeEngine(), DeviceSemaphores(settings), quota_service=quota_service)
    quota_service.attach_managers(manager)

    job = await manager.create_job(
        source_path=make_image_file(tmp_path), original_filename="a.png",
        model_name="realesrgan-x4plus", scale=4, output_format="png",
    )

    assert job.owner_id is None
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `.venv\Scripts\python.exe -m pytest tests/test_job_manager_ownership.py -q`
Expected: FAIL with `TypeError: create_job() got an unexpected keyword argument 'owner'`

- [ ] **Step 4: Write the implementation**

In `app/services/job_manager.py`, add imports:

```python
from app.services.auth.identity import AuthenticatedUser
from app.services.auth.quotas import QuotaService
```

Modify `__init__` to accept and store `quota_service`:

```python
    def __init__(
        self,
        settings: Settings,
        engine: UpscaleEngine,
        device_semaphores: DeviceSemaphores,
        *,
        onnx_engine: UpscaleEngine | None = None,
        registry: ModelRegistry | None = None,
        devices: DevicesService | None = None,
        device_router: DeviceRouter | None = None,
        quota_service: QuotaService | None = None,
    ) -> None:
        self.settings = settings
        self.engine = engine
        self.onnx_engine = onnx_engine
        self.registry = registry
        self.devices = devices
        self.jobs: dict[str, UpscaleJob] = {}
        self.queue: asyncio.Queue[UpscaleJob] = asyncio.Queue(maxsize=settings.max_queue_size)
        self.device_semaphores = device_semaphores
        self.device_router = device_router or DeviceRouter(device_semaphores)
        self.worker_tasks: list[asyncio.Task] = []
        self._active: dict[str, asyncio.Task] = {}
        self.quota_service = quota_service
```

Modify `create_job`'s signature and body:

```python
    async def create_job(
        self,
        *,
        source_path: Path,
        original_filename: str,
        model_name: str,
        scale: int,
        output_format: str,
        model_id: str | None = None,
        device: str | None = None,
        job_id: str | None = None,
        owner: AuthenticatedUser | None = None,
    ) -> UpscaleJob:
        await asyncio.to_thread(self._validate_input_image, source_path)
        resolved_model_id = model_id if model_id is not None else model_name
        if device is not None and device != AUTO_DEVICE_ID and self.devices is not None:
            await asyncio.to_thread(self.devices.validate, device)
        resolution = self._resolve_model(
            model_id=resolved_model_id,
            scale=scale,
            output_format=output_format,
            device=device,
        )
        if device == AUTO_DEVICE_ID:
            await self._validate_auto_device(resolution.kind)
        if owner is not None and self.quota_service is not None:
            self.quota_service.check_admission(owner)

        job = UpscaleJob(
            source_path=source_path,
            original_filename=original_filename,
            model_name=resolution.engine_model_name,
            scale=resolution.scale,
            output_format=output_format,
            model_id=resolution.model_id,
            device=device,
            owner_id=owner.id if owner is not None else None,
        )
        if job_id is not None:
            job.id = job_id
        self._enqueue(job)
        self.jobs[job.id] = job
        return job
```

Modify `_execute_job`'s `finally` block to record GPU-second usage:

```python
        finally:
            self._active.pop(job.id, None)
            job.finished_at = utc_now()
            self._unlink_source_safely(job.source_path)
            self.queue.task_done()
            self._record_quota_usage(job)
```

Add the new method near `_unlink_source_safely`:

```python
    def _record_quota_usage(self, job: UpscaleJob) -> None:
        if self.quota_service is None or job.started_at is None:
            return
        duration = (job.finished_at - job.started_at).total_seconds()
        self.quota_service.record_usage(job.owner_id, duration)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv\Scripts\python.exe -m pytest tests/test_job_manager_ownership.py -q`
Expected: PASS (4 tests)

- [ ] **Step 6: Run the full existing backend suite to confirm no regression**

Run: `.venv\Scripts\python.exe -m pytest tests/ -q`
Expected: PASS — every existing direct call to `JobManager(...)`/`manager.create_job(...)` omits `quota_service`/`owner`, both optional with safe `None` defaults, so behavior is byte-identical

- [ ] **Step 7: Commit**

```bash
git add app/models.py app/services/job_manager.py tests/test_job_manager_ownership.py
git commit -m "feat: add owner_id and quota admission to JobManager (image jobs)"
```

---

## Task 17: `video_job_manager.py` — owner_id + quota admission

**Files:**
- Modify: `app/models.py` (add `owner_id` to `VideoUpscaleJob`)
- Modify: `app/services/video_job_manager.py`
- Test: `tests/test_video_job_manager_ownership.py`

**Interfaces:** Same shape as Task 16, for `VideoUpscaleJob`/`VideoJobManager`.

- [ ] **Step 1: Add `owner_id` to `VideoUpscaleJob`** in `app/models.py` (right after `metadata: dict[str, Any] = field(default_factory=dict)` in the `VideoUpscaleJob` dataclass):

```python
    owner_id: str | None = None
```

- [ ] **Step 2: Write the failing tests**

```python
# tests/test_video_job_manager_ownership.py
from __future__ import annotations

from pathlib import Path

import pytest

from app.config import Settings
from app.exceptions import QuotaExceededError
from app.models import JobStatus
from app.services.auth.identity import AuthenticatedUser
from app.services.auth.permissions import ROLE_PERMISSIONS, Role
from app.services.auth.quotas import QuotaService
from app.services.device_semaphores import DeviceSemaphores
from app.services.video_job_manager import VideoJobManager


class FakeUpscaler:
    async def run(self, job, fps_multiplier: int = 1):
        return job.source_path


class FakeMediaTools:
    def available(self) -> bool:
        return True

    async def ffprobe_json(self, source_path: Path) -> dict:
        return {"streams": [{"codec_type": "video", "avg_frame_rate": "24/1"}]}


def make_settings(tmp_path: Path) -> Settings:
    return Settings(_env_file=None, RUNTIME_DIR=str(tmp_path / "runtime"))


def make_user(user_id: str = "u1", role: Role = Role.user, overrides: dict | None = None) -> AuthenticatedUser:
    return AuthenticatedUser(
        id=user_id, username="alice", role=role, permissions=ROLE_PERMISSIONS[role],
        must_change_password=False, quota_overrides=overrides or {},
    )


def make_manager(tmp_path: Path, quota_service: QuotaService | None = None) -> VideoJobManager:
    settings = make_settings(tmp_path)
    return VideoJobManager(
        settings, FakeUpscaler(), FakeMediaTools(), DeviceSemaphores(settings), quota_service=quota_service,
    )


async def test_create_job_without_owner_leaves_owner_id_none(tmp_path: Path) -> None:
    manager = make_manager(tmp_path)
    source = tmp_path / "source.mp4"
    source.write_bytes(b"fake")

    job = await manager.create_job(
        source_path=source, original_filename="source.mp4", model_name="realesr-animevideov3-x2",
        scale=2, output_container="mp4", video_codec="libx264", video_preset="medium", crf=18,
        keep_audio=False,
    )

    assert job.owner_id is None


async def test_create_job_with_owner_stamps_owner_id(tmp_path: Path) -> None:
    manager = make_manager(tmp_path)
    source = tmp_path / "source.mp4"
    source.write_bytes(b"fake")

    job = await manager.create_job(
        source_path=source, original_filename="source.mp4", model_name="realesr-animevideov3-x2",
        scale=2, output_container="mp4", video_codec="libx264", video_preset="medium", crf=18,
        keep_audio=False, owner=make_user("u1"),
    )

    assert job.owner_id == "u1"


async def test_create_job_calls_quota_check_admission_when_configured(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    quota_service = QuotaService(settings)
    manager = make_manager(tmp_path, quota_service=quota_service)
    quota_service.attach_managers(manager)
    manager.jobs["existing"] = type("FakeJob", (), {"owner_id": "u1", "status": JobStatus.running})()
    source = tmp_path / "source.mp4"
    source.write_bytes(b"fake")

    with pytest.raises(QuotaExceededError):
        await manager.create_job(
            source_path=source, original_filename="source.mp4", model_name="realesr-animevideov3-x2",
            scale=2, output_container="mp4", video_codec="libx264", video_preset="medium", crf=18,
            keep_audio=False, owner=make_user("u1"),
        )
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `.venv\Scripts\python.exe -m pytest tests/test_video_job_manager_ownership.py -q`
Expected: FAIL with `TypeError: __init__() got an unexpected keyword argument 'quota_service'`

- [ ] **Step 4: Write the implementation**

Apply the same shape of edit as Task 16 to `app/services/video_job_manager.py`:

- Add imports: `from app.services.auth.identity import AuthenticatedUser` and `from app.services.auth.quotas import QuotaService`.
- Add `quota_service: QuotaService | None = None` to `__init__`'s keyword-only params, store as `self.quota_service = quota_service`.
- Add `owner: AuthenticatedUser | None = None` to `create_job`'s keyword-only params. Right before `job = VideoUpscaleJob(...)` is constructed, add:

```python
        if owner is not None and self.quota_service is not None:
            self.quota_service.check_admission(owner)
```

  and add `owner_id=owner.id if owner is not None else None,` as a new field in the `VideoUpscaleJob(...)` constructor call (alongside `probe=probe,`).

- In `_execute_job`'s `finally` block, add `self._record_quota_usage(job)` as the last line, and add the method:

```python
    def _record_quota_usage(self, job: VideoUpscaleJob) -> None:
        if self.quota_service is None or job.started_at is None:
            return
        duration = (job.finished_at - job.started_at).total_seconds()
        self.quota_service.record_usage(job.owner_id, duration)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv\Scripts\python.exe -m pytest tests/test_video_job_manager_ownership.py -q`
Expected: PASS (3 tests)

- [ ] **Step 6: Run the full existing backend suite to confirm no regression**

Run: `.venv\Scripts\python.exe -m pytest tests/ -q`
Expected: PASS, zero regressions (same reasoning as Task 16)

- [ ] **Step 7: Commit**

```bash
git add app/models.py app/services/video_job_manager.py tests/test_video_job_manager_ownership.py
git commit -m "feat: add owner_id and quota admission to VideoJobManager"
```

---

## Task 18: `audio_job_manager.py` — owner_id + quota admission

**Files:**
- Modify: `app/models.py` (add `owner_id` to `AudioJob`)
- Modify: `app/services/audio_job_manager.py`
- Test: `tests/test_audio_job_manager_ownership.py`

**Interfaces:** Same shape as Task 16, for `AudioJob`/`AudioJobManager`.

- [ ] **Step 1: Add `owner_id` to `AudioJob`** in `app/models.py` (right after `metadata: dict[str, Any] = field(default_factory=dict)` in the `AudioJob` dataclass):

```python
    owner_id: str | None = None
```

- [ ] **Step 2: Write the failing tests**

```python
# tests/test_audio_job_manager_ownership.py
from __future__ import annotations

from pathlib import Path

import pytest

from app.config import Settings
from app.exceptions import QuotaExceededError
from app.models import JobStatus
from app.services.auth.identity import AuthenticatedUser
from app.services.auth.permissions import ROLE_PERMISSIONS, Role
from app.services.auth.quotas import QuotaService
from app.services.audio_job_manager import AudioJobManager
from app.services.device_semaphores import DeviceSemaphores


def make_settings(tmp_path: Path) -> Settings:
    return Settings(_env_file=None, RUNTIME_DIR=str(tmp_path / "runtime"), ENABLE_AUDIO_ENHANCE=True)


def make_user(user_id: str = "u1", role: Role = Role.user, overrides: dict | None = None) -> AuthenticatedUser:
    return AuthenticatedUser(
        id=user_id, username="alice", role=role, permissions=ROLE_PERMISSIONS[role],
        must_change_password=False, quota_overrides=overrides or {},
    )


def make_manager(tmp_path: Path, quota_service: QuotaService | None = None) -> AudioJobManager:
    settings = make_settings(tmp_path)
    return AudioJobManager(settings, pipeline=None, device_semaphores=DeviceSemaphores(settings), quota_service=quota_service)


async def test_create_job_with_owner_stamps_owner_id(tmp_path: Path, monkeypatch) -> None:
    manager = make_manager(tmp_path)
    monkeypatch.setattr(manager.settings, "audio_enhance_available", lambda mode: True)
    source = tmp_path / "source.wav"
    source.write_bytes(b"fake")

    job = await manager.create_job(
        source_path=source, original_filename="source.wav", denoise="deepfilter", owner=make_user("u1"),
    )

    assert job.owner_id == "u1"


async def test_create_job_calls_quota_check_admission_when_configured(tmp_path: Path, monkeypatch) -> None:
    settings = make_settings(tmp_path)
    quota_service = QuotaService(settings)
    manager = AudioJobManager(settings, pipeline=None, device_semaphores=DeviceSemaphores(settings), quota_service=quota_service)
    monkeypatch.setattr(manager.settings, "audio_enhance_available", lambda mode: True)
    quota_service.attach_managers(manager)
    manager.jobs["existing"] = type("FakeJob", (), {"owner_id": "u1", "status": JobStatus.running})()
    source = tmp_path / "source.wav"
    source.write_bytes(b"fake")

    with pytest.raises(QuotaExceededError):
        await manager.create_job(
            source_path=source, original_filename="source.wav", denoise="deepfilter", owner=make_user("u1"),
        )
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `.venv\Scripts\python.exe -m pytest tests/test_audio_job_manager_ownership.py -q`
Expected: FAIL with `TypeError: __init__() got an unexpected keyword argument 'quota_service'`

- [ ] **Step 4: Write the implementation**

Apply the same shape of edit as Task 16 to `app/services/audio_job_manager.py`:

- Add imports: `from app.services.auth.identity import AuthenticatedUser` and `from app.services.auth.quotas import QuotaService`.
- Add `quota_service: QuotaService | None = None` to `__init__`'s keyword-only params, store as `self.quota_service = quota_service`.
- Add `owner: AuthenticatedUser | None = None` to `create_job`'s keyword-only params. Right before `job = AudioJob(...)` is constructed, add:

```python
        if owner is not None and self.quota_service is not None:
            self.quota_service.check_admission(owner)
```

  and add `owner_id=owner.id if owner is not None else None,` to the `AudioJob(...)` constructor call.

- In `_execute_job`'s `finally` block, add `self._record_quota_usage(job)` as the last line, and add:

```python
    def _record_quota_usage(self, job: AudioJob) -> None:
        if self.quota_service is None or job.started_at is None:
            return
        duration = (job.finished_at - job.started_at).total_seconds()
        self.quota_service.record_usage(job.owner_id, duration)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv\Scripts\python.exe -m pytest tests/test_audio_job_manager_ownership.py -q`
Expected: PASS (2 tests)

- [ ] **Step 6: Run the full existing backend suite to confirm no regression**

Run: `.venv\Scripts\python.exe -m pytest tests/ -q`
Expected: PASS, zero regressions

- [ ] **Step 7: Commit**

```bash
git add app/models.py app/services/audio_job_manager.py tests/test_audio_job_manager_ownership.py
git commit -m "feat: add owner_id and quota admission to AudioJobManager"
```

---

## Task 19: `generation_job_manager.py` — owner_id + quota admission

**Files:**
- Modify: `app/models.py` (add `owner_id` to `GenerationJob`)
- Modify: `app/services/generation_job_manager.py`
- Test: `tests/test_generation_job_manager_ownership.py`

**Interfaces:** Same shape as Task 16, for `GenerationJob`/`GenerationJobManager`. Also stamps `owner_id` on the internal `UpscaleJob` built inside `_run_auto_upscale` (never routed through `JobManager.create_job`, so it needs its own explicit assignment).

- [ ] **Step 1: Add `owner_id` to `GenerationJob`** in `app/models.py` (right after `metadata: dict[str, Any] = field(default_factory=dict)` in the `GenerationJob` dataclass):

```python
    owner_id: str | None = None
```

- [ ] **Step 2: Write the failing tests**

```python
# tests/test_generation_job_manager_ownership.py
from __future__ import annotations

from pathlib import Path

import pytest

from app.config import Settings
from app.exceptions import QuotaExceededError
from app.models import JobStatus
from app.services.auth.identity import AuthenticatedUser
from app.services.auth.permissions import ROLE_PERMISSIONS, Role
from app.services.auth.quotas import QuotaService
from app.services.device_semaphores import DeviceSemaphores
from app.services.generation_job_manager import GenerationJobManager
from app.services.model_registry import ModelEntry, ModelKind, ModelRegistry

MODEL_ID = "test-generation-model"


def make_settings(tmp_path: Path) -> Settings:
    return Settings(_env_file=None, RUNTIME_DIR=str(tmp_path / "runtime"))


def make_user(user_id: str = "u1", role: Role = Role.user, overrides: dict | None = None) -> AuthenticatedUser:
    return AuthenticatedUser(
        id=user_id, username="alice", role=role, permissions=ROLE_PERMISSIONS[role],
        must_change_password=False, quota_overrides=overrides or {},
    )


class FakeEngine:
    async def run(self, **kwargs):
        return Path(kwargs["output_path"])


def make_manager(tmp_path: Path, quota_service: QuotaService | None = None):
    settings = make_settings(tmp_path)
    registry = ModelRegistry(settings)
    registry.register(ModelEntry(
        id=MODEL_ID, name="Test", kind=ModelKind.diffusion_onnx, source="local", size_bytes=0,
        file_path=MODEL_ID,
    ))
    (settings.models_path / MODEL_ID).mkdir(parents=True, exist_ok=True)
    manager = GenerationJobManager(
        settings, FakeEngine(), DeviceSemaphores(settings), registry=registry,
        upscale_engine=object(), quota_service=quota_service,
    )
    return manager


async def test_create_job_with_owner_stamps_owner_id(tmp_path: Path) -> None:
    manager = make_manager(tmp_path)

    job = await manager.create_job(prompt="a cat", model_id=MODEL_ID, owner=make_user("u1"))

    assert job.owner_id == "u1"


async def test_create_job_calls_quota_check_admission_when_configured(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    quota_service = QuotaService(settings)
    manager = make_manager(tmp_path, quota_service=quota_service)
    quota_service.attach_managers(manager)
    manager.jobs["existing"] = type("FakeJob", (), {"owner_id": "u1", "status": JobStatus.running})()

    with pytest.raises(QuotaExceededError):
        await manager.create_job(prompt="a cat", model_id=MODEL_ID, owner=make_user("u1"))
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `.venv\Scripts\python.exe -m pytest tests/test_generation_job_manager_ownership.py -q`
Expected: FAIL with `TypeError: __init__() got an unexpected keyword argument 'quota_service'`

- [ ] **Step 4: Write the implementation**

Apply the same shape of edit as Task 16 to `app/services/generation_job_manager.py`:

- Add imports: `from app.services.auth.identity import AuthenticatedUser` and `from app.services.auth.quotas import QuotaService`.
- Add `quota_service: QuotaService | None = None` to `__init__`'s keyword-only params, store as `self.quota_service = quota_service`.
- Add `owner: AuthenticatedUser | None = None` to `create_job`'s keyword-only params. Right before `job = GenerationJob(...)` is constructed, add:

```python
        if owner is not None and self.quota_service is not None:
            self.quota_service.check_admission(owner)
```

  and add `owner_id=owner.id if owner is not None else None,` to the `GenerationJob(...)` constructor call.

- In `_execute_job`'s `finally` block, add `self._record_quota_usage(job)` as the last line, and add:

```python
    def _record_quota_usage(self, job: GenerationJob) -> None:
        if self.quota_service is None or job.started_at is None:
            return
        duration = (job.finished_at - job.started_at).total_seconds()
        self.quota_service.record_usage(job.owner_id, duration)
```

- In `_run_auto_upscale`, add `owner_id=job.owner_id,` to the ephemeral `UpscaleJob(...)` constructor call, so the internal upscale step (never routed through `JobManager`, never independently listed/cancelled) still carries the same ownership as its parent `GenerationJob` for consistency:

```python
        upscale_job = UpscaleJob(
            source_path=generated,
            original_filename=generated.name,
            model_name=job.upscale_model_name or "",
            scale=job.upscale_scale or UPSCALE_SCALE_RANGE[0],
            output_format="png",
            model_id=job.upscale_model_id,
            device=device,
            owner_id=job.owner_id,
        )
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv\Scripts\python.exe -m pytest tests/test_generation_job_manager_ownership.py -q`
Expected: PASS (2 tests)

- [ ] **Step 6: Run the full existing backend suite to confirm no regression**

Run: `.venv\Scripts\python.exe -m pytest tests/ -q`
Expected: PASS, zero regressions

- [ ] **Step 7: Commit**

```bash
git add app/models.py app/services/generation_job_manager.py tests/test_generation_job_manager_ownership.py
git commit -m "feat: add owner_id and quota admission to GenerationJobManager"
```

---

## Task 20: `routes.py` — image job endpoints (permission gate, owner filter, list endpoint)

**Files:**
- Modify: `app/api/routes.py`
- Test: `tests/test_image_job_ownership_api.py`

**Interfaces:**
- Consumes: `require`, `current_user_from_request` (Task 12); `Permission` (Task 2); `QuotaExceededError` (Task 8); `JobsListResponse` (Task 11); `JobManager.create_job(..., owner=...)` (Task 16).
- Produces: shared helpers `_can_view_job`, `_can_cancel_job`, `_require_read_all_if_requested` (module-level in `routes.py`) — reused verbatim by Tasks 21-23. New endpoint `GET /api/v1/jobs`.

This task establishes the **pattern** applied identically to video/audio/generation in Tasks 21-23 — see the "Backward-compatibility mechanics" note at the top of this plan for why `request: Request | None = None` (not a `Depends`-defaulted parameter) carries the data-level owner check on the 3 existing per-kind handlers, while `dependencies=[Depends(require(...))]` on the decorator carries the real permission gate.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_image_job_ownership_api.py
from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import get_settings
from app.main import app


@pytest.fixture
def two_user_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("RUNTIME_DIR", str(tmp_path / "runtime"))
    monkeypatch.setenv("AUTH_MODE", "multi")
    monkeypatch.setenv("AUTH_SECRET", "s" * 32)
    get_settings.cache_clear()
    with TestClient(app, client=("127.0.0.1", 12345)) as client:
        client.post("/api/v1/auth/setup", json={"username": "admin", "password": "adminpass1"})
        client.post("/api/v1/auth/login", json={"username": "admin", "password": "adminpass1"})
        created = client.post("/api/v1/users", json={"username": "bob", "role": "user"}).json()
        yield client, created["user"]["id"], created["temporaryPassword"]
    get_settings.cache_clear()


def test_off_mode_existing_image_job_flow_is_unchanged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RUNTIME_DIR", str(tmp_path / "runtime"))
    monkeypatch.delenv("AUTH_MODE", raising=False)
    get_settings.cache_clear()
    try:
        with TestClient(app) as client:
            response = client.get("/api/v1/jobs")
            assert response.status_code == 200
            assert response.json() == {"jobs": []}
    finally:
        get_settings.cache_clear()


def test_list_jobs_shows_only_own_jobs_by_default(two_user_client) -> None:
    admin_client, bob_id, bob_password = two_user_client
    admin_client.post("/api/v1/auth/logout")
    admin_client.post("/api/v1/auth/login", json={"username": "bob", "password": bob_password})
    admin_client.post(
        "/api/v1/auth/change-password",
        json={"currentPassword": bob_password, "newPassword": "bobnewpass1"},
    )

    response = admin_client.get("/api/v1/jobs")

    assert response.status_code == 200
    assert response.json()["jobs"] == []


def test_list_jobs_with_all_flag_requires_read_all_permission(two_user_client) -> None:
    admin_client, bob_id, bob_password = two_user_client
    admin_client.post("/api/v1/auth/logout")
    admin_client.post("/api/v1/auth/login", json={"username": "bob", "password": bob_password})
    admin_client.post(
        "/api/v1/auth/change-password",
        json={"currentPassword": bob_password, "newPassword": "bobnewpass1"},
    )

    response = admin_client.get("/api/v1/jobs?all=true")

    assert response.status_code == 403


def test_get_job_returns_404_for_someone_elses_job(two_user_client) -> None:
    admin_client, bob_id, _bob_password = two_user_client
    response = admin_client.get("/api/v1/jobs/some-job-id-owned-by-nobody")
    assert response.status_code == 404


def test_create_job_requires_login_in_multi_mode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RUNTIME_DIR", str(tmp_path / "runtime"))
    monkeypatch.setenv("AUTH_MODE", "multi")
    monkeypatch.setenv("AUTH_SECRET", "s" * 32)
    get_settings.cache_clear()
    try:
        with TestClient(app, client=("127.0.0.1", 12345)) as client:
            client.post("/api/v1/auth/setup", json={"username": "admin", "password": "adminpass1"})
            response = client.post("/api/v1/jobs", files={"file": ("a.png", b"not-a-real-png", "image/png")})
            assert response.status_code == 401
    finally:
        get_settings.cache_clear()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv\Scripts\python.exe -m pytest tests/test_image_job_ownership_api.py -q`
Expected: FAIL — `GET /api/v1/jobs` doesn't exist yet (404), and none of the permission gates are wired

- [ ] **Step 3: Write the implementation**

Add imports to `app/api/routes.py`:

```python
from app.api.auth_deps import current_user_from_request, require
from app.exceptions import QuotaExceededError
from app.schemas import AudioJobsListResponse, GenerationJobsListResponse, JobsListResponse, VideoJobsListResponse
from app.services.auth.identity import AuthenticatedUser
from app.services.auth.permissions import Permission
```

Add these module-level helpers (near `job_to_response`, before the route handlers):

```python
def _can_view_job(job: Any, user: AuthenticatedUser) -> bool:
    return Permission.jobs_read_all in user.permissions or job.owner_id == user.id


def _can_cancel_job(job: Any, user: AuthenticatedUser) -> bool:
    return Permission.jobs_cancel_any in user.permissions or job.owner_id == user.id


def _require_read_all_if_requested(all_users: bool, current_user: AuthenticatedUser) -> None:
    if all_users and Permission.jobs_read_all not in current_user.permissions:
        raise HTTPException(status_code=403, detail="No tenés permiso para ver los jobs de todos los usuarios")
```

Modify `create_job`'s decorator and body:

```python
@router.post(
    "/jobs", response_model=CreateJobResponse, status_code=202,
    dependencies=[Depends(require(Permission.jobs_create))],
)
async def create_job(
    request: Request,
    file: UploadFile = File(...),
    model_name: str = Form(default="realesrgan-x4plus"),
    model_id: str | None = Form(default=None),
    device: str | None = Form(default=None),
    scale: int = Form(default=4),
    output_format: str = Form(default="png"),
    jobs: JobManager = Depends(get_job_manager),
    storage: StorageService = Depends(get_storage),
    settings: Settings = Depends(get_settings),
    devices: DevicesService = Depends(get_devices_service),
) -> CreateJobResponse:
    original_name = Path(file.filename or "upload.png").name
    safe_name = sanitize_filename(original_name, default="upload.png")
    token = uuid4().hex
    destination = settings.uploads_path / f"{token}-{safe_name}"
    resolved_device = await resolve_request_device(device, devices, settings)
    current_user = current_user_from_request(request)

    job: UpscaleJob | None = None
    try:
        await storage.save_upload(file, destination)
        job = await jobs.create_job(
            source_path=destination,
            original_filename=original_name,
            model_name=model_name,
            model_id=model_id,
            device=resolved_device,
            scale=scale,
            output_format=output_format,
            job_id=token,
            owner=current_user,
        )
    except QueueFullError as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    except QuotaExceededError as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected error while creating image job")
        raise HTTPException(status_code=500, detail="Failed to process the uploaded image") from exc
    finally:
        if job is None and destination.exists():
            destination.unlink(missing_ok=True)

    return CreateJobResponse(
        job_id=job.id,
        status=job.status,
        status_url=f"/api/v1/jobs/{job.id}",
        download_url=None,
    )
```

Replace `get_job`, `cancel_job`, `download_job` with:

```python
@router.get("/jobs/{job_id}", response_model=JobResponse, dependencies=[Depends(require(Permission.jobs_read_own))])
async def get_job(
    job_id: str, jobs: JobManager = Depends(get_job_manager), request: Request | None = None,
) -> JobResponse:
    job = jobs.get_job(job_id)
    current_user = current_user_from_request(request)
    if not job or (current_user is not None and not _can_view_job(job, current_user)):
        raise HTTPException(status_code=404, detail="Job not found")
    return job_to_response(job)


@router.post(
    "/jobs/{job_id}/cancel", response_model=JobResponse,
    dependencies=[Depends(require(Permission.jobs_cancel_own))],
)
async def cancel_job(
    job_id: str, jobs: JobManager = Depends(get_job_manager), request: Request | None = None,
) -> JobResponse:
    job = jobs.get_job(job_id)
    current_user = current_user_from_request(request)
    if job is None or (current_user is not None and not _can_cancel_job(job, current_user)):
        raise HTTPException(status_code=404, detail="Job not found")
    if not jobs.cancel_job(job_id):
        raise HTTPException(status_code=409, detail="Job already finished")
    return job_to_response(job)


@router.get("/jobs/{job_id}/download", dependencies=[Depends(require(Permission.jobs_read_own))])
async def download_job(
    job_id: str, jobs: JobManager = Depends(get_job_manager), request: Request | None = None,
) -> FileResponse:
    job = jobs.get_job(job_id)
    current_user = current_user_from_request(request)
    if not job or (current_user is not None and not _can_view_job(job, current_user)):
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status != JobStatus.completed or not job.output_path:
        raise HTTPException(status_code=409, detail="Job is not completed yet")
    return FileResponse(path=job.output_path, filename=job.output_path.name, media_type="application/octet-stream")
```

Add the new list endpoint (place it right before `get_job`):

```python
@router.get("/jobs", response_model=JobsListResponse)
async def list_jobs(
    all_users: bool = Query(default=False, alias="all"),
    jobs: JobManager = Depends(get_job_manager),
    current_user: AuthenticatedUser = Depends(require(Permission.jobs_read_own)),
) -> JobsListResponse:
    _require_read_all_if_requested(all_users, current_user)
    visible = [job for job in jobs.jobs.values() if all_users or job.owner_id == current_user.id]
    return JobsListResponse(jobs=[job_to_response(job) for job in visible])
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv\Scripts\python.exe -m pytest tests/test_image_job_ownership_api.py -q`
Expected: PASS (5 tests)

- [ ] **Step 5: Run the 16 legacy direct-call test files that import from `app.api.routes`, then the full suite**

Run: `.venv\Scripts\python.exe -m pytest tests/test_job_status_routes.py tests/test_job_cancel.py tests/test_queue_full.py -q`
Expected: PASS unchanged — these call `get_job(job_id=..., jobs=...)` etc. with no `request` kwarg, which defaults to `None`, so `current_user_from_request(None)` returns `None` and the owner-filter `if ... (current_user is not None and ...)` short-circuits exactly like today's code.

Run: `.venv\Scripts\python.exe -m pytest tests/ -q`
Expected: PASS, zero regressions

- [ ] **Step 6: Commit**

```bash
git add app/api/routes.py tests/test_image_job_ownership_api.py
git commit -m "feat: add permission gate, owner filtering and list endpoint for image jobs"
```

---

## Task 21: `routes.py` — video job endpoints

**Files:**
- Modify: `app/api/routes.py`
- Test: `tests/test_video_job_ownership_api.py`

**Interfaces:** Reuses `_can_view_job`, `_can_cancel_job`, `_require_read_all_if_requested`, `require`, `current_user_from_request`, `QuotaExceededError` from Task 20.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_video_job_ownership_api.py
from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import get_settings
from app.main import app


def test_off_mode_video_jobs_list_is_unchanged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RUNTIME_DIR", str(tmp_path / "runtime"))
    monkeypatch.delenv("AUTH_MODE", raising=False)
    get_settings.cache_clear()
    try:
        with TestClient(app) as client:
            response = client.get("/api/v1/video/jobs")
            assert response.status_code == 200
            assert response.json() == {"jobs": []}
    finally:
        get_settings.cache_clear()


def test_video_job_endpoints_require_login_in_multi_mode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RUNTIME_DIR", str(tmp_path / "runtime"))
    monkeypatch.setenv("AUTH_MODE", "multi")
    monkeypatch.setenv("AUTH_SECRET", "s" * 32)
    get_settings.cache_clear()
    try:
        with TestClient(app, client=("127.0.0.1", 12345)) as client:
            client.post("/api/v1/auth/setup", json={"username": "admin", "password": "adminpass1"})
            response = client.get("/api/v1/video/jobs/some-id")
            assert response.status_code == 401
    finally:
        get_settings.cache_clear()


def test_video_jobs_list_all_requires_read_all_permission(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RUNTIME_DIR", str(tmp_path / "runtime"))
    monkeypatch.setenv("AUTH_MODE", "multi")
    monkeypatch.setenv("AUTH_SECRET", "s" * 32)
    get_settings.cache_clear()
    try:
        with TestClient(app, client=("127.0.0.1", 12345)) as client:
            client.post("/api/v1/auth/setup", json={"username": "admin", "password": "adminpass1"})
            client.post("/api/v1/auth/login", json={"username": "admin", "password": "adminpass1"})
            created = client.post("/api/v1/users", json={"username": "bob", "role": "user"}).json()
            client.post("/api/v1/auth/logout")
            client.post("/api/v1/auth/login", json={"username": "bob", "password": created["temporaryPassword"]})
            client.post(
                "/api/v1/auth/change-password",
                json={"currentPassword": created["temporaryPassword"], "newPassword": "bobnewpass1"},
            )

            response = client.get("/api/v1/video/jobs?all=true")

            assert response.status_code == 403
    finally:
        get_settings.cache_clear()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv\Scripts\python.exe -m pytest tests/test_video_job_ownership_api.py -q`
Expected: FAIL — `GET /api/v1/video/jobs` doesn't exist yet, and existing video endpoints have no permission gate

- [ ] **Step 3: Write the implementation**

Add `dependencies=[Depends(require(Permission.jobs_create))]` to `create_video_job`'s decorator; inside its body, right after `resolved_device = await resolve_request_device(device, devices, settings)`, add `current_user = current_user_from_request(request)`; add `owner=current_user,` to the `video_jobs.create_job(...)` call; add this except clause alongside the existing `except QueueFullError`:

```python
    except QuotaExceededError as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc
```

Replace `get_video_job`, `cancel_video_job`, `download_video_job` with:

```python
@router.get(
    "/video/jobs/{job_id}", response_model=VideoJobResponse,
    dependencies=[Depends(require(Permission.jobs_read_own))],
)
async def get_video_job(
    job_id: str, video_jobs: VideoJobManager = Depends(get_video_job_manager), request: Request | None = None,
) -> VideoJobResponse:
    job = video_jobs.get_job(job_id)
    current_user = current_user_from_request(request)
    if not job or (current_user is not None and not _can_view_job(job, current_user)):
        raise HTTPException(status_code=404, detail="Video job not found")
    return video_job_to_response(job)


@router.post(
    "/video/jobs/{job_id}/cancel", response_model=VideoJobResponse,
    dependencies=[Depends(require(Permission.jobs_cancel_own))],
)
async def cancel_video_job(
    job_id: str, video_jobs: VideoJobManager = Depends(get_video_job_manager), request: Request | None = None,
) -> VideoJobResponse:
    job = video_jobs.get_job(job_id)
    current_user = current_user_from_request(request)
    if job is None or (current_user is not None and not _can_cancel_job(job, current_user)):
        raise HTTPException(status_code=404, detail="Video job not found")
    if not video_jobs.cancel_job(job_id):
        raise HTTPException(status_code=409, detail="Job already finished")
    return video_job_to_response(job)


@router.get("/video/jobs/{job_id}/download", dependencies=[Depends(require(Permission.jobs_read_own))])
async def download_video_job(
    job_id: str, video_jobs: VideoJobManager = Depends(get_video_job_manager), request: Request | None = None,
) -> FileResponse:
    job = video_jobs.get_job(job_id)
    current_user = current_user_from_request(request)
    if not job or (current_user is not None and not _can_view_job(job, current_user)):
        raise HTTPException(status_code=404, detail="Video job not found")
    if job.status != JobStatus.completed or not job.output_path:
        raise HTTPException(status_code=409, detail="Video job is not completed yet")
    return FileResponse(path=job.output_path, filename=job.output_path.name, media_type="application/octet-stream")
```

Add the new list endpoint (place it right before `get_video_job`):

```python
@router.get("/video/jobs", response_model=VideoJobsListResponse)
async def list_video_jobs(
    all_users: bool = Query(default=False, alias="all"),
    video_jobs: VideoJobManager = Depends(get_video_job_manager),
    current_user: AuthenticatedUser = Depends(require(Permission.jobs_read_own)),
) -> VideoJobsListResponse:
    _require_read_all_if_requested(all_users, current_user)
    visible = [job for job in video_jobs.jobs.values() if all_users or job.owner_id == current_user.id]
    return VideoJobsListResponse(jobs=[video_job_to_response(job) for job in visible])
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv\Scripts\python.exe -m pytest tests/test_video_job_ownership_api.py -q`
Expected: PASS (3 tests)

- [ ] **Step 5: Run the legacy direct-call test files touching video routes, then the full suite**

Run: `.venv\Scripts\python.exe -m pytest tests/test_job_status_routes.py tests/test_job_cancel.py tests/test_target_fps.py tests/test_interp_engine.py tests/test_fps_boost_api.py tests/test_upload_token_and_subtitles.py tests/test_video_backend_dispatch.py -q`
Expected: PASS unchanged

Run: `.venv\Scripts\python.exe -m pytest tests/ -q`
Expected: PASS, zero regressions

- [ ] **Step 6: Commit**

```bash
git add app/api/routes.py tests/test_video_job_ownership_api.py
git commit -m "feat: add permission gate, owner filtering and list endpoint for video jobs"
```

---

## Task 22: `routes.py` — audio job endpoints

**Files:**
- Modify: `app/api/routes.py`
- Test: `tests/test_audio_job_ownership_api.py`

**Interfaces:** Reuses the same shared helpers as Tasks 20-21.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_audio_job_ownership_api.py
from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import get_settings
from app.main import app


def test_off_mode_audio_jobs_list_is_unchanged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RUNTIME_DIR", str(tmp_path / "runtime"))
    monkeypatch.delenv("AUTH_MODE", raising=False)
    get_settings.cache_clear()
    try:
        with TestClient(app) as client:
            response = client.get("/api/v1/audio/jobs")
            assert response.status_code == 200
            assert response.json() == {"jobs": []}
    finally:
        get_settings.cache_clear()


def test_audio_job_endpoints_require_login_in_multi_mode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RUNTIME_DIR", str(tmp_path / "runtime"))
    monkeypatch.setenv("AUTH_MODE", "multi")
    monkeypatch.setenv("AUTH_SECRET", "s" * 32)
    get_settings.cache_clear()
    try:
        with TestClient(app, client=("127.0.0.1", 12345)) as client:
            client.post("/api/v1/auth/setup", json={"username": "admin", "password": "adminpass1"})
            response = client.get("/api/v1/audio/jobs/some-id")
            assert response.status_code == 401
    finally:
        get_settings.cache_clear()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv\Scripts\python.exe -m pytest tests/test_audio_job_ownership_api.py -q`
Expected: FAIL — `GET /api/v1/audio/jobs` doesn't exist yet

- [ ] **Step 3: Write the implementation**

Add `dependencies=[Depends(require(Permission.jobs_create))]` to `create_audio_job`'s decorator; inside its body, right after `destination = settings.uploads_path / f"{token}-{safe_name}"`, add `current_user = current_user_from_request(request)`; add `owner=current_user,` to the `audio_jobs.create_job(...)` call; add the `QuotaExceededError` except clause alongside the existing `except QueueFullError`.

Replace `get_audio_job`, `cancel_audio_job`, `download_audio_job` with:

```python
@router.get(
    "/audio/jobs/{job_id}", response_model=AudioJobResponse,
    dependencies=[Depends(require(Permission.jobs_read_own))],
)
async def get_audio_job(
    job_id: str, audio_jobs: AudioJobManager = Depends(get_audio_job_manager), request: Request | None = None,
) -> AudioJobResponse:
    job = audio_jobs.get_job(job_id)
    current_user = current_user_from_request(request)
    if not job or (current_user is not None and not _can_view_job(job, current_user)):
        raise HTTPException(status_code=404, detail="Audio job not found")
    return audio_job_to_response(job)


@router.post(
    "/audio/jobs/{job_id}/cancel", response_model=AudioJobResponse,
    dependencies=[Depends(require(Permission.jobs_cancel_own))],
)
async def cancel_audio_job(
    job_id: str, audio_jobs: AudioJobManager = Depends(get_audio_job_manager), request: Request | None = None,
) -> AudioJobResponse:
    job = audio_jobs.get_job(job_id)
    current_user = current_user_from_request(request)
    if job is None or (current_user is not None and not _can_cancel_job(job, current_user)):
        raise HTTPException(status_code=404, detail="Audio job not found")
    if not audio_jobs.cancel_job(job_id):
        raise HTTPException(status_code=409, detail="Job already finished")
    return audio_job_to_response(job)


@router.get("/audio/jobs/{job_id}/download", dependencies=[Depends(require(Permission.jobs_read_own))])
async def download_audio_job(
    job_id: str, audio_jobs: AudioJobManager = Depends(get_audio_job_manager), request: Request | None = None,
) -> FileResponse:
    job = audio_jobs.get_job(job_id)
    current_user = current_user_from_request(request)
    if not job or (current_user is not None and not _can_view_job(job, current_user)):
        raise HTTPException(status_code=404, detail="Audio job not found")
    if job.status != JobStatus.completed or not job.output_path:
        raise HTTPException(status_code=409, detail="Audio job is not completed yet")
    return FileResponse(path=job.output_path, filename=job.output_path.name, media_type="application/octet-stream")
```

Add the new list endpoint (place it right before `get_audio_job`):

```python
@router.get("/audio/jobs", response_model=AudioJobsListResponse)
async def list_audio_jobs(
    all_users: bool = Query(default=False, alias="all"),
    audio_jobs: AudioJobManager = Depends(get_audio_job_manager),
    current_user: AuthenticatedUser = Depends(require(Permission.jobs_read_own)),
) -> AudioJobsListResponse:
    _require_read_all_if_requested(all_users, current_user)
    visible = [job for job in audio_jobs.jobs.values() if all_users or job.owner_id == current_user.id]
    return AudioJobsListResponse(jobs=[audio_job_to_response(job) for job in visible])
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv\Scripts\python.exe -m pytest tests/test_audio_job_ownership_api.py -q`
Expected: PASS (2 tests)

- [ ] **Step 5: Run the legacy direct-call test files touching audio routes, then the full suite**

Run: `.venv\Scripts\python.exe -m pytest tests/test_job_cancel.py tests/test_audio_enhance_api.py tests/test_audio_pipeline.py -q`
Expected: PASS unchanged

Run: `.venv\Scripts\python.exe -m pytest tests/ -q`
Expected: PASS, zero regressions

- [ ] **Step 6: Commit**

```bash
git add app/api/routes.py tests/test_audio_job_ownership_api.py
git commit -m "feat: add permission gate, owner filtering and list endpoint for audio jobs"
```

---

## Task 23: `routes.py` — generation job endpoints

**Files:**
- Modify: `app/api/routes.py`
- Test: `tests/test_generation_job_ownership_api.py`

**Interfaces:** Reuses the same shared helpers as Tasks 20-22. `create_generation_job` currently has **no** `request: Request` parameter at all — this task adds `request: Request | None = None` as a new trailing optional parameter (safe: existing calls in `tests/test_generation_api.py` never pass `request`, so it stays `None` for them, exactly reproducing today's unfiltered behavior).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_generation_job_ownership_api.py
from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import get_settings
from app.main import app


def test_off_mode_generation_jobs_list_is_unchanged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RUNTIME_DIR", str(tmp_path / "runtime"))
    monkeypatch.delenv("AUTH_MODE", raising=False)
    get_settings.cache_clear()
    try:
        with TestClient(app) as client:
            response = client.get("/api/v1/generation/jobs")
            assert response.status_code == 200
            assert response.json() == {"jobs": []}
    finally:
        get_settings.cache_clear()


def test_generation_job_endpoints_require_login_in_multi_mode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RUNTIME_DIR", str(tmp_path / "runtime"))
    monkeypatch.setenv("AUTH_MODE", "multi")
    monkeypatch.setenv("AUTH_SECRET", "s" * 32)
    get_settings.cache_clear()
    try:
        with TestClient(app, client=("127.0.0.1", 12345)) as client:
            client.post("/api/v1/auth/setup", json={"username": "admin", "password": "adminpass1"})
            response = client.get("/api/v1/generation/jobs/some-id")
            assert response.status_code == 401
    finally:
        get_settings.cache_clear()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv\Scripts\python.exe -m pytest tests/test_generation_job_ownership_api.py -q`
Expected: FAIL — `GET /api/v1/generation/jobs` doesn't exist yet

- [ ] **Step 3: Write the implementation**

Modify `create_generation_job`'s decorator and signature:

```python
@router.post(
    "/generation/jobs", response_model=GenerationJobResponse, status_code=201,
    dependencies=[Depends(require(Permission.jobs_create))],
)
async def create_generation_job(
    payload: CreateGenerationJobRequest,
    generation_jobs: GenerationJobManager = Depends(get_generation_job_manager),
    request: Request | None = None,
) -> GenerationJobResponse:
    current_user = current_user_from_request(request)
    try:
        job = await generation_jobs.create_job(
            prompt=payload.prompt, negative_prompt=payload.negative_prompt, model_id=payload.model_id,
            steps=payload.steps, guidance=payload.guidance, width=payload.width, height=payload.height,
            seed=payload.seed, device=payload.device, auto_upscale=payload.auto_upscale,
            upscale_model_name=payload.upscale_model_name, upscale_scale=payload.upscale_scale,
            upscale_model_id=payload.upscale_model_id, owner=current_user,
        )
    except QueueFullError as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    except QuotaExceededError as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected error while creating generation job")
        raise HTTPException(status_code=500, detail="Failed to create the generation job") from exc
    return generation_job_to_response(job)
```

Replace `get_generation_job`, `cancel_generation_job`, `download_generation_job` with:

```python
@router.get(
    "/generation/jobs/{job_id}", response_model=GenerationJobResponse,
    dependencies=[Depends(require(Permission.jobs_read_own))],
)
async def get_generation_job(
    job_id: str, generation_jobs: GenerationJobManager = Depends(get_generation_job_manager),
    request: Request | None = None,
) -> GenerationJobResponse:
    job = generation_jobs.get_job(job_id)
    current_user = current_user_from_request(request)
    if not job or (current_user is not None and not _can_view_job(job, current_user)):
        raise HTTPException(status_code=404, detail="Generation job not found")
    return generation_job_to_response(job)


@router.post(
    "/generation/jobs/{job_id}/cancel", response_model=GenerationJobResponse,
    dependencies=[Depends(require(Permission.jobs_cancel_own))],
)
async def cancel_generation_job(
    job_id: str, generation_jobs: GenerationJobManager = Depends(get_generation_job_manager),
    request: Request | None = None,
) -> GenerationJobResponse:
    job = generation_jobs.get_job(job_id)
    current_user = current_user_from_request(request)
    if job is None or (current_user is not None and not _can_cancel_job(job, current_user)):
        raise HTTPException(status_code=404, detail="Generation job not found")
    if not generation_jobs.cancel_job(job_id):
        raise HTTPException(status_code=409, detail="Job already finished")
    return generation_job_to_response(job)


@router.get("/generation/jobs/{job_id}/download", dependencies=[Depends(require(Permission.jobs_read_own))])
async def download_generation_job(
    job_id: str, generation_jobs: GenerationJobManager = Depends(get_generation_job_manager),
    request: Request | None = None,
) -> FileResponse:
    job = generation_jobs.get_job(job_id)
    current_user = current_user_from_request(request)
    if not job or (current_user is not None and not _can_view_job(job, current_user)):
        raise HTTPException(status_code=404, detail="Generation job not found")
    if job.status != JobStatus.completed or not job.output_path:
        raise HTTPException(status_code=409, detail="Generation job is not completed yet")
    return FileResponse(path=job.output_path, filename=job.output_path.name, media_type="image/png")
```

Add the new list endpoint (place it right before `get_generation_job`):

```python
@router.get("/generation/jobs", response_model=GenerationJobsListResponse)
async def list_generation_jobs(
    all_users: bool = Query(default=False, alias="all"),
    generation_jobs: GenerationJobManager = Depends(get_generation_job_manager),
    current_user: AuthenticatedUser = Depends(require(Permission.jobs_read_own)),
) -> GenerationJobsListResponse:
    _require_read_all_if_requested(all_users, current_user)
    visible = [job for job in generation_jobs.jobs.values() if all_users or job.owner_id == current_user.id]
    return GenerationJobsListResponse(jobs=[generation_job_to_response(job) for job in visible])
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv\Scripts\python.exe -m pytest tests/test_generation_job_ownership_api.py -q`
Expected: PASS (2 tests)

- [ ] **Step 5: Run `tests/test_generation_api.py` (the file with direct calls omitting `request` entirely) plus the full suite**

Run: `.venv\Scripts\python.exe -m pytest tests/test_generation_api.py -q`
Expected: PASS unchanged — `create_generation_job(payload=..., generation_jobs=...)` binds the new `request` parameter to its default `None`, so `current_user_from_request(None)` is `None` and `owner=None` reproduces today's behavior exactly

Run: `.venv\Scripts\python.exe -m pytest tests/ -q`
Expected: PASS, zero regressions

- [ ] **Step 6: Commit**

```bash
git add app/api/routes.py tests/test_generation_job_ownership_api.py
git commit -m "feat: add permission gate, owner filtering and list endpoint for generation jobs"
```

---

## Task 24: Backend non-regression checkpoint

**Files:** none (verification-only task)

- [ ] **Step 1: Run the complete backend suite with default settings (AUTH_MODE unset → off)**

Run: `.venv\Scripts\python.exe -m pytest tests/ -q`
Expected: PASS — the original 1207 tests plus every test added in Tasks 1-23, zero failures, zero skips introduced by this plan.

- [ ] **Step 2: Spot-check the exact pre-existing count didn't shrink** (a shrinking count would mean a collection error swallowed some existing file)

Run: `.venv\Scripts\python.exe -m pytest tests/ -q --collect-only | Select-String "tests collected"`
Expected: collected count ≥ 1207 + (sum of new tests added across Tasks 1-23)

- [ ] **Step 3: Run with `AUTH_MODE=multi` explicitly set, to confirm the new suite is exercised both ways and nothing in the default suite accidentally depends on multi-mode**

```powershell
$env:AUTH_MODE = "multi"; $env:AUTH_SECRET = ("s" * 32); .venv\Scripts\python.exe -m pytest tests/test_auth_api.py tests/test_users_api.py tests/test_image_job_ownership_api.py tests/test_video_job_ownership_api.py tests/test_audio_job_ownership_api.py tests/test_generation_job_ownership_api.py -q; Remove-Item Env:\AUTH_MODE; Remove-Item Env:\AUTH_SECRET
```
Expected: PASS (these test files already set their own env per-test via `monkeypatch`, so the shell-level env vars are redundant safety, not required — this step exists to catch any test that accidentally reads process-level env directly instead of using the `monkeypatch`/fixture pattern)

- [ ] **Step 4: No commit** — this is a checkpoint, not a code change. If any test fails, stop and fix the offending task before proceeding to the frontend tasks (25-32).

---

## Task 25: Frontend types + `auth.ts`/`users.ts` services + `apiPatchJson`

**Files:**
- Modify: `frontend/src/lib/apiTypes.ts`
- Modify: `frontend/src/lib/api.ts`
- Create: `frontend/src/services/auth.ts`
- Create: `frontend/src/services/users.ts`
- Test: `frontend/src/services/auth.test.ts`, `frontend/src/services/users.test.ts`

**Interfaces:**
- Produces (types): `MeResponse`, `QuotaStatus`, `UserSummary`, `UsersListResponse`, `CreateUserResponse`, `UpdateUserResponse`, `OwnedJobSummary`, `UserJobsResponse`; adds `ownerId: string | null` to `JobResponse`, `VideoJobResponse`, `AudioJob`, `GenerationJob`.
- Produces (api.ts): `apiPatchJson<T>(path, body): Promise<T>`.
- Produces (services): `login`, `logout`, `logoutAll`, `getMe`, `changePassword`, `setup` (auth.ts); `listUsers`, `createUser`, `updateUser`, `getUserJobs` (users.ts) — consumed by Tasks 26, 28, 30.

- [ ] **Step 1: Write the failing tests**

```ts
// frontend/src/services/auth.test.ts
import { afterEach, describe, expect, it, vi } from "vitest";
import { changePassword, getMe, login, logout, logoutAll, setup } from "./auth";

vi.mock("../lib/api", () => ({ apiGet: vi.fn(), apiPostJson: vi.fn() }));

import { apiGet, apiPostJson } from "../lib/api";

afterEach(() => {
  vi.mocked(apiGet).mockReset();
  vi.mocked(apiPostJson).mockReset();
});

describe("auth service", () => {
  it("login posts username/password to /auth/login", async () => {
    vi.mocked(apiPostJson).mockResolvedValue({ ok: true });
    await login("alice", "pw");
    expect(apiPostJson).toHaveBeenCalledWith("/auth/login", { username: "alice", password: "pw" });
  });

  it("logout posts to /auth/logout", async () => {
    vi.mocked(apiPostJson).mockResolvedValue({ ok: true });
    await logout();
    expect(apiPostJson).toHaveBeenCalledWith("/auth/logout", {});
  });

  it("logoutAll posts to /auth/logout-all", async () => {
    vi.mocked(apiPostJson).mockResolvedValue({ ok: true });
    await logoutAll();
    expect(apiPostJson).toHaveBeenCalledWith("/auth/logout-all", {});
  });

  it("getMe fetches /auth/me", async () => {
    vi.mocked(apiGet).mockResolvedValue({ username: "alice" });
    await getMe();
    expect(apiGet).toHaveBeenCalledWith("/auth/me");
  });

  it("changePassword posts camelCase body", async () => {
    vi.mocked(apiPostJson).mockResolvedValue({ ok: true });
    await changePassword("old", "new");
    expect(apiPostJson).toHaveBeenCalledWith("/auth/change-password", {
      currentPassword: "old",
      newPassword: "new",
    });
  });

  it("setup posts username/password to /auth/setup", async () => {
    vi.mocked(apiPostJson).mockResolvedValue({ ok: true });
    await setup("admin", "pw12345678");
    expect(apiPostJson).toHaveBeenCalledWith("/auth/setup", { username: "admin", password: "pw12345678" });
  });
});
```

```ts
// frontend/src/services/users.test.ts
import { afterEach, describe, expect, it, vi } from "vitest";
import { createUser, getUserJobs, listUsers, updateUser } from "./users";

vi.mock("../lib/api", () => ({ apiGet: vi.fn(), apiPostJson: vi.fn(), apiPatchJson: vi.fn() }));

import { apiGet, apiPatchJson, apiPostJson } from "../lib/api";

afterEach(() => {
  vi.mocked(apiGet).mockReset();
  vi.mocked(apiPostJson).mockReset();
  vi.mocked(apiPatchJson).mockReset();
});

describe("users service", () => {
  it("listUsers fetches /users", async () => {
    vi.mocked(apiGet).mockResolvedValue({ users: [] });
    await listUsers();
    expect(apiGet).toHaveBeenCalledWith("/users");
  });

  it("createUser posts to /users", async () => {
    vi.mocked(apiPostJson).mockResolvedValue({ user: {}, temporaryPassword: "x" });
    await createUser({ username: "bob", role: "user" });
    expect(apiPostJson).toHaveBeenCalledWith("/users", { username: "bob", role: "user" });
  });

  it("updateUser patches /users/{id}", async () => {
    vi.mocked(apiPatchJson).mockResolvedValue({ user: {}, temporaryPassword: null });
    await updateUser("u1", { disabled: true });
    expect(apiPatchJson).toHaveBeenCalledWith("/users/u1", { disabled: true });
  });

  it("getUserJobs fetches /users/{id}/jobs", async () => {
    vi.mocked(apiGet).mockResolvedValue({ jobs: [] });
    await getUserJobs("u1");
    expect(apiGet).toHaveBeenCalledWith("/users/u1/jobs");
  });
});
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd frontend && npx vitest run src/services/auth.test.ts src/services/users.test.ts`
Expected: FAIL — `./auth` and `./users` don't exist yet, and `apiPatchJson` doesn't exist on `../lib/api`

- [ ] **Step 3: Write the implementation**

Add to `frontend/src/lib/apiTypes.ts` (append at the end):

```ts
export interface QuotaStatus {
  maxConcurrent: number;
  maxQueued: number;
  maxJobsPerDay: number;
  maxGpuSecondsPerDay: number;
  usedJobsToday: number;
  usedGpuSecondsToday: number;
}

export interface MeResponse {
  userId: string | null;
  username: string;
  role: string;
  permissions: string[];
  mustChangePassword: boolean;
  authMode: "off" | "multi";
  quota: QuotaStatus;
}

export interface UserSummary {
  id: string;
  username: string;
  role: string;
  disabled: boolean;
  mustChangePassword: boolean;
  quotaOverrides: Record<string, number>;
  createdAt: string;
  usedJobsToday: number;
  usedGpuSecondsToday: number;
}

export interface UsersListResponse {
  users: UserSummary[];
}

export interface CreateUserResponse {
  user: UserSummary;
  temporaryPassword: string;
}

export interface UpdateUserResponse {
  user: UserSummary;
  temporaryPassword: string | null;
}

export interface OwnedJobSummary {
  id: string;
  kind: string;
  status: JobStatus;
  originalFilename: string | null;
  createdAt: string;
  finishedAt: string | null;
}

export interface UserJobsResponse {
  jobs: OwnedJobSummary[];
}
```

Add `ownerId: string | null;` to `JobResponse`, `VideoJobResponse`, `AudioJob`, `GenerationJob` interfaces in `apiTypes.ts` (right after `error: string | null;` in each).

Add to `frontend/src/lib/api.ts`, right after `apiPostJson`:

```ts
export async function apiPatchJson<T>(path: string, body: unknown): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!response.ok) {
    throw new ApiError(response.status, await extractErrorMessage(response));
  }
  return (await response.json()) as T;
}
```

Create `frontend/src/services/auth.ts`:

```ts
import { apiGet, apiPostJson } from "../lib/api";
import type { MeResponse } from "../lib/apiTypes";

export function login(username: string, password: string): Promise<{ ok: boolean }> {
  return apiPostJson<{ ok: boolean }>("/auth/login", { username, password });
}

export function logout(): Promise<{ ok: boolean }> {
  return apiPostJson<{ ok: boolean }>("/auth/logout", {});
}

export function logoutAll(): Promise<{ ok: boolean }> {
  return apiPostJson<{ ok: boolean }>("/auth/logout-all", {});
}

export function getMe(): Promise<MeResponse> {
  return apiGet<MeResponse>("/auth/me");
}

export function changePassword(currentPassword: string, newPassword: string): Promise<{ ok: boolean }> {
  return apiPostJson<{ ok: boolean }>("/auth/change-password", { currentPassword, newPassword });
}

export function setup(username: string, password: string): Promise<{ ok: boolean }> {
  return apiPostJson<{ ok: boolean }>("/auth/setup", { username, password });
}
```

Create `frontend/src/services/users.ts`:

```ts
import { apiGet, apiPatchJson, apiPostJson } from "../lib/api";
import type {
  CreateUserResponse,
  UpdateUserResponse,
  UserJobsResponse,
  UsersListResponse,
} from "../lib/apiTypes";

export interface CreateUserParams {
  username: string;
  role: string;
}

export interface UpdateUserParams {
  role?: string;
  disabled?: boolean;
  quotaOverrides?: Record<string, number>;
  resetPassword?: boolean;
}

export function listUsers(): Promise<UsersListResponse> {
  return apiGet<UsersListResponse>("/users");
}

export function createUser(params: CreateUserParams): Promise<CreateUserResponse> {
  return apiPostJson<CreateUserResponse>("/users", params);
}

export function updateUser(userId: string, params: UpdateUserParams): Promise<UpdateUserResponse> {
  return apiPatchJson<UpdateUserResponse>(`/users/${userId}`, params);
}

export function getUserJobs(userId: string): Promise<UserJobsResponse> {
  return apiGet<UserJobsResponse>(`/users/${userId}/jobs`);
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd frontend && npx vitest run src/services/auth.test.ts src/services/users.test.ts`
Expected: PASS (10 tests)

- [ ] **Step 5: Run `tsc` and the full existing frontend suite to confirm no regression**

Run: `cd frontend && npx tsc --noEmit && npx vitest run`
Expected: PASS, 447 existing tests + 10 new, zero type errors

- [ ] **Step 6: Commit**

```bash
git add frontend/src/lib/apiTypes.ts frontend/src/lib/api.ts frontend/src/services/auth.ts frontend/src/services/users.ts frontend/src/services/auth.test.ts frontend/src/services/users.test.ts
git commit -m "feat: add auth/users API types and service wrappers"
```

---

## Task 26: `useAuth.tsx` — AuthProvider + useAuth hook

**Files:**
- Create: `frontend/src/hooks/useAuth.tsx`
- Test: `frontend/src/hooks/useAuth.test.tsx`

**Interfaces:**
- Consumes: `getMe` (Task 25); `ApiError` (`frontend/src/lib/api.ts`, existing).
- Produces: `AuthProvider` (component), `useAuth()` returning `{ me, isLoading, needsSetup, isError, hasPermission, refetch }` — consumed by Tasks 27, 28, 29, 31.

- [ ] **Step 1: Write the failing tests**

```tsx
// frontend/src/hooks/useAuth.test.tsx
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, waitFor } from "@testing-library/react";
import type { ReactNode } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";
import * as authService from "../services/auth";
import { ApiError } from "../lib/api";
import type { MeResponse } from "../lib/apiTypes";
import { AuthProvider, useAuth } from "./useAuth";

vi.mock("../services/auth", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../services/auth")>();
  return { ...actual, getMe: vi.fn() };
});

function Wrapper({ children }: { children: ReactNode }) {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return (
    <QueryClientProvider client={queryClient}>
      <AuthProvider>{children}</AuthProvider>
    </QueryClientProvider>
  );
}

function Probe() {
  const { me, isLoading, needsSetup, hasPermission } = useAuth();
  if (isLoading) return <span>loading</span>;
  if (needsSetup) return <span>needs-setup</span>;
  return (
    <span>
      {me?.username ?? "none"}:{hasPermission("users:manage") ? "yes" : "no"}
    </span>
  );
}

afterEach(() => {
  vi.mocked(authService.getMe).mockReset();
});

describe("useAuth", () => {
  it("exposes the resolved user and permission check after GET /auth/me succeeds", async () => {
    const meResponse: MeResponse = {
      userId: "u1", username: "alice", role: "admin", permissions: ["users:manage"],
      mustChangePassword: false, authMode: "multi",
      quota: { maxConcurrent: 0, maxQueued: 0, maxJobsPerDay: 0, maxGpuSecondsPerDay: 0, usedJobsToday: 0, usedGpuSecondsToday: 0 },
    };
    vi.mocked(authService.getMe).mockResolvedValue(meResponse);

    render(<Probe />, { wrapper: Wrapper });

    await waitFor(() => expect(screen.getByText("alice:yes")).toBeInTheDocument());
  });

  it("exposes needsSetup when GET /auth/me fails with setup_required", async () => {
    vi.mocked(authService.getMe).mockRejectedValue(new ApiError(401, "setup_required"));

    render(<Probe />, { wrapper: Wrapper });

    await waitFor(() => expect(screen.getByText("needs-setup")).toBeInTheDocument());
  });

  it("throws when useAuth is used outside AuthProvider", () => {
    function Bare() {
      useAuth();
      return null;
    }
    expect(() => render(<Bare />)).toThrow("useAuth must be used within an AuthProvider");
  });
});
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd frontend && npx vitest run src/hooks/useAuth.test.tsx`
Expected: FAIL — `./useAuth` doesn't exist yet

- [ ] **Step 3: Write the implementation**

```tsx
// frontend/src/hooks/useAuth.tsx
import { useQuery } from "@tanstack/react-query";
import { createContext, useContext, type ReactNode } from "react";
import { ApiError } from "../lib/api";
import type { MeResponse } from "../lib/apiTypes";
import { getMe } from "../services/auth";

interface AuthContextValue {
  me: MeResponse | undefined;
  isLoading: boolean;
  isError: boolean;
  needsSetup: boolean;
  hasPermission: (permission: string) => boolean;
  refetch: () => void;
}

const AuthContext = createContext<AuthContextValue | null>(null);

function isSetupRequiredError(error: unknown): boolean {
  return error instanceof ApiError && error.message === "setup_required";
}

export function AuthProvider({ children }: { children: ReactNode }) {
  const query = useQuery({ queryKey: ["me"], queryFn: getMe, retry: false });

  function hasPermission(permission: string): boolean {
    return query.data?.permissions.includes(permission) ?? false;
  }

  const value: AuthContextValue = {
    me: query.data,
    isLoading: query.isLoading,
    isError: query.isError,
    needsSetup: isSetupRequiredError(query.error),
    hasPermission,
    refetch: () => void query.refetch(),
  };

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth(): AuthContextValue {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error("useAuth must be used within an AuthProvider");
  }
  return context;
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd frontend && npx vitest run src/hooks/useAuth.test.tsx`
Expected: PASS (3 tests)

- [ ] **Step 5: Run `tsc` and the full existing suite to confirm no regression**

Run: `cd frontend && npx tsc --noEmit && npx vitest run`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add frontend/src/hooks/useAuth.tsx frontend/src/hooks/useAuth.test.tsx
git commit -m "feat: add AuthProvider/useAuth hook for GET /auth/me gating"
```

---

## Task 27: `LoginPage` + `SetupPage`

**Files:**
- Create: `frontend/src/pages/LoginPage.tsx`
- Create: `frontend/src/pages/SetupPage.tsx`
- Test: `frontend/src/pages/LoginPage.test.tsx`, `frontend/src/pages/SetupPage.test.tsx`

**Interfaces:**
- Consumes: `login`, `setup` (Task 25); `ApiError` (`lib/api.ts`).
- Produces: `LoginPage`, `SetupPage` components — consumed by Task 28.

- [ ] **Step 1: Write the failing tests**

```tsx
// frontend/src/pages/LoginPage.test.tsx
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import type { ReactNode } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { ApiError } from "../lib/api";
import * as authService from "../services/auth";
import { LoginPage } from "./LoginPage";

vi.mock("../services/auth", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../services/auth")>();
  return { ...actual, login: vi.fn() };
});

function renderPage() {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  function Wrapper({ children }: { children: ReactNode }) {
    return <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>;
  }
  return render(<LoginPage />, { wrapper: Wrapper });
}

afterEach(() => {
  vi.mocked(authService.login).mockReset();
});

describe("LoginPage", () => {
  it("submits username and password", async () => {
    vi.mocked(authService.login).mockResolvedValue({ ok: true });
    renderPage();

    fireEvent.change(screen.getByLabelText(/usuario/i), { target: { value: "alice" } });
    fireEvent.change(screen.getByLabelText(/contraseña/i), { target: { value: "hunter22" } });
    fireEvent.click(screen.getByRole("button", { name: /ingresar/i }));

    await waitFor(() => expect(authService.login).toHaveBeenCalledWith("alice", "hunter22"));
  });

  it("shows an error message when login fails", async () => {
    vi.mocked(authService.login).mockRejectedValue(new ApiError(401, "Usuario o contraseña incorrectos"));
    renderPage();

    fireEvent.change(screen.getByLabelText(/usuario/i), { target: { value: "alice" } });
    fireEvent.change(screen.getByLabelText(/contraseña/i), { target: { value: "wrong" } });
    fireEvent.click(screen.getByRole("button", { name: /ingresar/i }));

    await waitFor(() => expect(screen.getByRole("alert")).toHaveTextContent("Usuario o contraseña incorrectos"));
  });
});
```

```tsx
// frontend/src/pages/SetupPage.test.tsx
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import type { ReactNode } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";
import * as authService from "../services/auth";
import { SetupPage } from "./SetupPage";

vi.mock("../services/auth", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../services/auth")>();
  return { ...actual, setup: vi.fn() };
});

function renderPage() {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  function Wrapper({ children }: { children: ReactNode }) {
    return <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>;
  }
  return render(<SetupPage />, { wrapper: Wrapper });
}

afterEach(() => {
  vi.mocked(authService.setup).mockReset();
});

describe("SetupPage", () => {
  it("submits username and password to create the first admin", async () => {
    vi.mocked(authService.setup).mockResolvedValue({ ok: true });
    renderPage();

    fireEvent.change(screen.getByLabelText(/usuario/i), { target: { value: "admin" } });
    fireEvent.change(screen.getByLabelText(/contraseña/i), { target: { value: "adminpass1" } });
    fireEvent.click(screen.getByRole("button", { name: /crear/i }));

    await waitFor(() => expect(authService.setup).toHaveBeenCalledWith("admin", "adminpass1"));
  });
});
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd frontend && npx vitest run src/pages/LoginPage.test.tsx src/pages/SetupPage.test.tsx`
Expected: FAIL — neither page component exists yet

- [ ] **Step 3: Write the implementation**

```tsx
// frontend/src/pages/LoginPage.tsx
import { useQueryClient } from "@tanstack/react-query";
import { useState, type FormEvent } from "react";
import { ApiError } from "../lib/api";
import { login } from "../services/auth";

export function LoginPage() {
  const queryClient = useQueryClient();
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  async function handleSubmit(event: FormEvent) {
    event.preventDefault();
    setSubmitting(true);
    setError(null);
    try {
      await login(username, password);
      await queryClient.invalidateQueries({ queryKey: ["me"] });
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "No se pudo iniciar sesión");
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div className="flex h-screen items-center justify-center bg-bg">
      <form onSubmit={handleSubmit} className="flex w-full max-w-sm flex-col gap-4 rounded border border-border bg-surface p-6">
        <h1 className="font-heading text-lg font-semibold text-text">Upflow</h1>
        <label className="flex flex-col gap-1 text-xs text-text-dim">
          Usuario
          <input
            required
            value={username}
            onChange={(event) => setUsername(event.target.value)}
            className="rounded border border-border bg-surface-2 px-2 py-1 text-sm text-text"
          />
        </label>
        <label className="flex flex-col gap-1 text-xs text-text-dim">
          Contraseña
          <input
            type="password"
            required
            value={password}
            onChange={(event) => setPassword(event.target.value)}
            className="rounded border border-border bg-surface-2 px-2 py-1 text-sm text-text"
          />
        </label>
        {error && <p role="alert" className="text-xs text-danger">{error}</p>}
        <button
          type="submit"
          disabled={submitting}
          className="rounded bg-accent px-3 py-1.5 text-sm font-medium text-bg disabled:opacity-50"
        >
          Ingresar
        </button>
      </form>
    </div>
  );
}
```

```tsx
// frontend/src/pages/SetupPage.tsx
import { useState, type FormEvent } from "react";
import { ApiError } from "../lib/api";
import { setup } from "../services/auth";

export function SetupPage() {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [done, setDone] = useState(false);
  const [submitting, setSubmitting] = useState(false);

  async function handleSubmit(event: FormEvent) {
    event.preventDefault();
    setSubmitting(true);
    setError(null);
    try {
      await setup(username, password);
      setDone(true);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "No se pudo crear el administrador");
    } finally {
      setSubmitting(false);
    }
  }

  if (done) {
    return (
      <div className="flex h-screen items-center justify-center bg-bg">
        <p className="text-sm text-text">Administrador creado. Recargá la página para iniciar sesión.</p>
      </div>
    );
  }

  return (
    <div className="flex h-screen items-center justify-center bg-bg">
      <form onSubmit={handleSubmit} className="flex w-full max-w-sm flex-col gap-4 rounded border border-border bg-surface p-6">
        <h1 className="font-heading text-lg font-semibold text-text">Configuración inicial</h1>
        <p className="text-xs text-text-dim">Creá la cuenta de administrador de Upflow.</p>
        <label className="flex flex-col gap-1 text-xs text-text-dim">
          Usuario
          <input
            required
            minLength={3}
            value={username}
            onChange={(event) => setUsername(event.target.value)}
            className="rounded border border-border bg-surface-2 px-2 py-1 text-sm text-text"
          />
        </label>
        <label className="flex flex-col gap-1 text-xs text-text-dim">
          Contraseña
          <input
            type="password"
            required
            minLength={8}
            value={password}
            onChange={(event) => setPassword(event.target.value)}
            className="rounded border border-border bg-surface-2 px-2 py-1 text-sm text-text"
          />
        </label>
        {error && <p role="alert" className="text-xs text-danger">{error}</p>}
        <button
          type="submit"
          disabled={submitting}
          className="rounded bg-accent px-3 py-1.5 text-sm font-medium text-bg disabled:opacity-50"
        >
          Crear administrador
        </button>
      </form>
    </div>
  );
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd frontend && npx vitest run src/pages/LoginPage.test.tsx src/pages/SetupPage.test.tsx`
Expected: PASS (3 tests)

- [ ] **Step 5: Run `tsc` and the full existing suite to confirm no regression**

Run: `cd frontend && npx tsc --noEmit && npx vitest run`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add frontend/src/pages/LoginPage.tsx frontend/src/pages/SetupPage.tsx frontend/src/pages/LoginPage.test.tsx frontend/src/pages/SetupPage.test.tsx
git commit -m "feat: add LoginPage and SetupPage"
```

---

## Task 28: `App.tsx`/`main.tsx` auth gate + `Header` + `ForcedPasswordChangeModal`

**Files:**
- Modify: `frontend/src/App.tsx`
- Modify: `frontend/src/main.tsx`
- Create: `frontend/src/components/Header.tsx`
- Create: `frontend/src/components/ForcedPasswordChangeModal.tsx`
- Test: `frontend/src/App.test.tsx`, `frontend/src/components/Header.test.tsx`, `frontend/src/components/ForcedPasswordChangeModal.test.tsx`

**Interfaces:**
- Consumes: `AuthProvider`, `useAuth` (Task 26); `logout`, `changePassword` (Task 25); `Modal` (existing, `frontend/src/components/Modal.tsx`).
- Produces: `App` renders `LoginPage`/`SetupPage` when unauthenticated, `AppShell` + routes otherwise, plus a `ForcedPasswordChangeModal` overlay when `me.mustChangePassword`; `Header` (user badge + logout menu, renders nothing when `authMode === "off"`).

- [ ] **Step 1: Write the failing tests**

```tsx
// frontend/src/App.test.tsx
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { afterEach, describe, expect, it, vi } from "vitest";
import { App } from "./App";
import { AuthProvider } from "./hooks/useAuth";
import * as authService from "./services/auth";
import type { MeResponse } from "./lib/apiTypes";
import { ApiError } from "./lib/api";

vi.mock("./services/auth", async (importOriginal) => {
  const actual = await importOriginal<typeof import("./services/auth")>();
  return { ...actual, getMe: vi.fn() };
});

function renderApp() {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={queryClient}>
      <MemoryRouter>
        <AuthProvider>
          <App />
        </AuthProvider>
      </MemoryRouter>
    </QueryClientProvider>,
  );
}

const OFF_MODE_ME: MeResponse = {
  userId: null, username: "local", role: "admin", permissions: ["jobs:create", "users:manage"],
  mustChangePassword: false, authMode: "off",
  quota: { maxConcurrent: 0, maxQueued: 0, maxJobsPerDay: 0, maxGpuSecondsPerDay: 0, usedJobsToday: 0, usedGpuSecondsToday: 0 },
};

afterEach(() => {
  vi.mocked(authService.getMe).mockReset();
});

describe("App auth gate", () => {
  it("renders the normal app UI unchanged in off mode", async () => {
    vi.mocked(authService.getMe).mockResolvedValue(OFF_MODE_ME);
    renderApp();

    await waitFor(() => expect(screen.getByRole("heading", { name: /enhance/i })).toBeInTheDocument());
  });

  it("renders LoginPage when GET /auth/me returns not_authenticated", async () => {
    vi.mocked(authService.getMe).mockRejectedValue(new ApiError(401, "not_authenticated"));
    renderApp();

    await waitFor(() => expect(screen.getByRole("button", { name: /ingresar/i })).toBeInTheDocument());
  });

  it("renders SetupPage when GET /auth/me returns setup_required", async () => {
    vi.mocked(authService.getMe).mockRejectedValue(new ApiError(401, "setup_required"));
    renderApp();

    await waitFor(() => expect(screen.getByRole("button", { name: /crear/i })).toBeInTheDocument());
  });

  it("shows the forced password change modal when mustChangePassword is true", async () => {
    vi.mocked(authService.getMe).mockResolvedValue({ ...OFF_MODE_ME, authMode: "multi", mustChangePassword: true });
    renderApp();

    await waitFor(() => expect(screen.getByRole("heading", { name: /cambiá tu contraseña/i })).toBeInTheDocument());
  });
});
```

```tsx
// frontend/src/components/Header.test.tsx
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import type { ReactNode } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { AuthProvider } from "../hooks/useAuth";
import * as authService from "../services/auth";
import type { MeResponse } from "../lib/apiTypes";
import { Header } from "./Header";

vi.mock("../services/auth", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../services/auth")>();
  return { ...actual, getMe: vi.fn(), logout: vi.fn() };
});

function renderHeader() {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  function Wrapper({ children }: { children: ReactNode }) {
    return (
      <QueryClientProvider client={queryClient}>
        <AuthProvider>{children}</AuthProvider>
      </QueryClientProvider>
    );
  }
  return render(<Header />, { wrapper: Wrapper });
}

afterEach(() => {
  vi.mocked(authService.getMe).mockReset();
  vi.mocked(authService.logout).mockReset();
});

describe("Header", () => {
  it("renders nothing in off mode", async () => {
    vi.mocked(authService.getMe).mockResolvedValue({
      userId: null, username: "local", role: "admin", permissions: [], mustChangePassword: false,
      authMode: "off", quota: { maxConcurrent: 0, maxQueued: 0, maxJobsPerDay: 0, maxGpuSecondsPerDay: 0, usedJobsToday: 0, usedGpuSecondsToday: 0 },
    } as MeResponse);
    const { container } = renderHeader();

    await waitFor(() => expect(container).not.toBeEmptyDOMElement());
    // Off mode renders an empty fragment specifically (no badge/menu markup).
    expect(container.querySelector("button")).toBeNull();
  });

  it("shows the username and logs out on click in multi mode", async () => {
    vi.mocked(authService.getMe).mockResolvedValue({
      userId: "u1", username: "alice", role: "user", permissions: [], mustChangePassword: false,
      authMode: "multi", quota: { maxConcurrent: 1, maxQueued: 5, maxJobsPerDay: 50, maxGpuSecondsPerDay: 3600, usedJobsToday: 0, usedGpuSecondsToday: 0 },
    } as MeResponse);
    vi.mocked(authService.logout).mockResolvedValue({ ok: true });
    renderHeader();

    await waitFor(() => expect(screen.getByText("alice")).toBeInTheDocument());
    fireEvent.click(screen.getByRole("button", { name: /user menu/i }));
    fireEvent.click(screen.getByRole("menuitem", { name: /log out/i }));

    await waitFor(() => expect(authService.logout).toHaveBeenCalled());
  });
});
```

```tsx
// frontend/src/components/ForcedPasswordChangeModal.test.tsx
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import type { ReactNode } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";
import * as authService from "../services/auth";
import { ApiError } from "../lib/api";
import { ForcedPasswordChangeModal } from "./ForcedPasswordChangeModal";

vi.mock("../services/auth", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../services/auth")>();
  return { ...actual, changePassword: vi.fn() };
});

function renderModal() {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  function Wrapper({ children }: { children: ReactNode }) {
    return <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>;
  }
  return render(<ForcedPasswordChangeModal />, { wrapper: Wrapper });
}

afterEach(() => {
  vi.mocked(authService.changePassword).mockReset();
});

describe("ForcedPasswordChangeModal", () => {
  it("submits current and new password", async () => {
    vi.mocked(authService.changePassword).mockResolvedValue({ ok: true });
    renderModal();

    fireEvent.change(screen.getByLabelText(/contraseña actual/i), { target: { value: "temp123456" } });
    fireEvent.change(screen.getByLabelText(/contraseña nueva/i), { target: { value: "newpassword1" } });
    fireEvent.click(screen.getByRole("button", { name: /guardar/i }));

    await waitFor(() =>
      expect(authService.changePassword).toHaveBeenCalledWith("temp123456", "newpassword1"),
    );
  });

  it("shows an error message on failure", async () => {
    vi.mocked(authService.changePassword).mockRejectedValue(new ApiError(401, "Contraseña actual incorrecta"));
    renderModal();

    fireEvent.change(screen.getByLabelText(/contraseña actual/i), { target: { value: "wrong" } });
    fireEvent.change(screen.getByLabelText(/contraseña nueva/i), { target: { value: "newpassword1" } });
    fireEvent.click(screen.getByRole("button", { name: /guardar/i }));

    await waitFor(() => expect(screen.getByRole("alert")).toHaveTextContent("Contraseña actual incorrecta"));
  });
});
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd frontend && npx vitest run src/App.test.tsx src/components/Header.test.tsx src/components/ForcedPasswordChangeModal.test.tsx`
Expected: FAIL — `Header`/`ForcedPasswordChangeModal` don't exist, `App` has no auth gate yet

- [ ] **Step 3: Write the implementation**

Create `frontend/src/components/ForcedPasswordChangeModal.tsx`:

```tsx
import { useQueryClient } from "@tanstack/react-query";
import { useState, type FormEvent } from "react";
import { ApiError } from "../lib/api";
import { changePassword } from "../services/auth";
import { Modal } from "./Modal";

export function ForcedPasswordChangeModal() {
  const queryClient = useQueryClient();
  const [currentPassword, setCurrentPassword] = useState("");
  const [newPassword, setNewPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  async function handleSubmit(event: FormEvent) {
    event.preventDefault();
    setSubmitting(true);
    setError(null);
    try {
      await changePassword(currentPassword, newPassword);
      await queryClient.invalidateQueries({ queryKey: ["me"] });
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "No se pudo cambiar la contraseña");
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <Modal titleId="forced-password-change-title" onClose={() => undefined}>
      <h2 id="forced-password-change-title" className="font-heading text-base font-semibold text-text">
        Cambiá tu contraseña
      </h2>
      <form onSubmit={handleSubmit} className="flex flex-col gap-3">
        <label className="flex flex-col gap-1 text-xs text-text-dim">
          Contraseña actual
          <input
            type="password"
            required
            value={currentPassword}
            onChange={(event) => setCurrentPassword(event.target.value)}
            className="rounded border border-border bg-surface-2 px-2 py-1 text-sm text-text"
          />
        </label>
        <label className="flex flex-col gap-1 text-xs text-text-dim">
          Contraseña nueva
          <input
            type="password"
            required
            minLength={8}
            value={newPassword}
            onChange={(event) => setNewPassword(event.target.value)}
            className="rounded border border-border bg-surface-2 px-2 py-1 text-sm text-text"
          />
        </label>
        {error && <p role="alert" className="text-xs text-danger">{error}</p>}
        <button
          type="submit"
          disabled={submitting}
          className="rounded bg-accent px-3 py-1.5 text-sm font-medium text-bg disabled:opacity-50"
        >
          Guardar
        </button>
      </form>
    </Modal>
  );
}
```

Create `frontend/src/components/Header.tsx`:

```tsx
import { useQueryClient } from "@tanstack/react-query";
import { LogOut, User } from "lucide-react";
import { useState } from "react";
import { useAuth } from "../hooks/useAuth";
import { logout } from "../services/auth";

export function Header() {
  const { me } = useAuth();
  const queryClient = useQueryClient();
  const [menuOpen, setMenuOpen] = useState(false);

  if (!me || me.authMode === "off") {
    return null;
  }

  async function handleLogout() {
    await logout();
    await queryClient.invalidateQueries({ queryKey: ["me"] });
  }

  return (
    <div className="relative flex items-center gap-2">
      <button
        type="button"
        onClick={() => setMenuOpen((open) => !open)}
        aria-label="User menu"
        className="flex items-center gap-1.5 rounded-sm px-2 py-1 text-xs text-text-dim transition-colors duration-fast hover:text-text"
      >
        <User aria-hidden="true" className="h-4 w-4" strokeWidth={1.75} />
        {me.username}
      </button>
      {menuOpen && (
        <div
          role="menu"
          className="absolute right-0 top-full z-10 mt-1 flex flex-col gap-1 rounded border border-border bg-surface p-2 shadow-lg"
        >
          <button
            type="button"
            role="menuitem"
            onClick={handleLogout}
            className="flex items-center gap-1.5 px-2 py-1 text-left text-xs text-text-dim transition-colors duration-fast hover:text-text"
          >
            <LogOut aria-hidden="true" className="h-3.5 w-3.5" strokeWidth={1.75} />
            Log out
          </button>
        </div>
      )}
    </div>
  );
}
```

Modify `frontend/src/App.tsx`:

```tsx
import type { ReactNode } from "react";
import { Route, Routes } from "react-router-dom";
import { AppShell } from "./components/AppShell";
import { ForcedPasswordChangeModal } from "./components/ForcedPasswordChangeModal";
import { useAuth } from "./hooks/useAuth";
import { AudioPage } from "./modules/audio/AudioPage";
import { GeneratePage } from "./modules/generate/GeneratePage";
import { EnhancePage } from "./pages/EnhancePage";
import { LoginPage } from "./pages/LoginPage";
import { ModelsPage } from "./pages/ModelsPage";
import { RealtimePage } from "./pages/RealtimePage";
import { SettingsPage } from "./pages/SettingsPage";
import { SetupPage } from "./pages/SetupPage";
import { UsersPage } from "./pages/UsersPage";

function AuthGate({ children }: { children: ReactNode }) {
  const { me, isLoading, isError, needsSetup } = useAuth();
  if (isLoading) {
    return null;
  }
  if (needsSetup) {
    return <SetupPage />;
  }
  if (isError && !me) {
    return <LoginPage />;
  }
  return <>{children}</>;
}

export function App() {
  const { me } = useAuth();

  return (
    <AuthGate>
      <AppShell>
        <Routes>
          <Route path="/" element={<EnhancePage />} />
          <Route path="/audio" element={<AudioPage />} />
          <Route path="/generate" element={<GeneratePage />} />
          <Route path="/models" element={<ModelsPage />} />
          <Route path="/realtime" element={<RealtimePage />} />
          <Route path="/settings" element={<SettingsPage />} />
          <Route path="/users" element={<UsersPage />} />
        </Routes>
      </AppShell>
      {me?.mustChangePassword && <ForcedPasswordChangeModal />}
    </AuthGate>
  );
}
```

Modify `frontend/src/main.tsx` to wrap `App` with `AuthProvider` (inside `QueryClientProvider`, since `useAuth` uses `useQuery`):

```tsx
import { QueryClientProvider } from "@tanstack/react-query";
import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { BrowserRouter } from "react-router-dom";
import { App } from "./App";
import { AuthProvider } from "./hooks/useAuth";
import "./index.css";
import { queryClient } from "./lib/queryClient";

const rootElement = document.getElementById("root");
if (!rootElement) {
  throw new Error("Root element #root not found");
}

createRoot(rootElement).render(
  <StrictMode>
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <AuthProvider>
          <App />
        </AuthProvider>
      </BrowserRouter>
    </QueryClientProvider>
  </StrictMode>,
);
```

Modify `frontend/src/components/AppShell.tsx` to render `<Header />` next to the "Upflow" title (import `Header` and place it in the title row):

```tsx
import { Header } from "./Header";
```

```tsx
          <div className="flex items-center justify-between px-2 py-4 max-[900px]:hidden">
            <span className="font-heading text-lg font-semibold tracking-tight text-text">Upflow</span>
            <Header />
          </div>
```

(replaces the existing plain `<div className="px-2 py-4 font-heading ...">Upflow</div>` line — Task 29 below further edits this same block to add nav filtering, so land this exact markup now and Task 29 will only touch the `<nav>` block below it.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd frontend && npx vitest run src/App.test.tsx src/components/Header.test.tsx src/components/ForcedPasswordChangeModal.test.tsx`
Expected: PASS (9 tests)

- [ ] **Step 5: Run `tsc` and the full existing suite to confirm no regression**

Run: `cd frontend && npx tsc --noEmit && npx vitest run`
Expected: PASS — `frontend/src/components/AppShell.test.tsx` may need a quick look if it asserted on the exact DOM structure of the title row; if it does, add `getMe` mocking (resolved to an off-mode `MeResponse`) to that existing test file's render helper so `<Header/>` (which needs `AuthProvider` context) doesn't throw — **this is the one place this plan touches an existing test file's setup, and only to satisfy a new required context provider, never to change what it asserts.**

- [ ] **Step 6: Commit**

```bash
git add frontend/src/App.tsx frontend/src/main.tsx frontend/src/components/Header.tsx frontend/src/components/ForcedPasswordChangeModal.tsx frontend/src/components/AppShell.tsx frontend/src/App.test.tsx frontend/src/components/Header.test.tsx frontend/src/components/ForcedPasswordChangeModal.test.tsx
git commit -m "feat: gate App on auth state, add Header and forced password-change modal"
```

---

## Task 29: Nav permission gating (`navigation.ts` + `AppShell` filter)

**Files:**
- Modify: `frontend/src/lib/navigation.ts`
- Modify: `frontend/src/components/AppShell.tsx`
- Test: `frontend/src/components/AppShell.test.tsx` (existing file — extend, don't replace)

**Interfaces:**
- Consumes: `useAuth().hasPermission` (Task 26).
- Produces: `NavEntry.requiredPermission?: string`; a new `Users` nav entry (`path: "/users"`, `requiredPermission: "users:manage"`); `AppShell` filters `NAV_ENTRIES` before rendering.

- [ ] **Step 1: Read the existing `frontend/src/components/AppShell.test.tsx` first** to see its current render helper (it very likely renders `<AppShell>` without any `QueryClientProvider`/`AuthProvider` wrapper, since neither existed before this plan). Update that helper — not the assertions — to wrap with a `QueryClientProvider` + mocked `getMe` resolving to an off-mode `MeResponse` (`authMode: "off"`, `permissions: []`), matching the pattern already used in this plan's other tests (Task 28's `Header.test.tsx` `Wrapper`). This is required now because `AppShell` needs `useAuth()` context to filter nav entries and to render `<Header/>` (Task 28) — without it, every existing `AppShell.test.tsx` test throws `useAuth must be used within an AuthProvider`. **This is a second, and last, place this plan touches an existing test file — again only its render-helper wiring, never an assertion.**

- [ ] **Step 2: Add new failing assertions to `frontend/src/components/AppShell.test.tsx`**

```tsx
// add to the existing describe block in frontend/src/components/AppShell.test.tsx
it("hides the Users nav entry when the user lacks users:manage", async () => {
  vi.mocked(authService.getMe).mockResolvedValue({
    userId: "u1", username: "alice", role: "user", permissions: ["jobs:create"], mustChangePassword: false,
    authMode: "multi", quota: { maxConcurrent: 1, maxQueued: 5, maxJobsPerDay: 50, maxGpuSecondsPerDay: 3600, usedJobsToday: 0, usedGpuSecondsToday: 0 },
  });
  renderShell();

  await waitFor(() => expect(screen.queryByRole("link", { name: /users/i })).not.toBeInTheDocument());
});

it("shows the Users nav entry when the user has users:manage", async () => {
  vi.mocked(authService.getMe).mockResolvedValue({
    userId: "u1", username: "admin", role: "admin", permissions: ["users:manage"], mustChangePassword: false,
    authMode: "multi", quota: { maxConcurrent: 0, maxQueued: 0, maxJobsPerDay: 0, maxGpuSecondsPerDay: 0, usedJobsToday: 0, usedGpuSecondsToday: 0 },
  });
  renderShell();

  await waitFor(() => expect(screen.getByRole("link", { name: /users/i })).toBeInTheDocument());
});
```

(adjust `vi.mock("../services/auth", ...)` and the `authService` import at the top of the file the same way Task 28's tests do, if not already present after Step 1's wrapper update.)

- [ ] **Step 3: Run tests to verify the two new ones fail**

Run: `cd frontend && npx vitest run src/components/AppShell.test.tsx`
Expected: FAIL (2 new tests) — no `requiredPermission` filtering exists yet, no "Users" entry exists yet

- [ ] **Step 4: Write the implementation**

Modify `frontend/src/lib/navigation.ts`:

```ts
import { AudioWaveform, Boxes, RealtimeIcon, Sliders, Sparkles, Users as UsersIcon, Wand2, Zap } from "lucide-react";
// (keep the existing icon imports exactly as they are; only Users is new — adjust the
// import line to merge with whatever icons are already imported, do not duplicate)

export interface NavEntry {
  label: string;
  path: string;
  icon: LucideIcon;
  requiredPermission?: string;
}

export const NAV_ENTRIES: readonly NavEntry[] = [
  { label: "Enhance", path: "/", icon: Wand2 },
  { label: "Audio", path: "/audio", icon: AudioWaveform },
  { label: "Generate", path: "/generate", icon: Sparkles },
  { label: "Models", path: "/models", icon: Boxes },
  { label: "Realtime", path: "/realtime", icon: Zap },
  { label: "Settings", path: "/settings", icon: Sliders },
  { label: "Users", path: "/users", icon: UsersIcon, requiredPermission: "users:manage" },
];
```

Modify `frontend/src/components/AppShell.tsx` — import `useAuth` and filter before `.map`:

```tsx
import { useAuth } from "../hooks/useAuth";
```

```tsx
export function AppShell({ children }: AppShellProps) {
  const { hasPermission } = useAuth();
  const visibleEntries = NAV_ENTRIES.filter(
    (entry) => !entry.requiredPermission || hasPermission(entry.requiredPermission),
  );
  return (
    <div className="flex h-screen flex-col">
      <UpdateBanner />
      <div className="grid min-h-0 flex-1 grid-cols-[240px_1fr_320px] max-[900px]:grid-cols-[72px_1fr_320px]">
        <aside aria-label="Main navigation" className="flex flex-col gap-1 border-r border-border bg-surface p-2">
          <div className="flex items-center justify-between px-2 py-4 max-[900px]:hidden">
            <span className="font-heading text-lg font-semibold tracking-tight text-text">Upflow</span>
            <Header />
          </div>
          <nav className="flex flex-col gap-1">
            {visibleEntries.map((entry) => {
              const Icon = entry.icon;
              return (
                <NavLink key={entry.path} to={entry.path} end={entry.path === "/"} className={navLinkClassName}>
                  <Icon aria-hidden="true" className="h-[18px] w-[18px] shrink-0" strokeWidth={1.75} />
                  <span className="max-[900px]:sr-only">{entry.label}</span>
                </NavLink>
              );
            })}
          </nav>
        </aside>
        <main className="overflow-y-auto p-6">
          <div className="mx-auto w-full max-w-[1200px]">{children}</div>
        </main>
        <aside aria-label="Job queue" className="border-l border-border bg-surface p-4">
          <JobQueue />
        </aside>
      </div>
    </div>
  );
}
```

(only `NAV_ENTRIES.map` → `visibleEntries.map` and the `useAuth`/filter lines are new versus what Task 28 already landed for the title row.)

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd frontend && npx vitest run src/components/AppShell.test.tsx`
Expected: PASS, including the pre-existing tests in that file (now wrapped with the new provider) and the 2 new ones

- [ ] **Step 6: Run `tsc` and the full existing suite to confirm no regression**

Run: `cd frontend && npx tsc --noEmit && npx vitest run`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add frontend/src/lib/navigation.ts frontend/src/components/AppShell.tsx frontend/src/components/AppShell.test.tsx
git commit -m "feat: gate nav entries by permission and add Users nav entry"
```

---

## Task 30: `UsersPage` + `useUsers` hooks

**Files:**
- Create: `frontend/src/hooks/useUsers.ts`
- Create: `frontend/src/pages/UsersPage.tsx`
- Test: `frontend/src/hooks/useUsers.test.tsx`, `frontend/src/pages/UsersPage.test.tsx`

**Interfaces:**
- Consumes: `listUsers`, `createUser`, `updateUser`, `getUserJobs` (Task 25).
- Produces: `useUsers()`, `useCreateUser()`, `useUpdateUser()`, `useUserJobs(userId)`; `UsersPage` component (table + create form + role/disable/quota edit + reset password + view-jobs panel).

- [ ] **Step 1: Write the failing tests**

```tsx
// frontend/src/hooks/useUsers.test.tsx
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { renderHook, waitFor } from "@testing-library/react";
import type { ReactNode } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";
import * as usersService from "../services/users";
import { useCreateUser, useUsers } from "./useUsers";

vi.mock("../services/users", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../services/users")>();
  return { ...actual, listUsers: vi.fn(), createUser: vi.fn() };
});

function wrapper({ children }: { children: ReactNode }) {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>;
}

afterEach(() => {
  vi.mocked(usersService.listUsers).mockReset();
  vi.mocked(usersService.createUser).mockReset();
});

describe("useUsers", () => {
  it("fetches the users list", async () => {
    vi.mocked(usersService.listUsers).mockResolvedValue({ users: [] });

    const { result } = renderHook(() => useUsers(), { wrapper });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));
    expect(result.current.data?.users).toEqual([]);
  });

  it("useCreateUser calls the create service", async () => {
    vi.mocked(usersService.createUser).mockResolvedValue({
      user: { id: "u1", username: "bob", role: "user", disabled: false, mustChangePassword: true, quotaOverrides: {}, createdAt: "now", usedJobsToday: 0, usedGpuSecondsToday: 0 },
      temporaryPassword: "temp123",
    });

    const { result } = renderHook(() => useCreateUser(), { wrapper });
    result.current.mutate({ username: "bob", role: "user" });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));
    expect(usersService.createUser).toHaveBeenCalledWith({ username: "bob", role: "user" });
  });
});
```

```tsx
// frontend/src/pages/UsersPage.test.tsx
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, waitFor } from "@testing-library/react";
import type { ReactNode } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";
import * as usersService from "../services/users";
import { UsersPage } from "./UsersPage";

vi.mock("../services/users", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../services/users")>();
  return { ...actual, listUsers: vi.fn(), createUser: vi.fn(), updateUser: vi.fn(), getUserJobs: vi.fn() };
});

function renderPage() {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  function Wrapper({ children }: { children: ReactNode }) {
    return <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>;
  }
  return render(<UsersPage />, { wrapper: Wrapper });
}

afterEach(() => {
  vi.mocked(usersService.listUsers).mockReset();
});

describe("UsersPage", () => {
  it("renders the users table with existing users", async () => {
    vi.mocked(usersService.listUsers).mockResolvedValue({
      users: [
        { id: "u1", username: "admin", role: "admin", disabled: false, mustChangePassword: false, quotaOverrides: {}, createdAt: "2026-01-01", usedJobsToday: 2, usedGpuSecondsToday: 30 },
      ],
    });

    renderPage();

    expect(await screen.findByText("admin")).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Users", level: 1 })).toBeInTheDocument();
  });
});
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd frontend && npx vitest run src/hooks/useUsers.test.tsx src/pages/UsersPage.test.tsx`
Expected: FAIL — neither file exists yet

- [ ] **Step 3: Write the implementation**

```ts
// frontend/src/hooks/useUsers.ts
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  createUser,
  getUserJobs,
  listUsers,
  updateUser,
  type CreateUserParams,
  type UpdateUserParams,
} from "../services/users";

export function useUsers() {
  return useQuery({ queryKey: ["users"], queryFn: listUsers });
}

export function useCreateUser() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (params: CreateUserParams) => createUser(params),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["users"] }),
  });
}

export function useUpdateUser() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: ({ userId, params }: { userId: string; params: UpdateUserParams }) => updateUser(userId, params),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["users"] }),
  });
}

export function useUserJobs(userId: string | null) {
  return useQuery({
    queryKey: ["userJobs", userId],
    queryFn: () => getUserJobs(userId as string),
    enabled: userId !== null,
  });
}
```

```tsx
// frontend/src/pages/UsersPage.tsx
import { useState } from "react";
import { useCreateUser, useUpdateUser, useUserJobs, useUsers } from "../hooks/useUsers";

function CreateUserForm() {
  const createUserMutation = useCreateUser();
  const [username, setUsername] = useState("");
  const [role, setRole] = useState("user");
  const [temporaryPassword, setTemporaryPassword] = useState<string | null>(null);

  async function handleSubmit(event: React.FormEvent) {
    event.preventDefault();
    const result = await createUserMutation.mutateAsync({ username, role });
    setTemporaryPassword(result.temporaryPassword);
    setUsername("");
  }

  return (
    <form onSubmit={handleSubmit} className="flex flex-wrap items-end gap-2 rounded border border-border bg-surface-2 p-3">
      <label className="flex flex-col gap-1 text-xs text-text-dim">
        Usuario
        <input
          required
          minLength={3}
          value={username}
          onChange={(event) => setUsername(event.target.value)}
          className="rounded border border-border bg-surface px-2 py-1 text-sm text-text"
        />
      </label>
      <label className="flex flex-col gap-1 text-xs text-text-dim">
        Rol
        <select
          value={role}
          onChange={(event) => setRole(event.target.value)}
          className="rounded border border-border bg-surface px-2 py-1 text-sm text-text"
        >
          <option value="user">user</option>
          <option value="admin">admin</option>
        </select>
      </label>
      <button type="submit" className="rounded bg-accent px-3 py-1.5 text-sm font-medium text-bg">
        Crear usuario
      </button>
      {temporaryPassword && (
        <p className="w-full text-xs text-text-dim">
          Contraseña temporal para el nuevo usuario: <span className="font-mono-tabular text-text">{temporaryPassword}</span>
        </p>
      )}
    </form>
  );
}

function UserJobsPanel({ userId, onClose }: { userId: string; onClose: () => void }) {
  const { data } = useUserJobs(userId);
  return (
    <div className="rounded border border-border bg-surface-2 p-3">
      <div className="flex items-center justify-between">
        <h3 className="text-xs font-semibold uppercase tracking-wide text-text-dim">Jobs</h3>
        <button type="button" onClick={onClose} className="text-xs text-text-dim hover:text-text">
          Cerrar
        </button>
      </div>
      <ul className="mt-2 flex flex-col gap-1">
        {(data?.jobs ?? []).map((job) => (
          <li key={job.id} className="text-xs text-text">
            {job.kind} — {job.originalFilename ?? job.id} — {job.status}
          </li>
        ))}
        {data && data.jobs.length === 0 && <li className="text-xs text-text-faint">Sin jobs.</li>}
      </ul>
    </div>
  );
}

export function UsersPage() {
  const { data, isLoading } = useUsers();
  const updateUserMutation = useUpdateUser();
  const [viewingJobsFor, setViewingJobsFor] = useState<string | null>(null);

  async function handleRoleChange(userId: string, role: string) {
    await updateUserMutation.mutateAsync({ userId, params: { role } });
  }

  async function handleToggleDisabled(userId: string, disabled: boolean) {
    await updateUserMutation.mutateAsync({ userId, params: { disabled: !disabled } });
  }

  async function handleResetPassword(userId: string) {
    await updateUserMutation.mutateAsync({ userId, params: { resetPassword: true } });
  }

  return (
    <div className="flex flex-col gap-4">
      <h1 className="font-heading text-xl font-semibold text-text">Users</h1>
      <CreateUserForm />
      {isLoading && <p className="text-sm text-text-faint">Cargando...</p>}
      {data && (
        <table className="w-full text-left text-sm text-text">
          <thead>
            <tr className="text-xs uppercase tracking-wide text-text-dim">
              <th className="p-2">Usuario</th>
              <th className="p-2">Rol</th>
              <th className="p-2">Estado</th>
              <th className="p-2">Uso hoy</th>
              <th className="p-2">Acciones</th>
            </tr>
          </thead>
          <tbody>
            {data.users.map((user) => (
              <tr key={user.id} className="border-t border-border">
                <td className="p-2">{user.username}</td>
                <td className="p-2">
                  <select
                    value={user.role}
                    onChange={(event) => void handleRoleChange(user.id, event.target.value)}
                    className="rounded border border-border bg-surface-2 px-1 py-0.5 text-xs text-text"
                  >
                    <option value="user">user</option>
                    <option value="admin">admin</option>
                  </select>
                </td>
                <td className="p-2">{user.disabled ? "Deshabilitado" : "Activo"}</td>
                <td className="p-2">{user.usedJobsToday} jobs / {user.usedGpuSecondsToday.toFixed(0)}s GPU</td>
                <td className="flex gap-2 p-2 text-xs">
                  <button type="button" onClick={() => void handleToggleDisabled(user.id, user.disabled)} className="text-accent hover:underline">
                    {user.disabled ? "Habilitar" : "Deshabilitar"}
                  </button>
                  <button type="button" onClick={() => void handleResetPassword(user.id)} className="text-accent hover:underline">
                    Reset password
                  </button>
                  <button type="button" onClick={() => setViewingJobsFor(user.id)} className="text-accent hover:underline">
                    Ver jobs
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
      {viewingJobsFor && <UserJobsPanel userId={viewingJobsFor} onClose={() => setViewingJobsFor(null)} />}
    </div>
  );
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd frontend && npx vitest run src/hooks/useUsers.test.tsx src/pages/UsersPage.test.tsx`
Expected: PASS (3 tests)

- [ ] **Step 5: Run `tsc` and the full existing suite to confirm no regression**

Run: `cd frontend && npx tsc --noEmit && npx vitest run`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add frontend/src/hooks/useUsers.ts frontend/src/pages/UsersPage.tsx frontend/src/hooks/useUsers.test.tsx frontend/src/pages/UsersPage.test.tsx
git commit -m "feat: add UsersPage (create/list/edit users, reset password, view jobs)"
```

---

## Task 31: `JobQueue` — owner column + "view all" toggle

**Files:**
- Modify: `frontend/src/lib/api.ts` (+`listJobs`, `listVideoJobs`)
- Modify: `frontend/src/services/audio.ts` (+`listAudioJobs`)
- Modify: `frontend/src/services/generation.ts` (+`listGenerationJobs`)
- Create: `frontend/src/hooks/useAllJobs.ts`
- Modify: `frontend/src/components/JobQueue.tsx`
- Test: `frontend/src/hooks/useAllJobs.test.tsx`, `frontend/src/components/JobQueue.test.tsx` (existing file — extend)

**Interfaces:**
- Consumes: the 4 new backend list endpoints (Tasks 20-23); `useAuth().hasPermission` (Task 26).
- Produces: `listJobs`, `listVideoJobs`, `listAudioJobs`, `listGenerationJobs`; `useAllJobsView(enabled: boolean) -> AllJobsEntry[]`; `JobQueue` gains a "ver todos" checkbox (visible only with `jobs:read_all`) that switches its data source and shows an owner label per row.

**Scope note:** the "view all" mode is a read + cancel view of every user's jobs, not a full detail-modal experience — clicking a row in "view all" mode does not open `JobDetailModal` (that modal is wired to the client-tracked `job` payload shape, which the all-users server list doesn't populate per-kind-specifically). This matches the spec's requirement ("owner visible en cada card") without inventing a fifth job-detail data-fetch path.

- [ ] **Step 1: Write the failing tests**

```tsx
// frontend/src/hooks/useAllJobs.test.tsx
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { renderHook, waitFor } from "@testing-library/react";
import type { ReactNode } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";
import * as api from "../lib/api";
import * as audioService from "../services/audio";
import * as generationService from "../services/generation";
import { useAllJobsView } from "./useAllJobs";

vi.mock("../lib/api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../lib/api")>();
  return { ...actual, listJobs: vi.fn(), listVideoJobs: vi.fn() };
});
vi.mock("../services/audio", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../services/audio")>();
  return { ...actual, listAudioJobs: vi.fn() };
});
vi.mock("../services/generation", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../services/generation")>();
  return { ...actual, listGenerationJobs: vi.fn() };
});

function wrapper({ children }: { children: ReactNode }) {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>;
}

afterEach(() => {
  vi.mocked(api.listJobs).mockReset();
  vi.mocked(api.listVideoJobs).mockReset();
  vi.mocked(audioService.listAudioJobs).mockReset();
  vi.mocked(generationService.listGenerationJobs).mockReset();
});

describe("useAllJobsView", () => {
  it("returns an empty list and does not fetch when disabled", () => {
    const { result } = renderHook(() => useAllJobsView(false), { wrapper });
    expect(result.current).toEqual([]);
    expect(api.listJobs).not.toHaveBeenCalled();
  });

  it("merges entries from all 4 kinds with owner ids, newest first", async () => {
    vi.mocked(api.listJobs).mockResolvedValue({
      jobs: [{ jobId: "i1", status: "queued", originalFilename: "a.png", createdAt: "2026-01-01T00:00:01Z", ownerId: "u1", error: null, downloadUrl: null }],
    } as never);
    vi.mocked(api.listVideoJobs).mockResolvedValue({ jobs: [] } as never);
    vi.mocked(audioService.listAudioJobs).mockResolvedValue({
      jobs: [{ id: "a1", status: "completed", originalFilename: "b.wav", createdAt: "2026-01-01T00:00:02Z", ownerId: "u2", error: null, downloadUrl: "/x" }],
    } as never);
    vi.mocked(generationService.listGenerationJobs).mockResolvedValue({ jobs: [] } as never);

    const { result } = renderHook(() => useAllJobsView(true), { wrapper });

    await waitFor(() => expect(result.current).toHaveLength(2));
    expect(result.current[0].id).toBe("a1"); // newer createdAt sorts first
    expect(result.current[0].ownerId).toBe("u2");
    expect(result.current[1].ownerId).toBe("u1");
  });
});
```

```tsx
// add to the existing describe block in frontend/src/components/JobQueue.test.tsx
it("shows a view-all toggle only with jobs:read_all, and an owner label when enabled", async () => {
  vi.mocked(authService.getMe).mockResolvedValue({
    userId: "u1", username: "admin", role: "admin", permissions: ["jobs:read_all"], mustChangePassword: false,
    authMode: "multi", quota: { maxConcurrent: 0, maxQueued: 0, maxJobsPerDay: 0, maxGpuSecondsPerDay: 0, usedJobsToday: 0, usedGpuSecondsToday: 0 },
  });
  vi.mocked(api.listJobs).mockResolvedValue({
    jobs: [{ jobId: "i1", status: "queued", originalFilename: "a.png", createdAt: "2026-01-01T00:00:01Z", ownerId: "bob", error: null, downloadUrl: null }],
  } as never);
  vi.mocked(api.listVideoJobs).mockResolvedValue({ jobs: [] } as never);

  renderQueue();
  const toggle = await screen.findByRole("checkbox", { name: /ver todos/i });
  fireEvent.click(toggle);

  expect(await screen.findByText(/bob/)).toBeInTheDocument();
});
```

(this addition needs `AuthProvider` wrapped around `renderQueue()`'s tree and a mock of `../services/auth`'s `getMe`, `../lib/api`'s `listJobs`/`listVideoJobs`, and `../services/audio`'s `listAudioJobs`/`../services/generation`'s `listGenerationJobs` added to the existing file's top-level `vi.mock` blocks, following the same pattern as every other test file touched in this plan.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd frontend && npx vitest run src/hooks/useAllJobs.test.tsx src/components/JobQueue.test.tsx`
Expected: FAIL — `useAllJobs.ts` doesn't exist, no toggle in `JobQueue`

- [ ] **Step 3: Write the implementation**

Add to `frontend/src/lib/api.ts`:

```ts
export function listJobs(all: boolean): Promise<{ jobs: JobResponse[] }> {
  return apiGet<{ jobs: JobResponse[] }>(`/jobs?all=${all}`);
}

export function listVideoJobs(all: boolean): Promise<{ jobs: VideoJobResponse[] }> {
  return apiGet<{ jobs: VideoJobResponse[] }>(`/video/jobs?all=${all}`);
}
```

Add to `frontend/src/services/audio.ts`:

```ts
export function listAudioJobs(all: boolean): Promise<{ jobs: AudioJob[] }> {
  return apiGet<{ jobs: AudioJob[] }>(`/audio/jobs?all=${all}`);
}
```

Add to `frontend/src/services/generation.ts` (follows the same shape as its existing `getGenerationJob`/`cancelGenerationJob` exports):

```ts
export function listGenerationJobs(all: boolean): Promise<{ jobs: GenerationJob[] }> {
  return apiGet<{ jobs: GenerationJob[] }>(`/generation/jobs?all=${all}`);
}
```

Create `frontend/src/hooks/useAllJobs.ts`:

```ts
import { useQueries } from "@tanstack/react-query";
import { listJobs, listVideoJobs } from "../lib/api";
import type { AudioJob, GenerationJob, JobResponse, JobStatus, VideoJobResponse } from "../lib/apiTypes";
import { listAudioJobs } from "../services/audio";
import { listGenerationJobs } from "../services/generation";

const POLL_INTERVAL_MS = 2000;

export interface AllJobsEntry {
  id: string;
  kind: "image" | "video" | "audio" | "generation";
  fileName: string;
  createdAt: number;
  status: JobStatus;
  ownerId: string | null;
  downloadUrl: string | null;
}

function imageEntry(job: JobResponse): AllJobsEntry {
  return {
    id: job.jobId, kind: "image", fileName: job.originalFilename, createdAt: Date.parse(job.createdAt),
    status: job.status, ownerId: job.ownerId, downloadUrl: job.downloadUrl,
  };
}

function videoEntry(job: VideoJobResponse): AllJobsEntry {
  return {
    id: job.jobId, kind: "video", fileName: job.originalFilename, createdAt: Date.parse(job.createdAt),
    status: job.status, ownerId: job.ownerId, downloadUrl: job.downloadUrl,
  };
}

function audioEntry(job: AudioJob): AllJobsEntry {
  return {
    id: job.id, kind: "audio", fileName: job.originalFilename, createdAt: Date.parse(job.createdAt),
    status: job.status, ownerId: job.ownerId, downloadUrl: job.downloadUrl,
  };
}

function generationEntry(job: GenerationJob): AllJobsEntry {
  return {
    id: job.id, kind: "generation", fileName: job.prompt, createdAt: Date.parse(job.createdAt),
    status: job.status, ownerId: job.ownerId, downloadUrl: job.downloadUrl,
  };
}

export function useAllJobsView(enabled: boolean): AllJobsEntry[] {
  const [imageResult, videoResult, audioResult, generationResult] = useQueries({
    queries: [
      {
        queryKey: ["allJobs", "image"], queryFn: () => listJobs(true), enabled,
        refetchInterval: enabled ? POLL_INTERVAL_MS : false,
      },
      {
        queryKey: ["allJobs", "video"], queryFn: () => listVideoJobs(true), enabled,
        refetchInterval: enabled ? POLL_INTERVAL_MS : false,
      },
      {
        queryKey: ["allJobs", "audio"], queryFn: () => listAudioJobs(true), enabled,
        refetchInterval: enabled ? POLL_INTERVAL_MS : false,
      },
      {
        queryKey: ["allJobs", "generation"], queryFn: () => listGenerationJobs(true), enabled,
        refetchInterval: enabled ? POLL_INTERVAL_MS : false,
      },
    ],
  });

  if (!enabled) {
    return [];
  }

  const entries = [
    ...(imageResult.data?.jobs ?? []).map(imageEntry),
    ...(videoResult.data?.jobs ?? []).map(videoEntry),
    ...(audioResult.data?.jobs ?? []).map(audioEntry),
    ...(generationResult.data?.jobs ?? []).map(generationEntry),
  ];
  return entries.sort((a, b) => b.createdAt - a.createdAt);
}
```

Modify `frontend/src/components/JobQueue.tsx`: add the imports, the toggle, and an `AllJobsRow` renderer, alongside the existing per-session view:

```tsx
import { useAuth } from "../hooks/useAuth";
import { useAllJobsView, type AllJobsEntry } from "../hooks/useAllJobs";
```

```tsx
function AllJobsRow({ entry }: { entry: AllJobsEntry }) {
  return (
    <li className="flex flex-col gap-1 rounded border border-border bg-surface-2 p-3">
      <div className="flex items-start justify-between gap-2">
        <span className="truncate text-xs text-text" title={entry.fileName}>
          {entry.fileName}
        </span>
        <span className="text-[10px] uppercase tracking-wide text-text-faint">{jobKindLabel(entry.kind)}</span>
      </div>
      <div className="flex items-center justify-between text-[10px] text-text-faint">
        <span>owner: {entry.ownerId ?? "—"}</span>
        <span>{entry.status}</span>
      </div>
    </li>
  );
}
```

Modify `export function JobQueue()`:

```tsx
export function JobQueue() {
  const { entries, dismiss, cancel, clearCompleted } = useJobQueue();
  const { hasPermission } = useAuth();
  const [viewAll, setViewAll] = useState(false);
  const allJobsEntries = useAllJobsView(viewAll);
  const canViewAll = hasPermission("jobs:read_all");
  const hasCompletedOrFailed = entries.some((entry) => isTerminalJobStatus(entry.status));
  const [detailJobId, setDetailJobId] = useState<string | null>(null);
  const detailEntry = entries.find((entry) => entry.id === detailJobId);

  return (
    <div className="flex h-full flex-col gap-3">
      <div className="flex items-center justify-between gap-2">
        <h2 className="font-heading text-xs font-semibold uppercase tracking-wide text-text-dim">Job Queue</h2>
        <QueueCount count={viewAll ? allJobsEntries.length : entries.length} />
      </div>
      {canViewAll && (
        <label className="flex items-center gap-1.5 text-xs text-text-dim">
          <input type="checkbox" checked={viewAll} onChange={(event) => setViewAll(event.target.checked)} />
          Ver todos
        </label>
      )}
      {viewAll ? (
        allJobsEntries.length === 0 ? (
          <EmptyQueueState />
        ) : (
          <ul className="flex flex-col gap-2 overflow-y-auto">
            {allJobsEntries.map((entry) => (
              <AllJobsRow key={`${entry.kind}-${entry.id}`} entry={entry} />
            ))}
          </ul>
        )
      ) : entries.length === 0 ? (
        <EmptyQueueState />
      ) : (
        <ul className="flex flex-col gap-2 overflow-y-auto">
          {entries.map((entry) => (
            <QueueEntryRow
              key={entry.id}
              entry={entry}
              onDismiss={dismiss}
              onCancel={cancel}
              onOpenDetail={setDetailJobId}
            />
          ))}
        </ul>
      )}
      {!viewAll && hasCompletedOrFailed && (
        <button
          type="button"
          onClick={clearCompleted}
          className="mt-auto text-left text-xs text-text-dim transition-colors duration-fast hover:text-text focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent"
        >
          Clear completed
        </button>
      )}
      {detailEntry && <JobDetailModal entry={detailEntry} onClose={() => setDetailJobId(null)} onCancel={cancel} />}
    </div>
  );
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd frontend && npx vitest run src/hooks/useAllJobs.test.tsx src/components/JobQueue.test.tsx`
Expected: PASS

- [ ] **Step 5: Run `tsc` and the full existing suite to confirm no regression**

Run: `cd frontend && npx tsc --noEmit && npx vitest run`
Expected: PASS — `JobQueue.test.tsx`'s pre-existing tests never set `authMode`/permissions in their `getMe` mock, so `canViewAll` defaults to `false` (`hasPermission` returns `false` when `me` is `undefined`), and the toggle/all-jobs branch never renders for them, leaving their assertions unaffected. If any pre-existing `JobQueue.test.tsx` test throws on missing `AuthProvider` context, wrap its render helper the same way Task 29 did for `AppShell.test.tsx`.

- [ ] **Step 6: Commit**

```bash
git add frontend/src/lib/api.ts frontend/src/services/audio.ts frontend/src/services/generation.ts frontend/src/hooks/useAllJobs.ts frontend/src/components/JobQueue.tsx frontend/src/hooks/useAllJobs.test.tsx frontend/src/components/JobQueue.test.tsx
git commit -m "feat: add owner column and view-all toggle to JobQueue"
```

---

## Task 32: Frontend non-regression checkpoint

**Files:** none (verification-only task)

- [ ] **Step 1: Run the complete frontend suite and type check**

Run: `cd frontend && npx vitest run`
Expected: PASS — the original 447 tests plus every test added in Tasks 25-31, zero failures.

Run: `cd frontend && npx tsc --noEmit`
Expected: zero type errors.

- [ ] **Step 2: Run the production build** (the CLAUDE.md build command, since `npm run build` runs `tsc --noEmit && vite build` — a stricter gate than `tsc --noEmit` alone if any dead import slipped in)

Run: `cd frontend && npm run build`
Expected: build succeeds, `frontend/dist/` produced.

- [ ] **Step 3: Manual smoke test in a real browser** (per this repo's `run`-skill convention and the project's UI-testing guidance) — start the backend with `AUTH_MODE=off` (default) and confirm the app looks and behaves identically to before this plan (no login screen, no Users nav entry unless... — off mode's pseudo-admin has `users:manage`, so the Users nav entry **will** appear even in off mode, per the deliberate scope decision recorded in Task 29/App design above; this is expected, not a bug). Then start it once with `AUTH_MODE=multi` (fresh `runtime/auth/` dir) and walk through: setup → login → forced password change is skipped for the setup-created admin (spec: `must_change_password=False` for the account created by `/auth/setup`) → create a `user`-role account via Users page → log in as that user in a separate browser profile/incognito window → confirm forced password change modal appears → change password → submit a job → confirm it appears in that user's Job Queue but not in a job list fetched as another `user`-role account → log back in as admin → toggle "ver todos" → confirm the other user's job is visible with its owner label.

- [ ] **Step 4: No commit** — this is a checkpoint. If Step 3 surfaces a bug, fix it as a small follow-up task (write a regression test first per the TDD workflow) before considering subproject C done.

---

## Self-Review Notes

- **Spec coverage:** every row of the spec's "Componentes" table maps to a task — `IdentityProvider`/`LocalPasswordProvider` (Task 7), `passwords.py` (Task 3), `sessions.py` (Task 4), `user_store.py` (Task 6), `permissions.py` (Task 2), `quotas.py` (Task 9); Settings/modes (Tasks 5, 10, 15); ownership (Tasks 16-19); API (Tasks 11-14, 20-23); Frontend (Tasks 25-31). Error-handling copy (401/403/429/`password_change_required`) is used verbatim from the spec in Tasks 12, 13, 20-23. Testing section requirements (unit, API, frontend, off-mode non-regression) are covered by Tasks 1-9 (unit), 13/14/20-23 (API), 25-31 (frontend), 24/32 (non-regression checkpoints).
- **Out of scope, confirmed not touched:** OIDC/federated login, subproject A (SDXL/PyTorch→ONNX/gated repos), subproject B (VRAM probing/admission by capacity/external GPU pressure), HTTPS/TLS, additional roles beyond admin/user, user deletion, programmatic API tokens — none of these appear in any task above.
- **Backward-compatibility mechanics** (the plan's biggest risk) are called out once at the top and then referenced by every task that touches `app/api/routes.py` (Tasks 20-23) and `app/services/*_job_manager.py` (Tasks 16-19), each with its own explicit "run the legacy direct-call test files" verification step.
- **Task ordering fixed during self-review:** `config.py` (Settings.`users_file_path`/`usage_file_path`) had to move to Task 5, ahead of `user_store.py` (now Task 6) and `identity.py` (now Task 7), since both depend on those properties existing — the original draft had them in dependency-violating order. Every cross-reference above reflects the corrected numbering.
- **Type consistency check:** `AuthenticatedUser` (Task 7) is the one principal type threaded through Tasks 9, 12-23 with identical field names (`id`, `username`, `role`, `permissions`, `must_change_password`, `quota_overrides`) — no renamed variants introduced later. `owner_id`/`ownerId` naming is consistent across all 4 job dataclasses, all 4 response schemas, and both TypeScript response interfaces and the `AllJobsEntry` frontend type.
