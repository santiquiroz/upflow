# Multi-usuario y autenticación (subproyecto C) — Design

**Fecha:** 2026-07-24
**Estado:** Approved (pendiente de plan de implementación)

## Motivación

Upflow va a ser accesible desde la web (deployment aún sin definir: túnel personal, LAN o público). Antes de abrir el catálogo de modelos de terceros (subproyecto A) y la admisión por capacidad (subproyecto B), se construye la base multi-usuario para no refactorizar contratos después. Visión futura del usuario: login federado con su dominio ("entrar con Microsoft") y una plataforma propia que centralice usuarios entre sus proyectos — el diseño deja ambos enchufables.

Orden acordado de subproyectos: **C (este spec) → A (SDXL + conversión PyTorch→ONNX + gated + buscador) → B (capacidad/admisión: VRAM libre por DXGI, requisitos por modelo, y presión EXTERNA de GPU — ej. un juego corriendo → pausar/encolar inteligentemente)**.

## Decisiones tomadas (brainstorming 2026-07-24)

| Decisión | Elección | Razón |
|---|---|---|
| Alcance de acceso | Agnóstico del deployment | El usuario aún no define túnel/LAN/público |
| Identidad | `IdentityProvider` enchufable; hoy `LocalPasswordProvider` | Futuro: OIDC/plataforma central sin migración (usuarios con `subject_id` estable + `external_subject` nullable) |
| Roles | `admin` + `user`, enum extensible con tabla central de permisos | Pedido explícito: "posibilidad de roles en el futuro" |
| Privacidad | Cada usuario ve SOLO sus jobs; admin puede listar los de una persona específica y matarlos | Pedido explícito |
| Cuotas | Completas: concurrencia + cola + tope diario de jobs y de segundos de GPU, por rol con override individual | Pedido explícito ("Cuotas completas") |
| Altas | Solo admin crea usuarios (contraseña temporal + `must_change_password`); sin auto-registro | Pedido explícito |
| Modo local | `AUTH_MODE=off` default: escritorio sigue sin login; `multi` opt-in por parámetro/config | Pedido explícito ("dos opciones al inicio"); buena práctica CON guardarraíl loopback |
| Guardarraíl | `off` + bind no-loopback → 403 a requests remotos con mensaje accionable | Imposible exponerse sin querer (patrón Jupyter/code-server) |
| Sesiones | Cookie HttpOnly SameSite=Lax firmada HMAC-SHA256, stateless, TTL 30d renovable; logout global por `session_ver` | Sin store de sesiones; sin JWT en JS (XSS) |
| Passwords | `hashlib.scrypt` stdlib (N=2^14, r=8, p=1, salt por usuario) | Cero dependencias nuevas |
| Storage | `users.json` + `usage.json` en config dir, atomic-write + lock (patrón `ModelRegistry`); interfaz `UserStore` | Consistente con el repo; swap futuro a SQLite sin tocar consumidores |

## Componentes

### 1. `app/services/auth/` (módulo nuevo)

- `identity.py`: `IdentityProvider` (protocol: `authenticate(credentials) -> UserIdentity | None`) + `LocalPasswordProvider`.
- `passwords.py`: hash/verify con `hashlib.scrypt`; salt 16B urandom por usuario.
- `sessions.py`: firmar/verificar cookie (payload: `user_id`, `expires_at`, `session_ver`; HMAC-SHA256 con secret de 32B urandom autogenerado y persistido en `.env` al primer arranque como `AUTH_SECRET`). Verificación compara `session_ver` contra el usuario → incrementarlo = logout global.
- `user_store.py`: `UserStore` sobre `users.json` — campos por usuario: `id` (uuid), `username`, `password_hash`, `salt`, `role`, `disabled`, `must_change_password`, `session_ver`, `external_subject` (nullable), `quota_overrides` (dict opcional), `created_at`.
- `permissions.py`: `Role` (str enum: `admin`, `user`), `Permission` (str enum), `ROLE_PERMISSIONS: dict[Role, frozenset[Permission]]`. Permisos: `jobs:create`, `jobs:read_own`, `jobs:cancel_own`, `jobs:read_all`, `jobs:cancel_any`, `models:install`, `models:delete`, `users:manage`, `settings:read`, `settings:write`, `devices:read`, `queue:read_anonymized`.
- `quotas.py`: `QuotaService` — defaults por rol (`user`: 1 concurrente, 5 en cola, 50 jobs/día, 3600 s GPU/día; `admin`: 0 = ilimitado) + overrides por usuario; contadores diarios en `usage.json` (`user_id → {date, jobs, gpu_seconds}`) con **reset lazy** (si `date` ≠ hoy → cero); los consumos diarios se registran al TERMINAR el job usando la duración real (`started_at→finished_at`).

### 2. Settings y modos

- `.env`: `AUTH_MODE=off|multi` (default `off`), `AUTH_SECRET` (autogenerado si falta y modo `multi`).
- `off`: dependencias de auth resuelven a un pseudo-usuario local con rol `admin` → comportamiento y tests actuales intactos.
- `multi`: `/api/*` exige sesión salvo `/health` y `/auth/*`. Cero usuarios → solo `/auth/setup` habilitado (one-shot: crea el admin inicial).
- Guardarraíl: middleware — si `AUTH_MODE=off` y el request no viene de loopback → 403 `"Upflow está en modo single-user. Activá AUTH_MODE=multi para acceso remoto."`.

### 3. Ownership de jobs

- `UpscaleJob`, `VideoUpscaleJob`, `AudioJob`, `GenerationJob` ganan `owner_id: str | None = None`.
- `create_job` de los 4 managers recibe y guarda el owner; ANTES de encolar, `QuotaService.check_admission(user)` (concurrencia → cola → diarios) — excedido → `QuotaExceededError` → HTTP 429 con mensaje específico del límite y cuándo se libera.
- Listado/detalle/cancel/download: filtrado por `owner_id == user.id` salvo permiso `jobs:read_all`/`jobs:cancel_any`. En modo `off` todo owner es None y el pseudo-admin ve todo (igual que hoy).

### 4. API (`/api/v1/auth/*`, `/api/v1/users/*`)

- `POST /auth/login` → set-cookie; rate limit 5 intentos/min por IP (memoria).
- `POST /auth/logout` (limpia cookie), `POST /auth/logout-all` (incrementa `session_ver`).
- `GET /auth/me` → identidad + rol + permisos + cuotas restantes (+ `authMode` para que la SPA sepa si mostrar login). En `off` devuelve el pseudo-usuario.
- `POST /auth/change-password`; `must_change_password` bloquea todo lo demás salvo `/auth/*`.
- `POST /auth/setup` — solo `multi` + cero usuarios.
- `POST /users`, `GET /users`, `PATCH /users/{id}` (rol, `quota_overrides`, `disabled`, reset de contraseña → temporal + `must_change_password`) — permiso `users:manage`. No hay DELETE de usuarios en esta fase (disable cumple el rol; borrar rompería ownership histórico).
- `GET /users/{id}/jobs` — jobs de una persona (los 4 kinds mezclados, forma de la cola global) — `jobs:read_all`. Cancel: los endpoints existentes aceptan `jobs:cancel_any`.
- Dependencias FastAPI: `get_current_user`, `require(permission)` — aplicadas por router; los routers existentes ganan `require` según la tabla.

### 5. Frontend

- `AuthContext`: `GET /auth/me` al montar; 401 → Login; backend dice "setup requerido" → Setup. Modo `off` → UI idéntica a hoy.
- Páginas Login y Setup (estilo control-room actual, mínimas).
- Header: badge usuario + menú (cambiar contraseña, logout). Modal de cambio forzado si `must_change_password`.
- Página **Users** (nav visible solo con `users:manage`): tabla (username, rol, estado, uso del día), alta con contraseña temporal, editar rol/cuotas/disable, reset password, y "ver jobs" → vista de cola filtrada por ese usuario con cancel.
- JobQueue: usuario ve lo suyo (backend filtra); admin: toggle "ver todos" y owner visible en cada card.
- Gates de UI por permisos de `/auth/me` (Models install/delete, Settings, Users).

## Manejo de errores

- 401 sin/mala sesión (SPA → Login), 403 sin permiso o guardarraíl remoto, 429 cuotas con texto específico ("Tenés 1 job corriendo y tu límite es 1..." / "Límite diario alcanzado: 50 jobs. Se resetea a medianoche"), 423-equivalente para `must_change_password` (403 con código `password_change_required`).
- Nada de credenciales/hashes en logs. Errores de login genéricos ("usuario o contraseña incorrectos") — sin enumerar usuarios.

## Testing

- **Unit**: passwords (roundtrip, salt distinto), sessions (firma, expiración, `session_ver`, tampering), permissions (tabla completa por rol), quotas (admisión por cada límite, reset lazy de fecha, overrides, admin ilimitado), user_store (atomic write, corrupt file backup — patrón registry).
- **API**: login ok/fail/rate-limit, setup one-shot (segunda vez 403), guardarraíl remoto en `off` (client con IP no-loopback simulada), owner-filtering en listado/detalle/cancel/download de los 4 kinds, `users:manage` CRUD, `GET /users/{id}/jobs`, 429 de cada cuota, `must_change_password` bloqueando, modo `off` = suite existente SIN CAMBIOS (no-regresión).
- **Frontend**: login flow (401→login→me→app), setup flow, gates por permiso, página Users, cola filtrada + toggle admin, modal de cambio forzado.

## Fuera de alcance (fases futuras)

- OIDC / "entrar con Microsoft" / plataforma central de usuarios → segundo `IdentityProvider` (el contrato ya lo soporta vía `external_subject`).
- Subproyecto A: SDXL directo, conversión automática PyTorch→ONNX, repos gated (token HF + mensaje), buscador de generación, checkpoints single-file.
- Subproyecto B: sondeo de VRAM libre (DXGI multi-vendor), requisitos por modelo, admisión por capacidad con mensajes dicientes, y **presión externa de GPU** (ej. juego corriendo → pausar/encolar jobs inteligentemente).
- HTTPS/TLS (lo resuelve el túnel o reverse proxy del deployment que se elija), roles adicionales, borrado de usuarios, API tokens programáticos.
