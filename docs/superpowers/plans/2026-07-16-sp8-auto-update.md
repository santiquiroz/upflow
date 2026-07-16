# SP8 — Auto-update: avisar cuando hay una nueva release en GitHub

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development.

**Goal:** Que Upflow revise las releases del repo en GitHub y le avise al usuario (banner en la UI) si hay una versión más nueva que la instalada, con link a la release. Mecanismo **reutilizable** para otros proyectos del usuario (repo + versión inyectables por config; sin lógica hardcodeada al proyecto).

**Architecture:** Un `UpdateService` (backend) consulta `GET https://api.github.com/repos/{repo}/releases/latest`, compara el `tag_name` (sin la 'v') contra la versión instalada usando `packaging.version.Version`, y expone el resultado por `GET /api/v1/update-check`. El frontend consume ese endpoint (TanStack Query, staleTime largo) y muestra un banner descartable "Nueva versión vX disponible" con link. Nunca rompe la app si no hay red / rate-limit.

**Tech Stack:** httpx.AsyncClient (ya es dep; patrón de `hf_client.py` con transport inyectable para tests), `packaging.version` (ya instalado), `importlib.metadata.version("upflow")` (fuente de versión), FastAPI/pydantic, React+TS+TanStack Query, design system control-room.

## Global Constraints
- TDD; pytest verde (base 659) y vitest verde (base 290). Rama `feature/sp8-auto-update`. Commits español convencional; sin Co-Authored-By.
- **Nunca romper la app por el chequeo**: si GitHub no responde (offline, timeout, 403 rate-limit, 5xx), el endpoint responde 200 con `updateAvailable=false` y un campo `error` no-nulo; el banner simplemente no aparece. Sin excepciones que tumben el arranque ni el request.
- No bloquear el event loop: fetch async con httpx; timeout corto (default 5s).
- **Cache con TTL**: no pegarle a la API de GitHub en cada request (rate-limit anónimo = 60/h). Cachear el último resultado en memoria por `UPDATE_CHECK_TTL_SECONDS` (default 3600). Un fetch fallido no invalida un cache bueno previo.
- **Reutilizable**: `UpdateService` recibe `repo` y `current_version` desde config/params, sin nada específico de Upflow en la lógica. Documentar cómo reusarlo en otro proyecto (cambiar `UPDATE_REPO` + tener el paquete instalado con versión).
- Comparación robusta con `packaging.version.Version` (maneja pre-releases y evita el bug de comparar strings "0.10.0" < "0.9.0").

---

### Task 1: Servicio de versión + UpdateService + endpoint (backend)

**Files:** Create `app/services/update_service.py`, `app/core/version.py` (o helper en un módulo existente), `tests/test_update_service.py`, `tests/test_update_check_route.py`. Modify `app/config.py`, `app/schemas.py`, `app/api/routes.py`, `app/main.py` (wire `app.state.update_service`).
**Produces:**
- `get_app_version() -> str`: `importlib.metadata.version("upflow")`, con fallback a parsear `pyproject.toml` (`[project] version`) si el paquete no está instalado (dev/checkout sin `pip install -e .`). Nunca lanza; último fallback `"0.0.0"`.
- Config: `UPDATE_REPO` (default `"santiquiroz/upflow"`), `UPDATE_CHECK_ENABLED` (default `True`), `UPDATE_CHECK_TTL_SECONDS` (default `3600`), `UPDATE_API_TIMEOUT_SECONDS` (default `5.0`).
- `UpdateService(settings, transport=None)` (transport inyectable como `hf_client`): método `async check(force=False) -> UpdateStatus`. Hace GET a la Releases API (header `Accept: application/vnd.github+json`, User-Agent), lee `tag_name`/`html_url`/`published_at`, normaliza el tag (strip 'v'), compara con `get_app_version()` vía `packaging.version.Version`. Devuelve dataclass/objeto con: `current_version`, `latest_version` (str|None), `update_available` (bool), `release_url` (str|None), `published_at` (str|None), `checked_at`, `error` (str|None). Cache en memoria con TTL; `check()` devuelve el cache si es fresco. Si `UPDATE_CHECK_ENABLED=False` → `update_available=False`, sin fetch. Errores (httpx error, status != 200, JSON inválido, sin releases) → capturados, `error` seteado, `update_available=False`, conserva `latest_version` cacheada si había.
- Endpoint `GET /api/v1/update-check` (`get_update_service` dep desde `app.state.update_service`) → `UpdateCheckResponse` (pydantic en `schemas.py`) con esos campos en camelCase. Opción `?force=true` para saltar cache (para un botón "buscar ahora" futuro).
**Tests:** transport mock (httpx.MockTransport) devolviendo un release más nuevo → `update_available=True`, url/tag correctos; igual versión → False; release más vieja → False; pre-release semver; tag con y sin 'v'; timeout/500/403 → `update_available=False` + `error` seteado, sin excepción; cache: dos `check()` seguidos = un solo fetch (contar llamadas al transport); `force=True` re-fetchea; `UPDATE_CHECK_ENABLED=False` → sin fetch. `get_app_version` con paquete instalado y fallback a pyproject. Endpoint: shape camelCase correcto, 200 aun con error.

### Task 2: Banner de actualización (frontend) + docs + merge

**Files:** Create `frontend/src/hooks/useUpdateCheck.ts`, `frontend/src/components/UpdateBanner.tsx` (+ tests), `frontend/src/services/update.ts`, tipos en `apiTypes.ts`. Modify `frontend/src/components/Layout.tsx` (montar el banner), README. Tests vitest.
**Produces:**
- `services/update.ts`: `fetchUpdateCheck()` → GET `/api/v1/update-check`.
- `useUpdateCheck()`: TanStack Query, `staleTime` largo (ej. 1h), `retry:false`, no refetch agresivo. Devuelve el status.
- `UpdateBanner`: si `updateAvailable`, muestra una barra discreta (tokens control-room, no intrusiva) "Nueva versión {latestVersion} disponible" + link "Ver release" (`releaseUrl`, target _blank rel noopener) + botón descartar. **Descarte persistente por versión** en `localStorage` (`upflow.dismissedUpdate = latestVersion`): si el usuario descartó esa versión, no vuelve a aparecer hasta que salga una más nueva. Si `!updateAvailable` o hay `error` → no renderiza nada. Accesible (role, aria-label, foco), contraste AA, lucide icon.
- Montar en `Layout` (visible en todas las rutas, arriba). No romper el layout existente.
- README: sección "Actualizaciones" (cómo funciona el chequeo, que es opcional/silencioso, y **cómo reusar el patrón en otro proyecto**: cambiar `UPDATE_REPO`, tener el paquete con versión).
**Tests:** UpdateBanner: renderiza con updateAvailable + link correcto; no renderiza si no hay update o hay error; descartar oculta y persiste en localStorage; no reaparece para la misma versión descartada; reaparece para una versión más nueva. useUpdateCheck: mapea la respuesta. No debilitar tests existentes.

### Task 3: Smoke + review + merge + bump + release v0.2.0

- Smoke real: correr el backend, `curl /api/v1/update-check` → 200 con `currentVersion=0.1.0` (o la instalada); simular una release más nueva (o apuntar `UPDATE_REPO` a un repo con release mayor) y verificar `updateAvailable=true`; verificar que sin red no rompe (200 + error). Build del frontend verde; el banner compila.
- Review adversarial de rama (foco: que un fallo de red NUNCA rompa la app; rate-limit; que el cache no filtre entre requests de forma incorrecta; comparación semver correcta; el descarte persistente no oculte updates futuros).
- pytest + vitest + build verdes. Merge a master.
- **Bump versión 0.1.0 → 0.2.0** (pyproject) — esta release (multi-GPU + auto-update) es un minor. Regenerar setup.exe + zip como v0.2.0. Crear release **v0.2.0** en GitHub (notas: multi-GPU + auto-update) con ambos assets. (La v0.1.0 desplegada no tiene el código de auto-update; desde v0.2.0 en adelante el chequeo funciona.)

## Self-Review
- Cobertura: version source + UpdateService + endpoint ✓(T1), banner + persist-dismiss + docs ✓(T2), smoke+review+merge+bump+release ✓(T3). Nunca rompe por red ✓. Reutilizable (repo/version inyectables) ✓. Semver robusto ✓. Cache con TTL ✓.
