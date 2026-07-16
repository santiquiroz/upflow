# SP7 — Orquestador multi-GPU (colas por dispositivo + auto-router opcional)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development.

**Goal:** Jobs en GPUs físicas distintas corren EN PARALELO (hoy un semáforo GPU global los serializa). Opcional: un router que reparte los jobs encolados a la GPU libre compatible (batch de temporada).

**Architecture:** Reemplazar el `asyncio.Semaphore` GPU único compartido por un **registro de semáforos por dispositivo** (`DeviceSemaphores`): cada device_id expuesto (dml:0, dml:1, cpu…) tiene su propio semáforo. Un job adquiere el semáforo de SU dispositivo → distintos dispositivos no se bloquean entre sí. Suficientes workers para paralelismo intra-manager. Router auto opcional.

**Contexto actual (leer):** `app/main.py:44` crea `gpu_semaphore = asyncio.Semaphore(settings.gpu_concurrency)` e inyecta el MISMO a `JobManager` y `VideoJobManager`. Cada manager (`job_manager.py`, `video_job_manager.py`) corre `settings.gpu_concurrency` workers y hace `async with self.gpu_semaphore:` alrededor del engine run (job_manager.py:197, video_job_manager.py:274). El dedup de GPU (ya en master) hace que el 7800XT-doble se exponga solo como dml:0 → keyear por device_id es correcto (no doble-cuenta la misma GPU física).

## Global Constraints
- Preservar TODA la seguridad existente: manejo de CancelledError (jobs no-zombie), unlink de source en finally, task_done, watchdog de estancamiento, retention sweeper, límite de cola bounded. Ninguno depende del semáforo — no romperlos.
- Sin deadlocks ni starvation. El semáforo por-dispositivo es leaf lock; nunca sostener dos semáforos de dispositivo a la vez.
- cpu no necesita semáforo GPU (onnx en cpu está acotado por cores) — darle su propio semáforo con `CPU_CONCURRENCY` (o alto) para no serializar cpu contra GPU.
- TDD estricto; pytest verde (base 610). Commits español convencional; sin Co-Authored-By. Rama feature/sp7-multigpu.

---

### Task 1: Registro de semáforos por dispositivo (reemplaza el semáforo GPU único)

**Files:** Create `app/services/device_semaphores.py`, `tests/test_device_semaphores.py`. Modify `app/config.py`, `app/main.py`, `app/services/job_manager.py`, `app/services/video_job_manager.py`. Tests: `tests/test_multigpu_concurrency.py` + actualizar los tests de concurrencia existentes.
**Produces:**
- `DeviceSemaphores`: `acquire(device_id) -> async context manager` que devuelve/crea (lazy, con `threading.Lock`/`asyncio.Lock` para el get-or-create) un `asyncio.Semaphore` por device_id. GPU devices → `Semaphore(settings.per_device_gpu_concurrency)`; `cpu` → `Semaphore(settings.cpu_concurrency)`. Un helper `in_flight(device_id) -> int` (permits tomados) para el router de T2.
- Config: `PER_DEVICE_GPU_CONCURRENCY` (default 1, alias que reemplaza el rol de `gpu_concurrency` como "por GPU"), `CPU_CONCURRENCY` (default 2), `MAX_CONCURRENT_JOBS` (default = un valor razonable, ej. 4 — número de workers por manager, para permitir jobs en paralelo en distintos dispositivos dentro de un manager). Mantener `gpu_concurrency` como deprecated/compat si algo lo lee (health, .env.example) — o migrar limpio y documentar.
- `main.py`: crear UN `DeviceSemaphores` (compartido) e inyectarlo a ambos managers en vez del `gpu_semaphore`.
- Managers: `start()` spawnea `settings.max_concurrent_jobs` workers (no `gpu_concurrency`); el worker hace `async with self.device_semaphores.acquire(job.device):` alrededor del engine run, keyeado por el device del job. Jobs en distintos dispositivos → semáforos distintos → paralelo. Mismo dispositivo → serializan (per-device concurrency).
- `/health` (o el endpoint que reporte concurrencia) refleja el nuevo modelo (per-device) sin romper el schema.
**Tests:** con un fake engine que registra device + timestamps de in-flight: 2 jobs en dml:0 y dml:1 corren SOLAPADOS (paralelo); 2 jobs en el MISMO dml:0 con per_device=1 NO se solapan (serial); cpu + dml:0 en paralelo; cancelación/limpieza siguen (los tests de CancelledError/cleanup existentes verdes); gpu_concurrency=0 edge ya no aplica (o validar per_device>=1). Cross-manager: un video en dml:0 y una imagen en dml:1 corren en paralelo (el bug reportado).

### Task 2: Auto-router opcional (reparte encolados a GPU libre compatible)

**Files:** Modify `app/services/device_semaphores.py` (o nuevo `app/services/device_router.py`), `app/config.py`, `app/models.py`, `app/api/routes.py`, `app/services/job_manager.py`/`video_job_manager.py`, `app/schemas.py`, frontend (Settings toggle + device picker "Auto"). Tests.
**Produces:**
- Sentinel de dispositivo `"auto"` (además de cpu/dml:N) y/o setting `ENABLE_AUTO_ROUTE` (default False). Cuando un job tiene device="auto" (o el toggle global está on y el user no fijó device), el worker, al desencolar, elige un dispositivo COMPATIBLE libre:
  - Compatibilidad por kind del modelo: `builtin-ncnn` → solo GPUs Vulkan (NO cpu); `onnx` → cpu o cualquier GPU.
  - "Libre/menos cargado": elegir el dispositivo compatible con más capacidad disponible (menor `in_flight`/mayor permits libres); si ninguno libre, elegir el de menor carga y esperar su semáforo (bloqueante). Determinístico, sin deadlock.
  - Resolver el device elegido en el job (job.device = elegido) para que el resto del pipeline y la respuesta lo reflejen.
- Validación: "auto" válido en create_job/create_video_job; si no hay dispositivo compatible (ej. ncnn + solo cpu disponible) → ValueError claro 400.
- Router es OPCIONAL: default off → comportamiento actual (respeta el device elegido por job). On → auto-reparte los que sean "auto".
- UI: toggle "Auto-repartir entre GPUs" (Settings o cerca del device picker) + opción "Auto" en el DevicePicker. textContent, tokens.
**Tests:** router elige dml:1 cuando dml:0 está ocupado (ambos compatibles); ncnn nunca va a cpu; onnx puede ir a cpu; sin dispositivo compatible → 400; con router off, respeta el device del job; auto con 2 GPUs libres reparte (no todos a la misma).

### Task 3: Smoke real multi-GPU + docs + merge + release

- **Smoke real**: en la máquina (7800XT + iGPU Radeon Graphics), lanzar 2 jobs simultáneos en dispositivos distintos (ej. imagen onnx en dml:0 y otra en dml:1, o video en dml:0 + imagen en dml:1) y verificar por timestamps/GPU load que corren EN PARALELO (no serial) — el bug reportado resuelto. Con auto-router on, verificar que reparte. Documentar evidencia (timestamps de in-flight solapados).
- **Docs**: README — sección "Multi-GPU": colas por dispositivo (jobs en GPUs distintas en paralelo), cómo elegir dispositivo por job, el toggle de auto-reparto y qué modelos son compatibles con qué dispositivos (ncnn=GPU Vulkan, onnx=cpu/GPU). `.env.example`: PER_DEVICE_GPU_CONCURRENCY, CPU_CONCURRENCY, MAX_CONCURRENT_JOBS, ENABLE_AUTO_ROUTE.
- pytest + vitest + build verdes. Review final de rama (ADVERSARIAL: foco deadlock/starvation/paralelismo real). Merge a master. Rebuild setup.exe+zip. Re-subir assets release v0.1.0.

## Self-Review
- Cobertura: per-device semaphores (paralelo real) ✓(T1), auto-router opcional ✓(T2), smoke+docs+release ✓(T3). Sin deadlock (leaf locks, un semáforo a la vez). Seguridad existente preservada ✓. cpu no serializa contra GPU ✓.
