# SP10 — Cancelar jobs desde la UI (imagen / video / audio)

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development.

**Goal:** Botón "Cancel" en la UI que cancela un job (encolado o corriendo) de los 3 tipos. El backend ya tiene kill-on-cancel del subprocess; falta el cancel POR-JOB (método + endpoint) y el botón.

**Rama:** `feature/job-cancel`. Base pytest 701+ / vitest 314. Español convencional, sin Co-Authored-By. Backend = `app/**`,`tests/**`; frontend = `frontend/**`. Construcción PARALELA (un agente backend, un frontend); cada uno `git add` solo sus paths.

---

## CONTRATO FIJO

### Nuevo estado
`app/models.py`: `JobStatus.cancelled = "cancelled"`. Los 3 jobs (UpscaleJob/VideoUpscaleJob/AudioJob) lo usan.

### Patrón de cancelación (CRÍTICO — implementar EXACTO en los 3 managers)
Cada manager corre `settings.max_concurrent_jobs` workers que sacan de una `asyncio.Queue`. Cancelar UN job NO debe matar al worker (que sigue con otros jobs). Patrón:

1. El manager mantiene `self._active: dict[str, asyncio.Task] = {}` (job_id -> task que corre el engine).
2. En `_execute_job`, en vez de `await <engine>.run(job)` directo, envolver en una child task registrada por job_id:
```python
async def _execute_job(self, job) -> None:
    job.status = JobStatus.running
    job.started_at = utc_now()
    # ...(advance stage como ahora)...
    run_task = asyncio.ensure_future(self._do_engine_work(job))  # el await actual del engine
    self._active[job.id] = run_task
    try:
        await run_task
        job.status = JobStatus.completed
        # ...(complete stages como ahora)...
    except asyncio.CancelledError:
        run_task.cancel()  # asegura matar el subprocess si aun no
        if asyncio.current_task().cancelling():   # el WORKER fue cancelado -> SHUTDOWN
            job.status = JobStatus.failed
            job.error = "Job cancelled"
            raise                                  # propagar: el worker debe morir en shutdown
        job.status = JobStatus.cancelled           # cancel POR-JOB: worker sigue vivo
        job.error = None
    except Exception as exc:  # noqa: BLE001
        job.status = JobStatus.failed
        job.error = str(exc)
    finally:
        self._active.pop(job.id, None)
        job.finished_at = utc_now()
        self._unlink_source_safely(job.source_path)  # (o el cleanup existente del manager)
        self.queue.task_done()
```
   - `_do_engine_work(job)` = extraer el await del engine actual (image: `await engine.run(job)` devolviendo output_path; video/audio: su llamada actual). Debe setear `job.output_path` en éxito (o devolverlo y asignarlo en `_execute_job`).
   - **La distinción shutdown-vs-job-cancel es `asyncio.current_task().cancelling() > 0`** (Python 3.11+): en `stop()` se hace `worker_task.cancel()` -> el worker tiene cancelling()>=1 -> se re-lanza. En cancel-por-job solo se cancela la child (`run_task`), el worker no -> cancelling()==0 -> se marca cancelled y sigue.

3. **`cancel_job(self, job_id: str) -> bool`** (nuevo, en los 3 managers):
```python
def cancel_job(self, job_id: str) -> bool:
    job = self.jobs.get(job_id)
    if job is None:
        return False
    if job.status in (JobStatus.completed, JobStatus.failed, JobStatus.cancelled):
        return False  # ya terminó
    if job.status == JobStatus.queued:
        # aun no lo tomó un worker: marcar; el worker lo SALTEA al desencolar
        job.status = JobStatus.cancelled
        job.finished_at = utc_now()
        return True
    task = self._active.get(job_id)   # running
    if task is not None:
        task.cancel()
    return True
```

4. **`_worker` saltea los cancelled-mientras-encolados** (para no procesar un job ya cancelado que sigue en la cola):
```python
async def _worker(self):
    while True:
        job = await self.queue.get()
        if job.status == JobStatus.cancelled:      # cancelado mientras esperaba en cola
            self._unlink_source_safely(job.source_path)
            self.queue.task_done()
            continue
        # ...(resto igual: _run_pinned_job / _run_auto_job)...
```

Preservar TODA la seguridad existente: task_done balanceado (una vez por job, incluido el skip), unlink de source en finally, CancelledError de shutdown propaga, retention sweeper. Los tests de cancel/cleanup existentes deben seguir verdes.

### Endpoints (`app/api/routes.py`)
- `POST /api/v1/jobs/{job_id}/cancel` -> llama `job_manager.cancel_job(id)`. 200 con el JobResponse actualizado si True; 404 si el job no existe (cancel_job False por inexistente) — distinguir "no existe" de "ya terminó": si `get_job(id) is None` -> 404; si existe pero ya terminó -> 409 (o 200 idempotente con el estado actual). Elegir 409 "job already finished".
- `POST /api/v1/video/jobs/{job_id}/cancel` (idem video).
- `POST /api/v1/audio/jobs/{job_id}/cancel` (idem audio).

### Tests (pytest, TDD)
- cancelar un job QUEUED -> status cancelled, no se procesa (engine nunca corre), task_done balanceado, source unlinkeada.
- cancelar un job RUNNING (fake engine lento/cancelable) -> status cancelled, el subprocess/engine recibe CancelledError, el worker SIGUE VIVO y procesa el siguiente job.
- cancelar un job ya completed/failed -> 409, sin efecto.
- cancelar id inexistente -> 404.
- shutdown (`stop()`) sigue cancelando todos los workers y propaga (no confundir con job-cancel) — los tests de stop existentes verdes.
- Endpoints de los 3 tipos.
- Full pytest verde (base 701+).

## Task FRONTEND
- `services/*.ts`: `cancelJob(id)`, `cancelVideoJob(id)`, `cancelAudioJob(id)` -> POST a los endpoints.
- Botón **Cancel** en `JobCard.tsx` (y/o `JobDetailModal.tsx`) visible cuando el job está `queued` o `running` (no en completed/failed/cancelled). Click -> llama cancel + refetch del estado. Icono lucide (X/Ban), estilo danger sutil, con confirmacion inline o directa (elegir; directo está bien para un job). Estado `cancelled` se muestra con label+icono propios (no solo color) en JobCard/badge/stepper.
- `apiTypes.ts`: agregar "cancelled" al union de status si está tipado.
- Tests vitest: el botón aparece en queued/running y no en terminados; click dispara el cancel; estado cancelled se renderiza. No debilitar tests existentes. `npm --prefix frontend test -- --run` + build verdes.

## Self-Review
Cancel por-job sin matar worker (child task + `cancelling()`) ✓, queued-skip ✓, 3 tipos ✓, endpoints ✓, botón condicional ✓, seguridad existente preservada ✓.
