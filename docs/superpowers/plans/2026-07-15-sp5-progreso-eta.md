# SP5 — Progreso detallado + ETA + modal de job — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development.

**Goal:** Al hacer click en un job de la cola, abrir un modal con info del archivo, un stepper de todas las etapas (check/spinner/pendiente), barra de progreso % real y ETA honesto (solo cuando es fiable). Video + imagen.

**Architecture:** Backend calcula `progress` (0..1) ponderado por las etapas activas del job y `framesDone`/`framesTotal` (poller async que cuenta PNGs durante etapas de frames); lo expone en la respuesta del job. Frontend: click en item de cola → Modal (reusa `components/Modal.tsx`) con stepper + barra + ETA calculado client-side desde el ritmo de `progress` entre polls.

**Tech Stack:** FastAPI/pydantic (backend), React+TS (frontend), el Modal focus-trap existente.

## Global Constraints
- TDD; pytest verde (base 528) y vitest verde (base ~200); no romper jobs existentes.
- ETA HONESTO: se muestra solo cuando hay conteo de frames en vuelo y ritmo estable; en etapas no-cuantificables (probe/extract-audio/encode corto) mostrar el paso, no un número inventado.
- No bloquear el event loop: el poller de frames corre como task async con sleeps; usar `asyncio.to_thread` para el conteo de archivos si hace falta.
- El progreso es best-effort/monotónico no-decreciente en la UI (nunca retroceder visualmente).
- Reusar `components/Modal.tsx` (focus-trap, de SP2 T4) para el detalle.
- Diseño control-room existente (tokens, tabular mono para números/%/ETA/frames, lucide).
- Commits español convencional; sin Co-Authored-By. Rama `feature/sp5-progreso`.

---

### Task 1: Modelo de progreso backend (etapas ponderadas + framesTotal + progress en la respuesta)

**Files:** Modify `app/services/video_upscaler.py`, `app/services/job_manager.py`/`app/services/video_job_manager.py` (si aplica para imagen), `app/models.py`, `app/schemas.py`, `app/api/routes.py` (mapper de respuesta). Tests: `tests/test_progress.py`.
**Produces:**
- Un modelo de etapas ordenadas por tipo de job, con pesos, filtrando las inactivas para ESE job. Video activo: `probing`(2) → `extracting_frames`(8) → [`extracting_audio`/`enhancing_audio`(4) si keep_audio] → `upscaling_frames`(peso mayor, ~55) → [`interpolating_frames`(15) si fps_multiplier>1 o target_fps] → `encoding_video`(13). Pesos normalizados sobre las etapas activas.
- Helper puro `compute_progress(stages_done, current_stage, current_fraction) -> float` (0..1) y la lista `stages` (cada una: `key`, `label`, `weight`, `status` in {pending,active,done}).
- `job.metadata` gana: `stages` (lista ordenada con status), `progress` (0..1), `framesTotal` (int|None), `framesDone` (int|None), `stageStartedAt` (iso ts | None) — timestamps se estampan con utc_now del repo (no Date.now en tests).
- `framesTotal` para video: de ffprobe (`nb_frames` del stream de video; si falta, `round(duration * fps)` con el helper fps existente). Se setea tras probe.
- Respuesta del job (`schemas.py` VideoJobResponse + JobResponse) expone `progress` (reusar/poblar `progressPct` = progress*100 o un campo `progress` 0..1 — elegir uno y ser consistente; recomendado poblar el `progress_pct` existente además de exponer `stages`/`framesDone`/`framesTotal` vía metadata). El mapper (routes.py `video_job_to_response`/`job_to_response`) copia estos campos.
- Esta tarea setea progress en las TRANSICIONES de etapa (sin poller aún): al entrar a cada etapa marca las previas done, la actual active, progress = suma pesos done. framesDone/Total se exponen pero el conteo en vivo llega en T2.
- Imagen: etapas coarse `validating`(10) → `upscaling`(90) (marcadas en JobManager/engine); progress por transición.

### Task 2: Poller de frames en vivo (framesDone durante etapas de frames)

**Files:** Modify `app/services/video_upscaler.py`. Tests: extend `tests/test_progress.py` / `tests/test_pipeline_stage_order.py`.
**Consumes:** el modelo de T1.
**Produces:**
- Un context manager / helper `_track_frame_progress(job, output_dir, stage_key)` que lanza un task async: cada ~1s cuenta PNGs en `output_dir` (`_count_frames`, vía `asyncio.to_thread` para no bloquear), actualiza `job.metadata["framesDone"]` y recomputa `progress` (fracción de la etapa = framesDone/framesTotal, clamp 0..1). Se cancela y hace join limpio cuando la etapa termina (finally). Nunca deja el progreso retroceder.
- Envolver las etapas de frames con el tracker: `extracting_frames` (cuenta frames_in), `upscaling_frames` (frames_out), `interpolating_frames` (frames_interp). En cada una, framesDone avanza en vivo.
- Robustez: el poller nunca tumba el job (try/except dentro del loop + log), y su cancelación no enmascara errores de la etapa (patrón CancelledError ya usado en el repo).
- Test: con un fake que crea PNGs incrementalmente en un dir mientras "corre" la etapa, el tracker reporta framesDone creciente y progress monotónico; al terminar, framesDone==framesTotal; el poller se cancela sin dejar tasks colgados.

### Task 3: Modal de detalle del job (frontend) con stepper + barra + ETA

**Files:** Create `frontend/src/components/JobDetailModal.tsx`, `frontend/src/lib/eta.ts` (+ tests), `frontend/src/lib/jobProgress.ts` (deriva stepper/labels, + test). Modify `frontend/src/components/JobQueue.tsx` (click → abre modal), `frontend/src/components/JobCard.tsx` (barra determinada cuando hay progress), `apiTypes.ts` (campos nuevos: progress, stages, framesDone, framesTotal). Tests vitest.
**Consumes:** respuesta enriquecida de T1/T2.
**Produces:**
- Click en un item de la cola (JobQueue) abre `JobDetailModal` (reusa `Modal`): nombre del archivo, tipo, modelo, dispositivo, escala/perfil/fps/audio; **stepper vertical** de `stages` (check=done, spinner=active, gris=pending) con label legible; **barra de progreso %** (determinada, de `progress`); **frames X / Y** (tabular) cuando aplica; **ETA** vía `eta.ts`.
- `eta.ts`: `estimateEta(samples)` donde samples = `[{progress, t}]` recientes; calcula ritmo (Δprogress/Δt) de los últimos N samples; ETA = (1-progress)/ritmo. Devuelve null (oculto) si <2 samples, ritmo ≤0, o progreso estancado. Formatea "~2 min 30 s". El JobDetailModal/JobQueue mantiene un buffer de samples por job mientras poll-ea.
- Barra determinada también en el JobCard colapsado cuando hay `progress` (reemplaza la indeterminada); indeterminada solo si progress no disponible aún (queued/probing).
- Números tabular mono; estados con icono+texto (no solo color); reduced-motion en el spinner.
- Tests: eta.ts (varios ritmos, estancado→null, <2 samples→null, formato); jobProgress deriva stepper de stages; JobDetailModal renderiza pasos/barra/ETA; click en cola abre modal; Escape cierra (via Modal).

### Task 4: Progreso de imagen + smoke real + docs + merge + release

**Files:** Modify `app/services/engines/onnx_upscaler.py` (progreso por tiles: exponer framesDone=tilesDone/framesTotal=tilesTotal durante el tiling), image job stages; README; posibles ajustes menores. Tests.
**Produces:**
- Imagen ONNX con tiling: reportar progreso por tiles (tilesDone/tilesTotal) → barra real en el modal para imágenes grandes. ncnn imagen (subprocess único, rápido): etapas coarse validating→upscaling→done sin frame-count (honesto, sin ETA falso).
- **Smoke real**: correr un job de video corto y uno de imagen (con la app real, binarios vendored) y verificar que la respuesta del job expone progress creciente + stages + framesDone/Total, y que el modal (build) compila. Documentar evidencia.
- README: sección "Progreso y ETA" (qué muestra el modal, honestidad del ETA).
- pytest + vitest + build verdes. Review final de rama. Merge a master. Regenerar setup.exe + zip. Re-subir assets del release v0.1.0 (`gh release upload --clobber`).

## Self-Review
- Cobertura: modelo de progreso ponderado ✓(T1), frames en vivo ✓(T2), modal+stepper+barra+ETA honesto ✓(T3), imagen+smoke+docs+release ✓(T4). ETA nunca falso ✓. Reusa Modal ✓. No bloquea event loop (poller async + to_thread) ✓.
