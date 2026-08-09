# Auditoría 2026-08-08 — flujo de jobs, consolidación y MCP

Auditoría profunda de las ~150 unidades de `app/` (3 pasadas paralelas: mapa de
API, revisión del núcleo de jobs, escaneo de duplicación) hecha para la v0.56.0.
Este documento registra **lo que se corrigió**, **lo que se decidió no tocar** y
el **roadmap de consolidación** pendiente con esfuerzo/riesgo.

## Corregido en v0.56.0

1. **Resurrección de jobs cancelados** (CRÍTICO, 5 managers): con los defaults
   de producción (4 workers, 1 permiso GPU por device) un job dequeued esperando
   `device_semaphores.acquire()` seguía en `status=queued`; `cancel_job()` lo
   marcaba cancelled sin nada que cancelar y `_execute_job` lo pisaba con
   `running` al liberarse el permiso. Fix: re-chequeo de `cancelled` al entrar a
   `_execute_job` en image/video/audio/generation/transcribe.
   Test: `tests/test_job_cancel_semaphore_window.py`.
2. **Cuota descubierta** (CRÍTICO): `transcribe`, `shape3d` y `download` no
   estaban en `QuotaService.attach_managers` (sin límite de concurrencia/cola
   para usuarios no-admin) y `DownloadJobManager` jamás llamaba `record_usage`
   (exento de las 4 dimensiones de cuota — vector de DoS en AUTH_MODE=multi).
   Fix en `main.py` + `download_job_manager.py`.
3. **RetentionSweeper ciego** (CRÍTICO): no conocía transcribe/shape3d/download.
   El sweep horario podía borrar la fuente de un job de transcripción encolado o
   de un doblaje largo a mitad de ffmpeg, y los dicts de jobs de esos 3 managers
   crecían sin límite. Fix: los 7 managers registrados.
   Test: `tests/test_review_fixes_2026_08.py`.
4. **Cancel de shape3d corrupto** (HIGH): `_run_job` pisaba `cancelled` con
   `completed`/`failed` al terminar el hilo. Fix: `_finish()` con guarda.
5. **4 subprocess sin techo** (hang holes): `routes._decoded_upload`,
   `transcribe_onnx._decoded_copy`, `transcribe_job_manager._run_ffmpeg` (un
   cancel ahora MATA el ffmpeg del mux/burn), `video_upscaler._detect_scene_cuts`.
   Todos por `run_guarded_process`/timeout explícito.
6. **arnndn sin escapar** (latente): `voice_chain._denoise_filter` armaba
   `arnndn=m=C:\...` — el `:` de la unidad rompe el filtergraph en Windows.
   Hoy ningún caller pasa `denoise="rnnoise"` a la cadena de voz (siempre
   `fft`), pero quedaba armado para fallar. Mismo escape que
   `AudioEnhancer._escape_filter_path`.

## Decidido NO tocar (requiere decisión del dueño)

- **Doble loudnorm** (`voice_chain` paso `loudness` + `audio_mastering`): un job
  con `voice_steps=["...","loudness"]` **y** `master` aplica loudnorm dos veces
  (la segunda mide audio ya normalizado; además la del voice_chain es single-pass,
  la forma que el propio docstring de mastering explica que "bombea"). Es cambio
  de comportamiento audible → decidir si el paso `loudness` debe deshabilitarse
  cuando hay `master`, o quitarse del catálogo de voz.
- **Descargas por token sin ownership** (`/print/repaired/{token}`,
  `/print/parts/{token}`): cualquiera con el token de 32 hex baja el archivo.
  Irrelevante en single-user; en multi-user conviene atar el token al owner.
- **Base class `QueuedJobManager`**: la revisión dejó la tabla comparativa de
  los 8 managers (qué comparten, qué varía). El fix (1) hoy vive copiado en 5
  lugares — la base class lo dejaría en uno. Esfuerzo M, riesgo medio; hacerlo
  en una sesión dedicada con la suite verde antes y después.

## Roadmap de consolidación (por prioridad payoff/riesgo)

Estado actualizado 2026-08-09 (v0.57.0):

| # | Qué | Esfuerzo | Riesgo | Estado |
|---|---|---|---|---|
| 1 | Merge tablas img2img/inpaint (`generation_pipeline_modes.py`) | S | Bajo | **HECHO v0.57.0** |
| 2 | `OnnxAudioRestorer` base (apollo ⟷ audiosr) | S-M | Bajo | **HECHO v0.57.0** (`engines/audio_restore_base.py`) |
| 3 | `engines/onnx_common.py` + `frame_workers.py` (8 módulos importan `_`-privados) | M | Bajo | pendiente |
| 4 | `run_checked_process()` + `is_non_empty_file` únicos | S | Bajo | **HECHO v0.57.0** (sdcpp/voice_enhance parcial a propósito: sus mensajes assertados componen distinto) |
| 5 | `SingleWorkerJobQueue` base para los 6 installers (seam `_process_next`, 94 refs) | M | Medio | pendiente — nota: los 5 managers de MEDIA ya tienen su base (`job_manager_base.QueuedJobManager`, v0.57.0) |
| 6 | Unificar loudness (voice_chain ⟷ mastering) | M | Med-Alto | **HECHO v0.56.1** (skip con metadata) |
| 7 | Colapsar los dos preflights sobre `CompatStrategy` | M | Bajo-Med | pendiente |
| 8 | Extraer `generation_staging.py` (installer ⟷ converter) | M | Medio | pendiente |
| 9 | Un solo parser de `dml:` + constante | S | Bajo | **HECHO v0.57.0** (`dml_device.py`; eran 4 implementaciones, no 3) |
| 10 | Una sola fuente de "¿está listo?" (Settings ⟷ CATALOG ⟷ validators) | M | Medio | parcial — `audiosr` y `shap-e-img2img` ya están en CATALOG (v0.57.0); la unificación de mecanismo sigue pendiente |

Inconsistencias de API detectadas (para una v2 de la API, no urgentes):
`jobId` vs `id` según familia; 201 vs 202 en creates; `transcribe`/`download`/
`shape3d` sin endpoint de listado; 3 endpoints de búsqueda HF y 4 de install
paralelos con la misma forma; dos conceptos distintos bajo `/capabilities`.
La capa MCP (`app/mcp/`) ya normaliza todo esto para agentes.

## Descartado con medición (v0.57.0)

**Batching de frames en el upscaler ONNX de video**: hipótesis era llenar el
~30% de GPU ociosa batcheando inferencias. Spike real (RX 7800 XT,
animevideov3-x2 fp16, 1080p, re-export con batch dinámico verificado
bit-exacto): batch=2 = paridad con batch=1, batch=4 = **-8%**. El SRVGG
compacto ya satura los ALUs por frame; la GPU ociosa está ENTRE inferencias
(readback/pre-post), no dentro. La palanca real sería solapar cómputo con I/O
(doble sesión / IOBinding pipelined) — spike distinto, no hecho. Va a la
lista de callejones medidos junto con ReBAR y la fusión Olive sobre fp16.

## MCP (nuevo en v0.56.0)

`app/mcp/` — servidor FastMCP (stdio) con 24 tools, cliente HTTP fino sobre la
API local. Diseño: modelo de jobs unificado sobre las 7 familias + tools de
operación que aceptan rutas locales. Ver README § "Servidor MCP".
