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

**ROADMAP COMPLETO 10/10** (2026-08-09, v0.57.0 + v0.58.0):

| # | Qué | Estado |
|---|---|---|
| 1 | Merge tablas img2img/inpaint | **HECHO v0.57.0** (`generation_pipeline_modes.py`) |
| 2 | `OnnxAudioRestorer` base | **HECHO v0.57.0** (`engines/audio_restore_base.py`) |
| 3 | `onnx_common.py` + `frame_workers.py` | **HECHO v0.58.0** — onnx_upscaler/onnx_video ya no exportan `_`-privados; blending/cache-LRU/worker-loops de 2-3 copias a 1; frame_pipeline ya no importa de un engine |
| 4 | `run_checked_process()` + `is_non_empty_file` | **HECHO v0.57.0** (sdcpp/voice_enhance parcial a propósito) |
| 5 | Base de los 6 installers | **HECHO v0.58.0** (`install_queue_base.SingleWorkerJobQueue`; enums por familia se quedan; `_process_next` unificado al superset observacionalmente equivalente) |
| 6 | Unificar loudness | **HECHO v0.56.1** (skip con metadata) |
| 7 | Preflights sobre `CompatStrategy` | **HECHO v0.58.0** (`measure_capacity`/`classify_repo` en model_preflight; generation entra por su strategy) |
| 8 | `generation_staging.py` | **HECHO v0.58.0** — converter ya no importa privados del installer; call-counts intactos vía `files=` opcional |
| 9 | Parser `dml:` único | **HECHO v0.57.0** (`dml_device.py`) |
| 10 | Fuente única "¿está listo?" | **HECHO v0.58.0** (`capabilities.resolve_one`; validadores delegan; apollo ganó su `SettingRequirement` — la tarjeta ya no miente con el flag apagado). Fuera con motivo: `Settings.*_available` (ciclo de import), chequeos MÁS fuertes que CATALOG (audiosr completitud, interpolation, sdcpp), features sin tarjeta (gmfss/editor/rnnoise) |

Inconsistencias de API detectadas (para una v2 de la API, no urgentes):
`jobId` vs `id` según familia; 201 vs 202 en creates; `transcribe`/`download`/
`shape3d` sin endpoint de listado; 3 endpoints de búsqueda HF y 4 de install
paralelos con la misma forma; dos conceptos distintos bajo `/capabilities`.
La capa MCP (`app/mcp/`) ya normaliza todo esto para agentes.

## Descartado con medición (v0.58.0)

**Solapar cómputo/I-O en el upscaler ONNX de video**: la premisa del "~30% de
GPU ociosa entre inferencias" resultó FALSA — desglose medido (RX 7800 XT,
1080p x2 fp16): GPU pura 82.2 ms/frame de ~93 ms totales = la GPU ya está
~88% ocupada; el overhead no-GPU es ~11 ms (12%). Techo teórico +12.5%;
mejor variante real (iobinding prealocado + update_inplace) +5.1%, dentro
del ruido entre corridas. Tres hallazgos de mecanismo que matan la idea:
(1) run_with_iobinding en DML es submit ASÍNCRONO — el sync vive en el
readback; (2) el readback sincroniza contra TODA la command queue, no contra
su frame — solape imposible a nivel API ORT+DML; (3) cualquier llamada ORT
concurrente sobre el device DML produce DEVICE REMOVAL (DXGI 887A0005), no
contención — endurece el hallazgo GMFSS histórico. El presupuesto de
optimización futuro está en el cómputo mismo (los 82 ms), no en el I/O.
Micro-candidato pendiente de A/B controlado: cachear io_binding+buffers por
(sesión, shape) en _infer_iobinding (+5% nominal). Spike:
scratchpad/spike-overlap/ de la sesión 2026-08-09.

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
