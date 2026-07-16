# SP9 — Apartado de Audio (mejora de audio standalone + reuso en video)

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development.

**Goal:** Un apartado "Audio" propio (como Imagen/Video): subir un archivo de audio → mejorarlo (denoise y/o restauración de compresión) → descargar. El MISMO motor de mejora reusable dentro de los jobs de video (como ya se hace con `audio_enhance`). v1: **denoise sólido (DeepFilterNet/RNNoise, ya listo) + Apollo restore EXPERIMENTAL**.

**Rama:** `feature/audio-section`. Commits español convencional, sin Co-Authored-By. Base: pytest 693 + vitest 298 verdes.

**Construcción PARALELA** contra este contrato fijo: un agente backend, un agente frontend. NO se pisan (backend = `app/**`,`tests/**`; frontend = `frontend/**`). Cada uno `git add` SOLO sus paths.

---

## CONTRATO FIJO (ambos agentes se atan a esto)

### Cadena de procesamiento (standalone y en video)
`input audio → [denoise si se pидió] → [restore si se pidió] → output`. Denoise PRIMERO (limpia ruido), restore DESPUÉS (reconstruye banda perdida por códec). Cada paso opcional e independiente.

### API REST (respuestas camelCase, espejando JobResponse existente)
- `POST /api/v1/audio/jobs` (multipart): `file` (audio), `denoise` (str|None: `"deepfilter"`|`"rnnoise"`), `restore` (str|None: `"apollo"`), `device` (str|None, para restore ONNX). → **202** `{jobId, status, statusUrl, downloadUrl}`.
- `GET /api/v1/audio/jobs/{id}` → **AudioJobResponse**: `{id, status, originalFilename, denoise, restore, device, progressPct(0..100|None), stages(list|None), error(str|None), downloadUrl(str|None)}`.
- `GET /api/v1/audio/jobs/{id}/download` → archivo de audio mejorado (attachment).
- `GET /api/v1/audio/capabilities` → `{denoiseModes: string[] (solo los instalados), restoreAvailable: bool}`. La UI muestra SOLO lo disponible.

### Reuso en video
`POST /api/v1/video/jobs` gana un campo form nuevo `audio_restore` (str|None: `"apollo"`), aplicado al track de audio DESPUÉS de `audio_enhance` (denoise). Validación: requiere `keep_audio` + restore disponible; si no → 400 claro. `VideoJobResponse` expone `audioRestore`.

### Formatos de audio aceptados (standalone)
wav, mp3, flac, m4a, ogg, opus → se decodifican a WAV internamente con el ffmpeg vendored. Output standalone: WAV 44.1kHz (rate de Apollo) si hubo restore, si no conserva el rate de entrada. `MAX_AUDIO_UPLOAD_MB` (default 200).

### Config nueva (`app/config.py`)
- `ENABLE_AUDIO_RESTORE` (bool, default **False** — experimental).
- `APOLLO_RESTORE_MODEL` (str, default `vendor/apollo/apollo.onnx`).
- `AUDIO_RESTORE_CHUNK_SECONDS` (float, default **3.0** — DirectML rompe con T grande; chunk ≤3s).
- `MAX_AUDIO_UPLOAD_MB` (int, default 200).
- `AUDIO_RESTORE_MODES = {"apollo"}` (constante, como `AUDIO_ENHANCE_MODES`).
- Helper `audio_restore_available()` → `enable_audio_restore and Path(apollo_restore_model).exists()`.

---

## Task BACKEND (agente 1)

**Files:** Create `app/services/engines/apollo_restore.py`, `app/services/audio_job_manager.py`, `app/services/audio_pipeline.py`, tests. Modify `app/models.py`, `app/schemas.py`, `app/config.py`, `app/api/routes.py`, `app/main.py`, `app/services/video_job_manager.py`, `app/services/video_upscaler.py`.

**Leer primero:** `app/services/job_manager.py` (patrón manager: queue, workers, device_semaphores, create_job/get_job, retention — MIRROR); `app/services/engines/onnx_upscaler.py` (patrón multi-EP: device `dml:N`→`DmlExecutionProvider(device_id=N)`, `cpu`→`CPUExecutionProvider`, session cache — REUSAR ese patrón); `app/services/engines/audio_enhance.py` (`AudioEnhancer(settings,mode).available()/async run(in_wav,out_wav)` — REUSAR para denoise); `chunked_dml.py` en el port (`Desktop/upflow-audio-spike/apollo-port-code/chunked_dml.py`) para el chunking+overlap-add del restore.

**Produces:**
- `ApolloRestorer(settings)`: `available()` (= `settings.audio_restore_available()`), `async run(input_wav: Path, output_wav: Path, device: str) -> None`. Interno: resample input→44.1kHz mono (ffmpeg o soundfile+scipy), carga `apollo.onnx` con el EP del device (multi-provider, patrón OnnxUpscaler), procesa en chunks de `AUDIO_RESTORE_CHUNK_SECONDS` con Hann overlap-add (0.5s solape) — el modelo es audio→audio 44.1k self-contained (input `audio [1,1,n]`, output `restored [1,1,n]`). NUNCA rompe la app si el modelo falta (available()=False). onnx input name `audio`, output `restored`.
- `AudioJob` dataclass en models.py (source_path, original_filename, denoise, restore, device, id, status, created/started/finished, error, output_path, metadata).
- `audio_pipeline.py`: función/clase que orquesta la cadena denoise→restore reusando `AudioEnhancer` + `ApolloRestorer`, con archivos temp intermedios, limpieza en finally.
- `AudioJobManager` (mirror JobManager): queue bounded, workers = max_concurrent_jobs, device_semaphores compartido (restore usa GPU), create_job (valida denoise∈AUDIO_ENHANCE_MODES si no-None + disponible, restore∈AUDIO_RESTORE_MODES si no-None + disponible; al menos uno de los dos requerido; device válido), get_job, retención vía el sweeper existente. CancelledError-safe, unlink source en finally, task_done — preservar TODA la seguridad del patrón existente.
- Rutas + `get_audio_job_manager` dep + `audio_job_to_response` mapper + `/audio/capabilities`. Schemas `AudioJobResponse`, `AudioCapabilitiesResponse` (camelCase serialization_alias).
- `main.py`: `app.state.audio_job_manager = AudioJobManager(...)` + start()/stop() en el lifespan; el RetentionSweeper barre también audio jobs.
- Video: `audio_restore` en create_video_job + validación + aplicarlo en `video_upscaler` tras el denoise (extraer→denoise→restore→mux). `VideoJobResponse.audioRestore`.
- Config nueva (arriba) + `.env.example` + README (sección "Audio").

**Tests (TDD, pytest):** ApolloRestorer.available() gated (sin modelo/flag → False, sin excepción); run con un onnx fake/mock o el real si está (chunking correcto, output no vacío); create_audio_job valida modos + device + requiere al menos uno; cadena denoise→restore en orden; capabilities refleja lo instalado; 400 si restore pedido y no disponible; job lifecycle (queued→running→completed, download); video con audio_restore valida keep_audio. Full pytest verde (base 693 + nuevos). Usar el device_semaphores real; mock del subproceso/onnx donde convenga para no depender de binarios en CI.

## Task FRONTEND (agente 2)

**Files:** Create `frontend/src/modules/audio/AudioPage.tsx`, `frontend/src/modules/audio/AudioPanel.tsx`, `frontend/src/services/audio.ts`, `frontend/src/hooks/useAudioJob.ts` (+ tests). Modify `frontend/src/App.tsx` (ruta `/audio`), el nav (AppShell/Layout — agregar item "Audio" con icono lucide), `frontend/src/lib/apiTypes.ts`, y `frontend/src/modules/enhance/VideoPanel.tsx` (toggle de restore en la sección de audio).

**Leer primero:** `frontend/src/modules/enhance/ImagePanel.tsx` + `VideoPanel.tsx` (patrón de panel: upload, secciones acordeón, device picker, submit, JobQueue, progreso — MIRROR); `frontend/src/services/*.ts` (wrapper fetch); `frontend/src/hooks/*.ts`; `frontend/src/components/JobQueue.tsx`, `JobDetailModal.tsx`, `AccordionSection.tsx`, `Tooltip.tsx`, `DevicePicker.tsx` (REUSAR); tokens control-room.

**Produces:**
- `services/audio.ts`: `createAudioJob(form)`, `getAudioJob(id)`, `fetchAudioCapabilities()`.
- `useAudioJob.ts`: TanStack Query hooks (crear + poll status), + `useAudioCapabilities`.
- `AudioPanel`: upload de audio (accept audio/*), secciones acordeón: **Denoise** (none/DeepFilterNet/RNNoise — solo los de capabilities), **Restore** (none/Apollo con badge "Experimental" — solo si `restoreAvailable`), **Device** (DevicePicker, para restore). Validación: al menos denoise o restore elegido. Botón "Enhance audio" → crea job → JobQueue + JobDetailModal (progreso). Descarga del resultado. Copy UI en inglés (matchear la app). Tokens control-room, AA, lucide, mono-tabular solo números.
- `AudioPage` + ruta `/audio` en App.tsx + item "Audio" en el nav (icono ej. `AudioWaveform`/`Music`).
- `VideoPanel`: en la sección de audio existente, agregar toggle/select "Restore compression (Apollo — experimental)" que setea `audio_restore` en el submit del video job (solo si keep_audio + restoreAvailable).
- `apiTypes.ts`: `AudioJob`, `AudioCapabilities`, `audioRestore` en el tipo de video.

**Tests (vitest, TDD):** AudioPanel muestra solo modos disponibles (capabilities); requiere al menos un modo; submit arma el form correcto; restore muestra badge experimental y se oculta si no disponible; VideoPanel: toggle de restore aparece con keep_audio + restoreAvailable y setea el campo. No debilitar tests existentes. `npm --prefix frontend test -- --run` + `run build` verdes.

## Task 3 (main thread, yo): smoke real + review adversarial de rama + merge
- Smoke: correr backend, subir un audio, denoise + restore (con el apollo.onnx en vendor/), verificar output; capabilities; video con audio_restore. Frontend build.
- Review adversarial (foco: la cadena nunca rompe la app si falta el modelo; multi-EP correcto; seguridad del manager preservada; validaciones).
- Merge a master. (Modelo apollo.onnx: script de descarga `download-apollo.ps1` + subirlo como asset del release; NO se commitea, es vendored.)

## Self-Review
Cobertura: engine restore multi-EP ✓, audio job pipeline standalone ✓, reuso en video ✓, UI apartado Audio ✓, experimental gated (nunca rompe) ✓, denoise reusa lo existente ✓.
