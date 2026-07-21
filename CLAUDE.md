# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Servicio local de reescalado de imágenes y video (Real-ESRGAN NCNN Vulkan) optimizado para Windows + AMD GPU. FastAPI sirve web UI y REST API desde un solo backend.

## Commands

```powershell
# Setup (venv + pip install -e .)
powershell -ExecutionPolicy Bypass -File .\scripts\setup.ps1

# Descargar binarios vendored (gitignored — obligatorio tras clonar)
powershell -ExecutionPolicy Bypass -File .\scripts\download-realesrgan.ps1
powershell -ExecutionPolicy Bypass -File .\scripts\download-ffmpeg.ps1
powershell -ExecutionPolicy Bypass -File .\scripts\download-rife.ps1  # opcional, solo para FPS boost (ENABLE_INTERPOLATION=true)
powershell -ExecutionPolicy Bypass -File .\scripts\download-deepfilternet.ps1  # opcional, solo para mejora de audio (ENABLE_AUDIO_ENHANCE=true)

# Arrancar servidor
.\.venv\Scripts\uvicorn app.main:app --host 127.0.0.1 --port 8090 --reload

# Tests (pytest con asyncio_mode=auto)
.\.venv\Scripts\python -m pytest
.\.venv\Scripts\python -m pytest tests/test_health.py::test_health_endpoint
```

No hay linter/formatter configurado en el repo.

## Architecture

Todo se cablea en `app/main.py` dentro del `lifespan`: se instancian servicios (StorageService, RealEsrganNcnnEngine, RifeNcnnEngine, MediaTools, JobManager, VideoJobManager, VideoUpscaler, RetentionSweeper) y se cuelgan de `app.state`. Las rutas (`app/api/routes.py`, `app/web/routes.py`) los consumen desde ahí.

**Motor desacoplado**: `app/services/engines/base.py` define `UpscaleEngine` (ABC con `available()` y `run(job)`). `realesrgan_ncnn.py` es la implementación de upscaling: envuelve el ejecutable `vendor/realesrgan/realesrgan-ncnn-vulkan.exe` como subproceso. `rife_ncnn.py` sigue el mismo patrón (no implementa la ABC, expone `run(frames_in, frames_out, source_frame_count, multiplier)`) para interpolar fotogramas con `vendor/rife/rife-ncnn-vulkan.exe`. Para agregar otro motor (p.ej. DirectML), implementar la ABC correspondiente y cambiar la instanciación en `main.py`.

**Segundo motor de interpolación (GMFSS)**: `gmfss_engine.py` expone la misma firma que `rife_ncnn.py` (`run(frames_in, frames_out, source_frame_count, multiplier, *, target_frame_count=None, device=None)`), pero corre por ONNX Runtime/DirectML en vez de un binario NCNN — vendorea `app/services/engines/gmfss/` desde el repo público `santiquiroz/port-gmfss-onnx` (puerto propio de GMFSS_Fortuna a ONNX, primer port ONNX conocido de ese modelo). Es el motor de **máxima calidad, muy lento** (~10x o más que RIFE, medido 0.72-0.73fps@1080p 2x en RX 7800 XT) — opt-in vía `ENABLE_GMFSS` + modelos descargados con `scripts/download-gmfss-onnx.ps1` (no se bundlean en el instalador), seleccionable por job con `interp_engine=rife|gmfss` (`rife` default siempre). `_maybe_interpolate` en `video_upscaler.py` elige el motor según `job.interp_engine`. Cada sesión ONNX debe crearse con `graph_optimization_level=ORT_DISABLE_ALL` (el grafo MetricNet cuelga la GPU en DirectML con la optimización default — hallazgo real, no opcional).

**Colas de jobs**: `JobManager` (imágenes) y `VideoJobManager` (videos) son colas en memoria (`asyncio.Queue` con `maxsize=MAX_QUEUE_SIZE`, rechaza con `QueueFullError`/`429` cuando se llena) + dict de jobs, con `GPU_CONCURRENCY` worker tasks cada una. Ambos managers comparten un único `asyncio.Semaphore(GPU_CONCURRENCY)` (default 1 — no subir sin perfilar VRAM) para serializar el acceso real a la GPU entre imagen y video. Los jobs se pierden al reiniciar; no hay persistencia.

**Pipeline de video** (`video_upscaler.py`): ffprobe → extraer frames PNG con FFmpeg → upscale del lote de frames con el motor → interpolar con RIFE si `fps_multiplier > 1` (`_maybe_interpolate`, requiere `ENABLE_INTERPOLATION=true` y el binario instalado) → re-encode (libx264/libx265) → mux con audio (si `keep_audio` y `audio_enhance` está seteado, el audio pasa antes por `AudioEnhancer` — DeepFilterNet o el filtro `arnndn` de FFmpeg con RNNoise — requiere `ENABLE_AUDIO_ENHANCE=true` y los binarios de `download-deepfilternet.ps1`; si `audio_restore` está seteado, pasa además por Apollo/AudioSR vía `restore_multichannel` — `multichannel_restore.py` — preservando estéreo/surround por M/S en vez de downmixear a mono, con RMS-match por canal). El progreso se reporta vía `job.metadata["stage"]`; el FPS de salida queda en `job.metadata["outputFps"]`. Los threads de x265 se limitan (`FFMPEG_X265_THREADS`) porque libx265 en Windows falla con exceso de threads. Todo subproceso (engine/ffmpeg/rife/deep-filter) corre a través de `run_guarded_process` (`process_runner.py`), que aplica `SUBPROCESS_TIMEOUT` y mata el proceso al cancelar.

**Selección de pistas de audio/subtítulos + formato de salida**: `POST /api/v1/video/analyze` sube el archivo, corre `ffprobe -show_streams` y devuelve `uploadToken` + metadata de pistas de audio (índice/idioma/codec/canales/default) y subtítulos, sin crear un job. `POST /api/v1/video/jobs` acepta ese `uploadToken` en vez de `file` (exactamente uno de los dos), más `audio_track_indices` (lista; la primera es la pista **primaria** que pasa por enhance/restore, el resto se copia con `-c:a copy`) y `keep_subtitles` (copia todas las pistas de subtítulos con `-c:s copy`). Pedir subtítulos o `audio_output_format=auto|flac` con restore activo sube el contenedor a `.mkv` automáticamente si hacía falta (`VideoJobManager._resolve_output_container`, con el motivo en `job.metadata["containerUpgradedReason"]`, nunca en silencio). `audio_output_format` (`auto`/`flac`/`aac`, default `auto`) controla el codec de la pista restaurada; el módulo Audio standalone tiene su propio `output_format` (`wav`/`flac`/`mp3`, default `flac`).

**Retención**: `RetentionSweeper` corre un sweep inmediato al arrancar y luego cada hora; borra archivos en `outputs/` y jobs `completed`/`failed` más viejos que `OUTPUT_TTL_HOURS`. Un fallo en una iteración no detiene las siguientes.

**Catálogos en `app/config.py`**: `MODEL_CATALOG` (modelos con escalas válidas por modelo) y `VIDEO_PROFILE_CATALOG` (perfiles que combinan modelo + escala + codec + CRF + `fps_multiplier` por defecto). Agregar modelos o perfiles = editar esas listas; la UI y `GET /api/v1/engine` los exponen automáticamente. El modelo `realesr-animevideov3` es un preset que se resuelve a `-x2/-x3/-x4` según la escala (`resolve_engine_model_name`).

**Settings**: pydantic-settings lee `.env` (ver `.env.example`). `get_settings()` está cacheado con `lru_cache` — cambios a `.env` requieren reiniciar.

## Layout notes

- `vendor/` (binarios de Real-ESRGAN, FFmpeg, RIFE y DeepFilterNet) y `runtime/` (uploads/outputs/temp/video-work) están gitignored — se crean con los scripts de `scripts/` y en runtime.
- Validación de entrada vive en los job managers (escala permitida, formato, tamaño de imagen vía Pillow, `fps_multiplier` contra `ALLOWED_FPS_MULTIPLIERS`), no en las rutas.
