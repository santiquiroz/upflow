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

# Arrancar servidor
.\.venv\Scripts\uvicorn app.main:app --host 127.0.0.1 --port 8090 --reload

# Tests (pytest con asyncio_mode=auto)
.\.venv\Scripts\python -m pytest
.\.venv\Scripts\python -m pytest tests/test_health.py::test_health_endpoint
```

No hay linter/formatter configurado en el repo.

## Architecture

Todo se cablea en `app/main.py` dentro del `lifespan`: se instancian servicios (StorageService, RealEsrganNcnnEngine, MediaTools, JobManager, VideoJobManager, VideoUpscaler) y se cuelgan de `app.state`. Las rutas (`app/api/routes.py`, `app/web/routes.py`) los consumen desde ahí.

**Motor desacoplado**: `app/services/engines/base.py` define `UpscaleEngine` (ABC con `available()` y `run(job)`). `realesrgan_ncnn.py` es la única implementación: envuelve el ejecutable `vendor/realesrgan/realesrgan-ncnn-vulkan.exe` como subproceso. Para agregar otro motor (p.ej. DirectML), implementar la ABC y cambiar la instanciación en `main.py`.

**Colas de jobs**: `JobManager` (imágenes) y `VideoJobManager` (videos) son colas en memoria (`asyncio.Queue` + dict de jobs) con un worker task cada una. Un `asyncio.Semaphore(GPU_CONCURRENCY)` serializa el acceso a GPU (default 1 — no subir sin perfilar VRAM). Los jobs se pierden al reiniciar; no hay persistencia.

**Pipeline de video** (`video_upscaler.py`): ffprobe → extraer frames PNG con FFmpeg → upscale del lote de frames con el motor → re-encode (libx264/libx265) → mux con audio AAC. El progreso se reporta vía `job.metadata["stage"]`. Los threads de x265 se limitan (`FFMPEG_X265_THREADS`) porque libx265 en Windows falla con exceso de threads.

**Catálogos en `app/config.py`**: `MODEL_CATALOG` (modelos con escalas válidas por modelo) y `VIDEO_PROFILE_CATALOG` (perfiles que combinan modelo + escala + codec + CRF). Agregar modelos o perfiles = editar esas listas; la UI y `GET /api/v1/engine` los exponen automáticamente. El modelo `realesr-animevideov3` es un preset que se resuelve a `-x2/-x3/-x4` según la escala (`resolve_engine_model_name`).

**Settings**: pydantic-settings lee `.env` (ver `.env.example`). `get_settings()` está cacheado con `lru_cache` — cambios a `.env` requieren reiniciar.

## Layout notes

- `vendor/` (binarios de Real-ESRGAN y FFmpeg) y `runtime/` (uploads/outputs/temp/video-work) están gitignored — se crean con los scripts de `scripts/` y en runtime.
- Validación de entrada vive en los job managers (escala permitida, formato, tamaño de imagen vía Pillow), no en las rutas.
