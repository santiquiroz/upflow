<div align="center">

# ⚡ Upflow

### Upscaling con IA **+** interpolación de fotogramas — open source, nativo en Vulkan, pensado para AMD.

*Como Lossless Scaling, pero para tus archivos: reescala en lote anime y fotos, reconstruye video fotograma a fotograma y súbele los FPS con IA — todo en tu propia GPU, sin nube, sin depender de CUDA.*

[![License: MIT](https://img.shields.io/badge/License-MIT-22c55e.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Backend: Vulkan](https://img.shields.io/badge/backend-Vulkan-AC162C.svg?logo=vulkan&logoColor=white)](https://github.com/xinntao/Real-ESRGAN-ncnn-vulkan)
[![PRs welcome](https://img.shields.io/badge/PRs-welcome-8b5cf6.svg)](CONTRIBUTING.md)

</div>

---

## 🚀 Instalación rápida (usuarios)

**No hace falta tocar Python ni la consola.** Cuatro pasos:

1. Descargá el `.zip` del [último release](https://github.com/santiquiroz/upflow/releases) (por ejemplo `upflow-v0.1.0.zip`).
2. Extraelo en cualquier carpeta.
3. Doble click en **`Upflow.bat`**.
4. Esperá la primera descarga de binarios (~1 GB: motor de upscaling + FFmpeg + RIFE) — cuando el servidor está listo, el navegador se abre solo en `http://127.0.0.1:8090`.

**Requisitos:**

- Windows 10 u 11.
- Cualquier GPU con soporte Vulkan (NVIDIA, AMD o Intel).
- [Python 3.11+](https://www.python.org/downloads/) instalado y en el `PATH`. Si tenés `winget`, con esto alcanza: `winget install Python.Python.3.12`

> **¿Tenés una NVIDIA RTX?** Anda exactamente igual — Upflow corre sobre Vulkan, no necesita CUDA ni drivers especiales.

¿Preferís correrlo desde el código fuente, o contribuir al proyecto? Seguí con la sección de abajo.

---

## Qué es Upflow

La mayoría de buenos upscalers son CUDA-only, de código cerrado, o una pila de flags de CLI. Upflow es una **web UI + API REST** limpia, construida alrededor de un **motor desacoplado e intercambiable**, corriendo sobre **Real-ESRGAN NCNN + Vulkan** — lo que significa que vuela en **AMD Radeon** en Windows, donde DirectML y CUDA se quedan cortos.

> **Corre en cualquier GPU con Vulkan** (AMD, NVIDIA, Intel). Simplemente está *afinado* para AMD-en-Windows, donde las buenas opciones escasean.

## Características

- 🖼️ **Upscaling de imagen** — arrastrás el archivo, elegís modelo y escala (2×/3×/4× según el modelo), listo. Fotos y anime/line-art por igual.
- 🎬 **Upscaling de video** — pipeline completo con FFmpeg: extraer frames → upscale por lote → re-encodear preservando el audio. Perfiles listos para anime y contenido general.
- 🌊 **FPS boost con RIFE NCNN Vulkan** — interpolación de fotogramas 2×/3×/4× sobre el video ya reescalado, mismo backend Vulkan (sin CUDA). Se activa por config y aparece como dropdown en la UI.
- 🔊 **Mejora de audio con IA** — denoise opcional del audio del video con DeepFilterNet (red neuronal) o RNNoise (filtro `arnndn` de FFmpeg), como paso extra del pipeline antes de re-encodear. Se activa por config y se elige por job (`audio_enhance=deepfilter|rnnoise`).
- 🧹 **Retención automática** — un sweeper en background borra outputs y jobs terminados más viejos que `OUTPUT_TTL_HOURS`; corre al arrancar y luego cada hora. El disco ya no crece sin límite.
- 🚦 **Cola con límite + concurrencia de GPU compartida** — cada tipo de job (imagen/video) tiene su propia cola acotada (`MAX_QUEUE_SIZE`, responde `429` si se llena) y ambas comparten un único semáforo de GPU (`GPU_CONCURRENCY`) para no saturar la VRAM.
- 🛡️ **Hardening de subida** — nombres de archivo sanitizados (caracteres inválidos en Windows y nombres reservados como `CON`/`NUL`/`COM1`), timeout + kill automático de subprocesos colgados, validación de formato/tamaño/dimensiones de imagen y video, y un middleware que rechaza requests de escritura (`POST`/`PUT`/`PATCH`/`DELETE`) desde orígenes no permitidos.
- 🧩 **Motor desacoplado** — el backend de upscaling vive detrás de una interfaz (`UpscaleEngine`). NCNN/Vulkan hoy, lo que sea mañana.
- 🔌 **API REST** — encolá jobs y consultá su estado desde cualquier otra app.
- 🏠 **100% local** — tus archivos nunca salen de tu máquina.

## Requisitos

- Windows con una GPU compatible con Vulkan (AMD, NVIDIA o Intel).
- [Python 3.11+](https://www.python.org/) en el `PATH`.
- PowerShell (para correr los scripts de `scripts/`).

## Instalación paso a paso

```powershell
git clone https://github.com/santiquiroz/upflow.git
cd upflow

# 1. Entorno Python (crea .venv e instala el paquete en modo editable)
powershell -ExecutionPolicy Bypass -File .\scripts\setup.ps1

# 2. Motor de upscaling: Real-ESRGAN NCNN Vulkan (obligatorio)
powershell -ExecutionPolicy Bypass -File .\scripts\download-realesrgan.ps1

# 3. FFmpeg (obligatorio solo si vas a usar upscaling de video)
powershell -ExecutionPolicy Bypass -File .\scripts\download-ffmpeg.ps1

# 4. RIFE NCNN Vulkan (opcional, solo si querés el FPS boost — ver más abajo)
powershell -ExecutionPolicy Bypass -File .\scripts\download-rife.ps1

# 5. (opcional) copiar .env.example a .env y ajustar valores
copy .env.example .env

# 6. Arrancar el servidor
.\.venv\Scripts\uvicorn app.main:app --host 127.0.0.1 --port 8090 --reload
```

Abrí **http://127.0.0.1:8090**.

Todos los binarios de `vendor/` y todo lo de `runtime/` (uploads, outputs, temp, video-work) están en `.gitignore` — se generan localmente con los scripts de arriba y en tiempo de ejecución, nunca se commitean.

## Cómo usar

### Web UI

- **Imagen**: subís el archivo, elegís modelo (la lista de escalas se filtra automáticamente según lo que soporta cada modelo) y formato de salida.
- **Video**: subís el archivo y elegís un perfil (ver tabla de perfiles abajo). En "Advanced options" podés sobreescribir modelo, escala, contenedor, códec, preset, CRF, audio y el dropdown **FPS boost** (Off, o 2×/3×/4×; solo produce resultado si tenés `ENABLE_INTERPOLATION=true` y RIFE instalado — ver más abajo).
- La sección **Status** hace polling del job cada 2 segundos y muestra el JSON completo más el `stage` actual y, para video, el FPS de salida.

### API REST

Todos los endpoints viven bajo `/api/v1`. Los campos de formulario (subida) van en snake_case; las respuestas JSON usan camelCase (p. ej. subís `video_codec` como campo del form y la respuesta lo devuelve como `videoCodec`).

| Método | Endpoint | Descripción |
|---|---|---|
| `GET` | `/api/v1/health` | Healthcheck: motor activo, `gpuConcurrency` y profundidad de ambas colas |
| `GET` | `/api/v1/engine` | Estado del motor, si FFmpeg está disponible, catálogo de modelos y de perfiles de video |
| `GET` | `/api/v1/devices` | Dispositivos de cómputo disponibles (`cpu`, `dml:0`, `dml:1`...) y `defaultDeviceId` efectivo |
| `POST` | `/api/v1/jobs` | Crea un job de imagen (`202`) |
| `GET` | `/api/v1/jobs/{job_id}` | Estado de un job de imagen (`404` si no existe) |
| `GET` | `/api/v1/jobs/{job_id}/download` | Descarga el resultado (`404` si no existe, `409` si aún no terminó) |
| `POST` | `/api/v1/video/jobs` | Crea un job de video (`202`) |
| `GET` | `/api/v1/video/jobs/{job_id}` | Estado de un job de video, incluye `metadata` (stage, fps, dimensiones, `outputFps`) |
| `GET` | `/api/v1/video/jobs/{job_id}/download` | Descarga el video resultante (`404`/`409` igual que arriba) |

**Crear un job de imagen** — campos de formulario: `file` (requerido), `model_name` (default `realesrgan-x4plus`), `scale` (default `4`), `output_format` (`png`/`jpg`/`jpeg`/`webp`, default `png`):

```bash
curl -X POST http://127.0.0.1:8090/api/v1/jobs \
  -F "file=@input.png" \
  -F "model_name=realesrgan-x4plus-anime" \
  -F "scale=4" \
  -F "output_format=png"
```

**Crear un job de video** — campos de formulario: `file` (requerido), `profile_key` (default `anime-balanced-2x`), y overrides opcionales del perfil: `model_name`, `scale`, `output_container` (`mp4`/`mkv`), `video_codec` (`libx264`/`libx265`), `video_preset` (`medium`/`slow`/`veryslow`), `crf` (`10`-`28`), `keep_audio`, `fps_multiplier` (`1` = sin boost, o uno de `ALLOWED_FPS_MULTIPLIERS`), `audio_enhance` (`deepfilter`/`rnnoise`, omitido = sin mejora; requiere `keep_audio=true` y `ENABLE_AUDIO_ENHANCE=true` — ver "Cómo activar la mejora de audio" abajo):

```bash
curl -X POST http://127.0.0.1:8090/api/v1/video/jobs \
  -F "file=@input.mp4" \
  -F "profile_key=anime-balanced-2x" \
  -F "fps_multiplier=2"

# con mejora de audio (requiere ENABLE_AUDIO_ENHANCE=true y haber corrido download-deepfilternet.ps1)
curl -X POST http://127.0.0.1:8090/api/v1/video/jobs \
  -F "file=@input.mp4" \
  -F "profile_key=anime-balanced-2x" \
  -F "keep_audio=true" \
  -F "audio_enhance=deepfilter"
```

**Consultar y descargar:**

```bash
curl http://127.0.0.1:8090/api/v1/video/jobs/<job_id>
curl -OJ http://127.0.0.1:8090/api/v1/video/jobs/<job_id>/download
```

## Modelos

| Modelo | Ideal para | Escalas |
|---|---|---|
| `realesrgan-x4plus` | Fotos, imágenes generales | 4× |
| `realesrgan-x4plus-anime` | Anime fijo, ilustración, line art | 4× |
| `realesr-animevideov3-x2` / `-x3` / `-x4` | Fotogramas de video anime | 2× / 3× / 4× |
| `realesr-animevideov3` | Preset automático (resuelve a x2/x3/x4 según la escala pedida) | 2×–4× |

## Perfiles de video

| Perfil | Categoría | Modelo | Escala | Códec | Preset | CRF |
|---|---|---|---|---|---|---|
| `general-balanced-4x` | General | `realesrgan-x4plus` | 4× | `libx264` | `medium` | 18 |
| `general-hq-4x` | General | `realesrgan-x4plus` | 4× | `libx265` | `slow` | 17 |
| `anime-balanced-2x` (default) | Anime | `realesr-animevideov3-x2` | 2× | `libx264` | `medium` | 17 |
| `anime-quality-3x` | Anime | `realesr-animevideov3-x3` | 3× | `libx265` | `slow` | 16 |
| `anime-max-detail-4x` | Anime | `realesr-animevideov3-x4` | 4× | `libx265` | `slow` | 15 |

Cualquier campo del perfil puede sobreescribirse por request (ver "Crear un job de video" arriba). El catálogo completo vive en `app/config.py` (`MODEL_CATALOG` / `VIDEO_PROFILE_CATALOG`) — agregar un modelo o perfil ahí lo expone automáticamente en la web UI y en `GET /api/v1/engine`.

## Configuración

Todas las variables leen de `.env` (ver [`.env.example`](.env.example) con los defaults y comentarios). `get_settings()` cachea el resultado — reiniciá el servidor después de cambiar `.env`.

| Variable | Default | Descripción |
|---|---|---|
| `APP_NAME` | `Upflow` | Nombre interno del proceso FastAPI (`.env.example` lo sobreescribe a `Image Upscaler AMD`) |
| `APP_HOST` | `127.0.0.1` | Host de bind de uvicorn |
| `APP_PORT` | `8090` | Puerto de bind de uvicorn |
| `WEB_TITLE` | `Upflow` | Título mostrado en la web UI (`.env.example` lo sobreescribe a `AMD Image Upscaler`) |
| `MAX_UPLOAD_MB` | `50` | Tamaño máximo de subida para imágenes (MB) |
| `MAX_VIDEO_UPLOAD_MB` | `2048` | Tamaño máximo de subida para videos (MB) |
| `MAX_IMAGE_PIXELS` | `120000000` | Límite de píxeles (ancho × alto) para evitar decompression bombs |
| `GPU_CONCURRENCY` | `1` | Jobs simultáneos en GPU; semáforo compartido entre imagen y video — no subirlo sin perfilar VRAM |
| `CPU_FALLBACK_WORKERS` | `2` | Hilos de carga/guardado de frames para Real-ESRGAN NCNN en el pipeline de video |
| `SUBPROCESS_TIMEOUT` | `3600` | Segundos antes de matar cualquier subproceso (engine, FFmpeg, RIFE) |
| `FFMPEG_BINARY` | `vendor/ffmpeg/bin/ffmpeg.exe` | Ruta al binario de FFmpeg |
| `FFPROBE_BINARY` | `vendor/ffmpeg/bin/ffprobe.exe` | Ruta al binario de ffprobe |
| `FFMPEG_DECODE_THREADS` | `12` | Hilos para extraer frames del video de entrada |
| `FFMPEG_ENCODE_THREADS` | `24` | Hilos para re-encodear con `libx264` |
| `FFMPEG_X265_THREADS` | `8` | Hilos para re-encodear con `libx265` (limitado: falla en Windows con exceso de threads) |
| `RUNTIME_DIR` | `runtime` | Carpeta para uploads/outputs/temp/video-work (relativa a la raíz del proyecto, funciona sin importar el CWD) |
| `ENGINE` | `realesrgan-ncnn` | Identificador del motor de upscaling activo |
| `ENGINE_BINARY` | `vendor/realesrgan/realesrgan-ncnn-vulkan.exe` | Ruta al binario del motor |
| `ENGINE_MODELS_DIR` | `vendor/realesrgan/models` | Carpeta de modelos del motor |
| `DEFAULT_MODEL` | `realesrgan-x4plus` | Modelo preseleccionado en la UI y en `POST /api/v1/jobs` |
| `DEFAULT_SCALE` | `4` | Escala preseleccionada |
| `ALLOWED_SCALES` | `2,3,4` | Escalas permitidas por la API (lista separada por comas) |
| `DEFAULT_VIDEO_PROFILE` | `anime-balanced-2x` | Perfil de video preseleccionado en la UI |
| `OUTPUT_TTL_HOURS` | `24` | Horas antes de borrar outputs y jobs terminados (el sweep corre cada hora) |
| `ALLOWED_ORIGINS` | *(derivado de `APP_HOST`/`APP_PORT`; ej. `http://127.0.0.1:8090,http://localhost:8090`)* | Orígenes permitidos para requests que cambian estado (`POST`/`PUT`/`PATCH`/`DELETE`). Si no se define, se deriva automáticamente del `APP_HOST`/`APP_PORT` configurados; fijarlo explícitamente para sobreescribir |
| `MAX_QUEUE_SIZE` | `20` | Tamaño máximo de cada cola de jobs (imagen y video por separado); responde `429` si se llena |
| `RIFE_BINARY` | `vendor/rife/rife-ncnn-vulkan.exe` | Ruta al binario de RIFE NCNN Vulkan |
| `RIFE_MODELS_DIR` | `vendor/rife/models` | Carpeta de modelos de RIFE |
| `RIFE_MODEL` | `rife-v4.6` | Modelo RIFE usado para interpolar (recomendado, general-purpose) |
| `ENABLE_INTERPOLATION` | `false` | Habilita el FPS boost; requiere haber corrido `download-rife.ps1` |
| `ALLOWED_FPS_MULTIPLIERS` | `2,3,4` | Multiplicadores de FPS permitidos por la API (lista separada por comas) |
| `DEEPFILTER_BINARY` | `vendor/deepfilternet/deep-filter.exe` | Ruta al binario CLI de DeepFilterNet |
| `RNNOISE_MODEL` | `vendor/deepfilternet/models/sh.rnnn` | Ruta al modelo `.rnnn` usado por el filtro `arnndn` de FFmpeg |
| `ENABLE_AUDIO_ENHANCE` | `false` | Habilita la mejora de audio (`audio_enhance=deepfilter\|rnnoise`); requiere haber corrido `download-deepfilternet.ps1` |

## Cómo activar el FPS boost (RIFE)

El FPS boost está deshabilitado por defecto. Para activarlo:

```powershell
# 1. Descargar el motor RIFE NCNN Vulkan (fork TNTwise, incluye varios modelos v4.x)
powershell -ExecutionPolicy Bypass -File .\scripts\download-rife.ps1

# 2. En .env, habilitar la interpolación
ENABLE_INTERPOLATION=true
```

El dropdown "FPS boost" siempre está visible en la UI de video (con las opciones de `ALLOWED_FPS_MULTIPLIERS`), pero solo funciona una vez activado: pedir un `fps_multiplier > 1` (por UI o directo en `POST /api/v1/video/jobs`) sin `ENABLE_INTERPOLATION=true` o sin el binario de RIFE instalado devuelve `400`.

## Cómo activar la mejora de audio (DeepFilterNet / RNNoise)

La mejora de audio está deshabilitada por defecto. Para activarla:

```powershell
# 1. Descargar el binario de DeepFilterNet (~26 MB) + el modelo .rnnn para el filtro arnndn de FFmpeg (~300 KB)
powershell -ExecutionPolicy Bypass -File .\scripts\download-deepfilternet.ps1

# 2. En .env, habilitar la mejora de audio
ENABLE_AUDIO_ENHANCE=true
```

Con eso activado, un job de video con `keep_audio=true` puede pedir `audio_enhance=deepfilter` (red neuronal DeepFilterNet3, mejor calidad, más lento) o `audio_enhance=rnnoise` (filtro `arnndn` de FFmpeg, más liviano). Pedir `audio_enhance` sin `keep_audio=true`, sin `ENABLE_AUDIO_ENHANCE=true` o sin los binarios instalados devuelve `400`. Omitir `audio_enhance` deja el audio original intacto (remux con `-c:a copy`).

## Tests

```powershell
# instalar dependencias de desarrollo (pytest, pytest-asyncio) una sola vez
.\.venv\Scripts\python -m pip install -e ".[dev]"

# correr toda la suite
.\.venv\Scripts\python -m pytest

# un archivo o test puntual
.\.venv\Scripts\python -m pytest tests/test_health.py::test_health_endpoint

# con cobertura (requiere pytest-cov: pip install pytest-cov)
.\.venv\Scripts\python -m pytest --cov=app --cov-report=term-missing
```

## Arquitectura

```text
Browser / cliente API
        │
   FastAPI (app/)  ──  web UI (Jinja) + routers REST
        │
   Cola de jobs por tipo (imagen/video)  ──  workers async + semáforo de GPU compartido
        │
   ┌────┴──────────────────┐
   │                        │
Motor de imagen        Pipeline de video (FFmpeg)
(Real-ESRGAN            extraer frames → upscale por lote →
 NCNN Vulkan)            interpolar con RIFE (opcional) → re-encode + audio
```

El motor de upscaling vive detrás de una interfaz `UpscaleEngine` (`app/services/engines/base.py`), así que el backend Vulkan es un componente reemplazable. Un `RetentionSweeper` en background borra outputs y jobs vencidos según `OUTPUT_TTL_HOURS`.

## Roadmap

- [x] 🌊 FPS boost con RIFE NCNN Vulkan (2×/3×/4×, activable por config)
- [x] 🧹 Limpieza automática de disco + retención de jobs (TTL)
- [x] 🔊 Mejora de audio con IA (DeepFilterNet / RNNoise) — denoise como etapa opcional del pipeline, activable por config
- [ ] 📝 Subtítulos con IA (whisper.cpp) — generación + traducción, muxeados como pista blanda
- [ ] 🎚️ Slider calidad ↔ velocidad (presets Fast/Balanced/Best mapeados a los knobs reales de cada motor)
- [ ] 📦 Modo batch por temporada (subida múltiple, progreso agregado)

**Fuera de alcance:** interpolación en tiempo real estilo Lossless Scaling. Requiere captura del swapchain DirectX en vivo, arquitectónicamente incompatible con una app de archivos FastAPI/Python. Para eso, usá [Lossless Scaling](https://store.steampowered.com/app/993090/Lossless_Scaling/) o [Magpie](https://github.com/Blinue/Magpie) (open source) — Upflow se mantiene como pipeline offline de máxima calidad.

Ver el plan de ingeniería completo en **[`docs/IMPLEMENTATION_PLAN.md`](docs/IMPLEMENTATION_PLAN.md)** y la investigación detrás de estos ítems en **[`docs/RESEARCH_ANIME_SUITE.md`](docs/RESEARCH_ANIME_SUITE.md)**.

## Contribuir

Los PRs son bienvenidos. Ver [`CONTRIBUTING.md`](CONTRIBUTING.md) y el [plan de implementación](docs/IMPLEMENTATION_PLAN.md) para saber dónde ayuda más.

## Licencia

[MIT](LICENSE) © 2026 Santiago Quiroz. Hacé lo que quieras con esto.

---

<div align="center">
<sub>Construido con FastAPI · Real-ESRGAN · RIFE · NCNN · Vulkan · FFmpeg</sub>
</div>
