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

**No hace falta tocar Python ni la consola.** Dos opciones, misma app:

### Opción A: instalador (recomendada — sin Python)

1. Descargá `upflow-setup-v<version>.exe` del [último release](https://github.com/santiquiroz/upflow/releases).
2. Ejecutalo — no pide admin, se instala en tu carpeta de usuario (`%LOCALAPPDATA%\Upflow`) e incluye su propio Python embebido, así que **no necesitás tener Python instalado**.
3. Al terminar, tildá "iniciar Upflow" (o abrilo después desde el acceso directo del escritorio/menú inicio).
4. Esperá la primera descarga (~3-4 GB: motor de upscaling + FFmpeg + RIFE + dependencias de Python; puede tardar varios minutos según tu conexión) — cuando el servidor está listo, el navegador se abre solo en `http://127.0.0.1:8090`. Las siguientes veces arranca al instante.

Desinstalar preserva por defecto tus archivos y modelos (`runtime\`); hay un checkbox opcional durante la instalación para que el desinstalador también los borre. Ver [`installer/README.md`](installer/README.md) para el detalle del instalador.

### Opción B: zip portátil (requiere Python)

1. Descargá el `.zip` del [último release](https://github.com/santiquiroz/upflow/releases) (por ejemplo `upflow-v0.1.0.zip`).
2. Extraelo en cualquier carpeta.
3. Doble click en **`Upflow.bat`**.
4. Esperá la primera descarga de binarios (~1 GB: motor de upscaling + FFmpeg + RIFE; algunos minutos según tu conexión) — cuando el servidor está listo, el navegador se abre solo en `http://127.0.0.1:8090`.

Requiere [Python 3.11+](https://www.python.org/downloads/) instalado y en el `PATH`. Si tenés `winget`, con esto alcanza: `winget install Python.Python.3.12`

**Requisitos (ambas opciones):**

- Windows 10 u 11 de 64 bits.
- Cualquier GPU con soporte Vulkan (NVIDIA, AMD o Intel).

> **¿Tenés una NVIDIA RTX?** Anda exactamente igual — Upflow corre sobre Vulkan, no necesita CUDA ni drivers especiales.

¿Preferís correrlo desde el código fuente, o contribuir al proyecto? Seguí con la sección de abajo.

---

## Qué es Upflow

La mayoría de buenos upscalers son CUDA-only, de código cerrado, o una pila de flags de CLI. Upflow es una **SPA en React + API REST** limpia, construida alrededor de un **motor desacoplado e intercambiable**, corriendo sobre **Real-ESRGAN NCNN + Vulkan** — lo que significa que vuela en **AMD Radeon** en Windows, donde DirectML y CUDA se quedan cortos.

> **Corre en cualquier GPU con Vulkan** (AMD, NVIDIA, Intel). Simplemente está *afinado* para AMD-en-Windows, donde las buenas opciones escasean.

## Características

- 🖼️ **Upscaling de imagen** — arrastrás el archivo, elegís modelo, dispositivo y escala (2×/3×/4× según el modelo), listo. Fotos y anime/line-art por igual, con job en vivo y descarga directa desde la UI.
- 🎬 **Upscaling de video** — pipeline completo con FFmpeg: extraer frames → upscale por lote → re-encodear preservando el audio. Perfiles listos para anime y contenido general, con opciones avanzadas (modelo, escala, códec, preset, CRF, FPS boost, mejora de audio, formato de audio de salida).
- 🎞️ **Selección de pistas de audio y subtítulos** — `POST /api/v1/video/analyze` inspecciona el video subido (pistas de audio con idioma/codec/canales, pistas de subtítulos) antes de crear el job; elegís qué pista(s) de audio conservar (la primera de la lista es la primaria, la única que pasa por enhance/restore) y si querés preservar subtítulos, que sube el contenedor a `.mkv` automáticamente si hacía falta (con aviso en `job.metadata`, nunca en silencio).
- 🌊 **FPS boost con RIFE NCNN Vulkan** — interpolación de fotogramas 2×/3×/4× sobre el video ya reescalado, mismo backend Vulkan (sin CUDA). Se activa por config y aparece como dropdown en el módulo Enhance.
- 🎨 **GMFSS — interpolación de máxima calidad (experimental, opt-in)** — segundo motor de FPS boost, port ONNX propio de [GMFSS_Fortuna](https://github.com/santiquiroz/port-gmfss-onnx) (el mejor modelo de interpolación para anime, corriendo en cualquier GPU DirectX12 sin CUDA). Mucho más lento que RIFE (~10x o más — máxima calidad, no para uso cotidiano), se activa con `ENABLE_GMFSS=true` + `scripts/download-gmfss-onnx.ps1` y se elige por job (`interp_engine=rife|gmfss`, RIFE sigue siendo el default).
- 🔊 **Mejora de audio con IA** — denoise opcional del audio del video con DeepFilterNet (red neuronal) o RNNoise (filtro `arnndn` de FFmpeg), como paso extra del pipeline antes de re-encodear. Se activa por config y se elige por job (`audio_enhance=deepfilter|rnnoise`).
- 🧠 **Módulo Models** — buscá modelos de super-resolución en Hugging Face, instalalos con un click (con polling de progreso) y gestioná los ya instalados; elegí el dispositivo de cómputo (`cpu`/`dml:N`) por default o por job.
- ⚙️ **Módulo Settings** — estado del motor, disponibilidad de ffmpeg, concurrencia de GPU y profundidad de las colas de jobs, todo en vivo.
- 📡 **Módulo Realtime (roadmap)** — vista de estado que explica el plan de interpolación en tiempo real (Fase 7, no implementado todavía) y por qué no es viable hoy — ver [`docs/REALTIME_MODULE.md`](docs/REALTIME_MODULE.md).
- 🧹 **Retención automática** — un sweeper en background borra outputs y jobs terminados más viejos que `OUTPUT_TTL_HOURS`; corre al arrancar y luego cada hora. El disco ya no crece sin límite.
- 🚦 **Cola con límite + concurrencia de GPU compartida** — cada tipo de job (imagen/video) tiene su propia cola acotada (`MAX_QUEUE_SIZE`, responde `429` si se llena) y ambas comparten un único semáforo de GPU (`GPU_CONCURRENCY`) para no saturar la VRAM. La cola global de jobs se ve en vivo desde cualquier módulo.
- 🛡️ **Hardening de subida** — nombres de archivo sanitizados (caracteres inválidos en Windows y nombres reservados como `CON`/`NUL`/`COM1`), timeout + kill automático de subprocesos colgados, validación de formato/tamaño/dimensiones de imagen y video, y un middleware que rechaza requests de escritura (`POST`/`PUT`/`PATCH`/`DELETE`) desde orígenes no permitidos.
- 🧩 **Motor desacoplado** — el backend de upscaling vive detrás de una interfaz (`UpscaleEngine`). NCNN/Vulkan hoy, lo que sea mañana.
- 🔌 **API REST** — encolá jobs y consultá su estado desde cualquier otra app.
- 🏠 **100% local** — tus archivos nunca salen de tu máquina.

## Requisitos

- Windows con una GPU compatible con Vulkan (AMD, NVIDIA o Intel).
- [Python 3.11+](https://www.python.org/) en el `PATH`.
- PowerShell (para correr los scripts de `scripts/`).
- [Node.js 20+](https://nodejs.org/) en el `PATH` — **solo si corrés desde el código fuente** (para compilar la SPA de `frontend/`). El `.zip` de release ya trae `frontend/dist/` compilado, así que los usuarios finales no lo necesitan.

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

# 5. Frontend: compilar la SPA de React (necesario para correr desde codigo fuente,
#    requiere Node.js 20+; ver seccion "Desarrollo del frontend" mas abajo)
cd frontend
npm install
npm run build
cd ..

# 6. (opcional) copiar .env.example a .env y ajustar valores
copy .env.example .env

# 7. Arrancar el servidor
.\.venv\Scripts\uvicorn app.main:app --host 127.0.0.1 --port 8090 --reload
```

Abrí **http://127.0.0.1:8090**.

Todos los binarios de `vendor/` y todo lo de `runtime/` (uploads, outputs, temp, video-work) están en `.gitignore` — se generan localmente con los scripts de arriba y en tiempo de ejecución, nunca se commitean. Lo mismo para `frontend/dist/` y `frontend/node_modules/`: se generan con el paso 5, nunca se commitean.

> **Instalación más pesada de lo habitual:** el paso 1 (`pip install -e .`) instala también `onnxruntime-directml`, `torch` (CPU-only), `spandrel` y `onnx` — dependencias del módulo de modelos HF (ver sección "Modelos" abajo). Sumalas y son ~2-3 GB extra, la mayoría por `torch`. No hace falta ningún paso manual adicional, solo tener espacio en disco y paciencia la primera vez.

## Desarrollo del frontend

FastAPI sirve el build de producción de `frontend/dist/` en `/` (paso 5 arriba). Para desarrollar la UI con hot-reload, corré el backend y el dev server de Vite en paralelo:

```powershell
# Terminal 1: backend (API en :8090)
.\.venv\Scripts\uvicorn app.main:app --host 127.0.0.1 --port 8090 --reload

# Terminal 2: frontend con hot-reload (:5173, hace proxy de /api hacia :8090)
cd frontend
npm install
npm run dev
```

Abrí **http://localhost:5173** durante el desarrollo. `npm run build` genera el bundle de producción en `frontend/dist/` que consume FastAPI; `npm test` corre la suite de vitest.

## Cómo usar

### Web UI

La SPA de React tiene cuatro módulos, accesibles desde la barra lateral:

- **Enhance** (`/`) — imagen y video en la misma pantalla, con tabs:
  - *Imagen*: subís el archivo, elegís modelo, dispositivo de cómputo (`cpu`/`dml:N`) y escala (la lista se filtra automáticamente según lo que soporta cada modelo), formato de salida. Job en vivo con progreso y descarga directa al terminar.
  - *Video*: subís el archivo (se analiza automáticamente con `/video/analyze`) y elegís un perfil (ver tabla de perfiles abajo), con opciones avanzadas para sobreescribir modelo, escala, contenedor, códec, preset, CRF, audio, el dropdown **FPS boost** (Off, o 2×/3×/4×; solo produce resultado si tenés `ENABLE_INTERPOLATION=true` y RIFE instalado — ver más abajo), **mejora de audio** (Off/RNNoise/DeepFilterNet) y **formato de audio de salida** (Auto/FLAC/AAC). Si el video trae más de una pista de audio o subtítulos embebidos, aparece un **selector de pistas**: tildá cuáles pistas de audio conservar (la primera tildada es la primaria, la única que pasa por enhance/restore) y si querés preservar los subtítulos (sube el contenedor a `.mkv` automáticamente si hacía falta).
- **Models** (`/models`) — buscador de modelos de super-resolución en Hugging Face, instalación con un click (con polling de progreso hasta `done`/`error`), lista de modelos instalados con borrado, y selección de dispositivo por default.
- **Settings** (`/settings`) — estado del motor (disponibilidad, ffmpeg), concurrencia de GPU y profundidad de las colas de jobs, en vivo.
- **Realtime** (`/realtime`) — página de roadmap: explica el plan de interpolación en tiempo real (Fase 7) y por qué el frame generation en vivo no es viable todavía en Windows sin driver hooks propietarios.

Un panel de **cola de jobs** global (imagen + video) con progreso en vivo está disponible desde cualquier módulo.

### API REST

Todos los endpoints viven bajo `/api/v1`. Los campos de formulario (subida) van en snake_case; las respuestas JSON usan camelCase (p. ej. subís `video_codec` como campo del form y la respuesta lo devuelve como `videoCodec`).

| Método | Endpoint | Descripción |
|---|---|---|
| `GET` | `/api/v1/health` | Healthcheck: motor activo, `gpuConcurrency` y profundidad de ambas colas |
| `GET` | `/api/v1/engine` | Estado del motor, si FFmpeg está disponible, catálogo de modelos y de perfiles de video |
| `GET` | `/api/v1/devices` | Dispositivos de cómputo disponibles (`cpu`, `dml:0`, `dml:1`...) y `defaultDeviceId` efectivo |
| `GET` | `/api/v1/models` | Catálogo completo de modelos instalados (builtin + los instalados desde Hugging Face) |
| `GET` | `/api/v1/models/search?q=` | Busca modelos de super-resolución en Hugging Face Hub |
| `POST` | `/api/v1/models/install` | Instala un modelo desde HF por `repo_id` (`202`, devuelve `install_id`) |
| `GET` | `/api/v1/models/install/{install_id}` | Estado de una instalación en curso (`pending`/`downloading`/`converting`/`done`/`error`) |
| `DELETE` | `/api/v1/models/{model_id}` | Borra un modelo instalado (`204`; `403` si es builtin, `404` si no existe) |
| `POST` | `/api/v1/jobs` | Crea un job de imagen (`202`) |
| `GET` | `/api/v1/jobs/{job_id}` | Estado de un job de imagen (`404` si no existe) |
| `GET` | `/api/v1/jobs/{job_id}/download` | Descarga el resultado (`404` si no existe, `409` si aún no terminó) |
| `POST` | `/api/v1/video/analyze` | Analiza un video subido (pistas de audio y subtítulos vía `ffprobe`) sin crear un job; devuelve `uploadToken` reutilizable en `POST /api/v1/video/jobs` |
| `POST` | `/api/v1/video/jobs` | Crea un job de video (`202`) |
| `GET` | `/api/v1/video/jobs/{job_id}` | Estado de un job de video, incluye `metadata` (stage, fps, dimensiones, `outputFps`) |
| `GET` | `/api/v1/video/jobs/{job_id}/download` | Descarga el video resultante (`404`/`409` igual que arriba) |

**Crear un job de imagen** — campos de formulario: `file` (requerido), `model_name` (default `realesrgan-x4plus`, ignorado si se manda `model_id`), `model_id` (opcional: id de un modelo ONNX instalado desde HF, ver sección Modelos), `device` (opcional: `cpu`/`dml:N`, ver sección Dispositivos; omitido = `DEFAULT_DEVICE`), `scale` (default `4`), `output_format` (`png`/`jpg`/`jpeg`/`webp`, default `png`):

```bash
curl -X POST http://127.0.0.1:8090/api/v1/jobs \
  -F "file=@input.png" \
  -F "model_name=realesrgan-x4plus-anime" \
  -F "scale=4" \
  -F "output_format=png"

# con un modelo ONNX instalado desde Hugging Face, en la GPU dml:0
curl -X POST http://127.0.0.1:8090/api/v1/jobs \
  -F "file=@input.png" \
  -F "model_id=sceneworks--real-esrgan-onnx" \
  -F "device=dml:0" \
  -F "output_format=png"
```

**Crear un job de video** — campos de formulario: `file` **o** `upload_token` (exactamente uno de los dos: `file` sube el video directo, `upload_token` reutiliza el análisis previo de `POST /api/v1/video/analyze` sin volver a subir el archivo), `profile_key` (default `anime-balanced-2x`), y overrides opcionales del perfil: `model_name`, `model_id` (modelo ONNX instalado desde HF, ver sección Modelos), `device` (`cpu`/`dml:N`, ver sección Dispositivos), `scale`, `output_container` (`mp4`/`mkv`), `video_codec` (`libx264`/`libx265`), `video_preset` (`medium`/`slow`/`veryslow`), `crf` (`10`-`28`), `keep_audio`, `fps_multiplier` (`1` = sin boost, o uno de `ALLOWED_FPS_MULTIPLIERS`), `audio_enhance` (`deepfilter`/`rnnoise`, omitido = sin mejora; requiere `keep_audio=true` y `ENABLE_AUDIO_ENHANCE=true` — ver "Cómo activar la mejora de audio" abajo), `audio_track_indices` (índices de pista separados por coma, ej. `0,2`; omitido = ffmpeg elige la pista default como hoy — la primera pista de la lista es la **primaria**, la única que pasa por enhance/restore, el resto se copia sin procesar), `keep_subtitles` (default `false`; copia todas las pistas de subtítulos detectadas — sube el contenedor a `.mkv` automáticamente si hacía falta, con aviso en `job.metadata.containerUpgradedReason`), `audio_output_format` (`auto`/`flac`/`aac`, default `auto`: con `audio_restore` activo sube a FLAC lossless + `.mkv` automático, si no mantiene el comportamiento actual):

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

# analizar primero (pistas de audio/subtítulos), despues crear el job reusando el upload
curl -X POST http://127.0.0.1:8090/api/v1/video/analyze -F "file=@input.mkv"
# -> {"uploadToken": "...", "audioTracks": [...], "subtitleTracks": [...]}
curl -X POST http://127.0.0.1:8090/api/v1/video/jobs \
  -F "upload_token=<uploadToken>" \
  -F "profile_key=anime-balanced-2x" \
  -F "audio_track_indices=0,2" \
  -F "keep_subtitles=true"
```

**Consultar y descargar:**

```bash
curl http://127.0.0.1:8090/api/v1/video/jobs/<job_id>
curl -OJ http://127.0.0.1:8090/api/v1/video/jobs/<job_id>/download
```

## Progreso y ETA

La cola de jobs global muestra una barra de progreso en vivo para cada job; hacer click en un job abre un **modal de detalle** con:

- **Stepper de etapas** — cada tipo de job tiene sus propias etapas ponderadas (video: `probing` → `extracting_frames` → `extracting_audio`/`enhancing_audio`/`restoring_audio` (si aplica) → `upscaling_frames` → `interpolating_frames` (si el FPS boost está activo) → `encoding_video`; imagen: `validating` → `upscaling`), cada una con estado `pending`/`active`/`done`.
- **Frames X / Y** — en video, cuenta de frames procesados sobre el total real (extraídos del contenedor con `ffprobe`, o derivados de duración × fps cuando el origen es VFR y no trae `nb_frames`). En imagen, solo aparece para modelos **ONNX con tiling** (`ONNX_TILE_SIZE` activo en un lado más grande que el tile): cuenta tiles procesados sobre el total, actualizado entre cada tile de la grilla de inferencia. Los modelos **builtin NCNN** (subprocess único, sin conteo intermedio) y las imágenes **ONNX que caben en un solo tile** se quedan en etapas coarse (`validating`/`upscaling` sin frames) — a propósito: no hay conteo honesto que reportar ahí, así que no se inventa uno.
- **ETA** — solo se muestra cuando hay suficiente señal para ser confiable (frames/tiles con denominador real y throughput medido); si no, se omite en vez de mostrar un número inventado.

El progreso combinado (`progressPct` en la respuesta del job) es un promedio ponderado: cada etapa completada suma su peso completo, la etapa activa suma su peso proporcional a la fracción interna (frames o tiles procesados), y nunca retrocede.

Los jobs largos (videos de muchos frames, modelos ONNX pesados) ya **no se matan por un timeout fijo de duración**: un *stall watchdog* cancela el job solo si deja de haber progreso real (sin frames nuevos) durante `FRAME_STALL_TIMEOUT_SECONDS` (default 900s), no por exceder un techo de reloj arbitrario.

## Modelos

### Modelos builtin (NCNN Vulkan)

| Modelo | Ideal para | Escalas |
|---|---|---|
| `realesrgan-x4plus` | Fotos, imágenes generales | 4× |
| `realesrgan-x4plus-anime` | Anime fijo, ilustración, line art | 4× |
| `realesr-animevideov3-x2` / `-x3` / `-x4` | Fotogramas de video anime | 2× / 3× / 4× |
| `realesr-animevideov3` | Preset automático (resuelve a x2/x3/x4 según la escala pedida) | 2×–4× |

Estos modelos vienen empaquetados con el motor (`scripts/download-realesrgan.ps1`), corren siempre sobre Vulkan y **no aceptan `device=cpu`** (ver sección Dispositivos).

### Instalar modelos desde Hugging Face

Además del catálogo builtin, Upflow puede instalar cualquier modelo de super-resolución publicado en Hugging Face y correrlo con el motor ONNX Runtime + DirectML:

1. Buscar en el buscador de la web UI (sección Modelos) o con `GET /api/v1/models/search?q=<texto>` — pega directo a la Hub API de Hugging Face.
2. Instalar con `POST /api/v1/models/install` (`{"repo_id": "org/nombre-repo"}`) — devuelve un `install_id` para hacer polling en `GET /api/v1/models/install/{install_id}` hasta `status=done`.
3. Una vez instalado, el modelo aparece en `GET /api/v1/models` con `kind=onnx` y puede pasarse como `model_id` al crear un job de imagen o video.

**Formatos soportados:**

- **`.onnx` directo** — se copia tal cual a `MODELS_DIR`, sin conversión. Camino rápido (ej. `SceneWorks/real-esrgan-onnx`).
- **`.pth` / `.safetensors` (arquitecturas comunitarias tipo ESRGAN/Compact/SRVGG)** — se detecta la arquitectura vía [Spandrel](https://github.com/chaiNNer-org/spandrel) y se convierte a ONNX automáticamente con `torch.onnx.export` antes de dejarlo instalado. Requiere las dependencias `torch`/`spandrel`/`onnx` (ver "Instalación paso a paso" — se instalan solas con `pip install -e .`).

Si el repo de HF no expone un archivo compatible, el estado del install job pasa a `error` con el detalle.

Un modelo instalado puede borrarse con `DELETE /api/v1/models/{model_id}` (los 6 builtins están protegidos: devuelve `403`). El límite de tamaño de descarga es `MAX_MODEL_DOWNLOAD_MB` (default 2048 MB).

## Dispositivos

`GET /api/v1/devices` enumera los dispositivos de cómputo disponibles para el motor ONNX/DirectML:

- **`cpu`** — siempre presente. Válido **solo** para modelos ONNX instalados desde Hugging Face (`kind=onnx`). **Inválido para los 6 modelos builtin** (`kind=builtin-ncnn`): corren siempre sobre Vulkan y pedir `device=cpu` con un `model_id` builtin devuelve `400` ("Device 'cpu' is not supported for builtin model ... (requires a Vulkan GPU device)").
- **`dml:N`** — una GPU DirectML-capable, `N` = índice de adaptador DXGI (0, 1, 2...). El nombre real de cada GPU viene de `IDXGIFactory1::EnumAdapters1` (Windows, vía `ctypes`, sin dependencia extra) y `N` es exactamente el `device_id` que se le pasa a `DmlExecutionProvider` de onnxruntime — mapeo verificado empíricamente (ver `.superpowers/sdd/sp1-task-8-smoke-report.md`).

Es normal que una misma GPU física aparezca más de una vez (ej. `dml:0` y `dml:2` apuntando ambos a la misma dGPU) en máquinas con configuraciones de gráficos híbridos: Windows/DXGI expone LUIDs de adaptador distintos para el mismo silicio. Cada índice sigue siendo un `device_id` válido y funcional para DirectML — no es un bug, es comportamiento real de DXGI.

El dispositivo por defecto se controla con `DEFAULT_DEVICE` en `.env` (default `dml:0`); si el dispositivo configurado no está disponible, cae automáticamente a `cpu`.

**Selección de GPU en máquinas multi-adaptador — alcance de la garantía:** para modelos ONNX/HF (`kind=onnx`), `dml:N` se pasa directo como `device_id` a `DmlExecutionProvider`, que lo resuelve contra la misma lista ordenada por DXGI — el mapeo es exacto (verificado empíricamente, ver `.superpowers/sdd/sp1-task-8-smoke-report.md`). Para los modelos builtin (`kind=builtin-ncnn`), en cambio, `dml:N` se traduce a `-g N` (índice de dispositivo físico Vulkan del binario `realesrgan-ncnn-vulkan.exe`) — DXGI y Vulkan **no garantizan el mismo orden de enumeración** en una máquina con más de un adaptador. Solo el default de un único dGPU (`dml:0` → `-g 0`) está verificado end-to-end; en hardware multi-GPU, la selección de un `dml:N` con `N > 0` para un modelo builtin es best-effort, no exacta.

## Multi-GPU (colas por dispositivo + auto-router opcional)

Cada dispositivo tiene su **propia cola de concurrencia**: un job toma un permiso del semáforo de *su* dispositivo (`app/services/device_semaphores.py`), así que jobs en dispositivos distintos corren **en paralelo** en vez de serializarse detrás de un semáforo global. Un video reescalando en `dml:0` y una imagen en la iGPU `dml:1` (o en `cpu`) avanzan a la vez.

- **`PER_DEVICE_GPU_CONCURRENCY`** (default `1`) — jobs simultáneos **por GPU**. Imagen y video comparten el semáforo de esa GPU. No lo subas sin perfilar VRAM: dos jobs pesados en la misma GPU compiten por memoria.
- **`CPU_CONCURRENCY`** (default `2`) — jobs simultáneos en `cpu` (modelos ONNX). El `cpu` no compite con las GPUs.
- **`MAX_CONCURRENT_JOBS`** (default `4`) — workers por manager (imagen y video por separado). Debe **superar** la cantidad de dispositivos que quieras correr en paralelo, o no habrá worker libre para desencolar el segundo job.

**Auto-router opcional** (`ENABLE_AUTO_ROUTE`, default `False`): con el toggle activado, los jobs sin dispositivo fijo (o con `device="auto"`) se reparten al **primer dispositivo compatible libre** en vez de encolarse todos en `dml:0`. La compatibilidad depende del modelo:

| Tipo de modelo | Dispositivos compatibles |
|---|---|
| `builtin-ncnn` (los 6 builtin) | solo GPUs Vulkan (`dml:N`) — **nunca `cpu`** |
| `onnx` (instalados desde HF) | `cpu` o cualquier `dml:N` |

Si al desencolar todos los dispositivos compatibles están ocupados, el job espera al que **se libere primero** (sin bloqueo head-of-line: no se casa con un dispositivo saturado dejando otro libre ocioso). Si no existe ningún dispositivo compatible (ej. modelo builtin en una máquina sin GPU Vulkan), la creación del job responde `400`. El toggle vive en **Settings** de la web UI y se puede elegir **"Auto"** en el selector de dispositivo por job.

Caso de uso central: reescalar una **temporada completa** de anime encolando todos los episodios con auto-router on → se reparten entre las GPUs disponibles y terminan antes que en serie.

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
| `MAX_UPLOAD_MB` | `50` | Tamaño máximo de subida para imágenes (MB) |
| `MAX_VIDEO_UPLOAD_MB` | `2048` | Tamaño máximo de subida para videos (MB) |
| `MAX_IMAGE_PIXELS` | `120000000` | Límite de píxeles (ancho × alto) para evitar decompression bombs |
| `PER_DEVICE_GPU_CONCURRENCY` | `1` | Jobs simultáneos **por GPU** (semáforo por dispositivo; imagen y video comparten el de esa GPU) — no subirlo sin perfilar VRAM. Ver [Multi-GPU](#multi-gpu-colas-por-dispositivo--auto-router-opcional) |
| `CPU_CONCURRENCY` | `2` | Jobs simultáneos en `cpu` (modelos ONNX); no compite con las GPUs |
| `MAX_CONCURRENT_JOBS` | `4` | Workers por manager (imagen y video por separado); debe superar la cantidad de dispositivos a correr en paralelo |
| `ENABLE_AUTO_ROUTE` | `False` | Auto-router: reparte jobs sin dispositivo fijo (o `device="auto"`) al primer dispositivo compatible libre. Ver [Multi-GPU](#multi-gpu-colas-por-dispositivo--auto-router-opcional) |
| `SUBPROCESS_TIMEOUT` | `86400` | Backstop absoluto (24h) para matar cualquier subproceso; NO es el mecanismo real (ver `FRAME_STALL_TIMEOUT_SECONDS`) |
| `FRAME_STALL_TIMEOUT_SECONDS` | `900` | Watchdog real: mata la etapa solo si no produce frames/bytes nuevos por este tiempo (15 min); se reinicia con cada frame nuevo |
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
| `ENABLE_GMFSS` | `false` | Habilita el motor GMFSS (máxima calidad, muy lento); requiere `ENABLE_INTERPOLATION=true` además y haber corrido `download-gmfss-onnx.ps1` |
| `GMFSS_MODEL_DIR` | `vendor/gmfss` | Carpeta con los 4 `.onnx` + `manifest.json` del port GMFSS (+ `fusionnet_fp16.onnx` opcional) |
| `DEEPFILTER_BINARY` | `vendor/deepfilternet/deep-filter.exe` | Ruta al binario CLI de DeepFilterNet |
| `RNNOISE_MODEL` | `vendor/deepfilternet/models/sh.rnnn` | Ruta al modelo `.rnnn` usado por el filtro `arnndn` de FFmpeg |
| `ENABLE_AUDIO_ENHANCE` | `false` | Habilita la mejora de audio (`audio_enhance=deepfilter\|rnnoise`); requiere haber corrido `download-deepfilternet.ps1` |
| `DEFAULT_DEVICE` | `dml:0` | Dispositivo preferido (`cpu`/`dml:N`) para el motor ONNX/DirectML; cae a `cpu` si no está disponible (ver sección Dispositivos) |
| `MODELS_DIR` | `models` | Carpeta donde se guardan los modelos ONNX instalados desde Hugging Face (relativa a la raíz del proyecto) |
| `HF_TOKEN` | *(vacío)* | Token de Hugging Face opcional, para buscar/descargar modelos privados o evitar rate limiting anónimo |
| `MAX_MODEL_DOWNLOAD_MB` | `2048` | Tamaño máximo permitido para un archivo de modelo descargado desde HF (MB) |
| `ONNX_TILE_SIZE` | `256` | Tamaño de tile (px) para inferencia ONNX por partes, con blend de 16px de solape; `0` desactiva el tiling (imagen completa de una pasada) |

## Cómo activar el FPS boost (RIFE)

El FPS boost está deshabilitado por defecto. Para activarlo:

```powershell
# 1. Descargar el motor RIFE NCNN Vulkan (fork TNTwise, incluye varios modelos v4.x)
powershell -ExecutionPolicy Bypass -File .\scripts\download-rife.ps1

# 2. En .env, habilitar la interpolación
ENABLE_INTERPOLATION=true
```

El dropdown "FPS boost" siempre está visible en la UI de video (con las opciones de `ALLOWED_FPS_MULTIPLIERS`), pero solo funciona una vez activado: pedir un `fps_multiplier > 1` (por UI o directo en `POST /api/v1/video/jobs`) sin `ENABLE_INTERPOLATION=true` o sin el binario de RIFE instalado devuelve `400`.

## Cómo activar GMFSS (interpolación de máxima calidad)

GMFSS es un segundo motor de FPS boost — mucha más calidad que RIFE en anime, pero **~10x o más lento** (medido 0.72-0.73 fps @1080p 2x en una RX 7800 XT; pensalo como "máxima calidad, muy lento", no como reemplazo de RIFE para uso diario). Deshabilitado por defecto:

```powershell
# 1. Descargar los modelos ONNX de GMFSS (~55MB, port propio: santiquiroz/port-gmfss-onnx)
powershell -ExecutionPolicy Bypass -File .\scripts\download-gmfss-onnx.ps1

# 2. En .env, habilitar GMFSS (además de ENABLE_INTERPOLATION=true, arriba)
ENABLE_GMFSS=true
```

Con ambos motores disponibles, el selector RIFE/GMFSS aparece en el dropdown de FPS boost de la UI; por API se elige con `interp_engine=rife|gmfss` en `POST /api/v1/video/jobs` (default siempre `rife`, GMFSS es opt-in por job).

## Cómo activar la mejora de audio (DeepFilterNet / RNNoise)

La mejora de audio está deshabilitada por defecto. Para activarla:

```powershell
# 1. Descargar el binario de DeepFilterNet (~26 MB) + el modelo .rnnn para el filtro arnndn de FFmpeg (~300 KB)
powershell -ExecutionPolicy Bypass -File .\scripts\download-deepfilternet.ps1

# 2. En .env, habilitar la mejora de audio
ENABLE_AUDIO_ENHANCE=true
```

Con eso activado, un job de video con `keep_audio=true` puede pedir `audio_enhance=deepfilter` (red neuronal DeepFilterNet3, mejor calidad, más lento) o `audio_enhance=rnnoise` (filtro `arnndn` de FFmpeg, más liviano). Pedir `audio_enhance` sin `keep_audio=true`, sin `ENABLE_AUDIO_ENHANCE=true` o sin los binarios instalados devuelve `400`. Omitir `audio_enhance` deja el audio original intacto (remux con `-c:a copy`).

## Apartado de Audio (mejora standalone + restauración de compresión)

Además de imagen y video, Upflow tiene un **apartado de Audio** propio (ruta `/audio`): subís un archivo de audio (wav/mp3/flac/m4a/ogg/opus), elegís la mejora y descargás el resultado. La cadena es `entrada → [denoise] → [restore] → salida`, cada paso opcional.

- **Denoise** — quita ruido: `deepfilter` (DeepFilterNet3) o `rnnoise`. Es el mismo motor que ya se usa en video (ver sección anterior); requiere `ENABLE_AUDIO_ENHANCE=true` + `download-deepfilternet.ps1`.
- **Restore (EXPERIMENTAL)** — dos motores, elegibles por job:
  - `apollo`: reconstruye la banda de agudos perdida por compresión de códec (audio de WhatsApp/Telegram/redes). Rápido y liviano (~74 MB). Requiere `ENABLE_AUDIO_RESTORE=true` + `scripts/download-apollo.ps1`.
  - `audiosr`: **super-resolución de audio general por difusión latente** (cualquier banda → 48 kHz, UNet de 258M params). Techo de calidad muy superior a Apollo pero ~2 min de proceso por minuto de audio en GPU (50 pasos DDIM con CFG). Port ONNX propio — el primero conocido de AudioSR: [santiquiroz/port-audiosr-onnx](https://github.com/santiquiroz/port-audiosr-onnx). Requiere `ENABLE_AUDIOSR=true` + `scripts/download-audiosr-onnx.ps1` (~2.6 GB).

  Ambos motores son ONNX **multi-provider** (corren en cualquier GPU DirectX12 —AMD/NVIDIA/Intel— o CPU, igual que los modelos de imagen HF). Si un modelo no está instalado, ese modo simplemente no aparece — la app nunca se rompe por esto. **Preservan estéreo/surround**: en vez de downmixear a mono antes de restaurar, decodifican Mid/Side (restauran solo el Mid, el Side queda intacto) en estéreo, y en 5.1/7.1 restauran frente/rears por par M/S + centro directo + LFE intacto, con RMS-match por canal contra el original al final; un layout de canales no reconocido cae a mono con warning explícito (nunca en silencio).
- **Formato de salida** — `output_format: wav|flac|mp3`, default **`flac`** (sin pérdida, ~50% más liviano que WAV). `wav` para compatibilidad con editores viejos, `mp3` solo si el tamaño del archivo importa más que la calidad.

```powershell
# Restore experimental: descargar el modelo Apollo (~74 MB) y habilitarlo
powershell -ExecutionPolicy Bypass -File .\scripts\download-apollo.ps1
# en .env:  ENABLE_AUDIO_RESTORE=true
```

API: `POST /api/v1/audio/jobs` (multipart: `file`, `denoise?`, `restore?`, `output_format?` default `flac`, `device?`) → 202; `GET /api/v1/audio/jobs/{id}` (estado + progreso), `.../download` (resultado), `GET /api/v1/audio/capabilities` (qué motores están instalados; `restoreModes` lista los modos listos). El mismo `restore=apollo|audiosr` se puede pedir en un job de video vía el campo `audio_restore` (con `keep_audio=true`), aplicado después del denoise; el formato de salida de esa pista restaurada se controla con `audio_output_format` (ver "Crear un job de video" arriba).

> **Nota experimental:** el restore es un port ONNX del modelo Apollo (ver `docs/` y la guía del port). Funciona y es multi-provider, pero la calidad de reconstrucción y el rendimiento GPU aún se están evaluando — por eso va detrás de un flag y con badge "Experimental" en la UI.

## Actualizaciones

Upflow chequea **en silencio** si hay una release más nueva en GitHub y, si la hay, muestra un banner discreto arriba de la UI ("New version X available") con link a la release. El chequeo es opcional y a prueba de fallos: si no hay red, hay timeout o GitHub responde con rate-limit (`403`), el endpoint igual devuelve `200` con `updateAvailable=false` y un campo `error` — el banner simplemente no aparece y **la app nunca se rompe por el chequeo**. El resultado se cachea en memoria (`UPDATE_CHECK_TTL_SECONDS`, default 3600s) para no pegarle a la API de GitHub en cada request. El banner se puede descartar por versión: una vez descartado, esa versión no vuelve a aparecer, pero una versión más nueva sí.

- `GET /api/v1/update-check` → `{ currentVersion, latestVersion, updateAvailable, releaseUrl, publishedAt, checkedAt, error }` (camelCase). Acepta `?force=true` para saltar el cache.

Si un chequeo falla justo cuando el cache expira, el banner **no desaparece**: mientras hubo un resultado bueno en la sesión, el servicio sigue sirviéndolo (un parpadeo de red no oculta una actualización real). Un error sin ningún resultado bueno previo se cachea solo `UPDATE_ERROR_RETRY_SECONDS` (default 300s) para reintentar pronto, no por el TTL completo.

- `GET /api/v1/update-check` → `{ currentVersion, latestVersion, updateAvailable, releaseUrl, publishedAt, checkedAt, error }` (camelCase). Acepta `?force=true` para saltar el cache.

**Reusar el patrón en otro proyecto:** el chequeo no tiene nada hardcodeado a Upflow, así que se reusa cambiando dos variables de `.env`:

1. `UPDATE_REPO` → el repo destino, con formato `owner/nombre` (default `santiquiroz/upflow`).
2. `UPDATE_PACKAGE_NAME` → el nombre del paquete instalado cuya versión se compara contra el `tag_name` de la release (`importlib.metadata.version(...)`, con fallback al `[project] version` del `pyproject.toml`). También define el User-Agent del request.

Toggles: `UPDATE_CHECK_ENABLED` (default `true`) apaga el chequeo por completo, y `UPDATE_API_TIMEOUT_SECONDS` (default `5.0`) acota cuánto espera a GitHub.

## Tests

Backend (pytest):

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

Frontend (vitest):

```powershell
cd frontend
npm install
npm test              # correr toda la suite una vez
npm run test:watch    # modo watch
```

## Arquitectura

```text
Browser
        │
   SPA de React (frontend/, build de Vite servido desde frontend/dist/)
        │
   FastAPI (app/)  ──  sirve la SPA en "/" (fallback de rutas cliente) + routers REST en /api/v1
        │
   Cola de jobs por tipo (imagen/video)  ──  workers async + semáforo de GPU compartido
        │
   ┌────┴──────────────────┐
   │                        │
Motor de imagen        Pipeline de video (FFmpeg)
(Real-ESRGAN            extraer frames → upscale por lote →
 NCNN Vulkan)            interpolar con RIFE/GMFSS (opcional) → re-encode + audio
```

El motor de upscaling vive detrás de una interfaz `UpscaleEngine` (`app/services/engines/base.py`), así que el backend Vulkan es un componente reemplazable. Un `RetentionSweeper` en background borra outputs y jobs vencidos según `OUTPUT_TTL_HOURS`. La SPA (`frontend/`) se compila una sola vez a `frontend/dist/` — en release, `scripts/package-release.ps1` corre `npm ci && npm run build` antes de empaquetar el `.zip`; en desarrollo, se compila a mano o se corre con hot-reload (`npm run dev`, ver "Desarrollo del frontend" arriba).

## Roadmap

- [x] 🌊 FPS boost con RIFE NCNN Vulkan (2×/3×/4×, activable por config)
- [x] 🎨 GMFSS — segundo motor de interpolación de máxima calidad (ONNX, port propio [santiquiroz/port-gmfss-onnx](https://github.com/santiquiroz/port-gmfss-onnx)), opt-in, ~10x o más lento que RIFE
- [x] 🧹 Limpieza automática de disco + retención de jobs (TTL)
- [x] 🔊 Mejora de audio con IA (DeepFilterNet / RNNoise) — denoise como etapa opcional del pipeline, activable por config
- [x] 🧠 Modelos HF + selección de dispositivo — instalar cualquier modelo de super-resolución de Hugging Face (`.onnx` directo o `.pth`/`.safetensors` vía conversión Spandrel) y elegir `cpu`/`dml:N` por job
- [ ] 📝 Subtítulos con IA (whisper.cpp) — generación + traducción, muxeados como pista blanda
- [ ] 🎚️ Slider calidad ↔ velocidad (presets Fast/Balanced/Best mapeados a los knobs reales de cada motor)
- [ ] 📦 Modo batch por temporada (subida múltiple, progreso agregado)

**Fuera de alcance (por ahora):** interpolación en tiempo real estilo Lossless Scaling. Requiere captura del swapchain DirectX en vivo, arquitectónicamente incompatible con una app de archivos FastAPI/Python en el proceso principal. Diseño de Fase 7 (fork/vendor de Magpie como proceso helper separado, sin implementar todavía) documentado en **[`docs/REALTIME_MODULE.md`](docs/REALTIME_MODULE.md)**. Hasta que eso exista, usá [Lossless Scaling](https://store.steampowered.com/app/993090/Lossless_Scaling/) o [Magpie](https://github.com/Blinue/Magpie) (open source) — Upflow se mantiene como pipeline offline de máxima calidad.

Ver el plan de ingeniería completo en **[`docs/IMPLEMENTATION_PLAN.md`](docs/IMPLEMENTATION_PLAN.md)** y la investigación detrás de estos ítems en **[`docs/RESEARCH_ANIME_SUITE.md`](docs/RESEARCH_ANIME_SUITE.md)**.

## Contribuir

Los PRs son bienvenidos. Ver [`CONTRIBUTING.md`](CONTRIBUTING.md) y el [plan de implementación](docs/IMPLEMENTATION_PLAN.md) para saber dónde ayuda más.

## Licencia

[MIT](LICENSE) © 2026 Santiago Quiroz. Hacé lo que quieras con esto.

---

<div align="center">
<sub>Construido con FastAPI · Real-ESRGAN · RIFE · NCNN · Vulkan · FFmpeg</sub>
</div>
