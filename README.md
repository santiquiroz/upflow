<div align="center">

# ⚡ Upflow

### Tu estudio multimedia con IA, entero en tu computadora.

*Reescalá, restaurá, separá voces, transcribí, doblá, generá imágenes y video, y armá piezas 3D — todo en tu propia GPU. Sin nube, sin cuentas, sin subir un solo archivo. Y sin CUDA: anda en AMD, Intel y NVIDIA por igual.*

[![License: MIT](https://img.shields.io/badge/License-MIT-22c55e.svg)](LICENSE)
[![Descargar](https://img.shields.io/badge/⬇_Descargar-instalador_Windows-2563eb.svg)](https://github.com/santiquiroz/upflow/releases/latest)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![GPU: DirectML + Vulkan](https://img.shields.io/badge/GPU-AMD_·_Intel_·_NVIDIA-AC162C.svg?logo=amd&logoColor=white)](#dispositivos)
[![PRs welcome](https://img.shields.io/badge/PRs-welcome-8b5cf6.svg)](CONTRIBUTING.md)

</div>

---

## Qué podés hacer

Todo corre local. Cada función se baja aparte, cuando la usás por primera vez, así que instalás
la app y no 40 GB de modelos.

### 🎬 Video

| | |
|---|---|
| **Reescalar** | Subí la resolución de un video viejo o comprimido. Perfiles listos para anime y para imagen real. |
| **Más fluido** | Duplicá, triplicá o cuadruplicá los cuadros por segundo. Detecta los cortes de escena para no inventar cuadros fantasma entre dos planos distintos. |
| **Subtítulos** | Sacá la transcripción con tiempos, o devolvé el video con los subtítulos adentro o quemados en la imagen. En otro idioma si querés. |
| **Doblaje** | El video hablado en otro idioma, con el audio original conservado como segunda pista. |

### 🖼️ Imagen

| | |
|---|---|
| **Reescalar** | Fotos y anime, de a una o por lote, 2× a 4×. |
| **Borrar cosas** | Pintá lo que sobra —o tocá el objeto para seleccionarlo solo— y desaparece. |
| **Generar** | Texto a imagen e imagen a imagen con Stable Diffusion, SDXL o Flux. |

### 🔊 Audio y voz

| | |
|---|---|
| **Karaoke y pistas** | Separá la voz de la instrumental para cantar encima, o quedate solo con la voz. |
| **Limpiar** | Ruido, eco y reverberación, encadenables en una sola pasada. |
| **Restaurar** | Recuperá una grabación vieja o de baja calidad; hay un modo que directamente reinventa el agudo que se perdió. |
| **Nivelar** | Volumen al estándar de entrega de Spotify, YouTube o emisión (EBU R128). |
| **Convertir** | FLAC a MP3 y demás, sin tocar nada más del archivo. |
| **Transcribir** | Audio o video a texto, con tiempos. |
| **Hablar** | Escribís y la app lo dice, en varias voces. |
| **Cambiar de voz** | Le das una grabación y una muestra, y devuelve lo mismo con esa otra voz. |

### 🧱 Impresión 3D

| | |
|---|---|
| **¿Se imprime?** | Soltá un STL y te dice si sale bien en tu impresora y qué arreglarle. Sin instalar nada. |
| **Piezas con medidas** | Tubos, tacos, placas con agujeros, escuadras. Escribís las cotas y sale la pieza exacta, ya verificada. |
| **Generar formas** | Describís algo —o le das una foto— y sale una malla. Para formas, no para piezas que tienen que encajar. |
| **Reparar** | Cierra los agujeros de una malla rota, y vuelve a medir en vez de decirte que quedó bien. |

### ✨ Además

- **Generar video** desde un texto o una imagen, local.
- **Descargar** de internet lo que vas a procesar, sin salir de la app.
- **Tiempo real**: reescalá en vivo lo que estés mirando o jugando, en una ventana superpuesta.
- **API REST y servidor MCP**: encolá trabajos desde otro programa, o dejá que un agente de IA use la app como herramienta.
- **Varios usuarios** con permisos y cuotas, si la compartís en tu red.

> ¿Querés los números, los límites medidos y el porqué de cada decisión?
> Está todo en **[docs/FEATURES.md](docs/FEATURES.md)**.

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
- Cualquier GPU con Vulkan o DirectX 12 — AMD, Intel o NVIDIA. Varias funciones corren
  también en el procesador, más lento.

> **¿Tenés una NVIDIA RTX?** Anda exactamente igual: Upflow usa Vulkan y DirectML, así que no
> necesita CUDA ni drivers especiales. Si querés, después se le puede sumar TensorRT-RTX.

¿Preferís correrlo desde el código fuente, o contribuir al proyecto? Seguí con la sección de abajo.

---

## Qué es Upflow

Empezó siendo un reescalador para GPUs AMD, porque casi todo lo bueno era CUDA-only, de código
cerrado, o una pila de flags de consola. Creció hasta ser lo que usás cuando tenés **un archivo y
querés que quede mejor**: video, foto, audio, voz, texto o una pieza para imprimir.

Tres cosas no cambiaron desde el primer día:

- **Nada sale de tu máquina.** No hay cuenta, no hay nube, no hay subida. Tus archivos son tuyos.
- **Tu placa alcanza.** Corre por Vulkan y DirectML, así que anda igual en AMD, Intel y NVIDIA —
  y en el procesador cuando hace falta. CUDA es opcional, nunca un requisito.
- **Nada se decide en silencio.** Si algo salió por un camino más lento, si hubo que cambiar el
  formato o si una mejora falló, el trabajo te lo dice al terminar, con el detalle a la vista.

Por dentro es una SPA en React sobre una API REST en FastAPI, con los motores detrás de
interfaces: cambiar el modelo que hace el trabajo es una entrada en una tabla, no una cirugía.

## Requisitos

- Windows con una GPU compatible con Vulkan o DirectX 12 (AMD, Intel o NVIDIA).
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

La SPA de React se navega **por tarea**: la raíz pregunta qué querés hacer y te
lleva a la pantalla que corresponde, con el estado preseleccionado.

> **Qué se instala:** el asistente del instalador tiene una pantalla de
> componentes donde elegís las funciones extra —generar fotogramas, quitar ruido
> con IA, restaurar agudos— cada una con su tamaño. Vienen todas tildadas
> (perfil "Completa"), y hay un perfil "Mínima" que deja solo imagen y video. Dos
> paquetes no se ofrecen porque sin ellos la app no hace nada: el motor de
> upscaling y ffmpeg.
>
> Lo que no instales **no se pierde**: la pantalla de Tasks lo muestra con un
> botón que corre el mismo script de descarga, en contexto y con la explicación de
> para qué sirve. Ese es el mejor momento para decidir, no el de la instalación.
>
> El zip portable no tiene asistente, así que baja todo en el primer arranque. Para
> automatizar hay `-InstallAll` y `-SkipOptional` en el launcher.

- **Tasks** (`/`) — el árbol de capacidades, resuelto contra tu máquina: cuatro
  dominios (video, imágenes, audio, generar) y sus capacidades. Una capacidad
  lista te lleva a su pantalla; una que le falta un paquete ofrece bajarlo con
  un click (corre el mismo `scripts/download-*.ps1` que antes se corría a
  mano); y las que todavía no existen se muestran **inertes, con el motivo
  escrito**, bajo un encabezado de mapa de ruta. El status sale de mirar el
  disco y el registro, así que borrar una carpeta de `vendor/` a mano se refleja
  al instante.
- **Enhance** (`/enhance`, `/enhance/image`, `/enhance/video`) — imagen y video, con tabs:
  - *Imagen*: subís el archivo, elegís modelo, dispositivo de cómputo (`cpu`/`dml:N`) y escala (la lista se filtra automáticamente según lo que soporta cada modelo), formato de salida. Job en vivo con progreso y descarga directa al terminar.
  - *Video*: subís el archivo (se analiza automáticamente con `/video/analyze`) y elegís un perfil, que **rellena una pila de pasos** con lo que el job va a hacer de verdad (reescalar, interpolar, audio, subtítulos). Podés quitar y agregar pasos; el orden se muestra pero no se reordena, porque el backend lo tiene fijo y ofrecer reordenar sería un control que miente. Hay opciones avanzadas para sobreescribir modelo, escala, contenedor, códec, preset, CRF, audio, el dropdown **FPS boost** (Off, o 2×/3×/4×; solo produce resultado si tenés `ENABLE_INTERPOLATION=true` y RIFE instalado — ver más abajo), **mejora de audio** (Off/RNNoise/DeepFilterNet) y **formato de audio de salida** (Auto/FLAC/AAC). Si el video trae más de una pista de audio o subtítulos embebidos, aparece un **selector de pistas**: tildá cuáles pistas de audio conservar (la primera tildada es la primaria, la única que pasa por enhance/restore) y si querés preservar los subtítulos (sube el contenedor a `.mkv` automáticamente si hacía falta).
- **Models** (`/models`) — buscador de modelos de super-resolución en Hugging Face con **compatibilidad detectada** en cada resultado (si trae `.onnx` se instala directo; si trae pesos de PyTorch se convierten con Spandrel; si está restringido o no tiene pesos, lo dice antes de que aprietes instalar), instalación con un click con polling de progreso, lista de modelos instalados con borrado, y selección de dispositivo por default.
- **Settings** (`/settings`) — estado del motor (disponibilidad, ffmpeg), concurrencia de GPU y profundidad de las colas de jobs, en vivo, y **selector de idioma** (español / inglés, se recuerda en el equipo).
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
| `GET` | `/api/v1/models/search?q=` | Busca modelos de super-resolución en Hugging Face Hub, con compatibilidad detectada por resultado (sin requests extra: sale de la metadata que ya viene) |
| `GET` | `/api/v1/models/preflight?repoId=` | Antes de instalar un upscaler: veredicto de compatibilidad, tamaño real de la descarga, disco libre, VRAM libre por dispositivo y RAM. No estima pico de VRAM a propósito — un upscaler hace tiling, así que el factor de estimación de difusión no aplica |
| `GET` | `/api/v1/capabilities/tree` | El árbol de lo que la app puede hacer, resuelto contra esta máquina: `available`, `needs_setup` (con el paquete que falta) o `not_implemented` (con el motivo) |
| `POST` | `/api/v1/capabilities/{id}/provision` | Baja el paquete que le falta a una capacidad corriendo su `scripts/download-*.ps1` → 202 |
| `GET` | `/api/v1/capabilities/provision/{jobId}` | Estado de esa descarga |
| `POST` | `/api/v1/video/jobs` | Acepta `target_height` (opcional): se pide una RESOLUCION de salida en vez de un multiplicador. La app elige el escalado entero mas chico que la alcance y redimensiona a la medida exacta; si la fuente ya llega, no corre el modelo |
| `POST` | `/api/v1/generation/init-image` | Sube la imagen de partida para imagen a imagen y devuelve su token (201). Va aparte del job para que `POST /generation/jobs` siga siendo JSON |
| `GET` | `/api/v1/asr/models/search?q=` | Busca modelos de reconocimiento de voz en Hugging Face. El filtro por TAG es lo que decide que un repo es de ASR: los nombres de archivo no alcanzan para distinguirlo de otro modelo de audio |
| `POST` | `/api/v1/asr/models/install` | Instala un modelo de ASR: baja el par encoder/decoder no fusionado mas su metadata (~257 MB para whisper-tiny) → 202 |
| `GET` | `/api/v1/asr/models/install/{install_id}` | Estado de esa instalacion |
| `POST` | `/api/v1/transcribe/jobs` | Transcribe un audio a texto (multipart: `file`, `model_id`, `language?`, `device?`) → 202 |
| `GET` | `/api/v1/transcribe/jobs/{id}` | Estado del job. El TEXTO viaja en la respuesta; `.../download` da el .txt |
| `GET` | `/api/v1/audio/voice-catalog` | Los pasos de la cadena de mejora de voz **en su orden causal** y los destinos de entrega con sus números de loudness publicados |
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

**Listar trabajos** — las siete familias listan con el MISMO contrato, que es lo que vuelve seguro exponerlo en multiusuario: por defecto devuelve **solo los propios**, y `?all=true` devuelve los de todos pero exige el permiso `jobs:read_all` (sin él, `403`). Los terminados siguen apareciendo hasta que la poda los retira.

| Familia | Endpoint |
|---|---|
| Imagen | `GET /api/v1/jobs` |
| Video | `GET /api/v1/video/jobs` |
| Audio | `GET /api/v1/audio/jobs` |
| Generación | `GET /api/v1/generation/jobs` |
| Transcripción | `GET /api/v1/transcribe/jobs` |
| Descargas | `GET /api/v1/download/jobs` |
| 3D | `GET /api/v1/print/generate` |

Es lo que usa la interfaz para recuperar la cola al recargar el navegador: sin listado, un trabajo en curso seguía corriendo en el servidor pero se perdía de vista para siempre.

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

### Instalar modelos de generación (text-to-image)

Los modelos de generación tienen su propio buscador y su propio camino de instalación, con tres diferencias respecto del de upscalers.

**No hace falta saber el `repo_id`.** Con el buscador vacío, la sección Modelos muestra los pipelines text-to-image más descargados de Hugging Face. Cada tarjeta trae un badge de compatibilidad calculado de la metadata real del repo, sin descargar nada:

| Badge | Qué significa |
|---|---|
| **Listo** | Trae ONNX para todos sus componentes: se instala directo |
| **Requiere conversión** | Solo pesos PyTorch: se exporta a ONNX localmente (necesita `torch`, tarda) |
| **Gated** | Acceso restringido: hay que configurar `HF_TOKEN` y aceptar la licencia del repo |
| **Incompatible** | No es un pipeline diffusers (le falta `model_index.json`) |

**Precisión elegible.** En los repos que publican ambas variantes se puede elegir entre fp16 y fp32. La elección define **las dos cosas**: qué archivos se bajan y en qué precisión queda el ONNX exportado. fp16 pesa la mitad, usa menos VRAM y corre más rápido en DirectML; fp32 es más fiel y la única opción sensata en CPU. En un repo que ya viene en ONNX no hay elección: su precisión la fijó quien lo publicó.

**No hay techo de descarga, solo avisos.** Al expandir una tarjeta, Upflow mide el espacio libre real del disco destino y la VRAM libre de **cada** dispositivo, y avisa si el modelo probablemente no funcione bien ahí:

```
Precisión   (•) fp16 · baja 2.6 GB    ( ) fp32 · baja 5.2 GB

dml:0  RX 7900 XTX   libre 22.1 GB   ✓ entra
dml:1  RX 6600        libre 7.4 GB   ✗ no entra (~8.4 GB estimados a 512×512)
cpu    CPU                            ⚠ varios minutos por imagen
```

Son avisos, no bloqueos: el botón de instalar queda habilitado siempre. La decisión es de quien usa la app, que sabe de su máquina más que una constante en el código. El estimado de VRAM es eso — un estimado derivado del tamaño de los pesos, etiquetado con la resolución de referencia.

Endpoints: `GET /api/v1/generation/models/search?q=` (vacío = browse por descargas), `GET /api/v1/generation/models/preflight?repoId=<id>` y `POST /api/v1/generation/models` con `{"repoId": "...", "precision": "fp16"}`.

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

Además del conteo de jobs, cada permiso también puede exigir recursos libres reales del device antes de otorgarse (`app/services/resource_probes.py`):

- **`MIN_FREE_VRAM_MB`** (default `768`) — VRAM libre mínima para admitir un job GPU nuevo, medida en vivo vía `IDXGIAdapter3::QueryVideoMemoryInfo` (detecta presión de **otras apps**, no solo jobs propios). `0` = sin piso.
- **`MIN_FREE_RAM_MB`** (default `1024`) — RAM libre mínima para admitir un job `cpu` nuevo, vía `psutil`. `0` = sin piso.
- **`RESOURCE_POLL_INTERVAL_SECONDS`** (default `5`) — cada cuánto se re-chequea mientras un job espera gateado por recursos, para detectar presión externa que se libera sola.

Un job nunca falla por esto — si no hay recursos suficientes, espera en cola (igual que por conteo de jobs) hasta que se liberen, propios o ajenos.

Este gateo por recursos está **activo por defecto** (los valores de arriba no son `0`): un despliegue existente que actualice a esta versión sin tocar su `.env` empieza a admitir jobs según VRAM/RAM libre real de inmediato, aunque el umbral por defecto es generoso y no debería notarse en hardware típico. Para restaurar el comportamiento previo (solo conteo de jobs, sin piso de recursos), poné `MIN_FREE_VRAM_MB=0` y `MIN_FREE_RAM_MB=0`.

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
| `MIN_FREE_VRAM_MB` | `768` | VRAM libre mínima (MB) para admitir un job GPU nuevo; `0` = sin piso. Ver [Multi-GPU](#multi-gpu-colas-por-dispositivo--auto-router-opcional) |
| `MIN_FREE_RAM_MB` | `1024` | RAM libre mínima (MB) para admitir un job `cpu` nuevo; `0` = sin piso |
| `RESOURCE_POLL_INTERVAL_SECONDS` | `5` | Cada cuánto se re-chequea VRAM/RAM mientras un job espera por presión externa |
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
| `RIFE_MODEL` | `rife-v4.25` | Modelo RIFE usado para interpolar (recomendado, general-purpose) |
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
| `ENABLE_STREAM_PIPELINE` | `True` | Pipeline de frames en streaming (decode→interp→upscale→encode por colas en memoria, sin PNGs intermedios). Cae al camino clásico ante cualquier fallo. `False` fuerza siempre el camino clásico. Ver [Pipeline de frames en streaming](#pipeline-de-frames-en-streaming-enable_stream_pipeline) |
| `ONNX_VIDEO_MAX_PIPELINE_MB` | `1024` | Presupuesto de RAM para las colas de frames en vuelo. Es **global**: se reparte entre todas las colas del pipeline, no es por cola |
| `ENABLE_FILE_LOGGING` | `False` | Escribe los logs a `runtime/logs/upflow.log` (rotado). Apagado por defecto: en uso normal es ruido y disco. Se enciende desde **Settings** sin reiniciar, para que quien reporta un problema pueda reproducirlo y mandar el archivo |
| `LOG_FILE_MAX_MB` / `LOG_FILE_BACKUPS` | `10` / `3` | Techo por archivo antes de rotar y cuántos rotados se conservan |

## Optimization Center

El **Optimization Center** (en el módulo Settings) detecta cinco configuraciones del sistema que afectan el rendimiento de upscaling en GPUs DirectML de AMD; tres son corregibles con un click (requieren confirmación UAC), dos son diagnóstico de solo lectura:

| Detección | Qué es | Fija automáticamente | Requiere reboot |
|---|---|---|---|
| **HAGS** (Hardware Accelerated GPU Scheduling) | Necesario para DirectML moderno en AMD; activa el scheduler de GPU del OS | Sí, un click | Sí |
| **Disk write-cache** | Caché de escritura en disco — cuando está apagado, cada PNG intermedio se sincroniza a disco antes de continuar. Su peso bajó bastante desde el pipeline de streaming (los caminos que no escriben PNGs no lo sufren), pero sigue importando en el camino clásico: NCNN, modelos HF-ONNX y cualquier job que caiga al fallback | Sí, un click | Sí |
| **Defender en runtime/** | Exclusión del antivirus para `runtime/` — si está activado, el scanner de Windows ralentiza I/O de frames | Sí, un click | No |
| **PCIe link** | Velocidad y ancho del enlace PCIe entre la GPU y la CPU | Diagnóstico: solo lectura (no hay fix de software) | — |
| **ONNX CPU fallback** | Detecta qué operaciones de los modelos reales corren en CPU en vez de GPU — herramienta de diagnóstico para identificar cuellos de botella | Manual: ejecutar diagnóstico desde el panel | — |

Todos los fixes ejecutan vía **elevated PowerShell** (pide UAC una sola vez). El timeout de espera es `CAPABILITY_FIX_TIMEOUT_SECONDS` (default 120s) — si el usuario no responde al prompt UAC en ese tiempo, el fix falla sin romper nada.

### Resizable BAR y Above 4G Decoding

La interfaz también incluye un **checklist informativo** para Resizable BAR y Above 4G Decoding (configuraciones BIOS):

- **Resizable BAR** — permite que la CPU acceda a toda la VRAM de la GPU en una sola pasada (vs. el default de 256 MB por ventana). Típicamente aparece en BIOS como "Resizable BAR" o "Smart Access Memory (SAM)" según el fabricante. Hay un checkbox para confirmar manualmente que ya lo activaste (se persiste server-side en `REBAR_CONFIRMED`, con migración automática del valor que versiones viejas guardaban en el localStorage del navegador) — nunca bloquea lógica, es orientativo.
- **Above 4G Decoding** — permite direccionar framebuffers >4GB (relevante solo si tienes >8GB de VRAM en la GPU + resoluciones extremas). Misma mecánica: checklist informativa, sin enforce automático.

Ambas están **fuera del alcance de fix automatizado** (requieren cambios en BIOS/firmware, no en software).

#### ¿Cuánto ayuda realmente Resizable BAR acá? (medido)

Poco, y conviene ser honestos al respecto: **ReBAR acelera las escrituras CPU→GPU**, y en un upscaler esa es justamente la dirección barata. La transferencia grande va al revés.

Medido en una RX 7800 XT con `realesr-animevideov3-x4` sobre DirectML, desglosando un frame completo (subida del input, cómputo, readback del resultado):

| Entrada | Input subido | Output bajado | Subida | Total/frame | % que ReBAR podría tocar |
|---|---|---|---|---|---|
| 640×480 | 0.9 MB | 14.1 MB | 0.06 ms | 22.9 ms | **0.26%** |
| 960×720 | 2.0 MB | 31.6 MB | 0.08 ms | 55.2 ms | **0.15%** |
| 1280×720 | 2.6 MB | 42.2 MB | 0.12 ms | 73.1 ms | **0.16%** |

El desglose a 960×720: subida 0.2%, cómputo 90%, readback 10%. El output pesa **16x más que el input** (es un 4x: scale² en píxeles) y viaja GPU→CPU, que es un camino distinto que ReBAR no acelera.

El argumento cierra por los dos lados: si ReBAR ya está activo, la subida *ya* mide 0.2% del tiempo; si está inactivo, activarlo puede ganar como mucho ese 0.2%. En ambos casos es ruido frente al 90% de cómputo.

Donde sí podría notarse es en la **carga inicial de modelos grandes** a VRAM (los pipelines de generación pesan GB, no los 2.5 MB del upscaler de video), pero es un costo de una sola vez por sesión, no de throughput.

Conclusión: dejarlo activado no hace daño y es buena higiene general del sistema, pero **no esperes ganancia medible en upscaling de video por activarlo**. El checklist se mantiene como orientación, no como promesa de rendimiento.

### AudioSR e IOBinding de GMFSS (deferred)

- **Retrofit IOBinding** (Fase 0.2, Task 10): `ApolloRestorer` ganó un fast-path IOBinding para DirectML, pero `AudioSrRestorer`, `GmfssEngine` y `OnnxUpscaler` lo **difirieron explícitamente** — sus arquitecturas (AudioSR: loop DDIM con shapes dinámicas por step; GMFSS: 4 grafos con constraints frágiles `ORT_DISABLE_ALL`; OnnxUpscaler: auditoría pendiente) necesitan análisis y retrofit dedicados en passes futuras. El código de Apollo es la referencia.
- **Manual BIOS Checklist** (Fase 2): la comprobación de Resizable BAR/Above 4G no tiene backend de detección automática (requeriría acceso a BIOS/firmware propietario) — es un checklist guiado + confirmación manual en localStorage, nunca bloquea la lógica.

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

GMFSS es un segundo motor de FPS boost — mucha más calidad que RIFE en anime, pero **~10x o más lento** (medido 0.72-0.73 fps @1080p 2x en una RX 7800 XT **con splat OpenCL GPU activo** — ver nota de `pyopencl` abajo; sin él, GMFSS cae a splat por CPU y ronda 0.38fps. Pensalo como "máxima calidad, muy lento", no como reemplazo de RIFE para uso diario). Deshabilitado por defecto:

```powershell
# 1. Descargar los modelos ONNX de GMFSS (~55MB, port propio: santiquiroz/port-gmfss-onnx)
powershell -ExecutionPolicy Bypass -File .\scripts\download-gmfss-onnx.ps1

# 2. En .env, habilitar GMFSS (además de ENABLE_INTERPOLATION=true, arriba)
ENABLE_GMFSS=true

# 3. Opcional pero recomendado: splat GPU vía OpenCL (~2x más rápido que CPU-only,
#    ver benchmarks abajo). Sin este paso, GMFSS sigue funcionando, solo cae a CPU
#    para ese sub-paso (fallback automático, con warning único).
.\.venv\Scripts\python -m pip install -e ".[gpu-splat]"
```

Con ambos motores disponibles, el selector RIFE/GMFSS aparece en el dropdown de FPS boost de la UI; por API se elige con `interp_engine=rife|gmfss` en `POST /api/v1/video/jobs` (default siempre `rife`, GMFSS es opt-in por job).

### Nota histórica: fusión interpolar+escalar (eliminada)

La fusión GMFSS+upscale en una pasada (`ENABLE_INTERP_UPSCALE_FUSION`, Fase 2) midió ~1.7x MÁS LENTA que las dos pasadas a 4x/8K en una RX 7800 XT: era un generador secuencial de un solo hilo sin overlap load/compute/save. Fue eliminada y reemplazada por el pipeline de frames en streaming (ver `docs/superpowers/specs/2026-07-25-stream-frame-pipeline-design.md`), que conecta las etapas por colas con un thread por etapa.

### Pipeline de frames en streaming (`ENABLE_STREAM_PIPELINE`)

Activo por defecto. Conecta decode→(interpolación)→upscale→encode por colas acotadas en memoria (frames rgb24 crudos), con cada etapa en su propio thread — sin materializar PNGs intermedios. El presupuesto de RAM es el mismo `ONNX_VIDEO_MAX_PIPELINE_MB` del pipeline ONNX, repartido globalmente entre todas las colas.

| Camino | Con el pipeline | PNGs eliminados |
|---|---|---|
| Sin interpolación + upscale ONNX builtin | decode→stream→upscale→stream→encode | todos |
| GMFSS + upscale ONNX builtin | decode→stream→GMFSS→stream→upscale→stream→encode | todos |
| RIFE + upscale ONNX builtin | decode→PNG→RIFE→(lee sus PNGs)→upscale→**stream**→encode | los de salida (los más pesados) |
| Upscale NCNN (binario) / modelos HF-ONNX | camino clásico completo | ninguno |

Ante CUALQUIER excepción del pipeline, el job cae automáticamente al camino clásico desde cero (se registra en el log y en `job.metadata.streamPipelineFallback`) — el job nunca falla por culpa del camino nuevo. `ENABLE_STREAM_PIPELINE=false` restaura el comportamiento anterior completo (incluido el raw-pipe clásico). Diseño: `docs/superpowers/specs/2026-07-25-stream-frame-pipeline-design.md`.

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

Además de imagen y video, Upflow tiene un **apartado de Audio** propio (ruta `/audio`): subís un archivo de audio (wav/mp3/flac/m4a/ogg/opus), elegís la mejora y descargás el resultado. La cadena es `entrada → [limpieza] → [denoise] → [restore] → [voz] → [acabado] → salida`, cada paso opcional y todos combinables en el mismo trabajo (la única excepción es la separación de stems, que corre sola porque entrega dos archivos).

- **Limpieza (cadena)** — saca defectos de una grabación encadenando modelos de máscara del catálogo de UVR, y devuelve **un solo archivo**. Sirve para **cualquier** audio, música incluida: es la sección para limpiar música. Campo `cleanup_steps` (CSV de ids); combinable con denoise/restore/voz/acabado en el mismo trabajo. Ver "Cadena de limpieza" abajo.
- **Denoise (para VOZ)** — quita ruido de fondo: `deepfilter` (DeepFilterNet3) o `rnnoise`. Los dos están **entrenados con habla**: separan una voz de su ruido muy bien, y en música tratan a los instrumentos como ruido y pueden apagarlos — para música, usar la cadena de limpieza. Es el mismo motor que ya se usa en video (ver sección anterior); requiere `ENABLE_AUDIO_ENHANCE=true` + `download-deepfilternet.ps1`.
- **Restore (EXPERIMENTAL)** — dos motores, elegibles por job:
  - `apollo`: reconstruye la banda de agudos perdida por compresión de códec (audio de WhatsApp/Telegram/redes). Rápido y liviano (~74 MB). Requiere `ENABLE_AUDIO_RESTORE=true` + `scripts/download-apollo.ps1`.
  - `audiosr`: **super-resolución de audio general por difusión latente** (cualquier banda → 48 kHz, UNet de 258M params). Techo de calidad muy superior a Apollo pero ~2 min de proceso por minuto de audio en GPU (50 pasos DDIM con CFG). Port ONNX propio — el primero conocido de AudioSR: [santiquiroz/port-audiosr-onnx](https://github.com/santiquiroz/port-audiosr-onnx). Requiere `ENABLE_AUDIOSR=true` + `scripts/download-audiosr-onnx.ps1`. El script acepta `-Precision fp32|fp16`, y el botón de la tarjeta elige solo: **fp16 si el dispositivo por defecto es GPU** (1.26 GB en vez de 2.51 GB, 9% más rápido, salida a 59.4 dB SI-SDR de la fp32 — inaudible), **fp32 si es CPU**, porque el EP de CPU tiene muchos menos kernels fp16. Correr un pack fp16 en CPU falla a propósito, antes de crear la sesión, diciendo que hay que reinstalarlo en fp32. Medición completa: `docs/superpowers/specs/2026-08-17-audiosr-fp16-findings.md`.

  Ambos motores son ONNX **multi-provider** (corren en cualquier GPU DirectX12 —AMD/NVIDIA/Intel— o CPU, igual que los modelos de imagen HF). Si un modelo no está instalado, ese modo simplemente no aparece — la app nunca se rompe por esto. **Preservan estéreo/surround**: en vez de downmixear a mono antes de restaurar, decodifican Mid/Side (restauran solo el Mid, el Side queda intacto) en estéreo, y en 5.1/7.1 restauran frente/rears por par M/S + centro directo + LFE intacto, con RMS-match por canal contra el original al final; un layout de canales no reconocido cae a mono con warning explícito (nunca en silencio).
- **Formato de salida** — `output_format: wav|flac|mp3|m4a`, default **`flac`** (sin pérdida, ~50% más liviano que WAV). `wav` para compatibilidad con editores viejos; `m4a` (AAC) es el más compatible con teléfonos y con el ecosistema Apple; `mp3` lo lee todo, incluso equipos viejos. Para `mp3`/`m4a`, `lossy_quality: maximum|balanced|compact` (default **`maximum`** = MP3 320 kbps / AAC 256 kbps) elige el bitrate; en `wav`/`flac` se ignora. Ogg/Opus se aceptan de **entrada** pero no se ofrecen de salida: Vorbis pierde contra Opus en calidad y contra MP3/AAC en compatibilidad, y Opus gana en eficiencia pero pierde justo en el eje por el que existe esta selección.
- **Convertir de formato sin procesar nada** — un job **sin ningún paso** (sin limpieza, denoise, restauración, voz, mastering ni separación) es válido si el formato de salida es distinto al del archivo: convierte en **una sola pasada de ffmpeg** desde el original, **conservando la frecuencia de muestreo y la profundidad de bits** hasta donde el destino lo permita. Un FLAC de 44.1 kHz / 24 bits sale como WAV de 44.1 kHz / 24 bits, no como los 48 kHz / 16 bits a los que decodifica el camino de procesamiento (esa tasa la exigen DeepFilterNet, RNNoise y los separadores; una conversión pura no pasa por ahí). Lo que el destino **no** admite queda registrado en `job.metadata` y se muestra en el detalle del trabajo: un resample forzado (96 kHz → MP3 sale a 48 kHz) en `conversionResampled`, un downmix forzado (5.1 → MP3 sale estéreo) en `conversionDownmixed`, una profundidad recortada en `conversionBitDepthReduced`, y un remux sin recodificar en `conversionCopied` — nunca en silencio. Pedir el **mismo** formato que ya tiene el archivo sin ningún paso devuelve `400`: no hay nada que hacer.
- **Separación de stems (karaoke y limpieza)** — parte la mezcla en **dos archivos** con un modelo del catálogo de [Ultimate Vocal Remover](https://github.com/Anjok07/ultimatevocalremovergui). Corre **solo** (los demás pasos se aplicarían a un stem ambiguo): pedilos en un segundo trabajo sobre el stem que quieras. Cada modelo se baja por separado con `scripts/download-karaoke.ps1 -Model <id>` (o con el botón de descarga que la propia UI muestra al lado del modelo que falta); tener uno instalado alcanza para habilitar el modo.

  | Grupo | Id | Qué hace | Stems (el 1º es el que baja `downloadUrl`) |
  |---|---|---|---|
  | Karaoke | `inst_hq_3` | Saca la instrumental; la voz es el resto | `instrumental` + `vocals` |
  | Karaoke | `voc_ft` | Saca la voz; la instrumental es el resto | `instrumental` + `vocals` |
  | Karaoke | `mel_band_roformer_kim` | **Máxima calidad**, ~20x más lento; saca la voz, la instrumental es el resto | `instrumental` + `vocals` |
  | Limpieza | `reverb_hq` | Saca la cola de reverb; la pista limpia es la resta | `dry` + `wet` |
  | Limpieza | `deecho_normal` | Eco moderado, sin tocar el resto de la señal | `no_echo` + `echo` |
  | Limpieza | `deecho_aggressive` | Eco fuerte; pega más duro y puede apagar la señal | `no_echo` + `echo` |
  | Limpieza | `deecho_dereverb` | Eco **y** reverb de sala en una pasada | `no_reverb` + `reverb` |

  Los tres `deecho_*` son de **FoxJoy**, distribuidos por el canal oficial de descargas de UVR y exportados a ONNX por un port propio y público: [santiquiroz/port-uvr-deecho-onnx](https://github.com/santiquiroz/port-uvr-deecho-onnx) (MIT, paridad medida contra `python-audio-separator`: 61-65 dB SI-SDR). Son otra arquitectura (VR 5.1 CascadedNet) que los MDX-Net, pero eso no se elige: el catálogo es **una sola lista** y el backend resuelve el motor por el modelo que pediste. Medido en una RX 7800 XT (`dml:0`): **~21x tiempo real** — 5 minutos de audio en 14 segundos.

  `mel_band_roformer_kim` es el **carril de máxima calidad**, y es una elección explícita, no el default: es [Mel-Band RoFormer](https://huggingface.co/KimberleyJSN/melbandroformer) de **KimberleyJSN** (MIT — el roformer vocal con mejor SDR que declara licencia permisiva: 10.98 en el multisong de MSST, por encima del BS-RoFormer de viperx, que además no declara ninguna), exportado a ONNX por otro port propio y público: [santiquiroz/port-bs-roformer-onnx](https://github.com/santiquiroz/port-bs-roformer-onnx) (MIT). Tercera arquitectura del catálogo, misma lista única. Lo que hay que saber **antes** de elegirlo, y por eso la UI lo advierte al lado del modelo: pesa **931 MB**, necesita **~2,3 GB libres** en el dispositivo (el grafo fp32 más un intermedio de atención de ~1,3 GB — si no los hay, el trabajo falla al cargar con un mensaje que lo dice, no a mitad de camino) y cuesta **~20x** lo que `inst_hq_3`. Medido de punta a punta en una RX 7800 XT (`dml:0`, sesión caliente): **50 s de GPU por minuto de audio**, contra 2,5 s de `inst_hq_3`. En CPU no es un carril viable (0,69x tiempo real, más lento que reproducir el archivo). Mismo lugar de producto que GMFSS en video: elegilo cuando la calidad de la separación importa más que la espera.

  La descarga se pide por stem: `GET /api/v1/audio/jobs/{id}/download?stem=<id>` (sin `stem` sirve el primero). La respuesta del job trae `stems[]` con las dos URLs ya etiquetadas. **El contrato de dos salidas es exclusivo de este modo**: la cadena de limpieza entrega un archivo y no trae `stems[]`.

### Cadena de limpieza (`cleanup_steps`)

Los modelos del grupo "Limpieza" no son separadores aunque compartan motor: un separador parte una mezcla en dos cosas que querés, y estos **quitan un defecto** — entra audio, sale el mismo audio sin ruido, sin eco o sin reverb. Por eso además de correrse sueltos (modo separación, para escuchar qué sacan) se pueden **encadenar**, que es el flujo que la gente hace a mano en UVR.

`POST /api/v1/audio/jobs` con `cleanup_steps=<csv de ids>` corre **una pasada por id** y devuelve **un archivo**: el audio limpio. Los stems removidos de las pasadas intermedias no se guardan.

- **El orden lo fija el catálogo, no el request.** `denoise` → `deecho_*` → `reverb_hq`, siempre, mandes los ids en el orden que los mandes. Tiene causalidad: el ruido de banda ancha está en todo el espectro y en todo el tiempo, así que confunde a los modelos que vienen después; el eco son reflejos **discretos** y se modelan mientras sigan siendo copias reconocibles; la reverb es la **cola difusa** que queda cuando los reflejos ya no se distinguen, y sacarla primero le borraría al de-echo el material del que deduce los reflejos.
- **Exclusión por familia.** `deecho_normal` y `deecho_aggressive` son el mismo modelo en dos intensidades: elegí uno. `deecho_dereverb` hace eco **y** reverb en una pasada, así que excluye a los dos de-echo **y** a `reverb_hq`. Una combinación redundante devuelve `400` nombrando el par — no se normaliza en silencio, porque elegir un ganador entre dos intensidades sería inventar una decisión de calidad que es del usuario.
- **Se combina** con `denoise`/`restore`/`voice_steps`/`master` en el mismo trabajo. Corre después del decode y **antes** de todos ellos: limpiar antes de reconstruir y de nivelar.
- **Cada pasada es con pérdida** (son máscaras: descartan señal y no la devuelven). Desde la tercera, el job marca `metadata.cleanupOverprocessed` y la UI avisa que el resultado puede sonar sobreprocesado. No bloquea.
- `separate=true` **+** `cleanup_steps` devuelve `400`: la separación entrega dos archivos y la cadena uno, así que la combinación no define qué se entrega. Encadenalo en dos trabajos.
- El catálogo, en orden de ejecución, sale de `GET /api/v1/audio/capabilities` → `cleanupSteps[]` (`id`, `name`, `family`, `covers`, `installed`, `descriptionKey`) más `cleanupOverprocessingThreshold`. `covers` es lo que le permite a un cliente aplicar la exclusión sin hard-codear ids.

  Medido en una RX 7800 XT (`dml:0`), cadena de 2 pasadas sobre 12 s de audio estéreo 44,1 kHz: **4,44 s totales** — 0,10 s de decode, 2,17 s la pasada de `denoise`, 2,21 s la de `deecho_normal`. Cada pasada es una inferencia completa sobre el archivo entero, así que el costo es lineal en la cantidad de pasos.

```powershell
# Restore experimental: descargar el modelo Apollo (~74 MB) y habilitarlo
powershell -ExecutionPolicy Bypass -File .\scripts\download-apollo.ps1
# en .env:  ENABLE_AUDIO_RESTORE=true
```

API: `POST /api/v1/audio/jobs` (multipart: `file`, `cleanup_steps?` (CSV), `denoise?`, `restore?`, `voice_steps?` (CSV), `master?`, `output_format?` default `flac`, `lossy_quality?` default `maximum`, `device?`; **todos los pasos son opcionales**: solo `file` + un `output_format` distinto al del archivo ya es una conversión válida) → 202; `GET /api/v1/audio/jobs/{id}` (estado + progreso), `.../download` (resultado), `GET /api/v1/audio/capabilities` (qué motores están instalados; `restoreModes` lista los modos listos). El mismo `restore=apollo|audiosr` se puede pedir en un job de video vía el campo `audio_restore` (con `keep_audio=true`), aplicado después del denoise; el formato de salida de esa pista restaurada se controla con `audio_output_format` (ver "Crear un job de video" arriba).

> **Nota experimental:** el restore es un port ONNX del modelo Apollo (ver `docs/` y la guía del port). Funciona y es multi-provider, pero la calidad de reconstrucción y el rendimiento GPU aún se están evaluando — por eso va detrás de un flag y con badge "Experimental" en la UI.

## Servidor MCP (tools para agentes de IA)

Upflow expone toda su funcionalidad como **tools MCP** (Model Context Protocol) para que agentes de IA (Claude Code, Claude Desktop, o cualquier cliente MCP) puedan reescalar, transcribir, generar y procesar medios directamente.

- **24 tools** sobre las 7 familias de jobs de la API: upscale de imagen/video, audio (denoise/restore/master), transcripción/doblaje, descargas (yt-dlp), generación de imágenes/video, TTS y 3D imprimible.
- Modelo de jobs unificado: `upflow_job_status` / `upflow_wait_job` / `upflow_cancel_job` / `upflow_download_result` funcionan igual para cualquier familia (`image | video | audio | generation | transcribe | download | shape3d`).
- Las tools de creación aceptan **rutas de archivo locales** y resuelven la subida (multipart o staging por token) por sí solas.
- Es un cliente HTTP fino: el servidor Upflow tiene que estar corriendo; MCP y la web UI ven exactamente los mismos jobs.

Config para un cliente MCP (ej. `.mcp.json` de Claude Code):

```json
{
  "mcpServers": {
    "upflow": {
      "command": "C:/ruta/a/upflow/.venv/Scripts/python.exe",
      "args": ["-m", "app.mcp"],
      "cwd": "C:/ruta/a/upflow",
      "env": { "UPFLOW_URL": "http://127.0.0.1:8090" }
    }
  }
}
```

Variables: `UPFLOW_URL` (default `http://127.0.0.1:8090`); con `AUTH_MODE=multi`, `UPFLOW_USERNAME`/`UPFLOW_PASSWORD` hacen login automático. En modo single-user (default) no hace falta nada. También queda el script `upflow-mcp` instalado por `pip install -e .`.

Flujo típico de un agente: `upflow_status` → `upflow_upscale_image(file_path=..., destination_path=...)` (espera y descarga en un solo paso) o, para videos largos, `upflow_upscale_video(...)` → `upflow_wait_job` → `upflow_download_result`.

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
