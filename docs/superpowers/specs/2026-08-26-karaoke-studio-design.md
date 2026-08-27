# Karaoke Studio — apartado dedicado con pipeline configurable

Fecha: 2026-08-26. Aprobado en conversación (enfoque A).

## Problema

El modo karaoke de Transcribir es una caja cerrada: modelo de separación
hardcodeado, sin limpieza, sin mejora de resolución, sin elección de fondo, sin
revisar la letra antes de pagar el render. El usuario quiere un apartado
dedicado con libertad sobre cada paso.

## Alcance

- Página nueva `/karaoke` en la navegación.
- Job nuevo de DOS etapas con revisión en el medio.
- Reusa motores existentes; no duplica ninguno en memoria.

Fuera de alcance (decidido): URL de YouTube como entrada, un video por idioma
(se eligió bilingüe apilado), edición de tiempos por palabra.

## Arquitectura

### KaraokeJob (dataclass en `app/models.py`)

Campos de entrada (etapa 1): `source_path`, `original_filename`,
`separation_model_id` (default el del catálogo), `cleanup_steps: list[str]`
(ids de `cleanup_chain`), `restore_mode: str | None`
(`apollo` | `audiosr` | None), `asr_model_id`, `language`, `romanize: bool`,
`translate_to: str | None`, `device`.

Campos de render (etapa 2, llegan con el endpoint de render): `background_kind`,
`background_path`, estilo de subtítulo (`subtitle_size` = small|medium|large,
`subtitle_position` = bottom|top, `subtitle_color`,
`subtitle_highlight_color` — hex `#RRGGBB`, convertido a `&HAABBGGRR` ASS en
backend).

Campos de estado: `phase` (`preparing` → `review` → `rendering` →
`completed`/`failed`/`cancelled`), `segments`, `translated_lines: list[str]`,
`instrumental_path`, `work_dir`, `background_kind`
(`source` | `image` | `video` | `generated`), `background_path`,
`output_path`.

### KaraokeJobManager (`app/services/karaoke_job_manager.py`)

`QueuedJobManager`, 1 worker. Dependencias inyectadas desde `main.py`:
`settings`, `transcribe_engine`, `model_registry`, `separators` (dict
compartido), `restorers` (dict compartido), `devices`, `quota_service`.

**Etapa 1 — preparar** (al crear el job, corre en el worker):

1. Decode a WAV 44.1k estéreo (ffmpeg, mismo comando que audio).
2. Separación con `separation_model_id` vía `SEPARATION_MODELS` +
   `separators[spec.architecture]`. Stem principal = instrumental.
3. Limpieza: `cleanup_steps_from_selection()` valida (orden del catálogo,
   exclusión por familia) y cada pasada corre el separador de limpieza sobre el
   instrumental.
4. Restauración opcional: `restorers[restore_mode].run(in, out, device)`.
5. Transcripción del audio ORIGINAL completo (no del instrumental) con
   `transcribe_engine` + `asr_model_id`/`language`; romaji opcional con
   `romanize_segments`.
6. Traducción opcional: `TranslationEngine.translate([s.text], pair)` por
   segmento; par no instalado = error claro al CREAR el job, no al final.
7. El instrumental queda como FLAC en el work dir del job
   (`outputs/{id}.karaoke-studio/`); `phase = review`.

**Etapa 2 — render** (endpoint explícito, re-encola el job):

1. ASS bilingüe con estilo parametrizado (ver Subtítulos).
2. Fondo según `background_kind`:
   - `source`: el video original (si tiene imagen real, mismo probe de hoy).
   - `image`: imagen subida, `-loop 1` a la duración del instrumental.
   - `video`: video subido, `-stream_loop -1` + `-shortest` (se lupea o corta).
   - `generated`: fondo `lavfi` oscuro actual.
3. Un solo ffmpeg: fondo + subtítulos quemados + instrumental (extensión de
   `build_karaoke_command`).
4. `phase = completed`; el work dir se borra; el fuente se borra.

La letra es editable en `review`: endpoint que reemplaza `text` de cada
segmento y/o su línea traducida por índice. Tiempos intactos. Editar dispara
re-romanización NO (el texto editado es el que se quema tal cual: el usuario
tiene la última palabra).

### Subtítulos bilingües (`app/services/karaoke_subtitles.py`, extensión)

- `KaraokeStyle` dataclass: tamaño de fuente (mapea small/medium/large a
  puntos), alineación (2 = abajo, 8 = arriba), color primario, color de
  resaltado (SecondaryColour de ASS es el color ANTES de encenderse; el
  resaltado usa PrimaryColour — se documenta en el código).
- Línea original: `\k` por palabra real (lo de hoy).
- Línea traducida: estilo `Translation` — fuente ~70% de la principal, misma
  zona vertical, renglón debajo (MarginV menor) — con `\k` proporcional por
  cantidad de letras (`split_line_proportionally` existente).
- `render_karaoke_ass(lines, *, translations=None, style=DEFAULT)` mantiene
  compatibilidad con el llamador actual de Transcribir.

### API (`app/api/routes.py` + `app/schemas.py`)

- `POST /karaoke/jobs` — multipart: `file` + todos los parámetros de etapa 1.
  202 con job id. Valida TODO acá (modelo separación, cleanup, restore
  disponible, ASR, par de traducción instalado, device).
- `GET /karaoke/jobs` / `GET /karaoke/jobs/{id}` — estado, progreso, y en
  `review`: URL del instrumental + letra (segmentos con texto original,
  romaji ya aplicado, y traducción).
- `GET /karaoke/jobs/{id}/instrumental` — sirve el FLAC del work dir
  (FileResponse, sólo en `review`).
- `PUT /karaoke/jobs/{id}/lyrics` — JSON `{lines: [{index, text,
  translation?}]}`. Sólo en `review`.
- `POST /karaoke/jobs/{id}/render` — multipart opcional (`background_kind`,
  `background` file, estilo). Re-encola. Sólo en `review`.
- `POST /karaoke/jobs/{id}/cancel`, `GET /karaoke/jobs/{id}/download`.
- Permisos: mismos `Permission.jobs_create` / lectura que Transcribir.

### Frontend

- Ruta `/karaoke` + entrada en nav ("Karaoke").
- `services/karaoke.ts` (axios wrappers) + hook de polling tipo
  `useTranscribeJob`.
- `pages/KaraokePage.tsx` + `modules/karaoke/KaraokeStudioPanel.tsx`:
  - **Paso 1 (config)**: dropzone (audio/video), modelo de separación (reusa
    catálogo), checkboxes de limpieza (reusa copy del módulo Audio), radio
    restauración (Ninguna/Apollo/AudioSR), modelo ASR + idioma + romaji
    (sólo ja), traducción (pares instalados, "Ninguna" default), device.
  - **Paso 2 (revisión)**: player `<audio>` con el instrumental, letra
    editable línea por línea (original + traducción), selector de fondo
    (original si hay / imagen / video / generado) con upload, estilo
    (tamaño, posición, 2 colores), botón Renderizar.
  - **Paso 3**: video resultante + descarga.
- Job queue lateral: los jobs karaoke aparecen como los demás.

## Errores

- Toda validación al CREAR o al RENDERIZAR, nunca al final del trabajo.
- Motor faltante (separador/restaurador no construido, par no instalado,
  pack AudioSR/Apollo no descargado) = 400 con mensaje accionable.
- Fallo en etapa 1 borra el work dir; fallo en etapa 2 conserva el work dir y
  vuelve a `review` (lo caro —separar/limpiar/mejorar— no se pierde por un
  fondo corrupto).
- Job en `review` expira con el retention sweeper existente.

## Tests

- `tests/test_karaoke_style.py`: estilo ASS parametrizado + línea traducida
  proporcional + compatibilidad del llamador viejo.
- `tests/test_karaoke_video_background.py`: comandos ffmpeg por
  `background_kind` (image loop, video stream_loop, generated, source).
- `tests/test_karaoke_job_manager.py`: etapa 1 con motores fake (orden de
  pasos, cleanup inválido rechazado al crear, traducción sin par rechazada,
  fallo etapa 2 vuelve a review), edición de letra sólo en review.
- `tests/test_karaoke_api.py`: contrato de endpoints.
- Frontend: test del panel (config → params correctos; revisión → render).
