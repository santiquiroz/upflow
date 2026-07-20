# Selección de pistas de audio, subtítulos y restauración calidad-first — Design

**Fecha:** 2026-07-19
**Estado:** Approved (pendiente de plan de implementación)

## Motivación

Dos problemas detectados por el usuario en uso real:

1. Los videos que reescala suelen traer **múltiples pistas de audio** (idiomas/dubs) y **subtítulos embebidos** (típicamente ASS/SSA en rips de anime). El pipeline actual solo conserva la pista de audio que ffmpeg elige por default y descarta subtítulos siempre — sin excepción, sin aviso.
2. Comparando con un proyecto de un tercero (puerto CUDA equivalente), se identificó que **Apollo y AudioSR downmixean a mono antes de restaurar** (`data.mean(axis=1)` en ambos motores) — se pierde la imagen estéreo/surround completa. El proyecto de referencia hace M/S (mid-side) en estéreo y separación de canales en 5.1/7.1, más RMS-matching post-restauración.

Principio rector explícito del usuario: **calidad primero, portabilidad no es excusa para degradar resultado.** Este documento prioriza fidelidad sobre simplicidad de implementación donde ambas compiten.

## Estado actual verificado (no supuesto)

- `app/services/video_upscaler.py`: extracción de audio usa `-vn` sin `-map` explícito → ffmpeg elige una sola pista por heurística default. Sin `-map 0:s` en ningún lado → subtítulos siempre descartados.
- `app/services/engines/audiosr_restore.py::_load_mono_48k` y `app/services/engines/apollo_restore.py::_load_mono_44k`: ambos hacen `data.mean(axis=1)` — downmix a mono antes de restaurar, sin excepción.
- `app/services/engines/audiosr/driver.py`: normalización de input es solo peak-to-0.5 (`driver.py:212-213`), no hay RMS-matching post-restauración contra el original.
- `output_container` **ya existe** end-to-end (`app/schemas.py`, `app/models.py`, `app/api/routes.py`, `app/services/video_job_manager.py:263` valida `{"mp4", "mkv"}`) — infraestructura de contenedor reutilizable, no hay que construirla desde cero.
- Pista restaurada en jobs de video se re-encodea a AAC 192k lossy al mux-ear (`_prepare_processed_audio` retorna `["-c:a", "aac", "-b:a", "192k"]`); la pista original sin tocar usa `-c:a copy` (sin pérdida).
- Módulo Audio standalone (`app/services/audio_pipeline.py`): output SIEMPRE `.wav`, hardcodeado, sin selector de formato.

## Fase A — Análisis de pistas + selección + subtítulos

### Nuevo endpoint: `POST /api/v1/video/analyze`

- Acepta el mismo tipo de upload multipart que `POST /api/v1/video/jobs`.
- Guarda el archivo en `uploads/` con el mismo patrón de token único ya usado (`{token}-{safe_name}`).
- Corre `ffprobe -show_streams` sobre el archivo guardado.
- Responde con:
  ```json
  {
    "uploadToken": "<token>",
    "video": {"codec": "...", "width": 1920, "height": 1080, "fps": "24/1"},
    "audioTracks": [
      {"index": 0, "language": "jpn", "codec": "ac3", "channels": 2, "isDefault": true},
      {"index": 1, "language": "eng", "codec": "aac", "channels": 6, "isDefault": false}
    ],
    "subtitleTracks": [
      {"index": 0, "language": "eng", "codec": "ass"}
    ]
  }
  ```
- El archivo queda en `uploads/` sujeto al mismo TTL/sweep de `RetentionSweeper` que ya existe — no requiere limpieza nueva.

### `POST /api/v1/video/jobs` — nuevos campos

- `upload_token: str | None` — alternativa a `file`; si viene, referencia el upload ya analizado (mover/renombrar a la ubicación esperada por el job, no volver a subir). `file` sigue soportado para compatibilidad — comportamiento actual sin cambios si no se usa `analyze` primero.
- `audio_track_indices: list[int] | None` — `None` = comportamiento actual (ffmpeg elige default). Lista explícita = esas pistas se conservan.
- `keep_subtitles: bool` (default `False`) — si `True`, copia TODAS las pistas de subtítulos detectadas.
- Validación: si viene `file` y `upload_token` a la vez, o ninguno, 400.

### Mapeo de audio (ffmpeg)

- Sin `audio_track_indices`: comportamiento actual sin cambios (heurística default de ffmpeg, una pista).
- Con `audio_track_indices` de un solo elemento: `-map 0:a:<idx>` (equivalente a hoy pero explícito).
- Con múltiples: una pista designada **primaria** (la default detectada, o la primera de la lista si no hay default) pasa por el pipeline de enhance/restore existente si está activo; el resto se copian sin procesar (`-map 0:a:<idx> -c:a copy` por cada una). Enhance/restore NUNCA se aplica a más de una pista — evita multiplicar el costo de un modelo de difusión por N idiomas.

### Subtítulos

- `keep_subtitles=True` → `-map 0:s?` (todas, `?` evita fallo si no hay ninguna) + `-c:s copy`.
- Requiere contenedor `.mkv` (ASS/SSA no tiene codec nativo en MP4 sin pérdida de estilos). Si el job no pidió `.mkv` explícitamente, el `output_container` resuelto sube a `"mkv"` automáticamente — el response del job incluye una nota (`job.metadata["containerUpgradedReason"]`) explicando por qué, nunca silencioso.

## Fase B — Restauración M/S + RMS-match (Apollo y AudioSR)

Un módulo nuevo compartido (p.ej. `app/services/engines/multichannel_restore.py`) reemplaza las llamadas a `_load_mono_48k`/`_load_mono_44k` en ambos motores. Contrato:

```python
def restore_multichannel(
    audio: np.ndarray,       # [samples, channels], float32
    sample_rate: int,
    restore_mono: Callable[[np.ndarray], np.ndarray],  # el restore existente (AudioSR o Apollo), mono-in mono-out
) -> np.ndarray:             # [samples, channels], mismo layout de entrada
    ...
```

- **1 canal (mono)**: `restore_mono(audio[:, 0])` directo, sin cambios de comportamiento.
- **2 canales (estéreo)**: `mid = (L + R) / 2`, `side = (L - R) / 2` → `mid' = restore_mono(mid)` → `L' = mid' + side`, `R' = mid' - side`. `side` nunca pasa por el modelo.
- **6 canales (5.1) / 8 canales (7.1)**: mapeo por layout estándar ffmpeg (`FL FR FC LFE BL BR [SL SR]`). Frente (`FL`/`FR`) y surrounds/rears por M/S igual que estéreo; centro (`FC`) directo por `restore_mono` (ya es mono, contenido de diálogo); `LFE` se copia sin tocar (el modelo no aporta nada útil en esa banda).
- **RMS-match**: tras reconstruir, escalar la señal restaurada completa para que su RMS iguale al RMS de la señal de entrada original (por canal), antes de guardar. Función pura, testeable en aislado.
- Layouts no cubiertos (raros: 4.0, 7.1 side, etc.) → fallback documentado: tratar como N pares estéreo consecutivos + centro/LFE si están presentes; si no matchea ningún patrón conocido, downmix a mono con **warning explícito en logs y en `job.metadata`** (nunca degradar en silencio) — comportamiento de hoy, pero ahora es el fallback documentado, no el default.

## Fase C — Formato de salida elegible

### Video (`audio_output_format: "auto" | "flac" | "aac"`, default `"auto"`)

- `"auto"`: si `audio_restore` está activo → FLAC + contenedor `mkv` (igual lógica de auto-upgrade que subtítulos, mismo campo de aviso). Si no hay restore activo, comportamiento actual (AAC 192k si hay enhance, `copy` si no hay procesamiento).
- `"flac"` / `"aac"` explícitos: fuerzan el codec sin importar si hay restore o no (usuario decide contra la recomendación).

### Módulo Audio standalone

Nuevo campo `output_format: "wav" | "flac" | "mp3"`, default `"flac"` (cambia el default actual de facto de wav a flac — lossless y más chico, sin downside real). Copy amigable en la UI:

- **FLAC (recomendado)**: sin pérdida de calidad, ~50% más liviano que WAV.
- **WAV**: sin pérdida, sin comprimir, compatibilidad universal (algunos editores viejos no leen FLAC).
- **MP3**: con pérdida, el archivo más chico — solo si el tamaño importa más que la calidad.

## Testing

- `test_video_analyze.py`: ffprobe fake (subprocess mockeado) → parseo correcto de audioTracks/subtitleTracks, staged upload token reutilizable por `/jobs`.
- `test_multichannel_restore.py`: mono passthrough (identidad), round-trip M/S sin restaurar Mid (debe reproducir el original salvo error de redondeo float), RMS-match (verificar RMS post == RMS pre dentro de tolerancia), 5.1 con LFE intacto bit-exacto, fallback con layout desconocido logea warning.
- `test_video_job_manager.py` / `test_video_upscaler.py`: construcción de comando ffmpeg con `-map` múltiple (audio + subs), auto-upgrade de contenedor + mensaje en metadata, `audio_track_indices` de un elemento vs varios.
- `test_audio_pipeline.py`: los 3 formatos de salida producen el codec/extensión correcta.
- Frontend: nuevo flujo analyze→seleccionar→confirmar en `VideoPanel.test.tsx` (mock de `/analyze`), selector de formato en `AudioPanel.test.tsx`.

## Fuera de alcance (documentado, no implementar acá)

- Generación de subtítulos con IA (whisper.cpp) — sigue en el roadmap, es una feature distinta (crear subs, no preservar existentes).
- Layouts de canal exóticos más allá de mono/estéreo/5.1/7.1 estándar — caen al fallback mono documentado.
- Separación de fuentes (voz/música/efectos) — ni nosotros ni el proyecto de referencia lo hacen; ambos mejoran el mix conjunto.
