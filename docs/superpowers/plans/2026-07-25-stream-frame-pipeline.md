# Pipeline de frames en streaming (decode→tensor sin PNGs intermedios) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminar los round-trips PNG del pipeline de video conectando decode→(interpolación)→upscale→encode por colas acotadas en memoria con una etapa por thread (overlap real), con fallback automático al camino clásico y eliminación de la fusión GMFSS vieja (medida 1.7x más lenta).

**Architecture:** Un `FramePipeline` genérico (etapas `FrameStage` conectadas por `queue.Queue` acotadas, cada etapa en su propio thread, backpressure por cola llena) alimentado por un source ffmpeg rawvideo (`pipe:1`, espejo del writer raw-pipe existente) y drenado por el writer raw-pipe extraído a `RawPipeEncoder`. Las etapas envuelven los internals existentes: la sesión ONNX del upscaler builtin (tiling/fp16/whole-frame intactos, vía `build_frame_upscaler`) y GMFSS in-process (vía `build_stream_stage`). `video_upscaler.py` rutea: flag ON + backend ONNX + (sin interp | GMFSS) → pipeline completo; RIFE → híbrido (PNG de entrada para su tramo, stream de upscale→encode); NCNN o cualquier excepción → camino clásico. Spec fuente: `docs/superpowers/specs/2026-07-25-stream-frame-pipeline-design.md`.

**Tech Stack:** Python 3.11 + FastAPI, numpy, OpenCV (cv2), onnxruntime-directml (sesiones ya existentes, imports perezosos), ffmpeg vendored (rawvideo pipes), threading/queue de stdlib, pytest. Cero dependencias nuevas.

## Global Constraints

- Flag `ENABLE_STREAM_PIPELINE` **default true**; fallback automático al camino clásico ante CUALQUIER excepción del pipeline (log + flag en `job.metadata`, mismo patrón que el raw-pipe actual: `rawPipeFallback`).
- Eliminación de `enable_interp_upscale_fusion` y su camino (código+config+tests) como task propia (Task 1).
- Overlap real: cada etapa en su thread, colas `queue.Queue` con maxsize derivado del presupuesto `ONNX_VIDEO_MAX_PIPELINE_MB` (helper compartido extraído de `onnx_video_upscaler.py`, no duplicado). El presupuesto es GLOBAL entre TODAS las colas del pipeline, no por cola. GPU serializada por `GpuSessionCoordinator`/`DeviceSemaphores` existentes — este plan no agrega semáforos nuevos.
- Caminos: (sin interp | GMFSS) + backend ONNX → pipeline completo; RIFE → híbrido (PNG entrada de su tramo, stream de upscale→encode); upscale NCNN → fallback clásico completo.
- Cero deps nuevas (ffmpeg rawvideo pipe, mismo patrón del raw-pipe de salida). Comandos ffmpeg solo con flags no-deprecados (`-fps_mode passthrough`, **nunca** `-vsync`).
- Tests con motores/procesos fake (nunca binarios reales en unit); imports de torch/onnx perezosos en `app/services/`; mensajes de usuario en español; commits convencionales en español SIN `Co-Authored-By`.
- Suites de cierre por task: pytest del archivo tocado + al final la suite completa (`.venv\Scripts\python -m pytest -q`). El frontend NO se toca: el shape de `job.metadata` (claves `stage`/`stages`/`progress`/`framesDone`/`framesTotal`) no cambia.
- Smoke real manual (NO CI): sección final de este plan, no un task.
- La rama de trabajo contiene cambios sin commitear de OTRO agente en archivos de settings/capability — NO tocarlos ni incluirlos en commits. `git add` SIEMPRE por archivo explícito, nunca `git add -A`/`git add .`.
- `.claude/worktrees/**` contiene copias completas del repo: NO tocarlas, y todo grep de verificación debe excluirlas.

## Decisión delegada por el spec: reporte de etapas concurrentes en el stepper

El spec delega cómo reportan progreso las etapas concurrentes. Mirando `app/services/progress.py` y `frontend/src/lib/jobProgress.ts`: el frontend deriva el stepper directamente de `metadata.stages` (lista `{key, label, status}`) y muestra `framesDone / framesTotal` (con `interpFramesTotal` SOLO cuando `metadata.stage == "interpolating_frames"`). La opción más simple que mantiene claves y semántica intactas es la **etapa colapsada**, el mismo precedente que ya usan el raw-pipe y la fusión eliminada:

- El pipeline entero reporta bajo la etapa existente **`upscaling_frames`** (`advance_video_stage(job, "upscaling_frames")` al arrancar). `apply_stage_transition` marca `extracting_frames`/`interpolating_frames` como `done` — colapso honesto: esas etapas ya no existen como pasos separados.
- `framesDone` = frames ENTREGADOS al sink de encode (contador en memoria, monotónico, honesto — patrón `_track_streaming_progress` del raw-pipe actual).
- `job.metadata["framesTotal"]` se sobreescribe al arrancar el pipeline con el conteo de frames de SALIDA esperado (fuente × multiplicador o target count; `None` honesto si es indeterminable). Así el frontend muestra `N / M` correcto sin tocar `jobProgress.ts` ni usar `interpFramesTotal`.
- Al completar, `complete_video_stages` marca todo `done` (incluye `encoding_video`), igual que hoy con el raw-pipe.
- Cero cambios de frontend; cero claves nuevas de progreso. Claves informativas nuevas (no de progreso): `streamPipeline: true` en éxito, `streamPipelineFallback: "<motivo>"` en fallback (espejo de `rawPipe`/`rawPipeFallback`).

## Estructura de archivos (visión global)

| Archivo | Task | Responsabilidad |
|---|---|---|
| `app/config.py` (modif) | 1, 8 | Quitar `enable_interp_upscale_fusion`; agregar `enable_stream_pipeline` |
| `app/services/video_upscaler.py` (modif) | 1, 5, 8, 9, 10, 11 | Quitar fusión; ruteo de modos, `_try_stream_pipeline_*`, núcleo `_run_stream_pipeline` |
| `app/services/engines/gmfss_engine.py` (modif) | 1, 7 | Quitar `run_frames_fused`; `build_stream_stage` + `GmfssStreamStage` |
| `app/services/engines/onnx_video_upscaler.py` (modif) | 2, 6 | Extraer `derive_queue_maxsize`; `build_frame_upscaler` |
| `app/services/frame_pipeline.py` (nuevo) | 3, 9 | `FrameStage`, `MapStage`, `FramePipeline`, `derive_stream_queue_maxsizes`, `drain_stream`, `iter_png_frames` |
| `app/services/engines/ffmpeg_frame_source.py` (nuevo) | 4 | `FfmpegFrameSource`: decode rawvideo por pipe con seam fake |
| `app/services/engines/ffmpeg_frame_sink.py` (nuevo) | 5 | `RawPipeEncoder`: writer raw-pipe extraído, compartido con el camino clásico |
| `tests/test_video_upscaler.py` (borrar y recrear) | 1, 8, 9, 10, 11 | Gate de modos, híbrido, completo, GMFSS, fallback, cancel |
| `tests/test_gmfss_engine.py` (modif) | 1, 7 | Quitar tests de `run_frames_fused`; tests de `GmfssStreamStage` |
| `tests/test_onnx_video_upscaler.py` (modif) | 2, 6 | `derive_queue_maxsize`; `build_frame_upscaler` |
| `tests/test_frame_pipeline.py` (nuevo) | 3, 9 | Orden, 1→N, backpressure, errores, cancel, flush, maxsizes, `iter_png_frames` |
| `tests/test_ffmpeg_frame_source.py` (nuevo) | 4 | Comando, orden, truncado, exit≠0, kill-on-cancel (proceso fake) |
| `tests/test_ffmpeg_frame_sink.py` (nuevo) | 5 | Bytes crudos, conteo, finish/kill, tail de stderr (proceso fake) |
| `.env.example`, `README.md` (modif) | 1, 8, 12 | Quitar flag viejo, documentar el nuevo |

Convención de frames en TODO el pipeline: **NHWC uint8 RGB `[1,H,W,3]`** (el formato interno de `OnnxVideoUpscaler`). El sink le quita el batch (`frame[0]`, HWC) antes de `tobytes()` hacia ffmpeg — igual que el `_ordered_writer_loop` del raw-pipe actual.

---

### Task 1: Eliminar la fusión interpolar+escalar (`enable_interp_upscale_fusion`)

La fusión vieja midió ~1.7x MÁS LENTA que las dos pasadas (generador secuencial mono-hilo sin overlap — ver README "Benchmark real"). El spec la ELIMINA: este pipeline con threads la reemplaza. Se quita código, config, tests y docs en un solo task (es una remoción atómica: dejar mitades rompería imports).

**Files:**
- Modify: `app/services/video_upscaler.py` (quitar gate + camino fused, líneas 237-244 y 277-470 aprox.)
- Modify: `app/services/engines/gmfss_engine.py` (quitar `run_frames_fused` y helpers solo-fused)
- Modify: `app/config.py:347-356` (campo `enable_interp_upscale_fusion` + comentario)
- Modify: `.env.example:109` (línea `ENABLE_INTERP_UPSCALE_FUSION`)
- Modify: `README.md:420-433` (sección "Benchmark real: fusión interpolar+escalar")
- Delete: `tests/test_video_upscaler.py` (TODO su contenido actual es de la fusión; Task 8 lo recrea)
- Modify: `tests/test_gmfss_engine.py:283-379` (bloque `run_frames_fused`)

**Interfaces:**
- Consumes: nada nuevo.
- Produces: `video_upscaler.py` sin referencias a fusión — `_interpolate_and_upscale` arranca directo en `_maybe_interpolate`. `gmfss_engine.py` conserva `_prepare_pipeline`, `_chw_float_to_hwc_uint8`, `_rgb_to_padded_chw` (los usan `run()` y el Task 7). Los símbolos `_iter_interpolated_frames`, `_load_source_frame` y `_chw_float_to_nhwc_uint8` se eliminan; Task 7 re-agrega `_chw_float_to_nhwc_uint8` (one-liner) cuando vuelve a tener consumidor.

- [ ] **Step 1: Borrar los tests de la fusión**

```bash
git rm tests/test_video_upscaler.py
```

En `tests/test_gmfss_engine.py`, borrar el bloque completo entre el banner `# run_frames_fused (Fase 2 Task 7): ...` (línea ~283) y el final de `test_run_frames_fused_yields_nhwc_uint8_with_pixel_identical_source_frames` (línea ~379): las 3 funciones `test_run_frames_fused_calls_upscale_frame_for_every_output_frame`, `test_run_frames_fused_never_writes_intermediate_png`, `test_run_frames_fused_yields_nhwc_uint8_with_pixel_identical_source_frames` y su comentario-banner. No tocar nada más del archivo (los tests de `FP16_FUSIONNET_FILENAME` se QUEDAN — "fusionnet" es un grafo del modelo, no la fusión interp+upscale).

- [ ] **Step 2: Quitar el camino fused de `video_upscaler.py`**

Borrar íntegros estos métodos de `VideoUpscaler`:
- `_should_fuse_interpolate_upscale` (líneas ~277-305)
- `_run_fused_interpolate_upscale` (~307-342)
- `_run_fused_frames_shielded` (~344-379)
- `_run_fused_frames_blocking` (~381-416)
- `_build_builtin_onnx_upscale_callback` (~418-451)
- `_validate_fused_output_count` (~453-460)
- `_interpolated_encode_fps` (~462-470; Task 11 lo re-agrega con el mismo cuerpo cuando el pipeline GMFSS vuelve a necesitar esa cuenta)

En `_interpolate_and_upscale` (líneas ~211-244): borrar el bloque del gate fused (el `if await self._should_fuse_interpolate_upscale(...)` completo, líneas ~237-244) y el párrafo final del docstring que empieza con `Gate: when GMFSS interpolation is requested...` — el método queda arrancando directo en `upscale_src, encode_fps = await self._maybe_interpolate(...)`.

Limpiar imports que quedan huérfanos:
- Línea 10: `from collections.abc import AsyncIterator, Callable` → `from collections.abc import AsyncIterator`
- Línea 14: borrar `import numpy as np` (solo lo usaba el callback fused; Task 9 lo re-agrega si el linter lo exige para anotaciones nuevas)
- Líneas 19-23: quitar `get_builtin_onnx_model` del import de `backend_registry` (quedan `UpscaleBackend`, `resolve_upscale_backend`)
- Líneas 29-32: `from app.services.engines.onnx_video_upscaler import (OnnxVideoUpscaler, _save_frame as _onnx_video_save_frame)` → `from app.services.engines.onnx_video_upscaler import OnnxVideoUpscaler`

NO tocar: `import cv2` (lo usa `_output_dims` del raw-pipe), `compute_interpolated_fps`/`compute_target_frame_count`/`format_fps_fraction` (los usan `_interpolate_by_multiplier`/`_interpolate_to_target_fps`).

- [ ] **Step 3: Quitar el camino fused de `gmfss_engine.py`**

Borrar íntegros:
- `run_frames_fused` (líneas ~141-178)
- `_iter_interpolated_frames` (~226-254)
- `_load_source_frame` (~557-567)
- `_chw_float_to_nhwc_uint8` (~582-583)

Limpiar el import de línea 10: `from collections.abc import Callable, Iterator` → borrar la línea completa (ningún otro símbolo del módulo usa `Callable` ni `Iterator`).

NO tocar: `_prepare_pipeline` (lo usa `_run_blocking`), `_chw_float_to_hwc_uint8` (lo usa `_save_frame`), `_rgb_to_padded_chw`, `_decode_rgb`, `_load_padded_frame`.

- [ ] **Step 4: Quitar config y docs**

En `app/config.py`, borrar el campo y su comentario completo (líneas ~347-356):

```python
    # Fase 2 (Tasks 7-8): fusiona GMFSS interpolar + el escalador ONNX in-process
    # ... (todo el bloque de comentario) ...
    enable_interp_upscale_fusion: bool = Field(default=False, alias="ENABLE_INTERP_UPSCALE_FUSION")
```

En `.env.example`, borrar la línea 109 (`ENABLE_INTERP_UPSCALE_FUSION=False   # ...`).

En `README.md`, reemplazar la sección completa `### Benchmark real: fusión interpolar+escalar (Fase 2) vs dos pasadas (Fase 1)` (líneas 420-433, desde el heading hasta el párrafo `**Dado este resultado...**` inclusive) por esta nota breve:

```markdown
### Nota histórica: fusión interpolar+escalar (eliminada)

La fusión GMFSS+upscale en una pasada (`ENABLE_INTERP_UPSCALE_FUSION`, Fase 2) midió ~1.7x MÁS LENTA que las dos pasadas a 4x/8K en una RX 7800 XT: era un generador secuencial de un solo hilo sin overlap load/compute/save. Fue eliminada y reemplazada por el pipeline de frames en streaming (ver `docs/superpowers/specs/2026-07-25-stream-frame-pipeline-design.md`), que conecta las etapas por colas con un thread por etapa.
```

- [ ] **Step 5: Verificar que no quedan referencias**

Run: `grep -rn "interp_upscale_fusion\|INTERP_UPSCALE_FUSION\|run_frames_fused\|_run_fused\|_should_fuse" app tests README.md .env.example`
Expected: sin resultados (0 matches; `.claude/worktrees/**` queda fuera porque el grep se limita a `app tests README.md .env.example`).

- [ ] **Step 6: Correr la suite y verificar verde**

Run: `.venv\Scripts\python -m pytest tests/test_gmfss_engine.py tests/test_video_backend_dispatch.py tests/test_pipeline_stage_order.py -q && .venv\Scripts\python -m pytest -q`
Expected: PASS (ningún test restante importaba los símbolos borrados).

- [ ] **Step 7: Commit**

```bash
git add app/services/video_upscaler.py app/services/engines/gmfss_engine.py app/config.py .env.example README.md tests/test_gmfss_engine.py
git commit -m "refactor: elimina la fusión interpolar+escalar (ENABLE_INTERP_UPSCALE_FUSION), reemplazada por el stream pipeline"
```

(El `git rm tests/test_video_upscaler.py` del Step 1 ya está en el índice y entra en este mismo commit.)

---

### Task 2: Extraer el helper de presupuesto `derive_queue_maxsize`

La matemática de `ONNX_VIDEO_MAX_PIPELINE_MB` vive hoy dentro del método `OnnxVideoUpscaler._save_queue_maxsize`. El spec pide reusarla para TODAS las colas del pipeline nuevo sin duplicarla: se extrae a una función de módulo y el método existente la llama (semántica idéntica, tests existentes intactos).

**Files:**
- Modify: `app/services/engines/onnx_video_upscaler.py` (método `_save_queue_maxsize`, líneas ~418-433)
- Test: `tests/test_onnx_video_upscaler.py`

**Interfaces:**
- Produces: `derive_queue_maxsize(frame_bytes: int, budget_bytes: int, floor: int, ceiling: int) -> int` a nivel de módulo en `onnx_video_upscaler.py`. Task 3 la importa desde ahí (una sola dirección de import: `frame_pipeline` → `onnx_video_upscaler`, nunca al revés — sin ciclos).

- [ ] **Step 1: Escribir los tests que fallan**

En `tests/test_onnx_video_upscaler.py` (junto a los tests de `_save_queue_maxsize` existentes, línea ~355):

```python
from app.services.engines.onnx_video_upscaler import derive_queue_maxsize


def test_derive_queue_maxsize_budget_decides_between_floor_and_ceiling() -> None:
    # 48MB/frame, presupuesto 150MB -> 150//48 = 3, estrictamente entre piso 2 y techo 4.
    assert derive_queue_maxsize(48 * 1024 * 1024, 150 * 1024 * 1024, 2, 4) == 3


def test_derive_queue_maxsize_floors_and_ceils() -> None:
    assert derive_queue_maxsize(48 * 1024 * 1024, 1 * 1024 * 1024, 5, 10) == 5  # piso
    assert derive_queue_maxsize(1024, 1024 * 1024 * 1024, 2, 16) == 16  # techo


def test_derive_queue_maxsize_nonpositive_frame_bytes_returns_ceiling() -> None:
    # Sin tamaño de frame conocido no hay presupuesto que aplicar: techo (el
    # caso "sin frames" que _save_queue_maxsize ya resolvía con el default).
    assert derive_queue_maxsize(0, 100, 2, 8) == 8
    assert derive_queue_maxsize(-1, 100, 2, 8) == 8
```

- [ ] **Step 2: Correr y verificar que fallan**

Run: `.venv\Scripts\python -m pytest tests/test_onnx_video_upscaler.py -k derive_queue_maxsize -v`
Expected: FAIL con `ImportError: cannot import name 'derive_queue_maxsize'`

- [ ] **Step 3: Implementar**

En `onnx_video_upscaler.py`, a nivel de módulo (después de `_is_oom_error`, línea ~68):

```python
def derive_queue_maxsize(frame_bytes: int, budget_bytes: int, floor: int, ceiling: int) -> int:
    """Cuántos frames caben en cola bajo un presupuesto de RAM en bytes.

    Extraído de _save_queue_maxsize para compartirlo con las colas del stream
    pipeline (spec 2026-07-25-stream-frame-pipeline-design.md): mismo criterio,
    piso para no matar throughput y techo para no acumular de más.
    """
    if frame_bytes <= 0:
        return ceiling
    return max(floor, min(ceiling, budget_bytes // frame_bytes))
```

Y reescribir `_save_queue_maxsize` (líneas ~418-433) para delegarle la cuenta — mismo resultado que antes en todos los casos:

```python
    def _save_queue_maxsize(self, frame_paths: list[Path], scale: int, n_save: int) -> int:
        """Bound the save queue by a RAM budget instead of by thread count.

        Each queued item is a full 4x output frame (~44MB @ 5120x2880). Sizing by
        n_save*2 alone lets the queue hold ~1GB with no relation to memory, so we
        derive maxsize from ONNX_VIDEO_MAX_PIPELINE_MB and the real output frame
        size, with a floor of n_save so savers never starve.
        """
        default = n_save * 2
        if not frame_paths or scale < 1:
            return default
        out_bytes = self._output_frame_bytes(frame_paths[0], scale)
        budget_bytes = max(1, self.settings.onnx_video_max_pipeline_mb) * 1024 * 1024
        return derive_queue_maxsize(out_bytes, budget_bytes, floor=n_save, ceiling=default)
```

(El caso `out_bytes <= 0` que antes devolvía `default` ahora lo cubre `derive_queue_maxsize` devolviendo `ceiling`, que ES `default` — sin cambio de comportamiento.)

- [ ] **Step 4: Correr la suite del archivo y verificar verde**

Run: `.venv\Scripts\python -m pytest tests/test_onnx_video_upscaler.py -q`
Expected: PASS (incluye los 3 tests preexistentes de `_save_queue_maxsize`, sin modificarlos).

- [ ] **Step 5: Commit**

```bash
git add app/services/engines/onnx_video_upscaler.py tests/test_onnx_video_upscaler.py
git commit -m "refactor: extrae derive_queue_maxsize del presupuesto de cola del pipeline ONNX"
```

---

### Task 3: `frame_pipeline.py` — protocolo `FrameStage` y runner `FramePipeline`

El núcleo genérico: etapas conectadas por colas acotadas, un thread por etapa, backpressure por cola llena, orden garantizado (un solo thread por etapa + colas FIFO), teardown sin threads zombie. Sin imports de torch/onnx: solo stdlib + numpy. Reusa los helpers cancel-aware ya probados de `onnx_video_upscaler.py` (mismo precedente que `gmfss_engine.py`, que ya los importa de ahí).

**Files:**
- Create: `app/services/frame_pipeline.py`
- Test: `tests/test_frame_pipeline.py` (nuevo)

**Interfaces:**
- Consumes: `derive_queue_maxsize` (Task 2), `_put_until_cancelled`, `_drain_queue`, `_QUEUE_POLL_SECONDS`, `_THREAD_JOIN_TIMEOUT_SECONDS` de `app.services.engines.onnx_video_upscaler`.
- Produces (Tasks 4, 5, 7, 9, 10, 11 los consumen con estas firmas exactas):
  - protocolo `FrameStage`: `process(frame: np.ndarray) -> list[np.ndarray]`, `flush() -> list[np.ndarray]`
  - `MapStage(fn: Callable[[np.ndarray], np.ndarray])` (etapa 1→1)
  - `FramePipeline(source: Iterator[np.ndarray], stages: list[FrameStage], sink: Callable[[np.ndarray], None], queue_maxsizes: list[int])` con `run(cancel_event: threading.Event) -> int` (frames entregados al sink; re-lanza el primer error de cualquier etapa/source/sink tras el teardown)
  - `derive_stream_queue_maxsizes(input_frame_bytes: int, output_frame_bytes: int, n_stages: int, budget_bytes: int) -> list[int]` + constantes `STREAM_QUEUE_FLOOR = 2`, `STREAM_QUEUE_CEILING = 16`
  - `drain_stream(stream, sink: list[bytes]) -> None` (drenaje de stderr con tail acotado — cuerpo idéntico al `VideoUpscaler._drain_stream` actual; Task 5 elimina aquel y ambos procesos ffmpeg usan este)

- [ ] **Step 1: Escribir los tests que fallan (`tests/test_frame_pipeline.py`)**

```python
from __future__ import annotations

import threading

import numpy as np
import pytest

from app.services.frame_pipeline import (
    STREAM_QUEUE_CEILING,
    STREAM_QUEUE_FLOOR,
    FramePipeline,
    MapStage,
    derive_stream_queue_maxsizes,
)


def frame(value: int) -> np.ndarray:
    return np.full((1, 2, 3, 3), value % 256, dtype=np.uint8)


class DuplicateStage:
    """Etapa 1→2 (simula interpolación): emite el frame y una copia +100."""

    def process(self, f: np.ndarray) -> list[np.ndarray]:
        return [f, ((f.astype(np.int32) + 100) % 256).astype(np.uint8)]

    def flush(self) -> list[np.ndarray]:
        return []


class FlushStage:
    """Etapa que emite un frame extra recién en el flush (ventana tipo GMFSS)."""

    def process(self, f: np.ndarray) -> list[np.ndarray]:
        return [f]

    def flush(self) -> list[np.ndarray]:
        return [frame(99)]


class FailingStage:
    def __init__(self) -> None:
        self.seen = 0

    def process(self, f: np.ndarray) -> list[np.ndarray]:
        self.seen += 1
        if self.seen == 3:
            raise RuntimeError("boom en frame 3")
        return [f]

    def flush(self) -> list[np.ndarray]:
        return []


def run_pipeline(source_frames, stages, maxsizes=None):
    received: list[np.ndarray] = []
    pipeline = FramePipeline(
        iter(source_frames), stages, received.append, maxsizes or [2] * (len(stages) + 1)
    )
    delivered = pipeline.run(threading.Event())
    return delivered, received


def first_pixels(frames) -> list[int]:
    return [int(f[0, 0, 0, 0]) for f in frames]


def test_map_stage_preserves_count_and_order() -> None:
    delivered, received = run_pipeline([frame(i) for i in range(10)], [MapStage(lambda f: f * 2)])
    assert delivered == 10
    assert first_pixels(received) == [(i * 2) % 256 for i in range(10)]


def test_expanding_stage_emits_one_to_two_in_order() -> None:
    delivered, received = run_pipeline(
        [frame(i) for i in range(4)], [DuplicateStage(), MapStage(lambda f: f)]
    )
    assert delivered == 8
    assert first_pixels(received) == [0, 100, 1, 101, 2, 102, 3, 103]


def test_backpressure_with_maxsize_one_still_delivers_everything() -> None:
    delivered, _ = run_pipeline(
        [frame(i) for i in range(25)], [MapStage(lambda f: f)], maxsizes=[1, 1]
    )
    assert delivered == 25


def test_flush_frames_are_emitted_after_last_input() -> None:
    delivered, received = run_pipeline([frame(1)], [FlushStage()])
    assert delivered == 2
    assert first_pixels(received) == [1, 99]


def test_stage_error_propagates_and_joins_all_threads() -> None:
    threads_before = set(threading.enumerate())
    with pytest.raises(RuntimeError, match="boom en frame 3"):
        run_pipeline([frame(i) for i in range(10)], [FailingStage()])
    assert set(threading.enumerate()) <= threads_before, "quedó un thread vivo tras el error"


def test_source_error_propagates() -> None:
    def broken_source():
        yield frame(1)
        raise RuntimeError("decode roto")

    pipeline = FramePipeline(broken_source(), [MapStage(lambda f: f)], lambda f: None, [2, 2])
    with pytest.raises(RuntimeError, match="decode roto"):
        pipeline.run(threading.Event())


def test_sink_error_propagates() -> None:
    def bad_sink(f: np.ndarray) -> None:
        raise ValueError("sink roto")

    pipeline = FramePipeline(iter([frame(1)]), [MapStage(lambda f: f)], bad_sink, [2, 2])
    with pytest.raises(ValueError, match="sink roto"):
        pipeline.run(threading.Event())


def test_preset_cancel_delivers_nothing_and_leaks_no_threads() -> None:
    cancel = threading.Event()
    cancel.set()
    received: list[np.ndarray] = []
    threads_before = set(threading.enumerate())
    pipeline = FramePipeline(
        iter([frame(i) for i in range(5)]), [MapStage(lambda f: f)], received.append, [2, 2]
    )
    delivered = pipeline.run(cancel)
    assert delivered == 0
    assert received == []
    assert set(threading.enumerate()) <= threads_before


def test_queue_maxsizes_must_match_stage_count() -> None:
    with pytest.raises(ValueError, match="queue_maxsizes"):
        FramePipeline(iter([]), [MapStage(lambda f: f)], lambda f: None, [2])


def test_derive_stream_queue_maxsizes_splits_budget_globally() -> None:
    # Entrada 720p (1280x720x3 ≈ 2.6MB), salida 4x (16x px ≈ 42.2MB), 1 etapa
    # => 2 colas, presupuesto 256MB => 128MB por cola: la de entrada satura el
    # techo (16), la de salida da 128MB // 42.2MB = 3.
    input_bytes = 1280 * 720 * 3
    sizes = derive_stream_queue_maxsizes(input_bytes, input_bytes * 16, 1, 256 * 1024 * 1024)
    assert sizes == [STREAM_QUEUE_CEILING, 3]


def test_derive_stream_queue_maxsizes_floors_tiny_budget() -> None:
    sizes = derive_stream_queue_maxsizes(10_000_000, 160_000_000, 2, 1024)
    assert sizes == [STREAM_QUEUE_FLOOR] * 3
```

- [ ] **Step 2: Correr y verificar que fallan**

Run: `.venv\Scripts\python -m pytest tests/test_frame_pipeline.py -v`
Expected: FAIL con `ModuleNotFoundError: No module named 'app.services.frame_pipeline'`

- [ ] **Step 3: Implementar `app/services/frame_pipeline.py`**

```python
from __future__ import annotations

import logging
import queue
import threading
from collections.abc import Callable, Iterator
from typing import Protocol

import numpy as np

from app.services.engines.onnx_video_upscaler import (
    _QUEUE_POLL_SECONDS,
    _THREAD_JOIN_TIMEOUT_SECONDS,
    _drain_queue,
    _put_until_cancelled,
    derive_queue_maxsize,
)

logger = logging.getLogger(__name__)

# Piso/techo por cola del stream pipeline: el piso evita matar el overlap con
# presupuestos chicos; el techo evita que frames diminutos acumulen cientos de
# entradas sin beneficio (el productor solo necesita ir un puñado adelante).
STREAM_QUEUE_FLOOR = 2
STREAM_QUEUE_CEILING = 16


class FrameStage(Protocol):
    """Etapa del pipeline: recibe un frame NHWC uint8 RGB [1,H,W,3] y emite
    0..N frames en orden (upscaler 1→1; interpolador 1→1+extras con ventana de
    2). flush() emite lo retenido al agotarse la fuente."""

    def process(self, frame: np.ndarray) -> list[np.ndarray]: ...

    def flush(self) -> list[np.ndarray]: ...


class MapStage:
    """Etapa 1→1 sobre un callable puro por frame (p.ej. el closure de upscale
    de OnnxVideoUpscaler.build_frame_upscaler)."""

    def __init__(self, fn: Callable[[np.ndarray], np.ndarray]) -> None:
        self._fn = fn

    def process(self, frame: np.ndarray) -> list[np.ndarray]:
        return [self._fn(frame)]

    def flush(self) -> list[np.ndarray]:
        return []


def derive_stream_queue_maxsizes(
    input_frame_bytes: int, output_frame_bytes: int, n_stages: int, budget_bytes: int
) -> list[int]:
    """maxsize por cola bajo presupuesto GLOBAL (spec: repartido entre TODAS las
    colas del pipeline, no por cola). n_stages etapas => n_stages+1 colas; la
    ÚLTIMA transporta frames de salida (scale² más grandes), el resto de entrada.
    """
    n_queues = n_stages + 1
    per_queue_budget = max(1, budget_bytes // n_queues)
    sizes = [
        derive_queue_maxsize(input_frame_bytes, per_queue_budget, STREAM_QUEUE_FLOOR, STREAM_QUEUE_CEILING)
        for _ in range(n_queues - 1)
    ]
    sizes.append(
        derive_queue_maxsize(output_frame_bytes, per_queue_budget, STREAM_QUEUE_FLOOR, STREAM_QUEUE_CEILING)
    )
    return sizes


def drain_stream(stream, sink: list[bytes]) -> None:
    # ffmpeg llena su pipe de stderr; si nadie lo drena, el pipe de datos se
    # bloquea cuando el buffer se llena. Se conserva solo la cola para errores.
    # (Cuerpo idéntico al VideoUpscaler._drain_stream que Task 5 elimina.)
    try:
        for chunk in iter(lambda: stream.read(8192), b""):
            sink.append(chunk)
            if len(sink) > 64:
                del sink[:-64]
    except Exception:  # noqa: BLE001 - stream cerrado en un kill
        pass


class FramePipeline:
    """Etapas conectadas por colas acotadas, cada una en su propio thread.

    Backpressure: cola llena ⇒ el productor bloquea (vía _put_until_cancelled,
    que observa cancel_event — nunca un put ciego). Orden garantizado: un solo
    thread por etapa + colas FIFO. Fin de stream: sentinel None que cada etapa
    reenvía tras emitir su flush(). El sink corre en el thread llamador.
    """

    def __init__(
        self,
        source: Iterator[np.ndarray],
        stages: list[FrameStage],
        sink: Callable[[np.ndarray], None],
        queue_maxsizes: list[int],
    ) -> None:
        if len(queue_maxsizes) != len(stages) + 1:
            raise ValueError("queue_maxsizes debe tener len(stages) + 1 entradas")
        self._source = source
        self._stages = list(stages)
        self._sink = sink
        self._queues: list[queue.Queue] = [queue.Queue(maxsize=size) for size in queue_maxsizes]
        self._errors: list[Exception] = []

    def run(self, cancel_event: threading.Event) -> int:
        threads = [
            threading.Thread(
                target=self._source_loop, args=(cancel_event,), daemon=True, name="frame-pipeline-source"
            )
        ]
        for index, stage in enumerate(self._stages):
            threads.append(
                threading.Thread(
                    target=self._stage_loop,
                    args=(stage, self._queues[index], self._queues[index + 1], cancel_event),
                    daemon=True,
                    name=f"frame-pipeline-stage-{index}",
                )
            )
        for thread in threads:
            thread.start()
        try:
            delivered = self._sink_loop(cancel_event)
        finally:
            # Mismo teardown que OnnxVideoUpscaler._run_pipeline: drenar para
            # desbloquear productores y esperar los threads SIEMPRE, también en
            # el camino de error/cancel, para no filtrar threads zombie.
            if self._errors:
                cancel_event.set()
            for pending_queue in self._queues:
                _drain_queue(pending_queue)
            for thread in threads:
                thread.join(timeout=_THREAD_JOIN_TIMEOUT_SECONDS)
                if thread.is_alive():
                    logger.error("frame pipeline thread did not stop within timeout: %s", thread.name)
        if self._errors:
            raise self._errors[0]
        return delivered

    def _source_loop(self, cancel_event: threading.Event) -> None:
        out_q = self._queues[0]
        try:
            for source_frame in self._source:
                if cancel_event.is_set():
                    return
                if not _put_until_cancelled(out_q, source_frame, cancel_event):
                    return
        except Exception as exc:  # noqa: BLE001 - un decode roto es error de pipeline, no crash
            self._fail(exc, cancel_event)
            return
        _put_until_cancelled(out_q, None, cancel_event)

    def _stage_loop(
        self,
        stage: FrameStage,
        in_q: queue.Queue,
        out_q: queue.Queue,
        cancel_event: threading.Event,
    ) -> None:
        while True:
            if cancel_event.is_set() or self._errors:
                return
            try:
                item = in_q.get(timeout=_QUEUE_POLL_SECONDS)
            except queue.Empty:
                continue
            try:
                outputs = stage.flush() if item is None else stage.process(item)
            except Exception as exc:  # noqa: BLE001
                self._fail(exc, cancel_event)
                return
            for output_frame in outputs:
                if not _put_until_cancelled(out_q, output_frame, cancel_event):
                    return
            if item is None:
                _put_until_cancelled(out_q, None, cancel_event)
                return

    def _sink_loop(self, cancel_event: threading.Event) -> int:
        delivered = 0
        last_q = self._queues[-1]
        while True:
            if cancel_event.is_set() or self._errors:
                return delivered
            try:
                item = last_q.get(timeout=_QUEUE_POLL_SECONDS)
            except queue.Empty:
                continue
            if item is None:
                return delivered
            try:
                self._sink(item)
            except Exception as exc:  # noqa: BLE001 - ffmpeg muerto / broken pipe
                self._fail(exc, cancel_event)
                return delivered
            delivered += 1

    def _fail(self, exc: Exception, cancel_event: threading.Event) -> None:
        self._errors.append(exc)
        cancel_event.set()
```

- [ ] **Step 4: Correr y verificar verde**

Run: `.venv\Scripts\python -m pytest tests/test_frame_pipeline.py tests/test_onnx_video_upscaler.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add app/services/frame_pipeline.py tests/test_frame_pipeline.py
git commit -m "feat: FramePipeline con etapas en threads y colas acotadas por presupuesto"
```

---

### Task 4: `FfmpegFrameSource` — decode rawvideo por pipe

Espejo del writer raw-pipe de salida, pero en la dirección de entrada: `ffmpeg -i src ... -f rawvideo -pix_fmt rgb24 pipe:1`, leído frame a frame por el thread fuente del pipeline. Resolución del probe ya capturado en el job (el caller la pasa). El seam `_spawn` es monkeypatcheable para que los unit tests usen un proceso fake (nunca el binario real).

**Files:**
- Create: `app/services/engines/ffmpeg_frame_source.py`
- Test: `tests/test_ffmpeg_frame_source.py` (nuevo)

**Interfaces:**
- Consumes: `drain_stream` (Task 3).
- Produces: `FfmpegFrameSource(ffmpeg_binary: Path, source_path: Path, width: int, height: int, decode_threads: int)` con `build_command() -> list[str]` y `frames(cancel_event: threading.Event) -> Iterator[np.ndarray]` (frames NHWC uint8 `[1,H,W,3]` en orden; RuntimeError en truncado/exit≠0; kill del proceso en cancel o error). Task 10 lo consume como source del pipeline completo.

- [ ] **Step 1: Escribir los tests que fallan (`tests/test_ffmpeg_frame_source.py`)**

```python
from __future__ import annotations

import io
import threading
from pathlib import Path

import numpy as np
import pytest

from app.services.engines.ffmpeg_frame_source import FfmpegFrameSource


class FakeDecodeProc:
    """Popen fake: stdout con frames rgb24 crudos pre-armados, stderr fake."""

    def __init__(self, stdout_bytes: bytes, returncode: int = 0) -> None:
        self.stdout = io.BytesIO(stdout_bytes)
        self.stderr = io.BytesIO(b"fake ffmpeg stderr line")
        self.returncode: int | None = None
        self._final_returncode = returncode
        self.killed = False

    def wait(self, timeout: float | None = None) -> int:
        self.returncode = self._final_returncode
        return self.returncode

    def poll(self) -> int | None:
        return self.returncode

    def kill(self) -> None:
        self.killed = True
        self._final_returncode = -9


def make_source(tmp_path: Path, width: int = 4, height: int = 2) -> FfmpegFrameSource:
    return FfmpegFrameSource(Path("ffmpeg.exe"), tmp_path / "clip.mp4", width, height, decode_threads=2)


def raw_frames(count: int, width: int, height: int) -> bytes:
    # Frame i = todos los bytes en (i % 256): orden verificable por el primer byte.
    return b"".join(bytes([i % 256]) * (width * height * 3) for i in range(count))


def test_build_command_uses_fps_mode_passthrough_never_vsync(tmp_path: Path) -> None:
    command = make_source(tmp_path).build_command()
    assert "-vsync" not in command  # flag deprecado, prohibido por el spec
    assert command[command.index("-fps_mode") + 1] == "passthrough"
    assert command[command.index("-pix_fmt") + 1] == "rgb24"
    assert command[command.index("-f") + 1] == "rawvideo"
    assert command[-1] == "pipe:1"


def test_frames_yields_nhwc_uint8_in_order(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = make_source(tmp_path, width=4, height=2)
    fake = FakeDecodeProc(raw_frames(3, 4, 2))
    monkeypatch.setattr(source, "_spawn", lambda command: fake)

    frames = list(source.frames(threading.Event()))

    assert len(frames) == 3
    assert all(f.shape == (1, 2, 4, 3) and f.dtype == np.uint8 for f in frames)
    assert [int(f[0, 0, 0, 0]) for f in frames] == [0, 1, 2]


def test_frames_raises_on_truncated_tail(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = make_source(tmp_path, width=4, height=2)
    fake = FakeDecodeProc(raw_frames(1, 4, 2) + b"\x00" * 5)  # 5 bytes sueltos al final
    monkeypatch.setattr(source, "_spawn", lambda command: fake)

    with pytest.raises(RuntimeError, match="truncado"):
        list(source.frames(threading.Event()))
    assert fake.killed is True  # el proceso no queda huérfano tras el error


def test_frames_raises_when_ffmpeg_exits_nonzero(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = make_source(tmp_path, width=4, height=2)
    fake = FakeDecodeProc(raw_frames(2, 4, 2), returncode=1)
    monkeypatch.setattr(source, "_spawn", lambda command: fake)

    with pytest.raises(RuntimeError, match="exit 1"):
        list(source.frames(threading.Event()))


def test_frames_kills_process_when_cancelled(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = make_source(tmp_path, width=4, height=2)
    fake = FakeDecodeProc(raw_frames(10, 4, 2))
    monkeypatch.setattr(source, "_spawn", lambda command: fake)
    cancel = threading.Event()

    iterator = source.frames(cancel)
    next(iterator)
    cancel.set()
    remaining = list(iterator)

    assert remaining == []
    assert fake.killed is True
```

- [ ] **Step 2: Correr y verificar que fallan**

Run: `.venv\Scripts\python -m pytest tests/test_ffmpeg_frame_source.py -v`
Expected: FAIL con `ModuleNotFoundError: No module named 'app.services.engines.ffmpeg_frame_source'`

- [ ] **Step 3: Implementar `app/services/engines/ffmpeg_frame_source.py`**

```python
from __future__ import annotations

import subprocess
import threading
from collections.abc import Iterator
from pathlib import Path

import numpy as np

from app.services.frame_pipeline import drain_stream

_RGB24_BYTES_PER_PIXEL = 3
_STDERR_TAIL_CHARS = 500


class FfmpegFrameSource:
    """Decodea el video a frames rgb24 crudos por pipe (espejo del writer
    raw-pipe de salida ya probado en producción): cero PNGs intermedios y cero
    dependencias nuevas. Un thread del pipeline itera frames(); en cancel o
    error el proceso ffmpeg se mata SIEMPRE (nunca queda huérfano)."""

    def __init__(
        self, ffmpeg_binary: Path, source_path: Path, width: int, height: int, decode_threads: int
    ) -> None:
        self._ffmpeg_binary = ffmpeg_binary
        self._source_path = source_path
        self._width = width
        self._height = height
        self._decode_threads = decode_threads
        self._frame_bytes = width * height * _RGB24_BYTES_PER_PIXEL

    def build_command(self) -> list[str]:
        # Solo flags no-deprecados: -fps_mode passthrough (nunca -vsync), el
        # mismo passthrough que usa la extracción PNG actual para no
        # re-muestrear fuentes VFR.
        return [
            str(self._ffmpeg_binary),
            "-v", "error",
            "-i", str(self._source_path),
            "-fps_mode", "passthrough",
            "-threads", str(self._decode_threads),
            "-f", "rawvideo",
            "-pix_fmt", "rgb24",
            "pipe:1",
        ]

    def frames(self, cancel_event: threading.Event) -> Iterator[np.ndarray]:
        proc = self._spawn(self.build_command())
        stderr_buf: list[bytes] = []
        stderr_thread = threading.Thread(
            target=drain_stream, args=(proc.stderr, stderr_buf), daemon=True
        )
        stderr_thread.start()
        try:
            while not cancel_event.is_set():
                chunk = self._read_exact(proc.stdout, self._frame_bytes)
                if chunk is None:
                    break  # EOF limpio en un límite de frame
                yield self._to_frame(chunk)
            if cancel_event.is_set():
                return
            returncode = proc.wait()
            if returncode != 0:
                tail = b"".join(stderr_buf).decode("utf-8", errors="ignore").strip()
                raise RuntimeError(f"ffmpeg decode failed (exit {returncode}): {tail[-_STDERR_TAIL_CHARS:]}")
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()
            stderr_thread.join(timeout=5)

    def _spawn(self, command: list[str]) -> subprocess.Popen:
        # Seam monkeypatcheable: los unit tests lo reemplazan por un proceso
        # fake (nunca corre el binario real en unit).
        return subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    def _read_exact(self, stream, size: int) -> bytes | None:
        chunks: list[bytes] = []
        remaining = size
        while remaining > 0:
            data = stream.read(remaining)
            if not data:
                if chunks:
                    raise RuntimeError(
                        f"ffmpeg decode truncado: frame incompleto de {size - remaining} bytes"
                    )
                return None
            chunks.append(data)
            remaining -= len(data)
        return b"".join(chunks)

    def _to_frame(self, chunk: bytes) -> np.ndarray:
        frame = np.frombuffer(chunk, dtype=np.uint8).reshape(self._height, self._width, 3)
        # copy(): frombuffer devuelve un array read-only; aguas abajo GMFSS/ONNX
        # esperan poder pedir contigüidad/escritura sin sorpresas.
        return frame[np.newaxis, ...].copy()
```

- [ ] **Step 4: Correr y verificar verde**

Run: `.venv\Scripts\python -m pytest tests/test_ffmpeg_frame_source.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add app/services/engines/ffmpeg_frame_source.py tests/test_ffmpeg_frame_source.py
git commit -m "feat: FfmpegFrameSource decodea frames rgb24 por pipe sin PNGs intermedios"
```

---

### Task 5: `RawPipeEncoder` — extraer el writer raw-pipe y compartirlo

El spec exige que el sink sea el writer raw-pipe existente "extraído/reusado, no duplicado". Se extrae la gestión del proceso ffmpeg de encode (`Popen` + stdin + drenaje de stderr + kill/wait) de `VideoUpscaler._upscale_encode_streaming` a una clase síncrona reutilizable, y el camino raw-pipe clásico (flag OFF del pipeline) pasa a usarla — cero regresión verificada por los tests existentes de ese camino.

**Files:**
- Create: `app/services/engines/ffmpeg_frame_sink.py`
- Modify: `app/services/video_upscaler.py` (`_upscale_encode_streaming` líneas ~1155-1210, `_drain_stream` ~1212-1222)
- Test: `tests/test_ffmpeg_frame_sink.py` (nuevo)

**Interfaces:**
- Consumes: `drain_stream` (Task 3).
- Produces: `RawPipeEncoder(command: list[str], summarize_error: Callable[[bytes], str] | None = None)` con `start() -> None`, `write_frame(frame_hwc: np.ndarray) -> None` (HWC uint8 RGB; bloquea con el backpressure del pipe), `frames_written: int`, `finish() -> None` (cierra stdin, espera; RuntimeError con `summarize_error(stderr)` si exit≠0), `kill() -> None` (idempotente, seguro pre-start). Tasks 9/10 lo consumen como sink del pipeline.

- [ ] **Step 1: Escribir los tests que fallan (`tests/test_ffmpeg_frame_sink.py`)**

```python
from __future__ import annotations

import io

import numpy as np
import pytest

from app.services.engines.ffmpeg_frame_sink import RawPipeEncoder


class FakeStdin(io.BytesIO):
    """BytesIO cuyo close() solo marca la bandera: el buffer sigue legible
    para que el test inspeccione los bytes escritos."""

    def __init__(self) -> None:
        super().__init__()
        self.closed_by_encoder = False

    def close(self) -> None:  # type: ignore[override]
        self.closed_by_encoder = True


class FakeEncodeProc:
    def __init__(self, returncode: int = 0) -> None:
        self.stdin = FakeStdin()
        self.stderr = io.BytesIO(b"ffmpeg noise\nultima linea util")
        self.returncode: int | None = None
        self._final_returncode = returncode
        self.killed = False

    def wait(self, timeout: float | None = None) -> int:
        self.returncode = self._final_returncode
        return self.returncode

    def poll(self) -> int | None:
        return self.returncode

    def kill(self) -> None:
        self.killed = True
        self._final_returncode = -9


def make_encoder(
    monkeypatch: pytest.MonkeyPatch, returncode: int = 0
) -> tuple[RawPipeEncoder, FakeEncodeProc]:
    encoder = RawPipeEncoder(["ffmpeg.exe", "-fake"])
    fake = FakeEncodeProc(returncode)
    monkeypatch.setattr(encoder, "_spawn", lambda command: fake)
    encoder.start()
    return encoder, fake


def test_write_frame_pipes_raw_bytes_and_counts(monkeypatch: pytest.MonkeyPatch) -> None:
    encoder, fake = make_encoder(monkeypatch)
    frame = np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3)

    encoder.write_frame(frame)
    encoder.write_frame(frame)

    assert encoder.frames_written == 2
    assert fake.stdin.getvalue() == frame.tobytes() * 2


def test_finish_closes_stdin_and_passes_on_exit_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    encoder, fake = make_encoder(monkeypatch)
    encoder.finish()
    assert fake.stdin.closed_by_encoder is True


def test_finish_raises_with_stderr_tail_on_nonzero_exit(monkeypatch: pytest.MonkeyPatch) -> None:
    encoder, fake = make_encoder(monkeypatch, returncode=1)
    with pytest.raises(RuntimeError, match="ultima linea util"):
        encoder.finish()


def test_finish_uses_injected_summarizer(monkeypatch: pytest.MonkeyPatch) -> None:
    encoder = RawPipeEncoder(["ffmpeg.exe"], summarize_error=lambda stderr: "mensaje amigable")
    fake = FakeEncodeProc(returncode=1)
    monkeypatch.setattr(encoder, "_spawn", lambda command: fake)
    encoder.start()
    with pytest.raises(RuntimeError, match="mensaje amigable"):
        encoder.finish()


def test_kill_kills_live_process_and_is_idempotent(monkeypatch: pytest.MonkeyPatch) -> None:
    encoder, fake = make_encoder(monkeypatch)
    encoder.kill()
    encoder.kill()  # segunda llamada: no debe lanzar
    assert fake.killed is True


def test_kill_before_start_is_a_noop() -> None:
    RawPipeEncoder(["ffmpeg.exe"]).kill()  # no debe lanzar
```

- [ ] **Step 2: Correr y verificar que fallan**

Run: `.venv\Scripts\python -m pytest tests/test_ffmpeg_frame_sink.py -v`
Expected: FAIL con `ModuleNotFoundError: No module named 'app.services.engines.ffmpeg_frame_sink'`

- [ ] **Step 3: Implementar `app/services/engines/ffmpeg_frame_sink.py`**

```python
from __future__ import annotations

import subprocess
import threading
from collections.abc import Callable

import numpy as np

from app.services.frame_pipeline import drain_stream


def _default_summarize(stderr: bytes) -> str:
    text = stderr.decode("utf-8", errors="ignore")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else "ffmpeg encode failed"


class RawPipeEncoder:
    """Proceso ffmpeg de encode alimentado por stdin con frames rgb24 crudos.

    Extraído de VideoUpscaler._upscale_encode_streaming para que el raw-pipe
    clásico y el stream pipeline compartan el MISMO writer (spec: extraído/
    reusado, no duplicado). write_frame bloquea con el backpressure natural del
    pipe — por eso siempre se llama desde un worker thread, nunca del loop.
    """

    def __init__(
        self, command: list[str], summarize_error: Callable[[bytes], str] | None = None
    ) -> None:
        self._command = command
        self._summarize = summarize_error or _default_summarize
        self._proc: subprocess.Popen | None = None
        self._stderr_buf: list[bytes] = []
        self._stderr_thread: threading.Thread | None = None
        self.frames_written = 0

    def start(self) -> None:
        self._proc = self._spawn(self._command)
        self._stderr_thread = threading.Thread(
            target=drain_stream, args=(self._proc.stderr, self._stderr_buf), daemon=True
        )
        self._stderr_thread.start()

    def _spawn(self, command: list[str]) -> subprocess.Popen:
        # Seam monkeypatcheable: los unit tests lo reemplazan por un proceso fake.
        return subprocess.Popen(
            command, stdin=subprocess.PIPE, stderr=subprocess.PIPE, stdout=subprocess.DEVNULL
        )

    def write_frame(self, frame_hwc: np.ndarray) -> None:
        assert self._proc is not None, "write_frame antes de start()"
        self._proc.stdin.write(frame_hwc.tobytes())
        self.frames_written += 1

    def finish(self) -> None:
        assert self._proc is not None, "finish antes de start()"
        self._proc.stdin.close()
        returncode = self._proc.wait()
        self._join_stderr()
        if returncode != 0:
            raise RuntimeError(self._summarize(b"".join(self._stderr_buf)))

    def kill(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            self._proc.kill()
            self._proc.wait()
        self._join_stderr()

    def _join_stderr(self) -> None:
        if self._stderr_thread is not None:
            self._stderr_thread.join(timeout=5)
```

- [ ] **Step 4: Correr y verificar verde**

Run: `.venv\Scripts\python -m pytest tests/test_ffmpeg_frame_sink.py -q`
Expected: PASS

- [ ] **Step 5: Refactorizar `_upscale_encode_streaming` para usar `RawPipeEncoder`**

En `app/services/video_upscaler.py`:

1. Import nuevo junto a los de engines: `from app.services.engines.ffmpeg_frame_sink import RawPipeEncoder`.
2. Reemplazar el cuerpo de `_upscale_encode_streaming` (misma firma, mismos parámetros):

```python
    async def _upscale_encode_streaming(
        self,
        job: VideoUpscaleJob,
        frames_in: Path,
        output_path: Path,
        encoder: str,
        fps: str,
        audio_mux_path: Path | None,
        audio_codec_args: list[str],
        out_w: int,
        out_h: int,
    ) -> None:
        cmd = self._build_rawpipe_command(
            out_w, out_h, fps, audio_mux_path, audio_codec_args, output_path, job, encoder
        )
        # Mismo resumen de errores que el encode PNG (mensajes x265 amigables, etc.).
        pipe_encoder = RawPipeEncoder(cmd, summarize_error=lambda stderr: self._summarize_process_error(stderr, b""))
        pipe_encoder.start()

        counter = {"n": 0}

        def write_frame(frame_hwc_rgb) -> None:
            pipe_encoder.write_frame(frame_hwc_rgb)  # blocks on pipe backpressure (worker thread)
            counter["n"] = pipe_encoder.frames_written

        device = job.device or self.settings.default_device
        # Shield the engine task so a job cancel doesn't tear it down while a worker
        # thread is blocked writing to ffmpeg's pipe. We kill ffmpeg FIRST (which
        # unblocks that write with BrokenPipe so the engine unwinds), then await it.
        stream_task = asyncio.ensure_future(
            self.onnx_video_engine.run_frames_streaming(frames_in, job.model_name, device, write_frame)
        )
        try:
            async with self._track_streaming_progress(job, counter):
                expected = await asyncio.shield(stream_task)
            if counter["n"] != expected:
                raise RuntimeError(f"raw-pipe wrote {counter['n']}/{expected} frames")
            await asyncio.to_thread(pipe_encoder.finish)
        except BaseException:
            pipe_encoder.kill()
            with contextlib.suppress(BaseException):
                await stream_task
            raise
```

3. Borrar el método `_drain_stream` de `VideoUpscaler` (ya no tiene llamadores: el drenaje vive en `frame_pipeline.drain_stream` vía `RawPipeEncoder`). El import `subprocess` de la línea 8 queda sin uso en `video_upscaler.py` tras este cambio — quitarlo. `threading` SIGUE usado (`_run_fused` ya no existe, pero `threading.Thread`… verificar: tras Task 1 y este paso, `threading` solo lo usaba el raw-pipe viejo; Tasks 9-10 lo vuelven a usar para `cancel_event` — si el linter lo marca acá, quitarlo y re-agregarlo en Task 9).

- [ ] **Step 6: Correr la regresión del raw-pipe clásico y verificar verde**

Run: `.venv\Scripts\python -m pytest tests/test_video_encoder_dispatch.py tests/test_pipeline_stage_order.py tests/test_onnx_video_upscaler.py tests/test_ffmpeg_frame_sink.py -q`
Expected: PASS (el camino raw-pipe con flag OFF se comporta idéntico: mismos comandos, mismo conteo, mismos mensajes de error).

- [ ] **Step 7: Commit**

```bash
git add app/services/engines/ffmpeg_frame_sink.py app/services/video_upscaler.py tests/test_ffmpeg_frame_sink.py
git commit -m "refactor: extrae RawPipeEncoder y lo comparte con el raw-pipe clásico"
```

---

### Task 6: `OnnxVideoUpscaler.build_frame_upscaler` — closure por-frame del upscaler

La etapa de upscale del pipeline es un `MapStage` sobre un closure que reusa los internals EXACTOS del motor: misma sesión (`_get_session`, con `GpuSessionCoordinator.acquire` adentro — la serialización GPU existente), misma selección fp16 (`_select_model_file`), mismo `_upscale_one` (whole-frame → tiling sticky tras un OOM). Es la versión "método público del motor" del callback que la fusión eliminada construía manoseando privados desde `video_upscaler`.

**Files:**
- Modify: `app/services/engines/onnx_video_upscaler.py` (método nuevo después de `run_frames_streaming`, línea ~296)
- Test: `tests/test_onnx_video_upscaler.py`

**Interfaces:**
- Consumes: internals propios del motor (`_get_session`, `_select_model_file`, `_upscale_one`, `available`, `devices.validate`).
- Produces: `OnnxVideoUpscaler.build_frame_upscaler(engine_model_name: str, device: str) -> Callable[[np.ndarray], np.ndarray]` (NHWC uint8 → NHWC uint8 escalado). Tasks 9/10/11 lo envuelven en `MapStage`.

- [ ] **Step 1: Escribir los tests que fallan**

En `tests/test_onnx_video_upscaler.py` (usa `make_engine`, `touch_builtin_onnx` y `Double2xUint8Session` ya definidos en el archivo):

```python
def test_build_frame_upscaler_returns_working_closure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine = make_engine(tmp_path)
    touch_builtin_onnx(engine.settings, "realesr-animevideov3-x4-uint8.onnx")
    monkeypatch.setattr(engine, "_create_session", lambda model_path, device: Double2xUint8Session())

    upscale = engine.build_frame_upscaler("realesr-animevideov3-x4", "cpu")
    frame = np.random.default_rng(3).integers(0, 256, (1, 4, 6, 3), dtype=np.uint8)
    out = upscale(frame)

    assert out.shape == (1, 8, 12, 3)
    assert out.dtype == np.uint8


def test_build_frame_upscaler_raises_for_unconfigured_model(tmp_path: Path) -> None:
    engine = make_engine(tmp_path)
    with pytest.raises(RuntimeError, match="No ONNX export configured"):
        engine.build_frame_upscaler("does-not-exist", "cpu")


def test_build_frame_upscaler_raises_when_model_file_missing(tmp_path: Path) -> None:
    engine = make_engine(tmp_path)  # sin touch_builtin_onnx -> archivo ausente
    with pytest.raises(RuntimeError, match="ONNX model file not found"):
        engine.build_frame_upscaler("realesr-animevideov3-x4", "cpu")


def test_build_frame_upscaler_sticks_to_tiling_after_oom(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Mismo contrato sticky que _infer_loop: un OOM whole-frame degrada el RESTO
    # del run a tiling (el estado vive en el closure, un solo intento whole).
    engine = make_engine(tmp_path)
    touch_builtin_onnx(engine.settings, "realesr-animevideov3-x4-uint8.onnx")
    monkeypatch.setattr(engine, "_create_session", lambda model_path, device: Double2xUint8Session())
    calls = {"whole": 0, "tiled": 0}

    def whole_frame_oom(s, f, d):
        calls["whole"] += 1
        raise RuntimeError("Failed to allocate memory: out of memory (D3D12)")

    monkeypatch.setattr(engine, "_infer_frame", whole_frame_oom)
    monkeypatch.setattr(
        engine, "_infer_tiled", lambda s, f, d: (calls.__setitem__("tiled", calls["tiled"] + 1), f)[1]
    )

    upscale = engine.build_frame_upscaler("realesr-animevideov3-x4", "cpu")
    frame = np.zeros((1, 4, 6, 3), dtype=np.uint8)
    upscale(frame)
    upscale(frame)

    assert calls == {"whole": 1, "tiled": 2}  # el 2o frame va directo a tiling
```

- [ ] **Step 2: Correr y verificar que fallan**

Run: `.venv\Scripts\python -m pytest tests/test_onnx_video_upscaler.py -k build_frame_upscaler -v`
Expected: FAIL con `AttributeError: 'OnnxVideoUpscaler' object has no attribute 'build_frame_upscaler'`

- [ ] **Step 3: Implementar**

En `onnx_video_upscaler.py`, después de `run_frames_streaming` (línea ~296):

```python
    # --- per-frame closure for the stream pipeline ---------------------------

    def build_frame_upscaler(self, engine_model_name: str, device: str) -> "Callable[[np.ndarray], np.ndarray]":
        """Closure NHWC uint8 → NHWC uint8 sobre la MISMA sesión/tiling/fp16 que
        run_frames_builtin: la etapa de upscale del stream pipeline (MapStage).

        La sesión se resuelve UNA vez acá (cache + GpuSessionCoordinator.acquire
        — la serialización GPU existente); el flag sticky de tiling replica el
        contrato de _infer_loop para el resto del run.
        """
        if not self.available():
            raise RuntimeError("ONNX video engine is not available: onnxruntime and opencv are required")
        model = get_builtin_onnx_model(engine_model_name)
        if model is None:
            raise RuntimeError(f"No ONNX export configured for builtin model {engine_model_name!r}")
        onnx_path = self._select_model_file(model, device)
        if not onnx_path.exists():
            raise RuntimeError(f"ONNX model file not found: {onnx_path}")
        self.devices.validate(device)
        session = self._get_session(str(onnx_path), device)
        state = {"force_tiled": False}

        def upscale_frame(frame_nhwc: np.ndarray) -> np.ndarray:
            upscaled, state["force_tiled"] = self._upscale_one(
                session, frame_nhwc, device, state["force_tiled"]
            )
            return upscaled

        return upscale_frame
```

(`Callable` ya está importado en el módulo, línea 9.)

- [ ] **Step 4: Correr y verificar verde**

Run: `.venv\Scripts\python -m pytest tests/test_onnx_video_upscaler.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add app/services/engines/onnx_video_upscaler.py tests/test_onnx_video_upscaler.py
git commit -m "feat: build_frame_upscaler expone el closure por-frame del motor ONNX de video"
```

---

### Task 7: `GmfssEngine.build_stream_stage` — GMFSS como etapa de streaming 1→N

GMFSS entra al pipeline como `FrameStage` con ventana de 2 frames: recibe frames fuente en orden y emite `source[0], interp(pair0)…, source[1], …` — exactamente el orden de `_compute_loop`/`run()`. Reusa `_build_interpolation_plan` (mismo Bresenham exacto), el driver real (`GmfssDriver` + splat OpenCL) y las conversiones existentes. Los frames fuente pasan VERBATIM (pixel-idénticos, sin round-trip de resize), igual que `run()` los copia byte a byte.

**Files:**
- Modify: `app/services/engines/gmfss_engine.py` (método nuevo + clase `GmfssStreamStage` + helpers re-agregados; refactor chico de `_prepare_pipeline`)
- Test: `tests/test_gmfss_engine.py`

**Interfaces:**
- Consumes: `_build_interpolation_plan`, `_get_sessions` (con `GpuSessionCoordinator.acquire` adentro), `GmfssAssets.load`, `_graph_runner`, `softsplat_cl.splat_softmax`, `_rgb_to_padded_chw`, `_chw_float_to_hwc_uint8` (todos existentes).
- Produces: `GmfssEngine.build_stream_stage(source_frame_count: int, target_frame_count: int, device: str) -> GmfssStreamStage`; `GmfssStreamStage` cumple el protocolo `FrameStage` de Task 3 (`process`/`flush`). Task 11 lo inserta en el pipeline completo. También re-agrega `_chw_float_to_nhwc_uint8(frame_chw, original_hw) -> np.ndarray` y agrega `_nhwc_uint8_to_padded_chw(frame_nhwc, padded_hw) -> np.ndarray` (module-level).

- [ ] **Step 1: Escribir los tests que fallan**

En `tests/test_gmfss_engine.py` (usa `make_settings`, `fake_sessions`, `write_fake_source_frames`, `SOURCE_H/W` ya definidos ahí), al final del archivo:

```python
# ---------------------------------------------------------------------------
# build_stream_stage (stream pipeline): GMFSS como FrameStage 1→N con ventana
# de 2 frames — mismo orden de emisión y misma aritmética de plan que run().
# ---------------------------------------------------------------------------


def make_stream_source_frames(count: int) -> list[np.ndarray]:
    frames = []
    for index in range(count):
        value = (index * 17) % 256
        frames.append(np.full((1, SOURCE_H, SOURCE_W, 3), value, dtype=np.uint8))
    return frames


def collect_stage_outputs(stage, frames: list[np.ndarray]) -> list[np.ndarray]:
    outputs: list[np.ndarray] = []
    for source_frame in frames:
        outputs.extend(stage.process(source_frame))
    outputs.extend(stage.flush())
    return outputs


def make_stream_engine(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> GmfssEngine:
    engine = GmfssEngine(make_settings(tmp_path), GpuSessionCoordinator())
    monkeypatch.setattr(engine, "_create_sessions", fake_sessions)
    return engine


def test_stream_stage_emits_exact_count_order_and_dtype_for_2x(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine = make_stream_engine(tmp_path, monkeypatch)
    stage = engine.build_stream_stage(source_frame_count=3, target_frame_count=6, device="cpu")
    source = make_stream_source_frames(3)

    outputs = collect_stage_outputs(stage, source)

    # plan(3→6) = [1, 2] extras: source0, i(0.5), source1, i(1/3), i(2/3), source2
    assert len(outputs) == 6
    assert np.array_equal(outputs[0], source[0])  # fuente verbatim (t=0)
    assert np.array_equal(outputs[2], source[1])
    assert np.array_equal(outputs[5], source[2])  # fuente verbatim (t=1 del último par)
    for out in outputs:
        assert out.dtype == np.uint8
        assert out.shape == (1, SOURCE_H, SOURCE_W, 3)  # NHWC RGB a resolución fuente


def test_stream_stage_passes_through_when_no_extra_frames(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine = make_stream_engine(tmp_path, monkeypatch)
    stage = engine.build_stream_stage(source_frame_count=2, target_frame_count=2, device="cpu")
    source = make_stream_source_frames(2)

    outputs = collect_stage_outputs(stage, source)

    assert len(outputs) == 2
    assert all(np.array_equal(a, b) for a, b in zip(outputs, source))


async def test_stream_stage_matches_run_output_byte_for_byte(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Gate de calidad: la etapa de streaming produce EXACTAMENTE los mismos
    # píxeles que run() (misma sesión fake, mismo plan, mismas conversiones) —
    # solo desaparece el hop por disco. PNG es lossless, así que la comparación
    # decode-a-decode es byte-exacta.
    import cv2

    engine = make_stream_engine(tmp_path, monkeypatch)
    frames_in = tmp_path / "frames-in"
    write_fake_source_frames(frames_in, 3)
    frames_out = tmp_path / "frames-out"
    await engine.run(frames_in, frames_out, 3, 2, device="cpu")

    stage = engine.build_stream_stage(source_frame_count=3, target_frame_count=6, device="cpu")
    source = []
    for path in sorted(frames_in.glob("*.png")):
        rgb = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
        source.append(np.ascontiguousarray(rgb)[np.newaxis, ...])
    outputs = collect_stage_outputs(stage, source)

    expected_paths = sorted(frames_out.glob("*.png"))
    assert len(outputs) == len(expected_paths) == 6
    for index, path in enumerate(expected_paths):
        expected = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
        assert np.array_equal(outputs[index][0], expected), path.name


def test_stream_stage_flush_raises_on_missing_source_frames(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine = make_stream_engine(tmp_path, monkeypatch)
    stage = engine.build_stream_stage(source_frame_count=3, target_frame_count=6, device="cpu")
    stage.process(make_stream_source_frames(1)[0])

    with pytest.raises(RuntimeError, match="esperaba"):
        stage.flush()


def test_stream_stage_raises_on_extra_source_frames(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine = make_stream_engine(tmp_path, monkeypatch)
    stage = engine.build_stream_stage(source_frame_count=2, target_frame_count=4, device="cpu")
    frames = make_stream_source_frames(3)
    stage.process(frames[0])
    stage.process(frames[1])

    with pytest.raises(RuntimeError, match="más frames"):
        stage.process(frames[2])


def test_build_stream_stage_when_unavailable_raises_actionable_error(tmp_path: Path) -> None:
    engine = GmfssEngine(make_settings(tmp_path, enabled=False), GpuSessionCoordinator())
    with pytest.raises(RuntimeError, match="ENABLE_GMFSS"):
        engine.build_stream_stage(source_frame_count=2, target_frame_count=4, device="cpu")
```

- [ ] **Step 2: Correr y verificar que fallan**

Run: `.venv\Scripts\python -m pytest tests/test_gmfss_engine.py -k stream_stage -v`
Expected: FAIL con `AttributeError: 'GmfssEngine' object has no attribute 'build_stream_stage'`

- [ ] **Step 3: Implementar**

En `gmfss_engine.py`:

1. Refactor chico: extraer la construcción del driver de `_prepare_pipeline` a un método compartido (mismas 4 líneas, un solo dueño):

```python
    def _build_driver(self, device: str) -> tuple[GmfssDriver, tuple[int, int]]:
        sessions = self._get_sessions(device)
        assets = GmfssAssets.load(self.settings.gmfss_model_dir_path)
        driver = GmfssDriver(assets, _graph_runner(sessions), splat_fn=softsplat_cl.splat_softmax)
        return driver, assets.padded_hw
```

y en `_prepare_pipeline` reemplazar esas líneas por `driver, padded_hw = self._build_driver(device)` (el return pasa a usar `padded_hw` en lugar de `assets.padded_hw`).

2. Método público nuevo (después de `run()`):

```python
    def build_stream_stage(
        self, source_frame_count: int, target_frame_count: int, device: str
    ) -> "GmfssStreamStage":
        """GMFSS como FrameStage del stream pipeline: mismo plan exacto
        (_build_interpolation_plan) y mismo driver que run(), sin PNGs. El
        caller conoce source_frame_count por el probe (el gate del pipeline
        exige framesTotal conocido para elegir este camino)."""
        if not self.available():
            raise RuntimeError(
                "GMFSS interpolation engine is not available. Enable ENABLE_GMFSS and install the "
                "models (scripts/download-gmfss-onnx.ps1)."
            )
        plan = _build_interpolation_plan(source_frame_count, target_frame_count)
        driver, padded_hw = self._build_driver(device)
        return GmfssStreamStage(driver, padded_hw, plan)
```

3. Clase `GmfssStreamStage` (module-level, después de la clase `GmfssEngine`):

```python
class GmfssStreamStage:
    """FrameStage 1→N con ventana de 2: emite source[0], interp(pair0)...,
    source[1], ... — el MISMO orden que _compute_loop/run(). Los frames fuente
    pasan verbatim (pixel-idénticos, sin round-trip por la resolución padded);
    solo los interpolados atraviesan el driver. Corre en UN thread del
    pipeline: no necesita locks propios."""

    def __init__(
        self, driver: GmfssDriver, padded_hw: tuple[int, int], plan: list[list[float]]
    ) -> None:
        self._driver = driver
        self._padded_hw = padded_hw
        self._plan = plan
        self._pair_index = 0
        self._prev_chw: np.ndarray | None = None
        self._prev_hw: tuple[int, int] | None = None

    def process(self, frame: np.ndarray) -> list[np.ndarray]:
        chw = _nhwc_uint8_to_padded_chw(frame, self._padded_hw)
        original_hw = (frame.shape[1], frame.shape[2])
        if self._prev_chw is None:
            self._prev_chw, self._prev_hw = chw, original_hw
            return [frame]  # source[0] verbatim (t=0)
        if self._pair_index >= len(self._plan):
            raise RuntimeError(
                f"GMFSS recibió más frames fuente que los {len(self._plan) + 1} planificados"
            )
        timesteps = self._plan[self._pair_index]
        outputs: list[np.ndarray] = []
        if timesteps:  # un par con 0 extras se saltea reuse()+forward por completo
            for output_chw in self._driver.interpolate_pair(self._prev_chw, chw, timesteps):
                outputs.append(_chw_float_to_nhwc_uint8(output_chw, self._prev_hw))
        outputs.append(frame)  # source[i+1] verbatim (t=1)
        self._pair_index += 1
        self._prev_chw, self._prev_hw = chw, original_hw
        return outputs

    def flush(self) -> list[np.ndarray]:
        if self._pair_index != len(self._plan):
            raise RuntimeError(
                f"GMFSS esperaba {len(self._plan) + 1} frames fuente y recibió {self._pair_index + 1}"
            )
        return []
```

4. Helpers module-level (junto a `_chw_float_to_hwc_uint8`): re-agregar el one-liner que Task 1 eliminó y sumar la conversión de entrada en memoria:

```python
def _nhwc_uint8_to_padded_chw(frame_nhwc: np.ndarray, padded_hw: tuple[int, int]) -> np.ndarray:
    original_hw = (frame_nhwc.shape[1], frame_nhwc.shape[2])
    return _rgb_to_padded_chw(frame_nhwc[0], original_hw, padded_hw)


def _chw_float_to_nhwc_uint8(frame_chw: np.ndarray, original_hw: tuple[int, int]) -> np.ndarray:
    return _chw_float_to_hwc_uint8(frame_chw, original_hw)[np.newaxis, ...]
```

- [ ] **Step 4: Correr y verificar verde**

Run: `.venv\Scripts\python -m pytest tests/test_gmfss_engine.py -q`
Expected: PASS (incluye el test de paridad byte-exacta contra `run()`).

- [ ] **Step 5: Commit**

```bash
git add app/services/engines/gmfss_engine.py tests/test_gmfss_engine.py
git commit -m "feat: build_stream_stage expone GMFSS como etapa de streaming 1→N"
```

---

### Task 8: Flag `ENABLE_STREAM_PIPELINE` + gate de ruteo `_resolve_stream_pipeline_mode`

El ruteo del spec en un solo lugar: flag ON + backend ONNX builtin + (sin interp → completo | GMFSS con conteo conocido → completo | RIFE → híbrido); NCNN, modelos HF-ONNX, salidas chicas, jobs que necesitan el input original (pistas extra/subtítulos — solo el encode PNG sabe mapearlos) o GMFSS sin `framesTotal` honesto → `None` (camino clásico). Este task también recrea `tests/test_video_upscaler.py` (borrado en Task 1) con los fakes que reusan Tasks 9-11.

**Files:**
- Modify: `app/config.py` (campo nuevo después de `raw_pipe_min_output_pixels`, línea ~417)
- Modify: `.env.example` (línea nueva junto a `ENABLE_RAW_PIPE`)
- Modify: `app/services/video_upscaler.py` (constantes + método gate, sección raw-pipe)
- Create: `tests/test_video_upscaler.py` (recreado)

**Interfaces:**
- Consumes: `Settings.raw_pipe_min_output_pixels`, `_resolve_builtin_backend`, `_is_onnx_model`, `_needs_source_input`, `_interpolation_requested`, `GMFSS_ENGINE` (todos existentes).
- Produces: `Settings.enable_stream_pipeline: bool` (default `True`); constantes `STREAM_MODE_FULL = "full"` y `STREAM_MODE_HYBRID = "hybrid"` en `video_upscaler.py`; `VideoUpscaler._resolve_stream_pipeline_mode(job: VideoUpscaleJob, fps_multiplier: int) -> str | None`. Tasks 9/10 consumen el modo; los fakes del test file los reusan Tasks 9-11.

- [ ] **Step 1: Recrear `tests/test_video_upscaler.py` con los fakes + tests del gate que fallan**

```python
from __future__ import annotations

from pathlib import Path

import pytest

from app.config import Settings
from app.models import VideoUpscaleJob
from app.services.model_registry import ModelEntry, ModelKind, ModelRegistry, ModelStatus
from app.services.video_upscaler import (
    STREAM_MODE_FULL,
    STREAM_MODE_HYBRID,
    VideoUpscaler,
)

# ---------------------------------------------------------------------------
# Stream pipeline (spec 2026-07-25-stream-frame-pipeline-design.md) — ruteo de
# modos, tramo híbrido, pipeline completo y fallback clásico. Fakes calcados de
# tests/test_video_backend_dispatch.py; ningún binario real corre acá.
# ---------------------------------------------------------------------------


class FakeNcnnEngine:
    def available(self) -> bool:
        return True


class FakeMediaTools:
    def available(self) -> bool:
        return True

    async def ffprobe_json(self, source_path: Path) -> dict:
        return {
            "streams": [{"codec_type": "video", "width": 1280, "height": 720, "avg_frame_rate": "24/1"}],
            "format": {"duration": "2.0"},
        }


class FakeDevicesService:
    def __init__(self, valid_ids: tuple[str, ...] = ("cpu", "dml:0")) -> None:
        self._valid_ids = valid_ids

    def list_devices(self) -> list[dict]:
        # "name" presente: _device_name lo indexa al resolver el encoder "auto".
        return [{"id": device_id, "name": "Fake GPU"} for device_id in self._valid_ids]

    def validate(self, device_id: str) -> dict:
        if device_id not in self._valid_ids:
            raise ValueError(f"Unknown device id: {device_id!r}")
        return {"id": device_id}


class FakeOnnxVideoEngine:
    """Stand-in de OnnxVideoUpscaler para el gate: solo responde los probes de
    capacidad que _resolve_builtin_backend consulta."""

    def __init__(self, *, available: bool = True, gpu_ep: bool = True, builtin_available: bool = True) -> None:
        self._available = available
        self._gpu_ep = gpu_ep
        self._builtin_available = builtin_available

    def available(self) -> bool:
        return self._available

    def has_gpu_execution_provider(self) -> bool:
        return self._gpu_ep

    def builtin_onnx_available(self, engine_model_name: str) -> bool:
        return self._builtin_available


def make_onnx_entry(**overrides: object) -> ModelEntry:
    defaults: dict[str, object] = {
        "id": "fake-onnx-2x",
        "name": "Fake ONNX 2x",
        "kind": ModelKind.onnx,
        "source": "https://huggingface.co/example/fake-onnx-2x",
        "size_bytes": 1_000,
        "scale": 2,
        "arch": "fake",
        "file_path": "onnx/fake-onnx-2x.onnx",
        "status": ModelStatus.installed,
    }
    defaults.update(overrides)
    return ModelEntry(**defaults)


def make_stream_settings(tmp_path: Path, **overrides: object) -> Settings:
    kwargs: dict[str, object] = {
        "RUNTIME_DIR": str(tmp_path / "runtime"),
        "BUILTIN_ONNX_DIR": str(tmp_path / "builtin-onnx"),
    }
    kwargs.update(overrides)
    return Settings(_env_file=None, **kwargs)


def make_stream_upscaler(
    tmp_path: Path,
    *,
    gmfss_engine: object | None = object(),
    onnx_video_engine: object | None = None,
    registry: ModelRegistry | None = None,
    **settings_overrides: object,
) -> VideoUpscaler:
    settings = make_stream_settings(tmp_path, **settings_overrides)
    engine = FakeOnnxVideoEngine() if onnx_video_engine is None else onnx_video_engine
    return VideoUpscaler(
        settings,
        FakeNcnnEngine(),  # type: ignore[arg-type]
        FakeMediaTools(),  # type: ignore[arg-type]
        gmfss_engine=gmfss_engine,  # type: ignore[arg-type]
        onnx_video_engine=engine,  # type: ignore[arg-type]
        model_registry=registry if registry is not None else ModelRegistry(settings),
        devices=FakeDevicesService(),  # type: ignore[arg-type]
    )


def make_stream_job(tmp_path: Path, **overrides: object) -> VideoUpscaleJob:
    fields: dict[str, object] = dict(
        source_path=tmp_path / "clip.mp4",
        original_filename="clip.mp4",
        model_name="realesr-animevideov3-x4",
        scale=4,
        output_container="mp4",
        video_codec="libx264",
        video_preset="medium",
        crf=18,
        keep_audio=False,
        model_id=None,
        device="cpu",
        backend="onnx",
    )
    fields.update(overrides)
    job = VideoUpscaleJob(**fields)
    # Metadata que _run_pipeline estampa en el probe, precondición del gate:
    # 1280x720 x4 => 14.7Mpx de salida, sobre el umbral raw_pipe_min_output_pixels.
    job.metadata.update({"sourceWidth": 1280, "sourceHeight": 720, "framesTotal": 48})
    return job


# ---------------------------------------------------------------------------
# Gate: _resolve_stream_pipeline_mode
# ---------------------------------------------------------------------------


def test_enable_stream_pipeline_defaults_to_true() -> None:
    assert Settings(_env_file=None).enable_stream_pipeline is True


async def test_mode_full_when_no_interpolation(tmp_path: Path) -> None:
    upscaler = make_stream_upscaler(tmp_path)
    job = make_stream_job(tmp_path)
    assert await upscaler._resolve_stream_pipeline_mode(job, 1) == STREAM_MODE_FULL


async def test_mode_none_when_flag_off(tmp_path: Path) -> None:
    upscaler = make_stream_upscaler(tmp_path, ENABLE_STREAM_PIPELINE=False)
    job = make_stream_job(tmp_path)
    assert await upscaler._resolve_stream_pipeline_mode(job, 1) is None


async def test_mode_none_without_onnx_video_engine(tmp_path: Path) -> None:
    upscaler = make_stream_upscaler(tmp_path, onnx_video_engine=False)
    upscaler.onnx_video_engine = None
    job = make_stream_job(tmp_path)
    assert await upscaler._resolve_stream_pipeline_mode(job, 1) is None


async def test_mode_none_when_backend_resolves_ncnn(tmp_path: Path) -> None:
    upscaler = make_stream_upscaler(tmp_path)
    job = make_stream_job(tmp_path, backend="ncnn")
    assert await upscaler._resolve_stream_pipeline_mode(job, 1) is None


async def test_mode_none_for_hf_onnx_model(tmp_path: Path) -> None:
    settings_dir = tmp_path
    registry = ModelRegistry(make_stream_settings(settings_dir))
    registry.register(make_onnx_entry())
    upscaler = make_stream_upscaler(tmp_path, registry=registry)
    job = make_stream_job(tmp_path, model_name="fake-onnx-2x", model_id="fake-onnx-2x", scale=2)
    assert await upscaler._resolve_stream_pipeline_mode(job, 1) is None


async def test_mode_none_below_min_output_pixels(tmp_path: Path) -> None:
    upscaler = make_stream_upscaler(tmp_path)
    job = make_stream_job(tmp_path)
    job.metadata.update({"sourceWidth": 64, "sourceHeight": 64})  # 64x64x4x = chico
    assert await upscaler._resolve_stream_pipeline_mode(job, 1) is None


async def test_mode_none_when_job_needs_source_input(tmp_path: Path) -> None:
    # Pistas de audio extra / subtítulos: solo _build_encode_command (camino PNG)
    # sabe mapearlos desde el source original — misma restricción que el raw-pipe.
    upscaler = make_stream_upscaler(tmp_path)
    job = make_stream_job(tmp_path, keep_subtitles=True)
    assert await upscaler._resolve_stream_pipeline_mode(job, 1) is None


async def test_mode_hybrid_for_rife_interpolation(tmp_path: Path) -> None:
    upscaler = make_stream_upscaler(tmp_path)
    job = make_stream_job(tmp_path, fps_multiplier=2)
    assert await upscaler._resolve_stream_pipeline_mode(job, 2) == STREAM_MODE_HYBRID


async def test_mode_full_for_gmfss_with_known_frames_total(tmp_path: Path) -> None:
    upscaler = make_stream_upscaler(tmp_path)
    job = make_stream_job(tmp_path, interp_engine="gmfss", fps_multiplier=2)
    assert await upscaler._resolve_stream_pipeline_mode(job, 2) == STREAM_MODE_FULL


async def test_mode_none_for_gmfss_without_frames_total(tmp_path: Path) -> None:
    # VFR/indeterminable: sin conteo honesto no hay plan GMFSS posible.
    upscaler = make_stream_upscaler(tmp_path)
    job = make_stream_job(tmp_path, interp_engine="gmfss", fps_multiplier=2)
    job.metadata["framesTotal"] = None
    assert await upscaler._resolve_stream_pipeline_mode(job, 2) is None


async def test_mode_none_for_gmfss_without_engine(tmp_path: Path) -> None:
    upscaler = make_stream_upscaler(tmp_path, gmfss_engine=None)
    job = make_stream_job(tmp_path, interp_engine="gmfss", fps_multiplier=2)
    assert await upscaler._resolve_stream_pipeline_mode(job, 2) is None
```

- [ ] **Step 2: Correr y verificar que fallan**

Run: `.venv\Scripts\python -m pytest tests/test_video_upscaler.py -v`
Expected: FAIL con `ImportError: cannot import name 'STREAM_MODE_FULL'`

- [ ] **Step 3: Implementar config + gate**

En `app/config.py`, después de `raw_pipe_min_output_pixels` (línea ~417):

```python
    # Pipeline de frames en streaming (spec 2026-07-25): conecta decode→
    # (interpolación)→upscale→encode por colas acotadas en memoria, sin PNGs
    # intermedios, una etapa por thread. Fallback automático al camino clásico
    # ante cualquier excepción (patrón raw-pipe). False = camino clásico siempre.
    enable_stream_pipeline: bool = Field(default=True, alias="ENABLE_STREAM_PIPELINE")
```

En `.env.example`, después de la línea de `RAW_PIPE_MIN_OUTPUT_PIXELS`:

```text
ENABLE_STREAM_PIPELINE=True          # Pipeline de frames en streaming (decode->interp->upscale->encode por colas en memoria, sin PNGs). Cae al camino clasico ante cualquier fallo. False = siempre camino clasico
```

En `app/services/video_upscaler.py`, junto a `FRAME_POLL_INTERVAL_SECONDS` (línea ~57):

```python
# Modos del stream pipeline (spec 2026-07-25-stream-frame-pipeline-design.md):
# "full" = decode→(GMFSS)→upscale→encode sin ningún PNG; "hybrid" = RIFE
# conserva su tramo PNG y solo upscale→encode va en streaming; None = clásico.
STREAM_MODE_FULL = "full"
STREAM_MODE_HYBRID = "hybrid"
```

Y el método en `VideoUpscaler`, en la sección raw-pipe (antes de `_should_stream`):

```python
    async def _resolve_stream_pipeline_mode(
        self, job: VideoUpscaleJob, fps_multiplier: int
    ) -> str | None:
        """Elige el camino del stream pipeline. Requiere metadata del probe ya
        estampada (sourceWidth/sourceHeight/framesTotal). None = camino clásico.
        """
        if not self.settings.enable_stream_pipeline or self.onnx_video_engine is None:
            return None
        # Modelos HF-ONNX van por OnnxUpscaler (grafos fp32 arbitrarios), no por
        # el motor builtin de streaming — misma exclusión que el raw-pipe.
        if self._is_onnx_model(job.model_id):
            return None
        # Solo _build_encode_command (camino PNG) sabe mapear pistas de audio
        # extra / subtítulos desde el source original.
        if self._needs_source_input(job):
            return None
        out_pixels = (
            int(job.metadata.get("sourceWidth") or 0)
            * int(job.metadata.get("sourceHeight") or 0)
            * job.scale
            * job.scale
        )
        # Bajo el umbral el encode PNG es barato y el overhead no compensa —
        # mismo criterio (y mismo setting) que el raw-pipe.
        if out_pixels < self.settings.raw_pipe_min_output_pixels:
            return None
        # Off the loop: el primer resolve puede hacer un import frío de
        # onnxruntime + get_available_providers (mismo motivo que _should_stream).
        if await asyncio.to_thread(self._resolve_builtin_backend, job) != UpscaleBackend.onnx:
            return None
        if not self._interpolation_requested(fps_multiplier, job.target_fps):
            return STREAM_MODE_FULL
        if job.interp_engine == GMFSS_ENGINE:
            # El plan GMFSS necesita el conteo fuente exacto: sin framesTotal
            # honesto (VFR) no hay plan — clásico.
            if self.gmfss_engine is None or job.metadata.get("framesTotal") is None:
                return None
            return STREAM_MODE_FULL
        # RIFE: el binario exige el directorio PNG entero — híbrido (su tramo
        # queda en PNG; upscale→encode va en streaming).
        return STREAM_MODE_HYBRID
```

(`GMFSS_ENGINE` ya está importado en el módulo desde `app.config`, línea 16.)

- [ ] **Step 4: Correr y verificar verde**

Run: `.venv\Scripts\python -m pytest tests/test_video_upscaler.py tests/test_video_backend_dispatch.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add app/config.py .env.example app/services/video_upscaler.py tests/test_video_upscaler.py
git commit -m "feat: flag ENABLE_STREAM_PIPELINE y gate de modos del stream pipeline"
```

---

### Task 9: Tramo híbrido — `_try_stream_pipeline_from_dir` + núcleo `_run_stream_pipeline`

El corazón del wiring: el núcleo async `_run_stream_pipeline` (cancel_event + shield + watchdog vía `_track_streaming_progress`, patrón exacto del raw-pipe) y su runner blocking que arma etapas + colas presupuestadas + `RawPipeEncoder`. El primer consumidor es el modo híbrido (RIFE): sus PNGs interpolados se leen en streaming hacia upscale→encode. También agrega `iter_png_frames` (source de directorio) a `frame_pipeline.py`.

**Files:**
- Modify: `app/services/frame_pipeline.py` (`iter_png_frames`)
- Modify: `app/services/video_upscaler.py` (núcleo + tramo híbrido + wiring en `_interpolate_and_upscale`/`_run_pipeline`)
- Test: `tests/test_frame_pipeline.py`, `tests/test_video_upscaler.py`

**Interfaces:**
- Consumes: `FramePipeline`, `MapStage`, `derive_stream_queue_maxsizes` (Task 3), `FfmpegFrameSource` no todavía (Task 10), `RawPipeEncoder` (Task 5), `build_frame_upscaler` (Task 6), `STREAM_MODE_HYBRID` (Task 8), `_track_streaming_progress`/`_build_rawpipe_command`/`_summarize_process_error`/`VideoStallError` (existentes).
- Produces:
  - `iter_png_frames(frames_dir: Path, cancel_event: threading.Event) -> Iterator[np.ndarray]` en `frame_pipeline.py`
  - `VideoUpscaler._run_stream_pipeline(job, *, source_factory: Callable[[threading.Event], Iterator[np.ndarray]], gmfss_stage_factory: Callable[[str], FrameStage] | None, width: int, height: int, expected_output_count: int | None, output_path: Path, encode_fps: str, audio_mux_path: Path | None, audio_codec_args: list[str], encoder_name: str) -> None` (Tasks 10/11 lo reusan tal cual)
  - `VideoUpscaler._try_stream_pipeline_from_dir(job, frames_dir: Path, output_path: Path, encode_fps: str, audio_mux_path: Path | None, audio_codec_args: list[str]) -> bool`
  - `_interpolate_and_upscale(..., stream_mode: str | None)` (parámetro nuevo al final de la firma)

- [ ] **Step 1: Agregar `iter_png_frames` con sus tests**

En `tests/test_frame_pipeline.py`:

```python
from pathlib import Path

from app.services.frame_pipeline import iter_png_frames


def write_png(path: Path, value: int) -> None:
    import cv2

    frame_bgr = np.full((2, 3, 3), value, dtype=np.uint8)
    assert cv2.imwrite(str(path), frame_bgr)


def test_iter_png_frames_yields_in_name_order(tmp_path: Path) -> None:
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    for index in range(1, 4):
        write_png(frames_dir / f"{index:08d}.png", index * 10)

    frames = list(iter_png_frames(frames_dir, threading.Event()))

    assert [f.shape for f in frames] == [(1, 2, 3, 3)] * 3
    assert [f.dtype for f in frames] == [np.uint8] * 3
    # Valor uniforme por canal: BGR->RGB no cambia el primer píxel.
    assert [int(f[0, 0, 0, 0]) for f in frames] == [10, 20, 30]


def test_iter_png_frames_stops_on_preset_cancel(tmp_path: Path) -> None:
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    write_png(frames_dir / "00000001.png", 10)
    cancel = threading.Event()
    cancel.set()

    assert list(iter_png_frames(frames_dir, cancel)) == []
```

Run: `.venv\Scripts\python -m pytest tests/test_frame_pipeline.py -k iter_png -v` → FAIL (`ImportError`).

Implementar en `frame_pipeline.py` (agregar `from pathlib import Path` y sumar `_load_frame` al import existente de `onnx_video_upscaler`):

```python
def iter_png_frames(frames_dir: Path, cancel_event: threading.Event) -> Iterator[np.ndarray]:
    """Source del tramo híbrido: los PNGs %08d.png del directorio, en orden,
    como frames NHWC uint8. UN solo hilo los decodea (~30ms/frame vs ~116ms de
    infer: no es cuello — mismo racional que el loader único del raw-pipe)."""
    for path in sorted(frames_dir.glob("*.png")):
        if cancel_event.is_set():
            return
        yield _load_frame(path)
```

Run: `.venv\Scripts\python -m pytest tests/test_frame_pipeline.py -q` → PASS.

- [ ] **Step 2: Escribir los tests de wiring que fallan (`tests/test_video_upscaler.py`)**

Agregar al final del archivo (los helpers `make_stream_upscaler`/`make_stream_job` son de Task 8; `Double2xUint8Session` y `FakeEncodeProc` se copian acá porque los archivos de test no se importan entre sí):

```python
import cv2
import numpy as np

from app.services.devices_service import DevicesService
from app.services.engines.ffmpeg_frame_sink import RawPipeEncoder
from app.services.engines.onnx_video_upscaler import OnnxVideoUpscaler
from app.services.gpu_session_coordinator import GpuSessionCoordinator

SOURCE_H, SOURCE_W = 8, 12


def write_source_frames(directory: Path, count: int) -> None:
    # Primer píxel R = offset del frame (i*17): el orden queda verificable en
    # los bytes crudos que recibe el encoder fake.
    directory.mkdir(parents=True, exist_ok=True)
    row = np.arange(SOURCE_H, dtype=np.int32).reshape(SOURCE_H, 1)
    col = np.arange(SOURCE_W, dtype=np.int32).reshape(1, SOURCE_W)
    for index in range(count):
        offset = (index * 17) % 256
        frame = np.empty((SOURCE_H, SOURCE_W, 3), dtype=np.uint8)
        frame[:, :, 2] = (row * 0 + offset) % 256  # canal R en BGR de cv2
        frame[:, :, 1] = (col * 13 + offset * 2) % 256
        frame[:, :, 0] = (row + col * 5 + offset * 3) % 256
        assert cv2.imwrite(str(directory / f"{index + 1:08d}.png"), frame)


class _IoInfo:
    def __init__(self, name: str) -> None:
        self.name = name


class Double2xUint8Session:
    """Fake de sesión ONNX uint8: dobla H/W — copiado de tests/test_onnx_video_upscaler.py."""

    def __init__(self) -> None:
        self._input = _IoInfo("image")
        self._output = _IoInfo("upscaled")

    def get_inputs(self) -> list[_IoInfo]:
        return [self._input]

    def get_outputs(self) -> list[_IoInfo]:
        return [self._output]

    def run(self, output_names, input_feed):
        array = input_feed[self._input.name]
        assert array.dtype == np.uint8
        return [np.repeat(np.repeat(array, 2, axis=1), 2, axis=2)]


class FakeStdin(io.BytesIO):
    def close(self) -> None:  # type: ignore[override]
        pass  # el buffer sigue legible para inspeccionar los bytes escritos


class FakeEncodeProc:
    def __init__(self, returncode: int = 0) -> None:
        self.stdin = FakeStdin()
        self.stderr = io.BytesIO(b"")
        self.returncode: int | None = None
        self._final_returncode = returncode
        self.killed = False

    def wait(self, timeout=None) -> int:
        self.returncode = self._final_returncode
        return self.returncode

    def poll(self):
        return self.returncode

    def kill(self) -> None:
        self.killed = True
        self._final_returncode = -9


def make_streaming_upscaler_with_real_engines(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> VideoUpscaler:
    """VideoUpscaler con un OnnxVideoUpscaler REAL (sesión fake Double2x) — el
    FramePipeline y las etapas corren de verdad; solo sesión y procesos son fake."""
    settings = make_stream_settings(tmp_path)
    settings.builtin_onnx_path.mkdir(parents=True, exist_ok=True)
    (settings.builtin_onnx_path / "realesr-animevideov3-x4-uint8.onnx").write_bytes(b"fake")
    onnx_video = OnnxVideoUpscaler(
        settings, ModelRegistry(settings), DevicesService(settings), GpuSessionCoordinator()
    )
    monkeypatch.setattr(onnx_video, "_create_session", lambda model_path, device: Double2xUint8Session())
    return VideoUpscaler(
        settings,
        FakeNcnnEngine(),  # type: ignore[arg-type]
        FakeMediaTools(),  # type: ignore[arg-type]
        gmfss_engine=None,
        onnx_video_engine=onnx_video,
        model_registry=ModelRegistry(settings),
        devices=DevicesService(settings),
    )


# ---------------------------------------------------------------------------
# Tramo híbrido: ruteo + fallback + integración con FramePipeline real
# ---------------------------------------------------------------------------


async def test_hybrid_mode_streams_from_interp_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    upscaler = make_stream_upscaler(tmp_path)
    job = make_stream_job(tmp_path, fps_multiplier=2)
    frames_in = tmp_path / "frames-in"
    frames_in.mkdir(parents=True)
    interp_dir = tmp_path / "frames-interp"
    calls: dict = {}

    async def fake_interp(job_arg, frames_dir, fps, mult, target_fps=None):
        interp_dir.mkdir(parents=True, exist_ok=True)
        return interp_dir, "48/1"

    async def fake_from_dir(job_arg, frames_dir, output_path, encode_fps, mux, codec_args):
        calls["dir"] = frames_dir
        calls["fps"] = encode_fps
        output_path.write_bytes(b"fake-video")
        return True

    monkeypatch.setattr(upscaler, "_maybe_interpolate", fake_interp)
    monkeypatch.setattr(upscaler, "_try_stream_pipeline_from_dir", fake_from_dir)

    encode_dir, encode_fps = await upscaler._interpolate_and_upscale(
        job, frames_in, tmp_path / "frames-out", "24/1", 2, tmp_path / "out.mp4", None, [], STREAM_MODE_HYBRID
    )

    assert encode_dir is None  # el caller NO encodea: el tramo ya produjo el output
    assert encode_fps == "48/1"
    assert calls["dir"] == interp_dir


async def test_hybrid_fallback_uses_classic_png_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    upscaler = make_stream_upscaler(tmp_path)
    job = make_stream_job(tmp_path, fps_multiplier=2)
    frames_in = tmp_path / "frames-in"
    frames_in.mkdir(parents=True)
    frames_out = tmp_path / "frames-out"
    interp_dir = tmp_path / "frames-interp"

    async def fake_interp(job_arg, frames_dir, fps, mult, target_fps=None):
        interp_dir.mkdir(parents=True, exist_ok=True)
        (interp_dir / "00000001.png").write_bytes(b"png")
        return interp_dir, "48/1"

    async def failing_from_dir(job_arg, frames_dir, output_path, encode_fps, mux, codec_args):
        job_arg.metadata["streamPipelineFallback"] = "boom"
        return False

    upscaled = {"n": 0}

    async def fake_upscale(job_arg, src, dst):
        upscaled["n"] += 1
        dst.mkdir(parents=True, exist_ok=True)
        (dst / "00000001.png").write_bytes(b"png")

    monkeypatch.setattr(upscaler, "_maybe_interpolate", fake_interp)
    monkeypatch.setattr(upscaler, "_try_stream_pipeline_from_dir", failing_from_dir)
    monkeypatch.setattr(upscaler, "_upscale_frames", fake_upscale)

    encode_dir, encode_fps = await upscaler._interpolate_and_upscale(
        job, frames_in, frames_out, "24/1", 2, tmp_path / "out.mp4", None, [], STREAM_MODE_HYBRID
    )

    assert encode_dir == frames_out  # el caller encodea por el camino PNG clásico
    assert upscaled["n"] == 1
    assert job.metadata["streamPipelineFallback"] == "boom"


async def test_stream_pipeline_from_dir_streams_all_frames_in_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Integración real (spec "Testing/Integración", conteo 1→1): FramePipeline +
    # etapa de upscale reales; solo la sesión ONNX y el proceso ffmpeg son fake.
    upscaler = make_streaming_upscaler_with_real_engines(tmp_path, monkeypatch)
    frames_dir = tmp_path / "frames-interp"
    write_source_frames(frames_dir, 5)
    fake_proc = FakeEncodeProc()
    monkeypatch.setattr(RawPipeEncoder, "_spawn", lambda self, command: fake_proc)
    job = make_stream_job(tmp_path, device="cpu")

    ok = await upscaler._try_stream_pipeline_from_dir(
        job, frames_dir, tmp_path / "out.mp4", "24/1", None, []
    )

    assert ok is True
    assert job.metadata["streamPipeline"] is True
    assert job.metadata["framesTotal"] == 5  # denominador honesto del tramo
    data = fake_proc.stdin.getvalue()
    # La sesión fake dobla (no 4x como job.scale): el tamaño real escrito manda.
    frame_bytes = (SOURCE_H * 2) * (SOURCE_W * 2) * 3
    assert len(data) == 5 * frame_bytes
    # Orden 1..5 por el primer byte de cada frame (R de (0,0) = offset i*17).
    assert [data[i * frame_bytes] for i in range(5)] == [(i * 17) % 256 for i in range(5)]


async def test_stream_pipeline_from_dir_falls_back_on_encoder_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    upscaler = make_streaming_upscaler_with_real_engines(tmp_path, monkeypatch)
    frames_dir = tmp_path / "frames-interp"
    write_source_frames(frames_dir, 2)
    fake_proc = FakeEncodeProc(returncode=1)  # ffmpeg "falla" al cerrar
    monkeypatch.setattr(RawPipeEncoder, "_spawn", lambda self, command: fake_proc)
    job = make_stream_job(tmp_path, device="cpu")
    output_path = tmp_path / "out.mp4"
    output_path.write_bytes(b"parcial")

    ok = await upscaler._try_stream_pipeline_from_dir(job, frames_dir, output_path, "24/1", None, [])

    assert ok is False
    assert "streamPipelineFallback" in job.metadata
    assert not output_path.exists()  # el output parcial se borra antes del fallback
```

Agregar los imports nuevos arriba del archivo: `import io`, `import cv2`, `import numpy as np`, `from app.services.devices_service import DevicesService`, `from app.services.engines.ffmpeg_frame_sink import RawPipeEncoder`, `from app.services.engines.onnx_video_upscaler import OnnxVideoUpscaler`, `from app.services.gpu_session_coordinator import GpuSessionCoordinator`.

- [ ] **Step 3: Correr y verificar que fallan**

Run: `.venv\Scripts\python -m pytest tests/test_video_upscaler.py -k "hybrid or from_dir" -v`
Expected: FAIL con `TypeError: VideoUpscaler._interpolate_and_upscale() takes 9 positional arguments but 10 were given` (y `AttributeError` en `_try_stream_pipeline_from_dir`).

- [ ] **Step 4: Implementar el núcleo + tramo híbrido en `video_upscaler.py`**

1. Imports (re-agregar los que Task 1/5 limpiaron y sumar los nuevos):

```python
import numpy as np  # anotaciones del núcleo del stream pipeline
from collections.abc import AsyncIterator, Callable, Iterator
from app.services.frame_pipeline import (
    FramePipeline,
    FrameStage,
    MapStage,
    derive_stream_queue_maxsizes,
    iter_png_frames,
)
from app.services.engines.onnx_video_upscaler import OnnxVideoUpscaler, _load_frame
```

(`threading` y `RawPipeEncoder` ya están importados desde Task 5.)

2. Núcleo compartido (sección raw-pipe, después de `_resolve_stream_pipeline_mode`):

```python
    async def _run_stream_pipeline(
        self,
        job: VideoUpscaleJob,
        *,
        source_factory: "Callable[[threading.Event], Iterator[np.ndarray]]",
        gmfss_stage_factory: "Callable[[str], FrameStage] | None",
        width: int,
        height: int,
        expected_output_count: int | None,
        output_path: Path,
        encode_fps: str,
        audio_mux_path: Path | None,
        audio_codec_args: list[str],
        encoder_name: str,
    ) -> None:
        """Corre el FramePipeline completo en un worker thread con el patrón
        shield+cancel_event+watchdog del raw-pipe: el cancel señala y ESPERA a
        que el worker desenrolle (threads joineados, procesos ffmpeg muertos)
        antes de propagar, para que la limpieza del work-dir no corra contra
        escrituras vivas."""
        out_w, out_h = width * job.scale, height * job.scale
        command = self._build_rawpipe_command(
            out_w, out_h, encode_fps, audio_mux_path, audio_codec_args, output_path, job, encoder_name
        )
        counter = {"n": 0}
        cancel_event = threading.Event()
        worker = asyncio.ensure_future(
            asyncio.to_thread(
                self._run_stream_pipeline_blocking,
                job,
                source_factory,
                gmfss_stage_factory,
                job.device or self.settings.default_device,
                width,
                height,
                expected_output_count,
                command,
                counter,
                cancel_event,
            )
        )
        try:
            async with self._track_streaming_progress(job, counter):
                await asyncio.shield(worker)
        except BaseException:
            cancel_event.set()
            with contextlib.suppress(BaseException):
                await worker
            raise

    def _run_stream_pipeline_blocking(
        self,
        job: VideoUpscaleJob,
        source_factory: "Callable[[threading.Event], Iterator[np.ndarray]]",
        gmfss_stage_factory: "Callable[[str], FrameStage] | None",
        device: str,
        width: int,
        height: int,
        expected_output_count: int | None,
        command: list[str],
        counter: dict[str, int],
        cancel_event: threading.Event,
    ) -> None:
        stages: list[FrameStage] = []
        if gmfss_stage_factory is not None:
            # GMFSS primero, upscale después: el acquire del upscaler evicta la
            # ENTRADA de cache de GMFSS pero el driver ya retiene sus sesiones.
            # Ambos sets quedan residentes en VRAM durante el run — mismo
            # trade-off que documentaba la fusión eliminada; vigilarlo en el
            # smoke real. La serialización sigue en el coordinator/semáforos.
            stages.append(gmfss_stage_factory(device))
        upscale_frame = self.onnx_video_engine.build_frame_upscaler(job.model_name, device)
        stages.append(MapStage(upscale_frame))

        input_bytes = width * height * 3
        output_bytes = input_bytes * job.scale * job.scale
        budget_bytes = max(1, self.settings.onnx_video_max_pipeline_mb) * 1024 * 1024
        maxsizes = derive_stream_queue_maxsizes(input_bytes, output_bytes, len(stages), budget_bytes)

        encoder = RawPipeEncoder(
            command, summarize_error=lambda stderr: self._summarize_process_error(stderr, b"")
        )
        encoder.start()

        def sink(frame_nhwc: np.ndarray) -> None:
            encoder.write_frame(frame_nhwc[0])  # NHWC -> HWC; bloquea con backpressure del pipe
            counter["n"] = encoder.frames_written

        pipeline = FramePipeline(source_factory(cancel_event), stages, sink, maxsizes)
        try:
            delivered = pipeline.run(cancel_event)
        except BaseException:
            encoder.kill()
            raise
        if cancel_event.is_set():
            encoder.kill()
            return
        if delivered == 0:
            encoder.kill()
            raise RuntimeError("el stream pipeline no entregó ningún frame")
        if expected_output_count is not None and delivered != expected_output_count:
            encoder.kill()
            raise RuntimeError(
                f"el stream pipeline entregó {delivered}/{expected_output_count} frames"
            )
        encoder.finish()

    @staticmethod
    def _probe_png_dir(frames_dir: Path) -> tuple[int, int, int]:
        paths = sorted(frames_dir.glob("*.png"))
        if not paths:
            raise RuntimeError("no hay frames para streamear")
        first = _load_frame(paths[0])
        _, height, width, _ = first.shape
        return len(paths), width, height

    async def _try_stream_pipeline_from_dir(
        self,
        job: VideoUpscaleJob,
        frames_dir: Path,
        output_path: Path,
        encode_fps: str,
        audio_mux_path: Path | None,
        audio_codec_args: list[str],
    ) -> bool:
        """Tramo streaming del modo híbrido (RIFE): lee los PNGs interpolados y
        streamea upscale→encode. True = output_path finalizado por el pipeline;
        False = fallback al camino PNG clásico (cancel/stall SÍ propagan)."""
        try:
            frame_count, width, height = await asyncio.to_thread(self._probe_png_dir, frames_dir)
            encoder_name = await asyncio.to_thread(self._resolve_video_encoder, job)
            job.metadata["videoEncoder"] = encoder_name
            # Etapa colapsada (decisión de stepper del plan): denominador
            # honesto = frames reales del tramo (ya interpolados).
            job.metadata["framesTotal"] = frame_count
            advance_video_stage(job, "upscaling_frames")
            await self._run_stream_pipeline(
                job,
                source_factory=lambda cancel_event: iter_png_frames(frames_dir, cancel_event),
                gmfss_stage_factory=None,
                width=width,
                height=height,
                expected_output_count=frame_count,
                output_path=output_path,
                encode_fps=encode_fps,
                audio_mux_path=audio_mux_path,
                audio_codec_args=audio_codec_args,
                encoder_name=encoder_name,
            )
        except (asyncio.CancelledError, VideoStallError):
            raise
        except Exception as exc:  # noqa: BLE001 - CUALQUIER fallo -> camino clásico
            logger.warning("stream pipeline (híbrido) falló (%s); se usa el camino PNG clásico", exc)
            job.metadata["streamPipelineFallback"] = str(exc)
            output_path.unlink(missing_ok=True)
            return False
        job.metadata["streamPipeline"] = True
        job.metadata["outputFps"] = encode_fps
        return True
```

3. Wiring en `_interpolate_and_upscale`: agregar el parámetro `stream_mode: str | None = None` al final de la firma, y reemplazar el bloque del raw-pipe (el `if await self._should_stream(job):`) por:

```python
        # Tramo streaming: con el pipeline nuevo activo (modo híbrido) va por
        # FramePipeline; si no (flag OFF / no elegible / fallback del modo
        # full), el raw-pipe legacy queda intacto. Ambos caen al camino PNG
        # ante cualquier fallo — el if/elif garantiza que un fallo del híbrido
        # NO reintenta con el raw-pipe legacy (iría directo al PNG clásico).
        if stream_mode == STREAM_MODE_HYBRID:
            if await self._try_stream_pipeline_from_dir(
                job, upscale_src, output_path, encode_fps, audio_mux_path, audio_codec_args
            ):
                self._finalize_output(job, output_path)
                await asyncio.to_thread(self._safe_rmtree, upscale_src)
                return None, encode_fps
        elif await self._should_stream(job):
            streamed = await self._try_streaming(
                job, upscale_src, output_path, encode_fps, audio_mux_path, audio_codec_args
            )
            if streamed:
                self._finalize_output(job, output_path)
                await asyncio.to_thread(self._safe_rmtree, upscale_src)
                return None, encode_fps
```

4. Wiring en `_run_pipeline`: justo antes de la llamada a `_interpolate_and_upscale`, resolver y pasar el modo (en este task el modo `full` todavía cae al camino legacy — Task 10 lo intercepta antes de la extracción):

```python
        stream_mode = await self._resolve_stream_pipeline_mode(job, fps_multiplier)
        encode_frames_dir, encode_fps = await self._interpolate_and_upscale(
            job, frames_in, frames_out, fps, fps_multiplier, output_path,
            audio_mux_path, audio_codec_args, stream_mode
        )
```

- [ ] **Step 5: Correr y verificar verde**

Run: `.venv\Scripts\python -m pytest tests/test_video_upscaler.py tests/test_frame_pipeline.py tests/test_pipeline_stage_order.py tests/test_video_encoder_dispatch.py -q`
Expected: PASS (los tests legacy del raw-pipe siguen verdes: sus upscalers no tienen `onnx_video_engine`, así que el gate devuelve `None` y toman la rama `elif` intacta).

- [ ] **Step 6: Commit**

```bash
git add app/services/frame_pipeline.py app/services/video_upscaler.py tests/test_frame_pipeline.py tests/test_video_upscaler.py
git commit -m "feat: tramo híbrido del stream pipeline (PNGs interpolados → upscale → encode en streaming)"
```

---

### Task 10: Pipeline completo sin interpolación — `_try_stream_pipeline_full`

Primer camino 100% sin PNGs: decode (FfmpegFrameSource) → upscale → encode. `_run_pipeline` intercepta el modo `full` ANTES de la extracción PNG (que se saltea por completo); el audio se prepara antes (helper extraído `_prepare_audio_mux`, mismo código que hoy). Ante cualquier fallo, el camino clásico corre DESDE CERO (extracción incluida) — el job nunca falla por culpa del camino nuevo. En este task solo los jobs SIN interpolación entran al modo full; Task 11 suma GMFSS quitando una condición.

**Files:**
- Modify: `app/services/video_upscaler.py` (`_run_pipeline`, `_prepare_audio_mux` extraído, `_try_stream_pipeline_full`)
- Test: `tests/test_video_upscaler.py`

**Interfaces:**
- Consumes: `FfmpegFrameSource` (Task 4), `_run_stream_pipeline` (Task 9), `_resolve_stream_pipeline_mode`/`STREAM_MODE_FULL` (Task 8).
- Produces: `_prepare_audio_mux(job, has_audio: bool, audio_path: Path) -> tuple[Path | None, list[str]]`; `_try_stream_pipeline_full(job, fps_multiplier: int, output_path: Path, fps: str, audio_mux_path: Path | None, audio_codec_args: list[str]) -> bool` — Task 11 le agrega la rama de interpolación sin cambiar la firma.

- [ ] **Step 1: Escribir los tests que fallan**

En `tests/test_video_upscaler.py`, agregar (imports nuevos arriba: `import asyncio`, `import threading`, `import time`, `from app.services.engines.ffmpeg_frame_source import FfmpegFrameSource`):

```python
class FakeDecodeProc:
    """Popen fake de decode: stdout con frames rgb24 crudos — copiado de
    tests/test_ffmpeg_frame_source.py."""

    def __init__(self, stdout_bytes: bytes, returncode: int = 0) -> None:
        self.stdout = io.BytesIO(stdout_bytes)
        self.stderr = io.BytesIO(b"")
        self.returncode: int | None = None
        self._final_returncode = returncode
        self.killed = False

    def wait(self, timeout=None) -> int:
        self.returncode = self._final_returncode
        return self.returncode

    def poll(self):
        return self.returncode

    def kill(self) -> None:
        self.killed = True
        self._final_returncode = -9


def raw_source_frames(count: int) -> bytes:
    # Frame i uniforme en (i*17)%256 a resolución SOURCE_H x SOURCE_W.
    return b"".join(bytes([(i * 17) % 256]) * (SOURCE_H * SOURCE_W * 3) for i in range(count))


class RecordingRunProcessUpscaler(VideoUpscaler):
    """Registra los comandos de _run_process y fakea sus efectos (extract PNG /
    encode escribe el output) — patrón de StageTrackingVideoUpscaler de
    tests/test_pipeline_stage_order.py."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]
        self.commands: list[list[str]] = []

    async def _run_process(self, command: list[str]) -> None:
        self.commands.append(command)
        if "-fps_mode" in command:
            frames_dir = Path(command[-1]).parent
            frames_dir.mkdir(parents=True, exist_ok=True)
            (frames_dir / "00000001.png").write_bytes(b"png")
        elif "-framerate" in command:
            output = Path(command[-1])
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(b"fake-output-video")


def make_recording_upscaler(tmp_path: Path, **settings_overrides: object) -> RecordingRunProcessUpscaler:
    settings = make_stream_settings(tmp_path, **settings_overrides)
    return RecordingRunProcessUpscaler(
        settings,
        FakeNcnnEngine(),  # type: ignore[arg-type]
        FakeMediaTools(),  # type: ignore[arg-type]
        gmfss_engine=object(),  # type: ignore[arg-type]
        onnx_video_engine=FakeOnnxVideoEngine(),  # type: ignore[arg-type]
        model_registry=ModelRegistry(settings),
        devices=FakeDevicesService(),  # type: ignore[arg-type]
    )


# ---------------------------------------------------------------------------
# Pipeline completo (modo full): saltea la extracción PNG; fallback desde cero
# ---------------------------------------------------------------------------


async def test_full_mode_skips_png_extraction(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    upscaler = make_recording_upscaler(tmp_path)
    source_path = upscaler.settings.uploads_path / "clip.mp4"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_bytes(b"fake-video-bytes")
    job = make_stream_job(tmp_path, source_path=source_path)
    full_calls: dict = {}

    async def fake_full(job_arg, fps_multiplier, output_path, fps, mux, codec_args):
        full_calls["fps"] = fps
        output_path.write_bytes(b"fake-video")
        return True

    monkeypatch.setattr(upscaler, "_try_stream_pipeline_full", fake_full)

    output = await upscaler.run(job)

    assert output.exists()
    assert full_calls["fps"] == "24/1"  # el fps del probe de FakeMediaTools
    assert all("-fps_mode" not in command for command in upscaler.commands), "corrió extracción PNG en modo full"
    assert job.metadata["progress"] == 1.0


async def test_full_mode_falls_back_to_classic_from_scratch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    upscaler = make_recording_upscaler(tmp_path)
    source_path = upscaler.settings.uploads_path / "clip.mp4"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_bytes(b"fake-video-bytes")
    job = make_stream_job(tmp_path, source_path=source_path)

    async def failing_full(job_arg, fps_multiplier, output_path, fps, mux, codec_args):
        job_arg.metadata["streamPipelineFallback"] = "boom"
        return False

    async def fake_upscale(job_arg, src, dst):
        dst.mkdir(parents=True, exist_ok=True)
        (dst / "00000001.png").write_bytes(b"png")

    monkeypatch.setattr(upscaler, "_try_stream_pipeline_full", failing_full)
    monkeypatch.setattr(upscaler, "_upscale_frames", fake_upscale)

    output = await upscaler.run(job)

    assert output.exists()
    assert job.metadata["streamPipelineFallback"] == "boom"
    # El camino clásico corrió DESDE CERO: extracción PNG + encode PNG.
    assert any("-fps_mode" in command for command in upscaler.commands)
    assert any("-framerate" in command for command in upscaler.commands)


async def test_stream_pipeline_full_integration_no_interp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Integración real (spec "Testing/Integración", conteo 1→1 y orden):
    # FramePipeline + FfmpegFrameSource + etapa de upscale reales; sesión ONNX
    # y ambos procesos ffmpeg son fakes.
    upscaler = make_streaming_upscaler_with_real_engines(tmp_path, monkeypatch)
    job = make_stream_job(tmp_path, device="cpu")
    job.metadata.update({"sourceWidth": SOURCE_W, "sourceHeight": SOURCE_H, "framesTotal": 3})
    decode_proc = FakeDecodeProc(raw_source_frames(3))
    sink_proc = FakeEncodeProc()
    monkeypatch.setattr(FfmpegFrameSource, "_spawn", lambda self, command: decode_proc)
    monkeypatch.setattr(RawPipeEncoder, "_spawn", lambda self, command: sink_proc)

    ok = await upscaler._try_stream_pipeline_full(job, 1, tmp_path / "out.mp4", "24/1", None, [])

    assert ok is True
    assert job.metadata["streamPipeline"] is True
    data = sink_proc.stdin.getvalue()
    frame_bytes = (SOURCE_H * 2) * (SOURCE_W * 2) * 3  # la sesión fake dobla
    assert len(data) == 3 * frame_bytes
    assert [data[i * frame_bytes] for i in range(3)] == [0, 17, 34]  # orden fuente


async def test_full_pipeline_cancel_waits_for_worker_before_reraising(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Mismo contrato shield+await del raw-pipe/motores: al propagar el cancel,
    # el worker YA terminó — la limpieza del work-dir no corre contra threads vivos.
    upscaler = make_stream_upscaler(tmp_path)
    job = make_stream_job(tmp_path)
    finished = threading.Event()

    def blocking(job_arg, source_factory, stage_factory, device, width, height, expected, command, counter, cancel_event):
        cancel_event.wait(timeout=10)
        time.sleep(0.2)  # simula un teardown no interrumpible en vuelo
        finished.set()

    monkeypatch.setattr(upscaler, "_run_stream_pipeline_blocking", blocking)

    task = asyncio.create_task(
        upscaler._try_stream_pipeline_full(job, 1, tmp_path / "out.mp4", "24/1", None, [])
    )
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert finished.is_set(), "el cancel propagó antes de que el worker terminara"
```

- [ ] **Step 2: Correr y verificar que fallan**

Run: `.venv\Scripts\python -m pytest tests/test_video_upscaler.py -k "full" -v`
Expected: FAIL con `AttributeError: 'RecordingRunProcessUpscaler' object has no attribute '_try_stream_pipeline_full'` (monkeypatch sobre atributo inexistente) y equivalentes.

- [ ] **Step 3: Implementar**

En `video_upscaler.py`:

1. Import nuevo: `from app.services.engines.ffmpeg_frame_source import FfmpegFrameSource`.

2. Extraer `_prepare_audio_mux` (el bloque inline de `_run_pipeline`, líneas ~176-184, movido tal cual):

```python
    async def _prepare_audio_mux(
        self, job: VideoUpscaleJob, has_audio: bool, audio_path: Path
    ) -> tuple[Path | None, list[str]]:
        # Extraído del cuerpo de _run_pipeline para poder prepararlo ANTES del
        # stream pipeline completo (que encodea sin pasar por el camino PNG).
        if job.keep_audio and has_audio:
            prepared_audio_path, audio_codec_args = await self._prepare_audio(job, audio_path)
            return self._usable_audio_or_none(prepared_audio_path), audio_codec_args
        if job.keep_audio and job.audio_enhance:
            job.metadata["audioEnhanced"] = "skipped_no_audio"
        elif job.keep_audio and job.audio_restore:
            job.metadata["audioRestored"] = "skipped_no_audio"
        return None, []
```

3. `_try_stream_pipeline_full` (sección raw-pipe, junto a `_try_stream_pipeline_from_dir`):

```python
    async def _try_stream_pipeline_full(
        self,
        job: VideoUpscaleJob,
        fps_multiplier: int,
        output_path: Path,
        fps: str,
        audio_mux_path: Path | None,
        audio_codec_args: list[str],
    ) -> bool:
        """Pipeline completo decode→(GMFSS)→upscale→encode, sin ningún PNG.
        True = output_path finalizado; False = fallback clásico DESDE CERO
        (cancel/stall SÍ propagan). fps_multiplier lo consume la rama de
        interpolación (Task 11)."""
        width = int(job.metadata.get("sourceWidth") or 0)
        height = int(job.metadata.get("sourceHeight") or 0)
        if width <= 0 or height <= 0:
            return False
        try:
            encoder_name = await asyncio.to_thread(self._resolve_video_encoder, job)
            job.metadata["videoEncoder"] = encoder_name
            # Sin interp la salida iguala a la fuente; framesTotal None (VFR) es
            # honesto: sin validación estricta de conteo en ese caso.
            expected_output_count = job.metadata.get("framesTotal")
            encode_fps = fps
            gmfss_stage_factory = None
            # Etapa colapsada (decisión de stepper del plan): extract/interp
            # quedan "done" al avanzar; framesDone lo cuenta el sink de encode.
            advance_video_stage(job, "upscaling_frames")
            source = FfmpegFrameSource(
                self.settings.ffmpeg_binary_path,
                job.source_path,
                width,
                height,
                self.settings.ffmpeg_decode_threads,
            )
            await self._run_stream_pipeline(
                job,
                source_factory=source.frames,
                gmfss_stage_factory=gmfss_stage_factory,
                width=width,
                height=height,
                expected_output_count=expected_output_count,
                output_path=output_path,
                encode_fps=encode_fps,
                audio_mux_path=audio_mux_path,
                audio_codec_args=audio_codec_args,
                encoder_name=encoder_name,
            )
        except (asyncio.CancelledError, VideoStallError):
            raise
        except Exception as exc:  # noqa: BLE001 - CUALQUIER fallo -> clásico desde cero
            logger.warning(
                "stream pipeline (completo) falló (%s); se usa el camino clásico desde cero", exc
            )
            job.metadata["streamPipelineFallback"] = str(exc)
            output_path.unlink(missing_ok=True)
            return False
        job.metadata["streamPipeline"] = True
        job.metadata["outputFps"] = encode_fps
        return True
```

4. Reestructurar `_run_pipeline`: entre el estampado de metadata del probe y la extracción, insertar el intento full y mover el audio al helper. El tramo queda así (la extracción y todo lo posterior no cambian, salvo el `if not audio_prepared`):

```python
        job.metadata["framesTotal"] = resolve_frames_total(probe, video_stream, fps)

        output_path = self.settings.outputs_path / f"{job.id}.{job.output_container}"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        stream_mode = await self._resolve_stream_pipeline_mode(job, fps_multiplier)
        audio_mux_path: Path | None = None
        audio_codec_args: list[str] = []
        audio_prepared = False
        # Task 11 quita la condición de no-interpolación al habilitar GMFSS en
        # el modo full; mientras tanto los jobs GMFSS siguen el camino clásico.
        if stream_mode == STREAM_MODE_FULL and not self._interpolation_requested(
            fps_multiplier, job.target_fps
        ):
            audio_mux_path, audio_codec_args = await self._prepare_audio_mux(job, has_audio, audio_path)
            audio_prepared = True
            if await self._try_stream_pipeline_full(
                job, fps_multiplier, output_path, fps, audio_mux_path, audio_codec_args
            ):
                self._finalize_output(job, output_path)
                return output_path
            stream_mode = None  # fallback: camino clásico desde cero

        advance_video_stage(job, "extracting_frames")
        async with self._track_frame_progress(job, frames_in, "extracting_frames"):
            await self._run_process(
                [
                    str(self.settings.ffmpeg_binary_path),
                    "-y",
                    "-i",
                    str(job.source_path),
                    "-fps_mode",
                    "passthrough",
                    "-threads",
                    str(self.settings.ffmpeg_decode_threads),
                    # Extracted frames are throwaway input for the upscaler, so pay
                    # the cheapest zlib level instead of ffmpeg's default.
                    "-compression_level",
                    "1",
                    str(frames_in / "%08d.png"),
                ]
            )

        if not audio_prepared:
            audio_mux_path, audio_codec_args = await self._prepare_audio_mux(job, has_audio, audio_path)

        encode_frames_dir, encode_fps = await self._interpolate_and_upscale(
            job, frames_in, frames_out, fps, fps_multiplier, output_path,
            audio_mux_path, audio_codec_args, stream_mode
        )
```

(Las dos definiciones viejas de `output_path` y el bloque inline de audio se eliminan — quedaron cubiertos arriba. La resolución de `stream_mode` que Task 9 puso justo antes de `_interpolate_and_upscale` se mueve acá.)

- [ ] **Step 4: Correr y verificar verde**

Run: `.venv\Scripts\python -m pytest tests/test_video_upscaler.py tests/test_pipeline_stage_order.py tests/test_video_backend_dispatch.py -q`
Expected: PASS (los tests de `test_pipeline_stage_order.py` siguen intactos: sus upscalers no tienen `onnx_video_engine`, el gate devuelve `None` y el flujo clásico es idéntico).

- [ ] **Step 5: Commit**

```bash
git add app/services/video_upscaler.py tests/test_video_upscaler.py
git commit -m "feat: pipeline completo decode→upscale→encode sin PNGs para jobs sin interpolación"
```

---

### Task 11: GMFSS dentro del pipeline completo (decode→interp→upscale→encode)

El camino que corrige el 1.7x de la fusión eliminada: GMFSS entra como etapa CON su propio thread entre decode y upscale (overlap real). Se habilita quitando la condición de no-interpolación en `_run_pipeline`, agregando la rama de interpolación en `_try_stream_pipeline_full` y re-agregando `_interpolated_encode_fps` (mismo cuerpo que el que Task 1 eliminó — vuelve a tener consumidor).

**Files:**
- Modify: `app/services/video_upscaler.py` (`_try_stream_pipeline_full`, `_run_pipeline`, `_interpolated_encode_fps`)
- Test: `tests/test_video_upscaler.py`

**Interfaces:**
- Consumes: `GmfssEngine.build_stream_stage` (Task 7), `compute_target_frame_count`/`compute_interpolated_fps`/`format_fps_fraction` (existentes, ya importados).
- Produces: `_interpolated_encode_fps(fps: str, fps_multiplier: int, target_fps: str | None) -> str` (staticmethod); la rama de interpolación de `_try_stream_pipeline_full` (misma firma).

- [ ] **Step 1: Escribir los tests que fallan**

En `tests/test_video_upscaler.py`, agregar (imports nuevos arriba: `import json`, `from app.services.engines.gmfss.assets import GRAPH_NAMES`, `from app.services.engines.gmfss_engine import GmfssEngine`):

```python
FULL_H, FULL_W = 16, 24  # resolución "padded" GMFSS de juguete (no cuadrada a propósito)


def make_combined_settings(tmp_path: Path) -> Settings:
    """Settings que satisfacen al GMFSS engine (model dir + ENABLE_GMFSS) y al
    motor ONNX de video (builtin dir aislado) — patrón del viejo test de fusión."""
    gmfss_dir = tmp_path / "gmfss"
    gmfss_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "resolution": {"fixed_padded_hw": [FULL_H, FULL_W]},
        "required_files": ["manifest.json"] + [f"{name}.onnx" for name in GRAPH_NAMES],
    }
    (gmfss_dir / "manifest.json").write_text(json.dumps(manifest))
    for name in GRAPH_NAMES:
        (gmfss_dir / f"{name}.onnx").write_bytes(b"fake")
    return make_stream_settings(tmp_path, ENABLE_GMFSS=True, GMFSS_MODEL_DIR=str(gmfss_dir))


class FakeGmfssSession:
    """Sesión fake determinista de los 4 grafos GMFSS — copiada de
    tests/test_gmfss_engine.py (FakeSession)."""

    def __init__(self, name: str) -> None:
        self.name = name

    def run(self, _outputs, feeds):
        if self.name == "featurenet":
            n, _c, h, w = feeds["img"].shape
            return [
                np.full((n, ch, h // div, w // div), 1.0, dtype=np.float32)
                for ch, div in zip((4, 6, 8), (2, 4, 8))
            ]
        if self.name == "gmflow":
            n, _c, h, w = feeds["img0_half"].shape
            return [np.full((n, 2, h, w), 2.0, dtype=np.float32)]
        if self.name == "metricnet":
            n, _c, h, w = feeds["img0_half"].shape
            metric = np.zeros((n, 1, h, w), dtype=np.float32)
            return [metric.copy(), metric.copy()]
        if self.name == "fusionnet":
            n = feeds["fusion_rgb"].shape[0]
            h_half, w_half = feeds["fusion_rgb"].shape[2], feeds["fusion_rgb"].shape[3]
            return [np.full((n, 3, h_half * 2, w_half * 2), 0.5, dtype=np.float32)]
        raise AssertionError(self.name)


def fake_gmfss_sessions(_device: str):
    return {name: FakeGmfssSession(name) for name in GRAPH_NAMES}


def make_gmfss_streaming_upscaler(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> VideoUpscaler:
    settings = make_combined_settings(tmp_path)
    settings.builtin_onnx_path.mkdir(parents=True, exist_ok=True)
    (settings.builtin_onnx_path / "realesr-animevideov3-x4-uint8.onnx").write_bytes(b"fake")
    gmfss = GmfssEngine(settings, GpuSessionCoordinator())
    monkeypatch.setattr(gmfss, "_create_sessions", fake_gmfss_sessions)
    onnx_video = OnnxVideoUpscaler(
        settings, ModelRegistry(settings), DevicesService(settings), GpuSessionCoordinator()
    )
    monkeypatch.setattr(onnx_video, "_create_session", lambda model_path, device: Double2xUint8Session())
    return VideoUpscaler(
        settings,
        FakeNcnnEngine(),  # type: ignore[arg-type]
        FakeMediaTools(),  # type: ignore[arg-type]
        gmfss_engine=gmfss,
        onnx_video_engine=onnx_video,
        model_registry=ModelRegistry(settings),
        devices=DevicesService(settings),
    )


# ---------------------------------------------------------------------------
# GMFSS dentro del pipeline completo: conteo 1→2x, orden y fallback
# ---------------------------------------------------------------------------


async def test_stream_pipeline_full_with_gmfss_doubles_frame_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    upscaler = make_gmfss_streaming_upscaler(tmp_path, monkeypatch)
    job = make_stream_job(tmp_path, interp_engine="gmfss", fps_multiplier=2, device="cpu")
    job.metadata.update({"sourceWidth": SOURCE_W, "sourceHeight": SOURCE_H, "framesTotal": 3})
    decode_proc = FakeDecodeProc(raw_source_frames(3))
    sink_proc = FakeEncodeProc()
    monkeypatch.setattr(FfmpegFrameSource, "_spawn", lambda self, command: decode_proc)
    monkeypatch.setattr(RawPipeEncoder, "_spawn", lambda self, command: sink_proc)

    ok = await upscaler._try_stream_pipeline_full(job, 2, tmp_path / "out.mp4", "24/1", None, [])

    assert ok is True
    assert job.metadata["streamPipeline"] is True
    assert job.metadata["framesTotal"] == 6  # denominador honesto: salida interpolada
    assert job.metadata["outputFps"] == "48/1"
    data = sink_proc.stdin.getvalue()
    frame_bytes = (SOURCE_H * 2) * (SOURCE_W * 2) * 3  # la sesión fake dobla
    assert len(data) == 6 * frame_bytes  # conteo 1→2x (3 fuente -> 6 salida)
    # Orden: s0, interp, s1, interp, interp, s2 (plan(3→6) = [1, 2]); los frames
    # fuente pasan verbatim y doblados conservan su primer byte uniforme.
    assert data[0 * frame_bytes] == 0
    assert data[2 * frame_bytes] == 17
    assert data[5 * frame_bytes] == 34


async def test_stream_pipeline_full_gmfss_falls_back_on_source_count_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # El decode entrega 2 frames pero el probe prometió 3: el plan GMFSS no
    # cierra, la etapa revienta en flush() y el job cae al clásico (no falla).
    upscaler = make_gmfss_streaming_upscaler(tmp_path, monkeypatch)
    job = make_stream_job(tmp_path, interp_engine="gmfss", fps_multiplier=2, device="cpu")
    job.metadata.update({"sourceWidth": SOURCE_W, "sourceHeight": SOURCE_H, "framesTotal": 3})
    decode_proc = FakeDecodeProc(raw_source_frames(2))
    sink_proc = FakeEncodeProc()
    monkeypatch.setattr(FfmpegFrameSource, "_spawn", lambda self, command: decode_proc)
    monkeypatch.setattr(RawPipeEncoder, "_spawn", lambda self, command: sink_proc)
    output_path = tmp_path / "out.mp4"

    ok = await upscaler._try_stream_pipeline_full(job, 2, output_path, "24/1", None, [])

    assert ok is False
    assert "streamPipelineFallback" in job.metadata
    assert not output_path.exists()


async def test_run_routes_gmfss_job_through_full_pipeline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    upscaler = make_recording_upscaler(tmp_path)
    source_path = upscaler.settings.uploads_path / "clip.mp4"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_bytes(b"fake-video-bytes")
    job = make_stream_job(tmp_path, source_path=source_path, interp_engine="gmfss", fps_multiplier=2)
    seen: dict = {}

    async def fake_full(job_arg, fps_multiplier, output_path, fps, mux, codec_args):
        seen["fps_multiplier"] = fps_multiplier
        output_path.write_bytes(b"fake-video")
        return True

    monkeypatch.setattr(upscaler, "_try_stream_pipeline_full", fake_full)

    output = await upscaler.run(job, fps_multiplier=2)

    assert output.exists()
    assert seen["fps_multiplier"] == 2
    assert all("-fps_mode" not in command for command in upscaler.commands), "GMFSS full no debe extraer PNGs"
```

- [ ] **Step 2: Correr y verificar que fallan**

Run: `.venv\Scripts\python -m pytest tests/test_video_upscaler.py -k gmfss -v`
Expected: FAIL — `test_run_routes_gmfss_job_through_full_pipeline` extrae PNGs (la condición de no-interpolación de Task 10 sigue puesta) y los de integración validan contra el conteo fuente (3) en vez del interpolado (6).

- [ ] **Step 3: Implementar**

En `video_upscaler.py`:

1. Re-agregar el helper (junto a `_finalize_output`; mismo cuerpo que el eliminado en Task 1 — vuelve a tener consumidor):

```python
    @staticmethod
    def _interpolated_encode_fps(fps: str, fps_multiplier: int, target_fps: str | None) -> str:
        # Mismos valores de encode-fps que devuelven _interpolate_to_target_fps /
        # _interpolate_by_multiplier, para que el pipeline encodee exactamente a
        # la misma tasa que el camino de dos pasadas.
        if target_fps is not None:
            return format_fps_fraction(target_fps)
        new_rate = compute_interpolated_fps(fps, fps_multiplier)
        return f"{new_rate.numerator}/{new_rate.denominator}"
```

2. En `_try_stream_pipeline_full`, reemplazar las tres líneas

```python
            expected_output_count = job.metadata.get("framesTotal")
            encode_fps = fps
            gmfss_stage_factory = None
```

por la versión con la rama de interpolación:

```python
            expected_output_count = job.metadata.get("framesTotal")
            encode_fps = fps
            gmfss_stage_factory = None
            if self._interpolation_requested(fps_multiplier, job.target_fps):
                # El gate garantiza framesTotal conocido y gmfss_engine presente
                # para este camino (RIFE nunca llega acá: es modo híbrido).
                source_count = int(job.metadata["framesTotal"])
                expected_output_count = (
                    compute_target_frame_count(source_count, fps, job.target_fps)
                    if job.target_fps is not None
                    else source_count * fps_multiplier
                )
                encode_fps = self._interpolated_encode_fps(fps, fps_multiplier, job.target_fps)
                # Denominador honesto del stepper colapsado: la salida interpolada.
                job.metadata["framesTotal"] = expected_output_count
                gmfss_stage_factory = lambda device: self.gmfss_engine.build_stream_stage(  # noqa: E731
                    source_count, expected_output_count, device
                )
```

3. En `_run_pipeline`, quitar la condición transitoria de Task 10: el gate del modo full queda solo `if stream_mode == STREAM_MODE_FULL:` (borrar `and not self._interpolation_requested(fps_multiplier, job.target_fps)` y su comentario "Task 11 quita...").

- [ ] **Step 4: Correr y verificar verde**

Run: `.venv\Scripts\python -m pytest tests/test_video_upscaler.py tests/test_gmfss_engine.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add app/services/video_upscaler.py tests/test_video_upscaler.py
git commit -m "feat: GMFSS dentro del pipeline completo decode→interp→upscale→encode con overlap real"
```

---

### Task 12: Documentación + suite completa verde

Cierre: README documenta el pipeline nuevo (el `.env.example` ya quedó al día en Tasks 1/8) y la suite backend entera corre verde. El frontend NO se toca (el shape de metadata no cambió — verificado por los tests de Tasks 9-11 que solo usan las claves existentes).

**Files:**
- Modify: `README.md` (sección nueva después de la "Nota histórica" que dejó Task 1)
- Test: suite completa

**Interfaces:**
- Consumes: todo lo anterior. Produces: nada nuevo de código.

- [ ] **Step 1: Documentar en README**

Insertar después de la sección `### Nota histórica: fusión interpolar+escalar (eliminada)` (Task 1):

```markdown
### Pipeline de frames en streaming (`ENABLE_STREAM_PIPELINE`)

Activo por defecto. Conecta decode→(interpolación)→upscale→encode por colas acotadas en memoria (frames rgb24 crudos), con cada etapa en su propio thread — sin materializar PNGs intermedios. El presupuesto de RAM es el mismo `ONNX_VIDEO_MAX_PIPELINE_MB` del pipeline ONNX, repartido globalmente entre todas las colas.

| Camino | Con el pipeline | PNGs eliminados |
|---|---|---|
| Sin interpolación + upscale ONNX builtin | decode→stream→upscale→stream→encode | todos |
| GMFSS + upscale ONNX builtin | decode→stream→GMFSS→stream→upscale→stream→encode | todos |
| RIFE + upscale ONNX builtin | decode→PNG→RIFE→(lee sus PNGs)→upscale→**stream**→encode | los de salida (los más pesados) |
| Upscale NCNN (binario) / modelos HF-ONNX | camino clásico completo | ninguno |

Ante CUALQUIER excepción del pipeline, el job cae automáticamente al camino clásico desde cero (se registra en el log y en `job.metadata.streamPipelineFallback`) — el job nunca falla por culpa del camino nuevo. `ENABLE_STREAM_PIPELINE=false` restaura el comportamiento anterior completo (incluido el raw-pipe clásico). Diseño: `docs/superpowers/specs/2026-07-25-stream-frame-pipeline-design.md`.
```

- [ ] **Step 2: Correr la suite completa y verificar verde**

Run: `.venv\Scripts\python -m pytest -q`
Expected: PASS (0 failed). Si algo falla, arreglarlo ANTES de commitear — este es el gate de cierre del plan.

- [ ] **Step 3: Verificación final de referencias muertas**

Run: `grep -rn "interp_upscale_fusion\|run_frames_fused" app tests README.md .env.example`
Expected: sin resultados.

- [ ] **Step 4: Commit**

```bash
git add README.md
git commit -m "docs: documenta ENABLE_STREAM_PIPELINE y los caminos del pipeline de frames en streaming"
```

---

## Smoke real manual (gate de promoción — NO es un task, NO va a CI)

Lección GMFSS: la fusión anterior pasó todos los unit tests y midió 1.7x MÁS LENTA en hardware real. **Criterio de éxito del spec: ≥ paridad de throughput en los 3 caminos y cero regresión con flag OFF. Si no mide igual o mejor, no se promueve** (se deja `ENABLE_STREAM_PIPELINE=false` por default en un commit de ajuste y se investiga).

En la máquina real (RX 7800 XT), con un clip 720p de ~1-2 min y perfil 4x (`general-balanced-4x` o `anime-max-detail-4x`):

1. **Baseline (flag OFF)** — en `.env`: `ENABLE_STREAM_PIPELINE=false`. Levantar el server, correr y cronometrar (wall time del job en la UI o `finished_at - started_at` del JSON de estado):
   - Camino A: sin interpolación.
   - Camino B: `interp_engine=gmfss`, `fps_multiplier=2` (requiere `ENABLE_GMFSS=true` + modelos).
   - Camino C: `interp_engine=rife`, `fps_multiplier=2`.
2. **Pipeline (flag ON)** — `ENABLE_STREAM_PIPELINE=true` (default), repetir los 3 jobs idénticos.
3. Comparar por camino: `wall_ON <= wall_OFF` (± ruido de ~3%). Verificar en el estado del job que `metadata.streamPipeline == true` (si aparece `streamPipelineFallback`, el pipeline NO corrió — investigar el motivo en el log antes de medir).
4. Sanity de salida: mismos conteos de frames (`ffprobe -count_frames`), duración y fps de salida idénticos entre ON y OFF; reproducir ambos outputs.
5. Vigilar durante el camino B (GMFSS): VRAM (ambos sets de sesiones quedan residentes — trade-off documentado) y RAM del proceso (el techo teórico es `ONNX_VIDEO_MAX_PIPELINE_MB` + overhead constante).
6. Regresión flag OFF: el job A con flag OFF debe seguir usando el raw-pipe clásico (`metadata.rawPipe == true`) — cero cambio de comportamiento.

Registrar los números en el commit de cierre o en `docs/` si se desea (fuera del alcance de este plan).

---

## Self-review (checklist del skill, aplicado al terminar la redacción)

**1. Cobertura del spec (`2026-07-25-stream-frame-pipeline-design.md`):**
- Decisión "alcance motores externos / RIFE adapter PNG" → Tasks 8 (modo híbrido) y 9 (tramo streaming desde su directorio PNG). ✔
- Decisión "memoria: colas acotadas + presupuesto `ONNX_VIDEO_MAX_PIPELINE_MB`" → Tasks 2 (helper extraído, no duplicado) y 3 (`derive_stream_queue_maxsizes`, presupuesto GLOBAL repartido). ✔
- Decisión "rollout: `ENABLE_STREAM_PIPELINE` default ON + fallback automático" → Task 8 (flag) + Tasks 9/10 (fallback con log + `streamPipelineFallback`, patrón raw-pipe). ✔
- Decisión "se ELIMINA la fusión vieja" → Task 1 (task propia, código+config+tests+docs). ✔
- Arquitectura `FramePipeline`/protocolo `FrameStage`/threads/backpressure → Task 3. Source ffmpeg rawvideo → Task 4. Sink raw-pipe extraído/reusado → Task 5. Etapas envolviendo motores existentes (sesión/tiling/fp16 intactos; GMFSS in-process) → Tasks 6 y 7. Ruteo en `video_upscaler` (tabla de caminos completa, NCNN→clásico vía gate) → Tasks 8-11. ✔
- Manejo de errores (drenar colas, matar ffmpeg, fallback desde cero; cancel via `cancel_event` chequeado en get/put con timeout, procesos muertos en teardown) → Tasks 3/4/5 (unit) + 9/10 (wiring, shield+await). ✔
- Progreso (audio/probing intactos; `framesDone` = frames encodeados por el sink; `framesTotal` con multiplicador; decisión de stepper delegada → resuelta como "etapa colapsada" en la sección propia del header, sin cambios de frontend). ✔
- Testing del spec: unit (colas/backpressure/orden/maxsize/fallback/cancel sin zombies → Task 3; fallback ante excepción de etapa → Tasks 9/10/11; cancel limpio → Tasks 3/10) + integración (source/sink fake + motores fake, conteos 1→1 en Tasks 9/10 y 1→2x en Task 11, orden verificado por bytes) + smoke real manual como sección final (no task). ✔
- Riesgos: overlap (threads por etapa desde el día uno + smoke como gate), RAM 4K (presupuesto global), pipes en Windows (mismo patrón raw-pipe + kill en teardown), VFR (`framesTotal=None` honesto → gate excluye GMFSS y salta la validación estricta de conteo). ✔
- Fases futuras (RIFE→ONNX, decode GPU, chunked-RIFE): fuera de alcance, no se tocan. ✔

**2. Placeholder scan:** sin "TBD/TODO/implementar después/similar a Task N sin código". Todos los pasos de código llevan el código completo; los tests llevan asserts reales; los comandos llevan salida esperada. Los fakes repetidos entre archivos de test (`Double2xUint8Session`, `FakeEncodeProc`, `FakeDecodeProc`, `FakeGmfssSession`) se copian a propósito con la nota "copiado de X" — los archivos de test no se importan entre sí en este repo. Dos re-adiciones deliberadas y anunciadas en ambos extremos: `_chw_float_to_nhwc_uint8` (eliminado T1 → re-agregado T7) y `_interpolated_encode_fps` (eliminado T1 → re-agregado T11).

**3. Consistencia de tipos/firmas entre tasks (verificada cruzando los bloques Interfaces):**
- `derive_queue_maxsize(frame_bytes, budget_bytes, floor, ceiling) -> int` — definida T2, consumida T3 con esa firma. ✔
- `FrameStage.process/flush -> list[np.ndarray]` — T3; `GmfssStreamStage` (T7) y `MapStage` (T3) la cumplen; `_run_stream_pipeline_blocking` (T9) tipa `list[FrameStage]`. ✔
- `FramePipeline(source, stages, sink, queue_maxsizes).run(cancel_event) -> int` — T3, consumida T9 exactamente así. ✔
- `FfmpegFrameSource(...).frames(cancel_event)` — T4; T10 pasa `source.frames` como `source_factory` (firma `Callable[[threading.Event], Iterator[np.ndarray]]`). ✔
- `RawPipeEncoder(command, summarize_error).start()/write_frame(hwc)/frames_written/finish()/kill()` — T5; consumida en T5 (refactor raw-pipe legacy) y T9 (sink). ✔
- `build_frame_upscaler(engine_model_name, device) -> Callable[[np.ndarray], np.ndarray]` — T6, consumida T9. ✔
- `build_stream_stage(source_frame_count, target_frame_count, device) -> GmfssStreamStage` — T7, consumida T11 vía lambda `Callable[[str], FrameStage]`. ✔
- `_resolve_stream_pipeline_mode(job, fps_multiplier) -> str | None` con `STREAM_MODE_FULL/"full"`, `STREAM_MODE_HYBRID/"hybrid"` — T8, consumida T9 (paso a `_interpolate_and_upscale`) y T10 (`_run_pipeline`). ✔
- `_try_stream_pipeline_from_dir(job, frames_dir, output_path, encode_fps, audio_mux_path, audio_codec_args) -> bool` — T9, monkeypatcheada en tests T9 con la misma aridad. ✔
- `_try_stream_pipeline_full(job, fps_multiplier, output_path, fps, audio_mux_path, audio_codec_args) -> bool` — T10, misma firma en T11 y en los fakes de tests T10/T11. ✔
- `_interpolate_and_upscale(..., stream_mode)` — T9 agrega el 9º parámetro; los tests T9 lo llaman con 10 argumentos posicionales (self implícito + 9). ✔
- Convención de frames NHWC uint8 `[1,H,W,3]` declarada en el header y respetada por source (T4 `_to_frame`), etapas (T6/T7) y sink (T9 `frame_nhwc[0]` → HWC). ✔

Gaps detectados y corregidos durante esta revisión: (a) el gate necesita `sourceWidth/sourceHeight` para el umbral de píxeles → se documentó la precondición "metadata del probe ya estampada" y los tests la priman; (b) el fallback del modo full re-usa el audio ya preparado (`audio_prepared`) para no duplicar `audioEnhanced/audioRestored`; (c) los tests legacy de `test_pipeline_stage_order.py` no se rompen porque sus upscalers no tienen `onnx_video_engine` (gate → `None`) — verificado en los pasos de suite de T9/T10.
