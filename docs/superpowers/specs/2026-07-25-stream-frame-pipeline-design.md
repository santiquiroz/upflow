# Pipeline de frames en streaming (decode→tensor sin PNGs intermedios) — Design

**Fecha:** 2026-07-25
**Estado:** Approved (brainstormeado con el usuario en sesión; pendiente de plan de implementación)

## Motivación

El pipeline de video materializa **todos** los frames como PNG en disco entre cada etapa: decode→PNG, interpolación→PNG, upscale→PNG, encode. En un episodio 720p→4x son ~45.000 lecturas+escrituras de PNG por etapa (los de salida 4x pesan ~44MB c/u), con el disco como cuello observado en jobs reales (disco al 88% durante extracción mientras la GPU está al 14%). El raw-pipe de salida existente (`ENABLE_RAW_PIPE`) ya demostró la ganancia del enfoque (1.24x en salidas grandes) pero solo cubre el último tramo y solo con interpolación OFF.

Pedido explícito del usuario: la optimización debe servir para **cualquier algoritmo de generación de frames** (RIFE hoy, GMFSS hoy, futuros ports), no solo el camino de upscale.

Antecedente crítico: la fusión GMFSS+upscale existente (`ENABLE_INTERP_UPSCALE_FUSION`) midió **1.7x MÁS LENTA** que las dos pasadas por ser un generador secuencial mono-hilo sin overlap load/compute/save. Esa lección define el requisito núcleo de este diseño: **overlap real entre etapas o no sirve**.

## Decisiones tomadas (brainstorming 2026-07-25)

| Decisión | Elección | Razón |
|---|---|---|
| Alcance motores externos | Opción A: pipeline streaming + RIFE como adapter PNG en su tramo | RIFE es binario NCNN sin modo pipe; chunkearlo = relanzar el binario cientos de veces (model-load c/u). Port de RIFE a ONNX = fase futura |
| Memoria | Colas acotadas con backpressure, presupuesto reusando la matemática de `ONNX_VIDEO_MAX_PIPELINE_MB` | Nunca vive el video entero en RAM; peor caso = presupuesto configurado (~20 frames 4x en vuelo con el default de 1GB) |
| Rollout | Flag `ENABLE_STREAM_PIPELINE` **default ON** + fallback automático al camino PNG ante cualquier fallo | Patrón raw-pipe existente; el fallback acota el riesgo |
| Fusión GMFSS vieja | Se ELIMINA (`enable_interp_upscale_fusion` y su camino) | Reemplazada por este pipeline con overlap; era opt-in, apagada y más lenta |

## Arquitectura

**`FramePipeline`**: etapas conectadas por colas acotadas en memoria (frames rgb24 uint8 como `np.ndarray` + índice), cada etapa en su propio thread. Backpressure: cola llena ⇒ el productor bloquea. La GPU sigue serializada por el `GpuSessionCoordinator`/`DeviceSemaphores` existentes.

- **Protocolo `FrameStage`**: recibe un frame, emite 1..N (upscaler 1→1; interpolador 1→2x con el frame siguiente — mantiene una ventana de 2).
- **Source**: ffmpeg subprocess `-f rawvideo -pix_fmt rgb24 pipe:1` leído por thread (espejo del writer raw-pipe existente; cero dependencias nuevas). Resolución/fps del probe ya capturado en el job.
- **Sink**: writer raw-pipe existente hacia ffmpeg stdin (extraído/reusado, no duplicado).
- **Adapters para binarios**: RIFE conserva materialización PNG completa de SU tramo (el binario exige el directorio entero); sus PNGs de salida se leen en streaming hacia el upscale.

### Qué gana cada camino

| Camino | Con pipeline | PNGs eliminados |
|---|---|---|
| Sin interpolación + upscale ONNX | decode→stream→upscale→stream→encode | todos |
| GMFSS + upscale ONNX | decode→stream→GMFSS→stream→upscale→stream→encode | todos (y corrige el 1.7x de la fusión) |
| RIFE + upscale ONNX | decode→PNG→RIFE→(lee sus PNGs)→upscale→**stream**→encode | los de salida 4x (los más pesados); la entrada de RIFE queda en PNG |
| Upscale NCNN (binario) | fallback al camino clásico completo | ninguno (el binario exige PNGs en ambos lados) |

## Componentes

- `app/services/frame_pipeline.py` (nuevo): protocolo `FrameStage`, runner de threads por etapa, colas con maxsize derivado del presupuesto RAM (helper compartido extraído de `onnx_video_upscaler.py`).
- `app/services/engines/ffmpeg_frame_source.py` (nuevo): decode rawvideo por pipe bajo el guarded process runner actual (timeout + kill-on-cancel).
- Etapas envolviendo motores existentes reusando sus internals: sesión ONNX del upscaler (tiling/fp16/whole-frame intactos), GMFSS in-process.
- `video_upscaler.py`: ruteo — flag ON + backend ONNX + (sin interp | GMFSS) → pipeline completo; RIFE → híbrido (PNG entrada, stream salida); NCNN o cualquier excepción del pipeline → fallback clásico (log + flag en metadata).
- `config.py`: `ENABLE_STREAM_PIPELINE: bool = true`. Se elimina `enable_interp_upscale_fusion`.

## Manejo de errores

- Fallo de cualquier etapa → drenar colas, matar procesos ffmpeg, **fallback al camino clásico desde cero** (el job nunca falla por culpa del camino nuevo; mismo patrón que el raw-pipe actual).
- Cancel → mismo `cancel_event` de hoy; threads lo chequean en cada get/put con timeout; procesos bajo el guarded runner mueren con el cancel.

## Progreso

- Etapas de audio/probing intactas. Las etapas de frames concurrentes reportan por el sink: `framesDone` = frames encodeados (monotónico, honesto). `framesTotal`/ETA con la lógica existente (multiplicador de interpolación incluido). La representación exacta del stepper (etapas paralelas activas vs etapa colapsada "processing frames") la decide el plan mirando el UI actual.

## Testing

- Unit: colas/backpressure/orden con motores fake; maxsize correcto según tamaño real de frame y presupuesto; fallback ante excepción de cualquier etapa; cancel limpio sin threads zombie ni procesos huérfanos.
- Integración: pipeline completo con source/sink fake y motores fake verificando conteos 1→1 y 1→2x y orden de frames.
- **Smoke real manual (criterio de éxito, lección GMFSS)**: 720p→4x en la máquina real midiendo vs camino clásico — **≥ paridad de throughput en los 3 caminos** y cero regresión con flag OFF. Si no mide igual o mejor, no se promueve.

## Riesgos aceptados

| Riesgo | Mitigación |
|---|---|
| Overlap mal implementado repite el 1.7x de la fusión | Criterio de éxito explícito de paridad medida; threads por etapa desde el día uno; el smoke real es gate de promoción |
| RAM con videos 4K de entrada (frames de entrada grandes) | Presupuesto global compartido entre TODAS las colas del pipeline, no por cola |
| rawvideo pipe en Windows (bloqueos de pipe) | Mismo patrón ya probado en producción por el raw-pipe de salida; guarded runner con watchdog |
| VFR / fps raros en el decode a rawvideo | `-fps_mode passthrough` como ya usa la extracción; frames_total honesto (None) cuando no es determinable |

## Fases futuras (fuera de este spec)

1. **Port de RIFE a ONNX in-process** (como GMFSS) — cierra la última fila con PNGs.
2. **Decode por GPU (D3D11VA)** — solo si el decode CPU llega a ser cuello medido tras eliminar los PNGs (hoy no lo es).
3. **Chunked-RIFE** si upstream agrega modo pipe/stdin.
