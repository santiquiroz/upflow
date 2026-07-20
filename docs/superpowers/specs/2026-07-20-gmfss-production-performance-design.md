# GMFSS en producción: contención de sesiones GPU + fusión interpolar→escalar — Design

**Fecha:** 2026-07-20
**Estado:** Approved (pendiente de plan de implementación)

## Motivación

Job real (episodio completo, 79930 frames objetivo, `interp_engine=gmfss`, `audio_restore=audiosr`, `device=dml:0`) midió ~0.056 fps — muy por debajo de los 0.38-0.72 fps medidos en benchmarks aislados del puerto (`port-gmfss-onnx` Task 3.2). Investigación en caliente (sin re-ejecutar el job) encontró dos causas reales, independientes:

1. **`pyopencl` no estaba instalado** en el install local → el splat GPU (OpenCL, Task 3.1 del puerto) caía siempre a CPU. Ya corregido de forma ad-hoc en el install real (`pip install pyopencl` en el python embebido) — este spec formaliza el fix a nivel de código/documentación, no solo del install de un usuario.
2. **Contención de sesiones DirectML entre motores**: ninguno de `AudioSrRestorer`, `ApolloRestorer`, `GmfssEngine`, `OnnxVideoUpscaler`, `OnnxUpscaler` libera sus sesiones ONNX Runtime al terminar su etapa. En el job real, las sesiones DirectML de AudioSR (etapa `restoring_audio`, ya terminada) seguían residentes en memoria/GPU cuando arrancó `interpolating_frames` (GMFSS, 4 sesiones más) en el mismo proceso y mismo device. El puerto (`port-gmfss-onnx` Task 3.2) ya documentó que DirectML se degrada con múltiples sesiones ONNX concurrentes en un proceso (root-cause real, no arreglable a nivel de onnxruntime sin su API C) — en producción el problema es peor porque conviven sesiones de motores DISTINTOS, no solo las 4 de GMFSS entre sí.

Además, aprovechando la investigación: **interpolar y escalar corren como dos pasadas separadas con un viaje completo a disco entre medio** (GMFSS escribe PNGs interpolados → el escalador los vuelve a leer). Para el caso donde ambos motores corren in-process (GMFSS + escalador ONNX, sin binarios externos de por medio) esto es evitable.

## Alcance

Dos piezas, un spec, en fases (igual que audio/subs — comparten área del pipeline de video):

- **Fase 1 — Coordinador de sesiones GPU**: elimina la contención real, impacto esperado grande (posible 3-7x en el caso medido), cambio acotado.
- **Fase 2 — Fusión interpolar→escalar (GMFSS+ONNX in-process)**: elimina un round-trip de disco completo, impacto menor y honesto (el cuello real es cómputo de red, no I/O) pero real.

## Fase 1 — Coordinador de sesiones GPU

### `GpuSessionOwner` — protocolo compartido

```python
class GpuSessionOwner(Protocol):
    def release_device(self, device: str) -> None: ...
```

Cada motor con cache de sesiones (`AudioSrRestorer`, `ApolloRestorer`, `GmfssEngine`, `OnnxVideoUpscaler`, `OnnxUpscaler`) implementa `release_device(device)`: vacía su entrada de ese device específico en su `_session_cache` (si existe), deja intactas las entradas de otros devices. Método idempotente — llamarlo cuando no hay nada cacheado para ese device no hace nada.

### `GpuSessionCoordinator` — registro por device, no por job

```python
class GpuSessionCoordinator:
    def __init__(self) -> None:
        self._owners: dict[str, GpuSessionOwner] = {}
        self._lock = threading.Lock()

    def acquire(self, device: str, owner: GpuSessionOwner) -> None:
        with self._lock:
            previous = self._owners.get(device)
            if previous is not None and previous is not owner:
                previous.release_device(device)
            self._owners[device] = owner
```

Mutuamente excluyente por device (no por job): dos jobs en devices distintos nunca se pisan; dos motores distintos en el MISMO device sí se desalojan entre sí, sin importar si vienen del mismo job o de jobs consecutivos — es una propiedad del device, no del job.

Cada motor llama `self.gpu_coordinator.acquire(device, self)` como primer paso de su método de construcción/obtención de sesión (`_get_session`/`_get_sessions`/`_create_sessions`), ANTES de construir sesiones nuevas para ese device. Wiring: `app/main.py` construye UN `GpuSessionCoordinator` en el lifespan, se inyecta a los 5 motores (mismo patrón que `devices_service` ya se inyecta a varios).

**Trade-off aceptado explícitamente**: alternar el mismo device entre motores distintos varias veces en una corrida (ej. AudioSR→GMFSS→AudioSR) recarga sesión en el tercer paso — se pierde el ahorro del LRU entre esos dos usos. Se acepta porque la alternativa (dejarlas convivir) es exactamente la contención que se está arreglando.

## Fase 2 — Fusión interpolar→escalar (in-process únicamente)

### Gate de elegibilidad

Solo aplica cuando `job.interp_engine == "gmfss"` Y el escalador resuelto es el backend ONNX in-process (no NCNN, no RIFE — ambos corren como subproceso externo, fuera de alcance de esta fusión). `video_upscaler.py` ya tiene la lógica de resolución de backend (`_resolve_video_encoder`/equivalente para el motor de upscale) — reusar esa resolución, no duplicarla.

### Contrato

`GmfssEngine` gana un modo `run_frames_fused` (nombre a confirmar en el plan) que, en vez de escribir cada frame interpolado a PNG, invoca un callback `upscale_frame: Callable[[np.ndarray], np.ndarray]` inyectado por `video_upscaler.py`, y escribe a PNG solo el resultado YA escalado. El callback lo construye `OnnxVideoUpscaler` a partir de su sesión ya cargada (mismo patrón de inyección que `GraphRunner`/`SplatFn` en el puerto).

Cuando el gate de elegibilidad no se cumple, el pipeline sigue exactamente el camino actual (dos pasadas, sin cambios) — la fusión es un atajo opcional, nunca un requisito para que el job funcione.

### Verificación de impacto (obligatoria, no se asume)

Antes de reclamar cualquier mejora en README/UI: benchmark real en la 7800 XT, mismo clip, camino fusionado vs camino actual, con Fase 1 YA aplicada (para no confundir las dos mejoras). Reportar el número real, cualquiera sea.

## Testing

- **Fase 1**: `test_gpu_session_coordinator.py` — `acquire` de un device nuevo no libera nada; `acquire` del mismo device con otro dueño llama `release_device` en el anterior exactamente una vez; `acquire` repetido del mismo dueño no llama `release_device` (idempotencia); cada motor expone `release_device` y de verdad vacía su cache para ese device (test por motor, 5 motores).
- **Fase 2**: parity pixel-exacta entre salida fusionada y salida del camino actual (mismo frame de entrada, mismo seed/modelo) — sin tolerancia, deben ser bit-idénticos ya que es la MISMA operación de escalado, solo sin el round-trip PNG de por medio. Benchmark real documentado, no estimado.

## Fuera de alcance

- Fusión para RIFE (subproceso externo — requeriría streaming inter-proceso, otro proyecto).
- Arreglar la causa raíz de la contención DirectML a nivel de onnxruntime (requiere su API C, ya descartado como fuera de alcance por el puerto en Task 3.2).
- Cualquier cambio a los pesos/arquitectura de GMFSS — esto es puramente orquestación de recursos en Upflow.
