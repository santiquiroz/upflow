# Admisión de jobs por capacidad real de dispositivo (subproyecto B) — Design

**Fecha:** 2026-07-24
**Estado:** Approved (pendiente de plan de implementación)

## Motivación

Hoy la admisión de jobs a un device (`app/services/device_semaphores.py`, `DeviceSemaphores`) solo cuenta jobs concurrentes propios: `capacity_for(device_id)` devuelve `PER_DEVICE_GPU_CONCURRENCY`/`CPU_CONCURRENCY` fijo, y un job entra mientras `in_flight < capacity`. Esto no ve el estado real del recurso:

- **Presión externa**: procesos fuera de Upflow (juegos, otras apps GPU-intensive, navegador con muchas pestañas) compiten por la misma VRAM/RAM. `DeviceSemaphores` no tiene forma de saberlo hoy.
- **Jobs heterogéneos bajo el mismo conteo**: un upscale ONNX simple y una generación SD1.5 (pipeline ~4GB) o GMFSS (4 sesiones ONNX simultáneas) cuentan igual como "1 slot" — el conteo no refleja el consumo real de VRAM.

Un job puede admitirse por conteo y fallar en ejecución por falta de memoria, cuando la causa real es memoria ya comprometida (propia o externa) que el semáforo actual no mide.

`app/services/devices_service.py` ya enumera adapters GPU vía DXGI (`_enumerate_dxgi_adapter_names`, `_DxgiAdapterDesc1`) pero solo lee `DedicatedVideoMemory` (capacidad TOTAL del adapter, no uso en vivo) y ese dato ni siquiera se expone hoy en `DeviceInfo`/`list_devices()`. La infraestructura de enumeración existe; falta la parte de medición en vivo.

Este subproyecto agrega un segundo criterio de admisión, ortogonal al conteo de jobs: recursos libres reales del device seleccionado por el job, con vista a que el hardware/plataforma disponible va a cambiar en el futuro (hoy AMD/DirectML/Windows; mañana puede ser otro vendor de GPU, otro backend, otro SO) — el diseño no asume el hardware actual como fijo.

## Alcance

**MVP (este spec):**
- Protocolo `ResourceProbe` genérico por tipo de device (gpu/cpu/npu), pluggable.
- `DxgiVramProbe` (gpu, Windows/DirectML — hardware real disponible hoy): VRAM libre en vivo vía `IDXGIAdapter3::QueryVideoMemoryInfo` (`Budget - CurrentUsage`), no el campo `DedicatedVideoMemory` (total) que ya existe.
- `SystemRamProbe` (cpu — hardware real disponible hoy): RAM libre vía `psutil.virtual_memory().available`.
- Integración en `DeviceSemaphores`: admisión exige `in_flight < capacity` **y** `probe.free_capacity_mb(device_id) >= MIN_FREE_MB`, bajo el mismo `asyncio.Condition` (atómico, sin gate separado).
- Re-chequeo periódico del predicado mientras hay waiters gateados por recursos (cubre liberación de VRAM/RAM por causas externas, que no dispara `notify_all()` interno).
- Comportamiento sin capacidad: **esperar en cola** (no rechazar), consistente con el `acquire()` bloqueante ya existente.
- Fail-open: cualquier fallo de un probe (API nativa, dependencia ausente) degrada a "sin chequeo de recursos", nunca bloquea ni tira un job.

**Explícitamente fuera del MVP:**
- `NullProbe` para NPU: protocolo listo, sin lógica real — sin hardware NPU disponible para validar (confirmado: `devices_service.py` ya documenta "NPU detection has no enumeration story yet").
- Probes reales para plataformas no-Windows (Linux, ROCm nativo, CUDA nativo) — la interfaz `ResourceProbe` es agnóstica de plataforma; implementaciones concretas quedan para cuando exista ese hardware/target.
- Estimación de necesidad por-modelo (tabla de tamaños o medición durante instalación) — el MVP usa un umbral de headroom genérico, no estimación por job. Ver "Fases futuras".

## Decisiones tomadas (brainstorming 2026-07-24)

| Decisión | Elección | Razón |
|---|---|---|
| Comportamiento sin recursos libres | Esperar en cola (bloqueo) | Mismo patrón que `DeviceSemaphores.acquire()` ya usa para conteo; pedido explícito del usuario — sin fricción de reintento manual |
| Estimación de necesidad por job | Umbral de headroom libre genérico (`MIN_FREE_MB`), no tabla por modelo/kind | Escala a cualquier modelo HF que se instale a futuro (subproyecto A) sin mantener una tabla de tamaños; pedido explícito del usuario |
| Integración arquitectónica | Extender `DeviceSemaphores` directamente, no un gate separado por encima | Reusa el `asyncio.Condition` único ya probado (SP7): check-and-reserve atómico, sin riesgo de dos loops de espera desincronizados |
| Recursos con lógica real en MVP | GPU/VRAM (DXGI) + CPU/RAM (psutil) | Único hardware disponible para validar hoy; pedido explícito del usuario ("el requisito es mirar los recursos del dispositivo seleccionado") |
| NPU | Protocolo listo (`ResourceProbe`), implementación `NullProbe` sin lógica | Sin hardware NPU para probar hoy; pedido explícito del usuario de dejarlo preparado para cuando alguien lo tenga |
| Presión externa (fuera de Upflow) | Re-chequeo periódico del predicado, además de `notify_all()` interno | `notify_all()` solo se dispara cuando un job PROPIO libera un permit; VRAM/RAM liberada por un proceso externo no dispara nada — sin poll, un waiter gateado por recursos quedaría esperando para siempre |

## Componentes

### 1. `ResourceProbe` (protocolo) — `app/services/resource_probes.py`

```python
class ResourceProbe(Protocol):
    def free_capacity_mb(self, device_id: str) -> int | None: ...
    # None = "no puedo medir esto" (NPU hoy, plataforma no soportada, fallo de API nativa) -> fail-open, no gatea
```

Implementaciones en el mismo módulo:

- **`DxgiVramProbe`**: extiende los bindings ctypes DXGI que ya existen en `devices_service.py` con `IDXGIAdapter3` (`QueryVideoMemoryInfo`, distinta interfaz COM de la que ya usan `GetDesc1`/`IDXGIAdapter1` para nombre/memoria total). Devuelve `Budget - CurrentUsage` del `DXGI_MEMORY_SEGMENT_GROUP_LOCAL` del adapter mapeado a `device_id` (`dml:N`). Nunca lanza — cualquier fallo COM (`QueryInterface` no soportado, adapter no encontrado) devuelve `None`, mismo patrón que `_enumerate_gpu_adapter_names()` ya sigue hoy.
- **`SystemRamProbe`**: `psutil.virtual_memory().available // (1024 * 1024)`. Nunca lanza — `ImportError` de `psutil` ausente devuelve `None` (lazy import, tolerante, mismo patrón que `_probe_onnxruntime()`).
- **`NullProbe`**: siempre `None`. Usado para `npu` y como fallback en plataformas no-Windows para `gpu`.

Un registro `dict[str, ResourceProbe]` keyed por `device kind` (`"gpu"`, `"cpu"`, `"npu"`) se construye en `main.py` (mismo lugar donde se instancian `DevicesService`/`GpuSessionCoordinator` hoy) y se inyecta en `DeviceSemaphores`.

### 2. `devices_service.py` — extensión

Nuevos bindings ctypes para `IDXGIAdapter3::QueryVideoMemoryInfo` (GUID, vtable slots, `DXGI_QUERY_VIDEO_MEMORY_INFO` struct), siguiendo el mismo estilo (`_com_vtable`, `_release_com`, funciones libres nunca-lanzan) que ya usa el código de enumeración de adapters. No se modifica `list_devices()`/`DeviceInfo` — el probe consulta el adapter directamente por índice derivado de `device_id` (`dml:N` → adapter N), reusando `_enumerate_dxgi_adapter_names`'s índice.

### 3. `device_semaphores.py` — extensión

- Constructor recibe `resource_probes: dict[str, ResourceProbe] | None = None` (default `None` = comportamiento idéntico al actual, sin chequeo de recursos — no rompe el semáforo en tests/entornos que no lo inyecten).
- `_reserve_if_free(device_id)` gana una segunda condición: mapea `device_id` → `kind` con un prefijo liviano ya implícito en la convención de ids existente (`dml:*` → `gpu`, `cpu` exacto → `cpu`, cualquier otro prefijo futuro → `npu`/lo que corresponda) — sin depender de `DevicesService` para no acoplar la primitiva de concurrencia a un servicio externo. Si hay un probe registrado para ese `kind`, exige `probe.free_capacity_mb(device_id) >= MIN_FREE_MB[kind]`. Si el probe devuelve `None` (o no hay probe registrado para ese `kind`), esa condición se salta (fail-open).
- El loop de espera en `acquire()` cambia de `await condition.wait_for(pred)` a un loop con `asyncio.wait_for(condition.wait(), timeout=RESOURCE_POLL_INTERVAL_SECONDS)` + catch de `TimeoutError` + re-evaluación del predicado — el timeout SOLO importa cuando el predicado depende de un probe de recursos (si no hay probes inyectados, el comportamiento es idéntico al `wait_for` puro de hoy, sin overhead de polling).

### 4. `config.py`

- `MIN_FREE_VRAM_MB` (default: 768 — headroom conservador por debajo del pipeline SD1.5 más chico ya medido en el módulo de generación, ~4500MB; validar empíricamente en el plan).
- `MIN_FREE_RAM_MB` (default: 1024).
- `RESOURCE_POLL_INTERVAL_SECONDS` (default: 5).

### 5. `pyproject.toml`

- `psutil` — nueva dependencia (el proyecto declara deps en `pyproject.toml`, no `requirements.txt`), pin de versión a resolver en el plan.

## Manejo de errores

Todo probe es fail-open y nunca lanza, mismo contrato que `_enumerate_gpu_adapter_names()` ya establece hoy ("Never raises: returns [] on any failure"). Si `QueryVideoMemoryInfo` falla (driver viejo, adapter LUID inválido, `IDXGIAdapter3` no soportado en hardware pre-Windows 10 1511) o `psutil` no está instalado, la admisión se comporta exactamente como hoy: solo gatea por conteo de jobs. Un job nunca se cuelga ni falla por un probe roto — en el peor caso, pierde la protección extra de esta feature, nunca gana un fallo nuevo.

## Testing

- **`resource_probes.py`**: fakes por protocolo (probe con valor fijo, probe que devuelve `None`, probe con secuencia de valores para simular liberación externa entre llamadas). `DxgiVramProbe`/`SystemRamProbe` reales solo ejercitados donde hay hardware real (mismo patrón de smoke manual que SP7/SP1-T8 ya usan para DXGI).
- **`device_semaphores.py`**: extender los tests de paralelismo por timestamps de SP7 con: (a) 2 jobs caben por conteo pero NO por VRAM → uno espera; (b) probe pasa de "sin capacidad" a "con capacidad" entre polls → el waiter se libera sin necesitar un `release()` propio; (c) probe ausente (`None` inyectado) → comportamiento bit-idéntico al `DeviceSemaphores` actual (regresión cero).
- **Config**: validación de defaults, igual patrón que `test_config_auth.py` (subproyecto C) para los nuevos campos.
- **Smoke real (manual, no CI)**: correr un juego/carga GPU externa mientras se encola un job GPU-pinned, confirmar que el job espera hasta que se libera VRAM externa (o hasta que termina el juego).

## Riesgos aceptados

| Riesgo | Mitigación |
|---|---|
| `IDXGIAdapter3::QueryVideoMemoryInfo` no soportado en hardware/driver viejo | Fail-open: probe devuelve `None`, degrada a solo-conteo (comportamiento actual) |
| `psutil` como dependencia nueva — posible fricción de instalación/plataforma | Lazy import tolerante (mismo patrón que `onnxruntime` en `devices_service.py`), fail-open si falta |
| `RESOURCE_POLL_INTERVAL_SECONDS` mal calibrado: muy corto desperdicia wakeups, muy largo demora la reacción a que se libere presión externa | Configurable, default conservador a validar en smoke; no afecta el camino feliz sin contención (el polling solo corre cuando hay waiters gateados por recursos) |
| `MIN_FREE_MB` mal calibrado: muy conservador desperdicia capacidad real, muy agresivo no previene OOM | Sin dato preciso por-modelo en el MVP (decisión deliberada — ver "Fases futuras"), requiere ajuste empírico post-smoke con hardware real, documentado como riesgo abierto igual que el ajuste de VRAM budget en GMFSS (SP14) |
| Mapeo `device_id` → adapter index (`dml:N` → adapter N) puede desalinearse si DXGI reordena adapters entre boots | Mismo riesgo que ya existe hoy en `devices_service.py` para nombre/kind (ya vive con él); no es nuevo de este subproyecto |

## Fases futuras (fuera de este spec)

1. **Estimación por-modelo afinada** — cuando subproyecto A (installer de modelos HF arbitrarios) exista, su validación por forward-pass puede medir VRAM pico real y guardarla en `ModelRegistry`; la admisión usaría ese dato como override del headroom genérico cuando esté disponible, cayendo al umbral genérico si no.
2. **Probes reales no-Windows** — Linux/ROCm nativo, CUDA nativo, cuando el proyecto soporte esas plataformas.
3. **`NpuProbe` real** — cuando haya hardware NPU disponible para validar contra una API real (WinML, DirectML NPU EP, o lo que exista para entonces).
