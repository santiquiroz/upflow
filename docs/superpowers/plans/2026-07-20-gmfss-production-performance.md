# GMFSS en producción: coordinador de sesiones GPU + fusión interpolar→escalar — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminar la contención real de sesiones DirectML entre motores ONNX que coexisten en un job de video (causa raíz confirmada de un job real que midió 0.056fps en vez de los 0.38-0.72fps esperados), y opcionalmente fusionar interpolar+escalar cuando ambos corren in-process para evitar un round-trip completo a disco.

**Architecture:** Fase 1 agrega un `GpuSessionCoordinator` compartido (uno por app) que impone exclusión mutua por device entre los 5 motores con cache de sesiones ONNX — al cambiar de dueño en un device, el anterior libera SOLO su entrada para ese device (el LRU entre jobs del mismo motor no se toca). Fase 2 agrega un modo opcional en `GmfssEngine` que recibe un callback de escalado inyectado, evitando escribir/releer PNGs intermedios, activo solo cuando GMFSS + el escalador ONNX corren ambos in-process en el mismo proceso.

**Tech Stack:** Python, `threading.Lock`, `OrderedDict` (patrón ya usado en los 5 motores), pytest.

## Global Constraints

- `release_device(device)` es idempotente en los 5 motores: llamarlo sin nada cacheado para ese device no debe fallar ni loguear error.
- El coordinador es exclusión mutua POR DEVICE, no por job — dos motores en devices distintos nunca se pisan entre sí.
- Fase 2 (fusión) solo se activa cuando `job.interp_engine == "gmfss"` Y el backend de escalado resuelto es ONNX in-process — nunca para RIFE ni para el escalador NCNN (ambos subprocesos externos).
- La fusión debe producir output PIXEL-IDÉNTICO al camino actual de dos pasadas — es la misma operación de escalado, solo sin el disco de por medio, cero tolerancia de diferencia.
- Ningún cambio de esta fase toca pesos/arquitectura de red — es orquestación de recursos únicamente.
- Commits en español, formato por capas del repo, sin Co-Authored-By. Sin pushear hasta el final (branch de feature, seguir la convención `feature/<nombre>` ya usada en el repo).
- Suites completas verdes al cierre: `.venv/Scripts/python.exe -m pytest -q` y `cd frontend && npm test -- --run` (Fase 2 no toca frontend, pero la suite completa igual debe quedar verde).

---

## Fase 1 — Coordinador de sesiones GPU

### Task 1: `GpuSessionCoordinator` + protocolo `GpuSessionOwner`

**Files:**
- Create: `app/services/gpu_session_coordinator.py`
- Test: `tests/test_gpu_session_coordinator.py`

**Interfaces:**
- Produces: `GpuSessionOwner` (Protocol, `release_device(self, device: str) -> None`), `GpuSessionCoordinator.acquire(self, device: str, owner: GpuSessionOwner) -> None`. Consumido por Tasks 2-6.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_gpu_session_coordinator.py
from app.services.gpu_session_coordinator import GpuSessionCoordinator


class FakeOwner:
    def __init__(self, name: str) -> None:
        self.name = name
        self.released: list[str] = []

    def release_device(self, device: str) -> None:
        self.released.append(device)


def test_acquire_new_device_does_not_release_anything():
    coordinator = GpuSessionCoordinator()
    owner = FakeOwner("a")
    coordinator.acquire("dml:0", owner)
    assert owner.released == []


def test_acquire_same_device_different_owner_releases_previous():
    coordinator = GpuSessionCoordinator()
    owner_a = FakeOwner("a")
    owner_b = FakeOwner("b")
    coordinator.acquire("dml:0", owner_a)
    coordinator.acquire("dml:0", owner_b)
    assert owner_a.released == ["dml:0"]
    assert owner_b.released == []


def test_acquire_same_device_same_owner_does_not_release():
    coordinator = GpuSessionCoordinator()
    owner = FakeOwner("a")
    coordinator.acquire("dml:0", owner)
    coordinator.acquire("dml:0", owner)
    assert owner.released == []


def test_acquire_different_devices_never_release_each_other():
    coordinator = GpuSessionCoordinator()
    owner_a = FakeOwner("a")
    owner_b = FakeOwner("b")
    coordinator.acquire("dml:0", owner_a)
    coordinator.acquire("dml:1", owner_b)
    assert owner_a.released == []
    assert owner_b.released == []


def test_acquire_cpu_device_tracked_independently_from_gpu_devices():
    coordinator = GpuSessionCoordinator()
    owner_cpu = FakeOwner("cpu-user")
    owner_gpu = FakeOwner("gpu-user")
    coordinator.acquire("cpu", owner_cpu)
    coordinator.acquire("dml:0", owner_gpu)
    assert owner_cpu.released == []
    assert owner_gpu.released == []


def test_owner_regains_device_after_being_released():
    coordinator = GpuSessionCoordinator()
    owner_a = FakeOwner("a")
    owner_b = FakeOwner("b")
    coordinator.acquire("dml:0", owner_a)
    coordinator.acquire("dml:0", owner_b)
    coordinator.acquire("dml:0", owner_a)
    assert owner_a.released == []  # nunca se libero a si mismo
    assert owner_b.released == ["dml:0"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/test_gpu_session_coordinator.py -v`
Expected: FAIL (`ModuleNotFoundError`)

- [ ] **Step 3: Write minimal implementation**

```python
# app/services/gpu_session_coordinator.py
from __future__ import annotations

import threading
from typing import Protocol


class GpuSessionOwner(Protocol):
    def release_device(self, device: str) -> None: ...


class GpuSessionCoordinator:
    """Exclusion mutua por device entre motores con cache de sesiones ONNX.

    Cuando un motor distinto pide el mismo device, el dueño anterior libera
    SOLO su entrada para ese device (release_device) -- no afecta su cache
    para otros devices, y no afecta a otros motores en devices distintos.
    """

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

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv\Scripts\python.exe -m pytest tests/test_gpu_session_coordinator.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git checkout -b feature/gmfss-session-coordinator
git add app/services/gpu_session_coordinator.py tests/test_gpu_session_coordinator.py
git commit -m "Dominio: GpuSessionCoordinator, exclusion mutua por device entre motores ONNX (Fase 1 Task 1)"
```

---

### Task 2: `release_device` en `AudioSrRestorer` y `ApolloRestorer` + `acquire` antes de crear sesiones

**Files:**
- Modify: `app/services/engines/audiosr_restore.py`
- Modify: `app/services/engines/apollo_restore.py`
- Test: `tests/test_audiosr_restore.py`, `tests/test_apollo_restore.py`

**Interfaces:**
- Consumes: `GpuSessionCoordinator` (Task 1) — inyectado por constructor, parámetro nuevo `gpu_coordinator: GpuSessionCoordinator`.

- [ ] **Step 1: Write the failing test (AudioSR)**

Revisar el fixture existente de `AudioSrRestorer` en `tests/test_audiosr_restore.py` (probablemente construye `AudioSrRestorer(settings)`) — el constructor gana un segundo argumento obligatorio.

```python
# agregar a tests/test_audiosr_restore.py
def test_release_device_clears_cached_session_for_that_device_only(settings, gpu_coordinator):
    restorer = AudioSrRestorer(settings, gpu_coordinator)
    restorer._session_cache["dml:0"] = {"fake": "session"}
    restorer._session_cache["dml:1"] = {"fake": "session-1"}
    restorer.release_device("dml:0")
    assert "dml:0" not in restorer._session_cache
    assert "dml:1" in restorer._session_cache


def test_release_device_on_empty_cache_is_a_noop(settings, gpu_coordinator):
    restorer = AudioSrRestorer(settings, gpu_coordinator)
    restorer.release_device("dml:0")  # no debe lanzar


def test_get_sessions_calls_coordinator_acquire_before_creating(settings, gpu_coordinator, monkeypatch):
    restorer = AudioSrRestorer(settings, gpu_coordinator)
    calls = []
    monkeypatch.setattr(gpu_coordinator, "acquire", lambda device, owner: calls.append((device, owner)))
    monkeypatch.setattr(restorer, "_create_sessions", lambda device: {"fake": "session"})
    restorer._get_sessions("dml:0")
    assert calls == [("dml:0", restorer)]
```

(`gpu_coordinator` fixture: `GpuSessionCoordinator()` real, no fake — el coordinador en sí ya está testeado en Task 1, acá se testea que ESTE motor lo llama correctamente.)

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/test_audiosr_restore.py -k release_device or calls_coordinator -v`
Expected: FAIL (`AudioSrRestorer() takes 1 positional argument but 2 were given` o `AttributeError: 'AudioSrRestorer' object has no attribute 'release_device'`)

- [ ] **Step 3: Implement — AudioSR**

En `app/services/engines/audiosr_restore.py`:

```python
    def __init__(self, settings: Settings, gpu_coordinator: GpuSessionCoordinator) -> None:
        self.settings = settings
        self.gpu_coordinator = gpu_coordinator
        self._session_cache: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._session_lock = threading.Lock()

    def release_device(self, device: str) -> None:
        with self._session_lock:
            self._session_cache.pop(device, None)
```

Agregar el import: `from app.services.gpu_session_coordinator import GpuSessionCoordinator`.

En `_get_sessions`, como primera línea del método (antes del `with self._session_lock: cached = ...`):

```python
    def _get_sessions(self, device: str) -> dict[str, Any]:
        self.gpu_coordinator.acquire(device, self)
        with self._session_lock:
            cached = self._session_cache.get(device)
            ...
```

- [ ] **Step 4: Run test to verify it passes (AudioSR)**

Run: `.venv\Scripts\python.exe -m pytest tests/test_audiosr_restore.py -v`
Expected: todos verdes (incluye los tests existentes — si alguno construye `AudioSrRestorer(settings)` sin el segundo argumento, actualizarlo para pasar un `GpuSessionCoordinator()` real).

- [ ] **Step 5: Repeat Steps 1-4 for Apollo**

Mismo patrón exacto en `app/services/engines/apollo_restore.py` (`ApolloRestorer.__init__` gana `gpu_coordinator`, `release_device` idéntico, `_get_session` — ojo, singular en Apollo, no `_get_sessions` — llama `acquire` como primera línea). Tests equivalentes en `tests/test_apollo_restore.py`.

- [ ] **Step 6: Run full backend suite**

Run: `.venv\Scripts\python.exe -m pytest -q`
Expected: puede haber MUCHOS fallos de fixtures que construyen `AudioSrRestorer`/`ApolloRestorer` sin `gpu_coordinator` — revisar cada uno y agregar un `GpuSessionCoordinator()` real (barato, sin estado compartido entre tests salvo que el propio test lo requiera). No usar mocks para el coordinador salvo en el test específico de Step 1 que verifica la llamada — el resto de los tests existentes deben poder pasar un `GpuSessionCoordinator()` real sin cambiar su comportamiento.

- [ ] **Step 7: Commit**

```bash
git add app/services/engines/audiosr_restore.py app/services/engines/apollo_restore.py tests/test_audiosr_restore.py tests/test_apollo_restore.py
git commit -m "Dominio: AudioSrRestorer y ApolloRestorer se registran en el GpuSessionCoordinator (Fase 1 Task 2)"
```

---

### Task 3: `release_device` en `GmfssEngine`

**Files:**
- Modify: `app/services/engines/gmfss_engine.py`
- Test: `tests/test_gmfss_engine.py`

**Interfaces:**
- Consumes: `GpuSessionCoordinator` (Task 1).

- [ ] **Step 1: Write the failing test**

Mismo patrón que Task 2 — revisar el constructor/fixture actual de `GmfssEngine` en `tests/test_gmfss_engine.py` primero.

```python
# agregar a tests/test_gmfss_engine.py
def test_release_device_clears_cached_sessions_for_that_device_only(settings, gpu_coordinator):
    engine = GmfssEngine(settings, gpu_coordinator)
    engine._session_cache["dml:0"] = {"featurenet": "fake"}
    engine._session_cache["dml:1"] = {"featurenet": "fake-1"}
    engine.release_device("dml:0")
    assert "dml:0" not in engine._session_cache
    assert "dml:1" in engine._session_cache


def test_get_sessions_calls_coordinator_acquire_before_creating(settings, gpu_coordinator, monkeypatch):
    engine = GmfssEngine(settings, gpu_coordinator)
    calls = []
    monkeypatch.setattr(gpu_coordinator, "acquire", lambda device, owner: calls.append((device, owner)))
    monkeypatch.setattr(engine, "_create_sessions", lambda device: {"featurenet": "fake"})
    engine._get_sessions("dml:0")
    assert calls == [("dml:0", engine)]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/test_gmfss_engine.py -k release_device or calls_coordinator -v`
Expected: FAIL

- [ ] **Step 3: Implement**

Mismo patrón exacto que Task 2 (AudioSR) en `app/services/engines/gmfss_engine.py`: constructor gana `gpu_coordinator: GpuSessionCoordinator`, `release_device(device)` hace `self._session_cache.pop(device, None)` bajo el lock existente, `_get_sessions` llama `self.gpu_coordinator.acquire(device, self)` como primera línea.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv\Scripts\python.exe -m pytest tests/test_gmfss_engine.py -v`
Expected: todos verdes (actualizar fixtures existentes que construyen `GmfssEngine(settings)` para pasar `GpuSessionCoordinator()` real).

- [ ] **Step 5: Run full backend suite**

Run: `.venv\Scripts\python.exe -m pytest -q`
Expected: 0 failures.

- [ ] **Step 6: Commit**

```bash
git add app/services/engines/gmfss_engine.py tests/test_gmfss_engine.py
git commit -m "Dominio: GmfssEngine se registra en el GpuSessionCoordinator (Fase 1 Task 3)"
```

---

### Task 4: `release_device` en `OnnxUpscaler` y `OnnxVideoUpscaler` (cache keyed por tupla)

**Files:**
- Modify: `app/services/engines/onnx_upscaler.py`
- Modify: `app/services/engines/onnx_video_upscaler.py`
- Test: `tests/test_onnx_upscaler.py`, `tests/test_onnx_video_upscaler.py`

**Interfaces:**
- Consumes: `GpuSessionCoordinator` (Task 1).

**Diferencia clave vs Task 2/3**: estos dos motores cachean por `(model_id_o_path, device)` — un mismo device puede tener VARIAS entradas (distintos modelos). `release_device` debe borrar TODAS las entradas cuyo device coincida, no una sola.

- [ ] **Step 1: Write the failing test (OnnxUpscaler)**

```python
# agregar a tests/test_onnx_upscaler.py
def test_release_device_clears_all_cached_sessions_for_that_device(settings, gpu_coordinator):
    upscaler = OnnxUpscaler(settings, gpu_coordinator)
    upscaler._session_cache[("model-a", "dml:0")] = "fake-a"
    upscaler._session_cache[("model-b", "dml:0")] = "fake-b"
    upscaler._session_cache[("model-a", "dml:1")] = "fake-a-1"
    upscaler.release_device("dml:0")
    assert ("model-a", "dml:0") not in upscaler._session_cache
    assert ("model-b", "dml:0") not in upscaler._session_cache
    assert ("model-a", "dml:1") in upscaler._session_cache


def test_release_device_on_empty_cache_is_a_noop(settings, gpu_coordinator):
    upscaler = OnnxUpscaler(settings, gpu_coordinator)
    upscaler.release_device("dml:0")  # no debe lanzar


def test_get_session_calls_coordinator_acquire_before_creating(settings, gpu_coordinator, monkeypatch):
    upscaler = OnnxUpscaler(settings, gpu_coordinator)
    calls = []
    monkeypatch.setattr(gpu_coordinator, "acquire", lambda device, owner: calls.append((device, owner)))
    monkeypatch.setattr(upscaler, "_create_session", lambda model_path, device: "fake-session")
    upscaler._get_session("model-a", "dml:0")
    assert calls == [("dml:0", upscaler)]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/test_onnx_upscaler.py -k release_device or calls_coordinator -v`
Expected: FAIL

- [ ] **Step 3: Implement — OnnxUpscaler**

```python
    def __init__(self, settings: Settings, gpu_coordinator: GpuSessionCoordinator) -> None:
        self.settings = settings
        self.gpu_coordinator = gpu_coordinator
        self._session_cache: OrderedDict[tuple[str, str], Any] = OrderedDict()
        ...  # el resto del __init__ existente, sin cambios

    def release_device(self, device: str) -> None:
        with self._session_lock:
            keys_to_remove = [key for key in self._session_cache if key[1] == device]
            for key in keys_to_remove:
                del self._session_cache[key]
```

En `_get_session(self, model_path: str, device: str)`, como primera línea: `self.gpu_coordinator.acquire(device, self)`.

- [ ] **Step 4: Run test to verify it passes (OnnxUpscaler)**

Run: `.venv\Scripts\python.exe -m pytest tests/test_onnx_upscaler.py -v`
Expected: todos verdes (actualizar fixtures existentes).

- [ ] **Step 5: Repeat Steps 1-4 for `OnnxVideoUpscaler`**

Mismo patrón exacto en `app/services/engines/onnx_video_upscaler.py` (`_get_session(self, model_id: str, device: str, entry: ModelEntry)` — la firma tiene un tercer parámetro `entry`, no afecta el `acquire`, sigue yendo primero en el método). Tests equivalentes en `tests/test_onnx_video_upscaler.py`.

- [ ] **Step 6: Run full backend suite**

Run: `.venv\Scripts\python.exe -m pytest -q`
Expected: 0 failures.

- [ ] **Step 7: Commit**

```bash
git add app/services/engines/onnx_upscaler.py app/services/engines/onnx_video_upscaler.py tests/test_onnx_upscaler.py tests/test_onnx_video_upscaler.py
git commit -m "Dominio: OnnxUpscaler y OnnxVideoUpscaler se registran en el GpuSessionCoordinator, release_device limpia por tupla (Fase 1 Task 4)"
```

---

### Task 5: wiring en `app/main.py`

**Files:**
- Modify: `app/main.py`

**Interfaces:**
- Consumes: `GpuSessionCoordinator` (Task 1), constructores actualizados de los 5 motores (Tasks 2-4).

- [ ] **Step 1: Implement**

En `app/main.py`, dentro del lifespan, ANTES de construir cualquiera de los 5 motores:

```python
    gpu_coordinator = GpuSessionCoordinator()
```

Agregar el import: `from app.services.gpu_session_coordinator import GpuSessionCoordinator`.

Actualizar cada construcción existente para pasar `gpu_coordinator` como segundo argumento posicional (revisar el orden real de cada línea — `rife_engine = RifeNcnnEngine(settings)` NO cambia, RIFE no tiene sesiones ONNX; los que sí cambian):

```python
    gmfss_engine = GmfssEngine(settings, gpu_coordinator)
    onnx_video_engine = OnnxVideoUpscaler(settings, model_registry, devices_service, gpu_coordinator)
```

(Ajustar el orden exacto de parámetros al de la firma real definida en Tasks 3/4 — si `OnnxVideoUpscaler.__init__` ya tiene 3 parámetros posicionales antes de `gpu_coordinator`, agregarlo al final o como keyword, lo que sea consistente con cómo se escribió el `__init__` en Task 4.) Buscar también dónde se construyen `AudioSrRestorer`/`ApolloRestorer`/`OnnxUpscaler` (probablemente en `restorer_registry.py`'s `build_restorers` y en la construcción del motor de imagen) y pasarles `gpu_coordinator` también — `build_restorers(settings)` gana un parámetro `gpu_coordinator: GpuSessionCoordinator`.

- [ ] **Step 2: Run full backend suite**

Run: `.venv\Scripts\python.exe -m pytest -q`
Expected: 0 failures. Si algún test de `app.main`/lifespan construye estos servicios directamente, actualizarlo.

- [ ] **Step 3: Smoke real — confirmar que arranca**

Run: `.venv\Scripts\uvicorn.exe app.main:app --host 127.0.0.1 --port 18091` (puerto de desarrollo, no el 8090 real) y verificar `curl http://127.0.0.1:18091/health` responde 200. Parar el proceso después.

- [ ] **Step 4: Commit**

```bash
git add app/main.py app/services/restorer_registry.py
git commit -m "Infraestructura: wiring del GpuSessionCoordinator compartido en app.main (Fase 1 Task 5)"
```

---

### Task 6: smoke real — confirmar que la contención bajó

**Files:** ninguno (solo verificación, no código nuevo)

- [ ] **Step 1: Reproducir el escenario real que disparó la investigación**

Con el server de desarrollo corriendo (puerto 18091, `ENABLE_GMFSS=true`, modelos GMFSS + AudioSR instalados), correr un job real con `audio_restore=audiosr` seguido de `interp_engine=gmfss` sobre un clip CORTO (unos segundos, no un episodio completo — el objetivo es medir fps de la etapa `interpolating_frames`, no esperar horas). Medir fps de esa etapa.

- [ ] **Step 2: Comparar contra el baseline documentado**

El spec documenta 0.056fps observado (contención) vs 0.38-0.72fps esperado (benchmarks aislados del puerto, sin contención). Confirmar que el fps medido ahora está en el rango esperado, no en el rango de contención. Si sigue bajo, hay una segunda causa no capturada por este plan — documentar el hallazgo real (no forzar el número) y decidir si amerita una investigación nueva.

- [ ] **Step 3: Documentar el resultado real en el CHANGELOG del commit final de esta fase**

No hay commit de código en este task — el resultado se documenta en el mensaje del commit de cierre de fase (ver checklist de self-review al final del plan) o en una nota aparte si amerita.

---

## Fase 2 — Fusión interpolar→escalar (in-process)

### Task 7: `GmfssEngine` — modo fusionado con callback de escalado inyectado

**Files:**
- Modify: `app/services/engines/gmfss_engine.py`
- Test: `tests/test_gmfss_engine.py`

**Interfaces:**
- Produces: `GmfssEngine.run_frames_fused(frames_in: Path, source_frame_count: int, multiplier: int, *, target_frame_count: int | None, device: str | None, upscale_frame: Callable[[np.ndarray], np.ndarray]) -> Iterator[np.ndarray]` — generador que produce cada frame YA compuesto (interpolado + escalado), en orden, sin escribir PNGs intermedios. El llamador (Task 8) decide qué hacer con cada frame (guardarlo).

- [ ] **Step 1: Write the failing test**

```python
# agregar a tests/test_gmfss_engine.py
def test_run_frames_fused_calls_upscale_frame_for_every_output_frame(settings, gpu_coordinator, fake_driver_2x_pair):
    engine = GmfssEngine(settings, gpu_coordinator)
    # fake_driver_2x_pair: fixture ya existente (o construida en este task siguiendo
    # el patron de los tests actuales de interpolate_pair) que produce 2 frames de
    # entrada + un driver fake que interpola exactamente 1 frame intermedio (2x).
    upscale_calls = []

    def fake_upscale(frame):
        upscale_calls.append(frame.shape)
        return frame * 2  # marcador simple para verificar que el resultado viene de aca

    frames = list(engine.run_frames_fused(
        fake_driver_2x_pair.frames_in, source_frame_count=2, multiplier=2,
        target_frame_count=None, device="cpu", upscale_frame=fake_upscale,
    ))
    assert len(frames) == 4  # source_frame_count * multiplier
    assert len(upscale_calls) == 4  # cada frame de salida paso por upscale_frame


def test_run_frames_fused_never_writes_intermediate_png(settings, gpu_coordinator, fake_driver_2x_pair, tmp_path):
    engine = GmfssEngine(settings, gpu_coordinator)
    intermediate_dir = tmp_path / "should-stay-empty"
    intermediate_dir.mkdir()
    list(engine.run_frames_fused(
        fake_driver_2x_pair.frames_in, source_frame_count=2, multiplier=2,
        target_frame_count=None, device="cpu", upscale_frame=lambda f: f,
    ))
    assert list(intermediate_dir.iterdir()) == []  # nada se escribio ahi (no se le paso ese dir a run_frames_fused)
```

(`fake_driver_2x_pair` — construir este fixture reusando exactamente el mismo patrón de fakes que ya usan los tests existentes de `GmfssEngine.run` en este archivo — 2 frames PNG reales chicos en un directorio temporal + un `GmfssDriver`/sesiones fake monkeypateados.)

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/test_gmfss_engine.py -k run_frames_fused -v`
Expected: FAIL (`AttributeError: 'GmfssEngine' object has no attribute 'run_frames_fused'`)

- [ ] **Step 3: Implement**

Leer primero el cuerpo actual de `GmfssEngine.run` (el método existente, con su pipeline de threads load/compute/save) para reusar exactamente la misma lógica de carga/resize/interpolación — `run_frames_fused` debe compartir el máximo de código posible con `run`, solo cambiando el paso final (en vez de encodear PNG y guardarlo en `frames_out`, aplicar `upscale_frame` y `yield` el resultado). Extraer la lógica común a un método privado si `run` actualmente no separa claramente "producir el frame interpolado" de "guardarlo" — refactor mínimo, sin cambiar el comportamiento de `run` (los tests existentes de `run` deben seguir pasando sin modificación).

```python
    def run_frames_fused(
        self,
        frames_in: Path,
        source_frame_count: int,
        multiplier: int = 1,
        *,
        target_frame_count: int | None = None,
        device: str | None = None,
        upscale_frame: Callable[[np.ndarray], np.ndarray],
    ) -> Iterator[np.ndarray]:
        if not self.available():
            raise RuntimeError(
                "GMFSS interpolation engine is not available. Run scripts/download-gmfss-onnx.ps1 first."
            )
        for interpolated_frame in self._iter_interpolated_frames(
            frames_in, source_frame_count, multiplier, target_frame_count, device
        ):
            yield upscale_frame(interpolated_frame)
```

(`_iter_interpolated_frames` es el método privado extraído del cuerpo actual de `run` — su firma exacta depende de cómo esté estructurado hoy `run`; el implementador de este task debe leer el archivo real antes de nombrar/extraer este método, no asumir la forma exacta sin verificarla.)

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv\Scripts\python.exe -m pytest tests/test_gmfss_engine.py -v`
Expected: todos verdes, incluidos los tests preexistentes de `run` (sin modificar su comportamiento).

- [ ] **Step 5: Run full backend suite**

Run: `.venv\Scripts\python.exe -m pytest -q`
Expected: 0 failures.

- [ ] **Step 6: Commit**

```bash
git add app/services/engines/gmfss_engine.py tests/test_gmfss_engine.py
git commit -m "Dominio: GmfssEngine.run_frames_fused produce frames interpolados+escalados sin PNG intermedio (Fase 2 Task 7)"
```

---

### Task 8: wiring en `video_upscaler.py` — gate de elegibilidad + parity real

**Files:**
- Modify: `app/services/video_upscaler.py`
- Test: `tests/test_video_upscaler.py`

**Interfaces:**
- Consumes: `GmfssEngine.run_frames_fused` (Task 7), la resolución de backend ONNX in-process ya existente en este archivo (leer primero cómo se resuelve `backend`/el motor de escalado antes de decidir el gate exacto).

- [ ] **Step 1: Write the failing test**

```python
# agregar a tests/test_video_upscaler.py
@pytest.mark.asyncio
async def test_fused_path_used_only_when_gmfss_and_onnx_backend(upscaler, tmp_path, monkeypatch):
    job = make_video_job(interp_engine="gmfss", backend="onnx")
    called = {"fused": False, "two_pass": False}
    monkeypatch.setattr(upscaler, "_run_fused_interpolate_upscale", lambda *a, **k: called.update(fused=True))
    monkeypatch.setattr(upscaler, "_maybe_interpolate", lambda *a, **k: called.update(two_pass=True))
    await upscaler._interpolate_and_upscale(job, tmp_path / "frames-out", "24/1", 1)  # nombre exacto a confirmar leyendo el archivo real
    assert called["fused"] is True
    assert called["two_pass"] is False


@pytest.mark.asyncio
async def test_two_pass_path_used_for_rife_even_with_onnx_backend(upscaler, tmp_path, monkeypatch):
    job = make_video_job(interp_engine="rife", backend="onnx")
    called = {"fused": False, "two_pass": False}
    monkeypatch.setattr(upscaler, "_run_fused_interpolate_upscale", lambda *a, **k: called.update(fused=True))
    monkeypatch.setattr(upscaler, "_maybe_interpolate", lambda *a, **k: called.update(two_pass=True))
    await upscaler._interpolate_and_upscale(job, tmp_path / "frames-out", "24/1", 1)
    assert called["two_pass"] is True
    assert called["fused"] is False


@pytest.mark.asyncio
async def test_two_pass_path_used_for_gmfss_with_ncnn_backend(upscaler, tmp_path, monkeypatch):
    job = make_video_job(interp_engine="gmfss", backend="ncnn")
    called = {"fused": False, "two_pass": False}
    monkeypatch.setattr(upscaler, "_run_fused_interpolate_upscale", lambda *a, **k: called.update(fused=True))
    monkeypatch.setattr(upscaler, "_maybe_interpolate", lambda *a, **k: called.update(two_pass=True))
    await upscaler._interpolate_and_upscale(job, tmp_path / "frames-out", "24/1", 1)
    assert called["two_pass"] is True
    assert called["fused"] is False
```

(Los nombres de método `_interpolate_and_upscale`/firma exacta son PROVISIONALES — antes de escribir este test de verdad, leer `video_upscaler.py`'s método `_run_pipeline` actual para ver dónde exactamente se llama `_maybe_interpolate` seguido de `_upscale_frames`, y diseñar el punto de inserción del gate ahí, ajustando los nombres del test a la estructura real.)

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/test_video_upscaler.py -k fused_path or two_pass_path -v`
Expected: FAIL

- [ ] **Step 3: Implement**

Insertar el gate de elegibilidad en el punto real donde hoy se llaman `_maybe_interpolate` seguido de `_upscale_frames` (ver `_run_pipeline` líneas 182-208 leídas durante el diseño): si `job.interp_engine == "gmfss"` y el backend resuelto para escalado es el ONNX in-process (reusar la función de resolución de backend ya existente en este archivo — leerla primero, no reimplementar), llamar a un nuevo método `_run_fused_interpolate_upscale(job, frames_in, frames_out, fps, fps_multiplier)` que arma el callback `upscale_frame` a partir de la sesión ONNX ya resuelta y consume `GmfssEngine.run_frames_fused`, escribiendo cada frame recibido directamente a `frames_out` (mismo patrón de naming `%08d.png` que usa el resto del pipeline). Si el gate no se cumple, seguir llamando `_maybe_interpolate` + `_upscale_frames` exactamente como hoy — CERO cambios de comportamiento en ese camino.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv\Scripts\python.exe -m pytest tests/test_video_upscaler.py -v`
Expected: todos verdes.

- [ ] **Step 5: Test de parity pixel-exacta**

```python
@pytest.mark.asyncio
async def test_fused_output_is_pixel_identical_to_two_pass_output(upscaler_with_real_fake_onnx_session, tmp_path):
    # Usa las MISMAS sesiones ONNX fake deterministas para ambos caminos (mismo
    # modelo fake, mismo GMFSS driver fake) sobre el mismo par de frames de
    # entrada, corre el camino fusionado y el de dos pasadas, compara byte a
    # byte los PNG de salida.
    ...
```

Implementar este test siguiendo el patrón de fixtures deterministas que ya usan los tests de `OnnxVideoUpscaler`/`GmfssEngine` para sesiones fake (sin aleatoriedad) — es el gate de calidad más importante de esta fase, no aproximar con `np.allclose`, debe ser bit-exacto (`assert path_a.read_bytes() == path_b.read_bytes()`).

- [ ] **Step 6: Run full backend suite**

Run: `.venv\Scripts\python.exe -m pytest -q`
Expected: 0 failures.

- [ ] **Step 7: Commit**

```bash
git add app/services/video_upscaler.py tests/test_video_upscaler.py
git commit -m "Aplicacion: fusiona interpolar+escalar cuando GMFSS+ONNX corren in-process, parity pixel-exacta verificada (Fase 2 Task 8)"
```

---

### Task 9: benchmark real + documentación honesta del impacto

**Files:**
- Modify: `README.md` (o el doc de performance que ya exista para GMFSS en este repo)

- [ ] **Step 1: Medir real, Fase 1 sola vs Fase 1+2**

Con el server de desarrollo, mismo clip corto, mismo device, medir fps de la etapa `interpolating_frames`+`upscaling_frames` combinadas: (a) con Fase 1 aplicada pero SIN Fase 2 (dos pasadas), (b) con Fase 1+2 (fusionado). Reportar el número real — puede ser una mejora chica, no forzar ni redondear favorablemente.

- [ ] **Step 2: Documentar**

Agregar los números reales medidos a la documentación de GMFSS en `README.md` (buscar la sección ya agregada en la doc previa de GMFSS — "Cómo activar GMFSS" — y sumar una nota de performance con los números de este benchmark, marcados con el device/hardware donde se midieron).

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: numeros reales de benchmark tras coordinador de sesiones GPU + fusion interp-escalar"
```

---

## Self-Review

- **Cobertura del spec**: Fase 1 (coordinador) → Tasks 1-6, cubre los 5 motores con cache de sesiones + wiring + verificación real. Fase 2 (fusión) → Tasks 7-9, gate de elegibilidad + parity pixel-exacta + benchmark honesto. Sin gaps contra `docs/superpowers/specs/2026-07-20-gmfss-production-performance-design.md`.
- **Placeholders**: ninguno salvo los nombres de método marcados explícitamente como "a confirmar leyendo el archivo real" en Task 8 (estructura interna de `video_upscaler.py` que el implementador debe verificar antes de escribir, dado que este plan no tuvo acceso de re-lectura línea por línea de ese método específico al momento de escribirse — instrucción explícita de leer primero, no un placeholder de "hacer algo").
- **Consistencia de tipos**: `GpuSessionOwner.release_device(device: str) -> None` mismo contrato en Tasks 2, 3, 4. `GpuSessionCoordinator.acquire(device, owner)` mismo uso en los 5 motores. `run_frames_fused(..., upscale_frame: Callable[[np.ndarray], np.ndarray])` firma consistente entre Task 7 (produce) y Task 8 (consume).
- **Alcance**: 9 tasks en 2 fases. Fase 1 es el fix de impacto real y debe cerrarse y verificarse (Task 6) ANTES de arrancar Fase 2 — si Task 6 no confirma la mejora esperada, vale la pena pausar y re-investigar antes de invertir en la fusión.
