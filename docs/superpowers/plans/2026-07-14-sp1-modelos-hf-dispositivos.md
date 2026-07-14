# SP1 — Motor de modelos HF + dispositivos — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task.

**Goal:** Modelos de upscaling instalables desde Hugging Face (buscador + catálogo builtin intacto) ejecutados vía ONNX Runtime DirectML en el dispositivo que el usuario elija (CPU/GPU-N/NPU si existe).

**Architecture:** Nuevo motor `OnnxUpscaler` conviviendo con los ncnn actuales; registry local de modelos; cliente HF (búsqueda/descarga); installer con validación y conversión Spandrel→ONNX; servicio de dispositivos DirectML; API REST nueva; jobs aceptan `model_id` + `device`.

**Tech Stack:** onnxruntime-directml, huggingface_hub (o REST directo), spandrel + torch CPU (siempre incluido), numpy, Pillow (ya presente).

**Spec:** docs/superpowers/specs/2026-07-14-upflow-v2-ux-design.md (sección SP1 — binding).

## Global Constraints

- TDD estricto; suite completa verde en cada commit (base actual: 340 tests).
- Sin binarios/modelos/red reales en unit tests — fakes; la validación real va en la tarea de smoke.
- Motores ncnn existentes intactos (Real-ESRGAN, RIFE): builtin, no-eliminables, siguen siendo default.
- Settings: patrón `Field(default, alias)` + propiedades `*_path` con `resolve_against_project_root`.
- Subprocesos no aplican al motor ONNX (in-process); operaciones pesadas SIEMPRE via `asyncio.to_thread` (no bloquear event loop) y gateadas por el semáforo GPU compartido cuando usan GPU.
- Descargas solo HTTPS huggingface.co; preferir `.safetensors` sobre `.pth` automáticamente; límite `MAX_MODEL_DOWNLOAD_MB` (default 2048).
- Commits español convencional; sin Co-Authored-By. Ramas: `feature/...`.

---

### Task 1: devices_service + API /devices + settings

**Files:** Create `app/services/devices_service.py`, `tests/test_devices.py`. Modify `app/config.py`, `app/api/routes.py`, `app/schemas.py`, `pyproject.toml` (dep `onnxruntime-directml`), `app/main.py` (DI).
**Produces:** `DevicesService.list_devices() -> list[DeviceInfo]` donde `DeviceInfo = {id: str, kind: "cpu"|"gpu"|"npu", name: str, backend: "cpu"|"directml"|"winml"}`; ids estables `cpu`, `dml:0`, `dml:1`, `npu:0`. `DevicesService.validate(device_id) -> DeviceInfo | ValueError`. Settings: `DEFAULT_DEVICE` (default `dml:0`, fallback automático a `cpu` si no hay GPU). Endpoint `GET /api/v1/devices`.
Enumeración: onnxruntime `get_available_providers()` + DirectML device enumeration (usar `onnxruntime.capi` adapter info si disponible; si la lib no expone nombres, enumerar vía `dxdiag`-free camino: ctypes DXGI o aceptar nombres genéricos "GPU 0" — decidir en implementación, documentar). Import de onnxruntime lazy + tolerante (app arranca sin la lib con solo `cpu`).

### Task 2: model_registry

**Files:** Create `app/services/model_registry.py`, `tests/test_model_registry.py`. Modify `app/config.py` (`MODELS_DIR` default `{runtime}/models`).
**Produces:** `ModelRegistry(settings)`: `list() -> list[ModelEntry]`, `get(model_id)`, `register(entry)`, `remove(model_id)` (ValueError si builtin), persistencia JSON atómica (`registry.json` + write-temp-rename). `ModelEntry` dataclass/pydantic: `id, name, kind("builtin-ncnn"|"onnx"), source(str), scale(int|None), arch(str|None), file_path(str|None), size_bytes, status("installed"|"converting"|"error"), error(str|None), created_at`. Seed automático de builtins desde el catálogo ncnn existente (ids = nombres actuales de MODEL_CATALOG).

### Task 3: hf_client

**Files:** Create `app/services/hf_client.py`, `tests/test_hf_client.py`.
**Produces:** `HfClient(settings)`: `async search(query, limit=20) -> list[HfModelSummary]` (REST `/api/models?search=&filter=&limit=` con tags de tarea image-to-image/super-resolution; httpx ya disponible vía fastapi/testclient — si no, agregar httpx dep), `async repo_files(repo_id) -> list[HfFile{path,size}]`, `async download(repo_id, filename, dest, progress_cb) -> Path` (streaming, límite MAX_MODEL_DOWNLOAD_MB, HTTPS only, token opcional `HF_TOKEN`). `pick_weight_file(files) -> HfFile` (prioridad: .onnx > .safetensors > .pth; el más grande si varios). Tests con transport mockeado (httpx MockTransport).

### Task 4: engines/onnx_upscaler

**Files:** Create `app/services/engines/onnx_upscaler.py`, `tests/test_onnx_upscaler.py`.
**Consumes:** `ModelEntry` (Task 2), device ids (Task 1).
**Produces:** `OnnxUpscaler(settings, registry, devices)`: `async run(job)` — misma semántica que RealEsrganNcnnEngine.run (lee job.source_path, escribe output, valida non-empty) pero además `job.model_id`/`job.device`; `available() -> bool` (onnxruntime importable). Internals: `_get_session(model_id, device)` con caché LRU(2) (sesiones ONNX pesan); providers: `cpu`→CPUExecutionProvider, `dml:N`→DmlExecutionProvider con device_id=N; inferencia en `asyncio.to_thread`; **tiling**: `ONNX_TILE_SIZE` (default 256, 0=sin tiles) con solape 16px y blending en seams; entrada RGB uint8→float32 0-1 NCHW; escala detectada del output shape (registrada en el entry). Errores claros (VRAM/desconocido → RuntimeError con mensaje). Unit tests con una sesión fake inyectable (interfaz `_create_session` monkeypatcheable) — un "modelo" numpy 2x que duplica tamaño sirve para validar tiling/blending sin onnxruntime real.

### Task 5: model_installer (.onnx directo) + API modelos

**Files:** Create `app/services/model_installer.py`, `tests/test_model_installer.py`, `tests/test_models_api.py`. Modify `app/api/routes.py`, `app/schemas.py`, `app/main.py`.
**Consumes:** HfClient (T3), ModelRegistry (T2), OnnxUpscaler validación (T4).
**Produces:** `ModelInstaller(settings, registry, hf_client)`: `async install_from_hf(repo_id) -> install_id` (job en cola propia asyncio 1-worker CPU; estados progress: downloading(pct)/validating/converting/installed/error), `status(install_id)`, `async delete(model_id)`. Validación `.onnx`: cargar sesión CPU, verificar 1 input 4D float, inferir escala con tensor 32x32, registrar. API: `GET /api/v1/models`, `GET /api/v1/models/search?q=`, `POST /api/v1/models/install {repo_id}`, `GET /api/v1/models/install/{install_id}`, `DELETE /api/v1/models/{model_id}` (409 si builtin, 404 si no existe). Lifespan: construir registry/hf/installer, start/stop del worker de instalación.

### Task 6: conversión .pth/.safetensors (Spandrel + torch)

**Files:** Create `app/services/model_converter.py`, `tests/test_model_converter.py`. Modify `pyproject.toml` (deps `spandrel`, `torch` CPU — index-url pypi normal, torch CPU wheel), `app/services/model_installer.py`.
**Produces:** `convert_to_onnx(weight_path, out_onnx, progress_cb) -> ConversionResult{arch, scale}`: Spandrel `ModelLoader().load_from_file` (safetensors preferido; pth con torch weights_only si la arch lo permite) → `torch.onnx.export` (dynamic axes H/W, opset 17, fp32) → validar con onnxruntime CPU. Installer usa converter cuando el peso elegido no es .onnx (estado converting). Corre en `asyncio.to_thread`. Tests: fake spandrel model (nn.Module 2x conv) end-to-end export+validate real con torch CPU (torch SÍ permitido en unit tests aquí — es dependencia dura; mantener el modelo de prueba diminuto <1s).

### Task 7: jobs con model_id + device (imagen y video)

**Files:** Modify `app/models.py`, `app/api/routes.py`, `app/services/job_manager.py`, `app/services/video_job_manager.py`, `app/services/video_upscaler.py`, `app/services/engines/realesrgan_ncnn.py`, `app/main.py`, `app/schemas.py`. Tests: `tests/test_jobs_model_device.py` + actualizar existentes.
**Produces:** Form params `model_id` (compat: `model_name` sigue aceptado, mapea a builtin id) y `device` (default settings.DEFAULT_DEVICE, validado vía DevicesService). Routing por kind: `builtin-ncnn` → motores actuales (mapear `dml:N`→`-g N`; `cpu` inválido para ncnn → 400 claro); `onnx` → OnnxUpscaler. Video pipeline: paso de upscale por-frame usa el motor según modelo (frames dir in/out — OnnxUpscaler gana método `run_frames(frames_in, frames_out, model_id, device)` con misma validación de conteo que RIFE). Respuestas exponen `modelId`/`device`. Semáforo GPU compartido cubre AMBOS motores cuando device != cpu.

### Task 8: smoke real + docs + review final

Descargar e instalar modelo real .onnx pequeño de HF + un .pth comunitario (ej. de openmodeldb/HF: 2x-compact); job imagen en `dml:0` y `cpu`; job video corto con modelo onnx; verificar builtin intacto y RIFE ok. Docs: README (módulo modelos, API, dispositivos), .env.example (DEFAULT_DEVICE, MODELS_DIR, HF_TOKEN, MAX_MODEL_DOWNLOAD_MB, ONNX_TILE_SIZE), launcher (deps pip nuevas se instalan con pip install -e . normal). Review final de rama + merge a master.
