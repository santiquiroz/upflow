# Módulo de generación de imágenes (difusión local, multi-vendor) — Design

**Fecha:** 2026-07-22
**Estado:** Approved (pendiente de plan de implementación)

## Motivación

Upflow ya cubre el post-procesado completo (upscale RealESRGAN/ONNX, interpolación RIFE/GMFSS, restauración de audio). El paso natural es generar el medio de origen localmente: texto→imagen primero, video después cuando exista un camino Windows viable. El patrón "generar barato → escalar/interpolar con el pipeline existente" es un workflow establecido de la comunidad ComfyUI (documentado en docs.comfy.org), no una idea especulativa.

Investigación de esta sesión (deep-research con verificación adversarial, 2026-07-22):

- La colección `amd/` en Hugging Face publica modelos de difusión ONNX reales y mantenidos (SD1.5, SDXL, SDXL Turbo, SD3.5) en formato `diffusers` estándar (`model_index.json` + carpetas por componente), cargables vía `optimum[onnxruntime]` (`ORTStableDiffusionPipeline`).
- DirectML es multi-vendor por diseño (AMD/NVIDIA/Intel — abstracción DirectX, no exclusiva de AMD): un solo backend cubre a los tres fabricantes de entrada. Backends específicos por marca (AMD GPU EP/ROCm, CUDA/TensorRT, OpenVINO/NPU) quedan como mejoras opt-in de fases futuras.
- **Video local queda fuera del MVP por imposibilidad técnica, no por precaución**: AMD Hummingbird (único candidato de video AMD-friendly) requiere PyTorch+ROCm 6.1+Flash Attention en Linux/MI300 — cero soporte ONNX/DirectML/Windows verificado. LTX-Desktop es CUDA-only local. Se re-evalúa cuando exista un export ONNX real.
- Licencias: SD1.5/SDXL 1.0 bajo CreativeML Open RAIL++-M (comercial, sin piso de revenue). Modelos Core nuevos de Stability: gratis bajo $1M USD/año de revenue.
- ONNX ≠ DirectML-portable automáticamente: existen builds SDXL ONNX explícitamente CUDA-only (ej. `tlwu/stable-diffusion-xl-base-1.0-onnxruntime`). El installer DEBE validar con un forward-pass real, no asumir por formato.

El `model_installer.py` actual es incompatible con esto por construcción: valida exactamente 1 input 4D float, descarga UN archivo de peso por repo, cap de 2GB, y su validación mide ratio de escala (semántica de upscaler). No se toca — se construye un installer paralelo.

## Alcance

**MVP (este spec):** texto→imagen, un solo backend (DirectML vía la infraestructura ONNX existente), installer generalizado para repos HF en formato diffusers, nuevo tipo de job con panel propio, y auto-upscale opcional reusando el upscaler existente.

**Explícitamente fuera del MVP:** video local (bloqueado por ecosistema), backends por marca (ROCm/CUDA/OpenVINO), img2img/inpainting, LoRA/checkpoints custom, encadenado con interpolación.

## Decisiones tomadas (brainstorming 2026-07-22)

| Decisión | Elección | Razón |
|---|---|---|
| Backends MVP | Solo DirectML | Ya cubre AMD/NVIDIA/Intel; evita abstracción prematura |
| Modalidad MVP | Solo imagen | Hummingbird verificado como Linux/ROCm-only; sin camino Windows |
| Modelo default | SD1.5 (`amd/` collection) | Más liviano en VRAM (GPU compartida con upscale/interpolate), el único benchmarkeado a fondo en DirectML/AMD (9-13x Olive confirmado) |
| Encadenado | Ambas opciones: job separado por defecto + flag `auto_upscale` | Pedido explícito del usuario: dar las dos |
| Mensajes de incompatibilidad | Obligatorios y específicos por hardware | Pedido explícito del usuario: NVIDIA/Intel/NPU deben recibir mensajes claros, no crashes |

## Componentes

### 1. `GenerationModelInstaller` (`app/services/generation_installer.py`)

Paralelo a `ModelInstaller`, no lo reemplaza ni lo modifica.

- **Descarga multi-archivo**: baja el repo HF completo (o subset necesario: `model_index.json` + carpetas de componentes) a un staging dir bajo `settings.temp_path`. Reusa `HfClient` existente (validación de `repo_id` idéntica: mismo `_validate_repo_id`, extraído a módulo compartido si hace falta).
- **Validación estructural**: existe `model_index.json` y las carpetas que declara. Sin eso → error temprano antes de descargar gigas.
- **Validación funcional**: forward-pass mínimo (prompt dummy, 1 step, resolución mínima) vía `ORTStableDiffusionPipeline` con los providers del device por defecto. Acá se detectan los modelos CUDA-only → mensaje claro (ver Manejo de errores).
- **Promoción atómica**: staging → `settings.models_path / "generation" / {model_id}/` (carpeta, no archivo). Mismo patrón tmp-then-replace del installer actual.
- **Cap de descarga propio**: `MAX_GENERATION_MODEL_DOWNLOAD_MB` (default 8192) — separado del cap de 2GB de upscalers.
- **Cola propia single-worker** (mismo patrón asyncio.Queue del installer actual): las descargas no compiten con jobs de GPU.

### 2. `ModelRegistry` extendido

- Nuevo `ModelKind.diffusion_onnx`. Para esta clase, `file_path` apunta a la CARPETA del pipeline (relativa a `models_path`), no a un archivo. `scale` queda `None`.
- `ModelInstaller.delete` y el borrado de archivos ganan una rama: si `kind == diffusion_onnx`, borrar recursivo de la carpeta (con el mismo guard de "resuelve dentro de models_path").
- El seed de builtins no cambia — no hay modelos de generación builtin; todos llegan por instalación HF.

### 3. `GenerationEngine` (`app/services/engines/generation_onnx.py`)

- Envuelve `ORTStableDiffusionPipeline` de `optimum.onnxruntime`. NO ensambla componentes a mano — optimum ya sabe leer `model_index.json`.
- Providers vía `_build_providers(device)` existente (`onnx_upscaler.py`) — cpu/dml:N, sin código de backend nuevo.
- Cache de pipeline LRU(1) keyed por `(model_id, device)` — mismo patrón de `GmfssEngine._session_cache`. Un pipeline SD1.5 completo pesa ~4GB en VRAM; nunca más de uno caliente.
- Implementa `release_device(device)` y llama `gpu_coordinator.acquire(device, self)` antes de construir el pipeline — se integra al `GpuSessionCoordinator` existente igual que los otros 5 motores (evita convivir con sesiones de AudioSR/GMFSS en el mismo device).
- Inferencia en `asyncio.to_thread` con el patrón shield+cancel_event de los otros motores. Callback de progreso por step del scheduler → `job.metadata["progress"]`.
- Parámetros expuestos: `prompt`, `negative_prompt`, `steps` (default 25, cap 100), `guidance` (default 7.5), `width`/`height` (default 512×512, cap 1024, múltiplos de 64), `seed` (opcional, para reproducibilidad).

### 4. `GenerationJobManager` (`app/services/generation_job_manager.py`) — 4to job kind

- Mismo esqueleto que `JobManager`/`AudioJobManager`: dataclass `GenerationJob` en `models.py` (con `created_at`/`started_at`/`finished_at` desde el día uno — la fila Duration ya existe en el frontend), cola asyncio con `max_queue_size`, workers, cancelación, TTL sweep igual que los demás.
- Request: JSON puro (prompt + params + `model_id` + `device` opcional + `auto_upscale` + params de upscale opcionales). No hay upload de archivo.
- `auto_upscale=true`: al terminar la generación, el MISMO worker llama directo a la función de upscale de imagen existente (la que usa `JobManager._process`, extraída a helper si hace falta) sobre el PNG generado, dentro del mismo job — no re-encola por HTTP, no crea un segundo job. El stage aparece en `job.metadata["stages"]` como etapa adicional (`generating` → `upscaling`), reusando `progress.py`.
- Output: PNG en `runtime/outputs`, `downloadUrl` estándar.

### 5. API (`app/api/routes.py` + `schemas.py`)

- `POST /api/v1/generation/jobs` — crear job (validación pydantic de params, límites arriba).
- `GET /api/v1/generation/jobs/{id}`, `POST .../cancel`, `GET .../download` — calcados de audio.
- `GET /api/v1/generation/capabilities` — lista modelos instalados (`kind=diffusion_onnx`), devices compatibles, y el flag `cpuOnly` para la advertencia proactiva del frontend.
- `POST /api/v1/generation/models` + `GET /api/v1/generation/models/install/{id}` — instalar/status desde HF (paralelo a los endpoints de modelos actuales).

### 6. Frontend

- Nueva ruta/página **Generate** (`frontend/src/modules/generate/GeneratePanel.tsx` + `useGenerationJob`): prompt, negative prompt, selector de modelo instalado, steps/guidance/size/seed, toggle "Escalar automáticamente al terminar" (con selector de modelo de upscale y escala cuando está activo), botón Generate.
- Reusa `JobCard`/`JobDetailModal`/`JobQueue` — el job de generación entra a la misma cola visual (nuevo `kind: "generation"` en `useJobQueue`).
- Instalación de modelos: sección en la página Models existente, filtrada por tipo — input de repo_id HF + progreso de descarga (mismo componente/patrón del installer actual).
- Preview de la imagen generada en el card completado (igual que `ImageCompletedDetails` ya hace).

## Manejo de errores (multi-vendor, mensajes específicos — requisito explícito)

Tres puntos de fallo por hardware, cada uno con mensaje accionable vía `job.error` → UI existente:

1. **Modelo incompatible con el hardware en la instalación** (ej. SDXL build CUDA-only en máquina AMD/Intel): el forward-pass de validación falla → el install job termina en `error` con: *"Este modelo requiere GPU NVIDIA (CUDA) y no es compatible con DirectML en tu hardware. Buscá una versión compatible (ej. colección `amd/` en Hugging Face, formato ONNX+DirectML)."* Detección: parsear el error de provider de onnxruntime (patrón `_wrap_onnx_error` existente, extendido con esta clase).
2. **VRAM insuficiente al generar**: error de allocation DML atrapado → *"Sin memoria de GPU para generar con esta configuración. Probá menor resolución, un modelo más liviano, o esperá a que terminen otros trabajos de GPU."*
3. **Sin GPU compatible (solo CPU)**: chequeo PROACTIVO en `GET /generation/capabilities` (vía `DevicesService` existente) → el frontend muestra advertencia ANTES de encolar: *"No se detectó GPU compatible (DirectX 12). Generar en CPU tarda varios minutos por imagen. ¿Continuar igual?"* — con confirmación explícita. Difusión en CPU es un orden de magnitud peor que upscale en CPU; advertir después de encolar sería tarde.

Además: si `optimum`/dependencias de generación no están instaladas (install viejo, extra opcional), `generation/capabilities` responde `available: false` con razón — el panel muestra el estado con instrucción de fix, mismo patrón que GMFSS ausente en `video/capabilities`.

## Dependencias nuevas

- `optimum[onnxruntime]` — pin de versión a resolver en el plan de implementación (verificar compatibilidad con onnxruntime-directml 1.24.x instalado). Riesgo conocido: optimum asume el paquete `onnxruntime` vanilla en algunos checks de import — validar contra `onnxruntime-directml` en la primera task del plan (spike). Si hay fricción, fallback documentado: ensamblar las 4-5 sesiones ONNX a mano (más código, cero dependencia nueva) — decisión diferida al spike.
- `transformers` (tokenizer CLIP) — llega transitivo con optimum.

## Testing

- **Installer**: fakes de `HfClient` + pipeline (patrón `test_model_installer.py`): repo válido multi-archivo, repo sin `model_index.json`, forward-pass que falla con error de provider CUDA (assert del mensaje amigable), cap de tamaño excedido, delete recursivo con guard de path.
- **Engine**: pipeline fake numpy inyectado (patrón `fake_sessions` de `test_gmfss_engine.py`); cache LRU por device; `release_device`; integración con `GpuSessionCoordinator` (acquire antes de construir).
- **JobManager**: cola/cancelación/TTL calcados de los tests de audio; rama `auto_upscale` con upscaler fakeado (assert: un solo job, dos stages).
- **API**: validación de params (steps cap, dimensiones múltiplo de 64), capabilities con y sin modelos instalados, `cpuOnly` flag.
- **Frontend**: `GeneratePanel.test.tsx` + `useGenerationJob.test.tsx` calcados de los de audio; advertencia CPU-only; toggle auto_upscale.
- **Smoke real (manual, no CI)**: instalar SD1.5 de `amd/` en la instalación local, generar 512×512 en dml:0, verificar imagen + duración reportada.

## Riesgos aceptados

| Riesgo | Mitigación |
|---|---|
| `optimum` + `onnxruntime-directml` fricción de imports | Spike primero (Task 1 del plan); fallback: ensamblado manual de sesiones |
| Sample oficial Microsoft Olive+SD removido de GitHub (mantenimiento incierto) | No dependemos de Olive: usamos los ONNX ya optimizados que publica `amd/` |
| VRAM: pipeline SD1.5 (~4GB) + upscaler conviviendo | `GpuSessionCoordinator` ya desaloja sesiones del device; mensaje de error 2 cubre el resto |
| Repos HF arbitrarios con estructura rara | Validación estructural + forward-pass; el installer rechaza con mensaje, nunca registra un modelo que no pasó validación |

## Fases futuras (fuera de este spec)

1. **Video local** — cuando Hummingbird (u otro) tenga export ONNX/Windows real; la arquitectura del installer/engine/job ya queda preparada (el installer es genérico por formato diffusers, no por modalidad).
2. **Backends por marca opt-in** — AMD GPU EP (ROCm/Windows, hoy technology preview con redistribución sin confirmar), CUDA/TensorRT, OpenVINO para NPU Intel. Entra como selección de provider en `_build_providers`, no como rediseño.
3. **Encadenado con interpolación** — generar secuencias → RIFE, cuando haya generación de video.
4. **img2img / inpainting / LoRA** — mismas sesiones, endpoints nuevos.
