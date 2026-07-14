# Upflow v2 — UX, modelos Hugging Face, dispositivos e instalador

Fecha: 2026-07-14. Estado: aprobado por el usuario (diseño verbal; torch incluido siempre; alcance sin módulo tiempo real ejecutable).

## Objetivo

Cuatro subproyectos, en orden:

1. **SP1 — Motor de modelos + dispositivos (backend)**: modelos instalables desde Hugging Face con buscador, ejecución en cualquier GPU (y NPU/CPU cuando existan) con selección de dispositivo.
2. **SP2 — UI React**: nueva interfaz modular (Vite + React + TS + Tailwind) que reemplaza la Jinja actual.
3. **SP3 — Instalador Inno Setup**: setup.exe amigable sin admin con Python embebido.
4. **SP4 — Módulo tiempo real**: SOLO documento de requisitos/arquitectura (Fase 7 futura); base técnica: extender Magpie (GPL-3, proceso separado) con efectos ONNX; sin frame generation v1.

Fundamento técnico: informe `.superpowers/sdd/research-v2-hf-realtime.md` (verificado en fuentes primarias, 2026-07-14).

## SP1 — Motor de modelos + dispositivos

### Decisiones de runtime
- **Base garantizada**: `onnxruntime-directml` — inferencia ONNX en cualquier GPU DX12 (AMD/NVIDIA/Intel). DirectML está en mantenimiento pero estable y universal.
- **Mejora opcional**: detección de Windows ML (ONNX Runtime del sistema con EPs de fabricante: MIGraphX/VitisAI/OpenVINO/QNN) vía `ort.get_ep_devices()` cuando el paquete/OS lo soporte; fallback transparente a DirectML.
- **Loader de modelos arbitrarios**: Spandrel (MIT) reconoce arquitecturas SR de la comunidad (.pth/.safetensors: ESRGAN, SwinIR, HAT, DAT, SPAN, etc.) → export único a ONNX al instalar (`torch.onnx.export`, torch CPU incluido SIEMPRE como dependencia) → inferencia siempre por onnxruntime.
- Los motores ncnn actuales (Real-ESRGAN, RIFE) **no cambian**: quedan como modelos "integrados/verificados". RIFE sigue ncnn.
- **Descartados** (investigación): torch-directml (congelado), OpenVINO como base (solo Intel), ONNX-Vulkan EP (no existe).

### Componentes (app/services/)
- `model_registry.py`: manifiesto local `{config_dir}/models/registry.json` + carpetas por modelo. Entrada: id, nombre, tipo (`builtin-ncnn` | `onnx`), origen (HF repo id / archivo local), escala(s), arquitectura detectada, tamaño, estado (installed/converting/error), hash. Los builtin aparecen en el registry como no-eliminables.
- `hf_client.py`: búsqueda (REST `https://huggingface.co/api/models?search=...` con filtros de tarea/tags), metadatos de repo (lista de archivos), descarga con progreso (streaming + eventos), sin token para repos públicos (settings `HF_TOKEN` opcional para privados/gated).
- `model_installer.py`: valida y prepara un modelo — `.onnx`: verificación de IO (1 input imagen NCHW, 1 output escalado; inferir escala con un tensor de prueba pequeño en CPU). `.pth/.safetensors`: carga Spandrel → metadata de arquitectura/escala → export ONNX (fp32; fp16 en inferencia vía DirectML) → valida el ONNX resultante. Errores claros por incompatibilidad. Conversión en background (cola propia de 1 worker CPU — no toca el semáforo GPU).
- `engines/onnx_upscaler.py`: motor de upscaling ONNX — sesión por (modelo, dispositivo) con caché, **tiling** configurable (default auto por VRAM, solape de bordes para evitar seams), fp16 IO donde el modelo lo permita, batch=1. Misma interfaz `run(job)` que RealEsrganNcnnEngine para encajar en JobManager/VideoUpscaler sin reescribir pipeline (el pipeline de video usa el motor por-frame igual que hoy).
- `devices_service.py`: enumeración de dispositivos — DirectML adapters (índice + nombre vía onnxruntime/DXGI), CPU siempre, NPU solo si un EP la expone. Expone lista estable con ids (`cpu`, `dml:0`, `dml:1`, `npu:0`). Default global en settings (`DEFAULT_DEVICE`), override por job (`device` en el form). Los motores ncnn mapean `dml:N` → `-g N` (mismo orden de adapters; documentar la suposición y permitir override manual).

### API nueva
- `GET /api/v1/devices` — lista de dispositivos.
- `GET /api/v1/models` — instalados (builtin + HF) con estado.
- `GET /api/v1/models/search?q=...` — proxy de búsqueda HF filtrada (tags image-to-image/super-resolution + heurística de archivos compatibles).
- `POST /api/v1/models/install` — `{repo_id}` o upload de archivo; job de instalación con progreso consultable (`GET /api/v1/models/install/{id}`).
- `DELETE /api/v1/models/{id}` — solo no-builtin.
- Jobs de imagen/video aceptan `model_id` (reemplaza gradualmente `model_name`, compat retro) y `device`.

### Seguridad
- Descargas: solo HTTPS huggingface.co; tamaño máximo configurable; hash registrado. `.pth` implica pickle → cargar SOLO vía Spandrel con `weights_only=True` cuando torch lo permita y documentar el riesgo residual; preferir `.safetensors` cuando el repo lo ofrezca (elegirlo automáticamente).
- Validación de nombres/paths como ya hace el repo (sanitización existente).

## SP2 — UI React

- **Stack**: Vite + React 18 + TypeScript + Tailwind; TanStack Query para estado servidor; sin librería de componentes pesada (sistema propio guiado por la skill ui-ux-pro-max: dirección visual definida en la primera tarea de UI — no template genérico).
- **Estructura** `frontend/`: `src/modules/enhance` (imagen+video: upload, perfiles, FPS boost, target fps, audio; cola y estado de jobs con polling), `src/modules/models` (buscador HF con install 1-click y progreso, instalados, selector de dispositivo default), `src/modules/realtime` (placeholder "próximamente" — tarjeta con la visión y link al roadmap), `src/modules/settings` (dispositivo, TTL, timeouts, rutas, toggles de features), `src/components` compartidos, `src/lib/api.ts` cliente tipado.
- **Servida por FastAPI**: build a `frontend/dist`, montada como SPA fallback (mismo patrón bipolar-code). Dev: proxy Vite → :8090.
- **Migración**: la UI Jinja vive hasta paridad; al final del SP2 se elimina `app/templates`/`app/web` y los tests de template se reemplazan por tests de API + build check del frontend (vitest para lógica de UI crítica; sin Playwright en esta iteración).
- **Accesibilidad**: navegación teclado, focus states, contraste AA (checklist ui-ux-pro-max).

## SP3 — Instalador (Inno Setup)

- `installer/upflow.iss`: instala en `{localappdata}\Upflow` (sin admin), incluye **Python embeddable** (3.12 amd64) + la app + wheels preinstalados en un venv creado post-install (o site-packages del embeddable con `pip` bootstrapped — decidir en implementación lo más robusto), accesos directos (escritorio + inicio), desinstalador que preserva `{localappdata}\Upflow\runtime` y modelos con opción de borrarlos.
- Primer arranque: launcher existente (descarga binarios vendored ~1GB + modelos builtin) — barra de progreso en consola amigable ya existente.
- `scripts/package-release.ps1` gana modo `-Installer`: compila con ISCC si está instalado (winget `JRSoftware.InnoSetup`), produce `dist/upflow-setup-v<version>.exe`. El zip actual se mantiene como alternativa portable.
- torch CPU incluido → tamaño estimado setup ~350-450MB (sin binarios vendored). Aceptado por el usuario.

## SP4 — Tiempo real (solo documento)

- `docs/REALTIME_MODULE.md`: requisitos y arquitectura — captura Windows.Graphics.Capture, present overlay, base Magpie fork (GPL-3 → binario separado invocado por Upflow, sin linking; UI de Upflow como front de configuración/launcher), efectos ONNX del branch onnx-preview2, modelos candidatos realtime (Anime4K shaders, SPAN/RTMoSR/compact-ESRGAN ONNX), sin frame generation v1 (no existe camino open-source Windows; re-evaluar lsfg-vk/Windows y FidelityFX en 6-12 meses), presupuesto de latencia y hardware mínimo. Fase 7 del IMPLEMENTATION_PLAN.

## Criterios de aceptación globales

- SP1: buscar "ESRGAN" en la UI/API devuelve resultados HF; instalar un `.onnx` y un `.pth` comunitario funciona end-to-end (job de imagen con el modelo nuevo en `dml:0` y en `cpu`); dispositivos enumerados correctamente; modelos builtin intactos.
- SP2: paridad funcional completa con la UI actual + módulos nuevos; build integrado al release; Jinja eliminada.
- SP3: setup.exe instala y la app abre en navegador con doble click del acceso directo, sin Python del sistema.
- Suite pytest verde en todo momento; smoke real por subproyecto.
- Commits español convencional; sin Co-Authored-By.

## Orden y dependencias

SP1 → SP2 (la UI consume las APIs nuevas) → SP3 (empaqueta el resultado). SP4 en paralelo como docs. Cada SP en su propia rama `feature/...` mergeada a master al aprobar su review final.
