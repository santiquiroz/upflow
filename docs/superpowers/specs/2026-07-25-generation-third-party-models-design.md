# Modelos de terceros para generación (SDXL/SD3.5/Turbo, conversión, repos gated, buscador) — subproyecto A — Design

**Fecha:** 2026-07-25
**Estado:** Approved (pendiente de plan de implementación)

## Motivación

El módulo de generación (`docs/superpowers/specs/2026-07-22-generation-module-design.md`, shippeado en `feature/generation-module`) solo instala pipelines diffusers ONNX con `_class_name: OnnxStableDiffusionPipeline` (SD1.5 legacy) — `generation_onnx.py::_load_pipeline_class()` está hardcodeada a `ORTStableDiffusionPipeline`. La colección `amd/` en Hugging Face publica también SDXL, SDXL Turbo y SD3.5, cada uno con su propia clase de pipeline; hoy esos repos no se pueden instalar. Además: repos que solo publican pesos PyTorch/`.safetensors` (sin ONNX pre-exportado) fallan duro con "no parece un pipeline diffusers ONNX"; repos gated (licencia a aceptar en huggingface.co + token) fallan con un error genérico de HTTP; y la búsqueda de modelos (`HfClient.search()`) solo indexa upscalers (`image-to-image`/`super-resolution`), no modelos de generación.

Motivación de fondo (explícita del usuario al arrancar este subproyecto): el objetivo no es optimizar para el hardware/colección de hoy, sino construir una plataforma que pueda absorber lo que Hugging Face publique a futuro sin rediseño — de ahí que el mapa de clases de pipeline y el mecanismo de settings editables se diseñen extensibles desde el día uno, no acotados al mínimo pedido.

## Alcance

**MVP (este spec):**
- Mapa extensible `_class_name` (declarado en `model_index.json`) → clase de pipeline `optimum.onnxruntime`, cubriendo las 4 variantes reales de la colección `amd/`: SD1.5 (ya existe), SDXL, SDXL Turbo, SD3.5.
- Nuevo job kind `conversion`: convierte un repo diffusers que solo publica pesos PyTorch/`.safetensors` a un pipeline ONNX completo vía `optimum.exporters.onnx`, como job separado y visible (progreso por componente), no el patrón inline-silencioso que ya usa el conversor Spandrel de upscalers.
- Auto-detección en `generation_installer.py`: un repo sin `model_index.json`+ONNX pero con layout diffusers reconocible rutea automáticamente a un job de conversión en vez de fallar.
- Mensajes accionables en `hf_client.py` para repos gated (401/403), y un mecanismo GENÉRICO de settings editables (whitelist + escritura atómica a `.env` + cache-clear) cuyo primer campo real es `HF_TOKEN`.
- Buscador de modelos de generación: extensión de `HfClient.search()` con un tag de búsqueda propio (`text-to-image`), conectado a la sección de instalación de generación en la página Models (hoy requiere pegar `repo_id` a mano).

**Explícitamente fuera del MVP:**
- img2img/inpainting/LoRA (ya excluido en el spec del módulo de generación original).
- Backends por marca (ROCm/CUDA nativo) — ya trackeado como fase futura del módulo de generación.
- Cualquier campo editable en el whitelist de settings más allá de `HF_TOKEN` — el mecanismo queda listo, la whitelist crece en subproyectos futuros sin rediseño.
- Gate de permisos real sobre quién puede editar settings — si subproyecto C (multi-usuario/auth) no está mergeado todavía, el endpoint de settings queda sin protección de permisos (equivalente a `AUTH_MODE=off` transparente), pero diseñado para enchufar `Permission.settings_write` (ya existe en la tabla de permisos de C) sin cambios de forma cuando C aterrice.
- Backends de conversión que no sean `optimum.exporters.onnx` (ej. ensamblado manual de sesiones) — mismo riesgo/fallback ya documentado para el spike de `optimum` del módulo original, se revisita solo si el spike de este subproyecto encuentra fricción bloqueante.

## Decisiones tomadas (brainstorming 2026-07-25)

| Decisión | Elección | Razón |
|---|---|---|
| Alcance del spec | Un spec combinado (las 4 piezas), no 4 specs separados | Pedido explícito del usuario; mismo patrón que subproyecto B — comparten los mismos archivos base |
| UX de la conversión PyTorch→ONNX | Job separado y visible (progreso por componente), no el patrón inline-silencioso de Spandrel | Pedido explícito — la conversión de difusión es órdenes de magnitud más cara (minutos, VRAM alta) que convertir un upscaler (segundos) |
| Repos gated | UI para pegar/gestionar el token en Settings, no solo mensaje+doc | Pedido explícito — más accesible que exigir editar `.env` a mano |
| Mecanismo de settings editables | Genérico y reusable desde el día uno, acotado solo por la whitelist inicial (`HF_TOKEN`) | Pedido explícito — evita rehacer el mecanismo la próxima vez que haga falta editar algo |
| Selección de clase de pipeline | Mapa extensible cubriendo SD1.5+SDXL+SDXL Turbo+SD3.5, no solo SDXL | Pedido explícito, coherente con el objetivo de plataforma-para-lo-que-venga planteado al arrancar la tarea completa (subproyectos A+B) |

## Componentes

### 1. `generation_onnx.py` — mapa extensible de clases de pipeline

```python
# Nombres de _class_name a VERIFICAR contra repos reales en el spike (Task 1
# del plan) -- igual que el spike de optimum del módulo original, estos
# strings vienen de inspeccionar model_index.json de repos reales, no de
# documentación (la colección amd/ no siempre usa el nombre "canónico" de
# diffusers para el _class_name declarado).
PIPELINE_CLASS_NAMES = {
    "OnnxStableDiffusionPipeline": "ORTStableDiffusionPipeline",  # SD1.5 legacy, ya existe
    "StableDiffusionXLPipeline": "ORTStableDiffusionXLPipeline",  # SDXL y SDXL Turbo -- Turbo es un checkpoint fine-tuneado de la MISMA arquitectura/clase (menos steps, scheduler distinto), no una clase de pipeline separada -- no necesita entrada propia en este mapa, solo defaults de inferencia distintos que se resuelven en generation_job_manager.py, no acá
    "StableDiffusion3Pipeline": "ORTStableDiffusion3Pipeline",    # SD3.5 -- transformer-based, distinto de UNet; validar en el spike si optimum-onnx expone esta clase o si SD3.5 necesita un camino aparte
}
```

`_load_pipeline_class(declared_class_name: str)` deja de recibir cero argumentos: lee el `_class_name` real de `model_index.json` (mismo patrón que `_read_declared_components` de `generation_installer.py` ya usa) y resuelve contra `PIPELINE_CLASS_NAMES`. Una clase declarada que no está en el mapa produce un error accionable listando las clases soportadas — nunca un `ImportError` crudo. El import de la clase optimum sigue siendo perezoso (dentro de la función), igual que hoy.

### 2. `app/services/generation_converter.py` (nuevo) — conversión PyTorch→ONNX

Paralelo a `model_converter.py`, no lo reemplaza ni lo reusa (ese es Spandrel: arquitectura de un solo conv-net, minutos→segundos; esto es un pipeline diffusers completo multi-componente, minutos reales, VRAM alta).

- Usa `optimum.exporters.onnx` (API Python, no subprocess — a confirmar en el spike si la API expone progreso por componente o si hay que invocar exportadores por componente por separado para tener ese detalle).
- Corre en su propia cola single-worker (mismo patrón que `GenerationModelInstaller`/`ModelInstaller`) — no compite por GPU con jobs de inferencia, pero sí puede competir por VRAM con ellos (ver Riesgos).
- Progreso por etapa: `unet` → `vae` → `text_encoder` (y `text_encoder_2`/`transformer` para SDXL/SD3.5 respectivamente) — cada componente exportado marca una etapa en `job.metadata["stages"]`, mismo mecanismo que `progress.py` ya usa para generación con auto-upscale.
- Al terminar, el resultado se trata como un staging dir diffusers normal: pasa por la MISMA validación funcional (`_validate_pipeline`, forward-pass real) y promoción atómica que ya existen en `generation_installer.py` — la conversión no introduce un segundo camino de instalación, solo un paso previo.

### 3. `ConversionJob` — nuevo job kind

Dataclass en `models.py`, mismo esqueleto que los otros 4 job kinds (`created_at`/`started_at`/`finished_at`/`status`/`error`/`metadata`). Endpoints:

- `POST /api/v1/generation/models/convert` — inicia conversión directa (uso explícito).
- `GET /api/v1/generation/models/convert/{id}` — status, calcado de `GET /api/v1/generation/models/install/{id}`.
- Auto-ruteo: `GenerationModelInstaller._download_and_register` gana una rama — si `_ensure_model_index_listed` falla pero el repo SÍ tiene un layout diffusers reconocible con pesos PyTorch (heurística a definir en el plan: presencia de `model_index.json` con `_class_name` conocido pero componentes en `.safetensors`/`.bin` en vez de carpetas ONNX), encola un `ConversionJob` en vez de lanzar el `ValueError` actual de "no parece un pipeline diffusers ONNX".

### 4. `hf_client.py` — mensajes accionables + búsqueda de generación

- `_is_retryable_download_error` ya excluye 401/403 de los reintentos (correcto, son errores permanentes) — falta el mensaje. Nueva función `_wrap_hf_auth_error(exc, repo_id) -> Exception` que envuelve un `httpx.HTTPStatusError` 401 con *"Tu HF_TOKEN no es válido o no está configurado — revisalo en Settings."* y un 403 con *"El repo {repo_id} requiere aceptar su licencia en huggingface.co/{repo_id} antes de poder descargarlo."* — aplicada en `repo_files()` y `download()`.
- `search()` gana un parámetro `task_tags: tuple[str, ...] | None = None` (default `None` = comportamiento actual, `SEARCH_TASK_TAGS`). Nueva constante `GENERATION_SEARCH_TASK_TAGS = ("text-to-image",)`. Nuevo endpoint `GET /api/v1/generation/models/search` (calcado de `GET /api/v1/models/search`, pasando `task_tags=GENERATION_SEARCH_TASK_TAGS`).

### 5. `app/services/settings_service.py` (nuevo) — settings editables genéricas

```python
EDITABLE_SETTINGS_WHITELIST = frozenset({"hf_token"})
```

- `update_setting(settings: Settings, key: str, value: str) -> None` — rechaza `key` fuera de la whitelist (400), valida `value` contra el tipo del campo pydantic correspondiente, escribe/actualiza `.env` de forma atómica. Extiende el patrón `_append_env_var`/`ensure_auth_secret` de `config.py` (que hoy solo hace *append-si-falta*) a un *update-si-existe-o-append* — necesita parsear las líneas existentes de `.env`, reemplazar la línea `KEY=...` si ya existe, y usar `write_text_atomically` para el resultado completo (nunca una escritura parcial).
- Tras escribir, llama `get_settings.cache_clear()` (mismo patrón ya documentado en `CLAUDE.md` para que un token nuevo tome efecto sin reiniciar el proceso).
- Endpoint `PATCH /api/v1/settings` con body `{"key": str, "value": str}`.
- Frontend: la página Settings (hoy 100% solo-lectura, confirmado en el código) gana un formulario para los campos de la whitelist — hoy solo `HF_TOKEN`, mostrado como campo tipo password/oculto.

## Manejo de errores

- Clase de pipeline no reconocida → error listando las clases soportadas por `PIPELINE_CLASS_NAMES`, mismo estilo que los mensajes de hardware-incompatible ya existentes en el módulo de generación (mensaje específico y accionable, nunca un stacktrace crudo).
- Conversión falla a mitad de un componente → mismo patrón fail-safe que la instalación normal: staging descartado por completo (`finally: shutil.rmtree`), nada queda medio-registrado en `ModelRegistry`.
- Token inválido/repo gated → mensajes accionables específicos por código (ver componente 4), nunca el `HTTPStatusError` crudo de httpx propagando hasta la UI.
- Escritura de setting inválida (campo fuera de whitelist, tipo incorrecto) → 400 con mensaje claro; la escritura a `.env` es atómica así que nunca deja el archivo a medio escribir aunque el proceso muera a mitad.

## Testing

- **Mapa de clases**: tabla de casos (`_class_name` conocido→clase correcta para cada una de las 4 variantes; `_class_name` desconocido→error con mensaje listando las soportadas).
- **Conversión**: fake de `optimum.exporters.onnx` (nunca bajar pesos reales en tests unitarios) verificando que el job pasa por sus etapas en orden, que un fallo a mitad de componente no deja archivos huérfanos, y que el resultado convertido pasa por la MISMA validación funcional que un install ONNX-nativo (no un camino paralelo sin cubrir).
- **Auto-ruteo no-ONNX→conversión**: test de integración de `generation_installer.py` con un repo fake que declara `model_index.json` pero sus componentes son `.safetensors`, confirmando que encola `ConversionJob` en vez de fallar.
- **HfClient**: extender los tests de retry existentes (ya cubren que 401/403 no reintentan) con casos que verifiquen el mensaje accionable envuelto; test de `search()` con `task_tags` override.
- **Settings**: whitelist rechaza campos no listados; escritura atómica no corrompe `.env` con escrituras concurrentes (mismo patrón ya probado para `json_store.py`/`config.py` en subproyecto C); `get_settings.cache_clear()` se llama tras cada escritura exitosa (test que un valor nuevo es visible sin recrear el proceso).
- **Smoke real (manual, no CI)**: instalar un SDXL real de `amd/` de punta a punta; convertir un repo diffusers PyTorch-only real y confirmar que el resultado genera una imagen; probar un repo gated real sin token (mensaje 403 esperado) y con token válido pegado desde Settings (instala OK).

## Riesgos aceptados

| Riesgo | Mitigación |
|---|---|
| `optimum.exporters.onnx` para pipelines completos es superficie mucho mayor que Spandrel (single conv-net) — riesgo real de fricción de versiones/dependencias | Spike dedicado como Task 1 del plan, mismo patrón que el spike de `optimum` del módulo de generación original (`docs/superpowers/specs/2026-07-22-optimum-spike-findings.md`) |
| SD3.5 es transformer-based, no solo UNet — el mapa de clases puede necesitar más que un simple lookup de nombre si sus componentes/parámetros de inferencia difieren sustancialmente | A validar en el spike; si SD3.5 no encaja en el mapa simple, se documenta como fase futura en vez de forzar una abstracción prematura |
| Conversión larga (minutos) + VRAM alta simultánea a otros jobs de GPU | Si subproyecto B (admisión por capacidad) ya está mergeado, la conversión hereda su protección automáticamente (mismo `DeviceSemaphores`). Si no, es una zona ciega de UX real — un job de conversión largo puede competir por VRAM sin aviso — a documentar explícitamente en el plan si B no aterrizó todavía |
| Endpoint de settings sin gate de permisos si subproyecto C no está mergeado | Aceptado para el MVP (mismo nivel de protección que el resto de la app hoy bajo `AUTH_MODE=off`); diseñado para enchufar `Permission.settings_write` sin cambios de forma |
| _class_name strings del mapa son suposiciones razonadas, no verificadas contra repos reales todavía | Explícito en el propio código (comentario) y en este spec — el spike del plan los verifica antes de construir el resto sobre ellos |

## Fases futuras (fuera de este spec)

1. **Backends por marca opt-in para conversión/inferencia** (ROCm/CUDA nativo) — ya trackeado como fase futura del módulo de generación original, aplica igual acá.
2. **Whitelist de settings ampliada** — más campos editables desde la UI a medida que haga falta, sin rediseño del mecanismo.
3. **Gate de permisos real** — enchufar `Permission.settings_write` de subproyecto C cuando esté mergeado.
4. **img2img/inpainting/LoRA** sobre los pipelines SDXL/SD3.5 que este subproyecto habilita — mismas sesiones, endpoints nuevos.
