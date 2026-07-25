# Spike findings: modelos de terceros (SDXL / SDXL Turbo / SD3.5) — `_class_name` reales, clases `optimum` y API de export

**Fecha:** 2026-07-25
**Pins verificados:** `optimum==2.1.0`, `optimum-onnx==0.1.0`, `onnxruntime-directml==1.24.4`, `diffusers==0.39.0`, `transformers==4.57.6` (venv del repo, `.venv\Scripts\python`, sin instalar nada nuevo).
**Script:** `scripts/spike_third_party_models.py` — `python scripts\spike_third_party_models.py` (steps 1-3) y `python scripts\spike_third_party_models.py --smoke-export` (step 4).

**Si este doc difiere de las hipótesis de `2026-07-25-generation-third-party-models-design.md`, gana este doc** (mismo patrón que el spike de `2026-07-22-optimum-spike-findings.md`).

## Veredicto ejecutivo por variante

| Variante | Veredicto | Resumen |
|---|---|---|
| **SDXL** | **GO** (directo, sin conversión) | `amd/stable-diffusion-xl-1.0_io16_amdgpu` es público, ya ONNX, y su `_class_name` declarado es literalmente `ORTStableDiffusionXLPipeline` — clase que existe y se importa OK desde `optimum.onnxruntime` en los pins del repo. |
| **SDXL Turbo** | **GO vía conversión** (NO-GO directo desde `amd/`) | Los 2 repos `amd/` candidatos para Turbo NO tienen `model_index.json` (404) — publican artefactos compilados de hardware (NPU/Vitis-AI), no el layout diffusers-ONNX que `optimum-onnx` espera. El equivalente público `stabilityai/sdxl-turbo` SÍ es un checkpoint diffusers normal (misma arquitectura que SDXL, fine-tuneado) — exportable con el mismo camino de `main_export` verificado en el smoke test, y cargable con la MISMA clase `ORTStableDiffusionXLPipeline` (Turbo no necesita clase propia, confirma la hipótesis del design doc). |
| **SD3.5** | **GO condicional vía conversión, con deuda de verificación** (NO-GO directo desde `amd/`) | El repo `amd/` para SD3.5 tiene el mismo problema que Turbo: 404 en `model_index.json`, solo artefactos compilados. La clase `ORTStableDiffusion3Pipeline` SÍ existe en `optimum-onnx==0.1.0` (confirmado por `dir()`) — contradice el riesgo aceptado del design doc de que pudiera faltar. Un equivalente público no-gated (`adamo1139/stable-diffusion-3.5-medium-ungated`) tiene layout diffusers PyTorch normal, en teoría exportable por el mismo camino de `main_export` — **pero esto NO se probó en este spike** (el smoke export de step 4 usa un modelo tiny de SD1.5, no un fixture SD3.5; SD3.5 es transformer-based con un tercer text encoder (`text_encoder_3`, T5) que main_export no ejercitó acá). Task 2 debería correr un smoke export real de un SD3.5 tiny/pequeño antes de darlo por cerrado. |
| **Export API (`main_export`)** | **GO** | `main_export(model_dir, output_dir, task="text-to-image", device="cpu", cache_dir=...)` exporta un pipeline diffusers PyTorch local a ONNX completo — verificado con forward-pass de validación real (steps 1-4 de submodelos, comparación de outputs con atol). Requiere el mismo workaround de detección de `onnxruntime-directml` ya documentado en el spike de 2026-07-22, pero en la ruta de EXPORT, no en la de inferencia (ver más abajo). |

---

## (a) Tabla `repo_id` real → `_class_name` declarado

Fuente: `HF_REPOSITORY_EVIDENCE` de `python scripts\spike_third_party_models.py` (API pública de HF, sin bajar pesos — solo `api/models` con `blobs=true` y `resolve/main/model_index.json`, ambos de tamaño KB).

### SDXL

| repo_id | api_status | gated | model_index_status | `_class_name` | componentes declarados | storage |
|---|---|---|---|---|---|---|
| `amd/stable-diffusion-xl-1.0_io16_amdgpu` | 200 | `false` | 200 | **`ORTStableDiffusionXLPipeline`** | feature_extractor, image_encoder, scheduler, text_encoder, text_encoder_2, tokenizer, tokenizer_2, unet, vae_decoder, vae_encoder | ONNX (los 4 componentes pesados: text_encoder, text_encoder_2, unet, vae_decoder, vae_encoder) |

### SDXL Turbo

| repo_id | api_status | gated | model_index_status | `_class_name` | componentes | storage |
|---|---|---|---|---|---|---|
| `amd/sdxl-turbo_amdgpu` | 200 | `false` | **404** | `null` | — (sin `model_index.json`) | solo "hardware-specific compiled artifacts" (extensiones tipo `.mxr`/`.ctrlpkt`/`.fconst`/`.state`/`.super` — compilado NPU/Vitis-AI) |
| `amd/stable-diffusion-sdxl-turbo-amdnpu-onnx` | 200 | `false` | **404** | `null` | — (sin `model_index.json`) | ONNX + artefactos compilados de hardware mezclados, sin layout diffusers estándar en la raíz |
| `stabilityai/sdxl-turbo` (secundario, público, equivalente) | 200 | `false` | 200 | `StableDiffusionXLPipeline` (nombre nativo de diffusers, NO ya-ORT) | feature_extractor, image_encoder, scheduler, text_encoder, text_encoder_2, tokenizer, tokenizer_2, unet, vae | ONNX + PyTorch weights mezclados por componente (unet/text_encoder/text_encoder_2 tienen ambos; vae solo PyTorch) |

### SD3.5

| repo_id | api_status | gated | model_index_status | `_class_name` | componentes | storage |
|---|---|---|---|---|---|---|
| `amd/stable-diffusion-3.5-medium_amdgpu` | 200 | `false` | **404** | `null` | — (sin `model_index.json`) | solo "hardware-specific compiled artifacts" — mismo patrón que SDXL Turbo `amd/` |
| `stabilityai/stable-diffusion-3.5-medium` (secundario, oficial) | 200 | **`"auto"` (gated)** | **401** | `null` | — (bloqueado, requiere token + aceptar licencia) | PyTorch weights (metadata visible vía API aunque el contenido esté gated) |
| `adamo1139/stable-diffusion-3.5-medium-ungated` (secundario, público, equivalente) | 200 | `false` | 200 | `StableDiffusion3Pipeline` (nativo diffusers) | scheduler, text_encoder, text_encoder_2, text_encoder_3, tokenizer, tokenizer_2, tokenizer_3, transformer, vae | PyTorch weights únicamente — sin ONNX, necesita conversión completa |

## (b) Clases `ORT*Pipeline` reales en `optimum.onnxruntime`

Fuente: `dir(optimum.onnxruntime)` filtrado por `"Diffusion"`/`"StableDiffusion"`, con los pins exactos del repo:

```
ORTDiffusionPipeline
ORTStableDiffusion3Img2ImgPipeline
ORTStableDiffusion3InpaintPipeline
ORTStableDiffusion3Pipeline
ORTStableDiffusionImg2ImgPipeline
ORTStableDiffusionInpaintPipeline
ORTStableDiffusionPipeline
ORTStableDiffusionXLImg2ImgPipeline
ORTStableDiffusionXLInpaintPipeline
ORTStableDiffusionXLPipeline
```

Confirmado: `ORTStableDiffusionXLPipeline` y `ORTStableDiffusion3Pipeline` **existen ambas** en `optimum-onnx==0.1.0`. La hipótesis de riesgo del design doc ("SD3.5 puede no tener clase ORT") **no se cumple** — la clase existe y se importa sin error (`from optimum.onnxruntime import ORTStableDiffusion3Pipeline` no lanzó excepción durante la introspección).

## (c) API de `optimum.exporters.onnx`

### Firma real de `main_export`

```python
main_export(
    model_name_or_path: str,
    output: str | Path,
    task: str = 'auto',
    opset: int | None = None,
    device: str = 'cpu',
    dtype: str | None = None,
    optimize: str | None = None,
    monolith: bool = False,
    no_post_process: bool = False,
    framework: str | None = 'pt',
    atol: float | None = None,
    pad_token_id: int | None = None,
    subfolder: str = '',
    revision: str = 'main',
    force_download: bool = False,
    local_files_only: bool = False,
    trust_remote_code: bool = False,
    cache_dir: str = '<hf cache>',
    token: bool | str | None = None,
    do_validation: bool = True,
    model_kwargs: dict[str, Any] | None = None,
    custom_onnx_configs: dict[str, OnnxConfig] | None = None,
    fn_get_submodels: Callable | None = None,
    use_subprocess: bool = False,
    _variant: str = 'default',
    library_name: str | None = None,
    no_dynamic_axes: bool = False,
    do_constant_folding: bool = True,
    slim: bool = False,
    **kwargs_shapes,
)
```

**Sí sirve para un pipeline diffusers local**: se le pasa un directorio local (no un `repo_id` remoto) como `model_name_or_path` y exporta el pipeline diffusers completo a `output_dir`, con `do_validation=True` por default (compara outputs ONNX vs PyTorch con `atol`).

**`task`**: `task="text-to-image"` funciona explícito y fue el usado en el smoke test (default es `"auto"`, que también hubiera funcionado — no se probó `"auto"` explícitamente para no introducir una variable no controlada en el smoke).

**Progreso por componente — NO hay callback.** `main_export_has_callback: false` (no existe parámetro `callback` en la firma). Lo que SÍ existe es log de progreso por componente vía el logger de `optimum.utils.logging` (activado con `set_verbosity_info()`), con este patrón exacto observado en el smoke test real:

```
***** Exporting submodel 1/4: CLIPTextModel *****
***** Exporting submodel 2/4: UNet2DConditionModel *****
***** Exporting submodel 3/4: AutoencoderKL *****
***** Exporting submodel 4/4: AutoencoderKL *****
```

(el 3/4 y 4/4 corresponden a `vae_encoder` y `vae_decoder`, ambos `AutoencoderKL` — el log no distingue nombre de componente, solo la clase del modelo y el índice).

**Camino por-componente (alternativa a parsear logs)**, confirmado disponible en `optimum.exporters.onnx.convert`:

```python
export_models(
    models_and_onnx_configs: dict[str, tuple[PreTrainedModel | ModelMixin, OnnxConfig]],
    output_dir: Path,
    opset: int | None = None,
    output_names: list[str] | None = None,
    device: str = 'cpu',
    ...
) -> tuple[list[list[str]], list[list[str]]]

export(
    model: PreTrainedModel | ModelMixin,
    config: OnnxConfig,
    output: Path,
    opset: int | None = None,
    device: str = 'cpu',
    ...
) -> tuple[list[str], list[str]]
```

**Recomendación para Task 2 (`generation_converter.py`):** no hubo fricción bloqueante con `main_export` en este spike (el smoke export terminó OK), así que NO es necesario ensamblar manualmente `models_and_onnx_configs` con `TasksManager` para tener progreso — más barato es adjuntar un `logging.Handler` al logger de `optimum.utils.logging` y parsear el patrón `***** Exporting submodel N/M: <ClassName> *****` para actualizar `job.metadata["stages"]` en tiempo real. Si ese parseo resulta demasiado frágil en la práctica, `export_models`/`export` quedan confirmados como el camino de fallback con control total por componente (a costa de reconstruir el armado de `models_and_onnx_configs` que hoy hace `main_export` internamente vía `TasksManager` — no se investigó ese armado a fondo por no ser necesario).

### Workaround de detección de `onnxruntime-directml`

`optimum.utils.import_utils.is_onnxruntime_available()` devuelve **`False`** con los pins de este repo (`optimum_detects_onnxruntime: false`) aunque el módulo `onnxruntime` (en realidad `onnxruntime-directml`) importa perfectamente y expone `DmlExecutionProvider`. Este es el MISMO hallazgo ya documentado en `docs/superpowers/specs/2026-07-22-optimum-spike-findings.md` (optimum no reconoce la distribución `onnxruntime-directml`), pero re-confirmado acá específicamente en la ruta de **export** (`optimum.exporters.onnx.base.is_onnxruntime_available`), no solo en inferencia. El smoke export solo funcionó parcheando ese guard:

```python
with patch("optimum.exporters.onnx.base.is_onnxruntime_available", return_value=True):
    main_export(...)
```

**Nota importante:** `generation_onnx.py::_create_pipeline` (la ruta de INFERENCIA ya shippeada) **no** tiene este patch y funciona igual — confirma que el guard problemático es específico del módulo de export, no del módulo de inferencia (`optimum.onnxruntime.ORTStableDiffusionPipeline.from_pretrained` no lo dispara). Task 2 necesita replicar este patch (o uno equivalente) SOLO dentro de `generation_converter.py`, no en el motor de inferencia existente.

## (d) Estructura real del output dir del smoke export

Comando exacto (dentro de `scripts/spike_third_party_models.py::smoke_export`):

```python
with patch("optimum.exporters.onnx.base.is_onnxruntime_available", return_value=True):
    main_export(
        str(source_dir), output_dir,
        task="text-to-image", device="cpu", cache_dir=str(cache_dir),
    )
```

Modelo usado: `hf-internal-testing/tiny-stable-diffusion-torch` (SD1.5-shape tiny, no baja pesos reales — pocos MB, a `%TEMP%`, borrado al final con `shutil.rmtree`).

Validación de `main_export` (real, con `do_validation=True` default) — 4 submodelos exportados y validados con outputs comparados por atol:

```
Validating ONNX model .../text_encoder/model.onnx...  [OK] last_hidden_state, pooler_output
Validating ONNX model .../unet/model.onnx...            [OK] sample
Validating ONNX model .../vae_encoder/model.onnx...      [OK] latent_parameters
Validating ONNX model .../vae_decoder/model.onnx...      [OK] sample
The ONNX export succeeded and the exported model was saved at: <output_dir>
```

Estructura final del `output_dir`:

```
model_index.json
feature_extractor/preprocessor_config.json
scheduler/scheduler_config.json
text_encoder/config.json
text_encoder/model.onnx
tokenizer/{merges.txt, special_tokens_map.json, tokenizer_config.json, vocab.json}
unet/config.json
unet/model.onnx
vae_decoder/config.json
vae_decoder/model.onnx
vae_encoder/config.json
vae_encoder/model.onnx
```

**Hallazgo crítico — el `_class_name` del `model_index.json` de SALIDA NO cambia con la conversión:**

```json
"exported_class_name": "StableDiffusionPipeline"
```

`main_export` deja el `_class_name` **igual al de origen** (el nombre nativo de diffusers, `StableDiffusionPipeline`), NO lo reescribe a `OnnxStableDiffusionPipeline` ni a ningún nombre `ORT*`. Esto significa que **el `_class_name` por sí solo NO alcanza para distinguir "este directorio ya tiene componentes ONNX" de "este directorio es PyTorch puro"** después de pasar por conversión — hay que seguir mirando las extensiones de archivo por componente (`.onnx` vs `.safetensors`/`.bin`), exactamente el mismo patrón que ya usa `_storage_kind()` en este script y que `generation_installer.py` debería aplicar también al resultado de una conversión, no solo al repo original.

Limpieza confirmada: `SMOKE_TEMP_REMOVED=True` en ambas corridas — `%TEMP%` no quedó con residuos, nada de esto se commiteó.

## Repos gated encontrados (dato para Task 3)

| repo_id | señal de gate |
|---|---|
| `stabilityai/stable-diffusion-3.5-medium` | `gated: "auto"` en la API pública de metadata; `resolve/main/model_index.json` devuelve **401** sin token |

Ningún repo de la colección `amd/` inspeccionada resultó gated (`gated: false` en los 4 candidatos `amd/`). El único 401/403 real observado en este spike vino del repo oficial de Stability AI, consistente con la necesidad de UI de token (`HF_TOKEN`) planteada en el componente 4 del design doc.

## Desviaciones respecto de la hipótesis del design doc

El mapa hipótesis (`2026-07-25-generation-third-party-models-design.md`, componente 1) asumía:

```python
PIPELINE_CLASS_NAMES = {
    "OnnxStableDiffusionPipeline": "ORTStableDiffusionPipeline",  # SD1.5
    "StableDiffusionXLPipeline": "ORTStableDiffusionXLPipeline",  # SDXL/Turbo
    "StableDiffusion3Pipeline": "ORTStableDiffusion3Pipeline",    # SD3.5
}
```

Evidencia real de este spike:

1. **La entrada SDXL no cubre el repo `amd/` real.** `amd/stable-diffusion-xl-1.0_io16_amdgpu` declara `_class_name: "ORTStableDiffusionXLPipeline"` — **ya el nombre ORT**, no `"StableDiffusionXLPipeline"` (el nombre nativo que sí usa el equivalente público `stabilityai/sdxl-turbo`). El mapa necesita soportar **ambos casos**: (a) `_class_name` ya con prefijo `ORT*` → passthrough/identidad (importar directo, sin traducción), y (b) `_class_name` nativo de diffusers → traducir vía el mapa. La hipótesis original solo cubre (b).
2. **La clase de SD3.5 SÍ existe** (`ORTStableDiffusion3Pipeline`, confirmado por `dir()`) — el riesgo aceptado del design doc ("SD3.5 puede no tener clase ORT, se documenta NO-GO si falta") no se materializó. La entrada del mapa hipótesis es correcta como traducción, pero el riesgo asociado queda descartado.
3. **Ningún repo `amd/` de SDXL Turbo o SD3.5 es instalable directo.** Ambos devuelven 404 en `model_index.json` — publican artefactos compilados de hardware (NPU/Vitis-AI), no el layout diffusers-ONNX. Esto no estaba explícito en el design doc como riesgo por variante; el `PIPELINE_CLASS_NAMES` alcanza igual (aplica sobre el resultado de la conversión, que sí produce `model_index.json` diffusers-shaped), pero confirma que **Turbo y SD3.5 dependen 100% del job de conversión (Task 2)** para ser usables desde esta colección — no hay atajo directo como con SDXL.
4. **La conversión no reescribe `_class_name`.** Consecuencia práctica: después de convertir un repo PyTorch (`StableDiffusionXLPipeline`, `StableDiffusion3Pipeline`, etc.) con `main_export`, el `model_index.json` de salida declara el MISMO `_class_name` nativo de origen — el mapa de traducción (`PIPELINE_CLASS_NAMES`) sigue aplicando igual sobre ese resultado (la traducción ocurre en `_load_pipeline_class`, no depende de que el archivo declare ya el nombre ORT), pero `generation_installer.py` no puede usar el CAMBIO de `_class_name` como señal de "conversión completa" — debe seguir confirmando por extensión de archivo (`.onnx` presente) como ya hace hoy para el repo original.
5. **SD3.5 no fue smoke-testeado end-to-end** (solo se confirmó la existencia de la clase y la firma de `main_export`; el smoke real de step 4 corrió sobre un fixture tiny de SD1.5, no de SD3.5). Queda como deuda de verificación explícita para Task 2, no un GO limpio.

## Archivos de este spike

- `scripts/spike_third_party_models.py` — commiteado. Cubre los 4 steps del brief: introspección de repos reales vía API de HF (sin bajar pesos, `model_index.json` de pocos KB), clases `ORT*Pipeline` vía `dir()`, firma de `main_export` + API por-componente de `optimum.exporters.onnx.convert`, y smoke export real con un modelo tiny a `%TEMP%` (auto-limpiado).
- `%TEMP%\upflow-third-party-spike-*` — NO commiteado, borrado automáticamente al final de cada corrida (`shutil.rmtree` en el `finally` del script, confirmado con `SMOKE_TEMP_REMOVED=True` en ambas corridas reales).
