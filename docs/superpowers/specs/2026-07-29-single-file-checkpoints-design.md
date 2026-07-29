# Instalar checkpoints single-file (formato Civitai) — Design

**Fecha:** 2026-07-29
**Estado:** Approved (pendiente de plan de implementación)

## Motivación

v0.15.1 arregló un badge que mentía: un repo de checkpoints sueltos se anunciaba como `ONNX LISTO` y fallaba al instalar. El arreglo fue honesto pero terminal — esos repos ahora dicen `INCOMPATIBLE` y no hay forma de usarlos.

El problema es que ese formato no es un caso raro. La [documentación de diffusers](https://huggingface.co/docs/diffusers/using-diffusers/other-formats) lo presenta como una de exactamente **dos** formas en que se publica un modelo de difusión, y el [wiki de Civitai](https://github.com/civitai/civitai/wiki/How-to-use-models) describe sus checkpoints como archivos `.safetensors` sueltos listos para usar. En el browse de Upflow, entre los text-to-image más descargados de Hugging Face, aparecen varios así.

Rechazarlos es rechazar un ecosistema entero.

## Investigación previa (2026-07-29)

Todo lo de abajo está **medido contra las librerías instaladas y contra repos reales**, no inferido.

### El techo del runtime son seis arquitecturas

`optimum.onnxruntime.ORTPipelineForText2Image.ort_pipelines_mapping` es la lista completa de lo que se puede **ejecutar**:

```
['latent-consistency', 'stable-diffusion', 'stable-diffusion-xl',
 'stable-diffusion-3', 'flux', 'sana']
```

diffusers 0.39 **carga** más que eso. `Flux2Pipeline`, `ZImagePipeline` y `QwenImagePipeline` existen en diffusers y no tienen entrada en ese mapping, así que FLUX.2, Z-Image y Qwen-Image quedan fuera por una dependencia, no por decisión nuestra ni por VRAM. Las alternativas —esperar upstream, contribuir las clases, montar otro runtime— están fuera de este trabajo.

**`ort_pipelines_mapping` es la única autoridad; los nombres de clase no.** Medido: `ORTSanaPipeline` **no existe** como símbolo en `optimum.onnxruntime`, y sin embargo `sana` **sí** está en el mapping (se despacha por la auto-clase). Preguntar "¿existe `ORT{_class_name}`?" da falsos negativos y no debe usarse como test.

**GGUF queda fuera por imposibilidad.** No hay camino GGUF → ONNX ([onnxruntime-genai #761](https://github.com/microsoft/onnxruntime-genai/issues/761), [#1279](https://github.com/microsoft/onnxruntime-genai/issues/1279), [onnx #6282](https://github.com/onnx/onnx/discussions/6282)); GGUF es un formato del linaje llama.cpp para modelos decoder-only. Además [ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF) documenta que la cuantización no era viable para UNet por las capas conv2d. Soportarlo sería otro runtime entero (`stable-diffusion.cpp`), no una extensión de este.

### La arquitectura se detecta leyendo ~360 KB, no 6.6 GB

`infer_diffusers_model_type(checkpoint)` es puro (sin red) y solo inspecciona **presencia de claves y `.shape`** — nunca valores de tensores. El header de un `.safetensors` trae `shape` de cada tensor y se lee con dos HTTP Range requests.

Verificado contra un checkpoint real de 6616 MB: header de **358 KB**, 2515 claves, devolvió `xl_base` correctamente. Así que el detector propio de diffusers (235 líneas mantenidas upstream) se usa tal cual, sin duplicar heurística.

### El detector NO es un validador — cae a `v1` por default

Este es el hallazgo que define el diseño. Medido sobre cinco archivos reales:

| archivo | claves | claves que el detector consulta | devuelve |
|---|---|---|---|
| `ponyDiffusionV6XL_v6StartWithThisOne.safetensors` | 2515 | 4 | `xl_base` |
| `sdxl_vae.safetensors` (VAE suelto) | 250 | **0** | **`v1`** |
| `ZI Ada Wong (vrtlAdaWong).safetensors` (LoRA) | 630 | **0** | **`v1`** |
| `ip-adapter-faceid-plusv2_sdxl_lora.safetensors` | 1680 | **0** | **`v1`** |
| `flux-2-klein-4b-fp8.safetensors` | 309 | 2 | `flux-2-dev` |

Con **cero** señales, el detector devuelve `v1` (SD1.5, su default histórico). Es un clasificador que asume que ya sabés que el archivo es un checkpoint.

Usarlo como gate reproduciría exactamente el bug de v0.15.1: **ausencia de señal leída como veredicto positivo**. Un LoRA se clasificaría SD1.5, se bajaría, y el badge volvería a mentir.

### Los modelos nuevos se publican componente-por-repo

`flux-2-klein-4b-fp8.safetensors` tiene prefijos `double_blocks`, `single_blocks`, `img_in`, `txt_in`: es **solo el backbone**, sin text encoder ni VAE. Por eso existe `black-forest-labs/FLUX.2-small-decoder` como repo aparte.

El checkpoint monolítico es una convención de la era UNet / linaje SD. Converge con el techo de arriba: los modelos que sí podemos correr son justamente los que se publican monolíticos.

## Alcance

Instalar un checkpoint single-file de las seis arquitecturas con clase ORT, eligiendo cuál archivo del repo instalar.

**Fuera de alcance:** GGUF (imposible, ver arriba). FLUX.2 / Z-Image / Qwen-Image (sin clase ORT; se detectan y se explican). LoRAs y adapters — son composición sobre un modelo base, otro problema. Tests E2E de instalación real.

## Decisiones tomadas

| Decisión | Elección | Razón |
|---|---|---|
| Repos con varios checkpoints | Se elige el archivo al instalar | El formato real: un repo suele traer varios modelos distintos. `model_id` incluye el nombre del archivo, así que se pueden tener varios instalados |
| Arquitecturas | Las seis con clase ORT | "Entre más compatibilidad mejor" — y seis es el techo, no una elección |
| Verificación de export | SDXL y SD1.x verificadas; SD3.5, Flux, LCM y Sana etiquetadas "no verificado" | El spike de 2026-07-25 dejó esa deuda abierta. Se intentan igual; si el export falla, el job reporta el error real |
| Detección de arquitectura | `infer_diffusers_model_type` de diffusers, alimentado con un shim del header | 235 líneas mantenidas upstream, contra duplicar heurística propia |
| Gate de instalabilidad | Regla de completitud estructural **propia**, separada del detector | El detector cae a `v1` sin señal; no puede ser el gate |
| Aviso de disco | Pico real (~3x el checkpoint) | Ya decidido en el spec de 2026-07-28. Confirmado por la comunidad: "at least like 20 to 30 GB of files" |
| Aviso de RAM | Nuevo, con `SystemRamProbe` | `from_single_file` carga el checkpoint entero en RAM. La sonda ya está cableada y sin usar en el pre-flight |
| Integración | Estrategia de staging dentro del converter | Las dos ramas convergen en "árbol diffusers en `src_root`", así que export, validación, promote y progreso se reusan verbatim |

## Componentes

### 1. `app/services/generation_single_file.py` (nuevo)

```python
Architecture = str  # el tipo que devuelve infer_diffusers_model_type, ej. "xl_base"

@dataclass(slots=True, frozen=True)
class CheckpointVerdict:
    installable: bool
    architecture: Architecture | None
    ort_pipeline_class: str | None
    reason: str

def classify_checkpoint(header: dict[str, dict]) -> CheckpointVerdict
def supported_architecture(detected: Architecture) -> str | None
def materialize(checkpoint_path: Path, out_dir: Path, architecture: Architecture) -> None
```

`classify_checkpoint` recibe el header parseado del `.safetensors` (`{clave: {"shape": [...], ...}}`) y aplica, **en este orden**:

1. **Marcadores de LoRA** → no instalable. Cualquier clave con `lora_up`, `lora_down`, `lora_A`, `lora_B`, `.alpha`, `lora_unet_` o `lora_te_`. Un LoRA es composición sobre un modelo base, no un modelo.
2. **Completitud estructural** → no instalable si falta alguno de los tres roles. Un pipeline autocontenido tiene los tres en el mismo archivo:

   | Rol | Prefijos aceptados |
   |---|---|
   | backbone | `model.diffusion_model.`, `double_blocks.`, `single_blocks.`, `joint_blocks.`, `diffusion_model.` |
   | text encoder | `conditioner.embedders.`, `cond_stage_model.`, `text_encoders.` |
   | vae | `first_stage_model.` |

   El rol vae acepta **solo** `first_stage_model.`. Un `decoder.`/`encoder.` en la raíz significa que el archivo **es** un VAE, no que contenga uno — medido en `sdxl_vae.safetensors`.
3. **Detección de arquitectura** → `infer_diffusers_model_type` sobre un shim `{clave: objeto-con-.shape}`. Solo se llama después de pasar 1 y 2, así que su default a `v1` nunca decide nada.
4. **Soporte del runtime** → `supported_architecture(detected)`. Si devuelve `None`, no instalable, con el motivo nombrando que optimum-onnx no soporta esa arquitectura.

**El mapeo entre los dos vocabularios va en una tabla explícita.** `infer_diffusers_model_type` devuelve nombres del vocabulario single-file de diffusers (`xl_base`, `v1`, `flux-dev`, `sd35_medium`); `ort_pipelines_mapping` usa el de optimum (`stable-diffusion-xl`, `stable-diffusion`, `flux`). No hay correspondencia derivable: intentar resolverla por red vía el repo base falla, porque los repos base de SD2.1, SD3.5, FLUX.1 y FLUX.2 están **gated** (HTTPError medido el 2026-07-29).

```python
_ARCHITECTURE_TO_ORT: dict[str, str] = {
    "v1": "stable-diffusion",
    "v2": "stable-diffusion",
    "xl_base": "stable-diffusion-xl",
    "playground-v2-5": "stable-diffusion-xl",
    "sd35_large": "stable-diffusion-3",
    "sd35_medium": "stable-diffusion-3",
    "flux-dev": "flux",
    "flux-schnell": "flux",
    "sana": "sana",
}
```

Deliberadamente **fuera** de la tabla, aunque diffusers los detecte: `xl_refiner` y las variantes `inpainting` (son img2img/inpaint, no text-to-image), y `flux-2-dev`, `z-image-turbo`, los Qwen y todo lo de video (sin soporte en el runtime).

La tabla se **valida al importar el módulo** contra `ORTPipelineForText2Image.ort_pipelines_mapping`: si un upgrade de optimum renombra una clave, el import falla ruidosamente en vez de degradar en silencio a "no soportado".

`materialize` corre `from_single_file()` y `save_pretrained(out_dir)`. Es torch pesado y bloqueante: va en `asyncio.to_thread`, como el export.

### 2. `hf_client.read_safetensors_header(repo_id, path) -> tuple[dict, int]`

Dos Range requests: 8 bytes para el largo del header, después el JSON. Devuelve el header parseado y su tamaño en bytes. Es red, por eso vive acá y no en el módulo puro.

### 3. Modificaciones

| Archivo | Cambio |
|---|---|
| `generation_compat.py` | Veredicto nuevo `single_file` cuando hay `.safetensors` en la raíz y no hay `model_index.json`. Es un **candidato**, no una promesa |
| `generation_preflight.py` | Para `single_file`, lee el header de cada candidato y devuelve `checkpoints: [{path, sizeBytes, architecture, installable, reason}]`. Aviso de RAM además del de VRAM |
| `generation_converter.py` | Rama de staging: si el job trae `checkpoint_path`, descarga ese archivo y llama `materialize` en vez de `select_for_precision` |
| `generation_installer.py` | `install_from_hf` acepta `checkpoint_path`; `_generation_model_id` lo incorpora al id |
| `app/models.py` | `ConversionJob` e `InstallJob` ganan `checkpoint_path: str \| None` |
| `app/schemas.py` | `InstallModelRequest` gana `checkpointPath`; `PreflightResponse` gana `checkpoints` y el aviso de RAM |
| `app/api/routes.py` | Pasa `checkpointPath` al installer |

### Flujo

Descubrimiento, sin bajar pesos:

```
search → siblings[].rfilename
  ├─ model_index.json + pesos en carpetas → camino diffusers (sin cambios)
  ├─ .safetensors en la raiz              → badge SINGLE-FILE (candidato)
  │    └─ al expandir: read_safetensors_header por candidato (~360 KB)
  │         ├─ LoRA / incompleto            → no se ofrece, con motivo
  │         ├─ arquitectura sin clase ORT   → no instalable, nombra la clase
  │         └─ soportada                    → instalable, entra al picker
  └─ ninguna                                → incompatible
```

Instalación:

```
descarga 1 checkpoint (unlimited=True)
  → materialize(): from_single_file + save_pretrained   [etapa nueva, visible]
  → src_root con arbol diffusers                        [converge aca]
  → main_export(dtype, atol)                            [reusado verbatim]
  → validacion funcional + promote                      [reusado verbatim]
```

**La autoridad es el install, no el header.** `from_single_file` reclasifica sobre el checkpoint completo real. El header solo existe para no desperdiciar 6.6 GB cuando se puede saber antes.

### UI

```
LyliaEngine/Pony_Diffusion_V6_XL              [SINGLE-FILE]
  183,405 ↓                                      [Install]

  ── expandida ────────────────────────────────────────
  Checkpoint
   (o) ponyDiffusionV6XL_v6StartWithThisOne     6.5 GB
       SDXL · pico estimado ~20 GB de disco
   ( ) sdxl_vae                                  335 MB
       no instalable: es un VAE, no un pipeline completo

  dml:0  RX 7800 XT   libre 15.0 GB   ✓ entra
  RAM    31.4 GB libres              ✓ alcanza

  ⚠ La conversión necesita ~20 GB de pico en D:\
```

## Manejo de errores

Mismo principio: el pre-flight informa, nunca bloquea. Install queda habilitado siempre.

| Falla | Comportamiento |
|---|---|
| No se pudo leer el header | Ese checkpoint dice "no se pudo evaluar". **Install habilitado**: `from_single_file` reclasifica sobre los pesos reales |
| LoRA / incompleto | No se ofrece en el picker. Si se fuerza por API, error antes de descargar |
| Arquitectura sin clase ORT | Error nombrando la clase ausente (`ORTFlux2Pipeline` no existe en optimum-onnx), para que no parezca bug nuestro |
| `materialize` sin RAM | `MemoryError` / `RuntimeError` de alocación → mensaje accionable con el tamaño del checkpoint. El pre-flight ya avisó |
| Disco lleno | `ENOSPC` accionable, reusado sin cambios |
| `checkpoint_path` que no existe en el repo | 422 con la lista de candidatos |
| `checkpoint_path` que escapa el staging | Ya cubierto por `_safe_staging_dest` |

## Testing

Fixtures de datos reales, capturados el 2026-07-29 (1940 bytes en total: solo las claves que el detector consulta más una representativa por rol). Van a `tests/assets/single_file_fixtures.json`.

Unitarios puros:

- `classify_checkpoint` contra los cinco: `pony_sdxl` → instalable `xl_base`; `vae_only` → no instalable por completitud; `zimage_lora` e `ipadapter_lora` → no instalables por marcadores LoRA; `flux2_backbone` → no instalable (backbone-only, y además sin clase ORT)
- **Regresión del bug de v0.15.1**: un LoRA nunca sale instalable, **aunque `infer_diffusers_model_type` devuelva `v1`**. El fixture prueba que devuelve `v1` con cero claves consultadas
- El rol vae exige `first_stage_model.`: un archivo con solo `decoder.`/`encoder.` no lo satisface
- `supported_architecture`: cada clave de la tabla devuelve su valor; `flux-2-dev`, `z-image-turbo`, `xl_refiner` y un tipo inventado devuelven `None`
- **La validación de la tabla al importar**: todo valor de `_ARCHITECTURE_TO_ORT` está en `ort_pipelines_mapping`. Es el test que avisa si un upgrade de optimum rompe el mapeo
- El orden importa: un archivo con marcadores LoRA se rechaza sin llamar al detector

Integración (`transport` mockeado):

- `read_safetensors_header` emite dos Range requests y parsea el JSON
- El preflight de un repo single-file devuelve una fila por candidato y marca solo los instalables
- Header ilegible → ese candidato queda `installable: null`, el resto del reporte intacto

`materialize` se mockea en los tests del converter; correr torch real ahí no va.

Frontend: picker de checkpoint con los no instalables visibles y deshabilitados **como opción del radio** (no como botón), Install siempre habilitado, aviso de RAM.

## Riesgo conocido

Las cuatro arquitecturas etiquetadas "no verificado" (SD3.5, Flux, LCM, Sana) heredan la deuda del spike de 2026-07-25: nunca se corrió un export real de ninguna. La primera conversión real de cada una es la que va a decir si `main_export` las soporta con el `atol` que usamos. El fallback, si falla, es el mismo que ya está documentado: subir `atol` con el número medido, no apagar la validación.
