# Descarga de modelos de generación: sin techo, con aviso y precisión elegible — Design

**Fecha:** 2026-07-28
**Estado:** Approved (pendiente de plan de implementación)

## Motivación

Un usuario con una RX 7900 XTX (24 GB) no pudo instalar `stable-diffusion-v1-5/stable-diffusion-v1-5`, el modelo text-to-image más descargado de Hugging Face. El mensaje:

```
La descarga (11121 MB) supera el limite de 8192 MB (MAX_GENERATION_MODEL_DOWNLOAD_MB).
```

El cap de 8192 MB estaba dimensionado contra la máquina del autor. Alguien con más VRAM y más disco queda bloqueado por una constante que no describe su hardware.

Al medirlo, el cap resultó ser el síntoma, no la causa. Los 11121 MB reales (verificado contra la API de HF el 2026-07-28, coincide exacto con el error):

| MB | archivo | se usa? |
|---|---|---|
| 3278 | `unet/diffusion_pytorch_model.safetensors` | sí |
| 3278 | `unet/diffusion_pytorch_model.non_ema.safetensors` | **no** |
| 1639 | `unet/diffusion_pytorch_model.fp16.safetensors` | **no** |
| 1159 | `safety_checker/model.safetensors` | sí |
| 579 | `safety_checker/model.fp16.safetensors` | **no** |
| 469 | `text_encoder/model.safetensors` | sí |
| 319 | `vae/diffusion_pytorch_model.safetensors` | sí |
| 234 | `text_encoder/model.fp16.safetensors` | **no** |
| 159 | `vae/diffusion_pytorch_model.fp16.safetensors` | **no** |
| 1 | `tokenizer/vocab.json` | sí |

`_select_conversion_files` deduplica `.bin` cuando existe el `.safetensors` par, pero **no deduplica variantes** dentro del mismo componente: baja las tres versiones del mismo unet. Y no se leen — `main_export` se invoca sin selector de variante, así que diffusers carga el `.safetensors` pelado. **5889 de los 11121 MB se descargan para nada.** Lo necesario son 5232 MB, que caben bajo el cap que rechazó la operación.

Dos problemas independientes, entonces: un cap que codifica el hardware del autor, y una selección de archivos que desperdicia 2.1x.

## Alcance

Tres cambios, decididos en brainstorming el 2026-07-28:

- **A — Nunca bloquear por tamaño.** El cap se elimina. En su lugar, un pre-flight informativo que advierte cuando algo probablemente no funcione bien, con el botón de instalar siempre habilitado.
- **B — Precisión elegible.** El usuario elige fp16 o fp32 entre las que el repo publica. La elección define qué se baja **y** en qué precisión queda el ONNX exportado.
- **C — Descubrimiento sin escribir.** El buscador muestra modelos por popularidad con query vacía, cada uno con un badge de compatibilidad detectada.

**Fuera de alcance:** multi-GPU por job (un job sigue aterrizando en un dispositivo; el diseño no lo estorba, ver §Dispositivos). Tests E2E de instalación real. El cap de upscalers (`MAX_MODEL_DOWNLOAD_MB`, 2048) **no se toca**: protege otro camino y nadie pidió cambiarlo.

## Decisiones tomadas

| Decisión | Elección | Razón |
|---|---|---|
| Bloqueo por tamaño | Eliminado, reemplazado por aviso | Pedido explícito: "no bloqueemos la descarga, solo advirtamos" |
| Base del aviso | Disco libre real + compat detectada + VRAM libre real | Pedido explícito. Las tres son medibles, ninguna es una constante del autor |
| Origen de la compat | **Detectada**, no curada | `gated`, `model_index.json` y presencia de `.onnx` salen de la metadata en vivo. Una tabla a mano envejece entre releases |
| VRAM requerida | Estimada de los bytes de pesos × factor por resolución | HF no la publica. Una fórmula cubre cualquier repo desde el día uno; una tabla curada solo cubre lo curado |
| Qué hace elegir fp | Descarga **y** runtime | El ONNX sale en esa precisión. Es lo que le da sentido a la elección para alguien con otra GPU |
| Catálogo | HF por descargas + badges | Cero lista mantenida a mano |
| Dónde se calcula el pre-flight | Endpoint aparte, bajo demanda al expandir una tarjeta | El browse queda en 1 request sin importar cuántos resultados; la sonda cara ocurre solo para la tarjeta que el usuario abre |
| Dependencia de VRAM | Mergear `feature/gpu-capacity-admission` primero | Ya trae `DxgiVramProbe` completo y con tests. Duplicar la sonda divergiría la rama |

## Dependencia previa

`feature/gpu-capacity-admission` (local, sin pushear) debe entrar a master **antes** de este trabajo. Aporta `app/services/resource_probes.py` (`ResourceProbe`, `NullProbe`, `SystemRamProbe`), `DxgiVramProbe` sobre `IDXGIAdapter3::QueryVideoMemoryInfo` en `devices_service.py`, y sus tests. Entra también la admisión de jobs por capacidad real, que no se pidió en este spec pero está terminada y testeada.

## Componentes

Cuatro módulos nuevos. Los tres primeros son puros: sin red, sin disco, sin estado.

### 1. `app/services/generation_variants.py`

Reemplaza la lógica ciega a variantes de `_select_conversion_files`.

```python
Precision = Literal["fp16", "fp32"]

def available_precisions(files: list[HfFile]) -> tuple[Precision, ...]
def select_for_precision(files: list[HfFile], declared: list[str], precision: Precision) -> list[HfFile]
```

Reglas de `select_for_precision`, en orden:

1. Descartar `CONVERSION_SKIP_SUFFIXES` (`.ckpt`, `.msgpack`, `.h5`, `.onnx`, `.onnx_data`, `.pb`) — sin cambio.
2. Descartar **siempre** cualquier archivo con `.non_ema.` en el nombre. Es un checkpoint de entrenamiento; la inferencia nunca lo lee.
3. Mantener solo componentes declarados en `model_index.json` + metadata top-level `.json`/`.txt` — sin cambio.
4. Por cada carpeta de componente, quedarse con **un solo** archivo de pesos: el de la precisión pedida si existe, si no el otro. `.fp16.` identifica la variante fp16; el nombre pelado es fp32.
5. Descartar `.bin` cuando la misma carpeta tiene `.safetensors` — sin cambio.

`available_precisions` devuelve `("fp16", "fp32")` cuando hay ambas, o la única que haya. Verificado contra repos reales el 2026-07-28: SD1.5 y SDXL base tienen ambas; `Tongyi-MAI/Z-Image-Turbo` solo fp32.

Resultado esperado para SD1.5: **fp16 → 2611 MB, fp32 → 5232 MB** (hoy: 11121 MB).

### 2. `app/services/generation_compat.py`

```python
CompatVerdict = Literal["ready_onnx", "needs_conversion", "gated", "incompatible"]

def classify(filenames: tuple[str, ...], gated: bool | str | None) -> tuple[CompatVerdict, str]
```

Se alimenta de `siblings[].rfilename` y `gated`, ambos presentes en la respuesta de búsqueda con `full=true` — **cero HTTP adicional**. Devuelve veredicto y motivo legible.

| Condición | Veredicto |
|---|---|
| `gated` es truthy | `gated` |
| falta `model_index.json` | `incompatible` |
| todo componente con pesos torch tiene `.onnx` propio | `ready_onnx` |
| algún componente solo tiene torch | `needs_conversion` |

El orden importa: `gated` gana sobre todo lo demás, porque sin token no se puede saber nada más del repo.

Verificado el 2026-07-28: SD1.5 → `needs_conversion`; `stabilityai/stable-diffusion-xl-base-1.0` → tiene `.onnx`; `black-forest-labs/FLUX.1-dev` → `gated: "auto"`; `wikeeyang/Flux2-Klein-9B-True-V2` → sin `model_index.json`, `incompatible`.

### 3. `app/services/vram_estimate.py`

```python
def estimate_peak_bytes(weight_bytes: int, width: int, height: int) -> int
```

`weight_bytes × factor(width × height)`. El factor arranca con estos valores, escalonados por cantidad de píxeles y con interpolación lineal entre escalones:

| píxeles | resolución típica | factor |
|---|---|---|
| ≤ 262 144 | 512×512 | 1.25 |
| 589 824 | 768×768 | 1.45 |
| ≥ 1 048 576 | 1024×1024 | 1.70 |

**Son una extrapolación, no una medición por modelo**, y el módulo lo dice en un comentario. El margen sobre los pesos cubre activaciones y buffers intermedios, que crecen con la resolución mientras los pesos no.

Monótona en las dos variables por construcción: más pesos o más píxeles nunca puede dar menos pico. Eso es lo que fijan los tests, no los valores exactos — así revisar los factores con mediciones reales no rompe la suite.

Se etiqueta siempre como estimación en la UI. No pretende precisión: pretende distinguir "entra holgado" de "no entra".

### 4. `app/services/generation_preflight.py`

Composición. Único módulo del grupo que toca red y disco.

```python
async def preflight(repo_id: str, width: int, height: int) -> PreflightReport
```

Hace dos requests (`repo_files` con `blobs=true` para tamaños, y `model_index.json` para componentes declarados), y calcula bytes para **cada** precisión disponible — así el picker muestra los dos números sin un segundo viaje.

```json
{ "repoId": "stable-diffusion-v1-5/stable-diffusion-v1-5",
  "compat": "needs_conversion",
  "compatReason": "...",
  "degraded": false,
  "precisions": [ {"precision": "fp16", "downloadBytes": 2737000000, "estimatedPeakBytes": 3300000000},
                  {"precision": "fp32", "downloadBytes": 5486000000, "estimatedPeakBytes": 6600000000} ],
  "referenceResolution": {"width": 512, "height": 512},
  "disk": {"targetPath": "D:\\...\\temp", "freeBytes": 6700000000},
  "devices": [ {"id": "dml:0", "name": "RX 7900 XTX", "kind": "gpu", "totalVramBytes": 25757000000, "freeVramBytes": 23700000000},
               {"id": "cpu", "name": "CPU", "kind": "cpu", "totalVramBytes": null, "freeVramBytes": null} ] }
```

El reporte lleva **hechos medidos, no veredictos**. Comparar el pico estimado de la precisión elegida contra `freeVramBytes` de cada dispositivo es derivación pura, y vive en el frontend junto al picker — así cambiar de precisión no re-consulta el servidor. Lo mismo con disco: el servidor informa `freeBytes`, el frontend decide si hay que avisar.

### Dispositivos y multi-GPU

`devices` es una fila **por cada dispositivo enumerado** por `DevicesService.list_devices()` (`dml:0`, `dml:1`, …, `cpu`), no un veredicto global. La comparación pico-vs-libre se aplica por fila.

Cuando exista multi-GPU por job, la misma comparación por dispositivo alimenta un agregador distinto (suma de VRAM entre los dispositivos elegidos). No hay que rediseñar el reporte: hay que agregar un consumidor.

### Modificaciones

| Archivo | Cambio |
|---|---|
| `app/config.py` | `MAX_GENERATION_MODEL_DOWNLOAD_MB` **eliminado**. Sin setting que lo reemplace |
| `app/services/hf_client.py` | `max_bytes: int \| None`, `None` = sin techo. `search()` acepta query vacía y manda `sort=downloads&direction=-1`; `HfModelSummary` expone `filenames` y `gated` |
| `app/services/generation_installer.py` | `_ensure_size_cap` **eliminado**. Acepta `precision`. Stagea bajo nombre canónico |
| `app/services/generation_converter.py` | Idem, más `main_export(..., dtype=...)`. `_select_conversion_files` reemplazado por `generation_variants` |
| `app/api/` | `GET /api/generation/models/preflight?repoId=<id>&width=512&height=512` (`width`/`height` opcionales, default 512); el POST de install acepta `precision` |
| `.env.example`, `README.md` | Quitar la fila del cap |

### Los dos mecanismos de la precisión

`main_export` **no expone** un parámetro para elegir la variante de pesos (su firma real, documentada en `2026-07-25-third-party-spike-findings.md`, tiene `dtype` y `_variant`; `_variant` es privado y refiere al layout del export, no a los archivos `.fp16.`).

Elegir precisión requiere **dos** cambios independientes, medidos por separado en el smoke del 2026-07-28 (ver §Verificación):

**1. Staging bajo nombre canónico → ahorro de descarga.** Al escribir `unet/diffusion_pytorch_model.fp16.safetensors` en `src_root`, se guarda como `unet/diffusion_pytorch_model.safetensors`. diffusers lo carga por ese nombre sin ningún kwarg, y safetensors lleva el dtype en su propio header. `_safe_staging_dest` ya es el único punto por donde pasan los destinos, así que la normalización del nombre vive ahí.

**2. `main_export(..., dtype="fp16", atol=1e-2)` → precisión de runtime.** Medido: cargar pesos fp16 y exportar sin `dtype` produce un ONNX **fp32** — optimum los sube. El staging por sí solo NO da precisión de runtime. `dtype="fp16"` sí produce initializers `FLOAT16`, pero con el `atol` por defecto la validación de forward-pass falla por ruido de redondeo normal (unet: diff `0.00390625` contra atol `1e-05`) y `main_export` levanta `RuntimeError` — aunque deja el modelo escrito.

`atol=1e-2` hace pasar la validación conservándola. Se prefiere sobre `do_validation=False` (que también funciona) porque mantener la comparación contra el modelo de referencia es una red de seguridad real: detecta un export roto, no solo redondeo.

Los dos mecanismos son necesarios y ninguno reemplaza al otro.

### Frontend

| Archivo | Cambio |
|---|---|
| `services/generation.ts` | `preflightGenerationModel()`; `installGenerationModel(repoId, precision)` |
| `modules/models/GenerationHfSearch.tsx` | Query vacía → browse por descargas en vez de `SearchEmptyState` |
| `modules/models/GenerationModelCard.tsx` (nuevo) | Badge de compat; al expandir dispara preflight una vez; picker de precisión; filas por dispositivo; avisos; Install **siempre habilitado** |

El browse no es un modo aparte: es la misma búsqueda con query vacía, mismo endpoint y mismo shape de respuesta.

```
[Listo]        stabilityai/stable-diffusion-xl-base-1.0
               1,476,978 ↓   ♥ 4,231                        [Install]

  ── expandida ──────────────────────────────────────────
  Precisión   (•) fp16 · baja 2.6 GB    ( ) fp32 · baja 5.2 GB

  dml:0  RX 7900 XTX   libre 22.1 GB   ✓ entra
  dml:1  RX 6600        libre 7.4 GB   ✗ no entra (~8.4 GB estimados a 512×512)
  cpu    CPU                            ⚠ varios minutos por imagen

  ⚠ Quedan 6.2 GB libres en D:\ y hace falta 2.6 GB
```

## Manejo de errores

**Principio: el pre-flight es diagnóstico. Si falla, no bloquea nada.** Misma regla que el log a archivo de v0.14.3 — si no se puede escribir, se sigue sin log.

| Falla | Comportamiento |
|---|---|
| Preflight revienta (HF 5xx, timeout) | 200 con `degraded: true` y los campos que no se pudieron calcular en `null`. La tarjeta dice que no se pudo evaluar. **Install sigue habilitado** |
| Probe de VRAM falla (sin DXGI, no-Windows) | `freeVramBytes: null` en esa fila; cae al `NullProbe`. La fila dice que no se pudo medir |
| `disk_usage` falla | `disk: null`, sin aviso de disco |
| Repo `gated` | Badge `gated`, pero se intenta igual si el usuario quiere: puede tener `HF_TOKEN` configurado. El 401 real lo traduce `_wrap_hf_auth_error`, sin cambios |
| `precision` inexistente en el POST | 422 con las disponibles. El frontend no puede llegar ahí; la API es defensiva igual |
| **Disco lleno a mitad de descarga** | Consecuencia directa de quitar el cap. El `finally` de `_download_and_register` ya hace `rmtree(staging_root)`, así que el parcial se limpia. Falta mapear `OSError` con `errno.ENOSPC` a "no queda espacio en D:\\" en vez de dejar subir el `OSError` crudo al job |

Ese último es el nuevo modo de falla real del sistema sin cap, y merece un mensaje accionable.

## Testing

Unitarios puros, sin red:

- `generation_variants` con el listado real de SD1.5 (36 archivos): fp16 → 2611 MB, fp32 → 5232 MB, `.non_ema.` nunca aparece, `.bin` excluido si hay `.safetensors`. `Z-Image-Turbo` → `available_precisions == ("fp32",)`
- `generation_compat`: los cuatro veredictos contra los repos reales verificados el 2026-07-28, incluida la precedencia de `gated` sobre el resto
- `vram_estimate`: monotonía en pesos y en resolución

Integración (`HfClient` ya inyecta `transport`):

- query vacía manda `sort=downloads` y **no** manda `search`
- `max_bytes=None` no aborta un stream grande — el test que fija que el techo se fue
- `preflight` devuelve una fila por cada dispositivo enumerado, con probe y disco mockeados
- `preflight` con HF caído devuelve `degraded: true` y HTTP 200, no 5xx

Regresión del bug de origen:

- SD1.5 fp16 pesa ≤ 3 GB (hoy: 11121 MB)
- ningún caller de `_ensure_size_cap` sobrevive en el árbol

Frontend (vitest, molde en `GenerationHfSearch.test.tsx`):

- query vacía renderiza resultados, no el cartel de texto
- badge por veredicto de compat
- expandir dispara preflight una sola vez
- **Install habilitado incluso con avisos presentes** — el test que fija el requisito central de A
- el picker ofrece solo las precisiones disponibles

Los tests del cap de **upscalers** (`test_hf_client.py:552,571`, `test_model_installer.py:517`) se quedan intactos. El único que se borra es la aserción de `max_generation_model_download_mb == 8192` en `test_hf_client.py:594`.

Sin E2E: bajar 2.6 GB en CI no va.

## Verificación

`main_export(..., dtype=...)` no había sido probado: el spike de 2026-07-25 exportó siempre en la precisión por defecto. Se cerró con un smoke el 2026-07-28 sobre el fixture tiny del propio spike (`hf-internal-testing/tiny-stable-diffusion-torch`), `device="cpu"`, torch 2.13.0+cpu, con el mismo patch de `is_onnxruntime_available` que el spike ya documentó.

| Invocación | Resultado | dtype de los initializers |
|---|---|---|
| sin `dtype` (baseline) | OK | `FLOAT` |
| `dtype="fp16"` | `RuntimeError` en validación, **modelo escrito igual** | `FLOAT16` |
| `dtype="fp16", atol=1e-2` | **OK, sin excepción** | `FLOAT16` |
| `dtype="fp16", do_validation=False` | OK | `FLOAT16` |
| pesos fp16 bajo nombre canónico, sin `dtype` | OK | `FLOAT` |

Tres conclusiones, todas medidas:

1. **`dtype="fp16"` funciona en CPU** para exportar. Lo que falla es la validación de forward-pass contra el modelo de referencia, por ruido de redondeo fp16 esperable (`0.00390625` ≈ 2⁻⁸ contra un atol de `1e-05`). Además, el propio formateo del error de optimum tropieza con un `IndexError: tuple index out of range` en ese camino — otra razón para no dejar que se dispare.
2. **`atol=1e-2` es la elección**: pasa la validación sin apagarla.
3. **El staging por sí solo no alcanza.** Cargar pesos fp16 bajo el nombre canónico exporta un ONNX fp32. La precisión de runtime viene únicamente de `dtype=`. De ahí que §Los dos mecanismos de la precisión pida los dos.

Sin deuda de verificación abierta al cerrar este spec.

## Riesgo conocido

`atol=1e-2` es la tolerancia verificada sobre un fixture *tiny*. Un modelo real (SD1.5, SDXL) tiene muchas más capas y por lo tanto más acumulación de error, así que su diff fp16 puede superar `1e-2` y volver a levantar `RuntimeError`. La primera conversión fp16 de un modelo real es el momento de confirmarlo; si pasa, el ajuste es subir `atol` para fp16 (no apagar la validación), y queda registrado acá con el número medido.
