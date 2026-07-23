# Spike findings: optimum + onnxruntime-directml 1.24.4

**Fecha:** 2026-07-23
**Veredicto: GO** — optimum + onnxruntime-directml 1.24.4 corren un forward-pass real de Stable Diffusion 1.5 en `DmlExecutionProvider` (`dml:0`) sin crash, con `use_io_binding=False` por default (no se necesita workaround para el riesgo documentado de IOBinding+DML de este repo).

Máquina de prueba: Windows 11, AMD RX 7800 XT (`dml:0`), Python 3.11.7. Venv aislado `.venv-spike` (borrado al final de este spike).

---

## (a) Pin exacto

```
onnxruntime-directml==1.24.4
optimum==2.1.0
optimum-onnx==0.1.0
transformers==4.57.6
diffusers==0.39.0
torch==2.13.0
huggingface_hub==0.36.2
onnx==1.22.0
numpy==2.4.6
pillow==12.3.0
```

`torch==2.13.0` cae dentro del rango que ya declara `pyproject.toml` (`torch>=2.2.0,<3.0.0`) — sin conflicto. `torch` SÍ es import-requerido en el path ONNX: `optimum/onnxruntime/modeling_diffusion.py` hace `import torch` a nivel de módulo (usado para `torch.Generator`, dtype/device helpers), aunque la inferencia en sí corre sobre sesiones ONNX Runtime, no sobre tensores torch.

## (b) Receta pip segura (paso a paso, la que funcionó)

**Riesgo confirmado:** `pip install --dry-run --report ... "optimum[onnxruntime]"` resuelve `onnxruntime==1.27.0` (vanilla) junto con `optimum==2.1.0` y `optimum-onnx==0.1.0`. Causa raíz exacta (leída del METADATA del wheel): el extra `optimum[onnxruntime]` de `optimum==2.1.0` declara `Requires-Dist: optimum-onnx[onnxruntime]; extra == "onnxruntime"`, y el extra `onnxruntime` de `optimum-onnx==0.1.0` declara `Requires-Dist: onnxruntime>=1.18.0; extra == "onnxruntime"` — es decir, instalar con el extra SIEMPRE trae el paquete vanilla `onnxruntime`, que en Windows convive mal con `onnxruntime-directml` (mismo namespace de import `onnxruntime`, un `pip install` puede desinstalar/pisar el otro).

Secuencia verificada que deja `onnxruntime-directml==1.24.4` intacto (`ort.get_available_providers()` con `DmlExecutionProvider` presente) tras terminar:

```powershell
cd C:\Users\santi\.openclaw\workspace\image-upscaler-amd
python -m venv .venv-spike
.venv-spike\Scripts\python -m pip install --upgrade pip

# 1. onnxruntime-directml primero, como paquete "protegido"
.venv-spike\Scripts\pip install onnxruntime-directml==1.24.4

# 2. optimum + optimum-onnx SIN el extra [onnxruntime] y SIN deps
#    (--no-deps evita que pip resuelva transformers/torch "óptimos" que
#    a su vez podrían arrastrar el extra onnxruntime transitivamente)
.venv-spike\Scripts\pip install --no-deps optimum==2.1.0 optimum-onnx==0.1.0

# 3. deps reales de optimum/optimum-onnx, instaladas a mano, con
#    transformers PINNEADO al rango que optimum-onnx==0.1.0 exige
#    (transformers<4.58.0,>=4.36 — ver optimum_onnx-0.1.0.dist-info/METADATA)
.venv-spike\Scripts\pip install "transformers>=4.36,<4.58" "torch>=2.2.0,<3.0.0"
.venv-spike\Scripts\pip install "diffusers>=0.30" onnx huggingface_hub packaging numpy pillow

# 4. verificar que -directml sigue intacto
.venv-spike\Scripts\python -c "import onnxruntime as ort; print(ort.__version__); print(ort.get_available_providers())"
# -> 1.24.4
# -> ['DmlExecutionProvider', 'CPUExecutionProvider']

.venv-spike\Scripts\pip check
# -> No broken requirements found.
```

**Nota importante sobre el pin de `optimum`:** el dry-run report (paso 1, con el resolver completo de pip) resolvió `optimum==2.1.0`. Pero instalar `optimum` sin versión y sin `--no-deps` resuelve la última (`optimum==2.2.0`), que es INCOMPATIBLE con `optimum-onnx==0.1.0` (`optimum-onnx` exige `optimum~=2.1.0`) y además arrastra `transformers==5.14.1` sin pin — que rompe con `optimum-onnx` (exige `transformers<4.58.0,>=4.36`) y con `torch` sin instalar. `pip check` marcó ambos conflictos exactamente. **Conclusión: pinnear `optimum==2.1.0` explícitamente, no confiar en "última versión" con `--no-deps`.**

**Nota sobre `optimum[onnxruntime]` directo (Receta B del brief):** no se probó como instalación real porque el dry-run del paso 1 ya demostró que trae `onnxruntime` vanilla — habría pisado `-directml`. Receta A (arriba) fue la única necesaria.

## (c) Clase del pipeline

Ambas clases existen e importan sin error en `optimum==2.1.0` / `optimum-onnx==0.1.0`:

```python
from optimum.onnxruntime import ORTStableDiffusionPipeline  # OK
from optimum.onnxruntime import ORTDiffusionPipeline         # OK (clase base)
```

`ORTStableDiffusionPipeline(ORTDiffusionPipeline, StableDiffusionPipeline)` — hereda `__call__` **directamente de `diffusers.StableDiffusionPipeline`**, no lo sobreescribe. Esto importa para (d): los kwargs de `__call__` son exactamente los de diffusers 0.39.0, no una API propia de optimum.

Warning cosmético al importar (no bloquea nada, confirmado con `pip check` limpio y el forward-pass real corriendo bien): `Multiple distributions found for package optimum. Picked distribution: optimum`. Viene de `diffusers/utils/import_utils.py` (`_is_package_available`), que escanea `top_level.txt`/archivos declarados de todas las distribuciones instaladas — como `optimum` y `optimum-onnx` ambas declaran archivos bajo el paquete de import `optimum`, diffusers reporta "múltiples distribuciones" para ese top-level name. Es ruido, no error.

## (d) Kwargs verificados

### `from_pretrained` (definido en `optimum/onnxruntime/modeling_diffusion.py::ORTDiffusionPipeline.from_pretrained`)

```python
ORTStableDiffusionPipeline.from_pretrained(
    model_name_or_path: str | Path,
    export: bool | None = None,                 # None = auto (exporta si no hay .onnx)
    provider: str = "CPUExecutionProvider",
    providers: Sequence[str] | None = None,
    provider_options: Sequence[dict] | dict | None = None,
    session_options: onnxruntime.SessionOptions | None = None,  # SÍ acepta session_options
    use_io_binding: bool | None = None,
    **kwargs,   # revision, cache_dir, force_download, token, slim, dtype, no_dynamic_axes, etc.
)
```

Confirmado que SÍ acepta `session_options` (pasa directo a `onnxruntime.SessionOptions`, no hay que fabricar la sesión a mano).

**`use_io_binding` — hallazgo clave para el riesgo de IOBinding+DML de este repo:** el default (`None`) resuelve a `False` para cualquier provider que NO sea `CUDAExecutionProvider` (código en `optimum/onnxruntime/base.py::ORTSessionMixin.initialize_ort_attributes`, línea ~63-72: `if use_io_binding is None: use_io_binding = True if provider == "CUDAExecutionProvider" else False`). Es decir, **con `DmlExecutionProvider` el default ya es `use_io_binding=False` sin necesidad de tocar nada** — el riesgo documentado del repo (no usar IOBinding con DML) se cumple por default en esta versión de optimum. Aun así, Tasks 4/5 deberían pasar `use_io_binding=False` explícito en el `from_pretrained` para blindarse ante un cambio de default en una versión futura de optimum.

Kwargs usados y verificados con forward-pass real:

```python
pipe = ORTStableDiffusionPipeline.from_pretrained(
    pipeline_dir,
    provider="DmlExecutionProvider",
    provider_options={"device_id": 0},
)
```

### `__call__` (heredado de `diffusers.StableDiffusionPipeline.__call__`, diffusers 0.39.0)

```python
result = pipe(
    prompt: str,
    negative_prompt: str | None = None,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    width: int | None = None,
    height: int | None = None,
    generator: torch.Generator | list[torch.Generator] | None = None,
    callback: Callable[[int, int, Tensor], None] | None = None,        # DEPRECADO, funciona con FutureWarning
    callback_steps: int | None = None,                                  # DEPRECADO, funciona con FutureWarning
    callback_on_step_end: Callable | None = None,                       # API recomendada actual
    callback_on_step_end_tensor_inputs: list[str] = ["latents"],
    ...
)
```

**`callback`/`callback_steps` SÍ funcionan** en diffusers 0.39.0 (probado empíricamente: 4 pasos → `callback steps: [1, 2, 3, 4]`), pero emiten `FutureWarning` (`deprecate(..., "1.0.0", ...)` — solo lanza `ValueError` si `diffusers.__version__ >= 1.0.0`, y 0.39.0 está lejos de eso). Task 4/5 pueden usar `callback`/`callback_steps` tal cual el brief, o migrar a `callback_on_step_end` para evitar el warning; ambas rutas son válidas hoy.

**`generator` — DESVIACIÓN IMPORTANTE respecto al script del brief:** el brief sugiere `generator=np.random.RandomState(42)`. Esto **falla** con `AttributeError: 'numpy.random.mtrand.RandomState' object has no attribute 'device'`, porque `ORTStableDiffusionPipeline.__call__` es el `__call__` nativo de diffusers (no una versión "onnx-friendly" con soporte numpy), y `diffusers.utils.torch_utils.randn_tensor` accede a `generator.device.type`. **Usar `generator=torch.Generator(device="cpu").manual_seed(42)`** — verificado, funciona sin error. (`randn_tensor` documenta que generators de CPU siempre crean el tensor en CPU, incluso si el resto del pipeline corre en `dml:0`; es el comportamiento esperado, no un bug.)

## (e) Tipo del `generator`

`torch.Generator` (no `np.random.RandomState`). Ver arriba — verificado con forward-pass real.

## Repo id de SD1.5 amd usado

El id sugerido en el brief (`amd/stable-diffusion-v1-5_io32_amdgpu`) no existe. El id real (buscado en https://huggingface.co/amd, filtrando por `author=amd, search=stable-diffusion`):

```
amd/stable-diffusion-1.5_io16_amdgpu
```

(también existe `amd/stable-diffusion-1.5_io32_amdgpu`, más pesado por incluir además un `controlnet/` de ~1.7GB y binarios `MXR/*` — MIGraphX/ROCm precompilados, irrelevantes para `onnxruntime-directml`). Se usó la variante `io16` y se excluyó `MXR/*` del download (`allow_patterns`) — el repo completo pesa ~16GB (io32) contando MXR+controlnet; excluyendo esos, `io16` pesa **~2.6 GB** (unet 1.7GB + safety_checker 608MB + text_encoder 246MB + vae_decoder/encoder ~170MB combinados). Descargado a `%TEMP%\spike-sd15` — no commiteado.

### Hallazgo adicional (bloqueante si no se corrige): el layout de archivos del repo `amd/` NO es compatible tal cual con `ORTStableDiffusionPipeline.from_pretrained`

`model_index.json` de este repo declara `_class_name: "OnnxStableDiffusionPipeline"` (la clase ONNX legacy de `diffusers`, con componentes tipados `["diffusers", "OnnxRuntimeModel"]`) — un layout más viejo que el que espera `optimum-onnx==0.1.0`. Cuando se intenta `ORTStableDiffusionPipeline.from_pretrained(pipeline_dir, provider="DmlExecutionProvider", ...)` sobre el repo tal cual se descarga:

```
ValueError: Configuration file for ORTUnet not found at ...\unet\config.json
```

`optimum-onnx`'s `ORTModelMixin.__init__` (en `optimum/onnxruntime/modeling_diffusion.py`) exige un `config.json` por subcarpeta de componente (`unet/config.json`, `text_encoder/config.json`, `vae_decoder/config.json`, `vae_encoder/config.json`, `safety_checker/config.json`) con la config de arquitectura estilo diffusers (`UNet2DConditionModel`, `CLIPTextConfig`, `AutoencoderKL`, etc.) — el repo `amd/` solo trae `model.onnx` (+ `.onnx.data`) por subcarpeta, sin esos `config.json`.

**Workaround verificado que funciona** (usado en este spike): copiar los `config.json` correspondientes desde un repo SD1.5 estándar en formato diffusers PyTorch (mismos pesos base, misma arquitectura — el export ONNX de AMD no cambia la arquitectura, solo el runtime objetivo):

```powershell
# de sd-legacy/stable-diffusion-v1-5 (sucesor oficial del runwayml/stable-diffusion-v1-5 retirado)
unet/config.json          -> <pipeline_dir>/unet/config.json
text_encoder/config.json  -> <pipeline_dir>/text_encoder/config.json
safety_checker/config.json -> <pipeline_dir>/safety_checker/config.json
vae/config.json           -> <pipeline_dir>/vae_decoder/config.json  (y copiar también a vae_encoder/config.json)
```

Con estos 5 archivos copiados (unos pocos KB cada uno, no requieren re-descargar pesos), `from_pretrained` carga OK con solo warnings no bloqueantes (`time_cond_proj_dim` faltante, `scaling_factor` faltante — atributos opcionales de exports viejos, diffusers rellena con `None`/defaults) y el forward-pass corre normal.

**Implicación para Tasks 4/5:** si el pipeline de producción usa directamente los pesos `amd/stable-diffusion-1.5_io16_amdgpu` (o `io32`), hay que empaquetar/generar estos 5 `config.json` como parte del setup/instalación (son pequeños, estáticos, se pueden vendorizar en el repo o generar en `setup.ps1`/`download-*.ps1`). Alternativa: buscar si existe un export SD1.5 ONNX ya en layout moderno de `optimum` (con `config.json` por componente) para evitar el parche manual — no se investigó a fondo por estar fuera del alcance de este spike (el GATE es sobre la compatibilidad optimum+onnxruntime-directml, no sobre qué repo de pesos usar).

## Evidencia del forward-pass real (GO)

Comando:
```powershell
.venv-spike\Scripts\python scripts\spike_optimum_directml.py C:\Users\santi\AppData\Local\Temp\spike-sd15 dml:0
```

Salida relevante:
```
onnxruntime 1.24.4, providers: ['DmlExecutionProvider', 'CPUExecutionProvider']
load: 2.4s
infer: 1.0s, callback steps: [1, 2, 3, 4]
saved ...\spike_output.png
```

- Sin crash DXGI_ERROR_NOT_FOUND ni ningún otro crash de DML.
- Callback invocado exactamente 4 veces (una por paso), confirma que `callback`/`callback_steps` funcionan con el pin de diffusers 0.39.0.
- `use_io_binding` quedó en `False` (default para `DmlExecutionProvider`, ver sección (d)) — consistente con la restricción dura del repo.
- Imagen de 256×256 generada correctamente (`std=16.8`, no plana/NaN) pero de baja fidelidad visual (256×256 no es la resolución nativa de SD1.5). Prueba adicional (no parte del script commiteado, solo para verificación) a 512×512 con 20 pasos produjo una silueta de manzana claramente reconocible sobre una mesa de madera — confirma que el pipeline es semánticamente correcto; usar 512×512 en Tasks 4/5, no 256×256.
- `pip check` limpio en el venv final: `No broken requirements found`.

## Confirmaciones adicionales del brief

- **`import_utils.py` de optimum:** no fue necesario inspeccionarlo — el import de `ORTStableDiffusionPipeline`/`ORTDiffusionPipeline` funcionó directo con la receta de instalación segura, sin ningún mensaje tipo "onnxruntime is not installed". El riesgo se manifestó en la fase de `pip install`, no en el import de Python.
- **torch:** import-requerido (confirmado por `import torch` a nivel de módulo en `modeling_diffusion.py`), sin conflicto de versión con el pin del repo (`torch>=2.2.0,<3.0.0`).

## Archivos de este spike

- `scripts/spike_optimum_directml.py` — commiteado. Difiere del snippet literal del brief en: (1) `generator=torch.Generator(device="cpu").manual_seed(42)` en vez de `np.random.RandomState(42)` (ver sección (d)), (2) `import torch` en vez de `import numpy as np` (numpy ya no se usa directamente).
- `.venv-spike/` — NO commiteado, borrado al final (`Remove-Item -Recurse -Force .venv-spike`).
- `%TEMP%\spike-sd15` — NO commiteado, pesos descargados (~2.6GB), queda en TEMP fuera del repo.
- `spike_output.png`, `spike_output_20steps.png`, `spike_output_512.png` — NO commiteados. `.gitignore` no los cubre (solo ignora `.venv/`); se borraron manualmente antes del commit (`git add` solo agregó los dos archivos listados arriba explícitamente, nunca `git add -A`).
