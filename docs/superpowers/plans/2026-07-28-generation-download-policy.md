# Descarga sin techo, aviso de pre-flight y precisión elegible — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminar el cap de descarga de modelos de generación, reemplazarlo por un pre-flight informativo que nunca bloquea, dejar elegir la precisión a bajar y exportar, y mostrar modelos por popularidad con badges de compatibilidad detectada.

**Architecture:** Tres módulos puros nuevos (`generation_variants`, `generation_compat`, `vram_estimate`) que no tocan red ni disco, más un módulo de composición (`generation_preflight`) que los junta con la sonda de VRAM ya mergeada y `shutil.disk_usage`. El pre-flight devuelve **hechos medidos, no veredictos**: comparar pico estimado contra VRAM libre es derivación pura y vive en el frontend, así cambiar de precisión no re-consulta el servidor.

**Tech Stack:** Python 3.11, FastAPI, pydantic v2, pytest + pytest-asyncio, httpx (mockeable por `transport`), React 18 + TanStack Query + vitest, optimum-onnx 0.1.0.

**Spec:** `docs/superpowers/specs/2026-07-28-generation-download-policy-design.md`

## Global Constraints

- Idioma de los comentarios de código: español cuando explican un POR QUÉ no obvio; nombres de símbolos en inglés. Sin docstrings de documentación redundante (ver `CLAUDE.md`).
- Los schemas de respuesta usan `serialization_alias` en camelCase; los de request usan `alias` + `populate_by_name=True`. Patrón establecido en `app/schemas.py:204-225`.
- El cap de **upscalers** (`MAX_MODEL_DOWNLOAD_MB` / `max_model_download_mb`, default 2048) **no se toca en ninguna tarea**. Los tests `tests/test_hf_client.py:552`, `tests/test_hf_client.py:571` y `tests/test_model_installer.py:517` deben seguir pasando sin modificarse.
- `Precision = Literal["fp16", "fp32"]` se define **una sola vez**, en `app/services/generation_variants.py`, y se importa desde ahí en todos los demás módulos.
- Las sondas de recurso devuelven **MB** (`int`) o `None` cuando no se puede medir (`app/services/resource_probes.py:16`). `None` es fail-open en todo el sistema: nunca bloquea.
- La precisión elegible aplica **solo al camino de conversión**. Repos `ready_onnx` devuelven `availablePrecisions: []` y no reciben picker.
- Comandos: tests con `.\.venv\Scripts\python.exe -m pytest`, frontend con `npm test` desde `frontend/`.
- Sin `Co-Authored-By` en los mensajes de commit.

---

### Task 1: `generation_variants` — una sola variante de pesos por componente

**Files:**
- Create: `app/services/generation_variants.py`
- Test: `tests/test_generation_variants.py`

**Interfaces:**
- Consumes: `HfFile` de `app.services.hf_client` (dataclass con `.path: str` y `.size: int`).
- Produces:
  - `Precision = Literal["fp16", "fp32"]` — la definición canónica, importada por Tasks 3, 5, 6, 7.
  - `available_precisions(files: list[HfFile]) -> tuple[Precision, ...]`
  - `select_for_precision(files: list[HfFile], declared: list[str], precision: Precision) -> list[HfFile]`
  - `CONVERSION_SKIP_SUFFIXES: tuple[str, ...]` — se mueve acá desde `generation_converter.py:35-42`.

- [ ] **Step 1: Write the failing test**

Crear `tests/test_generation_variants.py`. El fixture reproduce el listado real de SD1.5 medido el 2026-07-28 (tamaños en MB, convertidos a bytes) — es el caso que originó todo el trabajo.

```python
from __future__ import annotations

import pytest

from app.services.generation_variants import available_precisions, select_for_precision
from app.services.hf_client import HfFile

MB = 1024 * 1024

# Listado real de stable-diffusion-v1-5/stable-diffusion-v1-5 (API de HF,
# 2026-07-28). Los .ckpt de la raiz se omiten: CONVERSION_SKIP_SUFFIXES ya los
# descarta y no aportan al caso que este test fija.
SD15_FILES = [
    HfFile(path="model_index.json", size=1024),
    HfFile(path="tokenizer/vocab.json", size=1 * MB),
    HfFile(path="unet/config.json", size=2048),
    HfFile(path="unet/diffusion_pytorch_model.safetensors", size=3278 * MB),
    HfFile(path="unet/diffusion_pytorch_model.non_ema.safetensors", size=3278 * MB),
    HfFile(path="unet/diffusion_pytorch_model.fp16.safetensors", size=1639 * MB),
    HfFile(path="unet/diffusion_pytorch_model.bin", size=3278 * MB),
    HfFile(path="safety_checker/model.safetensors", size=1159 * MB),
    HfFile(path="safety_checker/model.fp16.safetensors", size=579 * MB),
    HfFile(path="text_encoder/model.safetensors", size=469 * MB),
    HfFile(path="text_encoder/model.fp16.safetensors", size=234 * MB),
    HfFile(path="vae/diffusion_pytorch_model.safetensors", size=319 * MB),
    HfFile(path="vae/diffusion_pytorch_model.fp16.safetensors", size=159 * MB),
]

SD15_DECLARED = [
    "feature_extractor", "safety_checker", "scheduler",
    "text_encoder", "tokenizer", "unet", "vae",
]


def _total_mb(files: list[HfFile]) -> int:
    return sum(f.size for f in files) // MB


def test_available_precisions_detects_both_when_repo_publishes_both():
    assert available_precisions(SD15_FILES) == ("fp16", "fp32")


def test_available_precisions_returns_only_fp32_when_no_fp16_variant():
    # Caso real: Tongyi-MAI/Z-Image-Turbo publica solo la variante plana.
    files = [
        HfFile(path="model_index.json", size=1024),
        HfFile(path="unet/diffusion_pytorch_model.safetensors", size=500 * MB),
    ]
    assert available_precisions(files) == ("fp32",)


def test_available_precisions_ignores_non_ema_when_deciding():
    # .non_ema. es un checkpoint de entrenamiento: no es una precision elegible.
    files = [
        HfFile(path="model_index.json", size=1024),
        HfFile(path="unet/diffusion_pytorch_model.non_ema.safetensors", size=500 * MB),
        HfFile(path="unet/diffusion_pytorch_model.safetensors", size=500 * MB),
    ]
    assert available_precisions(files) == ("fp32",)


def test_select_fp16_picks_only_the_fp16_weight_per_component():
    selected = select_for_precision(SD15_FILES, SD15_DECLARED, "fp16")
    paths = {f.path for f in selected}
    assert "unet/diffusion_pytorch_model.fp16.safetensors" in paths
    assert "unet/diffusion_pytorch_model.safetensors" not in paths
    assert "safety_checker/model.fp16.safetensors" in paths
    assert "text_encoder/model.fp16.safetensors" in paths
    assert "vae/diffusion_pytorch_model.fp16.safetensors" in paths


def test_select_fp32_picks_only_the_plain_weight_per_component():
    selected = select_for_precision(SD15_FILES, SD15_DECLARED, "fp32")
    paths = {f.path for f in selected}
    assert "unet/diffusion_pytorch_model.safetensors" in paths
    assert "unet/diffusion_pytorch_model.fp16.safetensors" not in paths


def test_non_ema_never_selected():
    for precision in ("fp16", "fp32"):
        selected = select_for_precision(SD15_FILES, SD15_DECLARED, precision)
        assert not any(".non_ema." in f.path for f in selected)


def test_bin_dropped_when_safetensors_sibling_exists():
    selected = select_for_precision(SD15_FILES, SD15_DECLARED, "fp32")
    assert not any(f.path.endswith(".bin") for f in selected)


def test_bin_kept_when_it_is_the_only_weight_in_the_component():
    files = [
        HfFile(path="model_index.json", size=1024),
        HfFile(path="unet/diffusion_pytorch_model.bin", size=500 * MB),
    ]
    selected = select_for_precision(files, ["unet"], "fp32")
    assert [f.path for f in selected] == ["unet/diffusion_pytorch_model.bin"]


def test_model_index_never_in_selection():
    # Se descarga aparte y antes que el resto, para conocer `declared`.
    selected = select_for_precision(SD15_FILES, SD15_DECLARED, "fp16")
    assert not any(f.path == "model_index.json" for f in selected)


def test_undeclared_component_dirs_excluded():
    files = SD15_FILES + [HfFile(path="controlnet/diffusion_pytorch_model.safetensors", size=999 * MB)]
    selected = select_for_precision(files, SD15_DECLARED, "fp32")
    assert not any(f.path.startswith("controlnet/") for f in selected)


@pytest.mark.parametrize(
    ("precision", "expected_mb", "tolerance_mb"),
    [("fp16", 2611, 2), ("fp32", 5232, 2)],
)
def test_selection_totals_match_the_measured_numbers(precision, expected_mb, tolerance_mb):
    # Numeros del spec (medidos contra la API de HF el 2026-07-28). Antes de este
    # modulo la seleccion pesaba 11121 MB y reventaba el cap de 8192.
    total = _total_mb(select_for_precision(SD15_FILES, SD15_DECLARED, precision))
    assert abs(total - expected_mb) <= tolerance_mb


def test_regression_selection_is_far_below_the_old_cap():
    # El bug original: 11121 MB > 8192 MB. Este test falla si vuelve a bajar
    # multiples variantes del mismo componente.
    assert _total_mb(select_for_precision(SD15_FILES, SD15_DECLARED, "fp16")) < 3072


def test_conversion_skip_suffixes_excluded():
    files = SD15_FILES + [
        HfFile(path="v1-5-pruned.ckpt", size=7700 * MB),
        HfFile(path="unet/model.onnx", size=100 * MB),
    ]
    selected = select_for_precision(files, SD15_DECLARED, "fp16")
    assert not any(f.path.endswith((".ckpt", ".onnx")) for f in selected)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_generation_variants.py -q`
Expected: FAIL con `ModuleNotFoundError: No module named 'app.services.generation_variants'`

- [ ] **Step 3: Write minimal implementation**

Crear `app/services/generation_variants.py`:

```python
from __future__ import annotations

from typing import Literal

from app.services.hf_client import HfFile

Precision = Literal["fp16", "fp32"]

MODEL_INDEX_FILENAME = "model_index.json"

# La conversion descarga los pesos PyTorch que el installer normal EXCLUYE
# (.safetensors/.bin son la fuente del export). .ckpt/.msgpack/.h5 fuera: son
# duplicados legacy de los mismos pesos. .onnx fuera: si el repo ya trae ONNX
# no deberia estar en este camino.
CONVERSION_SKIP_SUFFIXES = (".ckpt", ".msgpack", ".h5", ".onnx", ".onnx_data", ".pb")

WEIGHT_SUFFIXES = (".safetensors", ".bin")
_FP16_MARKER = ".fp16."
# Checkpoint de entrenamiento: la inferencia no lo lee nunca. En SD1.5 son
# 3278 MB que se bajaban para nada.
_NON_EMA_MARKER = ".non_ema."


def _is_weight(path: str) -> bool:
    return path.lower().endswith(WEIGHT_SUFFIXES)


def _is_usable_weight(path: str) -> bool:
    return _is_weight(path) and _NON_EMA_MARKER not in path.lower()


def _precision_of(path: str) -> Precision:
    return "fp16" if _FP16_MARKER in path.lower() else "fp32"


def _component_of(path: str) -> str:
    return path.rsplit("/", 1)[0]


def available_precisions(files: list[HfFile]) -> tuple[Precision, ...]:
    found = {
        _precision_of(f.path)
        for f in files
        if "/" in f.path and _is_usable_weight(f.path)
    }
    return tuple(p for p in ("fp16", "fp32") if p in found)


def _pick_weight_for_component(
    weights: list[HfFile], precision: Precision
) -> HfFile:
    """Un solo archivo de pesos por componente: la precision pedida si existe,
    si no la otra. Entre empatados gana .safetensors sobre .bin."""
    preferred = [f for f in weights if _precision_of(f.path) == precision]
    candidates = preferred or weights
    return min(
        candidates,
        key=lambda f: (0 if f.path.lower().endswith(".safetensors") else 1, f.path),
    )


def select_for_precision(
    files: list[HfFile], declared: list[str], precision: Precision
) -> list[HfFile]:
    declared_set = set(declared)
    kept: list[HfFile] = []
    weights_by_component: dict[str, list[HfFile]] = {}

    for hf_file in files:
        path = hf_file.path
        lowered = path.lower()
        if path == MODEL_INDEX_FILENAME or lowered.endswith(CONVERSION_SKIP_SUFFIXES):
            continue
        if "/" not in path:
            if lowered.endswith((".json", ".txt")):
                kept.append(hf_file)
            continue
        if path.split("/", 1)[0] not in declared_set:
            continue
        if _is_weight(path):
            if _is_usable_weight(path):
                weights_by_component.setdefault(_component_of(path), []).append(hf_file)
            continue
        kept.append(hf_file)

    for weights in weights_by_component.values():
        kept.append(_pick_weight_for_component(weights, precision))
    return kept
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_generation_variants.py -q`
Expected: PASS, 13 tests

- [ ] **Step 5: Commit**

```bash
git add app/services/generation_variants.py tests/test_generation_variants.py
git commit -m "feat: seleccion de una sola variante de pesos por componente

SD1.5 bajaba 11121 MB porque la seleccion no deduplicaba variantes:
las tres versiones del mismo unet (plana, fp16, non_ema). Ahora fp16
pesa 2611 MB y fp32 5232 MB, y .non_ema. nunca se baja."
```

---

### Task 2: `generation_compat` — clasificar un repo sin tocar la red

**Files:**
- Create: `app/services/generation_compat.py`
- Test: `tests/test_generation_compat.py`

**Interfaces:**
- Consumes: nada de tasks anteriores. Entra una tupla de nombres de archivo (`siblings[].rfilename` de la respuesta de búsqueda de HF) y el campo `gated`.
- Produces:
  - `CompatVerdict = Literal["ready_onnx", "needs_conversion", "gated", "incompatible"]`
  - `classify(filenames: tuple[str, ...], gated: bool | str | None) -> tuple[CompatVerdict, str]` — devuelve veredicto y motivo legible en español.

- [ ] **Step 1: Write the failing test**

Crear `tests/test_generation_compat.py`. Los casos son repos reales verificados contra la API de HF el 2026-07-28.

```python
from __future__ import annotations

from app.services.generation_compat import classify

SD15 = (
    "model_index.json",
    "unet/diffusion_pytorch_model.safetensors",
    "vae/diffusion_pytorch_model.safetensors",
    "text_encoder/model.safetensors",
)

SDXL_ONNX = (
    "model_index.json",
    "unet/model.onnx",
    "unet/diffusion_pytorch_model.safetensors",
    "vae_decoder/model.onnx",
    "vae_decoder/diffusion_pytorch_model.safetensors",
)


def test_repo_without_model_index_is_incompatible():
    # Caso real: wikeeyang/Flux2-Klein-9B-True-V2 (15 archivos, sin model_index).
    verdict, reason = classify(("config.json", "weights.safetensors"), False)
    assert verdict == "incompatible"
    assert "model_index.json" in reason


def test_torch_only_repo_needs_conversion():
    verdict, reason = classify(SD15, False)
    assert verdict == "needs_conversion"
    assert reason


def test_repo_with_onnx_for_every_torch_component_is_ready():
    verdict, _ = classify(SDXL_ONNX, False)
    assert verdict == "ready_onnx"


def test_repo_with_partial_onnx_needs_conversion():
    # Caso real documentado en el spike: stabilityai/sdxl-turbo publica ONNX
    # para unet pero solo pesos torch para vae. Bajar solo el ONNX deja un
    # pipeline parcial.
    files = SDXL_ONNX + ("vae/diffusion_pytorch_model.safetensors",)
    verdict, _ = classify(files, False)
    assert verdict == "needs_conversion"


def test_gated_repo_wins_over_everything_else():
    # black-forest-labs/FLUX.1-dev y stabilityai/stable-diffusion-3.5-medium
    # devuelven gated="auto" en la metadata publica.
    for gated in ("auto", "manual", True):
        verdict, reason = classify(SD15, gated)
        assert verdict == "gated"
        assert "token" in reason.lower()


def test_gated_wins_even_when_model_index_is_missing():
    # Sin token no se puede saber nada mas del repo, asi que gated gana.
    verdict, _ = classify(("config.json",), "auto")
    assert verdict == "gated"


def test_gated_false_and_none_are_not_gated():
    for gated in (False, None, ""):
        verdict, _ = classify(SD15, gated)
        assert verdict != "gated"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_generation_compat.py -q`
Expected: FAIL con `ModuleNotFoundError: No module named 'app.services.generation_compat'`

- [ ] **Step 3: Write minimal implementation**

Crear `app/services/generation_compat.py`:

```python
from __future__ import annotations

from typing import Literal

CompatVerdict = Literal["ready_onnx", "needs_conversion", "gated", "incompatible"]

MODEL_INDEX_FILENAME = "model_index.json"
_TORCH_SUFFIXES = (".safetensors", ".bin")
_ONNX_SUFFIX = ".onnx"


def _top_level_dirs_with(filenames: tuple[str, ...], suffixes: tuple[str, ...]) -> set[str]:
    return {
        name.split("/", 1)[0]
        for name in filenames
        if "/" in name and name.lower().endswith(suffixes)
    }


def classify(
    filenames: tuple[str, ...], gated: bool | str | None
) -> tuple[CompatVerdict, str]:
    # gated primero y sin excepcion: sin token no se puede leer nada mas del
    # repo, asi que cualquier otro veredicto seria una conjetura.
    if gated:
        return "gated", "Repo con acceso restringido: necesita un token de Hugging Face y aceptar la licencia."

    if MODEL_INDEX_FILENAME not in filenames:
        return (
            "incompatible",
            f"No es un pipeline diffusers: falta {MODEL_INDEX_FILENAME}.",
        )

    torch_dirs = _top_level_dirs_with(filenames, _TORCH_SUFFIXES)
    onnx_dirs = _top_level_dirs_with(filenames, (_ONNX_SUFFIX,))
    missing_onnx = sorted(torch_dirs - onnx_dirs)
    if missing_onnx:
        return (
            "needs_conversion",
            "Sin ONNX propio para " + ", ".join(missing_onnx) + ": requiere conversion local.",
        )
    return "ready_onnx", "Trae ONNX para todos los componentes: se instala directo."
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_generation_compat.py -q`
Expected: PASS, 7 tests

- [ ] **Step 5: Commit**

```bash
git add app/services/generation_compat.py tests/test_generation_compat.py
git commit -m "feat: clasificacion de compatibilidad de un repo sin tocar la red

Se alimenta de siblings[].rfilename y gated, ambos presentes en la
respuesta de busqueda con full=true, asi que los badges no cuestan
un request extra por resultado."
```

---

### Task 3: `vram_estimate` — pico estimado de VRAM

**Files:**
- Create: `app/services/vram_estimate.py`
- Test: `tests/test_vram_estimate.py`

**Interfaces:**
- Consumes: nada.
- Produces: `estimate_peak_bytes(weight_bytes: int, width: int, height: int) -> int`

- [ ] **Step 1: Write the failing test**

Los tests fijan la **monotonía y el rango**, no los valores exactos: los factores son una extrapolación que se va a revisar con mediciones reales, y la suite no debe romperse cuando eso pase.

```python
from __future__ import annotations

import pytest

from app.services.vram_estimate import estimate_peak_bytes

GB = 1024**3


def test_peak_is_always_above_the_weights_themselves():
    assert estimate_peak_bytes(2 * GB, 512, 512) > 2 * GB


def test_monotonic_in_weights():
    small = estimate_peak_bytes(1 * GB, 512, 512)
    large = estimate_peak_bytes(4 * GB, 512, 512)
    assert large > small


def test_monotonic_in_resolution():
    at_512 = estimate_peak_bytes(2 * GB, 512, 512)
    at_768 = estimate_peak_bytes(2 * GB, 768, 768)
    at_1024 = estimate_peak_bytes(2 * GB, 1024, 1024)
    assert at_512 < at_768 < at_1024


def test_saturates_above_the_top_step():
    # Mas alla de 1024x1024 el factor no sigue creciendo: el escalon superior
    # es el techo tabulado.
    assert estimate_peak_bytes(2 * GB, 2048, 2048) == estimate_peak_bytes(2 * GB, 1024, 1024)


def test_clamps_below_the_bottom_step():
    assert estimate_peak_bytes(2 * GB, 64, 64) == estimate_peak_bytes(2 * GB, 512, 512)


def test_non_square_resolutions_use_pixel_count():
    # 1024x256 y 512x512 tienen los mismos pixeles: mismo factor.
    assert estimate_peak_bytes(2 * GB, 1024, 256) == estimate_peak_bytes(2 * GB, 512, 512)


def test_zero_weights_gives_zero():
    assert estimate_peak_bytes(0, 512, 512) == 0


@pytest.mark.parametrize("width,height", [(512, 512), (768, 768), (1024, 1024)])
def test_sd15_fp16_estimate_stays_in_a_plausible_range(width, height):
    # SD1.5 fp16 son ~2.6 GB de pesos. Cualquier estimacion sana cae entre
    # los pesos y 3x los pesos; el test protege de un factor absurdo.
    peak = estimate_peak_bytes(2611 * 1024 * 1024, width, height)
    assert 2611 * 1024 * 1024 < peak < 3 * 2611 * 1024 * 1024
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_vram_estimate.py -q`
Expected: FAIL con `ModuleNotFoundError: No module named 'app.services.vram_estimate'`

- [ ] **Step 3: Write minimal implementation**

```python
from __future__ import annotations

# Factores EXTRAPOLADOS, no medidos por modelo (ver el spec de 2026-07-28,
# seccion vram_estimate). El margen sobre los pesos cubre activaciones y
# buffers intermedios, que crecen con la resolucion mientras los pesos no.
# Revisar con mediciones reales; los tests fijan monotonia, no estos numeros.
_STEPS: tuple[tuple[int, float], ...] = (
    (512 * 512, 1.25),
    (768 * 768, 1.45),
    (1024 * 1024, 1.70),
)


def _factor(pixels: int) -> float:
    if pixels <= _STEPS[0][0]:
        return _STEPS[0][1]
    if pixels >= _STEPS[-1][0]:
        return _STEPS[-1][1]
    for (low_px, low_f), (high_px, high_f) in zip(_STEPS, _STEPS[1:]):
        if low_px <= pixels <= high_px:
            ratio = (pixels - low_px) / (high_px - low_px)
            return low_f + ratio * (high_f - low_f)
    return _STEPS[-1][1]


def estimate_peak_bytes(weight_bytes: int, width: int, height: int) -> int:
    return int(weight_bytes * _factor(max(width, 0) * max(height, 0)))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_vram_estimate.py -q`
Expected: PASS, 11 tests

- [ ] **Step 5: Commit**

```bash
git add app/services/vram_estimate.py tests/test_vram_estimate.py
git commit -m "feat: estimador de pico de VRAM por pesos y resolucion

HF no publica la VRAM que un modelo requiere. Una formula cubre
cualquier repo desde el dia uno; una tabla curada solo cubriria lo
curado. Los factores son extrapolacion declarada, y los tests fijan
monotonia para que revisarlos con mediciones no rompa la suite."
```

---

### Task 4: `hf_client` — sin techo, browse por popularidad, y metadata para los badges

**Files:**
- Modify: `app/services/hf_client.py:95-103` (`HfModelSummary`), `:111-119` (`_parse_model_summary`), `:210-226` (`search`), `:254-262` (`download`)
- Test: `tests/test_hf_client.py` (agregar; **no** modificar los tests del cap de upscalers)

**Interfaces:**
- Consumes: nada de tasks anteriores.
- Produces:
  - `HfModelSummary` gana `filenames: tuple[str, ...]` y `gated: bool | str | None`.
  - `HfClient.search(query: str, limit: int = 20, task_tags: tuple[str, ...] | None = None, sort: str | None = None) -> list[HfModelSummary]` — acepta `query=""`.
  - `HfClient.download(..., max_bytes: int | None = None, unlimited: bool = False)` — `unlimited=True` desactiva el techo.

- [ ] **Step 1: Write the failing test**

Agregar al final de `tests/test_hf_client.py`. Seguir el patrón de mock de `transport` que el archivo ya usa.

```python
async def test_search_with_empty_query_omits_search_param_and_sorts_by_downloads():
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        return httpx.Response(200, json=[])

    client = HfClient(_settings(), transport=httpx.MockTransport(handler))
    await client.search("", task_tags=("text-to-image",), sort="downloads")

    url = str(captured["url"])
    assert "search=" not in url
    assert "sort=downloads" in url
    assert "direction=-1" in url
    assert "filter=text-to-image" in url


async def test_search_with_query_still_sends_search_param():
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        return httpx.Response(200, json=[])

    client = HfClient(_settings(), transport=httpx.MockTransport(handler))
    await client.search("sdxl", task_tags=("text-to-image",))
    assert "search=sdxl" in str(captured["url"])


async def test_search_summary_carries_filenames_and_gated():
    payload = [
        {
            "id": "owner/name",
            "author": "owner",
            "pipeline_tag": "text-to-image",
            "downloads": 5,
            "likes": 1,
            "tags": ["text-to-image"],
            "gated": "auto",
            "siblings": [
                {"rfilename": "model_index.json"},
                {"rfilename": "unet/diffusion_pytorch_model.safetensors"},
            ],
        }
    ]
    client = HfClient(
        _settings(),
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json=payload)),
    )
    [summary] = await client.search("x", task_tags=("text-to-image",))
    assert summary.gated == "auto"
    assert summary.filenames == (
        "model_index.json",
        "unet/diffusion_pytorch_model.safetensors",
    )


async def test_search_summary_defaults_when_siblings_and_gated_absent():
    payload = [{"id": "owner/name", "downloads": 0, "likes": 0, "tags": []}]
    client = HfClient(
        _settings(),
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json=payload)),
    )
    [summary] = await client.search("x")
    assert summary.filenames == ()
    assert summary.gated is None


async def test_download_unlimited_does_not_abort_a_stream_over_the_cap(tmp_path: Path):
    # El cap de generacion se elimino: una descarga de 3 GB (mas que cualquier
    # techo configurado) tiene que completarse. Este es el test que fija que el
    # techo se fue; se usa un payload chico pero con Content-Length declarado
    # muy por encima del cap para ejercitar los dos guards.
    body = b"x" * 4096
    settings = _settings()
    settings.max_model_download_mb = 1  # techo bajo a proposito

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=body, headers={"Content-Length": str(len(body))})

    client = HfClient(settings, transport=httpx.MockTransport(handler))
    dest = tmp_path / "weights.safetensors"
    await client.download("owner/name", "weights.safetensors", dest, unlimited=True)
    assert dest.read_bytes() == body


async def test_download_without_unlimited_still_enforces_the_upscaler_cap(tmp_path: Path):
    settings = _settings()
    settings.max_model_download_mb = 0
    client = HfClient(
        settings,
        transport=httpx.MockTransport(lambda request: httpx.Response(200, content=b"xx")),
    )
    with pytest.raises(HfDownloadTooLargeError):
        await client.download("owner/name", "f.onnx", tmp_path / "f.onnx")
```

Nota: `_settings()` es el helper que el archivo ya define para construir un `Settings` de test. Si su nombre real difiere, usar el que esté en uso — no crear uno nuevo.

- [ ] **Step 2: Run test to verify it fails**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_hf_client.py -q -k "empty_query or filenames_and_gated or unlimited or siblings_and_gated"`
Expected: FAIL — `search()` no acepta `sort`, `HfModelSummary` no tiene `filenames`, `download()` no acepta `unlimited`.

- [ ] **Step 3: Write minimal implementation**

En `app/services/hf_client.py`, cambiar `HfModelSummary` (línea 95):

```python
@dataclass(slots=True, frozen=True)
class HfModelSummary:
    id: str
    author: str | None
    pipeline_tag: str | None
    downloads: int
    likes: int
    tags: tuple[str, ...]
    # `full=true` trae siblings y gated en la MISMA respuesta de busqueda, asi
    # que la clasificacion de compatibilidad no cuesta un request por resultado.
    # siblings sin blobs=true solo trae rfilename (sin tamanos) -- alcanza para
    # los badges; los tamanos los pide el pre-flight aparte.
    filenames: tuple[str, ...] = ()
    gated: bool | str | None = None
```

`_parse_model_summary` (línea 111):

```python
def _parse_model_summary(item: dict) -> HfModelSummary:
    siblings = item.get("siblings") or []
    return HfModelSummary(
        id=item["id"],
        author=item.get("author"),
        pipeline_tag=item.get("pipeline_tag"),
        downloads=item.get("downloads", 0),
        likes=item.get("likes", 0),
        tags=tuple(item.get("tags", [])),
        filenames=tuple(s["rfilename"] for s in siblings if "rfilename" in s),
        gated=item.get("gated"),
    )
```

`search` (línea 210):

```python
    async def search(
        self,
        query: str,
        limit: int = 20,
        task_tags: tuple[str, ...] | None = None,
        sort: str | None = None,
    ) -> list[HfModelSummary]:
        tags = SEARCH_TASK_TAGS if task_tags is None else task_tags
        params: dict[str, object] = {
            "filter": list(tags),
            "limit": limit,
            "full": "true",
        }
        # Query vacia = browse: el Hub devuelve los mas descargados del tag sin
        # `search`. Mandar search="" filtraria por cadena vacia, no es lo mismo.
        if query:
            params["search"] = query
        if sort:
            params["sort"] = sort
            params["direction"] = -1
        async with self._build_client() as client:
            response = await client.get(
                f"{HF_API_BASE}/models", params=params, headers=self._auth_headers()
            )
            response.raise_for_status()
            payload = response.json()
        return [_parse_model_summary(item) for item in payload]
```

`download` (línea 254) — agregar el parámetro `unlimited` y hacer que los guards lo respeten:

```python
    async def download(
        self,
        repo_id: str,
        filename: str,
        dest: Path,
        progress_cb: ProgressCallback | None = None,
        max_bytes: int | None = None,
        unlimited: bool = False,
    ) -> Path:
        url = _download_url(repo_id, filename)
        _validate_https_huggingface_host(url)
        # unlimited: los modelos de generacion no tienen techo (ver spec
        # 2026-07-28). None distinto de unlimited: None sigue cayendo al cap
        # de upscalers, que NO se toca.
        effective_max: int | None
        if unlimited:
            effective_max = None
        elif max_bytes is None:
            effective_max = self.settings.max_model_download_mb * 1024 * 1024
        else:
            effective_max = max_bytes
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = dest.with_name(f"{dest.name}{PARTIAL_DOWNLOAD_SUFFIX}")

        for attempt in range(1, DOWNLOAD_ATTEMPTS + 1):
            try:
                await self._stream_to_file(url, tmp_path, effective_max, progress_cb)
                break
            except Exception as exc:  # noqa: BLE001 -- CancelledError is BaseException, so cancel still propagates
                tmp_path.unlink(missing_ok=True)
                if attempt == DOWNLOAD_ATTEMPTS or not _is_retryable_download_error(exc):
                    raise _wrap_hf_auth_error(exc, repo_id) from exc
                logger.warning(
                    "Hugging Face download attempt %d/%d failed (%s); retrying",
                    attempt,
                    DOWNLOAD_ATTEMPTS,
                    type(exc).__name__,
                )
                await asyncio.sleep(2 ** (attempt - 1))

        tmp_path.replace(dest)
        return dest
```

Y hacer que los guards acepten `None` = sin techo. `_reject_declared_size_over_limit` (línea 162) y `_write_response_to_file` (línea 169), más `_stream_to_file` (línea 286), cambian su parámetro a `int | None`:

```python
def _reject_declared_size_over_limit(total: int | None, max_bytes: int | None) -> None:
    if max_bytes is None or total is None:
        return
    if total > max_bytes:
        raise HfDownloadTooLargeError(
            f"Declared size {total} bytes exceeds MAX_MODEL_DOWNLOAD_MB limit ({max_bytes} bytes)"
        )
```

```python
async def _write_response_to_file(
    response: httpx.Response,
    tmp_path: Path,
    max_bytes: int | None,
    total: int | None,
    progress_cb: ProgressCallback | None,
) -> None:
    downloaded = 0
    async with aiofiles.open(tmp_path, "wb") as handle:
        async for chunk in response.aiter_bytes(DOWNLOAD_CHUNK_BYTES):
            downloaded += len(chunk)
            if max_bytes is not None and downloaded > max_bytes:
                raise HfDownloadTooLargeError(
                    f"Download exceeds MAX_MODEL_DOWNLOAD_MB limit ({max_bytes} bytes)"
                )
            await handle.write(chunk)
            if progress_cb is not None:
                progress_cb(downloaded, total)
```

En `_stream_to_file`, cambiar la anotación de `max_bytes: int` a `max_bytes: int | None`. El cuerpo no cambia.

- [ ] **Step 4: Run the full hf_client suite**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_hf_client.py -q`
Expected: PASS. Los tests preexistentes del cap de upscalers (`:552`, `:571`, `:610`) siguen verdes sin tocarse.

- [ ] **Step 5: Commit**

```bash
git add app/services/hf_client.py tests/test_hf_client.py
git commit -m "feat: descargas sin techo, browse por popularidad y metadata de compat

download(unlimited=True) desactiva el techo para los modelos de
generacion; sin el flag sigue cayendo al cap de upscalers, que no se
toca. search() acepta query vacia con sort=downloads (browse) y expone
siblings+gated, que ya venian en la respuesta con full=true."
```

---

### Task 5: Eliminar el cap de generación y hacer accionable el disco lleno

**Files:**
- Modify: `app/config.py:359` (borrar `max_generation_model_download_mb`)
- Modify: `app/services/generation_installer.py:178-184` (borrar `_ensure_size_cap`), `:304-326` (usar `unlimited=True`), `:317` (borrar la llamada al cap)
- Modify: `app/services/generation_converter.py:245-271` (idem)
- Modify: `app/exceptions.py` (agregar el mapeo de ENOSPC) — verificar el nombre real del módulo de excepciones antes de editar
- Modify: `tests/test_hf_client.py:594` (borrar la aserción del setting eliminado)
- Test: `tests/test_generation_installer.py`, `tests/test_generation_converter.py`

**Interfaces:**
- Consumes: `HfClient.download(..., unlimited=True)` de Task 4.
- Produces: `map_disk_full(exc: OSError) -> str | None` en `app/services/generation_installer.py` — devuelve un mensaje accionable cuando `exc.errno == errno.ENOSPC`, `None` en cualquier otro caso.

- [ ] **Step 1: Write the failing test**

Agregar a `tests/test_generation_installer.py`:

```python
import errno

from app.services.generation_installer import map_disk_full


def test_map_disk_full_returns_actionable_message_for_enospc(tmp_path):
    exc = OSError(errno.ENOSPC, "No space left on device")
    exc.filename = str(tmp_path / "unet" / "model.safetensors")
    message = map_disk_full(exc)
    assert message is not None
    assert "espacio" in message.lower()


def test_map_disk_full_ignores_other_oserrors():
    assert map_disk_full(OSError(errno.EACCES, "Permission denied")) is None


def test_generation_settings_no_longer_expose_a_download_cap():
    from app.config import Settings

    assert not hasattr(Settings(), "max_generation_model_download_mb")
```

Y un test que fija que ningún caller del cap sobrevive:

```python
def test_no_size_cap_remains_in_the_generation_paths():
    from app.services import generation_converter, generation_installer

    assert not hasattr(generation_installer, "_ensure_size_cap")
    assert not hasattr(generation_converter, "_ensure_size_cap")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_generation_installer.py -q -k "disk_full or download_cap or size_cap"`
Expected: FAIL — `map_disk_full` no existe y `_ensure_size_cap` sí.

- [ ] **Step 3: Write minimal implementation**

En `app/config.py`, borrar la línea 359 completa:

```python
    max_generation_model_download_mb: int = Field(default=8192, alias="MAX_GENERATION_MODEL_DOWNLOAD_MB")
```

En `app/services/generation_installer.py`, borrar `_ensure_size_cap` (líneas 178-184) y agregar:

```python
import errno


def map_disk_full(exc: OSError) -> str | None:
    """Sin cap de descarga, quedarse sin disco a mitad de una bajada es EL modo
    de falla del sistema. Un OSError crudo en el job no le dice nada a nadie."""
    if exc.errno != errno.ENOSPC:
        return None
    target = getattr(exc, "filename", None)
    where = f" en {target}" if target else ""
    return (
        f"No queda espacio en disco{where}. Liberá espacio y volvé a intentar; "
        "la descarga parcial ya se limpió."
    )
```

En `_download_and_register` (línea 304), borrar `max_file_bytes` y usar `unlimited=True` en las dos llamadas a `download`, y borrar la llamada a `_ensure_size_cap` de la línea 317. La selección pasa a `generation_variants`:

```python
        job.status = InstallStatus.downloading
        try:
            # Fase 1: model_index.json primero (KBs) para conocer los componentes
            # declarados y filtrar la descarga a lo que el pipeline realmente usa.
            await self.hf_client.download(
                job.repo_id,
                MODEL_INDEX_FILENAME,
                _safe_staging_dest(staging_root, MODEL_INDEX_FILENAME),
                unlimited=True,
            )
            declared = _read_declared_components(staging_root)
            selected = _filter_to_declared(_select_files(files), declared)

            total_bytes = sum(f.size for f in selected) or 1
            downloaded_bytes = 0
            for hf_file in selected:
                dest = _safe_staging_dest(staging_root, hf_file.path)
                dest.parent.mkdir(parents=True, exist_ok=True)
                await self.hf_client.download(
                    job.repo_id, hf_file.path, dest, unlimited=True
                )
                downloaded_bytes += hf_file.size
                job.progress_pct = round(downloaded_bytes / total_bytes * 100, 1)
```

En `_run_install` (línea 273), traducir el disco lleno:

```python
    async def _run_install(self, job: InstallJob) -> None:
        try:
            await self._download_and_register(job)
        except OSError as exc:
            job.status = InstallStatus.error
            job.error = map_disk_full(exc) or str(exc)
        except Exception as exc:  # noqa: BLE001 - el job reporta cualquier fallo
            job.status = InstallStatus.error
            job.error = str(exc)
```

En `app/services/generation_converter.py`, borrar el import de `_ensure_size_cap` (línea 19), borrar `max_file_bytes` (líneas 245-247) y su uso, y aplicar el mismo `except OSError` en `_run_conversion`:

```python
    async def _run_conversion(self, job: ConversionJob) -> None:
        job.status = JobStatus.running
        job.started_at = utc_now()
        try:
            await self._convert_and_register(job)
            job.status = JobStatus.completed
        except OSError as exc:
            job.status = JobStatus.failed
            job.error = map_disk_full(exc) or str(exc)
        except Exception as exc:  # noqa: BLE001 - el job reporta cualquier fallo
            job.status = JobStatus.failed
            job.error = str(exc)
        finally:
            job.finished_at = utc_now()
```

Agregar `map_disk_full` al import desde `generation_installer` en `generation_converter.py`.

Borrar de `tests/test_hf_client.py` la línea 594 (`assert settings.max_generation_model_download_mb == 8192`) y, si queda un test vacío, el test entero.

Quitar la fila del cap de `.env.example` y de la tabla de variables del `README.md`.

- [ ] **Step 4: Run the full suite**

Run: `.\.venv\Scripts\python.exe -m pytest -q`
Expected: PASS. Ningún test menciona ya `max_generation_model_download_mb`.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat: elimina el cap de descarga de modelos de generacion

MAX_GENERATION_MODEL_DOWNLOAD_MB=8192 estaba dimensionado contra la
maquina del autor y bloqueaba a cualquiera con mas disco. Se elimina sin
reemplazo: el aviso de pre-flight informa, no bloquea.

Consecuencia atendida: quedarse sin disco a mitad de descarga pasa a ser
el modo de falla real, asi que ENOSPC se traduce a un mensaje accionable
en vez de subir un OSError crudo al job. El parcial ya se limpiaba solo
en el finally.

El cap de upscalers (MAX_MODEL_DOWNLOAD_MB) no se toca."
```

---

### Task 6: Precisión elegible — staging canónico y `dtype` en el export

**Files:**
- Modify: `app/services/generation_converter.py:51-82` (borrar `_select_conversion_files`, usar `generation_variants`), `:110-143` (`_export_with_optimum` acepta `dtype`), `:194-202` (`convert_from_hf` acepta `precision`), `:234-271` (staging canónico)
- Modify: `app/services/generation_installer.py` (`install_from_hf` acepta `precision` y la pasa a `enqueue_conversion`)
- Test: `tests/test_generation_converter.py`

**Interfaces:**
- Consumes: `Precision`, `select_for_precision`, `available_precisions` de Task 1.
- Produces:
  - `canonical_weight_name(path: str) -> str` en `generation_variants.py` — quita el marcador `.fp16.` del nombre de archivo.
  - `GenerationModelConverter.convert_from_hf(repo_id: str, precision: Precision = "fp16") -> str`
  - `GenerationModelInstaller.install_from_hf(repo_id: str, precision: Precision = "fp16") -> str`
  - `_export_with_optimum(src_dir, out_dir, on_component, dtype: str | None = None)`

- [ ] **Step 1: Write the failing test**

Primero el helper puro, en `tests/test_generation_variants.py`:

```python
def test_canonical_weight_name_strips_the_fp16_marker():
    from app.services.generation_variants import canonical_weight_name

    assert (
        canonical_weight_name("unet/diffusion_pytorch_model.fp16.safetensors")
        == "unet/diffusion_pytorch_model.safetensors"
    )


def test_canonical_weight_name_leaves_plain_names_untouched():
    from app.services.generation_variants import canonical_weight_name

    for path in (
        "unet/diffusion_pytorch_model.safetensors",
        "tokenizer/vocab.json",
        "model_index.json",
    ):
        assert canonical_weight_name(path) == path
```

Y en `tests/test_generation_converter.py`, los dos comportamientos del camino de conversión. El archivo ya define `make_converter(tmp_path, export_fn) -> (converter, installer, settings, registry)`, `_pytorch_repo_files()` y `SOURCE_MODEL_INDEX`, e importa `FakeHfClient`/`make_settings` desde `test_generation_installer`. Se reusan tal cual; solo hace falta un repo con variante fp16, que hoy no existe:

```python
def _pytorch_repo_files_with_fp16() -> list[HfFile]:
    return _pytorch_repo_files() + [
        HfFile(path="unet/diffusion_pytorch_model.fp16.safetensors", size=500),
        HfFile(path="vae/diffusion_pytorch_model.fp16.safetensors", size=250),
    ]


def _recording_export(captured: dict[str, object]):
    def fake_export(src_dir, out_dir, on_component, dtype=None, atol=None):
        captured["dtype"] = dtype
        captured["atol"] = atol
        (out_dir / "model_index.json").write_text(SOURCE_MODEL_INDEX, encoding="utf-8")
        for component in ("unet", "vae"):
            (out_dir / component).mkdir(parents=True, exist_ok=True)
            (out_dir / component / "model.onnx").write_bytes(b"onnx")
        return ["unet", "vae"]

    return fake_export


@pytest.mark.asyncio
async def test_fp16_weights_are_staged_under_the_canonical_name(tmp_path: Path) -> None:
    # main_export no expone un selector de variante: la eleccion se materializa
    # renombrando en el staging. Verificado en el smoke del 2026-07-28.
    captured: dict[str, object] = {}
    converter, _installer, _settings, _registry = make_converter(
        tmp_path, _recording_export(captured)
    )
    converter.hf_client.files = _pytorch_repo_files_with_fp16()
    staged: list[str] = []
    original_download = converter.hf_client.download

    async def recording_download(repo_id, filename, dest, progress_cb=None,
                                 max_bytes=None, unlimited=False):
        staged.append(dest.name)
        return await original_download(
            repo_id, filename, dest, progress_cb, max_bytes, unlimited
        )

    converter.hf_client.download = recording_download  # type: ignore[method-assign]

    conversion_id = await converter.convert_from_hf("owner/name", precision="fp16")
    await converter._process_next()

    assert converter.status(conversion_id).status is JobStatus.completed
    assert "diffusion_pytorch_model.safetensors" in staged
    assert not any(".fp16." in name for name in staged)


@pytest.mark.asyncio
async def test_export_receives_fp16_dtype_and_relaxed_atol(tmp_path: Path) -> None:
    # Medido en el smoke del 2026-07-28: dtype="fp16" produce initializers
    # FLOAT16 pero su validacion falla con el atol por defecto por redondeo
    # esperable (unet: 0.00390625 contra 1e-05). atol=1e-2 la pasa sin apagarla.
    captured: dict[str, object] = {}
    converter, *_ = make_converter(tmp_path, _recording_export(captured))
    converter.hf_client.files = _pytorch_repo_files_with_fp16()

    await converter.convert_from_hf("owner/name", precision="fp16")
    await converter._process_next()

    assert captured["dtype"] == "fp16"
    assert captured["atol"] == pytest.approx(1e-2)


@pytest.mark.asyncio
async def test_fp32_export_passes_no_dtype_and_no_atol_override(tmp_path: Path) -> None:
    captured: dict[str, object] = {}
    converter, *_ = make_converter(tmp_path, _recording_export(captured))
    converter.hf_client.files = _pytorch_repo_files_with_fp16()

    await converter.convert_from_hf("owner/name", precision="fp32")
    await converter._process_next()

    assert captured["dtype"] is None
    assert captured["atol"] is None


@pytest.mark.asyncio
async def test_precision_falls_back_when_the_repo_lacks_the_requested_variant(
    tmp_path: Path,
) -> None:
    # Repo sin fp16 (caso real: Tongyi-MAI/Z-Image-Turbo). Pedir fp16 no debe
    # fallar: se cae a la unica precision publicada.
    captured: dict[str, object] = {}
    converter, *_ = make_converter(tmp_path, _recording_export(captured))
    converter.hf_client.files = _pytorch_repo_files()

    conversion_id = await converter.convert_from_hf("owner/name", precision="fp16")
    await converter._process_next()

    assert converter.status(conversion_id).status is JobStatus.completed
    assert captured["dtype"] is None
```

Nota: `_select_conversion_files` está importado en la línea 15 de ese archivo y hay tests que lo ejercitan. Al borrarlo en el Step 3, esos tests se migran a `tests/test_generation_variants.py` (donde Task 1 ya cubre las mismas reglas) y se quita el import.

- [ ] **Step 2: Run test to verify it fails**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_generation_converter.py tests/test_generation_variants.py -q -k "canonical or dtype or staged_under"`
Expected: FAIL — `canonical_weight_name` no existe, el export no recibe `dtype`.

- [ ] **Step 3: Write minimal implementation**

Agregar a `app/services/generation_variants.py`:

```python
def canonical_weight_name(path: str) -> str:
    """Quita el marcador de variante del nombre. main_export no expone un
    selector de variante, asi que la eleccion se materializa escribiendo el
    archivo elegido bajo el nombre que diffusers busca. safetensors lleva el
    dtype en su propio header, asi que el rename no pierde informacion."""
    if not _is_weight(path):
        return path
    directory, _, name = path.rpartition("/")
    replaced = name.replace(_FP16_MARKER, ".", 1) if _FP16_MARKER in name.lower() else name
    return f"{directory}/{replaced}" if directory else replaced
```

En `app/services/generation_converter.py`, borrar `_select_conversion_files` (líneas 51-82) y `CONVERSION_SKIP_SUFFIXES`/`SAFETENSORS_SUFFIX`/`TORCH_BIN_SUFFIX` (35-44), e importar desde `generation_variants`:

```python
from app.services.generation_variants import (
    Precision,
    available_precisions,
    canonical_weight_name,
    select_for_precision,
)

# Medido en el smoke del 2026-07-28: dtype="fp16" produce un ONNX con
# initializers FLOAT16, pero la validacion de forward-pass falla con el atol
# por defecto por ruido de redondeo esperable (unet: 0.00390625 contra 1e-05).
# 1e-2 la pasa CONSERVANDOLA -- preferido sobre do_validation=False, que
# tambien funciona pero pierde la deteccion de un export realmente roto.
FP16_EXPORT_ATOL = 1e-2
```

`_export_with_optimum` acepta y reenvía `dtype`/`atol`:

```python
def _export_with_optimum(
    src_dir: Path,
    out_dir: Path,
    on_component: Callable[[str], None],
    dtype: str | None = None,
    atol: float | None = None,
) -> list[str]:
    from optimum.exporters.onnx import main_export
    from optimum.utils import logging as optimum_logging

    seen: list[str] = []
    handler = _SubmodelProgressHandler(on_component, seen)
    optimum_logger = logging.getLogger("optimum")
    previous_verbosity = optimum_logging.get_verbosity()
    optimum_logger.addHandler(handler)
    extra: dict[str, Any] = {}
    if dtype is not None:
        extra["dtype"] = dtype
    if atol is not None:
        extra["atol"] = atol
    try:
        optimum_logging.set_verbosity_info()
        with _patch(
            "optimum.exporters.onnx.base.is_onnxruntime_available",
            return_value=True,
        ):
            main_export(
                str(src_dir),
                str(out_dir),
                task="text-to-image",
                device="cpu",
                **extra,
            )
    finally:
        optimum_logger.removeHandler(handler)
        optimum_logging.set_verbosity(previous_verbosity)
    return seen or ["pipeline"]
```

Actualizar el alias de tipo `ExportFn` para la firma nueva.

`convert_from_hf` y `_convert_and_register` aceptan `precision`; `ConversionJob` guarda la elección. En `_convert_and_register`, la selección y el staging:

```python
            declared = _read_declared_components(src_root)
            offered = available_precisions(files)
            precision: Precision = job.precision if job.precision in offered else (offered[0] if offered else "fp32")
            selected = select_for_precision(files, declared, precision)
            for hf_file in selected:
                dest = _safe_staging_dest(src_root, canonical_weight_name(hf_file.path))
                dest.parent.mkdir(parents=True, exist_ok=True)
                await self.hf_client.download(
                    job.repo_id, hf_file.path, dest, unlimited=True
                )
```

Y el export:

```python
            exported = await asyncio.to_thread(
                self.export_fn,
                src_root,
                out_root,
                on_component,
                "fp16" if precision == "fp16" else None,
                FP16_EXPORT_ATOL if precision == "fp16" else None,
            )
```

En `generation_installer.py`, `install_from_hf(repo_id, precision="fp16")` guarda la precisión en el `InstallJob` y la reenvía al encolar la conversión: `self.enqueue_conversion(job.repo_id, job.precision)`. Ajustar el tipo del callable `enqueue_conversion` y su cableado en `app/main.py`.

- [ ] **Step 4: Run the tests**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_generation_converter.py tests/test_generation_installer.py tests/test_generation_variants.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat: precision elegible en la conversion de modelos de generacion

Dos mecanismos, los dos necesarios y medidos en el smoke del 2026-07-28:
el staging bajo nombre canonico da el ahorro de descarga (main_export no
expone selector de variante), y dtype='fp16' da la precision del ONNX.
Cargar pesos fp16 sin dtype produce un ONNX fp32, asi que el staging solo
no alcanza.

dtype='fp16' va con atol=1e-2: con el atol por defecto la validacion
falla por redondeo esperable. Se relaja la tolerancia en vez de apagar
la validacion."
```

---

### Task 7: `generation_preflight` y su endpoint

**Files:**
- Create: `app/services/generation_preflight.py`
- Modify: `app/schemas.py` (agregar los schemas de respuesta después de `ModelSearchResponse`, línea 213)
- Modify: `app/api/routes.py` (ruta nueva junto a las de generación, después de la línea 1280)
- Test: `tests/test_generation_preflight.py`

**Interfaces:**
- Consumes: `available_precisions`/`select_for_precision` (Task 1), `classify` (Task 2), `estimate_peak_bytes` (Task 3), `HfClient.repo_files`/`download` (Task 4).
- Produces: `async def preflight(hf_client, devices_service, settings, probes, repo_id: str, width: int = 512, height: int = 512) -> PreflightReport` — dataclass con los campos del spec.

- [ ] **Step 1: Write the failing test**

```python
from __future__ import annotations

import pytest

from app.services.generation_preflight import preflight
from app.services.hf_client import HfFile

MB = 1024 * 1024


class FakeHf:
    def __init__(self, files, index=None, fail=False):
        self._files, self._index, self._fail = files, index or {}, fail

    async def repo_files(self, repo_id):
        if self._fail:
            raise RuntimeError("HF caido")
        return self._files

    async def download(self, repo_id, filename, dest, progress_cb=None,
                       max_bytes=None, unlimited=False):
        import json
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(self._index), encoding="utf-8")
        return dest


class FakeDevices:
    def __init__(self, devices):
        self._devices = devices

    def list_devices(self):
        return self._devices


class FakeProbe:
    def __init__(self, by_id):
        self._by_id = by_id

    def free_capacity_mb(self, device_id):
        return self._by_id.get(device_id)


FILES = [
    HfFile(path="model_index.json", size=1024),
    HfFile(path="unet/diffusion_pytorch_model.safetensors", size=3278 * MB),
    HfFile(path="unet/diffusion_pytorch_model.fp16.safetensors", size=1639 * MB),
]
INDEX = {"_class_name": "StableDiffusionPipeline", "unet": ["diffusers", "UNet2DConditionModel"]}
DEVICES = [
    {"id": "dml:0", "kind": "gpu", "name": "RX 7900 XTX", "backend": "directml"},
    {"id": "dml:1", "kind": "gpu", "name": "RX 6600", "backend": "directml"},
    {"id": "cpu", "kind": "cpu", "name": "CPU", "backend": "cpu"},
]


async def test_report_has_one_row_per_enumerated_device(tmp_path, settings_factory):
    report = await preflight(
        hf_client=FakeHf(FILES, INDEX),
        devices_service=FakeDevices(DEVICES),
        settings=make_settings(tmp_path),
        probes={"gpu": FakeProbe({"dml:0": 23700, "dml:1": 7400}), "cpu": FakeProbe({"cpu": 16000})},
        repo_id="owner/name",
    )
    assert [d.id for d in report.devices] == ["dml:0", "dml:1", "cpu"]
    assert report.devices[0].free_vram_bytes == 23700 * MB
    assert report.devices[1].free_vram_bytes == 7400 * MB


async def test_report_prices_every_available_precision(tmp_path, settings_factory):
    report = await preflight(
        hf_client=FakeHf(FILES, INDEX),
        devices_service=FakeDevices(DEVICES),
        settings=make_settings(tmp_path),
        probes={},
        repo_id="owner/name",
    )
    by_precision = {p.precision: p for p in report.precisions}
    assert set(by_precision) == {"fp16", "fp32"}
    assert by_precision["fp16"].download_bytes < by_precision["fp32"].download_bytes
    assert by_precision["fp16"].estimated_peak_bytes > by_precision["fp16"].download_bytes


async def test_unmeasurable_probe_yields_null_not_zero(tmp_path, settings_factory):
    report = await preflight(
        hf_client=FakeHf(FILES, INDEX),
        devices_service=FakeDevices(DEVICES),
        settings=make_settings(tmp_path),
        probes={"gpu": FakeProbe({})},
        repo_id="owner/name",
    )
    assert report.devices[0].free_vram_bytes is None


async def test_hf_failure_degrades_instead_of_raising(tmp_path, settings_factory):
    report = await preflight(
        hf_client=FakeHf(FILES, INDEX, fail=True),
        devices_service=FakeDevices(DEVICES),
        settings=make_settings(tmp_path),
        probes={},
        repo_id="owner/name",
    )
    assert report.degraded is True
    assert report.precisions == []
    # Los dispositivos siguen informandose: no dependen de HF.
    assert [d.id for d in report.devices] == ["dml:0", "dml:1", "cpu"]


async def test_disk_free_bytes_reported(tmp_path, settings_factory):
    report = await preflight(
        hf_client=FakeHf(FILES, INDEX),
        devices_service=FakeDevices(DEVICES),
        settings=make_settings(tmp_path),
        probes={},
        repo_id="owner/name",
    )
    assert report.disk is not None
    assert report.disk.free_bytes > 0
```

Nota: `make_settings` se importa de `test_generation_installer` (`from test_generation_installer import make_settings`), igual que hace `tests/test_generation_converter.py:24-28`. Construye `Settings(RUNTIME_DIR=str(tmp_path), _env_file=None)`. Los tests son `async`, así que llevan `@pytest.mark.asyncio` como el resto del repo.

- [ ] **Step 2: Run test to verify it fails**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_generation_preflight.py -q`
Expected: FAIL con `ModuleNotFoundError: No module named 'app.services.generation_preflight'`

- [ ] **Step 3: Write minimal implementation**

```python
from __future__ import annotations

import json
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.services.generation_compat import CompatVerdict, classify
from app.services.generation_variants import (
    MODEL_INDEX_FILENAME,
    Precision,
    available_precisions,
    select_for_precision,
)
from app.services.vram_estimate import estimate_peak_bytes

MB = 1024 * 1024


@dataclass(slots=True, frozen=True)
class PrecisionCost:
    precision: Precision
    download_bytes: int
    estimated_peak_bytes: int


@dataclass(slots=True, frozen=True)
class DeviceCapacity:
    id: str
    name: str
    kind: str
    free_vram_bytes: int | None


@dataclass(slots=True, frozen=True)
class DiskCapacity:
    target_path: str
    free_bytes: int


@dataclass(slots=True, frozen=True)
class PreflightReport:
    repo_id: str
    compat: CompatVerdict | None
    compat_reason: str | None
    degraded: bool
    reference_width: int
    reference_height: int
    precisions: list[PrecisionCost] = field(default_factory=list)
    devices: list[DeviceCapacity] = field(default_factory=list)
    disk: DiskCapacity | None = None


def _measure_disk(target: Path) -> DiskCapacity | None:
    # El directorio puede no existir todavia en una instalacion nueva: se sube
    # al primer ancestro que exista antes de medir.
    probe = target
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    try:
        return DiskCapacity(target_path=str(target), free_bytes=shutil.disk_usage(probe).free)
    except OSError:
        return None


def _measure_devices(devices_service: Any, probes: dict[str, Any]) -> list[DeviceCapacity]:
    rows: list[DeviceCapacity] = []
    for info in devices_service.list_devices():
        probe = probes.get(info["kind"])
        free_mb = probe.free_capacity_mb(info["id"]) if probe is not None else None
        rows.append(
            DeviceCapacity(
                id=info["id"],
                name=info.get("name") or info["id"],
                kind=info["kind"],
                free_vram_bytes=None if free_mb is None else free_mb * MB,
            )
        )
    return rows


async def _read_declared(hf_client: Any, repo_id: str) -> list[str]:
    scratch = Path(tempfile.mkdtemp(prefix="upflow-preflight-"))
    try:
        dest = scratch / MODEL_INDEX_FILENAME
        await hf_client.download(repo_id, MODEL_INDEX_FILENAME, dest, unlimited=True)
        index = json.loads(dest.read_text(encoding="utf-8"))
        return [
            name
            for name, value in index.items()
            if not name.startswith("_") and isinstance(value, list)
        ]
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


async def preflight(
    hf_client: Any,
    devices_service: Any,
    settings: Any,
    probes: dict[str, Any],
    repo_id: str,
    width: int = 512,
    height: int = 512,
) -> PreflightReport:
    # Los dispositivos y el disco no dependen de Hugging Face, asi que se miden
    # aunque la parte de red falle: un reporte degradado sigue siendo util.
    devices = _measure_devices(devices_service, probes)
    disk = _measure_disk(Path(settings.temp_path))

    try:
        files = await hf_client.repo_files(repo_id)
        declared = await _read_declared(hf_client, repo_id)
    except Exception:  # noqa: BLE001 - el pre-flight es diagnostico: nunca propaga
        return PreflightReport(
            repo_id=repo_id, compat=None, compat_reason=None, degraded=True,
            reference_width=width, reference_height=height,
            devices=devices, disk=disk,
        )

    verdict, reason = classify(tuple(f.path for f in files), None)
    costs: list[PrecisionCost] = []
    # Un repo ready_onnx no tiene paso de export: su precision la fijo quien lo
    # publico, asi que no se ofrece eleccion (ver el spec, alcance de B).
    if verdict == "needs_conversion":
        for precision in available_precisions(files):
            total = sum(f.size for f in select_for_precision(files, declared, precision))
            costs.append(
                PrecisionCost(
                    precision=precision,
                    download_bytes=total,
                    estimated_peak_bytes=estimate_peak_bytes(total, width, height),
                )
            )

    return PreflightReport(
        repo_id=repo_id, compat=verdict, compat_reason=reason, degraded=False,
        reference_width=width, reference_height=height,
        precisions=costs, devices=devices, disk=disk,
    )
```

Agregar a `app/schemas.py`, después de `ModelSearchResponse`:

```python
class PrecisionCostResponse(BaseModel):
    precision: str
    download_bytes: int = Field(serialization_alias="downloadBytes")
    estimated_peak_bytes: int = Field(serialization_alias="estimatedPeakBytes")


class DeviceCapacityResponse(BaseModel):
    id: str
    name: str
    kind: str
    free_vram_bytes: int | None = Field(default=None, serialization_alias="freeVramBytes")


class DiskCapacityResponse(BaseModel):
    target_path: str = Field(serialization_alias="targetPath")
    free_bytes: int = Field(serialization_alias="freeBytes")


class PreflightResponse(BaseModel):
    repo_id: str = Field(serialization_alias="repoId")
    compat: str | None = None
    compat_reason: str | None = Field(default=None, serialization_alias="compatReason")
    degraded: bool
    reference_width: int = Field(serialization_alias="referenceWidth")
    reference_height: int = Field(serialization_alias="referenceHeight")
    precisions: list[PrecisionCostResponse]
    devices: list[DeviceCapacityResponse]
    disk: DiskCapacityResponse | None = None
```

Y la ruta en `app/api/routes.py`, después de `search_generation_models`. `request: Request` va **primero y sin default** — un parámetro sin `Depends` ni default no puede ir después de los que sí lo tienen:

```python
@router.get("/generation/models/preflight", response_model=PreflightResponse)
async def preflight_generation_model(
    request: Request,
    repo_id: str = Query(..., alias="repoId"),
    width: int = Query(512, ge=64, le=4096),
    height: int = Query(512, ge=64, le=4096),
    hf_client: HfClient = Depends(get_hf_client),
    settings: Settings = Depends(get_settings),
) -> PreflightResponse:
    report = await preflight(
        hf_client=hf_client,
        devices_service=request.app.state.devices_service,
        settings=settings,
        probes=request.app.state.resource_probes,
        repo_id=repo_id,
        width=width,
        height=height,
    )
    return PreflightResponse(**asdict(report))
```

Requiere `from dataclasses import asdict` en `routes.py`. `asdict` convierte los dataclasses anidados a dicts y pydantic los coerce a los modelos hijos; los nombres de campo del dataclass y del schema coinciden en snake_case, así que el desempaquetado directo funciona.

Antes de escribir la ruta, verificar en `app/main.py` con qué nombres quedan `devices_service` y las probes en `app.state`. La rama de capacidad cableó las probes dentro de `DeviceSemaphores`; si no hay un `app.state.resource_probes` ya expuesto, agregarlo en el lifespan como parte de esta task:

```python
    app.state.resource_probes = {"gpu": DxgiVramProbe(), "cpu": SystemRamProbe()}
```

Reusar las mismas instancias que ya construye el wiring de `DeviceSemaphores` en vez de crear otras, para no duplicar sondas.

- [ ] **Step 4: Run the tests**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_generation_preflight.py -q`
Expected: PASS, 5 tests

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat: endpoint de pre-flight con disco, compat y VRAM por dispositivo

Devuelve hechos medidos, no veredictos: comparar el pico estimado contra
la VRAM libre es derivacion pura y vive en el frontend, asi cambiar de
precision no re-consulta el servidor.

Una fila por dispositivo enumerado, no un veredicto global -- cuando haya
multi-GPU por job la misma medicion alimenta otro agregador.

Si Hugging Face falla, degraded=true con 200: el pre-flight es
diagnostico y no puede bloquear una instalacion."
```

---

### Task 8: Búsqueda con query opcional y badges en la respuesta

**Files:**
- Modify: `app/api/routes.py:1270-1280` (`q` opcional, `sort`), y un mapper nuevo al lado de `_search_results_to_response` (`:995-1008`, que **no se toca**: lo usa el camino de upscalers)
- Modify: `app/schemas.py:204-211` (`HfModelSearchResultResponse` gana compat y precisiones)
- Modify: `app/services/generation_variants.py` (agregar `available_precisions_from_names`, del que pasa a depender `available_precisions` de Task 1)
- Test: `tests/test_generation_api.py` (agregar; el archivo ya existe)

**Interfaces:**
- Consumes: `classify` (Task 2), `HfModelSummary.filenames`/`.gated` (Task 4).
- Produces:
  - `available_precisions_from_names(filenames: tuple[str, ...]) -> list[Precision]` en `generation_variants.py`.
  - La respuesta de `/generation/models/search` gana `compat`, `compatReason` y `availablePrecisions` por resultado.

- [ ] **Step 1: Write the failing test**

Agregar a `tests/test_generation_api.py`. Ese archivo llama las corrutinas de ruta **directo con dobles**, sin `TestClient` (ver su comentario de cabecera, líneas 32-39); se sigue ese patrón.

```python
from app.api.routes import search_generation_models
from app.services.hf_client import HfModelSummary


class FakeSearchHfClient:
    def __init__(self, summaries):
        self._summaries = summaries
        self.calls: list[dict] = []

    async def search(self, query, limit=20, task_tags=None, sort=None):
        self.calls.append({"query": query, "task_tags": task_tags, "sort": sort})
        return self._summaries


def _summary(**overrides) -> HfModelSummary:
    base = dict(
        id="owner/name", author="owner", pipeline_tag="text-to-image",
        downloads=10, likes=2, tags=("text-to-image",),
        filenames=(
            "model_index.json",
            "unet/diffusion_pytorch_model.safetensors",
            "unet/diffusion_pytorch_model.fp16.safetensors",
        ),
        gated=False,
    )
    base.update(overrides)
    return HfModelSummary(**base)


@pytest.mark.asyncio
async def test_search_with_empty_query_browses_by_downloads() -> None:
    # Query vacia = browse. Antes exigia min_length=1 y el frontend mostraba un
    # cartel de texto en vez de resultados.
    hf_client = FakeSearchHfClient([_summary()])
    await search_generation_models(q="", hf_client=hf_client)
    assert hf_client.calls[0]["query"] == ""
    assert hf_client.calls[0]["sort"] == "downloads"


@pytest.mark.asyncio
async def test_search_with_query_does_not_force_a_sort() -> None:
    hf_client = FakeSearchHfClient([_summary()])
    await search_generation_models(q="sdxl", hf_client=hf_client)
    assert hf_client.calls[0]["query"] == "sdxl"
    assert hf_client.calls[0]["sort"] is None


@pytest.mark.asyncio
async def test_search_results_carry_compat_and_precisions() -> None:
    response = await search_generation_models(
        q="x", hf_client=FakeSearchHfClient([_summary()])
    )
    [result] = response.results
    assert result.compat == "needs_conversion"
    assert result.compat_reason
    assert result.available_precisions == ["fp16", "fp32"]


@pytest.mark.asyncio
async def test_gated_result_reports_gated_compat() -> None:
    response = await search_generation_models(
        q="x", hf_client=FakeSearchHfClient([_summary(gated="auto")])
    )
    assert response.results[0].compat == "gated"
    assert response.results[0].available_precisions == []


@pytest.mark.asyncio
async def test_ready_onnx_result_offers_no_precision_choice() -> None:
    # Sin paso de export no hay dtype que elegir (ver alcance de B en el spec).
    onnx_only = _summary(
        filenames=(
            "model_index.json",
            "unet/model.onnx",
            "unet/diffusion_pytorch_model.safetensors",
        )
    )
    response = await search_generation_models(
        q="x", hf_client=FakeSearchHfClient([onnx_only])
    )
    assert response.results[0].compat == "ready_onnx"
    assert response.results[0].available_precisions == []


@pytest.mark.asyncio
async def test_incompatible_result_when_model_index_missing() -> None:
    response = await search_generation_models(
        q="x", hf_client=FakeSearchHfClient([_summary(filenames=("config.json",))])
    )
    assert response.results[0].compat == "incompatible"


@pytest.mark.asyncio
async def test_search_response_serializes_camel_case() -> None:
    response = await search_generation_models(
        q="x", hf_client=FakeSearchHfClient([_summary()])
    )
    dumped = response.model_dump(by_alias=True)["results"][0]
    assert "availablePrecisions" in dumped
    assert "compatReason" in dumped
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_generation_api.py -q -k "browse_by_downloads or compat_and_precisions or gated_compat or no_precision_choice or model_index_missing"`
Expected: FAIL — `search_generation_models` no acepta `q=""` (`min_length=1`), no pasa `sort`, y la respuesta no tiene `compat`.

- [ ] **Step 3: Write minimal implementation**

`app/schemas.py`, ampliar `HfModelSearchResultResponse`:

```python
class HfModelSearchResultResponse(BaseModel):
    id: str
    author: str | None = None
    pipeline_tag: str | None = Field(default=None, serialization_alias="pipelineTag")
    downloads: int
    likes: int
    tags: list[str]
    compat: str | None = None
    compat_reason: str | None = Field(default=None, serialization_alias="compatReason")
    available_precisions: list[str] = Field(
        default_factory=list, serialization_alias="availablePrecisions"
    )
```

`app/api/routes.py`, la ruta:

```python
@router.get("/generation/models/search", response_model=ModelSearchResponse)
async def search_generation_models(
    q: str = Query("", max_length=200),
    hf_client: HfClient = Depends(get_hf_client),
) -> ModelSearchResponse:
    # Query vacia = browse por descargas: el usuario ve modelos sin tener que
    # saber el repo_id exacto de antemano.
    try:
        results = await hf_client.search(
            q, task_tags=GENERATION_SEARCH_TASK_TAGS, sort=None if q else "downloads"
        )
    except Exception as exc:
        logger.exception("Hugging Face generation search failed for query %r", q)
        raise HTTPException(status_code=502, detail="Hugging Face search failed") from exc
    return _generation_search_results_to_response(results)
```

Y el mapper nuevo, al lado de `_search_results_to_response` (que se deja intacto: lo usa el camino de upscalers):

```python
def _generation_search_results_to_response(results: list) -> ModelSearchResponse:
    enriched = []
    for item in results:
        verdict, reason = classify(item.filenames, item.gated)
        precisions = (
            [p for p in available_precisions_from_names(item.filenames)]
            if verdict == "needs_conversion"
            else []
        )
        enriched.append(
            HfModelSearchResultResponse(
                id=item.id,
                author=item.author,
                pipeline_tag=item.pipeline_tag,
                downloads=item.downloads,
                likes=item.likes,
                tags=list(item.tags),
                compat=verdict,
                compat_reason=reason,
                available_precisions=precisions,
            )
        )
    return ModelSearchResponse(results=enriched)
```

`available_precisions` de Task 1 recibe `list[HfFile]`, pero acá solo hay nombres (siblings sin tamaños). Agregar en `generation_variants.py` la variante por nombres, y hacer que la original delegue en ella:

```python
def available_precisions_from_names(filenames: tuple[str, ...]) -> list[Precision]:
    found = {
        _precision_of(name)
        for name in filenames
        if "/" in name and _is_usable_weight(name)
    }
    return [p for p in ("fp16", "fp32") if p in found]


def available_precisions(files: list[HfFile]) -> tuple[Precision, ...]:
    return tuple(available_precisions_from_names(tuple(f.path for f in files)))
```

- [ ] **Step 4: Run the tests**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_generation_api.py tests/test_generation_variants.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat: busqueda de generacion con query opcional y badges de compat

Query vacia devuelve los text-to-image mas descargados, para no exigir
que el usuario sepa el repo_id de antemano. Cada resultado lleva su
veredicto de compatibilidad y las precisiones disponibles, calculados de
siblings+gated que ya venian en la misma respuesta: cero requests extra.

El mapper de upscalers no se toca."
```

---

### Task 9: Servicio y hooks del frontend

**Files:**
- Modify: `frontend/src/lib/apiTypes.ts` (tipos nuevos)
- Modify: `frontend/src/services/generation.ts:72-78`
- Modify: `frontend/src/hooks/useGenerationJob.ts:151-158` (browse), `:207-216` (install con precisión)
- Test: `frontend/src/services/generation.test.ts`

**Interfaces:**
- Consumes: los endpoints de Tasks 7 y 8.
- Produces:
  - `preflightGenerationModel(repoId: string, width?: number, height?: number): Promise<PreflightResponse>`
  - `installGenerationModel(repoId: string, precision?: Precision): Promise<CreateInstallResponse>`
  - `useGenerationHfSearchResults(query)` con `enabled: true` siempre.
  - `useGenerationModelPreflight(repoId, enabled)`

- [ ] **Step 1: Write the failing test**

En `frontend/src/services/generation.test.ts`:

```ts
it("searches with an empty query for the browse view", async () => {
  const spy = vi.spyOn(api, "apiGet").mockResolvedValue({ results: [] });
  await searchGenerationModels("");
  expect(spy).toHaveBeenCalledWith("/generation/models/search?q=");
});

it("requests preflight with the reference resolution", async () => {
  const spy = vi.spyOn(api, "apiGet").mockResolvedValue({});
  await preflightGenerationModel("owner/name", 512, 512);
  expect(spy).toHaveBeenCalledWith(
    "/generation/models/preflight?repoId=owner%2Fname&width=512&height=512",
  );
});

it("sends the chosen precision when installing", async () => {
  const spy = vi.spyOn(api, "apiPostJson").mockResolvedValue({ installId: "1", statusUrl: "/x" });
  await installGenerationModel("owner/name", "fp16");
  expect(spy).toHaveBeenCalledWith("/generation/models", {
    repoId: "owner/name",
    precision: "fp16",
  });
});

it("omits precision when none is chosen", async () => {
  const spy = vi.spyOn(api, "apiPostJson").mockResolvedValue({ installId: "1", statusUrl: "/x" });
  await installGenerationModel("owner/name");
  expect(spy).toHaveBeenCalledWith("/generation/models", { repoId: "owner/name" });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd frontend && npm test -- generation.test.ts`
Expected: FAIL — `preflightGenerationModel` no existe; `installGenerationModel` ignora el segundo argumento.

- [ ] **Step 3: Write minimal implementation**

En `frontend/src/lib/apiTypes.ts`:

```ts
export type Precision = "fp16" | "fp32";
export type CompatVerdict = "ready_onnx" | "needs_conversion" | "gated" | "incompatible";

export interface PrecisionCost {
  precision: Precision;
  downloadBytes: number;
  estimatedPeakBytes: number;
}

export interface DeviceCapacity {
  id: string;
  name: string;
  kind: string;
  freeVramBytes: number | null;
}

export interface PreflightResponse {
  repoId: string;
  compat: CompatVerdict | null;
  compatReason: string | null;
  degraded: boolean;
  referenceWidth: number;
  referenceHeight: number;
  precisions: PrecisionCost[];
  devices: DeviceCapacity[];
  disk: { targetPath: string; freeBytes: number } | null;
}
```

Y agregar a `HfModelSearchResultResponse` los campos `compat`, `compatReason` y `availablePrecisions`.

En `frontend/src/services/generation.ts`:

```ts
export function preflightGenerationModel(
  repoId: string,
  width = 512,
  height = 512,
): Promise<PreflightResponse> {
  const params = new URLSearchParams({ repoId, width: String(width), height: String(height) });
  return apiGet<PreflightResponse>(`/generation/models/preflight?${params}`);
}

export function installGenerationModel(
  repoId: string,
  precision?: Precision,
): Promise<CreateInstallResponse> {
  const body: Record<string, unknown> = { repoId };
  if (precision) body.precision = precision;
  return apiPostJson<CreateInstallResponse>("/generation/models", body);
}
```

En `useGenerationJob.ts`, la búsqueda deja de estar gateada por la longitud (la query vacía es el browse):

```ts
export function useGenerationHfSearchResults(query: string) {
  const trimmed = query.trim();
  return useQuery<ModelSearchResponse>({
    queryKey: ["generation-hf-search", trimmed],
    queryFn: () => searchGenerationModels(trimmed),
  });
}

export function useGenerationModelPreflight(repoId: string, enabled: boolean) {
  return useQuery<PreflightResponse>({
    queryKey: ["generation-model-preflight", repoId],
    queryFn: () => preflightGenerationModel(repoId),
    enabled,
    staleTime: 5 * 60 * 1000,
  });
}
```

Y `useGenerationModelInstall` acepta la precisión en `install`:

```ts
  const startMutation = useMutation({
    mutationFn: ({ repoId, precision }: { repoId: string; precision?: Precision }) =>
      installGenerationModel(repoId, precision),
    onSuccess: (data) => setInstallId(data.installId),
  });
```

Ajustar el tipo de `install` en `UseGenerationModelInstallResult` a `(repoId: string, precision?: Precision) => void` y su implementación para llamar `startMutation.mutate({ repoId, precision })`.

- [ ] **Step 4: Run the tests**

Run: `cd frontend && npm test -- generation.test.ts`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat: servicio y hooks de pre-flight y precision en el frontend

La busqueda deja de estar gateada por longitud: query vacia es el browse
por popularidad. El pre-flight se cachea 5 min por repo porque su costo
son dos requests a Hugging Face."
```

---

### Task 10: Tarjeta de modelo con badge, picker y avisos

**Files:**
- Create: `frontend/src/modules/models/GenerationModelCard.tsx`
- Create: `frontend/src/modules/models/generationWarnings.ts`
- Modify: `frontend/src/modules/models/GenerationHfSearch.tsx:39-84`
- Test: `frontend/src/modules/models/generationWarnings.test.ts`, `frontend/src/modules/models/GenerationModelCard.test.tsx`

**Interfaces:**
- Consumes: `useGenerationHfSearchResults`, `useGenerationModelPreflight`, `useGenerationModelInstall` (Task 9).
- Produces: `buildWarnings(preflight, precision) -> Warning[]` — pura, testeable sin React. Es donde vive la comparación pico-vs-libre que el servidor deliberadamente no hace.

- [ ] **Step 1: Write the failing test**

`generationWarnings.test.ts` — la lógica de decisión, pura:

```ts
import { describe, expect, it } from "vitest";
import { buildWarnings } from "./generationWarnings";

const GB = 1024 ** 3;

const base = {
  repoId: "owner/name",
  compat: "needs_conversion" as const,
  compatReason: "Sin ONNX propio para unet",
  degraded: false,
  referenceWidth: 512,
  referenceHeight: 512,
  precisions: [{ precision: "fp16" as const, downloadBytes: 3 * GB, estimatedPeakBytes: 4 * GB }],
  devices: [
    { id: "dml:0", name: "RX 7900 XTX", kind: "gpu", freeVramBytes: 22 * GB },
    { id: "dml:1", name: "RX 6600", kind: "gpu", freeVramBytes: 7 * GB },
    { id: "cpu", name: "CPU", kind: "cpu", freeVramBytes: null },
  ],
  disk: { targetPath: "D:\\temp", freeBytes: 50 * GB },
};

it("warns about devices where the estimate does not fit", () => {
  const codes = buildWarnings(base, "fp16").map((w) => w.code);
  expect(codes).toContain("device_wont_fit");
});

it("does not warn about a device with room to spare", () => {
  const fits = { ...base, devices: [base.devices[0]] };
  expect(buildWarnings(fits, "fp16").map((w) => w.code)).not.toContain("device_wont_fit");
});

it("warns when free disk is below the download size", () => {
  const tight = { ...base, disk: { targetPath: "D:\\temp", freeBytes: 1 * GB } };
  expect(buildWarnings(tight, "fp16").map((w) => w.code)).toContain("disk_low");
});

it("never warns about disk when it could not be measured", () => {
  const noDisk = { ...base, disk: null };
  expect(buildWarnings(noDisk, "fp16").map((w) => w.code)).not.toContain("disk_low");
});

it("never warns about a device whose VRAM could not be measured", () => {
  const unmeasured = {
    ...base,
    devices: [{ id: "dml:0", name: "GPU", kind: "gpu", freeVramBytes: null }],
  };
  expect(buildWarnings(unmeasured, "fp16").map((w) => w.code)).not.toContain("device_wont_fit");
});

it("warns that CPU generation is slow", () => {
  expect(buildWarnings(base, "fp16").map((w) => w.code)).toContain("cpu_slow");
});

it("warns when the repo is gated", () => {
  const gated = { ...base, compat: "gated" as const };
  expect(buildWarnings(gated, "fp16").map((w) => w.code)).toContain("gated");
});

it("returns a degraded notice and nothing else when preflight failed", () => {
  const degraded = { ...base, degraded: true, precisions: [], compat: null };
  expect(buildWarnings(degraded, "fp16").map((w) => w.code)).toEqual(["degraded"]);
});
```

`GenerationModelCard.test.tsx` — el requisito central de A. Sigue el patrón de mocks y wrapper de `GenerationHfSearch.test.tsx:1-47`:

```tsx
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import type { ReactNode } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { HfModelSearchResultResponse, PreflightResponse } from "../../lib/apiTypes";
import * as generationService from "../../services/generation";
import { GenerationModelCard } from "./GenerationModelCard";

vi.mock("../../services/generation", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../../services/generation")>();
  return {
    ...actual,
    preflightGenerationModel: vi.fn(),
    installGenerationModel: vi.fn(),
    getGenerationInstallStatus: vi.fn(),
    getConversionStatus: vi.fn(),
  };
});

const GB = 1024 ** 3;

const RESULT: HfModelSearchResultResponse = {
  id: "owner/name",
  author: "owner",
  pipelineTag: "text-to-image",
  downloads: 10,
  likes: 2,
  tags: [],
  compat: "needs_conversion",
  compatReason: "Sin ONNX propio para unet",
  availablePrecisions: ["fp16", "fp32"],
};

const TIGHT_PREFLIGHT: PreflightResponse = {
  repoId: "owner/name",
  compat: "needs_conversion",
  compatReason: "Sin ONNX propio para unet",
  degraded: false,
  referenceWidth: 512,
  referenceHeight: 512,
  precisions: [
    { precision: "fp16", downloadBytes: 3 * GB, estimatedPeakBytes: 9 * GB },
    { precision: "fp32", downloadBytes: 6 * GB, estimatedPeakBytes: 18 * GB },
  ],
  devices: [{ id: "dml:0", name: "RX 6600", kind: "gpu", freeVramBytes: 7 * GB }],
  disk: { targetPath: "D:\\temp", freeBytes: 1 * GB },
};

function renderCard(result: HfModelSearchResultResponse = RESULT) {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  });
  function Wrapper({ children }: { children: ReactNode }) {
    return <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>;
  }
  return render(<GenerationModelCard result={result} />, { wrapper: Wrapper });
}

afterEach(() => {
  vi.mocked(generationService.preflightGenerationModel).mockReset();
  vi.mocked(generationService.installGenerationModel).mockReset();
});

describe("GenerationModelCard", () => {
  it("shows the compat badge without expanding and without calling preflight", () => {
    renderCard();
    expect(screen.getByText(/requiere conversión/i)).toBeInTheDocument();
    expect(generationService.preflightGenerationModel).not.toHaveBeenCalled();
  });

  it("keeps Install enabled even when every warning fires", async () => {
    vi.mocked(generationService.preflightGenerationModel).mockResolvedValue(TIGHT_PREFLIGHT);

    renderCard();
    fireEvent.click(screen.getByRole("button", { name: /detalles/i }));

    await waitFor(() => expect(screen.getByText(/no entra/i)).toBeInTheDocument());
    expect(screen.getByText(/libres en D:\\temp/i)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /^install$/i })).toBeEnabled();
  });

  it("offers only the precisions the repo publishes", async () => {
    vi.mocked(generationService.preflightGenerationModel).mockResolvedValue({
      ...TIGHT_PREFLIGHT,
      precisions: [{ precision: "fp32", downloadBytes: 6 * GB, estimatedPeakBytes: 8 * GB }],
    });

    renderCard({ ...RESULT, availablePrecisions: ["fp32"] });
    fireEvent.click(screen.getByRole("button", { name: /detalles/i }));

    await waitFor(() => expect(screen.getByRole("radio", { name: /fp32/i })).toBeInTheDocument());
    expect(screen.queryByRole("radio", { name: /fp16/i })).not.toBeInTheDocument();
  });

  it("fires preflight once even when toggled repeatedly", async () => {
    vi.mocked(generationService.preflightGenerationModel).mockResolvedValue(TIGHT_PREFLIGHT);

    renderCard();
    const toggle = screen.getByRole("button", { name: /detalles/i });
    fireEvent.click(toggle);
    await waitFor(() => expect(screen.getByText(/no entra/i)).toBeInTheDocument());
    fireEvent.click(toggle);
    fireEvent.click(toggle);

    await waitFor(() =>
      expect(generationService.preflightGenerationModel).toHaveBeenCalledTimes(1),
    );
  });

  it("installs with the selected precision", async () => {
    vi.mocked(generationService.preflightGenerationModel).mockResolvedValue(TIGHT_PREFLIGHT);
    vi.mocked(generationService.installGenerationModel).mockResolvedValue({
      installId: "1",
      statusUrl: "/x",
    });

    renderCard();
    fireEvent.click(screen.getByRole("button", { name: /detalles/i }));
    await waitFor(() => expect(screen.getByRole("radio", { name: /fp32/i })).toBeInTheDocument());
    fireEvent.click(screen.getByRole("radio", { name: /fp32/i }));
    fireEvent.click(screen.getByRole("button", { name: /^install$/i }));

    await waitFor(() =>
      expect(generationService.installGenerationModel).toHaveBeenCalledWith("owner/name", "fp32"),
    );
  });

  it("still allows install when preflight came back degraded", async () => {
    vi.mocked(generationService.preflightGenerationModel).mockResolvedValue({
      ...TIGHT_PREFLIGHT,
      degraded: true,
      compat: null,
      precisions: [],
      disk: null,
    });

    renderCard();
    fireEvent.click(screen.getByRole("button", { name: /detalles/i }));

    await waitFor(() => expect(screen.getByText(/no se pudo evaluar/i)).toBeInTheDocument());
    expect(screen.getByRole("button", { name: /^install$/i })).toBeEnabled();
  });
});
```

Requisito de accesibilidad que estos tests fijan: el toggle de expansión es un `<button>` con texto accesible que matchea `/detalles/i`, y el picker usa `role="radio"` con la precisión en su nombre accesible. El botón de instalar mantiene el nombre `Install` exacto (`/^install$/i`) para no colisionar con el toggle.

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd frontend && npm test -- generationWarnings GenerationModelCard`
Expected: FAIL — los módulos no existen.

- [ ] **Step 3: Write minimal implementation**

`generationWarnings.ts`:

```ts
import type { PreflightResponse, Precision } from "../../lib/apiTypes";

export type WarningCode =
  | "degraded"
  | "gated"
  | "incompatible"
  | "disk_low"
  | "device_wont_fit"
  | "cpu_slow";

export interface Warning {
  code: WarningCode;
  message: string;
}

function formatGb(bytes: number): string {
  return `${(bytes / 1024 ** 3).toFixed(1)} GB`;
}

// El servidor manda hechos medidos; la decision de que amerita un aviso vive
// aca, para que cambiar de precision no re-consulte el backend. `null` en una
// medicion significa "no se pudo medir": nunca genera aviso.
export function buildWarnings(
  preflight: PreflightResponse,
  precision: Precision,
): Warning[] {
  if (preflight.degraded) {
    return [
      {
        code: "degraded",
        message: "No se pudo evaluar este modelo. Podés instalarlo igual.",
      },
    ];
  }

  const warnings: Warning[] = [];

  if (preflight.compat === "gated") {
    warnings.push({
      code: "gated",
      message: "Repo con acceso restringido: necesitás un token de Hugging Face y aceptar la licencia.",
    });
  }
  if (preflight.compat === "incompatible") {
    warnings.push({
      code: "incompatible",
      message: preflight.compatReason ?? "No parece un pipeline diffusers.",
    });
  }

  const cost = preflight.precisions.find((p) => p.precision === precision);

  if (cost && preflight.disk && preflight.disk.freeBytes < cost.downloadBytes) {
    warnings.push({
      code: "disk_low",
      message:
        `Quedan ${formatGb(preflight.disk.freeBytes)} libres en ${preflight.disk.targetPath} ` +
        `y hace falta ${formatGb(cost.downloadBytes)}.`,
    });
  }

  if (cost) {
    const tooSmall = preflight.devices.filter(
      (d) => d.kind === "gpu" && d.freeVramBytes !== null && d.freeVramBytes < cost.estimatedPeakBytes,
    );
    for (const device of tooSmall) {
      warnings.push({
        code: "device_wont_fit",
        message:
          `${device.name}: no entra. Necesita ~${formatGb(cost.estimatedPeakBytes)} estimados ` +
          `a ${preflight.referenceWidth}×${preflight.referenceHeight} y tiene ` +
          `${formatGb(device.freeVramBytes as number)} libres.`,
      });
    }
  }

  if (preflight.devices.every((d) => d.kind !== "gpu")) {
    warnings.push({
      code: "cpu_slow",
      message: "Sin GPU compatible: generar en CPU tarda varios minutos por imagen.",
    });
  } else if (preflight.devices.some((d) => d.kind === "cpu")) {
    warnings.push({
      code: "cpu_slow",
      message: "En CPU tarda varios minutos por imagen.",
    });
  }

  return warnings;
}
```

`GenerationModelCard.tsx` renderiza: el badge desde el resultado de búsqueda (sin preflight), y al expandir dispara `useGenerationModelPreflight`, muestra el picker con `downloadBytes` por precisión, una fila por dispositivo, los avisos de `buildWarnings`, y el botón Install **sin `disabled` en ninguna rama**. Reusar `ResultMeta` e `InstallProgress`/`InstallError` de `HfResultCard.tsx`/`installUi.tsx` en vez de reimplementarlos.

`GenerationHfSearch.tsx` deja de ramificar por query vacía:

```tsx
export function GenerationHfSearch({
  debounceMs = DEFAULT_SEARCH_DEBOUNCE_MS,
}: GenerationHfSearchProps) {
  const [query, setQuery] = useState("");
  const debouncedQuery = useDebouncedValue(query, debounceMs);

  return (
    <div className="flex flex-col gap-4">
      <SearchInput value={query} onChange={setQuery} />
      <SearchResults query={debouncedQuery.trim()} />
    </div>
  );
}
```

`SearchResults` cambia `GenerationResultCard` por `GenerationModelCard` y su `NoResultsState` sigue cubriendo el caso de cero resultados.

- [ ] **Step 4: Run the tests**

Run: `cd frontend && npm test`
Expected: PASS, incluido `GenerationHfSearch.test.tsx` (ajustar el test preexistente que afirmaba el cartel de query vacía: ahora se esperan resultados).

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat: tarjeta de modelo con badge de compat, picker de precision y avisos

El boton de instalar no tiene rama disabled: los avisos informan y la
decision es del usuario. La comparacion pico-vs-VRAM libre vive en
buildWarnings, pura y testeada sin React.

Una medicion en null (VRAM no sondeable, disco no medible) nunca genera
aviso: no se avisa de lo que no se sabe."
```

---

### Task 11: Documentación

**Files:**
- Modify: `README.md` (sección de generación: browse, badges, precisión; quitar el cap de la tabla de variables si Task 5 no lo hizo)
- Modify: `.env.example` (idem)

- [ ] **Step 1: Verificar que el cap ya no aparece en ningún doc**

Run: `git grep -n "MAX_GENERATION_MODEL_DOWNLOAD_MB"`
Expected: sin resultados (fuera de los specs históricos en `docs/superpowers/`, que son registro y no se editan).

- [ ] **Step 2: Documentar el comportamiento nuevo en el README**

Agregar a la sección del módulo de generación: que la búsqueda sin escribir muestra los más descargados con badge de compatibilidad, que la precisión se elige por instalación y afecta descarga y runtime, y que no hay techo de descarga — solo avisos.

- [ ] **Step 3: Commit**

```bash
git add README.md .env.example
git commit -m "docs: busqueda por popularidad, badges de compat y precision elegible"
```

---

## Verificación final

- [ ] `.\.venv\Scripts\python.exe -m pytest -q` — toda la suite verde
- [ ] `cd frontend && npm test` — toda la suite verde
- [ ] `cd frontend && npm run build` — sin errores de tipos
- [ ] `git grep -n "_ensure_size_cap"` — sin resultados en `app/`
- [ ] `git grep -n "max_generation_model_download_mb"` — sin resultados en `app/` ni `tests/`
- [ ] Los tests del cap de upscalers siguen presentes y verdes: `pytest tests/test_hf_client.py tests/test_model_installer.py -q`

## Riesgo abierto

`FP16_EXPORT_ATOL = 1e-2` se verificó sobre un fixture *tiny*. Un modelo real acumula más error, así que su primera conversión fp16 puede volver a levantar `RuntimeError` en la validación. Si pasa, subir el valor con el número medido y anotarlo en el spec — **no** apagar `do_validation`.
