# Módulo de generación de imágenes — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Texto→imagen local (SD1.5 ONNX vía DirectML) como 4to job kind de Upflow, con installer HF paralelo, auto-upscale opcional y mensajes de error específicos por hardware.

**Architecture:** `GenerationEngine` envuelve `ORTStableDiffusionPipeline` de optimum reusando `_build_providers`/`GpuSessionCoordinator` existentes; `GenerationModelInstaller` paralelo al `ModelInstaller` actual (descarga multi-archivo formato diffusers, validación por forward-pass); `GenerationJobManager` calcado del esqueleto de `AudioJobManager`; frontend agrega kind `"generation"` a la cola visual existente.

**Tech Stack:** FastAPI + pydantic v2, `optimum[onnxruntime]` (pin resuelto en Task 1), onnxruntime-directml 1.24.4, React + TanStack Query + vitest.

**Spec:** `docs/superpowers/specs/2026-07-22-generation-module-design.md` (approved).

## Global Constraints

- Solo backend DirectML: providers vía `_build_providers(device)` de `onnx_upscaler.py` (`cpu` / `dml:N`). Nada de CUDA/ROCm/OpenVINO.
- Solo texto→imagen. Nada de video/img2img/inpainting/LoRA.
- `ModelInstaller` actual NO se toca, salvo la rama de borrado recursivo en `_delete_model_file` (Task 2).
- Cap de descarga de generación: `max_generation_model_download_mb`, default **8192** (separado del cap de 2GB de upscalers).
- Params de generación: `steps` default 25 cap 100; `guidance` default 7.5 (0–30); `width`/`height` default 512, rango 64–1024, múltiplos de 64; `seed` opcional ≥0.
- Mensajes de error EXACTOS (copiar verbatim, están en Task 4): CUDA-only, VRAM, CPU-only.
- Job de auto-upscale: UN solo job, dos stages (`generating` → `upscaling`), pesos 85/15.
- `optimum` debe convivir con `onnxruntime-directml` — Task 1 es gate: si falla, STOP y re-plan (fallback: ensamblado manual de sesiones).
- Tests: `pytest` backend (desde raíz del repo), `npm test` frontend (desde `frontend/`). Commits convencionales (`feat:`/`fix:`/`test:`), sin `Co-Authored-By`.
- Repo: `C:\Users\santi\.openclaw\workspace\image-upscaler-amd`. Python del proyecto: el venv que usa el repo (`.venv` en la raíz; si no existe, `python` del PATH con deps del pyproject instaladas).

## File Structure

| Acción | Path | Responsabilidad |
|---|---|---|
| Create | `scripts/spike_optimum_directml.py` | Spike compat optimum+DirectML (queda como herramienta de diagnóstico) |
| Create | `docs/superpowers/specs/2026-07-22-optimum-spike-findings.md` | Resultado del spike: pins, receta pip, nombre de clase, kwargs |
| Modify | `app/services/model_registry.py` | `ModelKind.diffusion_onnx` |
| Modify | `app/services/model_installer.py` | Rama borrado recursivo de carpeta en `_delete_model_file` |
| Modify | `app/config.py` | `max_generation_model_download_mb` |
| Modify | `app/services/hf_client.py` | Param `max_bytes` opcional en `download` |
| Modify | `pyproject.toml` | Dependencia optimum (pin del spike) |
| Create | `app/services/engines/generation_onnx.py` | `GenerationEngine` + mensajes de error + availability check |
| Create | `app/services/generation_installer.py` | `GenerationModelInstaller` |
| Modify | `app/services/progress.py` | Stages `generating`/`upscaling` |
| Modify | `app/services/job_manager.py` | Extraer `select_upscale_engine` module-level |
| Modify | `app/models.py` | Dataclass `GenerationJob` |
| Create | `app/services/generation_job_manager.py` | `GenerationJobManager` |
| Modify | `app/services/retention_sweeper.py` | Prune de jobs de generación |
| Modify | `app/schemas.py` | Request/response de generación |
| Modify | `app/api/routes.py` | 7 endpoints + `generation_job_to_response` |
| Modify | `app/main.py` | Wiring lifespan |
| Create | `tests/test_generation_engine.py`, `tests/test_generation_installer.py`, `tests/test_generation_job_manager.py`, `tests/test_generation_api.py` | Tests backend |
| Modify | `frontend/src/lib/api.ts` | `apiPostJson` |
| Modify | `frontend/src/lib/apiTypes.ts` | Tipos de generación + union `AnyJobResponse` |
| Create | `frontend/src/services/generation.ts` | Servicios API |
| Modify | `frontend/src/lib/jobQueueStore.ts`, `frontend/src/hooks/useJobQueue.ts` | Kind `"generation"` en cola |
| Modify | `frontend/src/components/JobCard.tsx`, `frontend/src/components/JobDetailModal.tsx` | Guard + details de generación |
| Modify | `frontend/src/components/ModelPicker.tsx` | Excluir `diffusion-onnx` del picker de upscale |
| Create | `frontend/src/hooks/useGenerationJob.ts` (+`.test.tsx`) | Hook de job |
| Create | `frontend/src/modules/generate/GeneratePage.tsx`, `GeneratePanel.tsx` (+`.test.tsx`) | Página Generate |
| Modify | `frontend/src/App.tsx` + componente de nav | Ruta `/generate` |
| Create | `frontend/src/modules/models/GenerationModelsSection.tsx` (+`.test.tsx`) | Instalación de modelos de generación |
| Modify | `frontend/src/pages/ModelsPage.tsx` | Montar la sección |

---

### Task 1: Spike — optimum + onnxruntime-directml 1.24.4 (GATE)

Riesgo documentado en el spec: optimum puede asumir el paquete vanilla `onnxruntime` en sus checks de import y `pip install optimum[onnxruntime]` puede pisar `onnxruntime-directml`. Este task lo resuelve empíricamente. **No es TDD — es un spike con entregable de documentación y decisión GO/NO-GO.**

**Files:**
- Create: `scripts/spike_optimum_directml.py`
- Create: `docs/superpowers/specs/2026-07-22-optimum-spike-findings.md`

**Interfaces:**
- Produces: findings doc con (a) pin exacto de optimum, (b) receta pip segura, (c) nombre de clase del pipeline (`ORTStableDiffusionPipeline` u `ORTDiffusionPipeline`), (d) kwargs verificados de `from_pretrained` y `__call__`, (e) tipo del `generator`. Tasks 4 y 5 consumen (a)–(e).

- [ ] **Step 1: venv aislado + instalación candidata**

```powershell
cd C:\Users\santi\.openclaw\workspace\image-upscaler-amd
python -m venv .venv-spike
.venv-spike\Scripts\python -m pip install --upgrade pip
.venv-spike\Scripts\pip install onnxruntime-directml==1.24.4
.venv-spike\Scripts\pip install --dry-run --quiet --report %TEMP%\optimum-report.json "optimum[onnxruntime]"
.venv-spike\Scripts\python -c "import json,os; r=json.load(open(os.environ['TEMP']+r'\optimum-report.json')); print([i['metadata']['name']+'=='+i['metadata']['version'] for i in r['install']])"
```

Anotar: ¿el report incluye distribución `onnxruntime` (vanilla)? ¿Qué versión de `optimum` resuelve pip?

- [ ] **Step 2: instalar con la receta que no pise -directml**

Si el dry-run trae `onnxruntime` vanilla, usar la receta A; si no, receta B directa:

```powershell
# Receta A (esperada): instalar sin deps y traer las deps reales a mano
.venv-spike\Scripts\pip install --no-deps optimum optimum-onnx
.venv-spike\Scripts\pip install "transformers>=4.44" "diffusers>=0.30" onnx huggingface_hub packaging numpy pillow
# Receta B: directa
.venv-spike\Scripts\pip install "optimum[onnxruntime]"
```

Nota: en versiones recientes optimum movió lo ONNX al paquete `optimum-onnx`; si `pip install --no-deps optimum-onnx` falla por no existir para la versión resuelta, quedarse solo con `optimum`. Tras instalar, verificar que `-directml` sigue intacto:

```powershell
.venv-spike\Scripts\python -c "import onnxruntime as ort; print(ort.__version__); print(ort.get_available_providers())"
```

Expected: `1.24.4` y `DmlExecutionProvider` en la lista. Si `pip` desinstaló/pisó `-directml`: `pip uninstall onnxruntime -y` + `pip install --force-reinstall onnxruntime-directml==1.24.4` y re-verificar. Documentar la secuencia exacta que funcionó — esa es la "receta pip segura" del findings doc.

- [ ] **Step 3: verificar import del pipeline y detectar cómo chequea onnxruntime**

```powershell
.venv-spike\Scripts\python -c "from optimum.onnxruntime import ORTStableDiffusionPipeline; print('ORTStableDiffusionPipeline OK')"
.venv-spike\Scripts\python -c "from optimum.onnxruntime import ORTDiffusionPipeline; print('ORTDiffusionPipeline OK')"
```

Anotar cuál(es) existen. Si el import falla con mensaje tipo "onnxruntime is not installed", abrir `**/optimum/utils/import_utils.py` del venv, citar el check en el findings doc (¿`importlib.util.find_spec("onnxruntime")` — pasa con -directml — o `importlib.metadata.version("onnxruntime")` — falla?) y decidir si hay workaround de una línea (p.ej. la env var que optimum respete) o NO-GO.

- [ ] **Step 4: escribir `scripts/spike_optimum_directml.py` con forward-pass real**

```python
"""Spike: valida optimum + onnxruntime-directml. Uso: python scripts/spike_optimum_directml.py <pipeline_dir> [device]"""
from __future__ import annotations

import sys
import time
from pathlib import Path


def main() -> int:
    pipeline_dir = Path(sys.argv[1])
    device = sys.argv[2] if len(sys.argv) > 2 else "dml:0"

    import numpy as np
    import onnxruntime as ort

    print(f"onnxruntime {ort.__version__}, providers: {ort.get_available_providers()}")

    from optimum.onnxruntime import ORTStableDiffusionPipeline  # ajustar si Step 3 dijo otra clase

    if device.startswith("dml:"):
        kwargs = {"provider": "DmlExecutionProvider", "provider_options": {"device_id": int(device.split(":")[1])}}
    else:
        kwargs = {"provider": "CPUExecutionProvider"}

    t0 = time.perf_counter()
    pipe = ORTStableDiffusionPipeline.from_pretrained(str(pipeline_dir), **kwargs)
    print(f"load: {time.perf_counter() - t0:.1f}s")

    steps_seen: list[int] = []

    def on_step(step: int, timestep, latents) -> None:
        steps_seen.append(step)

    t0 = time.perf_counter()
    result = pipe(
        prompt="a red apple on a wooden table",
        negative_prompt="blurry",
        num_inference_steps=4,
        guidance_scale=7.5,
        width=256,
        height=256,
        callback=on_step,
        callback_steps=1,
        generator=np.random.RandomState(42),
    )
    print(f"infer: {time.perf_counter() - t0:.1f}s, callback steps: {steps_seen}")
    out = Path("spike_output.png")
    result.images[0].save(out)
    print(f"saved {out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 5: correr el spike contra un pipeline real**

Bajar el más chico disponible (SD1.5 de la colección amd, o un tiny de CI de optimum para iterar rápido):

```powershell
.venv-spike\Scripts\pip install "huggingface_hub[cli]"
.venv-spike\Scripts\python -c "from huggingface_hub import snapshot_download; print(snapshot_download('amd/stable-diffusion-v1-5_io32_amdgpu', local_dir=r'%TEMP%\spike-sd15'))"
.venv-spike\Scripts\python scripts\spike_optimum_directml.py %TEMP%\spike-sd15 dml:0
```

Si el repo id exacto difiere, buscar en https://huggingface.co/amd (colección de difusión ONNX; anotar el id real — es también el default sugerido en la UI). Expected: `spike_output.png` generado, callback llamado 4 veces, sin crash DML. Si `callback`/`callback_steps` no son aceptados (API nueva usa `callback_on_step_end`), anotar los kwargs que SÍ funcionaron.

- [ ] **Step 6: escribir findings doc + decisión**

`docs/superpowers/specs/2026-07-22-optimum-spike-findings.md` con: pin exacto (`optimum==X.Y.Z` y, si aplica, `optimum-onnx==…`), receta pip segura paso a paso, clase e import path, kwargs verificados de `from_pretrained` (¿acepta `session_options`?) y `__call__` (callback + generator), si torch es import-requerido en el path ONNX (el repo ya trae `torch>=2.2.0` — solo confirmar que no hay conflicto de versión), y el repo id del SD1.5 de amd usado. **GATE: si no hubo forma de importar optimum con -directml o el forward-pass DML falla → STOP, no seguir con Tasks 2+; volver al usuario con el findings doc y re-planear con el fallback (ensamblado manual de las sesiones ONNX del pipeline).**

- [ ] **Step 7: limpiar y commitear**

```powershell
Remove-Item -Recurse -Force .venv-spike
git add scripts/spike_optimum_directml.py docs/superpowers/specs/2026-07-22-optimum-spike-findings.md
git commit -m "feat: spike de compatibilidad optimum + onnxruntime-directml"
```

---

### Task 2: `ModelKind.diffusion_onnx` + borrado recursivo de carpetas

**Files:**
- Modify: `app/services/model_registry.py:23-25` (enum `ModelKind`)
- Modify: `app/services/model_installer.py:222-238` (`_delete_model_file`)
- Test: `tests/test_model_installer.py`

**Interfaces:**
- Produces: `ModelKind.diffusion_onnx` (valor `"diffusion-onnx"`). Para esta clase `ModelEntry.file_path` apunta a CARPETA relativa a `models_path` y `scale is None`. `ModelInstaller.delete` borra carpetas. Consumido por Tasks 5, 8, 9, 11, 13.

- [ ] **Step 1: test del borrado recursivo (falla)**

En `tests/test_model_installer.py`, junto a los tests de delete existentes (usar los mismos fixtures `make_installer`/`make_settings` del archivo):

```python
@pytest.mark.anyio
async def test_delete_diffusion_model_removes_directory_recursively(tmp_path: Path) -> None:
    installer, registry, settings, _hf = make_installer(tmp_path, files=[])
    model_dir = settings.models_path / "generation" / "gen--amd--sd15"
    (model_dir / "unet").mkdir(parents=True)
    (model_dir / "model_index.json").write_text("{}", encoding="utf-8")
    (model_dir / "unet" / "model.onnx").write_bytes(b"onnx")
    registry.register(
        ModelEntry(
            id="gen--amd--sd15",
            name="amd/sd15",
            kind=ModelKind.diffusion_onnx,
            source="hf:amd/sd15",
            size_bytes=4,
            scale=None,
            file_path="generation/gen--amd--sd15",
        )
    )

    await installer.delete("gen--amd--sd15")

    assert not model_dir.exists()
    assert registry.get("gen--amd--sd15") is None


@pytest.mark.anyio
async def test_delete_diffusion_model_rejects_path_escaping_models_root(tmp_path: Path) -> None:
    installer, registry, settings, _hf = make_installer(tmp_path, files=[])
    outside = tmp_path / "outside-dir"
    outside.mkdir()
    registry.register(
        ModelEntry(
            id="gen--evil",
            name="evil",
            kind=ModelKind.diffusion_onnx,
            source="hf:evil/evil",
            size_bytes=1,
            scale=None,
            file_path="../outside-dir",
        )
    )

    await installer.delete("gen--evil")

    assert outside.exists()  # guard: nunca borrar fuera de models_path
```

Ajustar la tupla de retorno de `make_installer` al orden real del fixture del archivo (hoy devuelve installer/registry/settings/hf en algún orden — copiarlo de un test vecino). Si los tests async del archivo usan otro marker (`@pytest.mark.asyncio` o ninguno), copiar el estilo del test vecino.

- [ ] **Step 2: correr — deben fallar**

```powershell
python -m pytest tests/test_model_installer.py -k diffusion -v
```

Expected: FAIL — `AttributeError: diffusion_onnx` (el enum no existe).

- [ ] **Step 3: implementar**

`app/services/model_registry.py` — agregar al enum `ModelKind`:

```python
class ModelKind(str, Enum):
    builtin_ncnn = "builtin-ncnn"
    onnx = "onnx"
    diffusion_onnx = "diffusion-onnx"
```

`app/services/model_installer.py` — en `_delete_model_file` (líneas 222-238), después del guard `is_relative_to(models_root)` existente, reemplazar el `target.unlink(missing_ok=True)` final por:

```python
        if target.is_dir():
            shutil.rmtree(target, ignore_errors=True)
        else:
            target.unlink(missing_ok=True)
```

Agregar `import shutil` arriba si falta.

- [ ] **Step 4: correr tests — pasan + sin regresiones**

```powershell
python -m pytest tests/test_model_installer.py tests/test_model_registry.py -v
```

Expected: PASS todos (los dos nuevos + suite existente intacta).

- [ ] **Step 5: commit**

```powershell
git add app/services/model_registry.py app/services/model_installer.py tests/test_model_installer.py
git commit -m "feat: ModelKind.diffusion_onnx y borrado recursivo de carpetas de modelo"
```

---

### Task 3: Cap de descarga de generación + `max_bytes` en `HfClient.download`

Los pesos de un pipeline de difusión pueden superar el cap per-file de 2GB de upscalers (`max_model_download_mb`). Cap propio para generación + override por llamada en `HfClient`.

**Files:**
- Modify: `app/config.py:356-359` (campos Settings)
- Modify: `app/services/hf_client.py:218-248` (`download`)
- Test: `tests/test_hf_client.py` (o donde vivan los tests de `HfClient` — buscar `HfDownloadTooLargeError` en `tests/`), `tests/test_config.py` si existe

**Interfaces:**
- Produces: `Settings.max_generation_model_download_mb: int` (default 8192); `HfClient.download(..., max_bytes: int | None = None)` — `None` mantiene el comportamiento actual (cap de `settings.max_model_download_mb`). Consumido por Task 5.

- [ ] **Step 1: tests (fallan)**

En el archivo de tests que ya cubre `HfClient.download` (localizarlo con `grep -r "HfDownloadTooLargeError" tests/`), agregar, copiando el fixture de transport/mock del test de cap existente:

```python
def test_settings_generation_download_cap_defaults_to_8192(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    assert settings.max_generation_model_download_mb == 8192


@pytest.mark.anyio
async def test_download_honors_max_bytes_override(tmp_path: Path) -> None:
    # mismo arnés del test de cap existente, pero con settings.max_model_download_mb
    # generoso y un max_bytes explícito chico: debe cortar por el override.
    client = make_client_with_body(tmp_path, body=b"x" * 2048, max_model_download_mb=10)
    with pytest.raises(HfDownloadTooLargeError):
        await client.download("owner/name", "file.onnx", tmp_path / "file.onnx", max_bytes=1024)
```

(`make_settings`/`make_client_with_body`: usar los helpers reales del archivo; si no existen con ese nombre, copiar el arreglo del test de "too large" vecino y parametrizarlo.)

- [ ] **Step 2: correr — fallan**

```powershell
python -m pytest tests/ -k "generation_download_cap or max_bytes_override" -v
```

Expected: FAIL (`AttributeError` en Settings; `TypeError: unexpected keyword argument 'max_bytes'`).

- [ ] **Step 3: implementar**

`app/config.py` — junto a `max_model_download_mb` (línea ~358):

```python
    max_generation_model_download_mb: int = 8192
```

`app/services/hf_client.py` — firma de `download` (línea 218):

```python
    async def download(
        self,
        repo_id: str,
        filename: str,
        dest: Path,
        progress_cb: ProgressCallback | None = None,
        max_bytes: int | None = None,
    ) -> Path:
```

y donde hoy calcula `max_bytes = self.settings.max_model_download_mb * 1024 * 1024` (línea 227):

```python
        if max_bytes is None:
            max_bytes = self.settings.max_model_download_mb * 1024 * 1024
```

- [ ] **Step 4: correr — pasan + suite HfClient sin regresión**

```powershell
python -m pytest tests/ -k "hf_client or hf" -v
```

Expected: PASS.

- [ ] **Step 5: commit**

```powershell
git add app/config.py app/services/hf_client.py tests/
git commit -m "feat: cap de descarga para modelos de generacion y max_bytes en HfClient"
```

---

### Task 4: `GenerationEngine` + dependencia optimum

**Files:**
- Modify: `pyproject.toml` (dependencia con pin del findings doc de Task 1)
- Create: `app/services/engines/generation_onnx.py`
- Test: `tests/test_generation_engine.py`

**Interfaces:**
- Consumes: `_build_providers`, `_wrap_onnx_error` de `app/services/engines/onnx_upscaler.py`; `_tune_session_options_for_device` de `app/services/engines/gmfss_engine.py`; `GpuSessionCoordinator` (`acquire(device, owner)` / protocolo `release_device`); findings de Task 1 (clase, kwargs, generator).
- Produces (Tasks 5, 8, 9 dependen de esto):
  - `generation_dependencies_available() -> tuple[bool, str | None]`
  - `CUDA_ONLY_MESSAGE: str`, `VRAM_MESSAGE: str`, `CPU_ONLY_WARNING: str`
  - `_wrap_generation_error(exc: Exception) -> RuntimeError`
  - `_load_pipeline_class() -> Any`
  - `@dataclass GenerationRequest(prompt, negative_prompt, steps, guidance, width, height, seed)`
  - `class GenerationEngine(settings, gpu_coordinator)` con `release_device(device: str) -> None` y `async run(*, model_id: str, pipeline_dir: Path, request: GenerationRequest, device: str, output_path: Path, progress_cb: Callable[[int, int], None]) -> Path`

- [ ] **Step 1: agregar dependencia e instalar**

En `pyproject.toml`, sección de dependencias, agregar EXACTAMENTE (pins del findings doc de Task 1, §a/§b — NUNCA el extra `optimum[onnxruntime]`: ese extra arrastra `onnxruntime` vanilla que pisa `onnxruntime-directml`):

```toml
    "optimum==2.1.0",
    "optimum-onnx==0.1.0",
    "transformers>=4.36,<4.58",
    "diffusers>=0.30,<1.0",
```

(`optimum==2.2.0` es incompatible con `optimum-onnx==0.1.0`; `transformers>=4.58` rompe `optimum-onnx` — ambos conflictos verificados con `pip check` en el spike.) Instalar en el venv del repo (`pip install optimum==2.1.0 optimum-onnx==0.1.0 "transformers>=4.36,<4.58" "diffusers>=0.30,<1.0"` — sin extras, el orden no importa porque ninguno declara onnxruntime como dep base) y verificar:

```powershell
python -c "import onnxruntime as ort; assert ort.get_available_providers().__contains__('DmlExecutionProvider'), ort.get_available_providers(); import optimum.onnxruntime; print('OK')"
```

Expected: `OK`. Si acá se pisa `-directml`, aplicar el fix documentado en el findings (uninstall vanilla + force-reinstall) antes de seguir.

- [ ] **Step 2: tests del engine (fallan)**

`tests/test_generation_engine.py`:

```python
from __future__ import annotations

import asyncio
import threading
from pathlib import Path
from typing import Any, Callable

import pytest

from app.config import Settings
from app.services.engines.generation_onnx import (
    CUDA_ONLY_MESSAGE,
    GenerationEngine,
    GenerationRequest,
    VRAM_MESSAGE,
    _wrap_generation_error,
)


def make_settings(tmp_path: Path) -> Settings:
    return Settings(RUNTIME_DIR=str(tmp_path), _env_file=None)


def make_request(**overrides: Any) -> GenerationRequest:
    defaults: dict[str, Any] = {
        "prompt": "a red apple",
        "negative_prompt": None,
        "steps": 4,
        "guidance": 7.5,
        "width": 256,
        "height": 256,
        "seed": None,
    }
    defaults.update(overrides)
    return GenerationRequest(**defaults)


class FakeImage:
    def save(self, path: Path) -> None:
        Path(path).write_bytes(b"png")


class FakeResult:
    def __init__(self) -> None:
        self.images = [FakeImage()]


class FakePipeline:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def __call__(self, **kwargs: Any) -> FakeResult:
        self.calls.append(kwargs)
        callback = kwargs.get("callback")
        if callback is not None:
            for step in range(kwargs["num_inference_steps"]):
                callback(step, None, None)
        return FakeResult()


class RecordingCoordinator:
    def __init__(self) -> None:
        self.acquired: list[tuple[str, Any]] = []

    def acquire(self, device: str, owner: Any) -> None:
        self.acquired.append((device, owner))


def make_engine(tmp_path: Path, pipeline: Any | None = None) -> tuple[GenerationEngine, RecordingCoordinator, FakePipeline]:
    coordinator = RecordingCoordinator()
    engine = GenerationEngine(make_settings(tmp_path), coordinator)  # type: ignore[arg-type]
    fake = pipeline or FakePipeline()
    engine._create_pipeline = lambda pipeline_dir, device: fake  # type: ignore[method-assign]
    return engine, coordinator, fake


@pytest.mark.anyio
async def test_run_generates_png_and_reports_progress(tmp_path: Path) -> None:
    engine, coordinator, fake = make_engine(tmp_path)
    output = tmp_path / "out.png"
    progress: list[tuple[int, int]] = []

    result = await engine.run(
        model_id="gen--amd--sd15",
        pipeline_dir=tmp_path,
        request=make_request(),
        device="dml:0",
        output_path=output,
        progress_cb=lambda done, total: progress.append((done, total)),
    )

    assert result == output
    assert output.read_bytes() == b"png"
    assert progress == [(1, 4), (2, 4), (3, 4), (4, 4)]
    assert coordinator.acquired == [("dml:0", engine)]
    call = fake.calls[0]
    assert call["prompt"] == "a red apple"
    assert call["num_inference_steps"] == 4
    assert call["width"] == 256 and call["height"] == 256


@pytest.mark.anyio
async def test_run_passes_seeded_generator(tmp_path: Path) -> None:
    engine, _coordinator, fake = make_engine(tmp_path)

    await engine.run(
        model_id="m",
        pipeline_dir=tmp_path,
        request=make_request(seed=42),
        device="cpu",
        output_path=tmp_path / "o.png",
        progress_cb=lambda *_: None,
    )

    assert "generator" in fake.calls[0]


@pytest.mark.anyio
async def test_pipeline_cache_is_lru_one_across_devices(tmp_path: Path) -> None:
    engine, _coordinator, _fake = make_engine(tmp_path)
    created: list[str] = []
    engine._create_pipeline = lambda pipeline_dir, device: created.append(device) or FakePipeline()  # type: ignore[method-assign]

    common = dict(
        model_id="m", pipeline_dir=tmp_path, request=make_request(),
        output_path=tmp_path / "o.png", progress_cb=lambda *_: None,
    )
    await engine.run(device="dml:0", **common)
    await engine.run(device="dml:0", **common)  # cache hit
    await engine.run(device="dml:1", **common)  # evicts dml:0
    await engine.run(device="dml:0", **common)  # rebuilt

    assert created == ["dml:0", "dml:1", "dml:0"]


@pytest.mark.anyio
async def test_release_device_drops_cached_pipeline(tmp_path: Path) -> None:
    engine, _coordinator, _fake = make_engine(tmp_path)
    created: list[str] = []
    engine._create_pipeline = lambda pipeline_dir, device: created.append(device) or FakePipeline()  # type: ignore[method-assign]
    common = dict(
        model_id="m", pipeline_dir=tmp_path, request=make_request(),
        output_path=tmp_path / "o.png", progress_cb=lambda *_: None,
    )
    await engine.run(device="dml:0", **common)

    engine.release_device("dml:0")
    await engine.run(device="dml:0", **common)

    assert created == ["dml:0", "dml:0"]


def test_wrap_generation_error_maps_vram() -> None:
    wrapped = _wrap_generation_error(RuntimeError("DML allocation failed: out of memory"))
    assert VRAM_MESSAGE in str(wrapped)


def test_wrap_generation_error_maps_cuda_only() -> None:
    wrapped = _wrap_generation_error(RuntimeError("CUDAExecutionProvider is not available"))
    assert CUDA_ONLY_MESSAGE in str(wrapped)


@pytest.mark.anyio
async def test_run_wraps_pipeline_errors(tmp_path: Path) -> None:
    class ExplodingPipeline:
        def __call__(self, **kwargs: Any) -> Any:
            raise RuntimeError("CUDAExecutionProvider is not available")

    engine, _coordinator, _fake = make_engine(tmp_path, pipeline=ExplodingPipeline())
    with pytest.raises(RuntimeError, match="requiere GPU NVIDIA"):
        await engine.run(
            model_id="m", pipeline_dir=tmp_path, request=make_request(),
            device="dml:0", output_path=tmp_path / "o.png", progress_cb=lambda *_: None,
        )
```

(Si la suite usa `@pytest.mark.asyncio` u otro estilo async, copiar el del test vecino más parecido — `tests/test_gmfss_engine.py`.)

- [ ] **Step 3: correr — fallan por módulo inexistente**

```powershell
python -m pytest tests/test_generation_engine.py -v
```

Expected: FAIL `ModuleNotFoundError: app.services.engines.generation_onnx`.

- [ ] **Step 4: implementar `app/services/engines/generation_onnx.py`**

```python
from __future__ import annotations

import asyncio
import contextlib
import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from app.config import Settings
from app.services.engines.gmfss_engine import _tune_session_options_for_device
from app.services.engines.onnx_upscaler import _build_providers
from app.services.gpu_session_coordinator import GpuSessionCoordinator

GENERATION_IMPORT_ERROR_HINT = (
    "Las dependencias de generación no están instaladas (paquete optimum). "
    "Reinstalá Upflow o corré la receta de docs/superpowers/specs/2026-07-22-optimum-spike-findings.md."
)

CUDA_ONLY_MESSAGE = (
    "Este modelo requiere GPU NVIDIA (CUDA) y no es compatible con DirectML en tu hardware. "
    "Buscá una versión compatible (ej. colección `amd/` en Hugging Face, formato ONNX+DirectML)."
)

VRAM_MESSAGE = (
    "Sin memoria de GPU para generar con esta configuración. Probá menor resolución, "
    "un modelo más liviano, o esperá a que terminen otros trabajos de GPU."
)

CPU_ONLY_WARNING = (
    "No se detectó GPU compatible (DirectX 12). Generar en CPU tarda varios minutos por imagen. "
    "¿Continuar igual?"
)

_MEMORY_TOKENS = ("memory", "alloc", "oom")
_CUDA_TOKENS = ("cudaexecutionprovider", "cuda", "tensorrt")


class GenerationCancelled(Exception):
    pass


def generation_dependencies_available() -> tuple[bool, str | None]:
    try:
        import optimum.onnxruntime  # noqa: F401
    except Exception as exc:  # noqa: BLE001 - cualquier fallo de import = no disponible
        return False, f"{GENERATION_IMPORT_ERROR_HINT} ({exc})"
    return True, None


def _load_pipeline_class() -> Any:
    # Nombre confirmado por el spike (Task 1). Si el findings doc dice otro, cambiarlo SOLO acá.
    from optimum.onnxruntime import ORTStableDiffusionPipeline

    return ORTStableDiffusionPipeline


def _wrap_generation_error(exc: Exception) -> RuntimeError:
    message = str(exc)
    lowered = message.lower()
    if any(token in lowered for token in _MEMORY_TOKENS):
        return RuntimeError(f"{VRAM_MESSAGE} ({message})")
    if any(token in lowered for token in _CUDA_TOKENS):
        return RuntimeError(f"{CUDA_ONLY_MESSAGE} ({message})")
    return RuntimeError(f"Image generation failed: {message}")


def _build_seed_generator(seed: int) -> Any:
    # torch.Generator, NO np.random.RandomState: __call__ es el de diffusers y
    # randn_tensor accede a generator.device (findings §d/§e, verificado empirico).
    import torch

    return torch.Generator(device="cpu").manual_seed(seed)


@dataclass(slots=True, kw_only=True)
class GenerationRequest:
    prompt: str
    negative_prompt: str | None
    steps: int
    guidance: float
    width: int
    height: int
    seed: int | None


class GenerationEngine:
    def __init__(self, settings: Settings, gpu_coordinator: GpuSessionCoordinator) -> None:
        self.settings = settings
        self.gpu_coordinator = gpu_coordinator
        self._pipeline_cache: OrderedDict[tuple[str, str], Any] = OrderedDict()
        self._cache_lock = threading.Lock()

    def release_device(self, device: str) -> None:
        with self._cache_lock:
            stale_keys = [key for key in self._pipeline_cache if key[1] == device]
            for key in stale_keys:
                self._pipeline_cache.pop(key, None)

    async def run(
        self,
        *,
        model_id: str,
        pipeline_dir: Path,
        request: GenerationRequest,
        device: str,
        output_path: Path,
        progress_cb: Callable[[int, int], None],
    ) -> Path:
        cancel_event = threading.Event()
        worker = asyncio.ensure_future(
            asyncio.to_thread(
                self._run_blocking,
                model_id,
                pipeline_dir,
                request,
                device,
                output_path,
                cancel_event,
                progress_cb,
            )
        )
        try:
            return await asyncio.shield(worker)
        except asyncio.CancelledError:
            cancel_event.set()
            with contextlib.suppress(BaseException):
                await worker
            raise

    def _run_blocking(
        self,
        model_id: str,
        pipeline_dir: Path,
        request: GenerationRequest,
        device: str,
        output_path: Path,
        cancel_event: threading.Event,
        progress_cb: Callable[[int, int], None],
    ) -> Path:
        pipeline = self._get_pipeline(model_id, pipeline_dir, device)

        def _on_step(step: int, _timestep: Any, _latents: Any) -> None:
            if cancel_event.is_set():
                raise GenerationCancelled()
            progress_cb(step + 1, request.steps)

        call_kwargs: dict[str, Any] = {
            "prompt": request.prompt,
            "num_inference_steps": request.steps,
            "guidance_scale": request.guidance,
            "width": request.width,
            "height": request.height,
            "callback": _on_step,
            "callback_steps": 1,
        }
        if request.negative_prompt:
            call_kwargs["negative_prompt"] = request.negative_prompt
        if request.seed is not None:
            call_kwargs["generator"] = _build_seed_generator(request.seed)

        try:
            result = pipeline(**call_kwargs)
        except GenerationCancelled:
            raise
        except Exception as exc:
            raise _wrap_generation_error(exc) from exc

        output_path.parent.mkdir(parents=True, exist_ok=True)
        result.images[0].save(output_path)
        return output_path

    def _get_pipeline(self, model_id: str, pipeline_dir: Path, device: str) -> Any:
        self.gpu_coordinator.acquire(device, self)
        key = (model_id, device)
        with self._cache_lock:
            cached = self._pipeline_cache.get(key)
            if cached is not None:
                self._pipeline_cache.move_to_end(key)
                return cached
        try:
            pipeline = self._create_pipeline(pipeline_dir, device)
        except Exception as exc:
            raise _wrap_generation_error(exc) from exc
        with self._cache_lock:
            self._pipeline_cache[key] = pipeline
            self._pipeline_cache.move_to_end(key)
            while len(self._pipeline_cache) > 1:
                self._pipeline_cache.popitem(last=False)
        return pipeline

    def _create_pipeline(self, pipeline_dir: Path, device: str) -> Any:
        import onnxruntime as ort

        pipeline_cls = _load_pipeline_class()
        providers = _build_providers(device)
        sess_options = ort.SessionOptions()
        _tune_session_options_for_device(sess_options, device)
        primary = providers[0]
        # use_io_binding=False explicito: hoy es el default para DML pero se
        # blinda ante cambios de default de optimum (IOBinding+DML vetado en este repo).
        from_pretrained_kwargs: dict[str, Any] = {
            "session_options": sess_options,
            "use_io_binding": False,
        }
        if isinstance(primary, tuple):
            provider_name, provider_options = primary
            from_pretrained_kwargs["provider"] = provider_name
            from_pretrained_kwargs["provider_options"] = provider_options
        else:
            from_pretrained_kwargs["provider"] = primary
        return pipeline_cls.from_pretrained(str(pipeline_dir), **from_pretrained_kwargs)
```

Kwargs confirmados por el findings doc (§d): `session_options` aceptado; `callback`/`callback_steps` funcionan en diffusers 0.39.0 (deprecados, FutureWarning una vez por proceso — aceptado en MVP por ser la ruta VERIFICADA empíricamente; no migrar a `callback_on_step_end` en esta task).

- [ ] **Step 5: correr — pasan**

```powershell
python -m pytest tests/test_generation_engine.py -v
```

Expected: PASS los 8.

- [ ] **Step 6: commit**

```powershell
git add pyproject.toml app/services/engines/generation_onnx.py tests/test_generation_engine.py
git commit -m "feat: GenerationEngine ONNX DirectML con cache LRU(1) y errores especificos por hardware"
```

---

### Task 5: `GenerationModelInstaller`

**Files:**
- Create: `app/services/generation_installer.py`
- Create: `app/assets/generation/sd15_legacy_configs/{unet,text_encoder,vae_decoder,vae_encoder,safety_checker}/config.json` — vendorizados una sola vez desde `sd-legacy/stable-diffusion-v1-5` en Hugging Face (`unet/config.json`, `text_encoder/config.json`, `safety_checker/config.json`, y `vae/config.json` copiado DOS veces: a `vae_decoder/` y `vae_encoder/`). Son KBs estáticos; descargarlos con `curl`/`Invoke-WebRequest` de `https://huggingface.co/sd-legacy/stable-diffusion-v1-5/raw/main/<path>` y commitearlos. Razón: los repos `amd/` legacy no traen `config.json` por componente y `optimum-onnx` lo exige (findings doc, hallazgo bloqueante).
- Test: `tests/test_generation_installer.py`

**Interfaces:**
- Consumes: `InstallJob`, `InstallStatus`, `_validate_repo_id`, `PROMOTE_RETRY_DELAYS_SECONDS` de `app/services/model_installer.py`; `HfClient` (`repo_files`, `download(..., max_bytes=...)` de Task 3); `ModelRegistry`/`ModelEntry`/`ModelKind.diffusion_onnx` (Task 2); `_load_pipeline_class`, `_wrap_generation_error`, `generation_dependencies_available` de Task 4; `_build_providers` de `onnx_upscaler`.
- Produces (Task 9 depende): `class GenerationModelInstaller(settings, registry, hf_client)` con `async start()`, `async stop()`, `async install_from_hf(repo_id: str) -> str`, `status(install_id: str) -> InstallJob | None`. Los modelos quedan en `models_path/generation/{model_id}/` con `file_path="generation/{model_id}"`. `_generation_model_id("amd/sd15") == "gen--amd--sd15"`.

- [ ] **Step 1: tests (fallan)**

`tests/test_generation_installer.py` — reusar `FakeHfClient` de `tests/test_model_installer.py` (importarlo o copiarlo si no es importable; su `download` ya escribe bytes en `dest`):

```python
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from app.config import Settings
from app.services.generation_installer import (
    GenerationModelInstaller,
    _generation_model_id,
    _select_files,
)
from app.services.hf_client import HfFile
from app.services.model_installer import InstallStatus
from app.services.model_registry import ModelKind, ModelRegistry

MODEL_INDEX = json.dumps(
    {
        "_class_name": "OnnxStableDiffusionPipeline",
        "text_encoder": ["diffusers", "OnnxRuntimeModel"],
        "unet": ["diffusers", "OnnxRuntimeModel"],
        "vae_decoder": ["diffusers", "OnnxRuntimeModel"],
        "tokenizer": ["transformers", "CLIPTokenizer"],
        "scheduler": ["diffusers", "PNDMScheduler"],
    }
)

PIPELINE_FILES = [
    HfFile(path="model_index.json", size=len(MODEL_INDEX)),
    HfFile(path="text_encoder/model.onnx", size=10),
    HfFile(path="unet/model.onnx", size=10),
    HfFile(path="vae_decoder/model.onnx", size=10),
    HfFile(path="tokenizer/tokenizer_config.json", size=5),
    HfFile(path="scheduler/scheduler_config.json", size=5),
    HfFile(path="v1-5-pruned.ckpt", size=4_000_000_000),  # duplicado torch: debe saltearse
    HfFile(path="MXR/unet.mxr", size=5_000_000_000),  # binarios MIGraphX: carpeta NO declarada, debe saltearse
]


def make_settings(tmp_path: Path, **overrides: object) -> Settings:
    return Settings(RUNTIME_DIR=str(tmp_path), _env_file=None, **overrides)


def make_installer(tmp_path: Path, files: list[HfFile], **hf_kwargs: Any):
    settings = make_settings(tmp_path)
    registry = ModelRegistry(settings)
    hf = FakeHfClient(files=files, **hf_kwargs)
    installer = GenerationModelInstaller(settings, registry, hf)
    # el download fake debe escribir el model_index real para la validación estructural:
    hf.download_bytes_by_path = {"model_index.json": MODEL_INDEX.encode("utf-8")}
    return installer, registry, settings, hf


def install_and_drain(installer: GenerationModelInstaller, repo_id: str):
    async def _run():
        install_id = await installer.install_from_hf(repo_id)
        await installer._process_next()
        return installer.status(install_id)

    import asyncio

    return asyncio.run(_run())


def test_generation_model_id_is_prefixed_and_safe() -> None:
    assert _generation_model_id("amd/Stable-Diffusion-1.5") == "gen--amd--stable-diffusion-1.5"


def test_select_files_skips_torch_checkpoints() -> None:
    selected = _select_files(PIPELINE_FILES)
    assert all(not f.path.endswith((".ckpt", ".pth", ".safetensors", ".bin")) for f in selected)
    assert any(f.path == "unet/model.onnx" for f in selected)


def test_filter_to_declared_drops_undeclared_dirs_and_model_index() -> None:
    declared = ["text_encoder", "unet", "vae_decoder", "tokenizer", "scheduler"]
    kept = _filter_to_declared(_select_files(PIPELINE_FILES), declared)
    paths = [f.path for f in kept]
    assert "MXR/unet.mxr" not in paths          # carpeta no declarada
    assert "model_index.json" not in paths       # se baja aparte, en fase 1
    assert "unet/model.onnx" in paths
    assert "tokenizer/tokenizer_config.json" in paths


def test_install_happy_path_registers_diffusion_model(tmp_path: Path, monkeypatch) -> None:
    installer, registry, settings, hf = make_installer(tmp_path, files=PIPELINE_FILES)
    monkeypatch.setattr(installer, "_create_validation_pipeline", lambda pipeline_dir: FakeValidationPipeline())

    job = install_and_drain(installer, "amd/sd15")

    assert job.status == InstallStatus.installed
    entry = registry.get("gen--amd--sd15")
    assert entry is not None
    assert entry.kind == ModelKind.diffusion_onnx
    assert entry.scale is None
    assert entry.file_path == "generation/gen--amd--sd15"
    final_dir = settings.models_path / "generation" / "gen--amd--sd15"
    assert (final_dir / "model_index.json").is_file()
    # patch legacy: _class_name == OnnxStableDiffusionPipeline y los componentes
    # no traian config.json -> el installer los completa desde los vendorizados
    assert (final_dir / "unet" / "config.json").is_file()
    assert (final_dir / "text_encoder" / "config.json").is_file()
    # MXR/ no declarado: nunca se descargo
    assert not any("MXR" in call for call in map(str, hf.download_calls))


def test_install_rejects_repo_without_model_index(tmp_path: Path) -> None:
    files = [HfFile(path="model.onnx", size=10)]
    installer, registry, _settings, hf = make_installer(tmp_path, files=files)

    job = install_and_drain(installer, "someone/upscaler")

    assert job.status == InstallStatus.error
    assert "model_index.json" in (job.error or "")
    assert hf.download_calls == []  # error ANTES de bajar gigas
    assert registry.get("gen--someone--upscaler") is None


def test_install_rejects_when_total_size_exceeds_cap(tmp_path: Path) -> None:
    big = [
        HfFile(path="model_index.json", size=len(MODEL_INDEX)),
        HfFile(path="unet/model.onnx", size=9 * 1024 * 1024 * 1024),
    ]
    installer, _registry, _settings, hf = make_installer(tmp_path, files=big)

    job = install_and_drain(installer, "amd/sdxl-huge")

    assert job.status == InstallStatus.error
    # el cap se chequea DESPUES de bajar model_index.json (fase 1, KBs) pero
    # ANTES de bajar cualquier peso:
    assert [str(c) for c in hf.download_calls if "model_index" not in str(c)] == []


def test_install_cuda_only_model_fails_with_friendly_message(tmp_path: Path, monkeypatch) -> None:
    installer, registry, _settings, _hf = make_installer(tmp_path, files=PIPELINE_FILES)

    def explode(pipeline_dir: Path) -> Any:
        raise RuntimeError("CUDAExecutionProvider is not in available providers")

    monkeypatch.setattr(installer, "_create_validation_pipeline", explode)

    job = install_and_drain(installer, "tlwu/sdxl-cuda-only")

    assert job.status == InstallStatus.error
    assert "requiere GPU NVIDIA" in (job.error or "")
    assert registry.get("gen--tlwu--sdxl-cuda-only") is None


class FakeValidationPipeline:
    def __call__(self, **kwargs: Any) -> Any:
        class _R:
            images = [object()]

        return _R()
```

Notas de adaptación (resolver al escribir, son detalles del fake existente): si `FakeHfClient` no soporta `download_bytes_by_path`, extenderlo en este archivo con una subclase que escriba `MODEL_INDEX` cuando `filename == "model_index.json"` y `b"onnx"` para el resto; `download_calls` es el tracking de llamadas que el fake ya tiene (usar su nombre real). Además la subclase debe aceptar el kwarg `max_bytes=None` en `download(...)` (agregado en Task 3) — el fake original no lo tiene.

- [ ] **Step 2: correr — fallan**

```powershell
python -m pytest tests/test_generation_installer.py -v
```

Expected: FAIL `ModuleNotFoundError: app.services.generation_installer`.

- [ ] **Step 3: implementar `app/services/generation_installer.py`**

```python
from __future__ import annotations

import asyncio
import contextlib
import gc
import json
import shutil
import uuid
from pathlib import Path
from typing import Any

from app.config import Settings
from app.services.engines.generation_onnx import (
    _build_providers_for_validation,
    _load_pipeline_class,
    _wrap_generation_error,
    generation_dependencies_available,
)
from app.services.hf_client import HfClient, HfFile
from app.services.model_installer import (
    InstallJob,
    InstallStatus,
    PROMOTE_RETRY_DELAYS_SECONDS,
    _validate_repo_id,
)
from app.services.model_registry import ModelEntry, ModelKind, ModelRegistry, ModelStatus

MODEL_INDEX_FILENAME = "model_index.json"
GENERATION_MODELS_SUBDIR = "generation"
SKIP_WEIGHT_SUFFIXES = (".ckpt", ".pth", ".safetensors", ".bin", ".msgpack", ".h5")
LEGACY_PIPELINE_CLASS = "OnnxStableDiffusionPipeline"
LEGACY_CONFIGS_ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets" / "generation" / "sd15_legacy_configs"
VALIDATION_PROMPT = "validation"
VALIDATION_SIZE = 64
VALIDATION_STEPS = 1


def _generation_model_id(repo_id: str) -> str:
    return "gen--" + repo_id.lower().replace("/", "--")


def _select_files(files: list[HfFile]) -> list[HfFile]:
    return [f for f in files if not f.path.lower().endswith(SKIP_WEIGHT_SUFFIXES)]


def _read_declared_components(staging_root: Path) -> list[str]:
    index = json.loads((staging_root / MODEL_INDEX_FILENAME).read_text(encoding="utf-8"))
    return [
        name
        for name, value in index.items()
        if not name.startswith("_") and isinstance(value, list)
    ]


def _filter_to_declared(files: list[HfFile], declared: list[str]) -> list[HfFile]:
    # Solo componentes declarados en model_index + metadata chica top-level.
    # Evita bajar carpetas ajenas al pipeline (ej. MXR/ binarios MIGraphX ~GBs,
    # controlnet/ no declarado) presentes en los repos amd/ (findings, repo id real).
    kept: list[HfFile] = []
    for hf_file in files:
        if hf_file.path == MODEL_INDEX_FILENAME:
            continue  # se descarga aparte, antes que el resto
        top_segment = hf_file.path.split("/", 1)[0]
        if "/" in hf_file.path:
            if top_segment in declared:
                kept.append(hf_file)
        elif hf_file.path.lower().endswith((".json", ".txt")):
            kept.append(hf_file)
    return kept


def _patch_legacy_component_configs(staging_root: Path) -> None:
    # Los repos amd/ legacy (_class_name: OnnxStableDiffusionPipeline) no traen
    # config.json por componente y optimum-onnx lo exige (findings, hallazgo
    # bloqueante). Se completan desde los configs SD1.5 vendorizados en
    # app/assets/generation/sd15_legacy_configs/. Solo para esa clase legacy:
    # otros layouts o traen sus configs o fallan la validacion funcional con
    # mensaje accionable.
    index = json.loads((staging_root / MODEL_INDEX_FILENAME).read_text(encoding="utf-8"))
    if index.get("_class_name") != LEGACY_PIPELINE_CLASS:
        return
    for component in _read_declared_components(staging_root):
        component_dir = staging_root / component
        config_path = component_dir / "config.json"
        vendored = LEGACY_CONFIGS_ASSETS_DIR / component / "config.json"
        if component_dir.is_dir() and not config_path.exists() and vendored.is_file():
            shutil.copyfile(vendored, config_path)


def _ensure_model_index_listed(files: list[HfFile], repo_id: str) -> None:
    if not any(f.path == MODEL_INDEX_FILENAME for f in files):
        raise ValueError(
            f"El repo {repo_id!r} no parece un pipeline diffusers ONNX: falta {MODEL_INDEX_FILENAME}."
        )


def _ensure_size_cap(files: list[HfFile], cap_mb: int) -> None:
    total = sum(f.size for f in files)
    if total > cap_mb * 1024 * 1024:
        raise ValueError(
            f"La descarga ({total // (1024 * 1024)} MB) supera el límite de {cap_mb} MB "
            "(MAX_GENERATION_MODEL_DOWNLOAD_MB)."
        )


def _validate_structure(staging_root: Path) -> None:
    index_path = staging_root / MODEL_INDEX_FILENAME
    if not index_path.is_file():
        raise ValueError(f"Descarga incompleta: falta {MODEL_INDEX_FILENAME}.")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    declared = [
        name
        for name, value in index.items()
        if not name.startswith("_") and isinstance(value, list)
    ]
    missing = sorted(name for name in declared if not (staging_root / name).is_dir())
    if missing:
        raise ValueError(f"Faltan componentes del pipeline en el repo: {', '.join(missing)}.")


class GenerationModelInstaller:
    def __init__(self, settings: Settings, registry: ModelRegistry, hf_client: HfClient) -> None:
        self.settings = settings
        self.registry = registry
        self.hf_client = hf_client
        self._queue: asyncio.Queue[InstallJob] = asyncio.Queue()
        self._jobs: dict[str, InstallJob] = {}
        self._worker_task: asyncio.Task | None = None
        self._model_locks: dict[str, asyncio.Lock] = {}

    async def start(self) -> None:
        if self._worker_task is None:
            self._worker_task = asyncio.create_task(
                self._worker(), name="generation-install-worker"
            )

    async def stop(self) -> None:
        if self._worker_task is not None:
            self._worker_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._worker_task
            self._worker_task = None

    async def install_from_hf(self, repo_id: str) -> str:
        available, reason = generation_dependencies_available()
        if not available:
            raise ValueError(reason or "Generation dependencies missing")
        validated = _validate_repo_id(repo_id)
        job = InstallJob(id=uuid.uuid4().hex, repo_id=validated)
        self._jobs[job.id] = job
        await self._queue.put(job)
        return job.id

    def status(self, install_id: str) -> InstallJob | None:
        return self._jobs.get(install_id)

    def _lock_for(self, model_id: str) -> asyncio.Lock:
        lock = self._model_locks.get(model_id)
        if lock is None:
            lock = asyncio.Lock()
            self._model_locks[model_id] = lock
        return lock

    async def _worker(self) -> None:
        while True:
            job = await self._queue.get()
            await self._run_install(job)
            self._queue.task_done()

    async def _process_next(self) -> bool:
        try:
            job = self._queue.get_nowait()
        except asyncio.QueueEmpty:
            return False
        await self._run_install(job)
        self._queue.task_done()
        return True

    async def _run_install(self, job: InstallJob) -> None:
        try:
            await self._download_and_register(job)
        except Exception as exc:  # noqa: BLE001 - el job reporta cualquier fallo
            job.status = InstallStatus.error
            job.error = str(exc)

    async def _download_and_register(self, job: InstallJob) -> None:
        files = await self.hf_client.repo_files(job.repo_id)
        _ensure_model_index_listed(files, job.repo_id)

        model_id = _generation_model_id(job.repo_id)
        staging_root = self.settings.temp_path / f"gen-staging-{model_id}"
        if staging_root.exists():
            shutil.rmtree(staging_root, ignore_errors=True)
        staging_root.mkdir(parents=True, exist_ok=True)

        max_file_bytes = self.settings.max_generation_model_download_mb * 1024 * 1024
        job.status = InstallStatus.downloading
        try:
            # Fase 1: model_index.json primero (KBs) para conocer los componentes
            # declarados y filtrar la descarga a lo que el pipeline realmente usa.
            await self.hf_client.download(
                job.repo_id,
                MODEL_INDEX_FILENAME,
                staging_root / MODEL_INDEX_FILENAME,
                max_bytes=max_file_bytes,
            )
            declared = _read_declared_components(staging_root)
            selected = _filter_to_declared(_select_files(files), declared)
            _ensure_size_cap(selected, self.settings.max_generation_model_download_mb)

            total_bytes = sum(f.size for f in selected) or 1
            downloaded_bytes = 0
            for hf_file in selected:
                dest = staging_root / hf_file.path
                dest.parent.mkdir(parents=True, exist_ok=True)
                await self.hf_client.download(
                    job.repo_id, hf_file.path, dest, max_bytes=max_file_bytes
                )
                downloaded_bytes += hf_file.size
                job.progress_pct = round(downloaded_bytes / total_bytes * 100, 1)

            _validate_structure(staging_root)
            _patch_legacy_component_configs(staging_root)
            job.status = InstallStatus.validating
            await asyncio.to_thread(self._validate_pipeline, staging_root)

            final_dir = (
                self.settings.models_path / GENERATION_MODELS_SUBDIR / model_id
            )
            async with self._lock_for(model_id):
                await self._promote_staging_dir(staging_root, final_dir)
                entry = ModelEntry(
                    id=model_id,
                    name=job.repo_id,
                    kind=ModelKind.diffusion_onnx,
                    source=f"hf:{job.repo_id}",
                    size_bytes=sum(f.size for f in selected),
                    scale=None,
                    file_path=f"{GENERATION_MODELS_SUBDIR}/{model_id}",
                    status=ModelStatus.installed,
                )
                self.registry.register(entry)
            job.model_id = model_id
            job.status = InstallStatus.installed
        finally:
            if staging_root.exists():
                shutil.rmtree(staging_root, ignore_errors=True)

    async def _promote_staging_dir(self, staging_root: Path, final_dir: Path) -> None:
        final_dir.parent.mkdir(parents=True, exist_ok=True)
        if final_dir.exists():
            shutil.rmtree(final_dir)
        last_error: Exception | None = None
        for delay in (0.0, *PROMOTE_RETRY_DELAYS_SECONDS):
            if delay:
                await asyncio.sleep(delay)
            try:
                staging_root.replace(final_dir)
                return
            except PermissionError as exc:
                last_error = exc
        raise RuntimeError(f"Could not promote generation model into place: {last_error}")

    def _validate_pipeline(self, pipeline_dir: Path) -> None:
        pipeline = None
        try:
            pipeline = self._create_validation_pipeline(pipeline_dir)
            pipeline(
                prompt=VALIDATION_PROMPT,
                num_inference_steps=VALIDATION_STEPS,
                width=VALIDATION_SIZE,
                height=VALIDATION_SIZE,
            )
        except Exception as exc:
            raise _wrap_generation_error(exc) from exc
        finally:
            del pipeline
            gc.collect()

    def _create_validation_pipeline(self, pipeline_dir: Path) -> Any:
        pipeline_cls = _load_pipeline_class()
        kwargs = _build_providers_for_validation(self.settings.default_device)
        return pipeline_cls.from_pretrained(str(pipeline_dir), **kwargs)
```

Además, en `app/services/engines/generation_onnx.py`, agregar el helper que este módulo consume (los `from_pretrained` kwargs de validación salen del mismo lugar que los del engine):

```python
def _build_providers_for_validation(device: str) -> dict[str, Any]:
    primary = _build_providers(device)[0]
    kwargs: dict[str, Any] = {"use_io_binding": False}
    if isinstance(primary, tuple):
        provider_name, provider_options = primary
        kwargs.update(provider=provider_name, provider_options=provider_options)
    else:
        kwargs["provider"] = primary
    return kwargs
```

- [ ] **Step 4: correr — pasan**

```powershell
python -m pytest tests/test_generation_installer.py tests/test_generation_engine.py -v
```

Expected: PASS.

- [ ] **Step 5: commit**

```powershell
git add app/services/generation_installer.py app/services/engines/generation_onnx.py tests/test_generation_installer.py
git commit -m "feat: GenerationModelInstaller multi-archivo con validacion estructural y forward-pass"
```

---

### Task 6: Stages de generación en `progress.py`

**Files:**
- Modify: `app/services/progress.py`
- Test: `tests/test_progress.py` (o el archivo que hoy testea `apply_image_tile_progress` — localizar con `grep -r "apply_image_tile_progress" tests/`)

**Interfaces:**
- Consumes: `apply_stage_transition`, `compute_progress`, `frame_stage_fraction`, `build_image_stages` y el dataclass `Stage` existentes en `progress.py` (leer sus firmas reales ANTES de escribir — el constructor de `Stage` se copia de `build_image_stages`).
- Produces (Task 8 depende): `build_generation_stages(include_upscale: bool) -> list[Stage]` (pesos: `generating` 85 / `upscaling` 15; solo `generating` con peso 100 si `include_upscale=False`); `apply_generation_step_progress(job, steps_done: int, steps_total: int, include_upscale: bool) -> None`; `advance_generation_stage(job, stage_key: str, include_upscale: bool) -> None`; `complete_generation_stages(job, include_upscale: bool) -> None`. Todos escriben las MISMAS claves de metadata que las variantes de imagen: `stage`, `stages`, `framesDone`, `framesTotal`, `progress`.

- [ ] **Step 1: leer las funciones de imagen y escribir el test (falla)**

Leer `app/services/progress.py` completo (líneas 1-250). Después, en el archivo de tests de progress:

```python
def test_generation_step_progress_reports_generating_stage() -> None:
    job = make_generation_job()  # helper local: GenerationJob mínimo (Task 8 lo define; hasta entonces usar un dataclass dummy con .metadata dict)
    apply_generation_step_progress(job, steps_done=5, steps_total=25, include_upscale=False)

    assert job.metadata["stage"] == "generating"
    assert job.metadata["framesDone"] == 5
    assert job.metadata["framesTotal"] == 25
    assert 0.0 < job.metadata["progress"] < 1.0
    assert [s["key"] for s in job.metadata["stages"]] == ["generating"]


def test_generation_stages_include_upscaling_when_auto_upscale() -> None:
    job = make_generation_job()
    advance_generation_stage(job, "upscaling", include_upscale=True)

    assert [s["key"] for s in job.metadata["stages"]] == ["generating", "upscaling"]
    assert job.metadata["stage"] == "upscaling"


def test_complete_generation_stages_reaches_full_progress() -> None:
    job = make_generation_job()
    complete_generation_stages(job, include_upscale=True)
    assert job.metadata["progress"] == 1.0
```

Nota: para no depender de Task 8, `make_generation_job()` en este archivo puede ser `SimpleNamespace(metadata={})` — las funciones de progress solo tocan `.metadata`. Si el dict de stage usa otra clave que `"key"` (verificar contra `asdict(stage)` de las de imagen), ajustar los asserts al nombre real.

- [ ] **Step 2: correr — fallan** (`ImportError`).

- [ ] **Step 3: implementar en `app/services/progress.py`**

Replicar la construcción de `build_image_stages` (mismo constructor de `Stage`, mismos campos) con esta tabla:

```python
GENERATION_STAGE_DEFS: tuple[tuple[str, str, float], ...] = (
    ("generating", "Generating", 85.0),
    ("upscaling", "Upscaling", 15.0),
)
```

y las cuatro funciones espejo de las de imagen (`build_generation_stages(include_upscale)` filtra la tabla y, si `include_upscale` es False, deja solo `generating` con peso 100; `apply_generation_step_progress` calcado de `apply_image_tile_progress` con stage `"generating"`; `advance_generation_stage`/`complete_generation_stages` calcadas de `advance_image_stage`/`complete_image_stages` pero recibiendo `include_upscale` y pasando `build_generation_stages(include_upscale)`).

- [ ] **Step 4: correr — pasan** (`python -m pytest tests/ -k generation_stage -v` y la suite de progress entera).

- [ ] **Step 5: commit**

```powershell
git add app/services/progress.py tests/
git commit -m "feat: stages de progreso generating/upscaling"
```

---

### Task 7: Extraer `select_upscale_engine` de `JobManager`

**Files:**
- Modify: `app/services/job_manager.py:185-194` (`_select_engine`)
- Test: `tests/test_job_manager.py` (localizar el archivo real con `grep -r "_select_engine\|JobManager(" tests/ -l`)

**Interfaces:**
- Produces (Task 8 depende): función module-level en `app/services/job_manager.py`:
  `select_upscale_engine(job: UpscaleJob, registry: ModelRegistry | None, builtin_engine: UpscaleEngine, onnx_engine: UpscaleEngine | None) -> UpscaleEngine`

- [ ] **Step 1: test (falla)**

```python
def test_select_upscale_engine_prefers_onnx_for_onnx_model(...) -> None:
    # armar registry con entry kind=onnx y verificar que devuelve onnx_engine;
    # sin model_id devuelve builtin; model_id onnx sin onnx_engine -> RuntimeError.
```

Copiar el arnés (fakes de engine/registry) de los tests existentes de `_select_engine` si los hay; si no, construir: registry real con entry `ModelKind.onnx`, dos `object()` como engines, asserts de identidad. Tres casos: onnx→onnx_engine, sin model_id→builtin, onnx sin engine→`pytest.raises(RuntimeError)`.

- [ ] **Step 2: correr — falla** (`ImportError: select_upscale_engine`).

- [ ] **Step 3: implementar**

Mover el cuerpo de `JobManager._select_engine` (líneas 185-194) a módulo-level:

```python
def select_upscale_engine(
    job: UpscaleJob,
    registry: ModelRegistry | None,
    builtin_engine: UpscaleEngine,
    onnx_engine: UpscaleEngine | None,
) -> UpscaleEngine:
    if job.model_id is not None and registry is not None:
        entry = registry.get(job.model_id)
        if entry is not None and entry.kind == ModelKind.onnx:
            if onnx_engine is None:
                raise RuntimeError(
                    f"Model {job.model_id!r} requires the ONNX engine, which is not configured"
                )
            return onnx_engine
    return builtin_engine
```

y `JobManager._select_engine` queda:

```python
    def _select_engine(self, job: UpscaleJob) -> UpscaleEngine:
        return select_upscale_engine(job, self.registry, self.engine, self.onnx_engine)
```

(Respetar los nombres de atributo reales de `JobManager.__init__` — el builtin se guarda como `self.engine`.)

- [ ] **Step 4: correr — pasan + suite de job_manager sin regresión**

```powershell
python -m pytest tests/ -k "job_manager or select_upscale" -v
```

- [ ] **Step 5: commit**

```powershell
git add app/services/job_manager.py tests/
git commit -m "refactor: extraer select_upscale_engine para reuso desde generacion"
```

---

### Task 8: `GenerationJob` + `GenerationJobManager` + retention

**Files:**
- Modify: `app/models.py` (dataclass), `app/services/retention_sweeper.py`
- Create: `app/services/generation_job_manager.py`
- Test: `tests/test_generation_job_manager.py`, caso nuevo en el test del sweeper (localizar con `grep -r "RetentionSweeper" tests/ -l`)

**Interfaces:**
- Consumes: `GenerationEngine`/`GenerationRequest` (Task 4), `select_upscale_engine` (Task 7), funciones de progress (Task 6), `ModelKind.diffusion_onnx` (Task 2), `DeviceSemaphores` (patrón `async with self.device_semaphores.acquire(job.device)`), `AUTO_DEVICE_ID` de `devices_service`, `QueueFullError` (copiar el import desde `app/services/audio_job_manager.py`).
- Produces (Task 9 depende):
  - `models.py`: `@dataclass(slots=True) GenerationJob` (campos abajo)
  - `GenerationJobManager(settings, engine, device_semaphores, *, registry, upscale_engine, onnx_upscale_engine, devices=None)` con `async start()`, `async stop()`, `queue_depth() -> int`, `async create_job(*, prompt, negative_prompt=None, model_id, steps=25, guidance=7.5, width=512, height=512, seed=None, device=None, auto_upscale=False, upscale_model_name=None, upscale_scale=None, upscale_model_id=None, job_id=None) -> GenerationJob`, `get_job(job_id) -> GenerationJob | None`, `cancel_job(job_id) -> bool`, atributo público `jobs: dict[str, GenerationJob]`.

- [ ] **Step 1: dataclass en `app/models.py`**

Después de `AudioJob` (línea ~133):

```python
@dataclass(slots=True)
class GenerationJob:
    prompt: str
    model_id: str
    negative_prompt: str | None = None
    steps: int = 25
    guidance: float = 7.5
    width: int = 512
    height: int = 512
    seed: int | None = None
    device: str | None = None
    auto_upscale: bool = False
    upscale_model_name: str | None = None
    upscale_scale: int | None = None
    upscale_model_id: str | None = None
    id: str = field(default_factory=lambda: uuid4().hex)
    status: JobStatus = JobStatus.queued
    created_at: datetime = field(default_factory=utc_now)
    started_at: datetime | None = None
    finished_at: datetime | None = None
    error: str | None = None
    output_path: Path | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
```

- [ ] **Step 2: tests del manager (fallan)**

`tests/test_generation_job_manager.py`:

```python
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from app.config import Settings
from app.models import GenerationJob, JobStatus
from app.services.generation_job_manager import GenerationJobManager
from app.services.model_registry import ModelEntry, ModelKind, ModelRegistry


def make_settings(tmp_path: Path) -> Settings:
    return Settings(RUNTIME_DIR=str(tmp_path), _env_file=None)


class FakeGenerationEngine:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def run(self, **kwargs: Any) -> Path:
        self.calls.append(kwargs)
        output_path: Path = kwargs["output_path"]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"png")
        return output_path


class FakeUpscaleEngine:
    def __init__(self, tmp_path: Path) -> None:
        self.calls: list[Any] = []
        self.tmp_path = tmp_path

    async def run(self, job: Any) -> Path:
        self.calls.append(job)
        out = self.tmp_path / f"upscaled-{job.id}.png"
        out.write_bytes(b"bigpng")
        return out


def register_generation_model(registry: ModelRegistry, settings: Settings, model_id: str = "gen--amd--sd15") -> None:
    model_dir = settings.models_path / "generation" / model_id
    model_dir.mkdir(parents=True, exist_ok=True)
    registry.register(
        ModelEntry(
            id=model_id, name="amd/sd15", kind=ModelKind.diffusion_onnx,
            source="hf:amd/sd15", size_bytes=1, scale=None,
            file_path=f"generation/{model_id}",
        )
    )


def make_manager(tmp_path: Path) -> tuple[GenerationJobManager, FakeGenerationEngine, FakeUpscaleEngine, ModelRegistry, Settings]:
    settings = make_settings(tmp_path)
    registry = ModelRegistry(settings)
    engine = FakeGenerationEngine()
    upscaler = FakeUpscaleEngine(tmp_path)
    # DeviceSemaphores: importar y construir igual que en el arnés de tests de audio
    # (copiar la línea exacta de tests/test_audio_pipeline.py::make_manager).
    manager = GenerationJobManager(
        settings, engine, make_device_semaphores(settings),
        registry=registry, upscale_engine=upscaler, onnx_upscale_engine=None,
    )
    register_generation_model(registry, settings)
    return manager, engine, upscaler, registry, settings


@pytest.mark.anyio
async def test_generation_job_completes_and_sets_output(tmp_path: Path) -> None:
    manager, engine, _up, _reg, _settings = make_manager(tmp_path)
    job = await manager.create_job(prompt="a red apple", model_id="gen--amd--sd15", device="cpu")

    await manager._process_next()  # mismo seam de test que los otros managers; si no existe, drenar con worker start/stop

    final = manager.get_job(job.id)
    assert final is not None
    assert final.status == JobStatus.completed
    assert final.output_path is not None and final.output_path.exists()
    assert final.finished_at is not None
    assert engine.calls[0]["device"] == "cpu"


@pytest.mark.anyio
async def test_auto_upscale_runs_two_stages_in_one_job(tmp_path: Path) -> None:
    manager, engine, upscaler, _reg, _settings = make_manager(tmp_path)
    job = await manager.create_job(
        prompt="a red apple", model_id="gen--amd--sd15", device="cpu",
        auto_upscale=True, upscale_model_name="realesrgan-x4plus", upscale_scale=4,
    )

    await manager._process_next()

    final = manager.get_job(job.id)
    assert final.status == JobStatus.completed
    assert len(upscaler.calls) == 1
    assert upscaler.calls[0].scale == 4
    assert final.output_path.name.startswith("upscaled-")
    stage_keys = [s["key"] for s in final.metadata["stages"]]
    assert stage_keys == ["generating", "upscaling"]
    generated_intermediate = engine.calls[0]["output_path"]
    assert not generated_intermediate.exists()  # intermedio borrado tras upscale OK


@pytest.mark.anyio
async def test_create_job_rejects_unknown_model(tmp_path: Path) -> None:
    manager, *_ = make_manager(tmp_path)
    with pytest.raises(ValueError, match="model"):
        await manager.create_job(prompt="x", model_id="nope", device="cpu")


@pytest.mark.anyio
async def test_create_job_rejects_upscaler_model_id_as_generation_model(tmp_path: Path) -> None:
    manager, _e, _u, registry, _s = make_manager(tmp_path)
    registry.register(
        ModelEntry(id="up1", name="up", kind=ModelKind.onnx, source="hf:x/y", size_bytes=1, scale=2, file_path="up1.onnx")
    )
    with pytest.raises(ValueError):
        await manager.create_job(prompt="x", model_id="up1", device="cpu")


@pytest.mark.anyio
async def test_create_job_rejects_auto_device(tmp_path: Path) -> None:
    manager, *_ = make_manager(tmp_path)
    with pytest.raises(ValueError, match="auto"):
        await manager.create_job(prompt="x", model_id="gen--amd--sd15", device="auto")


@pytest.mark.anyio
async def test_create_job_requires_upscale_params_when_auto_upscale(tmp_path: Path) -> None:
    manager, *_ = make_manager(tmp_path)
    with pytest.raises(ValueError, match="upscale"):
        await manager.create_job(prompt="x", model_id="gen--amd--sd15", device="cpu", auto_upscale=True)


@pytest.mark.anyio
async def test_cancel_queued_job_skips_processing(tmp_path: Path) -> None:
    manager, engine, *_ = make_manager(tmp_path)
    job = await manager.create_job(prompt="x", model_id="gen--amd--sd15", device="cpu")

    assert manager.cancel_job(job.id) is True
    await manager._process_next()

    assert manager.get_job(job.id).status == JobStatus.cancelled
    assert engine.calls == []
```

Detalles a resolver copiando de `tests/test_audio_pipeline.py`: `make_device_semaphores` (la construcción real de `DeviceSemaphores` que use su `make_manager`), el marker async del archivo, y si los managers exponen `_process_next` o hay que drenar con `await manager.start()` + `await manager.queue.join()` + `await manager.stop()` (usar el mismo mecanismo que usen los tests de audio).

- [ ] **Step 3: correr — fallan** (`ModuleNotFoundError`).

- [ ] **Step 4: implementar `app/services/generation_job_manager.py`**

Esqueleto calcado de `app/services/audio_job_manager.py` (leerlo entero antes). Diferencias vs audio — todo lo demás (start/stop/_worker/_run_job con semáforo/_execute_job con la MISMA semántica de cancelación de `audio_job_manager.py:167-194`/cancel_job/_enqueue con `QueueFullError`/queue_depth/get_job) se copia adaptando `AudioJob → GenerationJob`:

```python
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from app.config import Settings
from app.models import GenerationJob, JobStatus, TERMINAL_JOB_STATUSES, UpscaleJob, utc_now
from app.services.devices_service import AUTO_DEVICE_ID, DevicesService
from app.services.engines.generation_onnx import GenerationEngine, GenerationRequest
from app.services.job_manager import QueueFullError, select_upscale_engine  # ajustar import de QueueFullError al módulo real
from app.services.model_registry import ModelKind, ModelRegistry
from app.services.progress import (
    advance_generation_stage,
    apply_generation_step_progress,
    complete_generation_stages,
)

MAX_STEPS = 100
MAX_DIMENSION = 1024
MIN_DIMENSION = 64
DIMENSION_MULTIPLE = 64
UPSCALE_SCALE_RANGE = (2, 4)


class GenerationJobManager:
    def __init__(
        self,
        settings: Settings,
        engine: GenerationEngine,
        device_semaphores: Any,
        *,
        registry: ModelRegistry,
        upscale_engine: Any,
        onnx_upscale_engine: Any | None = None,
        devices: DevicesService | None = None,
    ) -> None:
        self.settings = settings
        self.engine = engine
        self.device_semaphores = device_semaphores
        self.registry = registry
        self.upscale_engine = upscale_engine
        self.onnx_upscale_engine = onnx_upscale_engine
        self.devices = devices
        self.jobs: dict[str, GenerationJob] = {}
        self.queue: asyncio.Queue[GenerationJob] = asyncio.Queue(maxsize=settings.max_queue_size)
        self.worker_tasks: list[asyncio.Task] = []
        self._active: dict[str, asyncio.Task] = {}
```

`create_job` — validaciones antes de encolar:

```python
    async def create_job(self, *, prompt: str, model_id: str, negative_prompt: str | None = None,
                         steps: int = 25, guidance: float = 7.5, width: int = 512, height: int = 512,
                         seed: int | None = None, device: str | None = None, auto_upscale: bool = False,
                         upscale_model_name: str | None = None, upscale_scale: int | None = None,
                         upscale_model_id: str | None = None, job_id: str | None = None) -> GenerationJob:
        self._validate_generation_model(model_id)
        self._validate_params(prompt, steps, width, height)
        await self._validate_device(device)
        if auto_upscale:
            self._validate_upscale_params(upscale_model_name, upscale_scale, upscale_model_id)
        job = GenerationJob(
            prompt=prompt, model_id=model_id, negative_prompt=negative_prompt, steps=steps,
            guidance=guidance, width=width, height=height, seed=seed, device=device,
            auto_upscale=auto_upscale, upscale_model_name=upscale_model_name,
            upscale_scale=upscale_scale, upscale_model_id=upscale_model_id,
        )
        if job_id is not None:
            job.id = job_id
        self.jobs[job.id] = job
        self._enqueue(job)
        return job

    def _validate_generation_model(self, model_id: str) -> None:
        entry = self.registry.get(model_id)
        if entry is None or entry.kind != ModelKind.diffusion_onnx:
            raise ValueError(f"Unknown generation model: {model_id!r}")

    def _validate_params(self, prompt: str, steps: int, width: int, height: int) -> None:
        if not prompt.strip():
            raise ValueError("prompt must not be empty")
        if not 1 <= steps <= MAX_STEPS:
            raise ValueError(f"steps must be between 1 and {MAX_STEPS}")
        for label, value in (("width", width), ("height", height)):
            if not MIN_DIMENSION <= value <= MAX_DIMENSION or value % DIMENSION_MULTIPLE:
                raise ValueError(
                    f"{label} must be a multiple of {DIMENSION_MULTIPLE} between {MIN_DIMENSION} and {MAX_DIMENSION}"
                )

    async def _validate_device(self, device: str | None) -> None:
        if device is None:
            return
        if device == AUTO_DEVICE_ID:
            raise ValueError("device 'auto' is not supported for generation jobs; pin a concrete device (cpu|dml:N)")
        if self.devices is not None:
            await asyncio.to_thread(self.devices.validate, device)

    def _validate_upscale_params(self, model_name: str | None, scale: int | None, model_id: str | None) -> None:
        if scale is None or not UPSCALE_SCALE_RANGE[0] <= scale <= UPSCALE_SCALE_RANGE[1]:
            raise ValueError("auto_upscale requires upscale_scale between 2 and 4")
        if not model_name and not model_id:
            raise ValueError("auto_upscale requires an upscale model (name or id)")
        if model_id is not None:
            entry = self.registry.get(model_id)
            if entry is None or entry.kind != ModelKind.onnx:
                raise ValueError(f"Unknown upscale model: {model_id!r}")
```

`_run_engine` (llamado desde el `_execute_job` copiado de audio):

```python
    async def _run_engine(self, job: GenerationJob) -> None:
        entry = self.registry.get(job.model_id)
        if entry is None or entry.kind != ModelKind.diffusion_onnx:
            raise RuntimeError(f"Generation model not found: {job.model_id!r}")
        pipeline_dir = self._resolve_pipeline_dir(entry)
        device = job.device or self.settings.default_device
        include_upscale = job.auto_upscale
        advance_generation_stage(job, "generating", include_upscale)
        request = GenerationRequest(
            prompt=job.prompt, negative_prompt=job.negative_prompt, steps=job.steps,
            guidance=job.guidance, width=job.width, height=job.height, seed=job.seed,
        )

        def on_progress(done: int, total: int) -> None:
            apply_generation_step_progress(job, done, total, include_upscale)

        generated = await self.engine.run(
            model_id=job.model_id, pipeline_dir=pipeline_dir, request=request,
            device=device, output_path=self.settings.outputs_path / f"{job.id}.png",
            progress_cb=on_progress,
        )
        if not job.auto_upscale:
            job.output_path = generated
            complete_generation_stages(job, include_upscale)
            return
        advance_generation_stage(job, "upscaling", include_upscale)
        job.output_path = await self._run_auto_upscale(job, generated, device)
        complete_generation_stages(job, include_upscale)

    def _resolve_pipeline_dir(self, entry: Any) -> Path:
        models_root = self.settings.models_path.resolve()
        target = (self.settings.models_path / (entry.file_path or "")).resolve()
        if not target.is_relative_to(models_root):
            raise RuntimeError(f"Model path escapes models directory: {entry.file_path!r}")
        if not target.is_dir():
            raise RuntimeError(f"Model folder missing on disk: {entry.file_path!r}")
        return target

    async def _run_auto_upscale(self, job: GenerationJob, generated: Path, device: str) -> Path:
        upscale_job = UpscaleJob(
            source_path=generated,
            original_filename=generated.name,
            model_name=job.upscale_model_name or "",
            scale=job.upscale_scale or UPSCALE_SCALE_RANGE[0],
            output_format="png",
            model_id=job.upscale_model_id,
            device=device,
        )
        upscale = select_upscale_engine(
            upscale_job, self.registry, self.upscale_engine, self.onnx_upscale_engine
        )
        output = await upscale.run(upscale_job)
        generated.unlink(missing_ok=True)
        return output
```

Sin `_unlink_source_safely` (no hay upload). Si `QueueFullError` vive en otro módulo (mirar el import de `audio_job_manager.py`), corregir el import.

- [ ] **Step 5: retention sweeper**

`app/services/retention_sweeper.py`:
- `__init__`: agregar param `generation_job_manager: GenerationJobManager | None = None` (después de `audio_job_manager`) y guardarlo.
- `sweep_once`: agregar al final:

```python
        if self.generation_job_manager is not None:
            self._prune_finished_jobs(self.generation_job_manager.jobs)
```

- Ampliar los type hints de `_prune_finished_jobs`/`_is_finished`/`_is_expired` con `GenerationJob` (import desde `app.models`).
- `GenerationJob` no tiene `source_path` → `_active_source_paths` NO se toca.

Test en el archivo del sweeper (copiar el arnés del caso de audio): job de generación `completed` con `finished_at` viejo se poda; job `running` no.

- [ ] **Step 6: correr todo — pasa**

```powershell
python -m pytest tests/test_generation_job_manager.py tests/ -k "generation or sweeper or retention" -v
```

- [ ] **Step 7: commit**

```powershell
git add app/models.py app/services/generation_job_manager.py app/services/retention_sweeper.py tests/
git commit -m "feat: GenerationJobManager con auto-upscale en dos stages y retencion TTL"
```

---

### Task 9: API + schemas + wiring en `main.py`

**Files:**
- Modify: `app/schemas.py`, `app/api/routes.py`, `app/main.py`
- Test: `tests/test_generation_api.py`

**Interfaces:**
- Consumes: todo lo anterior. Reusa `CreateJobResponse`, `InstallModelRequest`, `CreateInstallResponse`, `InstallStatusResponse` existentes en `schemas.py` sin tocarlos.
- Produces (frontend Tasks 10-13 dependen): endpoints `POST /api/v1/generation/jobs`, `GET /api/v1/generation/jobs/{id}`, `POST /api/v1/generation/jobs/{id}/cancel`, `GET /api/v1/generation/jobs/{id}/download`, `GET /api/v1/generation/capabilities`, `POST /api/v1/generation/models`, `GET /api/v1/generation/models/install/{id}`. Payloads camelCase (aliases pydantic).

- [ ] **Step 1: tests de API (fallan)**

`tests/test_generation_api.py` — copiar el arnés app/TestClient del test de rutas de audio (`tests/test_audio_pipeline.py::test_create_audio_job_route_forwards_output_format`, línea ~227: cómo construye la app y sobreescribe dependencias):

```python
def test_create_generation_job_validates_steps_cap(client) -> None:
    response = client.post("/api/v1/generation/jobs", json={
        "prompt": "x", "modelId": "gen--amd--sd15", "steps": 101,
    })
    assert response.status_code == 422


def test_create_generation_job_validates_dimension_multiple(client) -> None:
    response = client.post("/api/v1/generation/jobs", json={
        "prompt": "x", "modelId": "gen--amd--sd15", "width": 500,
    })
    assert response.status_code == 422


def test_get_generation_job_unknown_returns_404(client) -> None:
    assert client.get("/api/v1/generation/jobs/nope").status_code == 404


def test_capabilities_reports_unavailable_without_deps(client, monkeypatch) -> None:
    monkeypatch.setattr(
        "app.api.routes.generation_dependencies_available", lambda: (False, "optimum missing")
    )
    payload = client.get("/api/v1/generation/capabilities").json()
    assert payload["available"] is False
    assert "optimum" in payload["reason"]


def test_capabilities_lists_installed_models_and_cpu_only_flag(client_with_model) -> None:
    payload = client_with_model.get("/api/v1/generation/capabilities").json()
    assert payload["available"] is True
    assert payload["models"] == [{"id": "gen--amd--sd15", "name": "amd/sd15"}]
    assert isinstance(payload["cpuOnly"], bool)


def test_create_and_poll_generation_job_roundtrip(client_with_model) -> None:
    created = client_with_model.post("/api/v1/generation/jobs", json={
        "prompt": "a red apple", "modelId": "gen--amd--sd15", "device": "cpu",
    })
    assert created.status_code in (200, 201)
    job_id = created.json()["id"]
    polled = client_with_model.get(f"/api/v1/generation/jobs/{job_id}").json()
    assert polled["prompt"] == "a red apple"
    assert polled["status"] in ("queued", "running", "completed")
```

Fixtures `client`/`client_with_model`: app real con manager cableado a `FakeGenerationEngine` (reusar el de Task 8) y registry con el modelo fake registrado — mismo mecanismo de override que use el arnés de audio.

- [ ] **Step 2: correr — fallan** (404 en todas las rutas).

- [ ] **Step 3: schemas**

En `app/schemas.py` (estilo idéntico a `AudioJobResponse`/`InstallModelRequest` — atributos con alias camelCase):

```python
class CreateGenerationJobRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    prompt: str = Field(min_length=1, max_length=2000)
    negative_prompt: str | None = Field(default=None, alias="negativePrompt", max_length=2000)
    model_id: str = Field(alias="modelId")
    steps: int = Field(default=25, ge=1, le=100)
    guidance: float = Field(default=7.5, ge=0, le=30)
    width: int = Field(default=512, ge=64, le=1024, multiple_of=64)
    height: int = Field(default=512, ge=64, le=1024, multiple_of=64)
    seed: int | None = Field(default=None, ge=0)
    device: str | None = None
    auto_upscale: bool = Field(default=False, alias="autoUpscale")
    upscale_model_name: str | None = Field(default=None, alias="upscaleModelName")
    upscale_scale: int | None = Field(default=None, alias="upscaleScale", ge=2, le=4)
    upscale_model_id: str | None = Field(default=None, alias="upscaleModelId")


class GenerationJobResponse(BaseModel):
    id: str
    status: JobStatus
    prompt: str
    negative_prompt: str | None = Field(default=None, serialization_alias="negativePrompt")
    model_id: str = Field(serialization_alias="modelId")
    steps: int
    guidance: float
    width: int
    height: int
    seed: int | None = None
    device: str | None = None
    auto_upscale: bool = Field(default=False, serialization_alias="autoUpscale")
    created_at: datetime = Field(serialization_alias="createdAt")
    started_at: datetime | None = Field(default=None, serialization_alias="startedAt")
    finished_at: datetime | None = Field(default=None, serialization_alias="finishedAt")
    progress_pct: float | None = Field(default=None, serialization_alias="progressPct")
    stages: list[dict[str, Any]] | None = None
    error: str | None = None
    download_url: str | None = Field(default=None, serialization_alias="downloadUrl")


class GenerationModelSummary(BaseModel):
    id: str
    name: str


class GenerationCapabilitiesResponse(BaseModel):
    available: bool
    reason: str | None = None
    models: list[GenerationModelSummary] = Field(default_factory=list)
    devices: list[str] = Field(default_factory=list)
    cpu_only: bool = Field(default=False, serialization_alias="cpuOnly")
```

- [ ] **Step 4: rutas**

En `app/api/routes.py`, junto a las de audio. Dependency getters espejo de `get_audio_job_manager` (leen `app.state`):

```python
def get_generation_job_manager(request: Request) -> GenerationJobManager:
    return request.app.state.generation_job_manager


def get_generation_installer(request: Request) -> GenerationModelInstaller:
    return request.app.state.generation_installer
```

Serializador (espejo de `audio_job_to_response`, routes.py:274-291):

```python
def generation_job_to_response(job: GenerationJob) -> GenerationJobResponse:
    download_url = (
        f"/api/v1/generation/jobs/{job.id}/download" if job.status == JobStatus.completed else None
    )
    return GenerationJobResponse(
        id=job.id, status=job.status, prompt=job.prompt, negative_prompt=job.negative_prompt,
        model_id=job.model_id, steps=job.steps, guidance=job.guidance, width=job.width,
        height=job.height, seed=job.seed, device=job.device, auto_upscale=job.auto_upscale,
        created_at=job.created_at, started_at=job.started_at, finished_at=job.finished_at,
        progress_pct=_progress_pct_from_metadata(job.metadata), stages=job.metadata.get("stages"),
        error=job.error, download_url=download_url,
    )
```

Endpoints — create/get/cancel/download calcados de los de audio (mismos códigos de error y decoradores; el download con `media_type="image/png"`); create con:

```python
    try:
        job = await generation_jobs.create_job(
            prompt=payload.prompt, negative_prompt=payload.negative_prompt, model_id=payload.model_id,
            steps=payload.steps, guidance=payload.guidance, width=payload.width, height=payload.height,
            seed=payload.seed, device=payload.device, auto_upscale=payload.auto_upscale,
            upscale_model_name=payload.upscale_model_name, upscale_scale=payload.upscale_scale,
            upscale_model_id=payload.upscale_model_id,
        )
    except QueueFullError as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
```

(los códigos 429/400: verificar contra el create de audio y copiar los reales). Capabilities:

```python
@router.get("/generation/capabilities", response_model=GenerationCapabilitiesResponse)
async def generation_capabilities(
    registry: ModelRegistry = Depends(get_model_registry),
    devices_service: DevicesService = Depends(get_devices_service),
) -> GenerationCapabilitiesResponse:
    available, reason = generation_dependencies_available()
    if not available:
        return GenerationCapabilitiesResponse(available=False, reason=reason, cpu_only=True)
    models = [
        GenerationModelSummary(id=entry.id, name=entry.name)
        for entry in registry.list()
        if entry.kind == ModelKind.diffusion_onnx
    ]
    device_infos = devices_service.list_devices()
    return GenerationCapabilitiesResponse(
        available=True,
        models=models,
        devices=[info["id"] for info in device_infos],
        cpu_only=all(info["kind"] != "gpu" for info in device_infos),
    )
```

Install endpoints — espejo de `install_model`/`get_install_status` (routes.py:818-846) apuntando al `GenerationModelInstaller`, `status_url=f"/api/v1/generation/models/install/{install_id}"`, y `except ValueError → 400` en el POST.

- [ ] **Step 5: wiring `app/main.py`**

En el lifespan, después de la construcción de engines (línea ~63):

```python
    generation_engine = GenerationEngine(settings, gpu_coordinator)
    generation_installer = GenerationModelInstaller(settings, model_registry, hf_client)
    generation_job_manager = GenerationJobManager(
        settings,
        generation_engine,
        device_semaphores,
        registry=model_registry,
        upscale_engine=engine,
        onnx_upscale_engine=onnx_engine,
        devices=devices_service,
    )
```

(usar los MISMOS nombres de variable que recibe `JobManager(...)` en ese archivo para builtin engine/semaphores/hf_client/registry/devices — leerlos ahí). Agregar `await generation_job_manager.start()` / `await generation_installer.start()` junto a los starts de los demás, los `stop()` en el shutdown en orden inverso, `app.state.generation_job_manager = ...` / `app.state.generation_installer = ...` junto a los otros `app.state`, y pasar `generation_job_manager=generation_job_manager` al constructor de `RetentionSweeper`.

- [ ] **Step 6: correr — API tests pasan + suite backend completa verde**

```powershell
python -m pytest tests/ -x -q
```

Expected: PASS completo. Regresiones acá = bug de wiring.

- [ ] **Step 7: commit**

```powershell
git add app/schemas.py app/api/routes.py app/main.py tests/test_generation_api.py
git commit -m "feat: endpoints de generacion (jobs, capabilities, install) y wiring en lifespan"
```

---

### Task 10: Frontend — tipos + servicios

**Files:**
- Modify: `frontend/src/lib/api.ts`, `frontend/src/lib/apiTypes.ts`
- Create: `frontend/src/services/generation.ts`
- Test: `frontend/src/services/generation.test.ts`

**Interfaces:**
- Produces (Tasks 11-13 dependen): `apiPostJson<T>(path, body)`; tipos `GenerationJob`, `GenerationCapabilities`, `GenerationModelSummary`; `GenerationJob` sumado a la union `AnyJobResponse` (y a `TrackedJobResponse` en Task 11); servicios `createGenerationJob`, `getGenerationJob`, `cancelGenerationJob`, `fetchGenerationCapabilities`, `installGenerationModel`, `getGenerationInstallStatus`.

- [ ] **Step 1: test del servicio (falla)**

`frontend/src/services/generation.test.ts` (mock de `../lib/api` con `vi.mock`, estilo de los tests de servicios existentes):

```typescript
import { beforeEach, describe, expect, it, vi } from "vitest";

import { apiGet, apiPost, apiPostJson } from "../lib/api";
import { createGenerationJob, getGenerationJob } from "./generation";

vi.mock("../lib/api", () => ({
  apiGet: vi.fn(),
  apiPost: vi.fn(),
  apiPostJson: vi.fn(),
}));

describe("generation service", () => {
  beforeEach(() => vi.clearAllMocks());

  it("posts camelCase body omitting empty optionals", async () => {
    vi.mocked(apiPostJson).mockResolvedValue({ id: "j1" });

    await createGenerationJob({
      prompt: "a red apple", negativePrompt: null, modelId: "gen--amd--sd15",
      steps: 25, guidance: 7.5, width: 512, height: 512, seed: null,
      device: null, autoUpscale: false, upscaleModelName: null,
      upscaleScale: null, upscaleModelId: null,
    });

    expect(apiPostJson).toHaveBeenCalledWith("/generation/jobs", {
      prompt: "a red apple", modelId: "gen--amd--sd15", steps: 25,
      guidance: 7.5, width: 512, height: 512, autoUpscale: false,
    });
  });

  it("includes upscale params only when autoUpscale", async () => {
    vi.mocked(apiPostJson).mockResolvedValue({ id: "j1" });

    await createGenerationJob({
      prompt: "x", negativePrompt: "blurry", modelId: "m", steps: 25, guidance: 7.5,
      width: 512, height: 512, seed: 42, device: "dml:0", autoUpscale: true,
      upscaleModelName: "realesrgan-x4plus", upscaleScale: 4, upscaleModelId: null,
    });

    const body = vi.mocked(apiPostJson).mock.calls[0][1] as Record<string, unknown>;
    expect(body.upscaleModelName).toBe("realesrgan-x4plus");
    expect(body.upscaleScale).toBe(4);
    expect(body.seed).toBe(42);
    expect(body.negativePrompt).toBe("blurry");
  });

  it("gets a job by id", async () => {
    vi.mocked(apiGet).mockResolvedValue({ id: "j1" });
    await getGenerationJob("j1");
    expect(apiGet).toHaveBeenCalledWith("/generation/jobs/j1");
  });
});
```

- [ ] **Step 2: correr — falla** (`cd frontend; npm test -- generation.test`): módulo inexistente.

- [ ] **Step 3: implementar**

`frontend/src/lib/api.ts` — `apiPostJson` calcado de `apiPost` (líneas 74-88: mismo base URL y manejo de respuesta/errores), agregando headers y body:

```typescript
export async function apiPostJson<T>(path: string, body: unknown): Promise<T> {
  // copiar el cuerpo de apiPost y agregar:
  //   method: "POST",
  //   headers: { "Content-Type": "application/json" },
  //   body: JSON.stringify(body),
}
```

`frontend/src/lib/apiTypes.ts`:

```typescript
export interface GenerationJob {
  id: string;
  status: JobStatus;
  prompt: string;
  negativePrompt: string | null;
  modelId: string;
  steps: number;
  guidance: number;
  width: number;
  height: number;
  seed: number | null;
  device: string | null;
  autoUpscale: boolean;
  createdAt: string;
  startedAt: string | null;
  finishedAt: string | null;
  progressPct: number | null;
  stages: JobStage[] | null;
  error: string | null;
  downloadUrl: string | null;
}

export interface GenerationModelSummary {
  id: string;
  name: string;
}

export interface GenerationCapabilities {
  available: boolean;
  reason: string | null;
  models: GenerationModelSummary[];
  devices: string[];
  cpuOnly: boolean;
}
```

y sumar `GenerationJob` a la union `AnyJobResponse` (localizarla en este archivo o donde la importe `JobCard.tsx`).

`frontend/src/services/generation.ts`:

```typescript
import { apiGet, apiPost, apiPostJson } from "../lib/api";
import type {
  CreateInstallResponse,
  CreateJobResponse,
  GenerationCapabilities,
  GenerationJob,
  InstallStatusResponse,
} from "../lib/apiTypes";

export interface CreateGenerationJobParams {
  prompt: string;
  negativePrompt: string | null;
  modelId: string;
  steps: number;
  guidance: number;
  width: number;
  height: number;
  seed: number | null;
  device: string | null;
  autoUpscale: boolean;
  upscaleModelName: string | null;
  upscaleScale: number | null;
  upscaleModelId: string | null;
}

function buildRequestBody(params: CreateGenerationJobParams): Record<string, unknown> {
  const body: Record<string, unknown> = {
    prompt: params.prompt,
    modelId: params.modelId,
    steps: params.steps,
    guidance: params.guidance,
    width: params.width,
    height: params.height,
    autoUpscale: params.autoUpscale,
  };
  if (params.negativePrompt) body.negativePrompt = params.negativePrompt;
  if (params.seed !== null) body.seed = params.seed;
  if (params.device) body.device = params.device;
  if (params.autoUpscale) {
    if (params.upscaleModelName) body.upscaleModelName = params.upscaleModelName;
    if (params.upscaleScale !== null) body.upscaleScale = params.upscaleScale;
    if (params.upscaleModelId) body.upscaleModelId = params.upscaleModelId;
  }
  return body;
}

// NOTA (contrato real, Task 9): POST /generation/jobs devuelve el GenerationJob
// completo (id, status, downloadUrl...) con 201 — NO el CreateJobResponse/202
// (jobId/statusUrl) de los otros kinds. Deviación intencional documentada; no
// "arreglarla" para parecerse a audio.
export function createGenerationJob(params: CreateGenerationJobParams): Promise<GenerationJob> {
  return apiPostJson<GenerationJob>("/generation/jobs", buildRequestBody(params));
}

export function getGenerationJob(jobId: string): Promise<GenerationJob> {
  return apiGet<GenerationJob>(`/generation/jobs/${jobId}`);
}

export function cancelGenerationJob(jobId: string): Promise<GenerationJob> {
  return apiPost<GenerationJob>(`/generation/jobs/${jobId}/cancel`);
}

export function fetchGenerationCapabilities(): Promise<GenerationCapabilities> {
  return apiGet<GenerationCapabilities>("/generation/capabilities");
}

export function installGenerationModel(repoId: string): Promise<CreateInstallResponse> {
  return apiPostJson<CreateInstallResponse>("/generation/models", { repoId });
}

export function getGenerationInstallStatus(installId: string): Promise<InstallStatusResponse> {
  return apiGet<InstallStatusResponse>(`/generation/models/install/${installId}`);
}
```

(Si `CreateInstallResponse`/`InstallStatusResponse` viven solo en `api.ts`, importar de ahí o mover el import al lugar real.)

- [ ] **Step 4: correr — pasa** (`npm test -- generation.test`), y `npx tsc --noEmit` limpio.

- [ ] **Step 5: commit**

```powershell
git add frontend/src/lib/api.ts frontend/src/lib/apiTypes.ts frontend/src/services/generation.ts frontend/src/services/generation.test.ts
git commit -m "feat: tipos y servicios frontend de generacion"
```

---

### Task 11: Frontend — cola visual + cards

**Files:**
- Modify: `frontend/src/lib/jobQueueStore.ts:1`, `frontend/src/hooks/useJobQueue.ts:11,31-51`, `frontend/src/components/JobCard.tsx`, `frontend/src/components/JobDetailModal.tsx`, `frontend/src/components/ModelPicker.tsx:15-25`
- Test: `frontend/src/hooks/useJobQueue.test.tsx`

**Interfaces:**
- Consumes: `getGenerationJob`/`cancelGenerationJob` (Task 10), `GenerationJob` en `AnyJobResponse`.
- Produces: kind `"generation"` trackeable en la cola (Task 12 lo usa vía `useGenerationJob`).

- [ ] **Step 1: test de cola (falla)**

En `useJobQueue.test.tsx`, calcar el caso de agregación de audio: mock de `../services/generation`, trackear `{ id: "g1", kind: "generation", fileName: "a red apple", createdAt: ... }`, mock de `getGenerationJob` devolviendo un `GenerationJob` completed con `downloadUrl`, assert de que la entry aparece con status/downloadUrl.

- [ ] **Step 2: correr — falla** (TS: `"generation"` no asignable a `TrackedJobKind`).

- [ ] **Step 3: implementar**

`jobQueueStore.ts:1`:

```typescript
export type TrackedJobKind = "image" | "video" | "audio" | "generation";
```

`useJobQueue.ts`:

```typescript
import { cancelGenerationJob, getGenerationJob } from "../services/generation";
import type { GenerationJob } from "../lib/apiTypes";

export type TrackedJobResponse = JobResponse | VideoJobResponse | AudioJob | GenerationJob;

const QUERY_KEY_BY_KIND: Record<TrackedJob["kind"], string> = {
  image: "job",
  video: "videoJob",
  audio: "audioJob",
  generation: "generationJob",
};

const CANCEL_BY_KIND: Record<TrackedJob["kind"], (id: string) => Promise<TrackedJobResponse>> = {
  image: cancelJob,
  video: cancelVideoJob,
  audio: cancelAudioJob,
  generation: cancelGenerationJob,
};

function fetchTrackedJob(tracked: TrackedJob): Promise<TrackedJobResponse> {
  if (tracked.kind === "image") {
    return getJob(tracked.id);
  }
  if (tracked.kind === "audio") {
    return getAudioJob(tracked.id);
  }
  if (tracked.kind === "generation") {
    return getGenerationJob(tracked.id);
  }
  return getVideoJob(tracked.id);
}
```

`JobCard.tsx` — guard ANTES de los existentes (líneas 22-28) + rama de details calcada de `ImageCompletedDetails` (misma `<img>` de preview con `downloadUrl`, tamaño `width×height`, `steps`, y `DurationDetailItem` como ya lo usan las otras ramas):

```typescript
function isGenerationJob(job: AnyJobResponse): job is GenerationJob {
  return "prompt" in job;
}
```

`JobDetailModal.tsx` — mismo guard + sección de detalles (prompt completo, negativePrompt, modelId, steps/guidance/size/seed, device, duración).

`ModelPicker.tsx` — en `groupModels` (líneas 15-25), asegurar que el grupo de instalados filtre `kind === "onnx"` explícitamente (no "todo lo no-builtin"), para que los `diffusion-onnx` NO aparezcan en el picker de upscale:

```typescript
    { label: "Installed", models: models.filter((model) => model.kind === "onnx") },
```

(usar el label real que tenga el grupo hoy).

- [ ] **Step 4: correr — pasan**

```powershell
cd frontend
npm test -- useJobQueue
npx tsc --noEmit
```

- [ ] **Step 5: commit**

```powershell
git add frontend/src/lib/jobQueueStore.ts frontend/src/hooks/useJobQueue.ts frontend/src/hooks/useJobQueue.test.tsx frontend/src/components/JobCard.tsx frontend/src/components/JobDetailModal.tsx frontend/src/components/ModelPicker.tsx
git commit -m "feat: kind generation en cola visual, cards y detalle"
```

---

### Task 12: Frontend — `useGenerationJob` + página Generate

**Files:**
- Create: `frontend/src/hooks/useGenerationJob.ts`, `frontend/src/hooks/useGenerationJob.test.tsx`, `frontend/src/modules/generate/GeneratePanel.tsx`, `frontend/src/modules/generate/GeneratePanel.test.tsx`, `frontend/src/modules/generate/GeneratePage.tsx`
- Modify: `frontend/src/App.tsx`, componente de navegación (localizar con `grep -r "to=\"/audio\"" frontend/src` — agregar el link ahí al lado)

**Interfaces:**
- Consumes: servicios Task 10, cola Task 11, `ModelPicker` (props: `value: string | null`, `onChange: (model: ModelResponse) => void`), `DevicePicker` (copiar props de su uso en `AudioPanel.tsx:127-214`), `JobCard`, `CPU_ONLY_WARNING` textual (el texto viene del backend vía spec — en FE va hardcodeado igual al spec).
- Produces: ruta `/generate` funcional.

- [ ] **Step 1: test del hook (falla)**

`useGenerationJob.test.tsx` — CALCAR `useAudioJob.test.tsx` (líneas 1-130: mismo wrapper de QueryClient, mismos casos) con `vi.mock("../services/generation")`: empieza idle; `submit(params)` crea y pollea hasta completed; error del job se expone; el job queda trackeado en la cola con `kind: "generation"` y `fileName` = prompt.

- [ ] **Step 2: correr — falla.**

- [ ] **Step 3: implementar `useGenerationJob.ts`**

Copiar la estructura completa de `useAudioJob.ts` (líneas 61-119) cambiando: servicios por los de generación, tipo `AudioJob → GenerationJob`, y el track en cola:

```typescript
    queue.track({
      id: response.id,
      kind: "generation",
      fileName: params.prompt.slice(0, 60),
      createdAt: Date.now(),
    });
```

(la firma real de `queue.track`/`jobQueueStore` se copia de `useAudioJob.ts`). Exportar el mismo shape: `{ phase, job, errorMessage, submit, cancel, reset }`.

- [ ] **Step 4: test del panel (falla)**

`GeneratePanel.test.tsx` — casos (estilo de `AudioPanel` tests / testing-library del repo):

1. capabilities `available: false` → banner con `reason`, form deshabilitado.
2. capabilities con modelos → select poblado; submit llama `createGenerationJob` con los params elegidos.
3. `cpuOnly: true` y device cpu → primer click en Generate NO llama al servicio, muestra la advertencia del spec ("No se detectó GPU compatible…"); click en "Continuar igual" → llama.
4. toggle auto-upscale off → body sin `upscaleScale`; on → aparece el picker de modelo de upscale + select de escala y el submit incluye ambos.

- [ ] **Step 5: implementar `GeneratePanel.tsx`**

Estructura espejo de `AudioPanel.tsx` (mismo layout de Card/controles/JobCard al pie). Estado local:

```typescript
const SIZE_OPTIONS = [256, 384, 512, 640, 768, 896, 1024];

const [prompt, setPrompt] = useState("");
const [negativePrompt, setNegativePrompt] = useState("");
const [modelId, setModelId] = useState<string | null>(null);
const [steps, setSteps] = useState(25);
const [guidance, setGuidance] = useState(7.5);
const [width, setWidth] = useState(512);
const [height, setHeight] = useState(512);
const [seed, setSeed] = useState<string>("");        // input texto, parse a int o null
const [device, setDevice] = useState<string | null>(null);
const [autoUpscale, setAutoUpscale] = useState(false);
const [upscaleModel, setUpscaleModel] = useState<ModelResponse | null>(null);
const [upscaleScale, setUpscaleScale] = useState(2);
const [cpuConfirmPending, setCpuConfirmPending] = useState(false);
```

- Capabilities: `useQuery({ queryKey: ["generationCapabilities"], queryFn: fetchGenerationCapabilities })`. `available === false` → render del banner con `reason` y return temprano.
- Select de modelo de generación sobre `capabilities.models` (id → label name). Sin modelos instalados → mensaje con link a la página Models.
- `DevicePicker` y textarea/inputs: reusar los componentes de formulario que usa `AudioPanel` (mismos imports).
- Auto-upscale on → `<ModelPicker value={upscaleModel?.id ?? null} onChange={setUpscaleModel} />` + select de escala 2/3/4. Al armar params: `upscaleModelName` = `upscaleModel.kind === "builtin-ncnn" ? upscaleModel.name : null`, `upscaleModelId` = `upscaleModel.kind === "onnx" ? upscaleModel.id : null` (mirar cómo `ImagePanel.tsx` traduce la selección del picker a params de job y copiar esa lógica).
- Submit:

```typescript
const needsCpuConfirm = capabilities?.cpuOnly === true && (device === null || device === "cpu");

function handleGenerate() {
  if (needsCpuConfirm && !cpuConfirmPending) {
    setCpuConfirmPending(true);
    return;
  }
  setCpuConfirmPending(false);
  submit(buildParams());
}
```

con el aviso renderizado cuando `cpuConfirmPending`: texto exacto del spec + botón "Continuar igual" (re-invoca `handleGenerate`) y "Cancelar" (limpia el flag).
- Debajo, `JobCard`/estado del job igual que `AudioPanel` hace con el suyo.

`GeneratePage.tsx`: wrapper de página idéntico en estructura a `AudioPage.tsx` (mismo layout/heading) montando `<GeneratePanel />`.

- [ ] **Step 6: ruta + nav**

`App.tsx`: `<Route path="/generate" element={<GeneratePage />} />`. En el componente de nav (donde está el link a `/audio`), agregar "Generate" apuntando a `/generate`, mismo estilo de item.

- [ ] **Step 7: correr — todo verde**

```powershell
cd frontend
npm test
npx tsc --noEmit
```

- [ ] **Step 8: commit**

```powershell
git add frontend/src
git commit -m "feat: pagina Generate con useGenerationJob, advertencia CPU-only y auto-upscale"
```

---

### Task 13: Frontend — instalación de modelos de generación en Models

**Files:**
- Create: `frontend/src/modules/models/GenerationModelsSection.tsx`, `frontend/src/modules/models/GenerationModelsSection.test.tsx`
- Modify: `frontend/src/pages/ModelsPage.tsx:1-23`

**Interfaces:**
- Consumes: `installGenerationModel`/`getGenerationInstallStatus` (Task 10), `getModels`/`deleteModel` existentes (`lib/api.ts:101,260`), patrón de polling de instalación existente (mirar `HfResultCard.tsx` y su hook `useModelInstall` — reusarlo si es genérico sobre las funciones de install; si está atado a los endpoints de upscalers, duplicar el hook parametrizando las dos funciones).

- [ ] **Step 1: test (falla)**

`GenerationModelsSection.test.tsx`:

1. Lista modelos instalados: mock `getModels` → uno `kind: "diffusion-onnx"` y uno `kind: "onnx"`; solo el primero se muestra.
2. Install: escribir `amd/sd15` en el input, click Install → `installGenerationModel("amd/sd15")`; polling con `getGenerationInstallStatus` mostrando `progressPct`; al llegar `status: "installed"` refresca la lista.
3. Error de install (`status: "error"`, `error` con el mensaje CUDA del backend) → el mensaje se muestra.
4. Delete: click en borrar → `deleteModel(id)`.

- [ ] **Step 2: correr — falla.**

- [ ] **Step 3: implementar**

`GenerationModelsSection.tsx`: card con heading "Generation models (Stable Diffusion)", input de repo HF + botón Install (sugerencia en placeholder: el repo id del findings doc de Task 1), lista de instalados (`useQuery(["models"])` filtrada por `kind === "diffusion-onnx"`, mostrando name + size + botón delete), y estado de instalación en curso (progress % / installing / validating / error). Polling: mismo intervalo y mecánica que el flujo de install existente en `HfResultCard`/`useModelInstall`.

`ModelsPage.tsx`: montar `<GenerationModelsSection />` debajo de las secciones existentes.

- [ ] **Step 4: correr — verde** (`npm test`, `npx tsc --noEmit`).

- [ ] **Step 5: commit**

```powershell
git add frontend/src/modules/models frontend/src/pages/ModelsPage.tsx
git commit -m "feat: seccion de instalacion de modelos de generacion en Models"
```

---

### Task 14: Smoke real (manual, no CI) + cierre

**Files:** ninguno nuevo (checklist manual + fixes que salgan).

- [ ] **Step 1: suite completa verde**

```powershell
python -m pytest tests/ -q
cd frontend && npm test && npm run build
```

- [ ] **Step 2: smoke local end-to-end**

Con el server corriendo (`uvicorn app.main:app --port 8090` o el script de arranque del repo — ANTES de matar un server ya corriendo, chequear subprocesos activos realesrgan/ffmpeg/rife):

1. `GET /api/v1/generation/capabilities` → `available: true`, devices con `dml:0`, `cpuOnly: false`.
2. En Models: instalar el SD1.5 de la colección `amd/` (repo id del findings doc). Ver progreso hasta `installed`.
3. En Generate: prompt "a red apple on a wooden table", 512×512, 25 steps, dml:0 → imagen generada, progreso por step visible, fila Duration poblada.
4. Repetir con seed fijo dos veces → misma imagen (reproducibilidad).
5. Generar con auto-upscale ×2 → un solo job con stages `generating → upscaling`, PNG final escalado.
6. Cancelar un job a mitad de generación → status `cancelled`, sin proceso colgado.
7. (Si hay solo CPU disponible, p.ej. forzando device cpu) → advertencia proactiva antes de encolar.

- [ ] **Step 3: commit de cierre (si hubo fixes del smoke) + actualizar memoria del proyecto**

Registrar en el findings/memoria: pin final de optimum, repo id de SD1.5 usado, tiempos reales de load/step en dml:0.

---

## Riesgos operativos para el ejecutor

- **Task 1 es GATE**: sin spike verde no se avanza. El fallback (ensamblado manual de sesiones) requiere re-plan, no improvisación.
- Los nombres de kwargs de optimum (`callback`, `generator`, `session_options`) pueden diferir por versión — el findings doc de Task 1 es la fuente de verdad; los puntos de ajuste están marcados en Tasks 4 y 5.
- No tocar `ModelInstaller._download_and_register` ni `pick_weight_file` — el installer viejo sigue siendo el de upscalers.
- VRAM: pipeline SD1.5 ~4GB; `GpuSessionCoordinator` desaloja a los otros motores al `acquire` — es el comportamiento esperado, no un bug.
- IOBinding en DirectML multi-GPU está vetado en este repo (crash DXGI_ERROR_NOT_FOUND documentado) — no "optimizar" el engine con IOBinding.
