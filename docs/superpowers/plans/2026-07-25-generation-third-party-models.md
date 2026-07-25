# Modelos de terceros para generación (subproyecto A) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Habilitar instalación de SDXL/SDXL Turbo/SD3.5 de la colección `amd/`, conversión PyTorch→ONNX como job visible, mensajes accionables para repos gated, settings editables genéricas (primer campo: `HF_TOKEN`) y buscador de modelos de generación.

**Architecture:** Extiende el módulo de generación existente (spec `2026-07-22-generation-module-design.md`, mergeado en master) sin caminos paralelos: el mapa de clases reemplaza el hardcode de `ORTStableDiffusionPipeline`, la conversión reusa la MISMA validación funcional y promoción atómica del installer, y el buscador/settings calcan endpoints ya existentes. Spec fuente: `docs/superpowers/specs/2026-07-25-generation-third-party-models-design.md`.

**Tech Stack:** FastAPI + pydantic v2, optimum==2.1.0 / optimum-onnx==0.1.0 / onnxruntime-directml==1.24.4 / diffusers 0.39.0 (pins ya en `pyproject.toml`), React + TanStack Query + Tailwind (frontend), pytest + vitest.

## Global Constraints

- **Pins de dependencias intocables** (`pyproject.toml`, verificados por spike previo): `optimum==2.1.0`, `optimum-onnx==0.1.0`, `transformers>=4.36,<4.58`, `diffusers>=0.30,<1.0`, `torch>=2.2.0,<3.0.0`. NUNCA instalar el extra `optimum[onnxruntime]` (arrastra `onnxruntime` vanilla que pisa `onnxruntime-directml`).
- **`use_io_binding=False` explícito** en todo `from_pretrained` de pipelines ORT (IOBinding+DML vetado en este repo).
- **Imports de optimum/torch siempre perezosos** (dentro de funciones), nunca a nivel de módulo en `app/services/` — patrón existente en `generation_onnx.py`.
- Mensajes de error orientados a usuario en **español**, accionables, nunca stacktrace crudo (patrón `CUDA_ONLY_MESSAGE`/`VRAM_MESSAGE` de `generation_onnx.py`).
- Commits en español, formato convencional (`feat:`/`fix:`/`docs:`/`refactor:`/`test:`), SIN `Co-Authored-By`.
- Subproyecto B (admisión por capacidad) NO está mergeado: no dependas de contabilidad de VRAM. Subproyecto C (multiusuario/auth) SÍ está mergeado (PR #2 en origin/master): USAR `app/services/json_store.py::write_text_atomically`, `app.config.ENV_FILE_PATH`, y gatear los endpoints nuevos con `Depends(require(Permission...))` de `app.api.auth_deps` siguiendo el patrón de los vecinos. Con `AUTH_MODE=off` (default, y el de los tests) los gates pasan solos — los tests del plan no necesitan fixtures de auth.
- Tests backend: `pytest` desde la raíz del repo (`.venv\Scripts\python -m pytest`). Tests frontend: `npm test -- --run <archivo>` desde `frontend/`.
- La suite entera debe quedar verde al final de cada task (`.venv\Scripts\python -m pytest -q` y `cd frontend && npx vitest run`).

## Estructura de archivos (visión global)

| Archivo | Task | Responsabilidad |
|---|---|---|
| `docs/superpowers/specs/2026-07-25-third-party-spike-findings.md` (nuevo) | 1 | Findings del spike: `_class_name` reales, clases optimum disponibles, API de export |
| `app/services/engines/generation_onnx.py` (modif) | 2 | `PIPELINE_CLASS_NAMES`, `_read_declared_class_name`, `_load_pipeline_class(declared)` |
| `app/services/generation_installer.py` (modif) | 2, 9, 12 | Caller del mapa; extracción `validate_and_promote`; auto-ruteo a conversión |
| `app/services/hf_client.py` (modif) | 3, 4 | `_wrap_hf_auth_error`; `search(task_tags=...)` |
| `app/exceptions.py` (modif) | 3 | `HfAuthError` |
| `app/api/routes.py` (modif) | 4, 6, 11 | `GET /generation/models/search`, `GET/PATCH /settings`, endpoints de conversión |
| `app/schemas.py` (modif) | 4, 6, 11, 12 | Schemas nuevos |
| `app/services/settings_service.py` (nuevo) | 6 | Whitelist + validación + escritura atómica de `.env` + cache-clear |
| `app/models.py` (modif) | 8 | `ConversionJob` |
| `app/services/progress.py` (modif) | 8 | Stages de conversión (dinámicos por componente) |
| `app/services/generation_converter.py` (nuevo) | 10 | Cola single-worker de conversión PyTorch→ONNX |
| `app/main.py` (modif) | 11 | Wiring converter + enlace auto-ruteo |
| `app/services/model_installer.py` (modif) | 12 | `InstallJob.conversion_id` |
| `frontend/src/lib/api.ts`, `frontend/src/lib/apiTypes.ts` (modif) | 5, 7, 13 | Tipos + fetchers nuevos |
| `frontend/src/services/generation.ts` (modif) | 5, 13 | `searchGenerationModels`, `convertGenerationModel`, `getConversionStatus` |
| `frontend/src/modules/models/GenerationHfSearch.tsx` (nuevo) | 5 | Buscador de generación en la página Models |
| `frontend/src/modules/models/hfSearchUi.tsx` (nuevo) | 5 | Estados presentacionales extraídos de `HfSearch.tsx` (reuso) |
| `frontend/src/services/settings.ts` (nuevo) + `frontend/src/hooks/useEditableSettings.ts` (nuevo) | 7 | PATCH/GET settings |
| `frontend/src/modules/settings/EditableSettingsSection.tsx` (nuevo) | 7 | Form HF_TOKEN en Settings |
| `frontend/src/hooks/useGenerationJob.ts` (modif) | 13 | Install flow sigue `conversionId` |

---

### Task 1: Spike — verificar `_class_name` reales, clases optimum y API de export

**Files:**
- Create: `docs/superpowers/specs/2026-07-25-third-party-spike-findings.md`
- Create: `scripts/spike_third_party_models.py` (script de evidencia, commiteado)

**Interfaces:**
- Produces: findings doc con (a) tabla `repo_id` real → `_class_name` declarado para SDXL, SDXL Turbo y SD3.5 de la colección `amd/`; (b) qué clases `ORT*Pipeline` existen en `optimum.onnxruntime` con los pins del repo; (c) forma exacta de invocar `optimum.exporters.onnx` para exportar un pipeline diffusers completo y si expone progreso por componente; (d) veredicto GO/NO-GO por variante.
- Las Tasks 2 y 10 usan estos strings/APIs confirmados. **Si el findings doc difiere de los valores hipótesis de este plan, gana el findings doc** (mismo patrón que el spike de `2026-07-22-optimum-spike-findings.md`).

Este task es investigación, no TDD. El `.venv` del proyecto ya tiene los pins (verificado en `pyproject.toml`) — no crear venv aparte.

- [ ] **Step 1: Identificar los repos reales de la colección `amd/`**

Con la API pública de HF (sin bajar pesos — `model_index.json` pesa KBs):

```powershell
# listar candidatos de la colección amd
curl.exe -s "https://huggingface.co/api/models?author=amd&search=stable-diffusion&limit=50" | .venv\Scripts\python -c "import json,sys; [print(m['id']) for m in json.load(sys.stdin)]"
# para cada candidato SDXL / Turbo / SD3.5 elegido, leer su model_index.json:
curl.exe -s "https://huggingface.co/<repo_id>/resolve/main/model_index.json"
```

Registrar en el findings doc: `repo_id` exacto, `_class_name` declarado, lista de componentes declarados, si los componentes son carpetas ONNX o pesos PyTorch. Si algún repo es gated (401/403 al resolver), registrarlo — es dato útil para Task 3.

- [ ] **Step 2: Verificar clases disponibles en optimum.onnxruntime**

```powershell
.venv\Scripts\python -c "import optimum.onnxruntime as m; print([n for n in dir(m) if 'Diffusion' in n or 'StableDiffusion' in n])"
```

Confirmar si existen `ORTStableDiffusionXLPipeline` y `ORTStableDiffusion3Pipeline` (o cómo se llaman realmente). Si SD3.5 NO tiene clase en `optimum-onnx==0.1.0`, registrarlo como NO-GO de SD3.5: la entrada del mapa se omite y se documenta como fase futura (riesgo ya aceptado en el spec).

- [ ] **Step 3: Verificar la API de `optimum.exporters.onnx`**

```powershell
.venv\Scripts\python -c "from optimum.exporters.onnx import main_export; import inspect; print(inspect.signature(main_export))"
```

Responder en el findings doc: ¿`main_export(model_id_local_dir, output_dir, task=...)` sirve para un pipeline diffusers local? ¿Qué `task` corresponde (`text-to-image`)? ¿Emite progreso por componente (logs/callbacks) o hay que exportar componente por componente para tener etapas? ¿Qué componentes produce para SD1.5 (mínimo verificable sin bajar SDXL entero)? Si hay fricción bloqueante con `main_export`, documentar el camino por-componente (`optimum.exporters.onnx.convert` / export de submodelos) — NO ensamblado manual de sesiones (fuera de alcance por spec).

- [ ] **Step 4: Smoke mínimo de export (CPU, repo chico)**

Para no bajar 7GB: exportar un pipeline diffusers PyTorch chico (ej. `hf-internal-testing/tiny-stable-diffusion-torch` o equivalente que el spike encuentre) a ONNX en `%TEMP%`, y confirmar que el output dir queda con `model_index.json` + carpetas de componentes con `.onnx`. Guardar el comando exacto y la estructura resultante en el findings doc. Script commiteado en `scripts/spike_third_party_models.py` (mismo patrón que `scripts/spike_optimum_directml.py`). Limpieza: borrar lo bajado a `%TEMP%`, no commitear pesos ni outputs.

- [ ] **Step 5: Escribir el findings doc y commitear**

```bash
git add docs/superpowers/specs/2026-07-25-third-party-spike-findings.md scripts/spike_third_party_models.py
git commit -m "docs: spike modelos de terceros — _class_name reales, clases optimum y API de export"
```

---

### Task 2: Mapa extensible de clases de pipeline

**Files:**
- Modify: `app/services/engines/generation_onnx.py` (líneas 52-56 `_load_pipeline_class`, línea 204-224 `_create_pipeline`)
- Modify: `app/services/generation_installer.py` (líneas 368-371 `_create_validation_pipeline`)
- Test: `tests/test_generation_engine.py` (agregar), `tests/test_generation_installer.py` (ajustar si algún test rompe)

**Interfaces:**
- Consumes: strings `_class_name` confirmados por el findings doc de Task 1.
- Produces: `PIPELINE_CLASS_NAMES: dict[str, str]`, `_read_declared_class_name(pipeline_dir: Path) -> str`, `_load_pipeline_class(declared_class_name: str) -> Any` (Tasks 9/10 los consumen sin cambios de firma).

- [ ] **Step 1: Escribir los tests que fallan**

En `tests/test_generation_engine.py`:

```python
import json

import pytest

from app.services.engines.generation_onnx import (
    PIPELINE_CLASS_NAMES,
    _load_pipeline_class,
    _read_declared_class_name,
)


@pytest.mark.parametrize(
    ("declared", "expected_ort_name"),
    [
        ("OnnxStableDiffusionPipeline", "ORTStableDiffusionPipeline"),
        ("StableDiffusionXLPipeline", "ORTStableDiffusionXLPipeline"),
        ("StableDiffusion3Pipeline", "ORTStableDiffusion3Pipeline"),
    ],
)
def test_pipeline_class_map_covers_known_variants(declared: str, expected_ort_name: str) -> None:
    # Turbo NO tiene entrada propia: es un checkpoint de la MISMA clase SDXL.
    assert PIPELINE_CLASS_NAMES[declared] == expected_ort_name


def test_load_pipeline_class_unknown_class_lists_supported() -> None:
    with pytest.raises(RuntimeError) as excinfo:
        _load_pipeline_class("KandinskyV22Pipeline")
    message = str(excinfo.value)
    assert "KandinskyV22Pipeline" in message
    for supported in PIPELINE_CLASS_NAMES:
        assert supported in message


def test_read_declared_class_name_reads_model_index(tmp_path) -> None:
    (tmp_path / "model_index.json").write_text(
        json.dumps({"_class_name": "StableDiffusionXLPipeline"}), encoding="utf-8"
    )
    assert _read_declared_class_name(tmp_path) == "StableDiffusionXLPipeline"


def test_read_declared_class_name_missing_class_is_actionable(tmp_path) -> None:
    (tmp_path / "model_index.json").write_text(json.dumps({}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="_class_name"):
        _read_declared_class_name(tmp_path)
```

NOTA: si el findings doc de Task 1 confirmó strings distintos (o descartó SD3.5), ajustar la tabla del parametrize Y el mapa a los valores reales — los de arriba son la hipótesis del spec.

- [ ] **Step 2: Correr los tests y verificar que fallan**

Run: `.venv\Scripts\python -m pytest tests/test_generation_engine.py -k "pipeline_class or declared_class" -v`
Expected: FAIL con `ImportError: cannot import name 'PIPELINE_CLASS_NAMES'`

- [ ] **Step 3: Implementar en `generation_onnx.py`**

Reemplazar `_load_pipeline_class` (líneas 52-56) por:

```python
# _class_name declarados verificados contra repos reales de la colección amd/
# en docs/superpowers/specs/2026-07-25-third-party-spike-findings.md. SDXL
# Turbo es un checkpoint fine-tuneado de la MISMA clase SDXL (menos steps,
# scheduler distinto), no una clase aparte -- sus defaults de inferencia los
# elige el usuario por job (steps/guidance ya son parámetros del request).
PIPELINE_CLASS_NAMES: dict[str, str] = {
    "OnnxStableDiffusionPipeline": "ORTStableDiffusionPipeline",
    "StableDiffusionXLPipeline": "ORTStableDiffusionXLPipeline",
    "StableDiffusion3Pipeline": "ORTStableDiffusion3Pipeline",
}

# Duplicado deliberado de generation_installer.MODEL_INDEX_FILENAME: ese módulo
# ya importa de este; importarlo acá sería un import circular.
_MODEL_INDEX_FILENAME = "model_index.json"


def _read_declared_class_name(pipeline_dir: Path) -> str:
    index_path = pipeline_dir / _MODEL_INDEX_FILENAME
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"No se pudo leer {_MODEL_INDEX_FILENAME} del pipeline en {pipeline_dir.name}: {exc}"
        ) from exc
    declared = index.get("_class_name")
    if not isinstance(declared, str) or not declared:
        raise RuntimeError(
            f"El {_MODEL_INDEX_FILENAME} del pipeline no declara _class_name -- no se puede elegir la clase de carga."
        )
    return declared


def _load_pipeline_class(declared_class_name: str) -> Any:
    ort_class_name = PIPELINE_CLASS_NAMES.get(declared_class_name)
    if ort_class_name is None:
        supported = ", ".join(sorted(PIPELINE_CLASS_NAMES))
        raise RuntimeError(
            f"Clase de pipeline no soportada: {declared_class_name!r}. Clases soportadas: {supported}."
        )
    import optimum.onnxruntime as ort_module

    return getattr(ort_module, ort_class_name)
```

Agregar `import json` al módulo. Actualizar los DOS callers:

En `_create_pipeline` (línea ~207): `pipeline_cls = _load_pipeline_class(_read_declared_class_name(pipeline_dir))`.

En `generation_installer._create_validation_pipeline` (línea ~369): `pipeline_cls = _load_pipeline_class(_read_declared_class_name(pipeline_dir))` — el import de `_read_declared_class_name` se suma al bloque de imports ya existente desde `generation_onnx`.

- [ ] **Step 4: Correr la suite y verificar verde**

Run: `.venv\Scripts\python -m pytest tests/test_generation_engine.py tests/test_generation_installer.py tests/test_generation_job_manager.py -q`
Expected: PASS (si algún test existente moqueaba `_load_pipeline_class` sin argumento, actualizarlo a la firma nueva).

- [ ] **Step 5: Commit**

```bash
git add app/services/engines/generation_onnx.py app/services/generation_installer.py tests/test_generation_engine.py tests/test_generation_installer.py
git commit -m "feat: mapa extensible de clases de pipeline (SD1.5/SDXL/SD3.5) por _class_name declarado"
```

---

### Task 3: Mensajes accionables para repos gated (401/403)

**Files:**
- Modify: `app/exceptions.py`
- Modify: `app/services/hf_client.py` (líneas 207-228 `repo_files`, 245-259 `download`)
- Test: `tests/test_hf_client.py`

**Interfaces:**
- Produces: `HfAuthError(ValueError)` en `app/exceptions.py`; `_wrap_hf_auth_error(exc: Exception, repo_id: str) -> Exception` en `hf_client.py`. Los installers no cambian: `HfAuthError` es `ValueError`, así que `POST /models/install` y `POST /generation/models` ya lo devuelven como 400 con el mensaje, y los jobs lo guardan en `job.error`.

- [ ] **Step 1: Escribir los tests que fallan**

En `tests/test_hf_client.py` (seguir el patrón MockTransport existente del archivo):

```python
import httpx
import pytest

from app.exceptions import HfAuthError
from app.services.hf_client import HfClient, _wrap_hf_auth_error


def _status_error(status_code: int) -> httpx.HTTPStatusError:
    request = httpx.Request("GET", "https://huggingface.co/api/models/x/y")
    response = httpx.Response(status_code, request=request)
    return httpx.HTTPStatusError("boom", request=request, response=response)


def test_wrap_hf_auth_error_401_points_to_settings() -> None:
    wrapped = _wrap_hf_auth_error(_status_error(401), "amd/some-model")
    assert isinstance(wrapped, HfAuthError)
    assert "HF_TOKEN" in str(wrapped)
    assert "Settings" in str(wrapped)


def test_wrap_hf_auth_error_403_names_the_repo_license_page() -> None:
    wrapped = _wrap_hf_auth_error(_status_error(403), "amd/some-model")
    assert isinstance(wrapped, HfAuthError)
    assert "huggingface.co/amd/some-model" in str(wrapped)
    assert "licencia" in str(wrapped)


def test_wrap_hf_auth_error_leaves_other_errors_untouched() -> None:
    original = _status_error(500)
    assert _wrap_hf_auth_error(original, "a/b") is original
    plain = ValueError("x")
    assert _wrap_hf_auth_error(plain, "a/b") is plain


@pytest.mark.anyio
async def test_repo_files_401_raises_actionable_auth_error(make_settings) -> None:
    transport = httpx.MockTransport(lambda request: httpx.Response(401))
    client = HfClient(make_settings(), transport=transport)
    with pytest.raises(HfAuthError, match="HF_TOKEN"):
        await client.repo_files("amd/gated-model")


@pytest.mark.anyio
async def test_download_403_raises_actionable_auth_error(make_settings, tmp_path) -> None:
    transport = httpx.MockTransport(lambda request: httpx.Response(403))
    client = HfClient(make_settings(), transport=transport)
    with pytest.raises(HfAuthError, match="huggingface.co/amd/gated-model"):
        await client.download("amd/gated-model", "model_index.json", tmp_path / "mi.json")
```

(Adaptar el fixture de settings al que ya use `tests/test_hf_client.py` — si no hay `make_settings`, construir `Settings` igual que los tests vecinos. Los tests de retry existentes ya cubren que 401/403 NO reintentan; no duplicarlos.)

- [ ] **Step 2: Correr y verificar que fallan**

Run: `.venv\Scripts\python -m pytest tests/test_hf_client.py -k auth -v`
Expected: FAIL con `ImportError: cannot import name 'HfAuthError'`

- [ ] **Step 3: Implementar**

En `app/exceptions.py` (junto a `HfDownloadTooLargeError`/`HfInvalidSourceError`, mismo estilo):

```python
class HfAuthError(ValueError):
    """401/403 de Hugging Face con mensaje accionable para el usuario."""
```

En `hf_client.py`:

```python
from app.exceptions import HfAuthError, HfDownloadTooLargeError, HfInvalidSourceError


def _wrap_hf_auth_error(exc: Exception, repo_id: str) -> Exception:
    if not isinstance(exc, httpx.HTTPStatusError):
        return exc
    status = exc.response.status_code
    if status == 401:
        return HfAuthError(
            "Tu HF_TOKEN no es válido o no está configurado — revisalo en Settings."
        )
    if status == 403:
        return HfAuthError(
            f"El repo {repo_id} requiere aceptar su licencia en "
            f"huggingface.co/{repo_id} antes de poder descargarlo."
        )
    return exc
```

En `repo_files` (línea ~220) y en `download` (línea ~251), reemplazar el `raise` de la rama no-reintentable:

```python
            except Exception as exc:  # noqa: BLE001 -- CancelledError is BaseException, so cancel still propagates
                if attempt == DOWNLOAD_ATTEMPTS or not _is_retryable_download_error(exc):
                    raise _wrap_hf_auth_error(exc, repo_id) from exc
```

(En `download` mantener el `tmp_path.unlink(missing_ok=True)` previo al check, tal como está.)

- [ ] **Step 4: Correr suite y verificar verde**

Run: `.venv\Scripts\python -m pytest tests/test_hf_client.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add app/exceptions.py app/services/hf_client.py tests/test_hf_client.py
git commit -m "feat: mensajes accionables para repos gated de Hugging Face (401 token / 403 licencia)"
```

---

### Task 4: Búsqueda de modelos de generación (backend)

**Files:**
- Modify: `app/services/hf_client.py` (líneas 68, 192-205 `search`)
- Modify: `app/api/routes.py` (después del bloque `/generation/models/install/{install_id}`, línea ~998)
- Test: `tests/test_hf_client.py`, `tests/test_generation_api.py`

**Interfaces:**
- Produces: `HfClient.search(query: str, limit: int = 20, task_tags: tuple[str, ...] | None = None)`; constante `GENERATION_SEARCH_TASK_TAGS = ("text-to-image",)`; endpoint `GET /api/v1/generation/models/search?q=...` con el mismo shape `ModelSearchResponse` que `/models/search` (el frontend de Task 5 lo consume tal cual).

- [ ] **Step 1: Escribir los tests que fallan**

En `tests/test_hf_client.py`:

```python
from app.services.hf_client import GENERATION_SEARCH_TASK_TAGS, SEARCH_TASK_TAGS


@pytest.mark.anyio
async def test_search_default_uses_upscaler_task_tags(make_settings) -> None:
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["params"] = request.url.params
        return httpx.Response(200, json=[])

    client = HfClient(make_settings(), transport=httpx.MockTransport(handler))
    await client.search("esrgan")
    assert captured["params"].get_list("filter") == list(SEARCH_TASK_TAGS)


@pytest.mark.anyio
async def test_search_task_tags_override(make_settings) -> None:
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["params"] = request.url.params
        return httpx.Response(200, json=[])

    client = HfClient(make_settings(), transport=httpx.MockTransport(handler))
    await client.search("sdxl", task_tags=GENERATION_SEARCH_TASK_TAGS)
    assert captured["params"].get_list("filter") == ["text-to-image"]
```

En `tests/test_generation_api.py` (usar el client/fixture de app existente del archivo; moquear `app.state.hf_client` con un fake que registre `task_tags`):

```python
def test_generation_search_endpoint_uses_generation_tags(client, app) -> None:
    calls: dict = {}

    class FakeHf:
        async def search(self, query, limit=20, task_tags=None):
            calls["query"] = query
            calls["task_tags"] = task_tags
            return []

    app.state.hf_client = FakeHf()
    response = client.get("/api/v1/generation/models/search", params={"q": "sdxl"})
    assert response.status_code == 200
    assert response.json() == {"results": []}
    assert calls["task_tags"] == ("text-to-image",)
```

- [ ] **Step 2: Correr y verificar que fallan**

Run: `.venv\Scripts\python -m pytest tests/test_hf_client.py -k task_tags -v`
Expected: FAIL con `ImportError: cannot import name 'GENERATION_SEARCH_TASK_TAGS'`

- [ ] **Step 3: Implementar**

En `hf_client.py`, junto a `SEARCH_TASK_TAGS` (línea 68):

```python
GENERATION_SEARCH_TASK_TAGS = ("text-to-image",)
```

Y en `search`:

```python
    async def search(
        self, query: str, limit: int = 20, task_tags: tuple[str, ...] | None = None
    ) -> list[HfModelSummary]:
        tags = SEARCH_TASK_TAGS if task_tags is None else task_tags
        params = {
            "search": query,
            "filter": list(tags),
            "limit": limit,
            "full": "true",
        }
```

En `routes.py` (import `GENERATION_SEARCH_TASK_TAGS` desde `app.services.hf_client`; el helper de mapeo se comparte con `/models/search` extrayendo el cuerpo a una función local):

```python
def _search_results_to_response(results: list) -> ModelSearchResponse:
    return ModelSearchResponse(
        results=[
            HfModelSearchResultResponse(
                id=item.id,
                author=item.author,
                pipeline_tag=item.pipeline_tag,
                downloads=item.downloads,
                likes=item.likes,
                tags=list(item.tags),
            )
            for item in results
        ]
    )


@router.get("/generation/models/search", response_model=ModelSearchResponse)
async def search_generation_models(
    q: str = Query(..., min_length=1),
    hf_client: HfClient = Depends(get_hf_client),
) -> ModelSearchResponse:
    try:
        results = await hf_client.search(q, task_tags=GENERATION_SEARCH_TASK_TAGS)
    except Exception as exc:
        logger.exception("Hugging Face generation search failed for query %r", q)
        raise HTTPException(status_code=502, detail="Hugging Face search failed") from exc
    return _search_results_to_response(results)
```

Refactorizar `search_models` existente (línea 822-844) para usar `_search_results_to_response` (mismo comportamiento).

- [ ] **Step 4: Correr suite y verificar verde**

Run: `.venv\Scripts\python -m pytest tests/test_hf_client.py tests/test_generation_api.py tests/test_models_api.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add app/services/hf_client.py app/api/routes.py tests/test_hf_client.py tests/test_generation_api.py
git commit -m "feat: búsqueda de modelos de generación (task tag text-to-image) con endpoint propio"
```

---

### Task 5: Búsqueda de modelos de generación (frontend)

**Files:**
- Create: `frontend/src/modules/models/hfSearchUi.tsx` (estados presentacionales extraídos)
- Modify: `frontend/src/modules/models/HfSearch.tsx` (usar los extraídos, sin cambio visual)
- Create: `frontend/src/modules/models/GenerationHfSearch.tsx`
- Modify: `frontend/src/modules/models/GenerationModelsSection.tsx` (montar el buscador arriba del form de repo_id)
- Modify: `frontend/src/services/generation.ts`, `frontend/src/hooks/useGenerationJob.ts` (hook de búsqueda)
- Test: `frontend/src/modules/models/GenerationHfSearch.test.tsx`

**Interfaces:**
- Consumes: `GET /api/v1/generation/models/search` (Task 4), `useGenerationModelInstall` (existente).
- Produces: `searchGenerationModels(query: string): Promise<ModelSearchResponse>` en `services/generation.ts`; `useGenerationHfSearchResults(query: string)` en `hooks/useGenerationJob.ts`; componente `GenerationHfSearch`.

- [ ] **Step 1: Extraer estados presentacionales de `HfSearch.tsx`**

Mover `SearchEmptyState` (parametrizando el copy), `NoResultsState`, `SearchErrorState`, `SearchLoadingState`, `SearchInput` y `DEFAULT_SEARCH_DEBOUNCE_MS` a `frontend/src/modules/models/hfSearchUi.tsx` y exportarlos; `HfSearch.tsx` los importa desde ahí. `SearchEmptyState` gana prop `message: string` (el actual pasa "Search Hugging Face for an ONNX upscaling model to install."). Sin cambio de comportamiento.

Run: `cd frontend && npx vitest run src/modules/models/HfSearch.test.tsx`
Expected: PASS (los tests existentes no cambian)

- [ ] **Step 2: Escribir el test que falla para `GenerationHfSearch`**

`frontend/src/modules/models/GenerationHfSearch.test.tsx` (calcar setup de `HfSearch.test.tsx`: QueryClientProvider wrapper + mock de servicios):

```tsx
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { GenerationHfSearch } from "./GenerationHfSearch";

const searchGenerationModels = vi.hoisted(() => vi.fn());
const installGenerationModel = vi.hoisted(() => vi.fn());
vi.mock("../../services/generation", async (importOriginal) => ({
  ...(await importOriginal<typeof import("../../services/generation")>()),
  searchGenerationModels,
  installGenerationModel,
}));

describe("GenerationHfSearch", () => {
  it("busca con el endpoint de generación y muestra resultados", async () => {
    searchGenerationModels.mockResolvedValue({
      results: [{ id: "amd/sdxl-onnx", author: "amd", pipelineTag: "text-to-image", downloads: 10, likes: 2, tags: [] }],
    });
    render(<GenerationHfSearch debounceMs={0} />, { wrapper: createWrapper() });
    await userEvent.type(screen.getByRole("searchbox"), "sdxl");
    await waitFor(() => expect(searchGenerationModels).toHaveBeenCalledWith("sdxl"));
    expect(await screen.findByText("amd/sdxl-onnx")).toBeInTheDocument();
  });

  it("instala con el endpoint de generación al clickear Install", async () => {
    searchGenerationModels.mockResolvedValue({
      results: [{ id: "amd/sdxl-onnx", author: "amd", pipelineTag: "text-to-image", downloads: 10, likes: 2, tags: [] }],
    });
    installGenerationModel.mockResolvedValue({ installId: "abc", statusUrl: "/x" });
    render(<GenerationHfSearch debounceMs={0} />, { wrapper: createWrapper() });
    await userEvent.type(screen.getByRole("searchbox"), "sdxl");
    await userEvent.click(await screen.findByRole("button", { name: /install/i }));
    await waitFor(() => expect(installGenerationModel).toHaveBeenCalledWith("amd/sdxl-onnx"));
  });
});
```

(`createWrapper` = el helper de QueryClient que ya usen los tests vecinos; copiarlo si es local a cada archivo.)

- [ ] **Step 3: Correr y verificar que falla**

Run: `cd frontend && npx vitest run src/modules/models/GenerationHfSearch.test.tsx`
Expected: FAIL (módulo no existe)

- [ ] **Step 4: Implementar**

`services/generation.ts`:

```ts
import type { ModelSearchResponse } from "../lib/apiTypes";

export function searchGenerationModels(query: string): Promise<ModelSearchResponse> {
  return apiGet<ModelSearchResponse>(`/generation/models/search?q=${encodeURIComponent(query)}`);
}
```

`hooks/useGenerationJob.ts`:

```ts
export function useGenerationHfSearchResults(query: string) {
  const trimmed = query.trim();
  return useQuery<ModelSearchResponse>({
    queryKey: ["generation-hf-search", trimmed],
    queryFn: () => searchGenerationModels(trimmed),
    enabled: trimmed.length > 0,
  });
}
```

`GenerationHfSearch.tsx`: mismo esqueleto que `HfSearch` pero con `useGenerationHfSearchResults` y una card de resultado que usa `useGenerationModelInstall` (en vez de `useModelInstall`). La card puede ser una variante local `GenerationResultCard` que copia el layout de `HfResultCard` cambiando solo el hook (los subcomponentes `InstalledIndicator`/`InstallButton`/`ResultMeta` se importan si `HfResultCard.tsx` los exporta — exportarlos como parte de este step). Copy del empty state: `"Search Hugging Face for a Stable Diffusion (text-to-image) pipeline to install."`.

`GenerationModelsSection.tsx`: montar `<GenerationHfSearch />` entre el `<h2>` y `RepoIdForm` (el form manual de repo_id se conserva como fallback).

- [ ] **Step 5: Correr suite frontend y verificar verde**

Run: `cd frontend && npx vitest run src/modules/models`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add frontend/src/modules/models frontend/src/services/generation.ts frontend/src/hooks/useGenerationJob.ts
git commit -m "feat: buscador de modelos de generación en la página Models"
```

---

### Task 6: Settings editables genéricas (backend)

**Files:**
- Create: `app/services/settings_service.py`
- Modify: `app/api/routes.py`, `app/schemas.py`
- Test: `tests/test_settings_service.py` (nuevo), `tests/test_settings_api.py` (nuevo)

**Interfaces:**
- Produces: `EDITABLE_SETTINGS_WHITELIST = frozenset({"hf_token"})`; `update_setting(key: str, value: str) -> None`; `editable_settings_status(settings: Settings) -> list[EditableSettingStatus]`; excepciones `SettingNotEditableError(ValueError)` y `SettingValueError(ValueError)`; endpoints `GET /api/v1/settings` y `PATCH /api/v1/settings`.
- El frontend (Task 7) consume `{ settings: [{ key: "hf_token", configured: bool }] }` y `PATCH {key, value} -> 200 {key}`.

Contexto: el subproyecto C YA está mergeado en master — `app/services/json_store.py::write_text_atomically` y `app.config.ENV_FILE_PATH` existen y se REUSAN (no duplicar). El spec pide extender el patrón `_append_env_var` de `config.py` (append-si-falta) a update-si-existe-o-append: esa lógica vive en `settings_service.py`. Gate de permisos: `GET /settings` → `Permission.settings_read`, `PATCH /settings` → `Permission.settings_write` (mismo patrón que `capability_routes.py` líneas 48/57/84).

- [ ] **Step 1: Escribir los tests que fallan (`tests/test_settings_service.py`)**

```python
import threading
from pathlib import Path

import pytest

from app.config import Settings, get_settings
from app.services import settings_service
from app.services.settings_service import (
    EDITABLE_SETTINGS_WHITELIST,
    SettingNotEditableError,
    SettingValueError,
    editable_settings_status,
    update_setting,
)


@pytest.fixture()
def env_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / ".env"
    monkeypatch.setattr(settings_service, "ENV_FILE_PATH", path)
    return path


def test_whitelist_only_contains_hf_token() -> None:
    assert EDITABLE_SETTINGS_WHITELIST == frozenset({"hf_token"})


def test_update_setting_rejects_key_outside_whitelist(env_file: Path) -> None:
    with pytest.raises(SettingNotEditableError, match="app_port"):
        update_setting("app_port", "9999")
    assert not env_file.exists()


def test_update_setting_appends_when_env_missing(env_file: Path) -> None:
    update_setting("hf_token", "hf_abc123")
    assert env_file.read_text(encoding="utf-8").strip() == "HF_TOKEN=hf_abc123"


def test_update_setting_replaces_existing_line_preserving_others(env_file: Path) -> None:
    env_file.write_text("APP_PORT=8090\nHF_TOKEN=hf_old\nDEFAULT_DEVICE=dml:0\n", encoding="utf-8")
    update_setting("hf_token", "hf_new")
    lines = env_file.read_text(encoding="utf-8").splitlines()
    assert lines == ["APP_PORT=8090", "HF_TOKEN=hf_new", "DEFAULT_DEVICE=dml:0"]


def test_update_setting_clears_get_settings_cache(env_file: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # get_settings lee env_file=".env" relativo al CWD -- se apunta el CWD al
    # tmp para que la lectura y la escritura miren el mismo archivo.
    monkeypatch.chdir(env_file.parent)
    get_settings.cache_clear()
    assert get_settings().hf_token is None
    update_setting("hf_token", "hf_fresh")
    assert get_settings().hf_token == "hf_fresh"
    get_settings.cache_clear()


def test_concurrent_updates_do_not_corrupt_env(env_file: Path) -> None:
    env_file.write_text("APP_PORT=8090\n", encoding="utf-8")
    errors: list[Exception] = []

    def writer(value: str) -> None:
        try:
            for _ in range(20):
                update_setting("hf_token", value)
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=writer, args=(f"hf_{i}",)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    lines = env_file.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "APP_PORT=8090"
    assert len([l for l in lines if l.startswith("HF_TOKEN=")]) == 1


def test_editable_settings_status_reports_configured_flag() -> None:
    configured = editable_settings_status(Settings(_env_file=None, HF_TOKEN="x"))
    assert configured == [{"key": "hf_token", "configured": True}]
    empty = editable_settings_status(Settings(_env_file=None))
    assert empty == [{"key": "hf_token", "configured": False}]
```

NOTA para el implementer: si `Settings(_env_file=None)` igual levanta `HF_TOKEN` del entorno del proceso en tu máquina de test, agregar `monkeypatch.delenv("HF_TOKEN", raising=False)` en los tests afectados.

- [ ] **Step 2: Correr y verificar que fallan**

Run: `.venv\Scripts\python -m pytest tests/test_settings_service.py -v`
Expected: FAIL con `ModuleNotFoundError: No module named 'app.services.settings_service'`

- [ ] **Step 3: Implementar `app/services/settings_service.py`**

```python
from __future__ import annotations

import threading
from typing import TypedDict

from pydantic import ValidationError

from app.config import ENV_FILE_PATH, Settings, get_settings
from app.services.json_store import write_text_atomically

# Primer campo real de la whitelist. Crece en subproyectos futuros sin tocar
# el mecanismo (spec 2026-07-25-generation-third-party-models-design.md §5).
EDITABLE_SETTINGS_WHITELIST = frozenset({"hf_token"})

# Serializa read-modify-write del .env entre requests concurrentes.
_ENV_WRITE_LOCK = threading.Lock()


class SettingNotEditableError(ValueError):
    pass


class SettingValueError(ValueError):
    pass


class EditableSettingStatus(TypedDict):
    key: str
    configured: bool


def _env_alias(key: str) -> str:
    field = Settings.model_fields[key]
    return field.alias or key.upper()


def _validate_value(key: str, value: str) -> None:
    # Reusa la validación pydantic real del campo: un valor inválido para el
    # tipo del campo revienta acá con 400, nunca llega al .env.
    try:
        Settings(_env_file=None, **{_env_alias(key): value})
    except ValidationError as exc:
        raise SettingValueError(
            f"Valor inválido para {key}: {exc.errors()[0].get('msg', 'validación fallida')}"
        ) from exc


def _render_env_text(existing_text: str, alias: str, value: str) -> str:
    prefix = f"{alias}="
    lines = existing_text.splitlines()
    replaced = False
    for index, line in enumerate(lines):
        if line.strip().startswith(prefix):
            lines[index] = f"{alias}={value}"
            replaced = True
            break
    if not replaced:
        lines.append(f"{alias}={value}")
    return "\n".join(lines) + "\n"


def update_setting(key: str, value: str) -> None:
    if key not in EDITABLE_SETTINGS_WHITELIST:
        raise SettingNotEditableError(f"El setting {key!r} no es editable desde la UI.")
    _validate_value(key, value)
    alias = _env_alias(key)
    with _ENV_WRITE_LOCK:
        # Extiende _append_env_var de config.py (append-si-falta) a
        # update-si-existe-o-append, con la misma escritura atómica.
        existing = ENV_FILE_PATH.read_text(encoding="utf-8") if ENV_FILE_PATH.exists() else ""
        write_text_atomically(ENV_FILE_PATH, _render_env_text(existing, alias, value))
    get_settings.cache_clear()


def editable_settings_status(settings: Settings) -> list[EditableSettingStatus]:
    return [
        {"key": key, "configured": bool(getattr(settings, key))}
        for key in sorted(EDITABLE_SETTINGS_WHITELIST)
    ]
```

- [ ] **Step 4: Correr y verificar verde**

Run: `.venv\Scripts\python -m pytest tests/test_settings_service.py -v`
Expected: PASS

- [ ] **Step 5: Escribir tests de API que fallan (`tests/test_settings_api.py`)**

Calcar el setup de client de `tests/test_models_api.py` (TestClient + app fixture):

```python
def test_get_settings_lists_editable_keys(client) -> None:
    response = client.get("/api/v1/settings")
    assert response.status_code == 200
    payload = response.json()
    assert payload["settings"][0]["key"] == "hf_token"
    assert isinstance(payload["settings"][0]["configured"], bool)


def test_patch_setting_outside_whitelist_is_400(client) -> None:
    response = client.patch("/api/v1/settings", json={"key": "app_port", "value": "1"})
    assert response.status_code == 400
    assert "no es editable" in response.json()["detail"]


def test_patch_hf_token_persists(client, tmp_path, monkeypatch) -> None:
    from app.services import settings_service

    env_path = tmp_path / ".env"
    monkeypatch.setattr(settings_service, "ENV_FILE_PATH", env_path)
    response = client.patch("/api/v1/settings", json={"key": "hf_token", "value": "hf_xyz"})
    assert response.status_code == 200
    assert response.json() == {"key": "hf_token"}
    assert "HF_TOKEN=hf_xyz" in env_path.read_text(encoding="utf-8")
```

- [ ] **Step 6: Implementar endpoints**

`app/schemas.py`:

```python
class EditableSettingStatusResponse(BaseModel):
    key: str
    configured: bool


class EditableSettingsResponse(BaseModel):
    settings: list[EditableSettingStatusResponse]


class UpdateSettingRequest(BaseModel):
    key: str = Field(min_length=1)
    value: str


class UpdateSettingResponse(BaseModel):
    key: str
```

`app/api/routes.py` (al final, sección nueva; imports de `settings_service`):

```python
# Gates con los permisos que C ya define (mismo patrón que capability_routes):
# settings_read para leer, settings_write para escribir. Con AUTH_MODE=off el
# usuario off-mode tiene todos los permisos y esto es transparente.
@router.get(
    "/settings", response_model=EditableSettingsResponse,
    dependencies=[Depends(require(Permission.settings_read))],
)
async def get_editable_settings(settings: Settings = Depends(get_settings)) -> EditableSettingsResponse:
    return EditableSettingsResponse(
        settings=[EditableSettingStatusResponse(**item) for item in editable_settings_status(settings)]
    )


@router.patch(
    "/settings", response_model=UpdateSettingResponse,
    dependencies=[Depends(require(Permission.settings_write))],
)
async def patch_setting(payload: UpdateSettingRequest) -> UpdateSettingResponse:
    try:
        await asyncio.to_thread(update_setting, payload.key, payload.value)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return UpdateSettingResponse(key=payload.key)
```

(`routes.py` ya importa `asyncio`; si no, agregarlo.)

- [ ] **Step 7: Correr suite y verificar verde**

Run: `.venv\Scripts\python -m pytest tests/test_settings_service.py tests/test_settings_api.py -q && .venv\Scripts\python -m pytest -q`
Expected: PASS

- [ ] **Step 8: Commit**

```bash
git add app/services/settings_service.py app/api/routes.py app/schemas.py tests/test_settings_service.py tests/test_settings_api.py
git commit -m "feat: settings editables genéricas con whitelist y escritura atómica de .env (primer campo: HF_TOKEN)"
```

---

### Task 7: Settings editables (frontend)

**Files:**
- Create: `frontend/src/services/settings.ts`
- Create: `frontend/src/hooks/useEditableSettings.ts`
- Create: `frontend/src/modules/settings/EditableSettingsSection.tsx`
- Modify: `frontend/src/modules/settings/SettingsPage.tsx` (montar la sección + suavizar `EnvExplanationNote`)
- Modify: `frontend/src/lib/apiTypes.ts`
- Test: `frontend/src/modules/settings/EditableSettingsSection.test.tsx`

**Interfaces:**
- Consumes: `GET/PATCH /api/v1/settings` (Task 6).
- Produces: `fetchEditableSettings(): Promise<EditableSettingsResponse>`, `patchSetting(key: string, value: string): Promise<{key: string}>`; componente `EditableSettingsSection` con input tipo password para HF_TOKEN.

- [ ] **Step 1: Escribir el test que falla**

`frontend/src/modules/settings/EditableSettingsSection.test.tsx`:

```tsx
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { EditableSettingsSection } from "./EditableSettingsSection";

const fetchEditableSettings = vi.hoisted(() => vi.fn());
const patchSetting = vi.hoisted(() => vi.fn());
vi.mock("../../services/settings", () => ({ fetchEditableSettings, patchSetting }));

describe("EditableSettingsSection", () => {
  it("muestra HF token como no configurado y guarda con PATCH", async () => {
    fetchEditableSettings.mockResolvedValue({ settings: [{ key: "hf_token", configured: false }] });
    patchSetting.mockResolvedValue({ key: "hf_token" });
    render(<EditableSettingsSection />, { wrapper: createWrapper() });
    expect(await screen.findByText(/not configured/i)).toBeInTheDocument();
    const input = screen.getByLabelText(/hugging face token/i);
    expect(input).toHaveAttribute("type", "password");
    await userEvent.type(input, "hf_secret");
    await userEvent.click(screen.getByRole("button", { name: /save/i }));
    await waitFor(() => expect(patchSetting).toHaveBeenCalledWith("hf_token", "hf_secret"));
    expect(await screen.findByText(/saved/i)).toBeInTheDocument();
  });

  it("muestra el error del backend si el PATCH falla", async () => {
    fetchEditableSettings.mockResolvedValue({ settings: [{ key: "hf_token", configured: true }] });
    patchSetting.mockRejectedValue(new Error("Valor inválido para hf_token"));
    render(<EditableSettingsSection />, { wrapper: createWrapper() });
    await userEvent.type(await screen.findByLabelText(/hugging face token/i), "x");
    await userEvent.click(screen.getByRole("button", { name: /save/i }));
    expect(await screen.findByRole("alert")).toHaveTextContent("Valor inválido");
  });
});
```

- [ ] **Step 2: Correr y verificar que falla**

Run: `cd frontend && npx vitest run src/modules/settings/EditableSettingsSection.test.tsx`
Expected: FAIL (módulo no existe)

- [ ] **Step 3: Implementar**

`lib/apiTypes.ts`:

```ts
export interface EditableSettingStatus {
  key: string;
  configured: boolean;
}

export interface EditableSettingsResponse {
  settings: EditableSettingStatus[];
}
```

`services/settings.ts` (`apiGet` y `apiPatchJson` YA existen en `lib/api.ts` — C agregó `apiPatchJson`; no crear otro):

```ts
import { apiGet, apiPatchJson } from "../lib/api";
import type { EditableSettingsResponse } from "../lib/apiTypes";

export function fetchEditableSettings(): Promise<EditableSettingsResponse> {
  return apiGet<EditableSettingsResponse>("/settings");
}

export function patchSetting(key: string, value: string): Promise<{ key: string }> {
  return apiPatchJson<{ key: string }>("/settings", { key, value });
}
```

`hooks/useEditableSettings.ts`: `useQuery({queryKey: ["editable-settings"], queryFn: fetchEditableSettings})` + `useMutation({mutationFn: ({key, value}) => patchSetting(key, value), onSuccess: invalidate ["editable-settings"]})`.

`EditableSettingsSection.tsx`: card con heading "Credentials", fila HF token: label "Hugging Face token", badge Configured/Not configured (dato del GET), `<input type="password">`, botón Save (deshabilitado con input vacío), estado "Saved" tras éxito (limpia el input), `<p role="alert">` con el mensaje si falla. Estilos calcados de las cards existentes de `SettingsPage.tsx`.

`SettingsPage.tsx`: montar `<EditableSettingsSection />` en el grid; actualizar el texto de `EnvExplanationNote` para que no afirme que TODO es solo-lectura (ej. "Most values come from .env at install time and are read-only here; the Credentials section below is editable.").

- [ ] **Step 4: Correr suite frontend y verificar verde**

Run: `cd frontend && npx vitest run src/modules/settings`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add frontend/src/services/settings.ts frontend/src/hooks/useEditableSettings.ts frontend/src/modules/settings frontend/src/lib
git commit -m "feat: sección de settings editables en la UI (HF token como password) con GET/PATCH /settings"
```

---

### Task 8: `ConversionJob` + stages de conversión

**Files:**
- Modify: `app/models.py`
- Modify: `app/services/progress.py`
- Test: `tests/test_progress.py`

**Interfaces:**
- Produces: dataclass `ConversionJob` (models.py) con campos `repo_id`, `id`, `status: JobStatus`, `created_at/started_at/finished_at`, `error`, `model_id`, `metadata`; en progress.py: `build_conversion_stages(component_keys: list[str]) -> list[Stage]`, `advance_conversion_stage(job, component_keys, stage_key)`, `complete_conversion_stages(job, component_keys)`. Stage keys: `"downloading"`, `"exporting:<componente>"` (uno por componente), `"validating"`.
- Task 10 consume las tres funciones y el dataclass.

- [ ] **Step 1: Escribir los tests que fallan (`tests/test_progress.py`)**

```python
from app.models import ConversionJob, JobStatus
from app.services.progress import (
    advance_conversion_stage,
    build_conversion_stages,
    complete_conversion_stages,
)


def test_build_conversion_stages_one_export_stage_per_component() -> None:
    stages = build_conversion_stages(["unet", "vae_decoder", "text_encoder"])
    keys = [stage.key for stage in stages]
    assert keys == [
        "downloading",
        "exporting:unet",
        "exporting:vae_decoder",
        "exporting:text_encoder",
        "validating",
    ]
    assert abs(sum(stage.weight for stage in stages) - 1.0) < 1e-9


def test_build_conversion_stages_without_components_uses_single_export_stage() -> None:
    # Antes de leer model_index.json no se conocen los componentes.
    keys = [stage.key for stage in build_conversion_stages([])]
    assert keys == ["downloading", "exporting", "validating"]


def test_advance_conversion_stage_writes_job_metadata() -> None:
    job = ConversionJob(repo_id="amd/x")
    advance_conversion_stage(job, ["unet"], "exporting:unet")
    assert job.metadata["stage"] == "exporting:unet"
    assert job.metadata["stages"][0]["status"] == "done"      # downloading
    assert job.metadata["stages"][1]["status"] == "active"    # exporting:unet
    assert 0.0 < job.metadata["progress"] < 1.0


def test_complete_conversion_stages_marks_all_done() -> None:
    job = ConversionJob(repo_id="amd/x")
    complete_conversion_stages(job, ["unet"])
    assert job.metadata["progress"] == 1.0
    assert all(stage["status"] == "done" for stage in job.metadata["stages"])


def test_conversion_job_defaults() -> None:
    job = ConversionJob(repo_id="amd/x")
    assert job.status == JobStatus.queued
    assert job.model_id is None and job.error is None
```

- [ ] **Step 2: Correr y verificar que fallan**

Run: `.venv\Scripts\python -m pytest tests/test_progress.py -k conversion -v`
Expected: FAIL con `ImportError: cannot import name 'ConversionJob'`

- [ ] **Step 3: Implementar**

`app/models.py` (después de `GenerationJob`):

```python
@dataclass(slots=True)
class ConversionJob:
    repo_id: str
    id: str = field(default_factory=lambda: uuid4().hex)
    status: JobStatus = JobStatus.queued
    created_at: datetime = field(default_factory=utc_now)
    started_at: datetime | None = None
    finished_at: datetime | None = None
    error: str | None = None
    model_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
```

`app/services/progress.py` (después del bloque de generación):

```python
CONVERSION_DOWNLOAD_STAGE = ("downloading", "Downloading weights", 15.0)
CONVERSION_VALIDATE_STAGE = ("validating", "Validating pipeline", 15.0)
CONVERSION_EXPORT_TOTAL_WEIGHT = 70.0


def build_conversion_stages(component_keys: list[str]) -> list[Stage]:
    if component_keys:
        export_weight = CONVERSION_EXPORT_TOTAL_WEIGHT / len(component_keys)
        export_stages = [
            (f"exporting:{key}", f"Exporting {key}", export_weight) for key in component_keys
        ]
    else:
        export_stages = [("exporting", "Exporting to ONNX", CONVERSION_EXPORT_TOTAL_WEIGHT)]
    raw_stages = [CONVERSION_DOWNLOAD_STAGE, *export_stages, CONVERSION_VALIDATE_STAGE]
    return _normalize_weights(raw_stages)


def advance_conversion_stage(job: HasMetadata, component_keys: list[str], stage_key: str) -> None:
    stages = apply_stage_transition(build_conversion_stages(component_keys), stage_key)
    _write_stage_metadata(job, stages, stage_key)


def complete_conversion_stages(job: HasMetadata, component_keys: list[str]) -> None:
    stages = mark_all_done(build_conversion_stages(component_keys))
    _write_stage_metadata(job, stages, "completed", progress_override=1.0)
```

(`HasMetadata` ya existe en progress.py; `ConversionJob` lo satisface.)

- [ ] **Step 4: Correr suite y verificar verde**

Run: `.venv\Scripts\python -m pytest tests/test_progress.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add app/models.py app/services/progress.py tests/test_progress.py
git commit -m "feat: ConversionJob y etapas de progreso por componente para conversión PyTorch→ONNX"
```

---

### Task 9: Refactor — extraer `validate_and_promote` del installer de generación

**Files:**
- Modify: `app/services/generation_installer.py` (líneas 252-314 `_download_and_register`)
- Test: `tests/test_generation_installer.py` (la suite existente debe seguir verde sin cambios de asserts; agregar un test directo)

**Interfaces:**
- Produces: método público `GenerationModelInstaller.validate_and_promote(staging_root: Path, repo_id: str, size_bytes: int) -> str` (devuelve `model_id`). Ejecuta, en este orden: `_validate_structure`, `_patch_legacy_component_configs`, validación funcional bajo `device_semaphores` (forward-pass real vía `_validate_pipeline`), promoción atómica y registro en `ModelRegistry`. Task 10 lo llama con el staging dir producido por la conversión — la conversión NO introduce un segundo camino de instalación.

- [ ] **Step 1: Escribir el test que falla**

En `tests/test_generation_installer.py` (reusar `make_installer`/`FakeValidationPipeline` existentes):

```python
def test_validate_and_promote_registers_from_arbitrary_staging(tmp_path: Path, monkeypatch) -> None:
    installer, settings, registry = make_installer(tmp_path)
    monkeypatch.setattr(installer, "_create_validation_pipeline", lambda pipeline_dir: FakeValidationPipeline())
    staging = settings.temp_path / "conv-staging"
    (staging / "unet").mkdir(parents=True)
    (staging / "model_index.json").write_text(
        json.dumps({"_class_name": "StableDiffusionXLPipeline", "unet": ["diffusers", "x"]}),
        encoding="utf-8",
    )
    model_id = asyncio.run(installer.validate_and_promote(staging, "amd/conv-model", 123))
    entry = registry.get(model_id)
    assert entry is not None
    assert entry.kind == ModelKind.diffusion_onnx
    assert entry.size_bytes == 123
    assert (settings.models_path / "generation" / model_id / "model_index.json").exists()
```

(Ajustar la tupla de retorno de `make_installer` a lo que realmente devuelva ese helper — el implementer lo lee en el archivo.)

- [ ] **Step 2: Correr y verificar que falla**

Run: `.venv\Scripts\python -m pytest tests/test_generation_installer.py -k validate_and_promote -v`
Expected: FAIL con `AttributeError: ... has no attribute 'validate_and_promote'`

- [ ] **Step 3: Implementar el refactor**

Extraer de `_download_and_register` (líneas 288-309) a:

```python
    async def validate_and_promote(self, staging_root: Path, repo_id: str, size_bytes: int) -> str:
        _validate_structure(staging_root)
        _patch_legacy_component_configs(staging_root)
        async with self.device_semaphores.acquire(self.settings.default_device):
            await asyncio.to_thread(self._validate_pipeline, staging_root)
        model_id = _generation_model_id(repo_id)
        final_dir = self.settings.models_path / GENERATION_MODELS_SUBDIR / model_id
        async with self._lock_for(model_id):
            await self._promote_staging_dir(staging_root, final_dir)
            entry = ModelEntry(
                id=model_id,
                name=repo_id,
                kind=ModelKind.diffusion_onnx,
                source=f"hf:{repo_id}",
                size_bytes=size_bytes,
                scale=None,
                file_path=f"{GENERATION_MODELS_SUBDIR}/{model_id}",
                status=ModelStatus.installed,
            )
            self.registry.register(entry)
        return model_id
```

`_download_and_register` queda: fase de descarga igual que hoy, luego `job.status = InstallStatus.validating`, `job.model_id = await self.validate_and_promote(staging_root, job.repo_id, sum(f.size for f in selected))`, `job.status = InstallStatus.installed`. El `finally: shutil.rmtree(staging_root)` no cambia. Comportamiento idéntico — es un refactor puro.

- [ ] **Step 4: Correr la suite completa del installer**

Run: `.venv\Scripts\python -m pytest tests/test_generation_installer.py -q`
Expected: PASS, cero asserts cambiados en tests preexistentes

- [ ] **Step 5: Commit**

```bash
git add app/services/generation_installer.py tests/test_generation_installer.py
git commit -m "refactor: extrae validate_and_promote reusable del installer de generación"
```

---

### Task 10: Servicio de conversión PyTorch→ONNX

**Files:**
- Create: `app/services/generation_converter.py`
- Test: `tests/test_generation_converter.py` (nuevo)

**Interfaces:**
- Consumes: `ConversionJob`/stages (Task 8), `installer.validate_and_promote` (Task 9), `HfClient` (descarga), `_generation_model_id`/`_read_declared_components`/`_safe_staging_dest`/`_ensure_model_index_listed`/`_ensure_size_cap`/`MODEL_INDEX_FILENAME` (import desde `generation_installer`).
- Produces: `GenerationModelConverter` con `start()/stop()`, `async convert_from_hf(repo_id: str) -> str` (job id), `status(conversion_id: str) -> ConversionJob | None`. Firma del exportador inyectable: `ExportFn = Callable[[Path, Path, Callable[[str], None]], list[str]]` — `(src_dir, out_dir, on_component) -> componentes exportados`. Tasks 11/12 consumen `convert_from_hf`/`status`.

Riesgo documentado (spec, B sin mergear): el export corre en CPU/RAM (torch CPU); la única fase GPU es la validación funcional, que YA pasa por `device_semaphores` dentro de `validate_and_promote`. La zona ciega restante es RAM/CPU del export simultáneo a otros jobs — aceptada para el MVP, sin admisión por capacidad hasta que B aterrice.

- [ ] **Step 1: Escribir los tests que fallan (`tests/test_generation_converter.py`)**

Calcar helpers de `tests/test_generation_installer.py` (FakeHfClient que escribe `model_index.json` real, make_settings). Tests:

```python
def _pytorch_repo_files() -> list[HfFile]:
    return [
        HfFile(path="model_index.json", size=100),
        HfFile(path="unet/diffusion_pytorch_model.safetensors", size=1000),
        HfFile(path="unet/config.json", size=10),
        HfFile(path="vae/diffusion_pytorch_model.safetensors", size=500),
        HfFile(path="vae/config.json", size=10),
    ]


def test_select_conversion_files_keeps_torch_weights_in_declared_dirs() -> None:
    files = _pytorch_repo_files() + [
        HfFile(path="unet/duplicate.ckpt", size=999),
        HfFile(path="undeclared/x.safetensors", size=999),
    ]
    kept = _select_conversion_files(files, ["unet", "vae"])
    paths = {f.path for f in kept}
    assert "unet/diffusion_pytorch_model.safetensors" in paths
    assert "vae/config.json" in paths
    assert "unet/duplicate.ckpt" not in paths          # .ckpt siempre fuera
    assert "undeclared/x.safetensors" not in paths     # solo componentes declarados
    assert "model_index.json" not in paths             # se baja aparte, primero


def test_select_conversion_files_prefers_safetensors_over_bin_in_same_dir() -> None:
    files = [
        HfFile(path="unet/diffusion_pytorch_model.safetensors", size=1000),
        HfFile(path="unet/diffusion_pytorch_model.bin", size=1000),
    ]
    kept = _select_conversion_files(files, ["unet"])
    assert [f.path for f in kept] == ["unet/diffusion_pytorch_model.safetensors"]


# Helper a nivel de módulo del archivo de test: export fake que escribe un
# staging ONNX válido y reporta componentes. EXPORTED_LOG se limpia por test.
EXPORTED_LOG: list[str] = []


def fake_export_ok(src_dir: Path, out_dir: Path, on_component) -> list[str]:
    assert (src_dir / "model_index.json").exists()          # las fuentes ya están staged
    (out_dir / "unet").mkdir(parents=True)
    (out_dir / "model_index.json").write_text(
        json.dumps({"_class_name": "StableDiffusionXLPipeline", "unet": ["diffusers", "x"]}),
        encoding="utf-8",
    )
    for name in ("unet", "vae"):
        on_component(name)
        EXPORTED_LOG.append(name)
    return ["unet", "vae"]


def test_convert_happy_path_exports_and_promotes(tmp_path, monkeypatch) -> None:
    # La validación funcional real del installer se moquea a nivel de
    # _create_validation_pipeline (mismo patrón que test_generation_installer);
    # make_converter deja ese mock puesto por default.
    EXPORTED_LOG.clear()
    converter, installer, settings, registry = make_converter(tmp_path, export_fn=fake_export_ok)
    job_id = convert_and_drain(converter, "amd/sdxl-torch")
    job = converter.status(job_id)
    assert job.status == JobStatus.completed
    assert EXPORTED_LOG == ["unet", "vae"]
    assert registry.get(job.model_id) is not None
    # metadata de stages quedó completa
    assert job.metadata["progress"] == 1.0


def test_convert_export_failure_leaves_no_orphans(tmp_path) -> None:
    def failing_export(src_dir, out_dir, on_component):
        raise RuntimeError("export reventó a mitad del unet")

    converter, installer, settings, registry = make_converter(tmp_path, export_fn=failing_export)
    job_id = convert_and_drain(converter, "amd/sdxl-torch")
    job = converter.status(job_id)
    assert job.status == JobStatus.failed
    assert "export reventó" in job.error
    assert registry.get(_generation_model_id("amd/sdxl-torch")) is None
    leftovers = [p for p in settings.temp_path.iterdir()] if settings.temp_path.exists() else []
    assert leftovers == []                                       # staging borrado, sin huérfanos


def test_converted_result_goes_through_real_validation(tmp_path, monkeypatch) -> None:
    # Si la validación funcional del installer rechaza el resultado convertido,
    # el job falla -- la conversión NO tiene un camino de instalación paralelo.
    def explode(pipeline_dir):
        raise RuntimeError("pipeline inválido")

    EXPORTED_LOG.clear()
    converter, installer, settings, registry = make_converter(tmp_path, export_fn=fake_export_ok)
    monkeypatch.setattr(installer, "_create_validation_pipeline", explode)
    job_id = convert_and_drain(converter, "amd/sdxl-torch")
    job = converter.status(job_id)
    assert job.status == JobStatus.failed
    assert registry.get(_generation_model_id("amd/sdxl-torch")) is None
```

Helpers del archivo de test: `make_converter(tmp_path, export_fn)` construye Settings+registry+FakeHfClient(_pytorch_repo_files())+installer real (con `_create_validation_pipeline` moqueado a `FakeValidationPipeline` salvo que el test lo pise) y el converter; `convert_and_drain` calca `install_and_drain` (encola y drena con `_process_next`).

- [ ] **Step 2: Correr y verificar que fallan**

Run: `.venv\Scripts\python -m pytest tests/test_generation_converter.py -v`
Expected: FAIL con `ModuleNotFoundError`

- [ ] **Step 3: Implementar `app/services/generation_converter.py`**

```python
from __future__ import annotations

import asyncio
import contextlib
import shutil
from collections.abc import Callable
from pathlib import Path

from app.config import Settings
from app.models import ConversionJob, JobStatus, utc_now
from app.services.generation_installer import (
    GenerationModelInstaller,
    MODEL_INDEX_FILENAME,
    _ensure_model_index_listed,
    _ensure_size_cap,
    _generation_model_id,
    _read_declared_components,
    _safe_staging_dest,
)
from app.services.hf_client import HfClient, HfFile
from app.services.model_installer import _validate_repo_id
from app.services.progress import (
    advance_conversion_stage,
    complete_conversion_stages,
)

# La conversión descarga los pesos PyTorch que el installer normal EXCLUYE
# (.safetensors/.bin son la fuente del export). .ckpt/.msgpack/.h5 siguen
# fuera: son duplicados legacy de los mismos pesos. .onnx fuera: si el repo
# ya trae ONNX no debería estar en este camino.
CONVERSION_SKIP_SUFFIXES = (".ckpt", ".msgpack", ".h5", ".onnx", ".onnx_data", ".pb")
SAFETENSORS_SUFFIX = ".safetensors"
TORCH_BIN_SUFFIX = ".bin"


def _select_conversion_files(files: list[HfFile], declared: list[str]) -> list[HfFile]:
    dirs_with_safetensors = {
        f.path.rsplit("/", 1)[0]
        for f in files
        if "/" in f.path and f.path.lower().endswith(SAFETENSORS_SUFFIX)
    }
    kept: list[HfFile] = []
    for hf_file in files:
        lowered = hf_file.path.lower()
        if hf_file.path == MODEL_INDEX_FILENAME or lowered.endswith(CONVERSION_SKIP_SUFFIXES):
            continue
        if "/" not in hf_file.path:
            if lowered.endswith((".json", ".txt")):
                kept.append(hf_file)
            continue
        top_segment = hf_file.path.split("/", 1)[0]
        if top_segment not in declared:
            continue
        parent = hf_file.path.rsplit("/", 1)[0]
        if lowered.endswith(TORCH_BIN_SUFFIX) and parent in dirs_with_safetensors:
            continue  # duplicado .bin de un .safetensors en el mismo dir
        kept.append(hf_file)
    return kept


def _export_with_optimum(src_dir: Path, out_dir: Path, on_component: Callable[[str], None]) -> list[str]:
    # Import perezoso: torch/optimum nunca a nivel de módulo en app/services.
    # Invocación confirmada por el spike (docs/superpowers/specs/
    # 2026-07-25-third-party-spike-findings.md) -- si el findings doc registró
    # una forma distinta (export por componente para tener progreso real),
    # implementar ESA forma y mantener esta firma.
    from optimum.exporters.onnx import main_export

    on_component("pipeline")
    main_export(str(src_dir), output=str(out_dir), task="text-to-image")
    return ["pipeline"]


ExportFn = Callable[[Path, Path, Callable[[str], None]], list[str]]


class GenerationModelConverter:
    """Cola single-worker de conversión, paralela a GenerationModelInstaller.

    NO reusa model_converter.py (Spandrel: un solo conv-net, segundos) -- esto
    exporta un pipeline diffusers completo multi-componente (minutos, RAM
    alta). El resultado pasa por installer.validate_and_promote: MISMA
    validación funcional y promoción atómica que un install ONNX-nativo.
    """

    def __init__(
        self,
        settings: Settings,
        installer: GenerationModelInstaller,
        hf_client: HfClient,
        export_fn: ExportFn | None = None,
    ) -> None:
        self.settings = settings
        self.installer = installer
        self.hf_client = hf_client
        self.export_fn = export_fn or _export_with_optimum
        self._queue: asyncio.Queue[ConversionJob] = asyncio.Queue()
        self._jobs: dict[str, ConversionJob] = {}
        self._worker_task: asyncio.Task | None = None

    async def start(self) -> None:
        if self._worker_task is None:
            self._worker_task = asyncio.create_task(self._worker(), name="generation-convert-worker")

    async def stop(self) -> None:
        if self._worker_task is not None:
            self._worker_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._worker_task
            self._worker_task = None

    async def convert_from_hf(self, repo_id: str) -> str:
        validated = _validate_repo_id(repo_id)
        job = ConversionJob(repo_id=validated)
        self._jobs[job.id] = job
        await self._queue.put(job)
        return job.id

    def status(self, conversion_id: str) -> ConversionJob | None:
        return self._jobs.get(conversion_id)

    async def _worker(self) -> None:
        while True:
            job = await self._queue.get()
            await self._run_conversion(job)
            self._queue.task_done()

    async def _process_next(self) -> bool:
        try:
            job = self._queue.get_nowait()
        except asyncio.QueueEmpty:
            return False
        await self._run_conversion(job)
        self._queue.task_done()
        return True

    async def _run_conversion(self, job: ConversionJob) -> None:
        job.status = JobStatus.running
        job.started_at = utc_now()
        try:
            await self._convert_and_register(job)
            job.status = JobStatus.completed
        except Exception as exc:  # noqa: BLE001 - el job reporta cualquier fallo
            job.status = JobStatus.failed
            job.error = str(exc)
        finally:
            job.finished_at = utc_now()

    async def _convert_and_register(self, job: ConversionJob) -> None:
        files = await self.hf_client.repo_files(job.repo_id)
        _ensure_model_index_listed(files, job.repo_id)
        model_id = _generation_model_id(job.repo_id)
        src_root = self.settings.temp_path / f"genconv-src-{model_id}"
        out_root = self.settings.temp_path / f"genconv-onnx-{model_id}"
        for root in (src_root, out_root):
            if root.exists():
                shutil.rmtree(root, ignore_errors=True)
            root.mkdir(parents=True, exist_ok=True)

        max_file_bytes = self.settings.max_generation_model_download_mb * 1024 * 1024
        component_keys: list[str] = []
        try:
            advance_conversion_stage(job, component_keys, "downloading")
            await self.hf_client.download(
                job.repo_id,
                MODEL_INDEX_FILENAME,
                _safe_staging_dest(src_root, MODEL_INDEX_FILENAME),
                max_bytes=max_file_bytes,
            )
            declared = _read_declared_components(src_root)
            selected = _select_conversion_files(files, declared)
            _ensure_size_cap(selected, self.settings.max_generation_model_download_mb)
            for hf_file in selected:
                dest = _safe_staging_dest(src_root, hf_file.path)
                dest.parent.mkdir(parents=True, exist_ok=True)
                await self.hf_client.download(job.repo_id, hf_file.path, dest, max_bytes=max_file_bytes)

            def on_component(name: str) -> None:
                if name not in component_keys:
                    component_keys.append(name)
                advance_conversion_stage(job, component_keys, f"exporting:{name}")

            exported = await asyncio.to_thread(self.export_fn, src_root, out_root, on_component)
            advance_conversion_stage(job, exported, "validating")
            size_bytes = sum(f.size for f in selected)
            job.model_id = await self.installer.validate_and_promote(out_root, job.repo_id, size_bytes)
            complete_conversion_stages(job, exported)
        finally:
            for root in (src_root, out_root):
                if root.exists():
                    shutil.rmtree(root, ignore_errors=True)
```

Nota TDD: `on_component` corre en el thread del export — muta `job.metadata` (dict compartido); es el mismo patrón de callback cross-thread que ya usa `apply_generation_step_progress`, aceptado en este repo.

- [ ] **Step 4: Correr suite y verificar verde**

Run: `.venv\Scripts\python -m pytest tests/test_generation_converter.py tests/test_generation_installer.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add app/services/generation_converter.py tests/test_generation_converter.py
git commit -m "feat: servicio de conversión PyTorch→ONNX con progreso por componente y validación compartida"
```

---

### Task 11: Endpoints de conversión + wiring

**Files:**
- Modify: `app/api/routes.py`, `app/schemas.py`, `app/main.py`
- Test: `tests/test_generation_api.py`

**Interfaces:**
- Produces: `POST /api/v1/generation/models/convert` (body `{repoId}`, 202, respuesta `{conversionId, statusUrl}`) y `GET /api/v1/generation/models/convert/{conversion_id}` (respuesta con `conversionId, repoId, status, progress, stage, stages, modelId, error`). `app.state.generation_converter` disponible para los Depends.
- El frontend (Task 13) consume ambos.

- [ ] **Step 1: Escribir los tests que fallan (`tests/test_generation_api.py`)**

```python
def test_convert_endpoint_enqueues_conversion(client, app) -> None:
    class FakeConverter:
        async def convert_from_hf(self, repo_id: str) -> str:
            assert repo_id == "amd/sdxl-torch"
            return "conv123"

    app.state.generation_converter = FakeConverter()
    response = client.post("/api/v1/generation/models/convert", json={"repoId": "amd/sdxl-torch"})
    assert response.status_code == 202
    assert response.json() == {
        "conversionId": "conv123",
        "statusUrl": "/api/v1/generation/models/convert/conv123",
    }


def test_conversion_status_endpoint_returns_job(client, app) -> None:
    from app.models import ConversionJob, JobStatus

    job = ConversionJob(repo_id="amd/sdxl-torch")
    job.status = JobStatus.running
    job.metadata["progress"] = 0.4
    job.metadata["stage"] = "exporting:unet"
    job.metadata["stages"] = [{"key": "downloading", "label": "Downloading weights", "weight": 0.15, "status": "done"}]

    class FakeConverter:
        def status(self, conversion_id: str):
            return job if conversion_id == job.id else None

    app.state.generation_converter = FakeConverter()
    response = client.get(f"/api/v1/generation/models/convert/{job.id}")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "running"
    assert payload["progressPct"] == 40.0
    assert payload["stage"] == "exporting:unet"
    assert payload["stages"][0]["key"] == "downloading"


def test_conversion_status_unknown_id_is_404(client, app) -> None:
    class FakeConverter:
        def status(self, conversion_id: str):
            return None

    app.state.generation_converter = FakeConverter()
    assert client.get("/api/v1/generation/models/convert/nope").status_code == 404
```

- [ ] **Step 2: Correr y verificar que fallan**

Run: `.venv\Scripts\python -m pytest tests/test_generation_api.py -k convert -v`
Expected: FAIL (404 de ruta inexistente / KeyError de schema)

- [ ] **Step 3: Implementar**

`app/schemas.py`:

```python
class CreateConversionResponse(BaseModel):
    conversion_id: str = Field(serialization_alias="conversionId")
    status_url: str = Field(serialization_alias="statusUrl")


class ConversionStatusResponse(BaseModel):
    conversion_id: str = Field(serialization_alias="conversionId")
    repo_id: str = Field(serialization_alias="repoId")
    status: JobStatus
    progress_pct: float | None = Field(default=None, serialization_alias="progressPct")
    stage: str | None = None
    stages: list[dict[str, Any]] | None = None
    model_id: str | None = Field(default=None, serialization_alias="modelId")
    error: str | None = None
```

`app/api/routes.py` (junto a los endpoints de generación; dependency `get_generation_converter` calcada de `get_generation_installer`):

```python
def get_generation_converter(request: Request) -> GenerationModelConverter:
    return request.app.state.generation_converter


@router.post(
    "/generation/models/convert", response_model=CreateConversionResponse, status_code=202,
    dependencies=[Depends(require(Permission.models_install))],
)
async def convert_generation_model(
    payload: InstallModelRequest,
    converter: GenerationModelConverter = Depends(get_generation_converter),
) -> CreateConversionResponse:
    try:
        conversion_id = await converter.convert_from_hf(payload.repo_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return CreateConversionResponse(
        conversion_id=conversion_id,
        status_url=f"/api/v1/generation/models/convert/{conversion_id}",
    )


@router.get("/generation/models/convert/{conversion_id}", response_model=ConversionStatusResponse)
async def get_conversion_status(
    conversion_id: str, converter: GenerationModelConverter = Depends(get_generation_converter)
) -> ConversionStatusResponse:
    job = converter.status(conversion_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Conversion job not found")
    progress = job.metadata.get("progress")
    return ConversionStatusResponse(
        conversion_id=job.id,
        repo_id=job.repo_id,
        status=job.status,
        progress_pct=round(progress * 100, 1) if progress is not None else None,
        stage=job.metadata.get("stage"),
        stages=job.metadata.get("stages"),
        model_id=job.model_id,
        error=job.error,
    )
```

ORDEN DE RUTAS: FastAPI matchea `/generation/models/convert` (POST) y `/generation/models/convert/{id}` (GET) sin conflicto con `/generation/models/search` y `/generation/models/install/{id}` por método+literal — mantener los literales (`search`, `convert`, `install`) declarados ANTES de cualquier futura ruta con parámetro en ese segmento.

`app/main.py` (junto al wiring del installer, líneas ~127-136 y shutdown ~169):

```python
    generation_converter = GenerationModelConverter(settings, generation_installer, hf_client)
    await generation_converter.start()
    app.state.generation_converter = generation_converter
    # shutdown:
    await generation_converter.stop()
```

- [ ] **Step 4: Correr suite y verificar verde**

Run: `.venv\Scripts\python -m pytest tests/test_generation_api.py tests/test_spa_serving.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add app/api/routes.py app/schemas.py app/main.py tests/test_generation_api.py
git commit -m "feat: endpoints de conversión de modelos de generación y wiring del converter"
```

---

### Task 12: Auto-ruteo repo-no-ONNX → job de conversión

**Files:**
- Modify: `app/services/generation_installer.py` (`_download_and_register`, línea ~252), `app/services/model_installer.py` (`InstallJob`), `app/api/routes.py` + `app/schemas.py` (`InstallStatusResponse.conversion_id`), `app/main.py` (enlace)
- Test: `tests/test_generation_installer.py`, `tests/test_generation_api.py`

**Interfaces:**
- Produces: `InstallJob.conversion_id: str | None = None` (model_installer.py, compartido — inofensivo para el installer de upscalers); atributo `GenerationModelInstaller.enqueue_conversion: Callable[[str], Awaitable[str]] | None = None` (seteado en main.py a `generation_converter.convert_from_hf` — rompe la dependencia circular converter↔installer); `InstallStatusResponse.conversion_id` (alias `conversionId`).
- Heurística de detección (sobre el listado `files`, ANTES de crear staging o bajar nada): tiene `model_index.json` + NINGÚN archivo `.onnx` + al menos un `.safetensors`/`.bin` ⇒ layout diffusers PyTorch ⇒ encolar `ConversionJob` en vez del `ValueError` actual. El chequeo de `_class_name` conocido ocurre dentro del propio job de conversión (validación funcional), no acá — el listado de archivos no trae el contenido del JSON.

- [ ] **Step 1: Escribir los tests que fallan**

En `tests/test_generation_installer.py`:

```python
def test_install_routes_pytorch_only_repo_to_conversion(tmp_path: Path) -> None:
    files = [
        HfFile(path="model_index.json", size=100),
        HfFile(path="unet/diffusion_pytorch_model.safetensors", size=1000),
        HfFile(path="unet/config.json", size=10),
    ]
    installer, settings, registry = make_installer(tmp_path, files=files)
    enqueued: list[str] = []

    async def fake_enqueue(repo_id: str) -> str:
        enqueued.append(repo_id)
        return "conv456"

    installer.enqueue_conversion = fake_enqueue
    job = install_and_drain(installer, "amd/sdxl-torch")
    assert enqueued == ["amd/sdxl-torch"]
    assert job.status == InstallStatus.converting
    assert job.conversion_id == "conv456"
    assert job.error is None


def test_install_pytorch_only_repo_without_converter_keeps_actionable_error(tmp_path: Path) -> None:
    files = [
        HfFile(path="model_index.json", size=100),
        HfFile(path="unet/diffusion_pytorch_model.safetensors", size=1000),
    ]
    installer, settings, registry = make_installer(tmp_path, files=files)
    installer.enqueue_conversion = None
    job = install_and_drain(installer, "amd/sdxl-torch")
    assert job.status == InstallStatus.error
    assert "conversión" in job.error


def test_install_repo_with_onnx_files_never_routes_to_conversion(tmp_path: Path, monkeypatch) -> None:
    # El happy path existente (repo ONNX) no debe tocar enqueue_conversion:
    # mismo arranque que test_install_happy_path_registers_diffusion_model,
    # con el hook de conversión instrumentado.
    installer, settings, registry = make_installer(tmp_path)   # files ONNX default del helper
    monkeypatch.setattr(installer, "_create_validation_pipeline", lambda pipeline_dir: FakeValidationPipeline())
    called: list[str] = []

    async def fake_enqueue(repo_id: str) -> str:
        called.append(repo_id)
        return "nope"

    installer.enqueue_conversion = fake_enqueue
    job = install_and_drain(installer, "amd/onnx-model")
    assert job.status == InstallStatus.installed
    assert called == []
```

En `tests/test_generation_api.py`, extender el test de status de install para cubrir `conversionId`:

```python
def test_generation_install_status_exposes_conversion_id(client, app) -> None:
    from app.services.model_installer import InstallJob, InstallStatus

    job = InstallJob(id="i1", repo_id="amd/x", status=InstallStatus.converting, conversion_id="conv456")

    class FakeInstaller:
        def status(self, install_id):
            return job if install_id == "i1" else None

    app.state.generation_installer = FakeInstaller()
    payload = client.get("/api/v1/generation/models/install/i1").json()
    assert payload["status"] == "converting"
    assert payload["conversionId"] == "conv456"
```

- [ ] **Step 2: Correr y verificar que fallan**

Run: `.venv\Scripts\python -m pytest tests/test_generation_installer.py -k "conversion or routes" -v`
Expected: FAIL (`InstallJob` sin `conversion_id` / sin atributo `enqueue_conversion`)

- [ ] **Step 3: Implementar**

`model_installer.py` — `InstallJob` gana campo:

```python
@dataclass(slots=True, kw_only=True)
class InstallJob:
    id: str
    repo_id: str
    status: InstallStatus = InstallStatus.downloading
    progress_pct: float | None = None
    model_id: str | None = None
    error: str | None = None
    conversion_id: str | None = None
```

`generation_installer.py` — helper + rama al inicio de `_download_and_register` (después de `repo_files`, antes de `_ensure_model_index_listed`):

```python
def _has_onnx_payload(files: list[HfFile]) -> bool:
    return any(f.path.lower().endswith(".onnx") for f in files)


def _has_torch_weights(files: list[HfFile]) -> bool:
    return any(f.path.lower().endswith((".safetensors", ".bin")) for f in files)
```

En `__init__`: `self.enqueue_conversion: Callable[[str], Awaitable[str]] | None = None` (import `Awaitable`, `Callable` de `collections.abc`).

En `_download_and_register`:

```python
        files = await self.hf_client.repo_files(job.repo_id)
        _ensure_model_index_listed(files, job.repo_id)
        if not _has_onnx_payload(files) and _has_torch_weights(files):
            # Layout diffusers PyTorch (pesos sin export ONNX): en vez del
            # ValueError de "no parece un pipeline ONNX", se auto-rutea a un
            # job de conversión separado y visible (spec §3).
            if self.enqueue_conversion is None:
                raise ValueError(
                    f"El repo {job.repo_id!r} publica pesos PyTorch sin ONNX y la conversión no está disponible."
                )
            job.conversion_id = await self.enqueue_conversion(job.repo_id)
            job.status = InstallStatus.converting
            return
```

`routes.py` — los DOS builders de `InstallStatusResponse` (en `get_install_status` y `get_generation_install_status`) agregan `conversion_id=job.conversion_id`; `schemas.py`:

```python
class InstallStatusResponse(BaseModel):
    install_id: str = Field(serialization_alias="installId")
    repo_id: str = Field(serialization_alias="repoId")
    status: str
    progress_pct: float | None = Field(default=None, serialization_alias="progressPct")
    model_id: str | None = Field(default=None, serialization_alias="modelId")
    error: str | None = None
    conversion_id: str | None = Field(default=None, serialization_alias="conversionId")
```

`main.py` — después de construir ambos:

```python
    generation_installer.enqueue_conversion = generation_converter.convert_from_hf
```

NOTA semántica: para el installer de generación, `InstallStatus.converting` + `conversion_id` es estado TERMINAL del install job (el progreso real vive en el conversion job). El frontend (Task 13) hace el hand-off. `isTerminalInstallStatus` del frontend hoy probablemente NO considere `converting` terminal — Task 13 lo maneja explícitamente.

- [ ] **Step 4: Correr suite y verificar verde**

Run: `.venv\Scripts\python -m pytest tests/test_generation_installer.py tests/test_generation_api.py tests/test_model_installer.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add app/services/generation_installer.py app/services/model_installer.py app/api/routes.py app/schemas.py app/main.py tests/
git commit -m "feat: auto-ruteo de repos diffusers PyTorch-only a job de conversión visible"
```

---

### Task 13: Conversión visible en el frontend

**Files:**
- Modify: `frontend/src/lib/apiTypes.ts` (tipos `ConversionStatusResponse`, `conversionId` en `InstallStatusResponse`)
- Modify: `frontend/src/services/generation.ts` (`convertGenerationModel`, `getConversionStatus`)
- Modify: `frontend/src/hooks/useGenerationJob.ts` (`useGenerationModelInstall` sigue el hand-off a conversión)
- Modify: `frontend/src/modules/models/installUi.tsx` (progreso con etiqueta de etapa de conversión)
- Test: `frontend/src/hooks/useGenerationJob.test.tsx`, `frontend/src/modules/models/GenerationModelsSection.test.tsx`

**Interfaces:**
- Consumes: `GET /generation/models/install/{id}` con `conversionId` (Task 12), `GET /generation/models/convert/{id}` (Task 11).
- Produces: `useGenerationModelInstall` extendido — cuando el install status llega con `status === "converting"` y `conversionId`, arranca un segundo poll al endpoint de conversión; `phase` pasa a `"converting"`, `progressPct` y una nueva `stageLabel: string | null` salen del conversion job; al completar la conversión invalida `MODELS_QUERY_KEY` y `phase` termina en `"installed"`.

- [ ] **Step 1: Escribir los tests que fallan (`frontend/src/hooks/useGenerationJob.test.tsx`)**

```tsx
it("sigue el hand-off a conversión y termina en installed", async () => {
  installGenerationModel.mockResolvedValue({ installId: "i1", statusUrl: "/x" });
  getGenerationInstallStatus.mockResolvedValue({
    installId: "i1", repoId: "amd/x", status: "converting", progressPct: null,
    modelId: null, error: null, conversionId: "c1",
  });
  getConversionStatus
    .mockResolvedValueOnce({
      conversionId: "c1", repoId: "amd/x", status: "running", progressPct: 40,
      stage: "exporting:unet", stages: [], modelId: null, error: null,
    })
    .mockResolvedValue({
      conversionId: "c1", repoId: "amd/x", status: "completed", progressPct: 100,
      stage: "completed", stages: [], modelId: "gen--amd--x", error: null,
    });
  const { result } = renderHook(() => useGenerationModelInstall(10), { wrapper: createWrapper() });
  act(() => result.current.install("amd/x"));
  await waitFor(() => expect(result.current.phase).toBe("converting"));
  await waitFor(() => expect(result.current.phase).toBe("installed"));
  expect(result.current.modelId).toBe("gen--amd--x");
});

it("propaga el error del conversion job", async () => {
  installGenerationModel.mockResolvedValue({ installId: "i1", statusUrl: "/x" });
  getGenerationInstallStatus.mockResolvedValue({
    installId: "i1", repoId: "amd/x", status: "converting", progressPct: null,
    modelId: null, error: null, conversionId: "c1",
  });
  getConversionStatus.mockResolvedValue({
    conversionId: "c1", repoId: "amd/x", status: "failed", progressPct: null,
    stage: null, stages: [], modelId: null, error: "export reventó",
  });
  const { result } = renderHook(() => useGenerationModelInstall(10), { wrapper: createWrapper() });
  act(() => result.current.install("amd/x"));
  await waitFor(() => expect(result.current.phase).toBe("error"));
  expect(result.current.errorMessage).toContain("export reventó");
});
```

(Mocks hoisted de `../services/generation` como en los tests existentes del archivo.)

- [ ] **Step 2: Correr y verificar que fallan**

Run: `cd frontend && npx vitest run src/hooks/useGenerationJob.test.tsx`
Expected: FAIL (`getConversionStatus` no existe)

- [ ] **Step 3: Implementar**

`apiTypes.ts`: agregar `conversionId?: string | null` a `InstallStatusResponse` y:

```ts
export interface ConversionStatusResponse {
  conversionId: string;
  repoId: string;
  status: JobStatus;
  progressPct: number | null;
  stage: string | null;
  stages: Array<{ key: string; label: string; weight: number; status: string }> | null;
  modelId: string | null;
  error: string | null;
}
```

`services/generation.ts`:

```ts
export function convertGenerationModel(repoId: string): Promise<{ conversionId: string; statusUrl: string }> {
  return apiPostJson("/generation/models/convert", { repoId });
}

export function getConversionStatus(conversionId: string): Promise<ConversionStatusResponse> {
  return apiGet<ConversionStatusResponse>(`/generation/models/convert/${conversionId}`);
}
```

`useGenerationJob.ts` — dentro de `useGenerationModelInstall`:

```ts
const conversionId = statusQuery.data?.status === "converting" ? (statusQuery.data.conversionId ?? null) : null;

const conversionQuery = useQuery({
  queryKey: ["generation-model-conversion", conversionId],
  queryFn: () => getConversionStatus(conversionId as string),
  enabled: conversionId !== null,
  refetchInterval: (query) =>
    isTerminalJobStatus(query.state.data?.status ?? "queued") ? false : pollIntervalMs,
});
```

Resolución de fase (reemplaza el return actual): si hay `conversionId` → `phase` = `"converting"` mientras el conversion job no es terminal; conversion `completed` → `"installed"` (y el `useEffect` de invalidación usa también `conversionQuery.data?.modelId`); conversion `failed` → `"error"` con `conversionQuery.data.error`. `progressPct` = `conversionQuery.data?.progressPct ?? statusQuery.data?.progressPct ?? null`. Exponer `stageLabel: string | null` derivado de `stages` (label del stage `active`) — `null` fuera de conversión. Ampliar `ModelInstallPhase` si `"converting"` no está en el union (revisar `useModels.ts` / `installStatus.ts`; `isTerminalInstallStatus` NO debe tratar `converting` como terminal para que el poll del install no se corte antes de leer `conversionId` — pero una vez que `conversionId` existe, cortar el poll del install: `refetchInterval` del install query devuelve `false` cuando `data?.conversionId != null`).

`installUi.tsx` — `InstallProgress` acepta prop opcional `stageLabel?: string | null` y la muestra junto a la barra (`Converting — Exporting unet`). `GenerationModelsSection.tsx` y `GenerationHfSearch` le pasan el `stageLabel` del hook.

- [ ] **Step 4: Correr suite frontend completa y verificar verde**

Run: `cd frontend && npx vitest run`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add frontend/src
git commit -m "feat: progreso de conversión visible en la UI de instalación de modelos de generación"
```

---

### Task 14: Verificación final integrada

**Files:** ninguno nuevo — gate de cierre.

- [ ] **Step 1: Suite backend completa**

Run: `.venv\Scripts\python -m pytest -q`
Expected: PASS, cero regresiones

- [ ] **Step 2: Suite frontend completa + typecheck + build**

Run: `cd frontend && npx tsc --noEmit && npx vitest run && npm run build`
Expected: PASS

- [ ] **Step 3: Commit final si hubo ajustes**

```bash
git add -A && git commit -m "test: ajustes de integración final subproyecto A"
```

(Solo si los steps 1-2 requirieron cambios.)

---

## Smoke real (manual, post-plan — NO parte de las tasks, requiere GPU y red)

Documentado por el spec como validación manual, no CI:
1. Instalar un SDXL real de `amd/` de punta a punta desde el buscador nuevo.
2. Convertir un repo diffusers PyTorch-only real y generar una imagen con el resultado.
3. Probar un repo gated sin token (esperar el 403 accionable) y con token pegado desde Settings (instala OK).

## Notas de alcance (del spec, verificadas contra master)

- **SDXL Turbo**: misma clase que SDXL — no hay tarea propia. Sus defaults de inferencia (steps bajos, guidance ~0) se eligen por job: `steps`/`guidance` ya son parámetros del request de generación. Documentado en el comentario del mapa (Task 2).
- **SD3.5**: si el spike (Task 1) determina que `optimum-onnx==0.1.0` no expone la clase, se quita la entrada del mapa y se registra como fase futura en el findings doc — el resto del plan no cambia.
- **B sin mergear**: la conversión no tiene admisión por capacidad; su única fase GPU (validación funcional) sí pasa por `device_semaphores` al reusar `validate_and_promote`. Zona ciega restante: RAM/CPU del export — aceptada (Task 10).
- **C mergeado** (PR #2, verificado en origin/master): `GET/PATCH /settings` nacen gateados con `Permission.settings_read`/`settings_write`; `write_text_atomically` se importa de `app/services/json_store.py` (cero duplicación); `POST /generation/models/convert` gateado con `Permission.models_install` como sus vecinos. Bajo `AUTH_MODE=off` (default) todo esto es transparente.
