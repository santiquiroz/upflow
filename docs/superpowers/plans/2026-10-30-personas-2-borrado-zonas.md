# Personas 2/4: borrado automático, zonas y retoque (backend) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Un motor "Borrado automático (LaMa + OSOR)" que no carga SDXL salvo que una zona lo necesite, que reporta por zona el motor, la caja y si quedó una persona, que se puede forzar por zona, con ajuste de borde, corrector puntual y acceso desde MCP.

**Architecture:** `auto_erase` pasa a devolver zonas con caja y gana una guarda de personas inyectable y un motor forzado. Un modelo virtual `eraser-auto` corre por el camino de los borradores del `GenerationJobManager` (sin el pipeline SDXL), con SDXL perezoso solo como respaldo. La respuesta del trabajo serializa `eraseZones`. Un endpoint síncrono `/editor/heal` rellena trazos chicos con LaMa.

**Tech Stack:** Python 3.11, FastAPI, numpy/OpenCV, torch CPU (LaMa, OSOR), pytest; MCP con FastMCP.

**Spec:** `docs/superpowers/specs/2026-10-30-editor-personas-design.md` (secciones "Guarda de personas", "Zonas en la respuesta", "Borrado automático sin SDXL", "Retoque y agentes", criterio 3 y la parte backend del 6).

**Depende de:** Plan 1 (`PersonDetector`, `PersonBox`).

## Global Constraints

- Las del Plan 1 (licencias, `ep_registry`, estilo de código, comando de tests, sin push).
- Motores válidos para `eraseEngine`: exactamente `"lama" | "osor" | "lama+osor" | "sdxl"`.
- Nombres en las zonas: `engine` usa los valores que ya produce `auto_erase` (`"lama"`, `"osor"`, `"lama+osor"`, `"hibrido"`, `"sdxl"`); `guard` es `"face" | "person" | None`.
- `eraser-auto` se reserva en `"cpu"` como LaMa (OSOR corre en CPU float32).
- Con `eraseEngine` forzado la guarda no cambia de motor: solo marca `residualPerson`.

## Review Focus

- Trabajo `eraser-auto` sin OSOR y sin ningún SDXL instalado con una zona grande: se llena con LaMa y la zona dice `"lama"`; nunca 500 ni carga de SDXL.
- `eraseEngine: "sdxl"` sin modelo SDXL de relleno instalado: 400 al crear el trabajo con el motivo, no fallo a mitad del trabajo.
- Máscara vacía después de restar la protección: el trabajo termina con la imagen igual y `eraseZones` vacío.
- `/editor/heal` con un trazo más grande que 700 px de lado: 400 con "usá Quitar", sin correr LaMa.
- Trabajos viejos (sin `eraseAuto` en metadata, o con el formato previo sin `box`): `GET` sigue respondiendo; `eraseZones` es `null` o se arma sin caja (`box: null`).

---

### Task 1: Zonas con caja en `auto_erase`

**Files:**
- Modify: `app/services/erase_auto.py` (:61-100)
- Test: `tests/test_erase_auto.py`

**Interfaces:**
- Produces: cada `choice` de `auto_erase` suma `"index": int` (orden del grupo, desde 0) y `"box": [l, t, r, b]` (la caja de recorte que devuelve `edit_groups`). El resto de claves no cambia.

- [ ] **Step 1: Write the failing test**

```python
def test_choices_carry_group_index_and_crop_box():
    base, mask = scene_two_regions()            # helper existente o nuevo: dos manchas separadas
    _, choices = auto_erase(base, mask, None, fill_with(10), sdxl_never)
    assert [c["index"] for c in choices] == [0, 1]
    assert all(len(c["box"]) == 4 and c["box"][0] < c["box"][2] for c in choices)
```

- [ ] **Step 2: Run** `pytest tests/test_erase_auto.py -q` → FAIL.
- [ ] **Step 3: Implement** — iterar `for index, (box, group_mask) in enumerate(edit_groups(...))` y sumar las dos claves.
- [ ] **Step 4: Run** → PASS.
- [ ] **Step 5: Commit** `feat: auto_erase informa índice y caja de cada zona`

### Task 2: Guarda de personas y persona residual

**Files:**
- Create: `app/services/fill_people.py`
- Modify: `app/services/erase_auto.py`
- Test: `tests/test_fill_people.py`, `tests/test_erase_auto.py`

**Interfaces:**
- Consumes: `PersonDetector.detect(rgb, device) -> list[PersonBox]` (Plan 1).
- Produces:
  - `PeopleFinder = Callable[[np.ndarray, np.ndarray], list[tuple[int, int, int, int]]]` (rgb, máscara del grupo → cajas).
  - `def people_inside(boxes: list[PersonBox], mask: np.ndarray, min_score: float, inside_ratio: float) -> list[tuple[int, int, int, int]]` — cajas con puntaje ≥ `min_score` cuyo área cae ≥ `inside_ratio` dentro de la máscara.
  - `def make_people_finder(detector: PersonDetector, device: str = "cpu", min_score: float = PERSON_MIN_SCORE, inside_ratio: float = PERSON_INSIDE_RATIO) -> PeopleFinder` — detecta sobre el recorte de la caja de la máscara agrandada 50 % (no sobre la foto entera) y devuelve cajas en coordenadas de la foto. Constantes iniciales `PERSON_MIN_SCORE = 0.3`, `PERSON_INSIDE_RATIO = 0.6` (medido en la sesión: con esos valores marcó 2272 g0 y 2222 g2, que Haar no ve).
  - `auto_erase(..., find_people: PeopleFinder | None = None)`. En un grupo grande con OSOR: si `find_faces` o `find_people` encuentran algo → LaMa + OSOR con `"guard": "face"` o `"person"`; después de repetir se vuelve a mirar y, si sigue, `"residualPerson": True`. Sin hallazgos: `"guard": None, "residualPerson": False`. La clave vieja `"face_guard"` se reemplaza por `"guard"` (actualizar `tests/test_generation_inpaint_api.py:250` si la mira).

- [ ] **Step 1: Write the failing tests**

```python
def test_people_inside_keeps_boxes_mostly_in_mask():
    mask = np.zeros((100, 100), bool); mask[10:60, 10:60] = True
    boxes = [PersonBox((12, 12, 50, 58), 0.8), PersonBox((40, 40, 99, 99), 0.9), PersonBox((15, 15, 30, 30), 0.1)]
    assert people_inside(boxes, mask, min_score=0.3, inside_ratio=0.6) == [(12, 12, 50, 58)]

def test_person_guard_retries_with_lama_and_flags_residual():
    # find_people devuelve una caja las dos veces → engine "lama+osor", guard "person", residualPerson True
    ...
def test_no_finding_keeps_osor_and_no_residual(): ...
def test_face_guard_reports_guard_face(): ...
```

- [ ] **Step 2: Run** → FAIL.
- [ ] **Step 3: Implement** `fill_people.py` and the guard flow in `auto_erase` (one helper per branch so `auto_erase` stays flat).
- [ ] **Step 4: Run** `pytest tests/test_fill_people.py tests/test_erase_auto.py tests/test_generation_inpaint_api.py -q` → PASS.
- [ ] **Step 5: Commit** `feat: guarda de personas en el borrado automático`

### Task 3: Calibrar la guarda (criterio 3)

**Files:**
- Create: `scripts/measure_guard.py`
- Modify: `app/services/fill_people.py` (constantes)
- Uses: `C:/personal/upflow-eval/2026-10-30-personas/` (Plan 1, Task 0)

**Interfaces:**
- Produces: `python scripts/measure_guard.py --eval <carpeta> [--truth guard_truth.json]` — corre `faces_inside` + `make_people_finder` sobre `_editada.jpg` y `mask_i.png` de cada grupo `osor` de `out8/all8.log` y `out9/fix9.log`; imprime aciertos sobre los grupos con persona y falsas alarmas sobre el resto, y escribe una hoja de contactos de las falsas alarmas.

Grupos con persona que quedó (verdad inicial del spec): v8 2145 g0, 2234 g2, 2268 g1, 2274 g0, 2199 g2, 2187 g0; v9 2145 g1, 2268 g1, 2274 g0, 2272 g0, 2077 g0, 2222 g2, 2163 g2, 2177 g4. En la sesión el detector también marcó grupos fuera de esa lista que a ojo sí tenían una persona residual menor (2243 g4, 2186 g1, 2220 g0, 2163 g2 en v8, 2272 g0 en v8, 2173 g3): la verdad hay que revisarla.

- [ ] **Step 1: Write the script**; run it and produce the false-alarm sheet.
- [ ] **Step 2: Fijar la verdad** — el orquestador revisa la hoja y escribe `guard_truth.json` (`{"persona": ["8:2243g4", ...], "limpio": [...]}`) agregando las personas residuales reales que falten en la lista.
- [ ] **Step 3: Calibrar** `PERSON_MIN_SCORE` y `PERSON_INSIDE_RATIO` hasta: aciertos ≥ 12 de los 14 grupos de la lista inicial y ≤ 2 falsas alarmas sobre los grupos limpios según `guard_truth.json`.
- [ ] **Step 4: Run** `pytest tests/test_fill_people.py tests/test_erase_auto.py -q` → PASS.
- [ ] **Step 5: Commit** `feat: guarda de personas calibrada` con los números en el cuerpo.

### Task 4: Motor forzado (`eraseEngine`)

**Files:**
- Modify: `app/services/erase_auto.py`, `app/schemas.py` (`CreateGenerationJobRequest` :1320-1357), `app/services/engines/generation_onnx.py` (`GenerationRequest` :346-381), `app/services/generation_job_manager.py` (`create_job` :120-185), `app/api/routes.py` (`create_generation_job` :2830-2876)
- Test: `tests/test_erase_auto.py`, `tests/test_generation_inpaint_api.py`

**Interfaces:**
- Produces:
  - `CreateGenerationJobRequest.erase_engine: Literal["lama","osor","lama+osor","sdxl"] | None` (alias `eraseEngine`); viaja a `create_job(..., erase_engine=...)` y a `GenerationRequest.erase_engine`.
  - `auto_erase(..., engine: str | None = None)`: con valor, todos los grupos (chicos y grandes) usan ese motor; `"lama+osor"` = LaMa y OSOR encima; `"osor"` sin `osor_fill` → `ValueError`; `"sdxl"` usa `sdxl_edit` con strength 1.0. La guarda solo marca `residualPerson`.
  - `create_job` valida: `erase_engine` sin máscara → `ValueError`; `"osor"` sin paquete → `ValueError(missing_pack_message("osor"))`; `"sdxl"` sin ningún SDXL de relleno instalado → `ValueError` con motivo.

- [ ] **Step 1: Write the failing tests** — `test_forced_lama_applies_to_large_groups`, `test_forced_engine_never_switches_but_flags_residual`, `test_forced_osor_without_pack_is_400`, `test_forced_sdxl_without_inpaint_model_is_400`.
- [ ] **Step 2: Run** → FAIL.
- [ ] **Step 3: Implement**.
- [ ] **Step 4: Run** `pytest tests/test_erase_auto.py tests/test_generation_inpaint_api.py tests/test_generation_api.py -q` → PASS.
- [ ] **Step 5: Commit** `feat: motor de borrado forzado por trabajo`

### Task 5: Ajustar borde (`maskGrowPx`, `maskFeatherPx`)

**Files:**
- Modify: `app/services/inpaint_mask.py` (junto a `dilate_mask` :183 y `feather_mask` :200), `app/schemas.py`, `generation_job_manager.py`, `generation_onnx.py` (`GenerationRequest`)
- Test: `tests/test_inpaint_mask.py`, `tests/test_generation_inpaint_api.py`

**Interfaces:**
- Produces:
  - `def adjust_mask(mask: np.ndarray, grow_px: int, feather_px: int) -> np.ndarray` — `grow_px > 0` dilata, `< 0` erosiona (elipse de radio `abs(grow_px)`), `feather_px > 0` desenfoca el borde (gaussiano); uint8 0-255 a la salida.
  - Request: `mask_grow_px: int = Field(0, ge=-200, le=200, alias="maskGrowPx")`, `mask_feather_px: int = Field(0, ge=0, le=100, alias="maskFeatherPx")`; se aplican con `adjust_mask` a la máscara de borrado antes de `auto_erase` (camino `eraser-auto` del Task 6 y camino `eraseStrategy: "auto"` actual).

- [ ] **Step 1: Write the failing tests**

```python
def test_adjust_mask_grow_moves_bbox_by_n():
    mask = np.zeros((200, 200), np.uint8); mask[50:150, 50:150] = 255
    grown = adjust_mask(mask, 10, 0)
    assert mask_bbox(grown) == (40, 40, 160, 160)
    assert mask_bbox(adjust_mask(mask, -10, 0)) == (60, 60, 140, 140)

def test_auto_erase_job_receives_grown_mask(): ...  # FakeEngine registra la máscara recibida
```

- [ ] **Step 2: Run** → FAIL. **Step 3: Implement.** **Step 4: Run** → PASS.
- [ ] **Step 5: Commit** `feat: expandir, contraer y suavizar la máscara de borrado`

### Task 6: Modelo virtual `eraser-auto` sin SDXL

**Files:**
- Create: `app/services/engines/auto_eraser.py`
- Modify: `app/services/generation_job_manager.py` (ramas de LaMa en :153, :188-191, :302-307, :376-378, :390-392, :486-518), `app/api/routes.py` (`generation_capabilities` :3078-3140), `app/main.py` (:156-180), `app/services/engines/generation_onnx.py` (exponer una fábrica de `sdxl_edit` perezosa)
- Test: `tests/test_auto_eraser.py`, `tests/test_lama_eraser.py` (capabilities), `tests/test_generation_api.py`

**Interfaces:**
- Consumes: `LamaEraser.fill_group`, `OsorEraser.fill_group`/`available()`, `make_people_finder` (Task 2), `adjust_mask` (Task 5), `auto_erase` (Tasks 1, 2, 4).
- Produces:
  - `ERASER_AUTO_MODEL_ID = "eraser-auto"`, `ERASER_AUTO_MODEL_LABEL = "Borrado automático (LaMa + OSOR)"`.
  - `SdxlEditFactory = Callable[[], SdxlEdit]`.
  - `class AutoEraser: __init__(self, settings, lama_eraser, osor_eraser, people_finder_factory: Callable[[str], PeopleFinder | None], sdxl_edit_factory: Callable[[], SdxlEditFactory | None]); available() -> bool` (= LaMa instalado); `async def erase(self, base_image, mask_image, device: str, *, protect_image=None, engine: str | None = None, grow_px: int = 0, feather_px: int = 0) -> tuple[Image.Image, list[dict]]`. Reparto por zona grande: OSOR si está; si no, SDXL solo si `sdxl_edit_factory()` devuelve algo (se pide recién cuando una zona lo necesita); si no, LaMa. Nunca toca el pipeline SDXL si ninguna zona lo pide.
  - En el manager, `eraser-auto` sigue las ramas de LaMa (validación, inpaint, reserva `"cpu"`, `_run_eraser`) y guarda `job.metadata["eraseAuto"] = choices`.
  - `generation_capabilities` inserta `eraser-auto` primero (antes de LaMa y MI-GAN) cuando `settings.lama_available()`, con `supports_inpaint=True, erase_only=True`.
  - `GenerationEngine.sdxl_edit_for_installed_inpaint(device) -> SdxlEdit | None`: busca un modelo SDXL de relleno instalado (patrón `_entry_inpaint_only`, routes.py :2957) y devuelve un `sdxl_edit` que carga el pipeline en la primera llamada.

- [ ] **Step 1: Write the failing tests** — `test_auto_eraser_never_builds_sdxl_when_osor_handles_large`, `test_auto_eraser_without_osor_uses_sdxl_factory_once`, `test_auto_eraser_without_osor_or_sdxl_falls_back_to_lama`, `test_capabilities_list_eraser_auto_first_when_lama_installed`, `test_eraser_auto_job_records_zones_and_reserves_cpu`.
- [ ] **Step 2: Run** → FAIL. **Step 3: Implement.** **Step 4: Run** `pytest tests/test_auto_eraser.py tests/test_lama_eraser.py tests/test_generation_api.py tests/test_generation_inpaint_api.py -q` → PASS.
- [ ] **Step 5: Commit** `feat: modelo de borrado automático sin SDXL`

### Task 7: `eraseZones` en la respuesta del trabajo

**Files:**
- Modify: `app/schemas.py` (`GenerationJobResponse` :1408-1450), `app/api/routes.py` (`generation_job_to_response` :668-690)
- Test: `tests/test_generation_api.py`

**Interfaces:**
- Produces: `class EraseZone(BaseModel): index: int; box: list[int] | None; engine: str; guard: Literal["face","person"] | None = None; residual_person: bool = Field(False, serialization_alias="residualPerson")`; `GenerationJobResponse.erase_zones: list[EraseZone] | None = Field(None, serialization_alias="eraseZones")`; `def erase_zones_from_metadata(raw: list[dict] | None) -> list[EraseZone] | None` (acepta el formato viejo: `face_guard: True` → `guard: "face"`, sin `index` → posición, sin `box` → `None`).

- [ ] **Step 1: Write the failing tests** — `test_job_response_serializes_erase_zones_camel_case`, `test_old_metadata_without_box_still_serializes`, `test_job_without_erase_auto_has_null_zones`.
- [ ] **Step 2: Run** → FAIL. **Step 3: Implement.** **Step 4: Run** → PASS.
- [ ] **Step 5: Commit** `feat: la API devuelve las zonas del borrado`

### Task 8: Corrector puntual `POST /api/v1/editor/heal`

**Files:**
- Modify: `app/schemas.py`, `app/api/editor_routes.py`, `app/main.py` (si hace falta exponer `lama_eraser` en `app.state`)
- Test: `tests/test_editor_api.py`

**Interfaces:**
- Consumes: `LamaEraser.fill_group(canvas, group_mask, protect)`; `_resolve_init_image`; el guardado de imágenes iniciales que usa `upload_init_image` (mismo almacenamiento y formato de token).
- Produces: `EditorHealRequest(image_token alias imageToken, mask_token alias maskToken)`; respuesta `InitImageResponse` (`initImageToken, originalFilename, width, height`) de la imagen nueva. 400 si la máscara está vacía o su caja tiene lado > 700 (`"La zona es grande: usá Quitar."`); 503 si LaMa falta.

- [ ] **Step 1: Write the failing tests** — `test_heal_returns_new_token_with_filled_region` (FakeLama que pinta la zona de un color), `test_heal_rejects_large_stroke`, `test_heal_without_lama_is_503`.
- [ ] **Step 2: Run** → FAIL. **Step 3: Implement.** **Step 4: Run** → PASS.
- [ ] **Step 5: Commit** `feat: corrector puntual con LaMa en el Editor`

### Task 9: MCP: detectar personas y borrar con zonas

**Files:**
- Modify: `app/mcp/server.py` (junto a `upflow_segment_object` :1515-1560 y `upflow_generate_image` :1033-1095), `app/mcp/jobs.py` (`normalize_job` :51-82)
- Test: `tests/test_mcp_server.py`, `tests/test_mcp_jobs.py`

**Interfaces:**
- Produces:
  - Tool `upflow_detect_people(image_token: str, device: str = "") -> str` (READ_ONLY): llama `POST /api/v1/editor/people` y devuelve la lista.
  - `upflow_generate_image` suma `protect_mask_path: str = ""` (se sube como imagen y viaja como `protect_mask_token`), `erase_engine: str = ""`, `mask_grow_px: int = 0`, `mask_feather_px: int = 0`.
  - `normalize_job` para la familia `generation` incluye `eraseZones` si la respuesta lo trae.

- [ ] **Step 1: Write the failing tests** (patrón `install_mock` con `httpx.MockTransport`) — `test_detect_people_tool_posts_token`, `test_generate_with_protect_mask_uploads_and_sends_token`, `test_generation_job_normalized_keeps_erase_zones`.
- [ ] **Step 2: Run** → FAIL. **Step 3: Implement.** **Step 4: Run** `pytest tests/test_mcp_server.py tests/test_mcp_jobs.py -q` → PASS.
- [ ] **Step 5: Commit** `feat: MCP detecta personas y devuelve las zonas del borrado`
