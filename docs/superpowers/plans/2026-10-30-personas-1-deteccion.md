# Personas 1/4: detección de personas (backend) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upflow detecta a las personas de una foto, propone sujeto/fondo y devuelve la silueta de cada una por caja, medido contra los clics reales de las 60 fotos de grado.

**Architecture:** Un servicio `PersonDetector` (YOLOX-s ONNX por `ep_registry`) y funciones puras de rol y cara; un endpoint `POST /api/v1/editor/people` con caché por imagen; `POST /api/v1/editor/segment` acepta una caja además del punto (MobileSAM con labels 2/3). Un script de medición fija la verdad y calibra los umbrales.

**Tech Stack:** Python 3.11, FastAPI, onnxruntime 1.24 (DirectML/CPU), OpenCV 4.13, pytest (asyncio auto).

**Spec:** `docs/superpowers/specs/2026-10-30-editor-personas-design.md` (secciones "Detector de personas", "Endpoints", "Frente o fondo", "Criterios de aceptación" 1, 2 y 4).

## Global Constraints

- Upflow es MIT: solo modelos y código MIT/Apache/BSD. Detector: YOLOX-s, Apache-2.0, `https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.onnx`, sha256 `c5c2d13e59ae883e6af3b45daea64af4833a4951c92d116ec270d9ddbe998063`, 35858002 bytes, entrada `images` [1,3,640,640] float32 BGR 0-255 letterbox con relleno 114, salida `output` [1,8400,85] (cx, cy, w, h, objectness, 80 clases; persona = clase 0 = columna 5).
- Sesiones ONNX siempre por `ep_registry.create_session(str(path), device, settings, ...)`, con `_create_session` como costura de tests (patrón `EditorSegmenter`).
- Código: funciones atómicas, complejidad baja, sin docstrings, comentario de una línea solo para un porqué no obvio (en español, como el repo).
- Mensajes al usuario en español con el voseo de la app; errores de paquete faltante con `missing_pack_message("person-detector")`.
- Backend tests: `cd C:/wt/upflow-borrado && PYTHONNOUSERSITE=1 C:/Users/santi/.openclaw/workspace/image-upscaler-amd/.venv/Scripts/python.exe -m pytest <archivos> -q` (siempre desde el worktree: el `.pth` del venv apunta al repo principal).
- No hacer push ni merge. Commits en `feature/borrado-pulido`, sin línea Co-Authored-By, mensajes en español con prefijo `feat:`/`test:`/`fix:`.

## Review Focus

- Foto sin personas o con el paquete sin instalar: `/editor/people` responde 200 con lista vacía (sin personas) o 503 con el mensaje del paquete (sin modelo); nunca 500.
- Caja de persona que se sale de la imagen (YOLOX devuelve coordenadas negativas o > ancho, visto en la prueba: `[-2, 380, ...]`): se recorta al borde antes de devolverla y antes de pasarla a MobileSAM.
- Foto vertical 4000x6000 y foto chica (< 640 px de lado): el letterbox y la escala de vuelta dan cajas en coordenadas de la original.
- Pedido de segmentación con `box` y con `x,y` a la vez, o con caja invertida (x0 ≥ x1): 422 con mensaje claro, no excepción.
- Dos pedidos de `/editor/people` seguidos para la misma imagen: el segundo sale de la caché (el detector corre una sola vez).

---

### Task 0: Datos de medición en una carpeta propia

**Files:**
- Create: `C:/personal/upflow-eval/2026-10-30-personas/` (fuera del repo: fotos de personas reales)

- [ ] **Step 1: Copiar los datos de la sesión**

Copiar de `C:\Users\santi\AppData\Local\Temp\claude\c--personal-bipolar-code-bipolar-code\82a622fd-b78e-4d2c-b39d-f5c14ea7e944\scratchpad\thomas\` a `C:/personal/upflow-eval/2026-10-30-personas/`: `specs/all8.json`, `out8/all8.log`, `out9/fix9.log`, y de cada `out8/_DSC*/` y `out9/_DSC*/` solo `*_editada.jpg` y `mask_*.png` (mismas subcarpetas). Las fotos originales quedan en `C:\Users\santi\Desktop\imagenesThomas\`.
Copiar también el modelo ya descargado y verificado `...\scratchpad\dl-yolox\yolox_s.onnx` a
`C:/personal/upflow-eval/models/yolox_s.onnx` y usarlo con `PERSON_DETECTOR_MODEL=<esa ruta>` en las
corridas reales hasta que el Plan 4 lo instale. No escribir en `vendor/`: en el worktree es un
junction al repo principal.

- [ ] **Step 2: Verificar**

Run: `ls C:/personal/upflow-eval/2026-10-30-personas/out9 | wc -l` → 32 carpetas; `python -c "import json;print(len(json.load(open('C:/personal/upflow-eval/2026-10-30-personas/specs/all8.json'))))"` → 59.

### Task 1: Configuración del detector

**Files:**
- Modify: `app/config.py` (junto a `editor_segmenter_encoder`, ~:635-643 y propiedades ~:1334)
- Modify: `app/services/missing_pack.py` (`PACK_LABELS`)
- Modify: `tests/conftest.py` (`isolated_runtime_dir`, :24-60)
- Test: `tests/test_person_detector.py`

**Interfaces:**
- Produces: `Settings.person_detector_model: str` (alias `PERSON_DETECTOR_MODEL`, default `"vendor/yolox/yolox_s.onnx"`), `Settings.person_detector_model_path -> Path` (vía `resolve_against_project_root`), `Settings.person_detector_available() -> bool`; `PACK_LABELS["person-detector"] = "el modelo que encuentra personas"`.

- [ ] **Step 1: Write the failing tests**

```python
def test_person_detector_path_default_and_override(tmp_path):
    default = Settings(_env_file=None)
    assert default.person_detector_model_path.as_posix().endswith("vendor/yolox/yolox_s.onnx")
    model = tmp_path / "det.onnx"
    model.write_bytes(b"x")
    custom = Settings(_env_file=None, PERSON_DETECTOR_MODEL=str(model))
    assert custom.person_detector_model_path == model
    assert custom.person_detector_available() is True

def test_person_detector_unavailable_when_file_missing(tmp_path):
    assert Settings(_env_file=None, PERSON_DETECTOR_MODEL=str(tmp_path / "no.onnx")).person_detector_available() is False
```

- [ ] **Step 2: Run to verify they fail** — `pytest tests/test_person_detector.py -q` → FAIL (AttributeError).
- [ ] **Step 3: Implement** the field, path property and `available()` mirroring `migan_model` (:1300-1305); add the pack label; in `isolated_runtime_dir` set `PERSON_DETECTOR_MODEL` to `tmp_path / "detector-no-instalado.onnx"`.
- [ ] **Step 4: Run** `pytest tests/test_person_detector.py tests/test_missing_pack.py -q` → PASS.
- [ ] **Step 5: Commit** `feat: configuración del detector de personas`

### Task 2: `PersonDetector` (YOLOX-s)

**Files:**
- Create: `app/services/person_detector.py`
- Test: `tests/test_person_detector.py`

**Interfaces:**
- Produces:
  - `@dataclass(frozen=True) class PersonBox: box: tuple[int, int, int, int]; score: float`
  - `def letterbox(rgb: np.ndarray, side: int = 640) -> tuple[np.ndarray, float]` → tensor [1,3,side,side] float32 BGR, factor `r`.
  - `def decode_yolox(raw: np.ndarray, side: int = 640) -> np.ndarray` → [8400,85] con cx,cy,w,h en píxeles del letterbox (grillas de strides 8, 16, 32).
  - `def select_people(decoded: np.ndarray, ratio: float, image_size: tuple[int, int], threshold: float = 0.3, iou: float = 0.45) -> list[PersonBox]` → puntaje = objectness × clase 0, NMS (`cv2.dnn.NMSBoxes`), cajas divididas por `ratio`, recortadas a `image_size` (ancho, alto), ordenadas de izquierda a derecha.
  - `class PersonDetector: __init__(self, settings: Settings); available() -> bool; detect(self, rgb: np.ndarray, device: str = "cpu") -> list[PersonBox]; _create_session(self, device: str) -> Any` (sesión cacheada por device bajo un lock).
  - `class PersonDetectorUnavailableError(RuntimeError)` con `missing_pack_message("person-detector")`.

- [ ] **Step 1: Write the failing tests**

```python
def test_letterbox_pads_with_114_and_keeps_ratio():
    rgb = np.zeros((400, 800, 3), np.uint8)
    tensor, r = letterbox(rgb)
    assert tensor.shape == (1, 3, 640, 640) and r == 0.8
    assert tensor[0, :, 600, 0].tolist() == [114.0, 114.0, 114.0]

def test_decode_places_first_cell_at_stride_8():
    raw = np.zeros((1, 8400, 85), np.float32)
    decoded = decode_yolox(raw)
    assert decoded[0, :4].tolist() == [0.0, 0.0, 8.0, 8.0]
    assert decoded[6400, 2] == 16.0  # primera celda del stride 16

def test_select_people_scales_back_clips_and_sorts():
    decoded = np.zeros((3, 85), np.float32)
    decoded[0, :6] = [300, 200, 100, 200, 0.9, 0.9]   # persona, más a la derecha
    decoded[1, :6] = [10, 100, 60, 100, 0.9, 0.9]     # persona que se sale por la izquierda
    decoded[2, :5] = [100, 100, 50, 50, 0.9]; decoded[2, 6] = 0.9  # otra clase
    people = select_people(decoded, ratio=0.5, image_size=(1000, 800))
    assert [p.box for p in people] == [(0, 100, 80, 300), (500, 200, 700, 600)]

def test_detect_runs_session_once_per_call_and_reports_missing_pack(tmp_path):
    # FakeSession devuelve un tensor con una persona; _create_session monkeypatcheado.
    ...
    with pytest.raises(PersonDetectorUnavailableError, match="encuentra personas"):
        PersonDetector(Settings(_env_file=None, PERSON_DETECTOR_MODEL=str(tmp_path / "no.onnx"))).detect(rgb)
```

- [ ] **Step 2: Run to verify they fail** → FAIL (ImportError).
- [ ] **Step 3: Implement** the functions and class above; `detect` converts RGB→BGR, letterbox, `session.run(None, {"images": tensor})[0][0]`, `decode_yolox`, `select_people`.
- [ ] **Step 4: Run** `pytest tests/test_person_detector.py -q` → PASS.
- [ ] **Step 5: Commit** `feat: detector de personas YOLOX-s por ep_registry`

### Task 3: Rol (sujeto o fondo) y cara

**Files:**
- Create: `app/services/person_roles.py`
- Test: `tests/test_person_roles.py`

**Interfaces:**
- Consumes: `PersonBox` (Task 2).
- Produces:
  - `Role = Literal["subject", "background"]`
  - `def role_scores(rgb: np.ndarray, people: list[PersonBox]) -> list[float]` — combina altura relativa de la caja, nitidez (varianza del laplaciano en la caja / la de toda la foto en gris) y cercanía al centro horizontal; pesos como constantes `HEIGHT_WEIGHT`, `SHARPNESS_WEIGHT`, `CENTER_WEIGHT`.
  - `def classify_roles(rgb: np.ndarray, people: list[PersonBox]) -> list[Role]` — sujeto si su puntaje ≥ `SUBJECT_RELATIVE` × el máximo de la foto y ≥ `SUBJECT_MIN`; constantes iniciales `SUBJECT_RELATIVE = 0.6`, `SUBJECT_MIN = 0.35` (Task 6 las calibra).
  - `def face_box(rgb: np.ndarray, box: tuple[int, int, int, int]) -> tuple[int, int, int, int] | None` — Haar (`haarcascade_frontalface_default.xml`, el mismo de `fill_faces`) en la mitad superior de la caja; la más grande, en coordenadas de la foto.
  - `def anchor_point(box, face) -> tuple[int, int]` — centro de la cara o, sin cara, centro del cuarto superior de la caja (lo usa el Editor como punto de la entrada).

- [ ] **Step 1: Write the failing tests**

```python
def test_big_sharp_centered_person_is_subject_small_blurred_edge_is_background():
    rgb = scene_with(big_sharp=(800, 200, 1400, 1900), small_blur=(50, 700, 200, 1000))  # helper del test
    roles = classify_roles(rgb, [PersonBox((50, 700, 200, 1000), 0.8), PersonBox((800, 200, 1400, 1900), 0.9)])
    assert roles == ["background", "subject"]

def test_all_similar_people_are_all_subjects():  # grupo posado
    ...
    assert set(classify_roles(rgb, three_similar)) == {"subject"}

def test_anchor_point_uses_face_center_or_upper_quarter():
    assert anchor_point((0, 0, 100, 400), (20, 10, 60, 50)) == (40, 30)
    assert anchor_point((0, 0, 100, 400), None) == (50, 50)

def test_face_box_none_on_blank():
    assert face_box(np.zeros((400, 300, 3), np.uint8), (0, 0, 300, 400)) is None
```

- [ ] **Step 2: Run to verify they fail** → FAIL.
- [ ] **Step 3: Implement** the four functions.
- [ ] **Step 4: Run** `pytest tests/test_person_roles.py -q` → PASS.
- [ ] **Step 5: Commit** `feat: rol sujeto/fondo y cara por persona`

### Task 4: `POST /api/v1/editor/people` y capacidad

**Files:**
- Modify: `app/schemas.py` (junto a `EditorSegmentRequest`, :1471)
- Create: `app/services/editor_people.py`
- Modify: `app/api/editor_routes.py` (:22-61), `app/main.py` (:369-376)
- Test: `tests/test_editor_api.py`

**Interfaces:**
- Consumes: `PersonDetector`, `classify_roles`, `face_box` (Tasks 2-3); `_resolve_init_image(settings, token)` de `app/api/routes.py:2755`.
- Produces:
  - Schemas: `EditorPeopleRequest(image_token: str alias "imageToken", device: str | None = None)`; `EditorPerson(id: int, box: list[int], score: float, role: Literal["subject","background"], face: list[int] | None)`; `EditorPeopleResponse(people: list[EditorPerson])`.
  - `class EditorPeople: __init__(self, detector: PersonDetector); available() -> bool; async def people(self, image_path: Path, device: str) -> list[EditorPerson]` — caché LRU de 4 por `(f"{path}:{mtime_ns}", device)`; corre en `asyncio.to_thread`. `id` = índice en orden izquierda→derecha.
  - Route `POST /api/v1/editor/people` (permiso `jobs_create`, mismo manejo de device que `/segment`: default `"cpu"`, 400 si no es `cpu`/`dml:*`), 503 si el paquete falta, 500 legible si no.
  - `GET /api/v1/editor/capabilities` suma `"people": bool`.
  - `app.state.editor_people` creado en `main.py` junto a `editor_segmenter`.

- [ ] **Step 1: Write the failing tests** (patrón de `tests/test_editor_api.py`: llamar la corrutina con dobles)

```python
async def test_people_returns_roles_and_faces(tmp_path):
    token = await upload_token(tmp_path)
    resp = await editor_people_route(EditorPeopleRequest(imageToken=token), people=FakePeople([...]))
    assert resp.people[0].role == "subject" and resp.people[0].box == [10, 20, 110, 220]

async def test_people_unavailable_is_503_with_pack_hint(): ...
async def test_people_second_call_hits_cache(tmp_path):  # EditorPeople con FakeDetector que cuenta llamadas
    assert detector.calls == 1
async def test_capabilities_reports_people(): ...
```

- [ ] **Step 2: Run to verify they fail** → FAIL.
- [ ] **Step 3: Implement** schemas, `EditorPeople`, route, wiring.
- [ ] **Step 4: Run** `pytest tests/test_editor_api.py -q` → PASS.
- [ ] **Step 5: Commit** `feat: endpoint /editor/people con caché por imagen`

### Task 5: Segmentar por caja

**Files:**
- Modify: `app/services/editor_segmenter.py` (:57-67, :102-114)
- Modify: `app/schemas.py` (`EditorSegmentRequest`), `app/api/editor_routes.py` (:29-61)
- Test: `tests/test_editor_segmenter.py`, `tests/test_editor_api.py`

**Interfaces:**
- Produces:
  - `def build_box_decoder_inputs(embeddings: np.ndarray, box: tuple[float, float, float, float], scale: float, orig_h: int, orig_w: int) -> dict[str, np.ndarray]` — `point_coords` `[[[x0*s, y0*s], [x1*s, y1*s]]]`, `point_labels` `[[2.0, 3.0]]`, resto igual que `build_decoder_inputs`.
  - `async def segment_box(self, image_path: Path, box: tuple[float, float, float, float], device: str) -> bytes` en `EditorSegmenter` (caja recortada a la imagen).
  - `EditorSegmentRequest`: `x`, `y` pasan a `float | None = Field(default=None, ge=0)`; nuevo `box: list[float] | None = None`; `model_validator`: exactamente uno de (`x` y `y`) o `box`; `box` de 4 valores con x0 < x1 y y0 < y1. La route despacha a `segment` o `segment_box`.

- [ ] **Step 1: Write the failing tests**

```python
def test_build_box_decoder_inputs_uses_corner_labels():
    inputs = build_box_decoder_inputs(np.zeros((1, 256, 64, 64), np.float32), (10, 20, 110, 220), 0.5, 400, 300)
    assert inputs["point_coords"].tolist() == [[[5.0, 10.0], [55.0, 110.0]]]
    assert inputs["point_labels"].tolist() == [[2.0, 3.0]]

def test_segment_request_requires_point_or_box():
    with pytest.raises(ValidationError): EditorSegmentRequest(imageToken="ab", x=1, y=1, box=[0, 0, 5, 5])
    with pytest.raises(ValidationError): EditorSegmentRequest(imageToken="ab")
    with pytest.raises(ValidationError): EditorSegmentRequest(imageToken="ab", box=[5, 0, 1, 5])

async def test_segment_route_with_box_calls_segment_box(tmp_path): ...  # FakeSegmenter registra la caja
```

- [ ] **Step 2: Run to verify they fail** → FAIL.
- [ ] **Step 3: Implement** the decoder builder, `segment_box`, schema validator and dispatch.
- [ ] **Step 4: Run** `pytest tests/test_editor_segmenter.py tests/test_editor_api.py -q` → PASS (incluidos los tests viejos de punto).
- [ ] **Step 5: Commit** `feat: segmentar con MobileSAM por caja de persona`

### Task 6: Medir, fijar la verdad y calibrar (criterios 1, 2 y 4)

**Files:**
- Create: `scripts/measure_people.py` (herramienta de desarrollo; no la usa la app)
- Modify: `app/services/person_roles.py` (constantes calibradas)
- Create (fuera del repo): `C:/personal/upflow-eval/2026-10-30-personas/people_truth.json`, `contactos/`

**Interfaces:**
- Consumes: `PersonDetector`, `classify_roles`, `EditorSegmenter` (para la máscara de cada clic), datos de Task 0.
- Produces: `python scripts/measure_people.py --eval C:/personal/upflow-eval/2026-10-30-personas --photos C:/Users/santi/Desktop/imagenesThomas [--truth people_truth.json] [--sheets contactos]` imprime recall, acierto de rol, sujetos grandes propuestos como fondo (por foto) y tiempo por foto (p50/p95), y escribe hojas de contactos de los objetivos sin cubrir.

Regla de conteo (del spec): objetivo = máscara MobileSAM del clic; objetivos de una foto que se solapan > 50 % se funden; detectado si una caja cubre ≥ 50 % de la máscara; rol esperado = tipo de clic (`protect` → sujeto, `bg` → fondo), asignado a la caja más chica que cubre ≥ 50 %.

- [ ] **Step 1: Write the script** and run it once with `--sheets` to produce the contact sheets of uncovered objectives.
- [ ] **Step 2: Fijar la verdad** — revisar las hojas a ojo y escribir `people_truth.json` (`{"_DSC2145": {"not_person": [ids...]}, ...}`): los objetivos que no son personas (diploma, ramo, sombra) se descartan; el resto sin cubrir cuenta como persona perdida. Esta revisión la hace el orquestador (necesita mirar imágenes), no el implementador.
- [ ] **Step 3: Run** with `--truth` and record the baseline numbers.
- [ ] **Step 4: Calibrar** `SUBJECT_RELATIVE`, `SUBJECT_MIN` y los pesos hasta cumplir: recall ≥ 95 %, rol ≥ 90 %, sujetos grandes como fondo en ≤ 2 de 59 fotos, detección < 3 s p95 en `dml:0` y en `cpu` sobre fotos 6000x4000. Si el recall no llega con YOLOX-s a 640: probar primero mosaico 2x2 solo para cajas chicas; si sigue sin llegar, cambiar a RT-DETR (Apache-2.0) y avisar al dueño antes de seguir.
- [ ] **Step 5: Run** `pytest tests/test_person_roles.py tests/test_person_detector.py tests/test_editor_api.py tests/test_editor_segmenter.py -q` → PASS (ajustar fixtures si cambian las constantes).
- [ ] **Step 6: Commit** `feat: umbrales de sujeto/fondo calibrados con las 60 fotos` con los números medidos en el cuerpo del commit.
