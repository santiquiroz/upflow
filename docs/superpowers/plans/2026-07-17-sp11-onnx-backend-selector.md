# SP11 — Backend ONNX/DirectML optimizado + selector de Runtime

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development.

**Goal:** Un backend de upscale **ONNX Runtime** optimizado que en video supera a NCNN ~2.1x en AMD (DirectML) y hace rápido a NVIDIA (CUDA EP) — con un **selector de Runtime** (Auto por dispositivo / manual) estilo Lossless Scaling. Multi-vendor con un solo motor.

**Datos del benchmark (RX 7800 XT, animevideov3-x4, 720p→4x, medido):** NCNN 5.4 fps; ONNX/DML optimizado **11.49 fps (2.1x)**; inferencia sola 73ms (13.7 fps). Claves que lo hacen ganar: (a) **pre/post horneado en el grafo → I/O uint8** (mata numpy 147ms + readback 177MB), (b) **whole-frame SIN tiling** (tiling en DML = 1.26 fps, catastrofico), (c) **IO binding**, (d) **pipeline de threads load/infer/save**, (e) **PNG rapido (OpenCV, NO PIL** — PIL 3.15s/frame vs raw 62ms). Prototipo en `scratchpad/onnx_opt/` (`export_uint8.py`, `pipeline.py`, `onnx/anime-x4-uint8.onnx`).

**Rama:** `feature/onnx-backend-selector`. Base pytest 716 / vitest 336. Español convencional, sin Co-Authored-By.

---

## CONTRATO / arquitectura

### Backends (registro)
- `ncnn` (Real-ESRGAN NCNN Vulkan) — el actual, cualquier GPU. **Se queda como default seguro / fallback.**
- `onnx` (ONNX Runtime) multi-EP: `DmlExecutionProvider` (cualquier GPU DX12), `CUDAExecutionProvider`/TensorRT (NVIDIA, si el package está), `CPUExecutionProvider`. **El rápido, cuando el modelo está en ONNX.**

### Selección
- Setting `UPSCALE_BACKEND` = `auto` (default) | `ncnn` | `onnx`. Y por-job override desde la UI.
- **Auto**: si el modelo tiene versión ONNX y hay una GPU capaz → `onnx` (DmlEP en AMD/Intel, CUDA en NVIDIA); si no → `ncnn`. CPU → onnx-cpu o ncnn segun disponibilidad. Regla explicita y testeable.
- El selector NO cambia el MODELO (el user sigue eligiendo animevideov3-x4 etc.), solo el RUNTIME que lo corre.

### Modelos builtin en ONNX (uint8-in/out)
- Exportar los builtin a ONNX con **pre/post en el grafo** (input uint8 NHWC → /255 → NCHW → SRVGG → ×255 → clamp → round → uint8 → NHWC): animevideov3 x2/x3/x4, realesrgan-x4plus, realesrgan-x4plus-anime. Reusar `scratchpad/onnx_opt/export_uint8.py` (ya exporta animevideov3-x4; extender a los demas desde los .pth oficiales de xinntao/Real-ESRGAN). Vendored en `vendor/realesrgan-onnx/` (gitignored, script `download`/export). opset 17, fp32 (fp16 no ayuda).

### Motor ONNX de video (lo que da la velocidad)
- Inferencia **whole-frame** (sin tiling) cuando el frame entra en VRAM; tiling solo como fallback para frames enormes / GPU de poca VRAM (heuristica por tamaño+EP).
- **IO binding** (bind input/output OrtValue en device).
- **Frame I/O rapido con OpenCV** (`opencv-python-headless`, ~dep nueva): `cv2.imread`/`cv2.imwrite` PNG — mucho mas rapido que PIL (PIL era el cuello, 3.15s/frame). El intercambio sigue siendo PNG (compatible con RIFE + ffmpeg).
- **Pipeline de threads**: load(N+1) + infer(N) GPU + save(N-1) solapados. Sostiene ~2x NCNN.
- Session cache por (modelo, device/EP), como OnnxUpscaler.
- Reusar/extender `app/services/engines/onnx_upscaler.py`; el path de imagen single puede quedar con tiling+PIL (fps irrelevante ahi) pero idealmente tambien uint8-graph.

### Integracion en el pipeline de video
- `video_upscaler._upscale_frames`: elegir backend (auto/onnx/ncnn) segun el setting/job + disponibilidad del modelo ONNX + EP. Si onnx → nuevo path optimizado; si ncnn → el actual (con los threads 2:24:12 ya subidos). Preservar el watchdog de estancamiento, kill-on-cancel, task_done, todo.

## Tasks
1. **Export builtin→ONNX uint8** (script + modelos) + `download`/generación. (backend)
2. **Motor ONNX video optimizado** (uint8 graph, whole-frame, IO binding, OpenCV I/O, threaded pipeline, multi-EP) + registro de backends + Auto-selección + wiring en video_upscaler. TDD. (backend)
3. **Selector de Runtime en la UI** (dropdown Auto/NCNN/ONNX en Video e Image; por-job) + apiTypes + services. TDD vitest. (frontend)
4. **Smoke real en el pipeline completo** (episodio corto, onnx vs ncnn, medir fps real + verificar salida) + review adversarial (foco: no romper el pipeline existente, fallback correcto, watchdog/cancel intactos) + merge + release. (yo)

## Riesgos
- **PNG rapido es imprescindible**: si OpenCV no alcanza, para interpolación-OFF usar pipe rawvideo directo a ffmpeg (sin frames en disco). Con RIFE-ON hace falta PNG rapido si o si. Verificar OpenCV PNG < ~150ms/frame en el smoke.
- opencv-python-headless suma ~40MB al install — justificado por 2x. 
- CUDA/TensorRT EP para NVIDIA = otro package (onnxruntime-gpu); evaluar en Task 3/follow-up (el DmlEP ya cubre NVIDIA en Windows aunque no tan rapido como CUDA).

## Self-Review
Backend ONNX 2.1x medido ✓, uint8-graph + whole-frame + IO binding + OpenCV + threads ✓, selector Auto/manual ✓, NCNN fallback ✓, multi-vendor (DML/CUDA/CPU) ✓, pipeline/watchdog/cancel preservados ✓.
