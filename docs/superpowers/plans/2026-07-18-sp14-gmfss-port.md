# SP14 — Port GMFSS a ONNX any-GPU (port-gmfss-onnx) + motor de interpolación de calidad en Upflow

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** portar GMFSS_Fortuna (la mejor interpolación de frames para anime, MIT) a ONNX multi-EP para que corra
en cualquier GPU DirectX12 (AMD/Intel/NVIDIA) sin CUDA — **el primer port ONNX conocido de GMFSS** — publicarlo
como código libre para la comunidad, e integrarlo en Upflow como segundo motor de interpolación junto a RIFE
(RIFE = rápido, GMFSS = máxima calidad).

**Architecture:** mismo patrón validado en SP13 (AudioSR): descomponer el modelo en grafos ONNX (las redes) +
un driver Python/numpy (el pegamento no exportable), con parity numérica obligatoria por componente contra la
referencia PyTorch. El único op no portable es **softsplat** (forward warping): DirectML rechaza
`ScatterND(reduction='add')` (verificado en el código fuente de onnxruntime), así que vive en el driver —
primero CPU (correctitud), después kernel OpenCL (velocidad, el riesgo real del proyecto, aislado en su propia fase).

**Tech Stack:** PyTorch CPU (solo toolkit de export), ONNX opset 17, onnxruntime-directml, numpy, pyopencl
(fase 4), pytest. Runtime en Upflow: cero deps nuevas hasta fase 4 (pyopencl opcional).

## Global Constraints

- Carpeta local del port: `C:\personal\GMFSS_port` (git init ahí; NO en .openclaw/workspace).
- Repo público: `github.com/santiquiroz/port-gmfss-onnx` — MIT, README comunidad en inglés.
- Publicar SIEMPRE con `env -u GITHUB_TOKEN -u GH_TOKEN gh ...` (token de trabajo pisa keyring personal).
- Fuentes permitidas: `98mxr/GMFSS_Fortuna` (MIT) y `HolyWu/vs-gmfss_fortuna` (MIT, pesos commiteados ~75MB).
  **PROHIBIDO derivar de `sniklaus/softmax-splatting`** (academic-only, trampa legal) — el softsplat se porta
  desde `GMFSS_Fortuna/model/softsplat_torch.py` (MIT, PyTorch puro).
- Variante objetivo: **Fortuna "PG" (pg104)** — la de máxima calidad anime según SVFI model-spec.
- Parity gates por componente: rel-err máx < 1e-3 vs referencia PyTorch (mismo umbral que AudioSR);
  softsplat driver = bit-parity vs `softsplat_torch.py`.
- NUNCA aproximar softsplat con backward-warp (reintroduce el ghosting que softmax splatting evita).
- Benchmarks reales en la RX 7800 XT en cada fase; ninguna decisión de perf sin medición.
- Kill criterion (fase 4): si el splat GPU no logra ≥1 fps @1080p 2x end-to-end en la 7800 XT, GMFSS
  queda como modo "Max quality (slow)" documentado y RIFE sigue siendo el default — el port se publica igual.
- Commits en español, formato por capas del repo Upflow; sin Co-Authored-By.
- Tests Upflow: no romper 870 backend + 361 frontend; los módulos nuevos con tests propios.
- Contrato entre repos: los artefactos (`*.onnx` + `manifest.json`) se publican como release assets del repo
  del port; Upflow los descarga on-demand (`scripts/download-gmfss-onnx.ps1`), NO se bundlean en el installer.

---

## Contexto técnico (del research docs/RESEARCH_GMFSS_PORT.md — leerlo entero antes de empezar)

Pipeline GMFSS (de `vsgmfss_fortuna/GMFSS.py`): `FeatureNet(img0,img1)` → `GMFlow` (flujo a scale=0.5,
reescalado bilinear) → `MetricNet(imgs, flows)` → **softsplat ×~7** (imágenes full-res + pirámide de features
en 3 escalas, ambas direcciones, pesado por métrica Z) → `FusionNet` (GridNet) sintetiza el frame intermedio.
`reuse()` cachea flow/features por par de frames.

| Componente | Tamaño | ONNX | DML | Riesgo |
|---|---|---|---|---|
| FeatureNet | 3.3MB | ✅ trivial | ✅ | Bajo |
| MetricNet | 0.5MB | ✅ trivial | ✅ | Bajo |
| FusionNet (GridNet) | 31MB | ✅ trivial | ✅ | Bajo |
| IFNet (RIFE HDv3, variante union) | 21MB | ✅ (ports existen) | ✅ | Bajo |
| GMFlow | 19MB | ✅ shapes fijos, pad /64 (precedente: ptlflow) | ✅ (GridSample 4D opset16 tiene shader D3D12 propio) | Medio |
| softsplat | op | ✅ expresable (ScatterND-add) | ❌ cae a CPU EP | **Alto → driver** |

Perf de referencia (enhancr, 1080p 2x): GMFSS ≈ 4.3fps en A4000 / 16.4fps en 4090 (CUDA). Estimado 7800 XT
vía DML: **2-5 fps con splat OpenCL; 0.2-0.6 fps con splat CPU** (solo dev/parity). GMFSS es 10-15× más lento
que RIFE POR DISEÑO — es el trade-off de calidad, comunicarlo en la UI.

---

## Fase 0 — Repo + referencia dorada sin CUDA (S/M)

### Task 0.1: esqueleto del repo público

**Files:**
- Create: `C:\personal\GMFSS_port\{.gitignore, LICENSE, README.md, docs/, toolkit/, artifacts/, refs/}`

- [ ] `git init -b master` en `C:\personal\GMFSS_port`.
- [ ] `LICENSE` = MIT (copyright Santiago Quiroz + nota de que los pesos/arquitectura son de 98mxr/GMFSS_Fortuna, MIT).
- [ ] `.gitignore`: `.venv/`, `artifacts/*.onnx`, `refs/*.npy`, `refs/golden/`, `__pycache__/`, `dist/`, `*.pth`, `*.pkl`.
- [ ] `README.md` inicial (ver Task 6.1 para el contenido final; arrancar con la sección de motivación y el
      status table vacío — el README crece con cada fase, con números medidos, nunca prometidos).
- [ ] Commit inicial en español. Publicar YA como repo público (mismo flujo que port-audiosr-onnx):
      `env -u GITHUB_TOKEN -u GH_TOKEN gh repo create port-gmfss-onnx --public --source . --push --description "First known ONNX port of GMFSS — the best anime frame interpolation, running on any DX12 GPU (AMD/Intel/NVIDIA), no CUDA required"`
- [ ] Verificar: `git remote -v` apunta a santiquiroz (NO a la cuenta de trabajo).

### Task 0.2: env del toolkit + referencia dorada corriendo en CPU

**Files:**
- Create: `toolkit/requirements.txt`, `toolkit/setup-env.ps1`, `toolkit/run_reference.py`

**Interfaces:**
- Produces: `refs/golden/` con tensores por componente y `refs/golden/interp_ref.png` (frame interpolado de
  referencia) — TODO el resto del proyecto se valida contra esto.

- [ ] Venv py3.11/3.12 con: torch CPU (moderno, sin pins legacy — GMFSS no es audiosr), numpy, onnx,
      onnxruntime-directml, opencv-python-headless, pytest. SIN cupy.
- [ ] Vendorear el código de inferencia: clonar `HolyWu/vs-gmfss_fortuna` (composición limpia + pesos
      commiteados: flownet 18.9MB, fusionnet 31.4MB, feat 3.3MB, metric 0.5MB, rife 21MB) y
      `98mxr/GMFSS_Fortuna` (por `model/softsplat_torch.py`). Documentar commits exactos vendoreados.
- [ ] `run_reference.py`: correr GMFSS_Fortuna PG en CPU **con softsplat_torch (sin cupy)** sobre un par de
      frames de anime 1920×1080 fijos (incluir 2-3 pares de prueba en `refs/inputs/`, frames propios o de
      clips CC), t=0.5. Instrumentar con hooks para dumpear a `refs/golden/`:
      entrada normalizada, features (3 escalas), flow01/flow10, metric Z, cada salida de softsplat (las ~7),
      salida de FusionNet, frame final. Guardar shapes en `refs/golden/meta.json`.
- [ ] **Verificación de fase**: `run_reference.py` reproduce un frame interpolado visualmente correcto
      (inspección + SSIM > 0.9 entre el frame generado y el frame real intermedio del clip de prueba).
      Medir y anotar el tiempo CPU por frame (baseline).
- [ ] Commit + push.

**Gotcha esperado**: softsplat_torch.py tiene control-flow no exportable (`if not finite_mask.any()`) — para la
REFERENCIA da igual (corre eager); solo importa en fase 3.

---

## Fase 1 — Export de las redes fáciles + parity (S/M)

### Task 1.1: exportar FeatureNet, MetricNet, FusionNet (e IFNet si la variante union la usa)

**Files:**
- Create: `toolkit/export_components.py`, `toolkit/validate_ort.py`
- Produce: `artifacts/{featurenet,metricnet,fusionnet}.onnx` + `<n>_in*.npy`/`<n>_ref.npy`

- [ ] Patrón EXACTO de port-audiosr-onnx (copiar la estructura de su `export_components.py` y
      `validate_ort.py` — ya resolvieron save_pair/parity/timing): wrapper `torch.nn.Module` por grafo si la
      firma tiene kwargs, export opset 17 con `dynamo=False` primero (legacy JIT); si un grafo tripea el
      tracer ("invalid unordered_map key" u otro), reintentar `dynamo=True` opset 18 — **la parity decide, no
      el exporter** (lección SP13: UNet solo salió por dynamo, vocoder solo por legacy).
- [ ] Shapes FIJOS por resolución objetivo: exportar a 1920×1088 (1080p pad /64). La resolución se hornea; el
      runtime elegirá el ONNX que matchee o padear/croppear (decisión en Task 5.2).
- [ ] `validate_ort.py`: cada grafo en CPU-EP y DirectML vs sus refs de fase 0 — rel-err máx < 1e-3, tabla de
      timing CPU vs DML (como la de AudioSR). Los inputs de parity son los tensores REALES de fase 0, no randn.
- [ ] Commit + push + actualizar tabla de status del README con los números medidos.

### Task 1.2: exportar GMFlow (el riesgo medio)

**Files:**
- Modify: `toolkit/export_components.py` (añadir grafo `gmflow`)

- [ ] Export a shapes fijos (input par de imágenes a scale=0.5 de 1080p → 960×544 pad /64).
      Puntos de fricción conocidos y sus salidas: `F.unfold` (se descompone en ops estándar en el export),
      rolls de shifted-window attention (exportan como Slice/Concat), InstanceNorm2d (op nativo).
- [ ] Validar en DML: GridSample 4D está soportado (shader propio) — si algún op NO mapea en DML, partir
      GMFlow en sub-grafos (backbone / transformer / refine) con glue numpy, exactamente el fallback que
      AudioSR tenía para el UNet.
- [ ] Parity vs flow de fase 0 (rel-err < 1e-3; el flow es sensible — si el max-err lo dominan outliers
      aislados, reportar también RMS como en validate_driver de AudioSR y decidir con los dos).
- [ ] Medir DML vs CPU. Commit + push + README.

**Verificación de fase 1**: los 4-5 grafos corren en DirectML en la 7800 XT con parity verde y timing tabulado.

---

## Fase 2 — Driver Python: softsplat portable + pipeline completo (M/L)

### Task 2.1: softsplat en el driver (correctitud primero)

**Files:**
- Create: `driver/softsplat.py` (en el repo del port; el paquete `driver/` es puro: numpy + torch-CPU opcional,
  CERO imports de upflow — igual que `app/services/engines/audiosr/` que se importa desde afuera)
- Test: `tests/test_softsplat.py`

**Interfaces:**
- Produces: `splat_softmax(tenIn: np.ndarray[N,C,H,W], tenFlow: np.ndarray[N,2,H,W], tenMetric: np.ndarray[N,1,H,W]) -> np.ndarray[N,C,H,W]`

- [ ] Port 1:1 de `softsplat_torch.py` (MIT) a numpy/torch-CPU: pesos bilineales de las 4 esquinas +
      `index_add_` sobre índices NHW aplanados (torch-CPU) o `np.bincount` por bloques de canal (numpy).
      Reescribir las máscaras booleanas como clamp-de-índices + peso-cero (sin control flow dependiente de datos).
- [ ] Test de **bit-parity** contra softsplat_torch.py sobre los tensores reales de fase 0 (las ~7 llamadas
      del pipeline) + casos sintéticos (flujo cero = identidad; flujo entero = shift exacto; colisiones).
- [ ] Commit.

### Task 2.2: driver end-to-end (grafos ONNX + splat CPU) con parity total

**Files:**
- Create: `driver/pipeline.py`, `driver/assets.py` (manifest), `toolkit/validate_driver.py`
- Test: `tests/test_pipeline.py`

**Interfaces:**
- Produces: `GmfssDriver(assets, run_graph).interpolate_pair(img0, img1, timesteps: list[float]) -> list[np.ndarray]`
  y `GmfssDriver.reuse(img0, img1)` (cache de flow/features por par, como el original — SIN recalcular por timestep).

- [ ] Ensamblar: FeatureNet → GMFlow (scale 0.5, resize bilinear del flow a full-res ×2 del valor) →
      MetricNet → softsplat×7 → FusionNet. Replicar `GMFSS.py` de HolyWu línea a línea (es la composición
      de referencia); `run_graph` inyectado como en AudioSrDriver.
- [ ] `manifest.json`: shapes por grafo, resoluciones soportadas, orden/semántica de las llamadas a splat,
      escala del flow, normalización de entrada (rangos), versión de pesos (pg104), licencia y créditos.
- [ ] `validate_driver.py` (el gate obligatorio, espejo del de AudioSR): compara CADA etapa contra
      `refs/golden/` + el frame final (rel-err < 1e-3 por etapa; frame final además SSIM > 0.99 vs referencia).
- [ ] Medir fps end-to-end @1080p con splat CPU (esperado 0.2-0.6 fps — es el modo parity, documentarlo así).
- [ ] Commit + push + README (tabla: "pipeline completo validado, splat CPU").

**Verificación de fase 2**: `validate_driver.py` = PARITY OK todas las etapas en CPU-EP y DML (grafos en DML,
splat en CPU).

---

## Fase 3 — Splat rápido: kernel OpenCL (el riesgo del proyecto, aislado) (M/L)

### Task 3.1: kernel scatter-add OpenCL

**Files:**
- Create: `driver/softsplat_cl.py` + `driver/kernels/splat.cl`
- Test: `tests/test_softsplat_cl.py`

- [ ] Kernel OpenCL (~50 líneas): scatter-add bilineal con atomics float (atomic_cmpxchg loop o
      cl_khr_int64_base_atomics según disponibilidad en el driver AMD). pyopencl como dep OPCIONAL
      (extra `[gpu-splat]` en el port; en Upflow se gatea por disponibilidad como todo lo demás).
- [ ] Bit-comparación (tolerancia float por orden de acumulación: rel-err < 1e-5) vs `softsplat.py` CPU sobre
      los tensores de fase 0.
- [ ] Bench aislado del kernel en la 7800 XT: objetivo <20ms por llamada @1080p (7 llamadas/frame → <140ms
      de splat por frame).
- [ ] Fallback automático: si pyopencl no está o el kernel falla al compilar → splat CPU con warning una vez.
- [ ] Alternativa B si OpenCL decepciona (documentada, no implementar de una): grafo ONNX con ScatterND-add —
      en CUDA EP corre en GPU (regalo para NVIDIA), en DML cae a CPU EP (= statu quo).

### Task 3.2: bench end-to-end + decisión

- [ ] fps @1080p 2x end-to-end con splat OpenCL + grafos DML, fp32 y fp16 (convertir grafos con
      onnxconverter-common; si el converter se ahoga como con AudioSR, medir solo fp32 y documentar).
- [ ] **GATE (kill criterion)**: ≥1 fps @1080p → GO integración con selector normal.
      <1 fps → GO igual pero UI lo marca "Max quality (very slow)" y el README documenta los números reales.
- [ ] Publicar release `models-v1.0` del repo del port con los `.onnx` + manifest (assets, como AudioSR).
- [ ] Commit + push + README con la tabla final de perf.

---

## Fase 4 — Integración en Upflow (M)

### Task 4.1: motor GmfssEngine con el MISMO contrato que RifeNcnnEngine

**Files:**
- Create: `app/services/engines/gmfss/` (paquete puro copiado del repo del port: `softsplat.py`,
  `softsplat_cl.py`, `pipeline.py`, `assets.py` — vendored como código, con nota de sync y versión) y
  `app/services/engines/gmfss_engine.py` (integración)
- Test: `tests/test_gmfss_engine.py`

**Interfaces:**
- Produces: `GmfssEngine.run(frames_in: Path, frames_out: Path, source_frame_count: int, multiplier: int = 1, *, target_frame_count: int | None = None, device: str | None = None) -> Path`
  — firma IDÉNTICA a `RifeNcnnEngine.run` → drop-in en `_maybe_interpolate` sin tocar el pipeline.

- [ ] `GmfssEngine`: sesiones ONNX por device con cache LRU(1) (patrón AudioSrRestorer), pools de threads
      para decode/encode PNG (lección SP11/RIFE: el I/O mata el throughput — reusar el patrón load/save
      threads de `onnx_video_upscaler.py`), `reuse()` por par consecutivo, timesteps desde multiplier o
      target_frame_count (misma aritmética que RIFE: `_resolve_target_frame_count`).
- [ ] `available()` = `settings.gmfss_available()` (flag + `GmfssAssets.is_complete(dir)` con
      `required_files` del manifest — patrón exacto de `AudioSrAssets`).
- [ ] Cancel cooperativo: chequear un `threading.Event` entre pares de frames (lección del review SP13:
      esperar el thread antes de re-lanzar CancelledError — copiar el patrón shield+await de AudioSrRestorer).
- [ ] Tests con grafos fake (patrón test_audiosr_driver): conteo de frames exacto, timesteps correctos para
      2x/3x/target_fps, cancel, engine no disponible → RuntimeError accionable.

### Task 4.2: selector de motor de interpolación (config + API + UI)

**Files:**
- Modify: `app/config.py` (`INTERP_ENGINES = frozenset({"rife", "gmfss"})`, `enable_gmfss: bool`,
  `gmfss_model_dir: str`, `interp_engine_available(engine)`), `app/services/video_upscaler.py`
  (`_maybe_interpolate` elige motor por `job.interp_engine`), `app/services/video_job_manager.py`
  (validación por modo, mensajes disabled vs not-installed — patrón `validate_restore_mode_ready`),
  `app/api/routes.py` + `app/schemas.py` (campo `interp_engine`, capabilities `interpEngines`),
  `app/main.py` (construcción).
- Create: `scripts/download-gmfss-onnx.ps1` (assets del release del port, patrón download-audiosr-onnx.ps1)
- Modify frontend: `FpsBoostControls`/`VideoPanel` — selector RIFE/GMFSS visible solo si hay >1 motor
  disponible, con hint de costo ("GMFSS: máxima calidad para anime, ~10× más lento que RIFE").
- Test: espejo de los tests de restore modes (backend) + tests de panel (frontend).

- [ ] Default SIEMPRE `rife` (GMFSS es opt-in por job). `interp_engine=gmfss` requiere ENABLE_GMFSS + modelos.
- [ ] `.env.example` documentado (flag, dir, nota de costo, TDR n/a — los grafos GMFSS son chicos).
- [ ] Suites completas verdes (870+ backend, 361+ frontend).

### Task 4.3: smoke real + review adversarial + release

- [ ] Smoke e2e en la 7800 XT: job real 720p→2x con `interp_engine=gmfss` vía API → completed, frames
      exactos, fps correcto; comparación visual RIFE vs GMFSS sobre el mismo clip de anime (guardar ambos
      outputs para evaluación por oído/ojo del usuario).
- [ ] Review adversarial (code-reviewer) del código nuevo; arreglar CRITICAL/HIGH con tests de regresión.
- [ ] Bump versión Upflow + merge master + push + package + `gh release` + actualizar install local
      (CHEQUEAR jobs activos antes de tocar el server; relanzar con -WorkingDirectory).
- [ ] Actualizar memoria del proyecto.

---

## Fase 5 (v2, NO en este plan — solo dejar documentado)

- **DRBA** (MIT): adjuster de timing para anime (doses/treses) sobre el flow — se apila sobre RIFE o GMFSS.
- Dedup de frames duplicados (30-50% en anime) — beneficia a los DOS motores.
- Fusión GMFSS↔raw-pipe (frames en memoria sin PNG intermedio).

---

## Task 6.1: README comunidad del repo del port (crece por fase; contenido final)

**Files:** `C:\personal\GMFSS_port\README.md`

Estructura obligatoria (espejo del README de port-audiosr-onnx, que ya validamos como formato):

- [ ] **Título + tagline**: "The first known ONNX port of GMFSS — state-of-the-art anime frame interpolation
      on any DirectX 12 GPU, not just NVIDIA."
- [ ] Badges: MIT, ONNX opset, DirectML (AMD | Intel | NVIDIA), Python.
- [ ] **Why this exists (la motivación pedida)**: GMFSS_Fortuna es el techo de calidad en interpolación de
      anime, pero está encadenado a CUDA/cupy (softsplat es un kernel CUDA); los ports TensorRT/NCNN
      existentes TAMBIÉN dejan el splat en CUDA. Quien tiene AMD/Intel solo puede usar RIFE. Este port
      descompone GMFSS en grafos ONNX + un driver con softsplat portable (CPU/OpenCL) para democratizar la
      mejor interpolación de anime sin lock-in de vendor — mismo motivo y patrón que nuestro port de AudioSR.
- [ ] **Tabla de resultados medidos** (RX 7800 XT, por grafo: CPU-EP vs DML, rel-err) + fps end-to-end
      (splat CPU vs OpenCL) — solo números reales, actualizada por fase.
- [ ] **Diagrama mermaid** de la descomposición (grafos ONNX vs driver, dónde vive el splat y por qué:
      cita al DmlOperatorScatter.cpp de onnxruntime que rechaza reduction).
- [ ] **How it works**: pipeline, reuse(), shapes fijos, timesteps.
- [ ] **Usage**: setup-env, export, validate, snippet mínimo del driver standalone (sin Upflow), link a la
      integración de Upflow como ejemplo de runtime.
- [ ] **Status table** por componente (export/parity/DML) — igual que AudioSR.
- [ ] **Créditos y licencias**: 98mxr/GMFSS_Fortuna (MIT — modelo y softsplat_torch), HolyWu/vs-gmfss_fortuna
      (MIT — composición + pesos), GMFlow (Apache-2.0), paper de softmax splatting (Niklaus & Liu, citar el
      paper, NO derivar del repo academic-only). Sibling ports: port-audiosr-onnx.
- [ ] **Contributing**: pedir benchmarks en otras GPUs (Arc, RDNA2/4, Ampere), mejoras del kernel, ports del
      driver a otros lenguajes.

---

## Riesgos y fallbacks

| Riesgo | Prob. | Mitigación |
|---|---|---|
| GMFlow no exporta limpio (unfold/rolls/InstanceNorm) | Media | Sub-grafos + glue numpy (fallback AudioSR); precedente ptlflow dice que sale |
| Algún op de GMFlow no corre en DML | Baja-media | Ese sub-grafo a CPU-EP (chico) o shader alternativo; medir impacto |
| Kernel OpenCL lento o atomics float no disponibles | Media | Alternativa B (ScatterND-add: GPU en CUDA, CPU en DML) + kill criterion honesto |
| fp16 rompe parity o el converter se ahoga (como AudioSR) | Media | Ship fp32 (los grafos son chicos, <80MB total; fp16 es nice-to-have acá) |
| Calidad percibida no justifica 10× vs RIFE | Baja | Comparación visual en el smoke; el usuario decide por ojo — RIFE sigue default |

## Esfuerzo

Fases 0-2 ≈ port AudioSR (~1-2 semanas focalizadas; con la infra/patrones de SP13 ya escritos, probablemente
menos). Fase 3 = días, acotada y testeable en aislamiento. Fase 4 = M (todos los patrones ya existen en
Upflow: registry, availability, download script, selector UI, tests espejo).

## Self-Review

- Carpeta pedida (`C:\personal\GMFSS_port`) ✓, repo público con nombre consistente (`port-gmfss-onnx`) ✓,
  README con motivación explícita (Task 6.1) ✓, código libre bien documentado (README por fase con números
  medidos + manifest + docs) ✓.
- Eficiencia: benchmarks por fase, fps gates, kernel aislado, I/O threads, reuse(), fp16 medido no asumido ✓.
- Calidad: parity por componente 1e-3 + bit-parity del splat + SSIM final + prohibido backward-warp ✓.
- Buenas prácticas: TDD en cada task con tests concretos, review adversarial, licencias auditadas (trampa
  academic-only identificada), commits por capas, kill criterion definido ✓.
- Tipos consistentes: `GmfssEngine.run` = firma de `RifeNcnnEngine.run` (drop-in) ✓; `GmfssDriver.interpolate_pair`
  definido en 2.2 y consumido en 4.1 ✓.
