# Personas 4/4: paquetes del instalador (detector, LaMa, OSOR) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Que un Upflow instalado pueda bajar el detector de personas y armar el paquete OSOR (y LaMa, si todavía no tiene camino) desde el propio Editor, con descargas fijadas por hash.

**Architecture:** Mismo mecanismo que `mobilesam`/`migan`: un script PowerShell por paquete en `scripts/`, registrado en `PACK_SCRIPTS` (`app/services/pack_provisioner.py`), con etiqueta en `missing_pack.py`, test de fijación de URL/hash y entrada en `THIRD_PARTY_NOTICES.md`. OSOR descarga de Hugging Face en revisiones fijas, calcula los embeddings del prompt fijo y llama a `scripts/build_osor_pack.py`. El Editor ofrece la descarga con `PackDownload`.

**Tech Stack:** PowerShell 5.1 (scripts de paquetes), Python 3.11 (`huggingface_hub`, `transformers`, `diffusers`, `peft`), FastAPI, React (`PackDownload`).

**Spec:** `docs/superpowers/specs/2026-10-30-editor-personas-design.md` (sección "Distribución"; decisión menor sobre `peft`).

**Depende de:** Plan 1 (setting `person_detector_model`), Plan 3 (Editor que muestra los motivos).

## Global Constraints

- Las del Plan 1 (licencias, estilo, tests, sin push).
- Detector: `https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.onnx`, sha256 `c5c2d13e59ae883e6af3b45daea64af4833a4951c92d116ec270d9ddbe998063`, 35858002 bytes → `vendor\yolox\yolox_s.onnx`. Licencia Apache-2.0.
- Base OSOR: `diffusers/stable-diffusion-xl-1.0-inpainting-0.1` en la revisión `115134f363124c53c7d878647567d04daf26e41e` (CreativeML Open RAIL++-M). Archivos y sha256:
  - `unet/diffusion_pytorch_model.fp16.safetensors` 5135178560 `6470840731e98cc16713ddf3ac7ee458c9fdbcb881a98c6727cd4a938f227d3f`
  - `vae/diffusion_pytorch_model.fp16.safetensors` 167335338 `4ad62825e5c8b31eefb77355ba6693785619bf7d669d9b1a6fd9f19dec6d65b3`
  - `text_encoder/model.fp16.safetensors` 246144867 `fc83cf401d930147807e7c44021c164bcc5508c9d4cc0ff35f4e354685ca9cd0`
  - `text_encoder_2/model.fp16.safetensors` 1389382884 `a8622bd41f8d359df484fdf9de091edb9337a2fd747f0a5c9c320a62e24d1fd3`
  - más los `config.json` de unet/vae/text_encoder/text_encoder_2, `scheduler/scheduler_config.json` y `tokenizer/`, `tokenizer_2/` (chicos, sin hash LFS: se fija la revisión).
- Pesos OSOR: `QinmingZhou/OSOR` en la revisión `9288156766b1df44e5f50fb697876a6f2fd8c062`, archivo `osor-sdxlinpainting/weights/sdxlinpainting_phase2.pth` 3112296806 bytes, sha256 `4f6abee0f152919fa8faca1895820ee531adb2b42b5de25db2e9a311aa47face` (RAIL++-M); código OSOR MIT.
- Prompt fijo de OSOR: `"Remove the instance of object"`; `prompt_embeds.pt` = dict `prompt_embeds` [1,77,2048], `pooled_prompt_embeds` [1,1280], `prompt` (texto).
- El paquete OSOR exige aceptar la licencia RAIL++-M (parámetro `acceptLicense` existente del provisionador).
- Nunca escribir en `vendor/` del worktree durante el desarrollo (es un junction al repo principal): los tests corren los scripts con destino temporal.

## Review Focus

- Descarga cortada o archivo corrupto del mismo tamaño: el script falla con código ≠ 0 y no deja archivos a medias (patrón de `test_lama_download_pin.py`).
- Disco sin espacio para OSOR (~9 GB de descarga + ~5 GB de paquete): el script lo dice antes de empezar (chequeo de espacio libre) en vez de fallar a mitad.
- Paquete OSOR ya instalado: volver a correrlo no rehace nada ni borra el paquete válido.
- Sin aceptar la licencia RAIL++-M: el provisionador se niega con un mensaje que nombra la licencia.
- Editor con el detector sin instalar: la herramienta Personas muestra el motivo y el botón de descarga; al terminar la descarga la herramienta se habilita sin recargar la app.

---

### Task 1: Paquete `person-detector`

**Files:**
- Create: `scripts/download-yolox.ps1`, `tests/test_yolox_download_pin.py`
- Modify: `app/services/pack_provisioner.py` (`PACK_SCRIPTS` :28-72), `THIRD_PARTY_NOTICES.md`, `tests/test_third_party_notices.py` si hace falta la entrada
- Test: `tests/test_yolox_download_pin.py`, `tests/test_missing_pack.py`, `tests/test_pack_provisioner.py`

**Interfaces:**
- Produces: `PACK_SCRIPTS["person-detector"] = "download-yolox.ps1"`; el script acepta `-Destino` (por defecto `vendor\yolox`) para que los tests usen una carpeta temporal.

- [ ] **Step 1: Write the failing test** copiando la estructura de `tests/test_lama_download_pin.py`: constantes `MODEL_URL`, `MODEL_SHA256`, `MODEL_SIZE` de las Global Constraints presentes en el script; corrida con `Invoke-WebRequest` falso que deja un archivo corrupto del mismo tamaño → código ≠ 0 y sin archivos en el destino.
- [ ] **Step 2: Run** `pytest tests/test_yolox_download_pin.py -q` → FAIL.
- [ ] **Step 3: Implement** el script con el mismo patrón que `download-lama.ps1` (`.download`, tamaño y SHA-256 con .NET, mover al final); registrar el paquete; entrada en `THIRD_PARTY_NOTICES.md` (YOLOX, Megvii, Apache-2.0).
- [ ] **Step 4: Run** `pytest tests/test_yolox_download_pin.py tests/test_missing_pack.py tests/test_pack_provisioner.py tests/test_third_party_notices.py -q` → PASS.
- [ ] **Step 5: Commit** `feat: paquete descargable del detector de personas`

### Task 2: Paquete `lama` (verificar)

**Files:**
- Read: `app/services/pack_provisioner.py`, `scripts/download-lama.ps1`, `installer/upflow.iss` (:72-73, :131-150), `scripts/upflow-launcher.ps1` (:335-397)

- [ ] **Step 1: Verificar** si `"lama"` está en `PACK_SCRIPTS`, en los componentes del instalador y en `$script:OptionalPacks` del lanzador (el mapa del código encontró señales contradictorias).
- [ ] **Step 2: Si falta en alguno**, agregarlo siguiendo el patrón de `migan` (componente `eraser`, línea en `optional-packs.txt`), con su test donde exista uno análogo; si está completo, no tocar nada y anotarlo en el resumen.
- [ ] **Step 3: Run** `pytest tests/test_pack_provisioner.py tests/test_lama_download_pin.py -q` → PASS.
- [ ] **Step 4: Commit** (solo si hubo cambios) `fix: LaMa se instala como paquete opcional`

### Task 3: Embeddings del prompt fijo de OSOR

**Files:**
- Create: `scripts/cache_osor_prompt.py`
- Modify: `scripts/build_osor_pack.py` (aceptar `--embeds` opcional y calcularlo si falta)
- Test: `tests/test_cache_osor_prompt.py`

**Interfaces:**
- Produces: `def encode_fixed_prompt(text_encoder, text_encoder_2, tokenizer, tokenizer_2, prompt: str = "Remove the instance of object") -> dict[str, torch.Tensor | str]` (como SDXL: penúltima capa oculta de los dos encoders concatenadas → [1,77,2048]; `text_embeds` del encoder 2 → [1,1280]); CLI `python scripts/cache_osor_prompt.py --base <carpeta> --out prompt_embeds.pt`.

- [ ] **Step 1: Write the failing test** — con encoders falsos de dimensiones chicas, la salida tiene las claves, la concatenación por la última dimensión y el texto del prompt.
- [ ] **Step 2: Run** → FAIL. **Step 3: Implement.**
- [ ] **Step 4: Verificar contra el archivo de la sesión** — calcular con la base del spike (`C:/wt/osor-spike/sdxl-inpainting`, si existe) y comparar con `C:/wt/osor-spike/src/osor-sdxlinpainting/pretrained_weights/fixed_prompt_embeds.pt`: diferencia máxima < 1e-3 en fp32. Si el spike ya no existe, saltar este paso y decirlo.
- [ ] **Step 5: Run** `pytest tests/test_cache_osor_prompt.py -q` → PASS.
- [ ] **Step 6: Commit** `feat: calcular los embeddings del prompt fijo de OSOR`

### Task 4: Paquete `osor`

**Files:**
- Create: `scripts/provision-osor.ps1`, `scripts/fetch_osor_sources.py`, `tests/test_osor_provision_pin.py`
- Modify: `app/services/pack_provisioner.py` (`PACK_SCRIPTS`, licencia requerida), `pyproject.toml` (extra opcional `osor = ["peft>=0.14"]` o equivalente que use el instalador), `THIRD_PARTY_NOTICES.md` (OSOR MIT; pesos y base RAIL++-M)

**Interfaces:**
- Produces:
  - `fetch_osor_sources.py --dest <carpeta>`: `huggingface_hub.hf_hub_download` por archivo con las revisiones fijas; verifica tamaño y sha256 de los cinco archivos grandes; chequea espacio libre (≥ 16 GB) antes de empezar; reanuda descargas parciales; sale ≠ 0 ante cualquier diferencia.
  - `provision-osor.ps1`: si `<models>\eraser-osor` ya es válido (`osor_available`), termina sin hacer nada; si no, corre `fetch_osor_sources.py` en una carpeta de trabajo, `cache_osor_prompt.py`, `build_osor_pack.py --out <models>\eraser-osor`, y borra la carpeta de trabajo al terminar bien.
  - `PACK_SCRIPTS["osor"] = "provision-osor.ps1"`, con aceptación de licencia obligatoria (mismo mecanismo que los paquetes que ya la piden).

- [ ] **Step 1: Write the failing test** — las revisiones, rutas y hashes de las Global Constraints están en `fetch_osor_sources.py`; sin `acceptLicense` el provisionador rechaza `osor`; con un `hf_hub_download` falso que devuelve un archivo con hash distinto, `fetch` sale ≠ 0 y no deja el archivo.
- [ ] **Step 2: Run** → FAIL. **Step 3: Implement.** **Step 4: Run** `pytest tests/test_osor_provision_pin.py tests/test_pack_provisioner.py tests/test_missing_pack.py tests/test_third_party_notices.py -q` → PASS.
- [ ] **Step 5: Corrida real** (larga, ~9 GB): `provision-osor.ps1` hacia una carpeta temporal fuera de `vendor/` y de `runtime/models`; verificar `build_report.json` (paridad 1 media fp32 < 1e-3) y comparar con el paquete existente `C:/Users/santi/AppData/Local/Upflow/runtime/models/eraser-osor` (mismos tamaños de archivo). Borrar la carpeta temporal.
- [ ] **Step 6: Commit** `feat: paquete OSOR armado por el instalador`

### Task 5: Instalar desde el Editor

**Files:**
- Modify: `frontend/src/modules/editor/EditorPanel.tsx` / `PeoplePanel.tsx` / panel Acción (Plan 3), `installer/upflow.iss`, `scripts/upflow-launcher.ps1` (`$script:OptionalPacks`), i18n
- Test: `frontend/src/modules/editor/PeoplePanel.test.tsx`, `EditorPanel.test.tsx`

**Interfaces:**
- Consumes: `<PackDownload pack variant? reason? onDone? />` (`frontend/src/components/PackDownload.tsx:27`), `provisionPack` (`services/capabilities.ts:22`).
- Produces: con `people: false` en las capacidades, el panel de Personas muestra el motivo y `PackDownload pack="person-detector"`; al terminar invalida `["editorCapabilities"]` y la herramienta se habilita. En el panel Acción, si `eraser-auto` está sin OSOR, una línea "OSOR mejora los huecos grandes" con `PackDownload pack="osor"` (con aceptación de licencia). El instalador ofrece `person-detector` como paquete opcional en el componente `editor` (OSOR no: pesa demasiado para el instalador; se baja desde la app).

- [ ] **Step 1: Write the failing tests** — sin capacidad `people`, el panel muestra el botón de descarga; `onDone` invalida las capacidades y habilita la herramienta.
- [ ] **Step 2: Run** → FAIL. **Step 3: Implement.** **Step 4: Run** `npx vitest run src/modules/editor` → PASS.
- [ ] **Step 5: Commit** `feat: el Editor ofrece bajar el detector y OSOR`
