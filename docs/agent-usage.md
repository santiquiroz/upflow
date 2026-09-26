# Upflow para agentes (Claude Code, Codex, cualquier cliente MCP)

Todo lo que un agente necesita para reescalar una imagen **sin abrir la UI y sin
levantar el servidor**: una CLI headless, un servidor MCP que sabe correr en
proceso, y una API con `/health` y parámetros de tiling explícitos.

## Instalación mínima

```powershell
git clone https://github.com/santiquiroz/upflow
cd upflow
python -m venv .venv
.venv\Scripts\pip install -e .
.venv\Scripts\pip install -e .[dev]     # solo para correr los tests
scripts\download-realesrgan.ps1        # pack ncnn (binario + modelos builtin) -> vendor/realesrgan
```

`pip install -e .` deja dos scripts en `.venv\Scripts\`: `upflow` (CLI) y
`upflow-mcp` (servidor MCP). Con el venv activado se llaman por nombre; sin
activarlo, por ruta (`.venv\Scripts\upflow.exe`).

## Tres comandos copiables

```powershell
# 1. ¿Hay GPU, pack ncnn y qué modelos puedo pedir?
upflow health --json

# 2. Reescalar 2x sin tocar la UI (el modelo corre a 4x y Upflow reduce con Lanczos)
upflow upscale --in foto.png --out foto-2x.webp --model realesrgan-x4plus --scale 2 --json

# 3. Modelos instalados (ids válidos para --model)
upflow models --json
```

`--json` imprime **una sola línea** al final; todo lo demás va a stderr. Sin
`--json` imprime una línea legible.

Salida de `upscale --json`:

```json
{"ok": true, "output": "C:/.../foto-2x.webp", "width": 2048, "height": 1152,
 "model": "realesrgan-x4plus", "engineModel": "realesrgan-x4plus",
 "engine": "realesrgan-ncnn-vulkan", "device": "dml:0", "scale": 2, "nativeScale": 4,
 "resized": true, "tile": {"size": 0, "overlap": 10, "meaning": "binary auto (heap-based)"},
 "format": "webp", "seconds": 6.7}
```

## Parámetros de `upflow upscale`

| Flag | Default | Qué hace |
|---|---|---|
| `--in PATH` | — | imagen de entrada (png/jpg/webp/bmp) |
| `--out PATH` | — | salida; la extensión define el formato salvo que se pase `--format` |
| `--model ID` | `realesrgan-x4plus` | builtin (`realesrgan-x4plus`, `realesrgan-x4plus-anime`, `realesr-animevideov3`) o un id de `upflow models` |
| `--scale 2\|3\|4` | `4` | escala pedida. Con un modelo x4 y `--scale 2/3` el motor corre a 4x y Upflow reduce con Lanczos (`resized: true`) |
| `--tile N` | omitido = auto | `auto`: ncnn deja al binario elegir por heap de VRAM (200 px en GPUs con >1.9 GB), ONNX usa `ONNX_TILE_SIZE`. `0`: sin tiling — ncnn elige el mayor tile que entra en la VRAM libre (tope: lado mayor de la imagen), ONNX un solo pase. `N>=32`: tile fijo |
| `--tile-overlap N` | `16` | solo motor ONNX (mezcla con feather). ncnn usa un prepadding fijo de 10 px, se reporta como `overlap: 10` |
| `--device ID` | `DEFAULT_DEVICE` | `dml:0`, `dml:1`, `cpu` (cpu solo con modelos ONNX) |
| `--format` | extensión de `--out` | `png`, `jpg`, `jpeg`, `webp` (los escribe el motor), `jxl`, `avif` (PNG del motor + ffmpeg vendorizado: `libjxl -distance 1.0`, `libaom-av1 -crf 18 -still-picture 1`) |

Otros subcomandos: `upflow models`, `upflow health`, `upflow preflight --repo <hf-repo>`,
`upflow install --repo <hf-repo> --yes` (sin `--yes` no descarga nada y sale con 2).

## Códigos de salida y errores

| Código | Significado | Ejemplo de `error` |
|---|---|---|
| `0` | ok | — |
| `2` | argumentos inválidos | `Scale must be one of [2, 3, 4]`, `unsupported output format 'tiff'`, `downloads need --yes` |
| `3` | modelo no instalado | `builtin model 'realesrgan-x4plus' needs the realesrgan-ncnn pack at ...`, `model 'x' is not installed (see upflow models)` |
| `4` | dispositivo | `Unknown device id 'dml:9'`, `device 'auto' needs the server's router; pass an explicit device` |
| `5` | fallo de inferencia u operación | `Real-ESRGAN NCNN reported a Vulkan failure (vkAllocateMemory failed -2); usually the tile does not fit in VRAM. Retry with a smaller tile_size`, `ffmpeg could not encode jxl: ...` |

Con `--json` los errores también salen como una línea: `{"ok": false, "error": "...", "code": 3}`.

## Determinismo

Mismo input + mismos parámetros ⇒ mismos bytes de salida. El binario ncnn es
determinista (tres corridas del mismo job dieron el mismo MD5), la reducción
Lanczos de Pillow también, y el id interno del job es un hash del contenido y
los parámetros (`cli-<sha1>`), así que no queda ningún nombre temporal
aleatorio en la metadata. `seconds` es el único campo que varía entre corridas.

## Video de cámaras de seguridad (`upflow cctv`, carril clásico)

Corre en proceso, en CPU y sin IA: filtros clásicos de ffmpeg con salida
determinista (sin publicar todavía).

```powershell
# 1. SHA-256 del archivo antes de tocarlo, contenedor (IMKH, DHAV, H.264 crudo...), índice de cuadros y diagnóstico
upflow cctv probe --in clip.mp4 --json

# 2. "Clarify video": copia sin pérdida, copia para ver, comparativo, cuadros, informe y paquete
upflow cctv clarify --in clip.mp4 --out-dir caso --preset night_ir `
  --osd 0,0,480,40 --osd 1500,1040,400,36 --osd-confirmed --trim 120:980 --frames 300,512 --json

# 3. "Check files are unchanged": vuelve a calcular cada SHA-256 de SHA256SUMS.txt
upflow cctv verify --dir caso/<jobId>.cctv --json
```

`probe --json` devuelve el mismo JSON que `POST /api/v1/video/cctv/analyze`
(`sourceSha256`, `receivedAt`, `container`, `video`, `frameIndex`, `gop`,
`quality`, `suggestedPreset`, `warnings`...) más `ok` y `presetSteps`: los pasos
clásicos de cada preset **para ese clip** (el desentrelazado y el aspecto dependen
del diagnóstico). `token` sale en `null` porque la CLI borra su copia de trabajo.

| Flag de `clarify` | Default | Qué hace |
|---|---|---|
| `--in PATH` | — | el clip tal como salió del grabador; no lo conviertas antes con otra herramienta |
| `--out-dir DIR` | — | ahí queda `<jobId>.cctv/`: `01_original` (copia verificada), `02_processed` (`.mkv` FFV1 y `.mp4` H.264), comparativo, cuadros, `report.html`/`report.json`, `frame_index.csv`, `SHA256SUMS.txt`, `reproduce.cmd` y el `.zip` de entrega |
| `--preset` | el sugerido por `probe` | `day`, `night_ir`, `analog`, `low_res` |
| `--osd X,Y,W,H` | — | una por caja de texto en pantalla (hora, cámara), en píxeles del cuadro guardado; ancho y alto pares. Esas cajas se copian del original sin denoise temporal |
| `--osd-confirmed` \| `--no-osd` | — | **hay que elegir uno**: las cajas tapan la hora y la cámara en un cuadro donde se ven, o el video no tiene texto en pantalla. Sin ninguno sale con `2` y `Confirm the on-screen text boxes on a frame where the time is visible, or choose 'No on-screen text'.` |
| `--trim A:B` | clip entero | primer y último cuadro, inclusive, contados en el índice |
| `--frames N,M` | ninguno | cuadros exportados como PNG (original y procesado, con su *framehash*) |

Salida de `clarify --json` (las rutas de `outputs` son relativas a `outputDir`):

```json
{"ok": true, "jobId": "8f0c...", "task": "clarify", "lane": "classic", "preset": "night_ir",
 "outputDir": "C:/.../caso/8f0c....cctv", "sourceSha256": "...", "receivedAt": {"utc": "...", "local": "..."},
 "framesIn": 861, "framesOut": 861,
 "outputs": {"analysis": "02_processed/...mkv", "viewing": "02_processed/...mp4", "comparison": "...",
             "stills": [{"frame": 300, "original": {...}, "processed": {...}}], "package": "....zip"},
 "report": "C:/.../report.html", "reportJson": "C:/.../report.json", "warnings": [], "seconds": 41.2}
```

`verify` sale con `0` si todo coincide y con `5` si algo cambió o falta
(`{"ok": false, "mismatches": [...], "missing": [...]}`). Detecta cambios
accidentales después de que Upflow recibió el archivo; no prueba que la
grabación sea auténtica ni dice qué pasó antes.

Códigos propios del modo: `2` para la decisión del OSD, un preset, recorte,
cuadro o caja inválidos; `3` si no hay ffmpeg o la build no trae FFV1/libx264;
`5` si el video no se puede decodificar o ffmpeg falla. Con `--json` el error
suma `key` (`cctv.error.*`), la misma clave que usa la API.

## MCP

`upflow-mcp` expone los mismos parámetros y el mismo JSON que la CLI:

| Tool | Equivale a |
|---|---|
| `upflow_health` | `upflow health --json` (con servidor: además el `/health` del servidor) |
| `upflow_upscale_image` | `upflow upscale` — acepta `tile_size`, `tile_overlap`; usa el servidor si está, si no corre en proceso |
| `upflow_upscale_image_headless` | `upflow upscale` siempre en proceso (sin servidor) |
| `upflow_list_models` | `upflow models --json` |
| `upflow_preflight_upscaler` | `upflow preflight --repo` (en proceso; `upflow_model_preflight` es la variante por servidor, multi-kind) |
| `upflow_install_upscaler(repo_id, confirm=true)` | `upflow install --repo --yes` (en proceso, espera a que termine; `upflow_install_model` es la variante por servidor, asíncrona) |
| `upflow_cctv_probe(file_path)` | `upflow cctv probe`. Con servidor sube el clip y espera el análisis aunque la API responda 202; en proceso la sesión queda viva para `upflow_cctv_clarify` (la barre el sweeper del servidor cuando arranca) |
| `upflow_cctv_clarify(token, preset, steps, osd_boxes, osd_confirmed, no_osd, trim, still_frames, acquisition, destination_dir)` | `upflow cctv clarify`. `steps` = `presetSteps[preset]` del probe (`[]` = sin filtros; preset sin `steps` es un error). Con servidor crea un job de la familia `video` (seguilo con `upflow_wait_job`; `cctv.artifacts` lista los archivos); en proceso espera y, con `destination_dir`, mueve ahí `<jobId>.cctv` |
| `upflow_cctv_check_unchanged(job_id, output_dir)` | `upflow cctv verify`; `output_dir` para una carpeta ya movida |

Modos (`--mode` o variable `UPFLOW_MCP_MODE`): `auto` (default: servidor si
responde en `UPFLOW_URL`, si no in-process), `server`, `inprocess`.
`--autostart` levanta `uvicorn app.main:app` en el puerto de `UPFLOW_URL`
(8090 por default) si no hay nadie escuchando, espera hasta 60 s y sigue en
in-process si no arranca.

### Registrar en Claude Code

```bash
claude mcp add upflow -- upflow-mcp --autostart
# o con ruta completa, sin activar el venv:
claude mcp add upflow -- C:/ruta/a/upflow/.venv/Scripts/upflow-mcp.exe --autostart
```

### Registrar en Codex

En `~/.codex/config.toml`:

```toml
[mcp_servers.upflow]
command = "C:/ruta/a/upflow/.venv/Scripts/upflow-mcp.exe"
args = ["--autostart"]
```

## API REST (si preferís hablar con el servidor)

```bash
curl http://127.0.0.1:8090/api/v1/health
# -> version, ncnnAvailable, onnxAvailable, devices[{id, freeVramMb}], defaultDevice,
#    modelsInstalled, tile{ncnnDefault, onnxTileSize, onnxTileOverlap}

curl -X POST http://127.0.0.1:8090/api/v1/jobs -F "file=@foto.png" \
  -F "model_name=realesrgan-x4plus" -F "scale=2" -F "output_format=webp" -F "tile_size=0"
curl http://127.0.0.1:8090/api/v1/jobs/<jobId>
# -> metadata.effective: {engine, command, nativeScale, requestedScale, tileSize, tileOverlap, resized}
```

## Por qué existe el parámetro de tiling (y la reducción Lanczos)

`realesrgan-ncnn-vulkan` no sabe reescalar a una escala distinta de la del
modelo. Con `-s 2` sobre `realesrgan-x4plus` producía una imagen del tamaño
correcto pero armada con el cuarto superior izquierdo de cada tile ampliado
x4: una rejilla de bloques de 400 px con saltos de brillo (PSNR 13 dB contra
el x4 real). Desde 2026-09-02 el motor corre siempre a la escala nativa y
Upflow reduce con Lanczos; `tests/test_seams.py` mide la rejilla y falla si
vuelve.

![antes / después](images/ncnn-scale2-seams-before-after.png)

Medido en una RX 7800 XT (16 GB) con `realesrgan-x4plus` a 4x sobre 1024×576:

| `-t` | pico de VRAM | tiempo | costuras |
|---|---|---|---|
| 0 (auto → 200) | 1.5 GB | 7.0 s | no |
| 400 | 4.1 GB | 5.8 s | no |
| 600 | 8.1 GB | 6.9 s | no |
| 900 | 11.7 GB | 6.9 s | no |
| 1000 / 1024 | — | — | `vkAllocateMemory failed -2`, imagen plana con exit 0 (ahora se detecta y falla con código 5) |

Tiles más grandes no acortan el tiempo ni cambian la calidad a escala nativa,
por eso el default sigue siendo el `auto` del binario y `tile_size=0` se
resuelve como "el mayor tile que entra en la VRAM libre" (≈ 600 MB + 0.0215 MB
por píxel de tile) en vez de "sin tiling a ciegas".
