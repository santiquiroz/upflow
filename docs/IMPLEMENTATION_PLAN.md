# Upflow — Implementation Plan

Engineering plan to (1) fix the audited defects and (2) ship the headline feature: **AI frame interpolation (FPS boost)**. Written to be executed in a fresh session.

- **Repo:** `https://github.com/santiquiroz/upflow`
- **Local path:** `C:\Users\santi\.openclaw\workspace\image-upscaler-amd`
- **Audit verdict:** BLOCK — 2 CRITICAL, 6 HIGH, several MEDIUM (details below, each with `file:line`).
- **Method:** TDD. Write the failing test first, then the fix. Run `pytest` after every phase.

---

## 0. Prerequisites (do once, at session start)

1. **Branch:** `git checkout -b fix/audit-and-fps` off `master`.
2. **Dev deps:** `pip install -e ".[dev]"` (pytest + pytest-asyncio already declared in `pyproject.toml`).
3. **Baseline:** run `pytest` — only `tests/test_health.py` exists today; it should pass.
4. **Test strategy for GPU code:** the Real-ESRGAN / FFmpeg / RIFE binaries are heavy and machine-specific. Unit-test everything *around* the subprocess by **injecting a fake `_run_process` / engine** (dependency injection or monkeypatch). Do **not** require real GPU work in CI.
5. **Codex note:** the local Codex CLI is outdated (`gpt-5.6-sol requires a newer version of Codex`). Run `codex upgrade` before delegating any part of this to Codex, otherwise stay inline.

**Global acceptance for the whole plan:** `pytest` green, `runtime/` no longer grows without bound, `GPU_CONCURRENCY` is either honored or removed, and a video job can optionally output at 2×/3×/4× its source FPS.

---

## Phase 1 — CRITICAL

### 1.1 Upload filename collision → data corruption
**Where:** `app/api/routes.py:122` (image), `:170` (video); `app/services/storage.py:28`.
**Problem:** destination is `pending-{original_name}` — no per-request uniqueness. Two concurrent uploads with the same name write the same file (`open("wb")`) and corrupt each other; the later `rename()` can raise `FileNotFoundError`.
**Fix:** generate the job id (or a `uuid4().hex` token) **before** writing, and save straight to the final path. Eliminate the `pending-` → `final` rename entirely.

- Pre-generate `token = uuid4().hex`; safe display name = `Path(file.filename or default).name`.
- Compute `destination = uploads_path / f"{token}-{safe_name}"` and write there directly.
- Pass the pre-generated id into `create_job` so `job.id == token` and `job.source_path == destination` from the start (add an optional `job_id` / `source_path` param to `create_job`, or construct the dataclass in the route and hand it to a `register(job)` method).
- Delete the temp file in a `finally`, not only on `ValueError`.

**Test:** `tests/test_upload_collision.py` — fire two concurrent `create_job`-equivalents with the same filename against a fake engine; assert two distinct `source_path`s and no truncation.

### 1.2 Blocking I/O on the event loop → one-request app-wide DoS
**Where:** `app/services/media_tools.py:24` (`subprocess.run`, no timeout), `app/services/job_manager.py:86` (`Image.open`), `app/services/storage.py:28-33` (sync file write). `aiofiles` is a declared dep but never imported.
**Problem:** all synchronous, called from async paths. A hanging ffprobe or a large upload on a slow disk freezes the whole server.
**Fix:**
- `MediaTools.ffprobe_json`: make an async variant using `asyncio.create_subprocess_exec` + `asyncio.wait_for(..., timeout=30)`, or wrap the sync call in `asyncio.to_thread`. Add a timeout either way. Update the two call sites (`video_job_manager.py:73`, `video_upscaler.py:33`) to `await` it.
- `JobManager._validate_input_image`: run `Image.open` validation via `await asyncio.to_thread(...)`.
- `StorageService.save_upload`: switch the write loop to `aiofiles.open(...)` (already a dep), keeping the chunked size check.

**Test:** `tests/test_no_blocking_io.py` — monkeypatch ffprobe to sleep; assert a concurrent `/health`-style coroutine still returns promptly (event loop not blocked).

**Phase 1 acceptance:** concurrent same-name uploads don't corrupt; a slow/hung probe doesn't stall other requests.

---

## Phase 2 — HIGH

### 2.1 Disk leak (root cause of the 2.6 GB `runtime/`)
**Where:** `app/services/video_upscaler.py:25-31` (work dirs created, never removed); `job_manager.py` / `video_job_manager.py` (no cleanup anywhere). Confirmed empirically: 1708 orphaned frames + a 0-byte output mp4 on disk.
**Fix:**
- In `VideoUpscaler.run`, wrap the body in `try/finally` and `shutil.rmtree(work_dir, ignore_errors=True)` in `finally` (keep only the final encoded output).
- In both managers' `_worker`, after a job finishes (success or fail), delete the source upload (`job.source_path.unlink(missing_ok=True)`).
- Add a retention sweep for `outputs/`: a background task that deletes outputs older than `OUTPUT_TTL_HOURS` (new setting, default 24). Wire start/stop in `main.py` lifespan.

**Test:** `tests/test_cleanup.py` — run a video job with a fake pipeline that creates dummy frame files; assert `work_dir` is gone and source upload removed afterward, output kept.

### 2.2 `GPU_CONCURRENCY` is a no-op **+ image/video share no GPU limit**
**Where:** `app/services/job_manager.py:19,24,95-109` — `Semaphore(gpu_concurrency)` but a single worker task, so the semaphore never gates >1. **Plus (Codex, new):** `job_manager.py:19` and `video_job_manager.py:19` create **separate** semaphores, and `main.py:25-27` wires them independently — an image job and a video job can both run Real-ESRGAN on `-g 0` simultaneously, saturating VRAM even though `/health` reports concurrency 1.
**Fix:**
- **A (honor it):** in `start()`, spawn `settings.gpu_concurrency` worker tasks; keep them in a list; cancel all in `stop()`.
- **B (remove it):** delete the setting + semaphore, document single-GPU-job behavior, drop from `/health` + `.env.example`.
- **Cross-manager (do regardless of A/B):** inject **one shared GPU semaphore/runner** into both `JobManager` and `VideoJobManager` (create it in `main.py` lifespan, pass to both) so total concurrent GPU jobs across image+video is bounded by a single limit.
- Recommendation: **A + shared semaphore**, since throughput is the point but VRAM is the real constraint.

**Test:** `tests/test_concurrency.py` — with the shared semaphore at 1, a fake engine recording max simultaneous in-flight jobs, assert an image job and a video job never overlap; with limit 2, assert 2 run at once.

### 2.5 `CancelledError` leaves a zombie job + orphan subprocess (Codex, new)
**Where:** `app/services/job_manager.py:95-109` and `video_job_manager.py:101-115` — the worker's `except Exception` does **not** catch `CancelledError` (it's `BaseException`), but the `finally` still sets `finished_at` and calls `task_done()`. On shutdown mid-job the job is left `status=running` with a `finished_at` set (impossible state), no error, and the child ffmpeg/realesrgan process is never killed.
**Fix:** add an explicit `except asyncio.CancelledError:` branch that kills the running subprocess (via the guarded runner from 2.3), marks the job `failed`/`cancelled`, then re-raises so the worker task actually stops. Track the live subprocess handle on the job/manager so cancel can reach it.
**Test:** `tests/test_worker_cancel.py` — start a job with a fake long-running runner, cancel the worker task, assert the job ends `failed`/`cancelled` (not `running`) and the fake subprocess received `kill()`.

### 2.3 Subprocess: no timeout, no kill on cancel → dead queue / orphans
**Where:** `app/services/video_upscaler.py:155-163` (`_run_process`), `app/services/engines/realesrgan_ncnn.py:47-55`.
**Fix:** wrap `process.communicate()` in `asyncio.wait_for(..., timeout=SUBPROCESS_TIMEOUT)` (new setting). On `TimeoutError`/`CancelledError`: `process.kill()`, `await process.wait()`, then re-raise as a clear error. Factor a shared helper so both the image engine and video pipeline use the same guarded runner.

**Test:** `tests/test_subprocess_timeout.py` — fake a subprocess that hangs; assert it's killed and a timeout error surfaces without blocking forever.

### 2.4 Non-`ValueError` exceptions leak files + return 500
**Where:** `app/api/routes.py:136` (`except ValueError` only); `job_manager.py:84-93` (only `UnidentifiedImageError` caught); `video_job_manager.py:73` (`CalledProcessError` uncaught).
**Fix:**
- In validation, catch `Image.DecompressionBombError` and `subprocess.CalledProcessError` and re-raise as `ValueError` with a clean message.
- In routes, broaden cleanup: `except Exception` → cleanup temp file → re-raise as `HTTPException` (400 for validation, 500 otherwise). Always clean the temp file in a `finally` (folds into 1.1).

**Test:** upload a decompression-bomb-sized image and a malformed video; assert clean 400 and no leftover file in `uploads/`.

**Phase 2 acceptance:** jobs clean up after themselves; `GPU_CONCURRENCY` honored; a hung binary times out and is killed; bad inputs return 400 with no orphaned files.

---

## Phase 3 — MEDIUM + hardening

| # | Issue | Where | Fix |
|---|---|---|---|
| 3.1 | `crf`/`scale` `or`-default drops explicit `0` | `routes.py:173,177` | Use `x if x is not None else profile[...]` (match `keep_audio` at `:178`). |
| 3.2 | `fps` fallback accepts ffprobe `"0/1"` | `video_upscaler.py:39` | Helper that treats `0/x`, `0/0`, empty as invalid → fall through to `r_frame_rate` → `"30/1"`. **Reused by Phase 4.** |
| 3.3 | Output "exists" but 0 bytes marked completed | `video_upscaler.py:123`, `realesrgan_ncnn.py:57` | Also assert `output_path.stat().st_size > 0`. |
| 3.4 | No image format allow-list | `job_manager.py:84-93` | After open, check `img.format in {"PNG","JPEG","WEBP","BMP"}`, else `ValueError`. |
| 3.5 | ADS/reserved-name filenames (Windows) | `routes.py:121,169` | Reject/strip `:` and `< > " \| ? *`; derive on-disk name from `job.id` only (folds into 1.1), keep original as display metadata. |
| 3.6 | No CSRF/origin check (localhost drive-by) | `main.py` | Add middleware allow-listing `Origin`/`Referer` for state-changing POSTs (config `ALLOWED_ORIGINS`, default `http://127.0.0.1:8090` + `http://localhost:8090`). |
| 3.7 | `jpeg_quality` setting is dead | `config.py:174` | Wire into engine command if the ncnn binary supports it, else remove. |
| 3.8 | Unbounded queue | `job_manager.py:18`, `video_job_manager.py:18` | `asyncio.Queue(maxsize=N)` (config), return `429` when full. |
| 3.9 | Hardcoded relative paths (`app/static`, `app/templates`, `runtime`) | `main.py:45`, `web/routes.py:10`, `config.py` | Resolve against `Path(__file__).parent` / an explicit base dir. |
| 3.10 | Mislabeled profile: `general-balanced-2x` actually uses `scale=4` (Codex, new) | `config.py:86-91` | Rename profile to `general-balanced-4x` **or** set `scale: 2` with a 2x-capable model, so the label matches the real scale (VRAM/time surprise otherwise). |

Each row gets a focused unit test where it has logic (3.1, 3.2, 3.3, 3.4, 3.8).

**Phase 3 acceptance:** explicit `crf=0`/`scale=0` rejected (not silently overridden); VFR videos with `0/1` avg rate encode correctly; 0-byte outputs fail the job; only whitelisted image formats accepted; cross-origin POSTs rejected.

---

## Phase 4 — 🌊 FPS interpolation (the headline feature)

Goal: optional stage that turns an upscaled video into a higher-FPS one via **RIFE frame interpolation**, staying on NCNN/Vulkan (no CUDA), consistent with the existing engine pattern.

### 4.1 Bring in the engine binary
- New script `scripts/download-rife.ps1` (mirror `download-realesrgan.ps1`): fetch `rife-ncnn-vulkan` (use the [TNTwise fork](https://github.com/TNTwise/rife-ncnn-vulkan) for the newer v4.x + anime models) into `vendor/rife/`, models into `vendor/rife/models/`.
- Recommended default model: **`rife-v4.6`** (general) with **`rife-v4.6-anime`/RIFE 4.8** selectable; GMFSS as an optional heavier/quality path later.

### 4.2 Config (`app/config.py`)
Add settings: `RIFE_BINARY` (default `vendor/rife/rife-ncnn-vulkan.exe`), `RIFE_MODELS_DIR` (`vendor/rife/models`), `RIFE_MODEL` (`rife-v4.6`), `ENABLE_INTERPOLATION` (default `False`), `ALLOWED_FPS_MULTIPLIERS` (`2,3,4`). Add an `interpolation_available()` helper analogous to the engine check.

### 4.3 New engine `app/services/engines/rife_ncnn.py`
Wrap the CLI (list argv, no shell):
```
rife-ncnn-vulkan -i <frames_in> -o <frames_out> -m <model> -n <target_frame_count> -g 0 -f %08d.png
```
- `-n target_frame_count` = `source_frame_count * multiplier` (RIFE defaults to 2× when `-n` omitted; pass `-n` for 3×/4×).
- Reuse the **guarded runner** from Phase 2.3 (timeout + kill).
- `available()` = binary + models dir exist.

### 4.4 Wire into the video pipeline (`app/services/video_upscaler.py`)
Insert **between** upscale and encode:
1. After `frames-out` (upscaled) is produced, if interpolation is enabled for the job:
   - `frames_interp = work_dir / "frames-interp"`.
   - Count frames in `frames-out`; `target = count * multiplier`.
   - Run RIFE `frames-out → frames-interp`.
   - Encode from `frames-interp`.
2. **FPS math (critical for correct duration):** encode `-framerate` must be `source_fps * multiplier` so playback duration stays identical and audio stays in sync. Parse `source_fps` with the fixed helper from 3.2. Compute `new_rate = Fraction(source_fps) * multiplier` and pass as `num/den`.
3. Add `frames-interp` to the `finally` cleanup (Phase 2.1).

### 4.5 Job model / schema / API
- `VideoUpscaleJob` (`app/models.py`): add `fps_multiplier: int = 1` (1 = off).
- `create_video_job` (`app/api/routes.py:149`): add `fps_multiplier: int | None = Form(default=None)`, resolve via profile/default with the `is not None` pattern; validate against `ALLOWED_FPS_MULTIPLIERS` in `VideoJobManager._validate_request`.
- `VideoJobResponse` + `video_job_to_response`: expose `fps_multiplier` and the resulting `outputFps` in metadata.
- Optionally extend `VideoProfile` with a default `fps_multiplier` per preset.

### 4.6 UI (`app/templates/index.html`)
Add an "FPS boost" dropdown in the video form: Off / 2× / 3× / 4×. Render resulting output FPS in the job status (keep using `textContent`, no `innerHTML`).

### 4.7 Tests
- `tests/test_rife_engine.py` — with a fake runner, assert argv is built correctly (correct `-n`, model, dirs) for 2×/3×/4×.
- `tests/test_fps_math.py` — assert `new_rate = source_fps * multiplier` for cases incl. `24000/1001`, `30/1`, and rejects `0/1`.
- `tests/test_pipeline_stage_order.py` — with fakes, assert interpolation runs after upscale and encode reads from `frames-interp` when enabled, from `frames-out` when off.

**Phase 4 acceptance:** a video job with `fps_multiplier=2` produces an output with ~2× the source frame rate, identical duration, audio in sync; with `fps_multiplier=1` behavior is unchanged.

---

## Phase 5 — Tests, docs, release

1. **Coverage:** target 80%+ on `app/services/**` and `app/api/routes.py` (mock all binaries). Add the collision + concurrency + cleanup + timeout tests from earlier phases.
2. **Docs:** update `README.md` (tick the RIFE roadmap box, add FPS-boost to features + API params), `.env.example` (new settings), and `CONTRIBUTING.md` if setup changes.
3. **`.env.example`:** add every new key with sane defaults (`ENABLE_INTERPOLATION=false`, `RIFE_*`, `OUTPUT_TTL_HOURS`, `SUBPROCESS_TIMEOUT`, `ALLOWED_ORIGINS`, queue `maxsize`).
4. **Commit style:** follow the repo's Spanish "Historia técnica" layered format (Dominio/Aplicación/Infraestructura/Configuración/Pruebas) — see root `CLAUDE.md`.
5. **PR:** branch → PR into `master` with the audit summary + before/after (`runtime/` growth fixed, FPS feature demoed).

---

## Phase 6 — Roadmap: suite completa de anime (investigación 2026-07-13)

Basado en investigación profunda (informe interno: frame generation, audio IA, AFMF/LSFG, subtitulado). Todo lo de esta fase es **selectable/activable** vía config + UI, siguiendo el patrón de motores vendored por subprocess. Orden sugerido = orden de la tabla.

### 6.1 Mejora de audio IA (esfuerzo: bajo)
- **Motor principal:** DeepFilterNet CLI (`deep-filter.exe`, MIT/Apache-2.0, binario standalone, CPU RTF ~0.04-0.2 — sin CUDA). Script `download-deepfilternet.ps1` + engine `engines/deepfilter.py` con el guarded runner.
- **Bonus gratis:** filtro `arnndn` (RNNoise) del FFmpeg ya vendored como opción ligera.
- Etapa opcional del pipeline de video: extraer audio → mejorar → mux. Toggle `ENABLE_AUDIO_ENHANCE` + dropdown (Off / RNNoise / DeepFilterNet).
- Descartados por ahora: AudioSR/FlashSR/Apollo (difusión GPU-CUDA, sin camino AMD verificado; Apollo es CC BY-SA).

### 6.2 Subtitulado IA (esfuerzo: bajo-medio)
- **Motor:** whisper.cpp (`whisper-cli.exe`, MIT, emite SRT nativo con `-osrt`; CPU ok, backend Vulkan corre en AMD ~8x tiempo real con modelo large). Modelos ggml vendored (`download-whisper.ps1`).
- JP→EN integrado (task translate de Whisper). JP→ES: fase 2 con MADLAD-400 3B GGUF vía llama.cpp CPU (Apache 2.0). Evitar NLLB (CC-BY-NC).
- Upgrade de calidad posterior: modelo anime-whisper (CER 13.0% vs 16.5% stock en anime) convertido a ggml; pre-pass opcional Demucs (vocals) para pistas ruidosas.
- Pipeline: extraer audio (FFmpeg) → STT en CPU **en paralelo** con el upscale GPU (sin contención) → mux subs blandos al encode (`-c:s srt/ass` MKV, `mov_text` MP4). UI: toggle + selector de idioma.

### 6.3 Slider calidad ↔ velocidad (esfuerzo: bajo)
Presets Fast/Balanced/Best mapeados a knobs reales por motor (tabla completa en el informe): Real-ESRGAN (modelo, tile size), RIFE (modelo v4.6 vs más nuevos/pesados), DeepFilterNet on/off, whisper (small/medium/large). Un solo control en UI que resuelve a un dict de settings por job.

### 6.4 Frame generation — veredicto (investigado a fondo, incluye follow-up AFMF/LSFG)
- AFMF y LSFG 3.x **sí** son motion-vector-free (optical flow sobre frames de color), pero **no existe camino offline a archivos**: AFMF vive en el present-path del driver DX11/12; LSFG captura compositor en vivo; lsfg-vk (GPL-3) descarta frames en su modo debug y requiere poseer Lossless Scaling. El requisito de motion vectors es solo del FSR3 FG in-engine.
- Calidad: los flujos real-time (bloques 8x8, presupuesto sub-frame) son más crudos que el optical flow aprendido per-pixel de RIFE — offline, RIFE gana sin desventaja.
- **Decisión:** RIFE ncnn Vulkan (fork TNTwise) es EL motor de frame-gen offline. GMFSS/FILM/EMA-VFI/VFIMamba: sin ports Vulkan mantenidos, solo CUDA — reevaluar en 6-12 meses.
- Modelos RIFE adicionales (v4.26, rife-anime, etc. — ya vienen en el zip vendored) como opciones seleccionables del dropdown de calidad.

### 6.5 Modo tiempo real estilo Lossless Scaling — NO se construye
Arquitectónicamente incompatible con una app de archivos FastAPI/Python (requiere captura DirectX + present de baja latencia nativos). Recomendación al usuario: Lossless Scaling (Steam, ~$7) o Magpie (open source, GPL-3) para gaming; Upflow queda como pipeline offline de máxima calidad. Documentado en README.

### 6.6 Batch por temporada (para el flujo "temporada completa de anime")
- Subida múltiple / carpeta de entrada observada; cola ya bounded + workers ya concurrentes (fases 1-3) — falta UX de lote: seleccionar N episodios, progreso agregado, reanudación.
- `fps_multiplier` fraccional o modo "target fps" (`-n` de RIFE ya es frame-count absoluto → 23.976→60 exacto es viable: target = ceil(count × 60/23.976), encode a 60000/1001). Diseñar como `TARGET_FPS` alternativo al multiplicador entero.

---

## Quick reference — audit findings → phase

| Severity | Finding | Phase |
|---|---|---|
| CRITICAL | Upload filename collision | 1.1 |
| CRITICAL | Blocking I/O on event loop | 1.2 |
| HIGH | Disk leak (no cleanup) | 2.1 |
| HIGH | `GPU_CONCURRENCY` no-op + image/video share no GPU limit | 2.2 |
| HIGH | Subprocess no timeout/kill | 2.3 |
| HIGH | `CancelledError` → zombie job + orphan subprocess | 2.5 |
| HIGH | Non-`ValueError` leaks files/500 | 2.4 |
| HIGH | Job dict retained forever | 2.1 (retention sweep) |
| MEDIUM | `crf`/`scale` `or`-default | 3.1 |
| MEDIUM | `fps` accepts `0/1` | 3.2 |
| MEDIUM | 0-byte output = completed | 3.3 |
| MEDIUM | No image format allow-list | 3.4 |
| MEDIUM | ADS/reserved filenames | 3.5 |
| MEDIUM | No CSRF/origin check | 3.6 |
| MEDIUM | Unbounded queue | 3.8 |
| MEDIUM | Mislabeled profile `general-balanced-2x` = scale 4 | 3.10 |
| LOW | Dead `jpeg_quality`; hardcoded paths; tests | 3.7 / 3.9 / 5 |
| — | **NEW: FPS interpolation** | **4** |

## Suggested order & effort

1. Phase 1 (CRITICAL) — highest risk, cheap fixes. ~half day.
2. Phase 2 (HIGH) — stops the disk bleed + fixes concurrency. ~1 day.
3. Phase 4 (FPS feature) — the fun part; depends on the guarded runner (2.3) and fps helper (3.2). ~1–2 days incl. binary integration.
4. Phase 3 remaining MEDIUM + Phase 5 polish. ~1 day.
