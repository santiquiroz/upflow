<div align="center">

# ⚡ Upflow

### Modern AI upscaling **+** frame interpolation — open source, Vulkan-native, built for AMD.

*Think Lossless Scaling, but for your files: batch-upscale anime and photos, rebuild video frame-by-frame, and (soon) interpolate to buttery-smooth high FPS — all on your own GPU, no cloud, no CUDA lock-in.*

[![License: MIT](https://img.shields.io/badge/License-MIT-22c55e.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Backend: Vulkan](https://img.shields.io/badge/backend-Vulkan-AC162C.svg?logo=vulkan&logoColor=white)](https://github.com/xinntao/Real-ESRGAN-ncnn-vulkan)
[![PRs welcome](https://img.shields.io/badge/PRs-welcome-8b5cf6.svg)](CONTRIBUTING.md)

</div>

---

## Why Upflow

Most good upscalers are either NVIDIA/CUDA-only, closed-source, or a pile of CLI flags. Upflow is a clean **web UI + REST API** wrapped around a **decoupled, swappable engine**, running on **Real-ESRGAN NCNN + Vulkan** — which means it screams on **AMD Radeon** hardware on Windows where DirectML and CUDA fall short.

- 🖼️ **Images** — drag, pick a model, upscale up to 4×. Photos and anime/line-art alike.
- 🎬 **Video** — full FFmpeg pipeline: extract frames → batch upscale → re-encode with audio preserved. Anime & general presets included.
- 🧩 **Decoupled engine** — the model backend is a component behind an interface. NCNN/Vulkan today, anything tomorrow.
- 🔌 **REST API** — queue jobs and poll status from any other app.
- 🏠 **100% local** — your media never leaves your machine.

> **Runs on any Vulkan GPU** (AMD, NVIDIA, Intel). It's simply *tuned* for AMD-on-Windows, where the good options are thin.

## ✨ Roadmap — the "modern" part

Upscaling is spatial super-resolution. The next leap is **temporal**: AI **frame interpolation** to turn 24 fps into smooth 48/60/120 fps — the same trick Lossless Scaling does in real time, but applied losslessly to your output file.

- [ ] 🌊 **FPS interpolation via RIFE** (`rife-ncnn-vulkan`) — 2×/3×/4× frame rate, anime-optimized models (RIFE 4.8 / GMFSS)
- [ ] 🧹 Automatic disk cleanup + job retention (TTL)
- [ ] 🗄️ Persistent job store (SQLite)
- [ ] 📊 Prometheus / OpenTelemetry metrics
- [ ] 🔑 Optional API-key auth + rate limiting for remote use

See the full engineering plan in **[`docs/IMPLEMENTATION_PLAN.md`](docs/IMPLEMENTATION_PLAN.md)**.

## 🚀 Quickstart (Windows / PowerShell)

```powershell
git clone https://github.com/santiquiroz/upflow.git
cd upflow

# 1. Python env + deps
powershell -ExecutionPolicy Bypass -File .\scripts\setup.ps1

# 2. Download the Real-ESRGAN NCNN Vulkan engine + models
powershell -ExecutionPolicy Bypass -File .\scripts\download-realesrgan.ps1

# 3. (video only) Download FFmpeg
powershell -ExecutionPolicy Bypass -File .\scripts\download-ffmpeg.ps1

# 4. Run
.\.venv\Scripts\uvicorn app.main:app --host 127.0.0.1 --port 8090 --reload
```

Open **http://127.0.0.1:8090**.

## 🧠 Models

| Model | Best for | Scales |
|---|---|---|
| `realesrgan-x4plus` | Photos, general images | 4× |
| `realesrgan-x4plus-anime` | Still anime, illustration, line art | 4× |
| `realesr-animevideov3-x2/x3/x4` | Anime video frames | 2× / 3× / 4× |
| `realesr-animevideov3` | Auto preset (picks x2/x3/x4 by scale) | 2–4× |

## 🔗 API

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/api/v1/health` | Healthcheck + queue depth |
| `GET` | `/api/v1/engine` | Engine status, models, video profiles |
| `POST` | `/api/v1/jobs` | Create image job |
| `GET` | `/api/v1/jobs/{id}` | Image job status |
| `GET` | `/api/v1/jobs/{id}/download` | Download upscaled image |
| `POST` | `/api/v1/video/jobs` | Create video job |
| `GET` | `/api/v1/video/jobs/{id}` | Video job status |
| `GET` | `/api/v1/video/jobs/{id}/download` | Download upscaled video |

```bash
curl -X POST http://127.0.0.1:8090/api/v1/jobs \
  -F "file=@input.png" \
  -F "model_name=realesrgan-x4plus-anime" \
  -F "scale=4" \
  -F "output_format=png"
```

## 🏗️ Architecture

```text
Browser / API client
        │
   FastAPI (app/)  ──  web UI (Jinja) + REST routers
        │
   Job queue (per media type)  ──  async worker + GPU semaphore
        │
   ┌────┴─────────────┐
   │                  │
Image engine     Video pipeline (FFmpeg)
(Real-ESRGAN     extract frames → upscale batch → re-encode + audio
 NCNN Vulkan)    (RIFE interpolation stage — coming)
```

The engine sits behind an `UpscaleEngine` interface (`app/services/engines/base.py`), so the Vulkan backend is a drop-in component.

## 🤝 Contributing

PRs welcome. See [`CONTRIBUTING.md`](CONTRIBUTING.md) and the [implementation plan](docs/IMPLEMENTATION_PLAN.md) for where help is most useful. Good first areas: disk cleanup, tests, and the RIFE interpolation stage.

## 📄 License

[MIT](LICENSE) © 2026 Santiago Quiroz. Do whatever you want with it.

---

<div align="center">
<sub>Built with FastAPI · Real-ESRGAN · NCNN · Vulkan · FFmpeg</sub>
</div>
