# Research: Frame Interpolation, Audio Enhancement, Quality/Speed Tuning, Real-Time Scaling (2025-2026)

Scope: Upflow — local Windows/AMD (NCNN/Vulkan, subprocess-driven, no CUDA) video/image upscaler. Current pipeline: `vendor/realesrgan/realesrgan-ncnn-vulkan-v0.2.0-windows` (xinntao's ncnn build, subprocess via `RealEsrganNcnnEngine` in `app/services/engines/realesrgan_ncnn.py`) + planned RIFE ncnn Vulkan + FFmpeg.

---

## 1. Executive Summary

1. RIFE ncnn Vulkan remains the correct choice for frame interpolation on AMD/no-CUDA — no competitor (GMFSS, FILM, IFRNet, EMA-VFI, VFIMamba, GIMM-VFI) has a maintained, subprocess-ready Vulkan/DirectML port.
2. Use **TNTwise/rife-ncnn-vulkan** (MIT, active, last release 2026-02-05) instead of the original nihui/rife-ncnn-vulkan (MIT, frozen since 2025-01-12) — same CLI contract, newer models, drop-in.
3. IFRNet has a real ncnn/Vulkan port (nihui/ifrnet-ncnn-vulkan, MIT) but it's stale since 2022 — usable as a low-effort secondary option, not a priority.
4. GMFSS has an ncnn port (Justin62628/gmfss-ncnn-vulkan, MIT) but it is a 10-star, unmaintained-since-2023 side project — do not depend on it.
5. FILM, EMA-VFI, VFIMamba, GIMM-VFI: PyTorch/CUDA research code only, zero ncnn/DirectML ports found on GitHub as of July 2026 — not viable without a full ONNX conversion + DirectML validation effort (High).
6. FSR3 Frame Generation, DLSS Frame Generation, and AMD AFMF are all **not usable for offline file processing** — confirmed via GPUOpen/NVIDIA docs: they require either live engine-provided motion vectors (FSR3, DLSS FG) or a live DirectX 11/12 swapchain to intercept (AFMF, driver-level). None expose a file-in/file-out path.
7. For audio, **DeepFilterNet's Rust `deep-filter` CLI** is the standout: standalone binary, MIT/Apache-2.0 dual license, no CUDA, RTF ≈0.04–0.2 on a single CPU core — directly matches the subprocess pattern already used for Real-ESRGAN.
8. Demucs (stem separation) is best sourced from **adefossez/demucs** (active, pushed within the last 2 days) — the original facebookresearch/demucs mirror is archived since April 2024.
9. AudioSR/FlashSR/Apollo (audio super-resolution/restoration) are diffusion-family models, GPU-oriented (validated on A6000/RTX 4090), no confirmed AMD DirectML path, and Apollo is CC BY-SA 4.0 (share-alike) — treat as optional/experimental, not a v1 target.
10. Real-time gaming-style scaling (Lossless Scaling, Magpie, AFMF) is architecturally incompatible with a Python/FastAPI offline-file app — verdict: do not build, recommend Magpie (GPL-3.0, open source) as an external tool link, keep Upflow file-based.

---

## 2. Comparison Tables

### Topic 1 — Frame interpolation / frame generation

| Candidate | Quality | Speed | AMD/no-CUDA path | Standalone CLI | License | Maturity | Integration effort |
|---|---|---|---|---|---|---|---|
| **RIFE (nihui/rife-ncnn-vulkan)** | Good, established baseline | Fast (ncnn, fp16) | Yes — Vulkan, works on Intel/AMD/Nvidia | Yes, `.exe` | MIT | Frozen since 2025-01-12 (still functional) | Low (already planned) |
| **RIFE (TNTwise/rife-ncnn-vulkan fork)** | Same RIFE weights, more model versions (v4.6–v4.25, lite variants) | Fast | Yes — same Vulkan backend | Yes, `.exe` | MIT | Active, last release 2026-02-05 | Low (same CLI flags as nihui's) |
| **IFRNet (nihui/ifrnet-ncnn-vulkan)** | Competitive with RIFE on some benchmarks, different architecture (feature-refine, no explicit flow warp) | Fast (ncnn) | Yes — Vulkan | Yes, `.exe` | MIT | Stale since 2022-07-20 | Low technically, but unmaintained |
| **GMFSS (Justin62628/gmfss-ncnn-vulkan)** | Higher quality than RIFE on large motion (per GMFSS papers), heavier | Slower than RIFE | Yes — Vulkan (claimed) | Yes | MIT | Unmaintained since 2023-07-03, 10 stars, 0 issues (likely near-zero usage) | Medium-High (unverified stability) |
| **GMFSS (HolyWu vs-gmfss_union/fortuna)** | High | Medium-slow | No — PyTorch + TensorRT/CUDA only | VapourSynth plugin, not a standalone CLI | Varies | Active but NVIDIA-only | Not viable (CUDA required) |
| **FILM (google-research/frame-interpolation)** | Very good on large motion, single-network | Slow (TF2, large model) | No confirmed AMD path; no ncnn/ONNX-DirectML port found | No official CLI, TF2 script only | Apache 2.0 | Maintained by community forks, original repo inactive research code | High (would need TF→ONNX conversion + DirectML validation) |
| **EMA-VFI** | High (attention-based) | Slow, transformer cost | No ncnn/ONNX/DirectML port found | No | Apache 2.0 (paper code) | Research code, CUDA-oriented | High |
| **VFIMamba** | SOTA on large-motion/high-res benchmarks (NeurIPS 2024) | Better FLOPs/quality ratio than transformer methods, still needs GPU | No ncnn/ONNX/DirectML port found | No | Research code license | Active research repo (MCG-NJU), no deployment tooling | High |
| **GIMM-VFI** | High, arbitrary-timestep continuous motion modeling (NeurIPS 2024) | Slow-medium | No ncnn/ONNX/DirectML port found | No | Research code license | Active, checkpoints public, training code released Nov 2024 | High |
| **Practical-RIFE / SVFI** | Same RIFE family, SVFI adds a GUI + AMD upscaling via DirectML for its *upscaling* module (not interpolation) | Fast | Partial — SVFI Pro adds DirectML super-resolution on AMD, interpolation itself still ncnn/RIFE-based | SVFI is GUI-only; Practical-RIFE is Python/PyTorch | MIT (RIFE core) | Active | Not a CLI fit — GUI app, not subprocess-friendly |
| **FSR3 Frame Generation** | N/A (real-time only) | N/A | **Not usable offline** — needs render-resolution motion vectors + depth from a live engine per-frame; no file-based API exists | No | Proprietary/GPUOpen SDK | N/A | Not applicable |
| **DLSS Frame Generation** | N/A (real-time only) | N/A | **Not usable offline** — NVIDIA-only (RTX 40+), consumes the final composited color buffer inside a live swapchain via Streamline SDK, fixed-size buffer per frame, no file I/O mode | No | Proprietary | N/A | Not applicable |
| **AMD AFMF (driver-level)** | N/A (real-time only) | N/A | **Not usable offline** — operates on consecutive frames inside the live DirectX 11/12 swapchain via the AMD driver; there is no file-based entry point, it never sees a video file, only rendered frames in flight | No | Proprietary (driver feature) | N/A | Not applicable |

Verified AMD/no-CUDA claims (2+ sources): RIFE ncnn Vulkan cross-vendor GPU support was confirmed both in the nihui repo's own release notes (mentions AMD/Intel/Nvidia GPU support and AMD driver links) and independently via TNTwise's actively maintained fork carrying the same Vulkan backend forward into 2026 releases. FSR3/DLSS FG/AFMF's offline-inapplicability was cross-checked against GPUOpen's official FSR3 integration docs (motion-vector requirement) and AMD's own AFMF announcement/PC Gamer coverage (driver intercepts the live DX11/12 swapchain, no game/file awareness beyond consecutive rendered frames).

### Topic 2 — Audio enhancement

| Candidate | Purpose | Quality | Speed (CPU, no CUDA) | AMD/no-CUDA path | Standalone CLI | License | Maturity | Integration effort |
|---|---|---|---|---|---|---|---|---|
| **DeepFilterNet (`deep-filter` Rust CLI)** | Denoise, 48kHz full-band speech enhancement | Good for speech/dialogue, less for music/SFX | RTF ≈0.04 (DeepFilterNet2, notebook i5, single thread) to 0.42 on Raspberry Pi 4 — i.e. much faster than realtime on any modern desktop CPU | Yes — pure Rust/ONNX-ish inference (tract), no CUDA, no GPU needed at all | **Yes**, prebuilt binary, `deep-filter -i in.wav -o outdir` | MIT / Apache-2.0 dual | Active-ish (last push 2024-10, stable, widely used) | **Low** — same subprocess-per-file pattern as Real-ESRGAN |
| **Demucs (adefossez/demucs)** | Stem separation (vocals/drums/bass/other) — useful for isolating dialogue before denoise, not a v1 requirement | High, SOTA-class (HT Demucs v4) | Minutes-scale on CPU for a full track (GPU: seconds); no official RTF on CPU published in this research, but community reports ~5 min CPU vs ~10s GPU for a 3-4 min song | Partial — PyTorch, CPU works out of the box; ROCm build possible on Linux, not on Windows; practically CPU-only on Windows/AMD | Yes, `demucs` CLI (Python entry point, not a single binary) | MIT | Active — pushed within the last 2 days (original repo `facebookresearch/demucs` is **archived** since 2024-04) | Medium — needs a Python/venv or PyInstaller-bundled runtime, not a plain `.exe` |
| **Resemble Enhance** | Denoise + bandwidth extension/enhancement | Good, designed for speech restoration to 44.1kHz | CUDA hardcoded as preferred device; multiple open GitHub issues (#16, #17) report it breaking or falling back poorly without an NVIDIA GPU | **No confirmed clean AMD/no-CUDA path** — CPU fallback exists but is reported buggy/slow, no DirectML build | No, Python (Gradio app / library), Windows fork exists but still PyTorch-CUDA-first | MIT | Active-ish | High (would need patching device selection + accept slow CPU fallback) |
| **VoiceFixer** | General speech restoration (noise, reverb, low-res, clipping) in one model | Good all-rounder for speech | Not benchmarked in this research; ResUNet + vocoder — moderate CPU cost expected | CPU supported (Docker CPU image referenced), no CUDA hard requirement documented | Python library/CLI wrapper, no single binary | MIT-family (verify per-file) | Active, pushed 2025-02 | Medium |
| **AudioSR** | Audio super-resolution, any→48kHz, general audio (not just speech) | High quality (latent diffusion + HiFi-GAN) | Diffusion-based — multi-step sampling, GPU-oriented; no CPU RTF published, expect well below realtime on CPU | No confirmed DirectML/ONNX build found | ComfyUI node / Python script, no standalone binary | Non-commercial-leaning research license (verify before shipping) | Active via community forks | High |
| **FlashSR** | One-step distilled diffusion audio SR, 16kHz→48kHz | Near-AudioSR quality, much faster | ~200-400x realtime **on GPU** (community fork claim); no CPU numbers published, but single-step diffusion is the most CPU-plausible of the diffusion SR models | No confirmed DirectML/ONNX build found | Python script (`FlashSR_Inference`), no packaged binary | Research code, check per-repo license | New (paper Jan 2025), early-stage tooling | High |
| **Apollo** | Restore lossy/MP3-compressed music to near-lossless | Outperforms SR-GAN baselines per paper | RTF benchmarked at 44.1kHz on both CPU/GPU in the paper (compact model, "high computational efficiency" claimed) but no public number for AMD/no-GPU path found here | Trained on 8x RTX 4090; no confirmed AMD/DirectML build | Python script + Colab notebook, no packaged binary | **CC BY-SA 4.0** — copyleft, share-alike required for redistribution/derivatives | Active (ICASSP 2025 paper), community Colab wrapper | High, plus license friction for a closed pipeline |
| **ffmpeg `arnndn` (RNNoise)** | Lightweight built-in denoise, baseline comparison | Lower than DeepFilterNet but real-time-trivial | Extremely fast, negligible CPU cost, no model download beyond the `.rnnn` file | Yes — it's already inside the FFmpeg binary Upflow vendors | Yes, it's an ffmpeg filter, zero new dependency | LGPL/GPL (ffmpeg's own) | Mature, stable | **Lowest possible** — one `-af arnndn=model=...` flag on the existing ffmpeg call |

Verified AMD/no-CUDA claim for the top pick: DeepFilterNet's Rust CLI binary requirement was checked against the official `Rikorose/DeepFilterNet` README/release page (binary usage documented, "no CUDA/PyTorch runtime" needed for the Rust path) and independently against the crates.io `deep_filter` package description confirming the pure-Rust inference runtime — consistent across both sources.

### Topic 3 — Quality-vs-speed knobs per engine

| Engine | Knob | Values / effect |
|---|---|---|
| **Real-ESRGAN ncnn (already vendored, v0.2.0)** | `-n` model | `realesr-animevideov3` (fastest, anime-tuned, smaller net) vs `realesrgan-x4plus-anime` / `realesrgan-x4plus` (higher quality, slower) vs `realesrnet-x4plus` (less "GAN sharpen" artifacting) |
| | `-s` scale | 2 / 3 / 4 — higher scale = more compute, some models only support specific scales |
| | `-t` tile size | `0` = auto (fits VRAM); explicit values (e.g. 128/256/400) trade VRAM for speed — smaller tiles = less VRAM, more overhead/seams risk, slower; larger tiles = faster but more VRAM |
| | `-j` load:proc:save threads | e.g. `1:2:2` or `4:4:4` — tune for I/O vs GPU-bound workloads; more proc threads help when GPU is the bottleneck and CPU has headroom |
| | `-x` TTA mode | Test-time augmentation — notably higher quality, ~8x slower (runs 8 augmented passes and averages) |
| | precision | fp16 is the ncnn build default; no CLI flag to force fp32 in this binary — precision is fixed at build time |
| **RIFE ncnn (planned)** | `-m` model | Version selection (v2.3 baseline up to v4.25+); newer = better quality/speed tradeoff, `rife-anime` tuned for animation |
| | `-n` target frame count / `-s` timestep | Controls interpolation factor (2x, 3x, 4x, or arbitrary timestep) |
| | `-u` UHD mode | Optimizes for 4K+ content (adjusts internal tiling) — use for high-res sources |
| | `-x` / `-z` spatial/temporal TTA | Quality boost at multiplicative cost, same tradeoff pattern as Real-ESRGAN's `-x` |
| | `-j` load:proc:save | Same threading tradeoff as Real-ESRGAN |
| | `-g` GPU id | `-1` forces CPU fallback — useful as a last-resort compatibility mode, much slower |
| **DeepFilterNet CLI** | `--pf` postfilter | Slightly higher perceptual quality, small added cost |
| | model selection | DeepFilterNet2 (faster, default) vs DeepFilterNet3 (if bundled — better quality, more compute) |
| | `--compensate-delay` | Correctness knob, not a speed one — align output length |
| **ffmpeg arnndn** | model file | Different `.rnnn` models trade denoise aggressiveness vs artifacts; effectively free either way (real-time+) |
| **Demucs (if adopted later)** | `-n` model (e.g. `htdemucs` vs `htdemucs_ft` vs `mdx_extra`) | `_ft` (fine-tuned) variants: notably higher quality, ~4x slower (per-source fine-tuned passes) |
| | `--segment` | Shorter segments = less RAM, more overhead; longer = faster but more memory |

**Proposed unified quality↔speed slider (3 presets):**

| Preset | Real-ESRGAN | RIFE | Audio |
|---|---|---|---|
| **Fast** | `realesr-animevideov3`, tile auto, no TTA | RIFE v4.25-lite, no TTA, 2x factor | ffmpeg `arnndn` only |
| **Balanced** (default) | `realesrgan-x4plus-anime`, tile auto, no TTA | RIFE v4.25, no TTA, 2x-3x | DeepFilterNet2, no postfilter |
| **Best** | `realesrgan-x4plus`, `-x` TTA on | RIFE v4.25 + spatial TTA | DeepFilterNet2/3 + `--pf` postfilter |

### Topic 4 — Real-time gaming scaling/frame-gen tools

| Tool | Architecture | Open source | AMD-friendly | Feasible inside Upflow (Python/FastAPI, offline files)? |
|---|---|---|---|---|
| **Lossless Scaling** | Desktop-compositor screen capture → overlay-composited frame interpolation/upscale → present as a system overlay on top of the target window; requires Windowed/Borderless (no Exclusive Fullscreen) | No, paid Steam app | Vendor-agnostic (works on any GPU via general compute shaders) | No |
| **Magpie** | Captures a window via Windows Graphics Capture (or alternatives) → runs a DirectX 11+ effect chain (FSR, Anime4K, NIS, xBRZ, RAVU, NNEDI3, CRT shaders, etc.) → presents in real time, sub-frame latency budget | Yes, GitHub `Blinue/Magpie`, GPL-3.0, active (14k+ stars, commits within days) | Yes — DX11 feature-level based, runs on AMD | No |
| **AMD AFMF** | Driver-level, intercepts the live DirectX 11/12 swapchain between consecutive rendered frames, generates and inserts interpolated frames before present | No, proprietary driver feature | AMD-only by definition | No |

**Verdict (short):** Not feasible as a mode inside Upflow. All three operate on a **live swapchain/compositor with sub-frame latency budgets** (microseconds to a few ms), driven by DirectX capture/present loops — fundamentally different from Upflow's file-in/file-out subprocess model, which processes whole frames/files with no latency constraint. Building this would mean a separate native Windows app (C++/DirectX or C#/Win2D), not a Python/FastAPI extension. Recommendation: do not build it. If the user wants real-time upscaling for games, point them at **Magpie** (open source, GPL-3.0, actively maintained, already does exactly this) rather than reinventing it. Keep Upflow strictly file-based.

---

## 3. Concrete Recommendations for Upflow

**Frame interpolation (replaces "planned RIFE ncnn Vulkan" with a specific, verified pick):**
- Vendor **`TNTwise/rife-ncnn-vulkan`** (MIT license, GitHub releases page, latest `20250112`-style tagged builds continuing into 2026) instead of `nihui/rife-ncnn-vulkan` (same MIT license, functionally frozen since Jan 2025). CLI flags are compatible (`-i`/`-o`/`-m`/`-g`/`-j`/`-x`/`-z`/`-u`), so it's a drop-in replacement for the binary path only — integrate exactly like `RealEsrganNcnnEngine`, i.e. a new `RifeNcnnEngine(UpscaleEngine)`-style class calling `run_guarded_process`.
- Optional secondary/experimental model: `nihui/ifrnet-ncnn-vulkan` (MIT) as an alternate interpolation backend for A/B testing — same integration shape, low effort, but unmaintained since 2022 so treat as best-effort only.
- Do not chase GMFSS/FILM/EMA-VFI/VFIMamba/GIMM-VFI for v1 — none have a maintained Vulkan/DirectML deployment path; would require months of ONNX-DirectML porting work for uncertain gains over RIFE on anime/general content.

**Audio enhancement (new capability):**
- Primary: vendor the **`deep-filter` Rust CLI binary** from `Rikorose/DeepFilterNet` releases (MIT/Apache-2.0). Add a new `services/engines/deepfilternet.py` (or `services/audio/`) module following the exact same `available()` / async `run()` / `run_guarded_process` pattern as `RealEsrganNcnnEngine`. No CUDA, no GPU at all — runs on any CPU, RTF well under 1.0.
- Cheap complementary step already available for free: add an `arnndn` pass to the existing FFmpeg encode step as a "fast" preset option — zero new binaries, zero new licensing.
- Defer stem separation (Demucs) and audio super-resolution (AudioSR/FlashSR/Apollo) to a later phase: Demucs needs a Python/PyTorch runtime (heavier packaging than a single `.exe`, though `adefossez/demucs` — MIT, actively maintained — is the correct upstream if pursued); the SR/restoration models are GPU-diffusion-oriented with no verified AMD path and, for Apollo specifically, a CC BY-SA 4.0 license that needs legal review before bundling into a closed pipeline.

**Quality/speed slider:** implement the 3-tier preset table in Topic 3 as the concrete mapping — it only requires changing existing CLI flags/model names per engine, no new dependencies.

**Real-time gaming mode:** do not build. Reference Magpie for users who want that use case.

---

## 4. Realtime-Gaming Verdict (short)

Not feasible inside Upflow. Lossless Scaling, Magpie, and AMD AFMF all hook a live DirectX swapchain/desktop compositor with sub-frame latency requirements — a different problem class from file-based subprocess upscaling. Recommend Magpie (open source, GPL-3.0) as the tool to point users to; do not attempt to replicate this inside the Python/FastAPI app.

---

## 4b. AFMF/LSFG Deep-Dive (user follow-up)

User's claim: "there are AFMF versions that don't require motion vectorization — Lossless Scaling uses that (or a modified version)." Investigated against primary sources (AMD GPUOpen manuals, FidelityFX SDK repo, lsfg-vk source code).

### (a) Was the user right that these don't need motion vectors? YES — confirmed.

- **AFMF 1/2/2.1 (driver-level)** performs frame generation from **color frames only**, using AMD's optical-flow algorithm inside the driver. It never receives engine motion vectors — that is precisely why it works on any DX11/12 game without integration. Confirmed via AMD's AFMF release notes and the GPUOpen FidelityFX Optical Flow manual, which states the component is "based on AMD Fluid Motion Frames technology" and takes a **single color buffer as its only input**, outputting 8x8-block motion vectors + a scene-change flag.
- **LSFG 3.x (Lossless Scaling)** likewise needs no motion vectors: it captures two consecutive presented frames from the desktop compositor and runs a **proprietary ML optical-flow model** (compute shaders shipped inside `Lossless.dll`) to synthesize intermediates. Vendor-agnostic — runs on AMD/Intel/NVIDIA including iGPUs, no tensor cores required. Confirmed via losslessscaling.com/lsfg-3 and the lsfg-vk project, which drives those exact shaders on Vulkan.
- Important distinction the user's claim conflates: **in-engine FSR3 Frame Generation is the one that needs motion vectors**. Verified in the GPUOpen FidelityFX Frame Interpolation manual: its dispatch requires `dilatedDepth`, `dilatedMotionVectors`, `reconstructPrevNearDepth` (game-provided) *in addition to* optical-flow vectors — it is a hybrid that blends both motion sources per-pixel and **cannot run in a color+optical-flow-only mode**. AFMF is not "FSR3 FG without motion vectors"; it is a separate, driver-internal implementation sharing the optical-flow tech.

### Key technical findings

1. **FidelityFX SDK OpticalFlow component (MIT-licensed SDK) CAN be driven standalone.** Per the GPUOpen manual, `ffxOpticalflowContextCreate`/`Dispatch` operate outside a swapchain: input = one color buffer, output = R16G16_SINT motion vectors at 8x8-block granularity, DX12 or Vulkan backend, SM 6.2. So the *optical flow* part is offline-drivable in principle. However:
   - The SDK's **FrameInterpolation** component (the part that actually synthesizes the frame) hard-requires game depth + game motion vectors (see above) — you cannot assemble an "offline AFMF" from the SDK's public pieces alone; you would have to write your own warp/blend/occlusion pass around the raw 8x8 block flow.
   - 8x8-block MSAD-based flow (max 512px tracking) is engineered for real-time cost, not quality — categorically cruder than RIFE's learned per-pixel flow. For an offline tool where latency is irrelevant, it is strictly worse quality than RIFE at a cost advantage nobody offline needs.
   - **No existing offline/file tool wires FidelityFX OpticalFlow to video files** — searched GitHub/VapourSynth/ffmpeg ecosystems; nothing found. The offline interpolation ecosystem (SVPflow, VapourSynth-RIFE, rife-ncnn-vulkan) uses its own flow estimators.
2. **AFMF itself has no file API of any kind.** It is enabled per-game in Adrenalin, lives inside the display driver, and consumes frames in flight in the DX11/12 present path. There is no documented entry point that accepts arbitrary frames or files, in any AFMF version (1, 2, 2.1).
3. **lsfg-vk (PancakeTAS/lsfg-vk, GPL-3.0, ~4.6k stars, active — pushed June 2026)** re-implements the LSFG host on Vulkan for Linux by loading the **proprietary shaders/model out of `Lossless.dll`** from a purchased Steam copy of Lossless Scaling (the repo requires you to own the app; the DLL is the model payload). Primary use: a Vulkan layer hooking live games. Ecosystem: Decky plugin `xXJSONDeruloXx/decky-lsfg-vk` (active, 1.7k stars) streamlines it on Steam Deck/SteamOS — proof the LSFG shader payload is portable across OS and GPU vendors.
4. **Closest thing to a file-based LSFG path found anywhere:** `lsfg-vk-cli`'s `debug` subcommand ("Run lsfg-vk on a set of images") — verified in source (`lsfg-vk-cli/src/tools/debug.cpp`): it ingests a numbered folder of DDS frames, uploads pairs to Vulkan images, and runs the real LSFG generation pipeline with `--multiplier`/`--flow`/`--performance-mode`/`--dll <path to Lossless.dll>` flags. **But it discards the generated frames** — renders to Vulkan images and waits on semaphores, never reads back to disk; it exists to validate/benchmark the pipeline, not to process video. Extending it into a real file-out tool is feasible engineering-wise (add readback + image encode), but the result would be: GPL-3.0 host code + a hard runtime dependency on the user owning Lossless Scaling and locating `Lossless.dll` + Linux-oriented plumbing (POSIX fd-based external memory) to port to Windows — and LSFG's model is tuned for real-time cost, so it would still not beat RIFE offline on quality.

### (b) Does this change the offline-pipeline recommendation? NO.

Motion-vector-free ≠ file-processable. The user is right about the mechanics, but every member of this family is either locked inside a live present path (AFMF: driver swapchain; LSFG: compositor capture; lsfg-vk: Vulkan layer), missing its synthesis stage for standalone use (FidelityFX SDK), or legally/practically encumbered (LSFG shaders require owning `Lossless.dll`; lsfg-vk is GPL-3.0). Meanwhile RIFE ncnn Vulkan already does learned optical flow + frame synthesis, offline, file-in/file-out, MIT, on AMD — and with no real-time budget constraint, offline learned flow (RIFE-class) is higher quality than the real-time block/ML flow these tools use. **RIFE-class interpolation remains the only sensible offline path on AMD.** (If AMD ever ships a file-based AFMF API or someone builds a maintained FFX-OpticalFlow offline tool, revisit — nothing exists as of July 2026.)

### (c) Does this change the realtime-mode verdict? NO — it reinforces it.

The deep-dive confirms the whole family depends on live capture/present infrastructure (compositor capture, swapchain interception, Vulkan layer injection) that cannot exist inside a Python/FastAPI file server. Verdict stands: keep Upflow file-based; point real-time users at Magpie (upscaling) and Lossless Scaling/AFMF (frame gen) as external tools.

### Deep-dive sources
- https://gpuopen.com/manuals/fidelityfx_sdk/techniques/optical-flow/ (FFX OpticalFlow: color-only input, standalone context API, AFMF-based)
- https://gpuopen.com/manuals/fidelityfx_sdk/fidelityfx_sdk-page_techniques_frame-interpolation/ (FSR3 FrameInterpolation: requires game depth + motion vectors, hybrid blend)
- https://github.com/GPUOpen-LibrariesAndSDKs/FidelityFX-SDK (MIT samples/SDK, active, pushed 2026-06)
- https://www.amd.com/en/resources/support-articles/release-notes/RN-RAD-WIN-23-30-AFMF-TECH-PREVIEW.html (AFMF driver-level, per-game toggle, no file API)
- https://losslessscaling.com/lsfg-3/ (LSFG 3: ML optical flow on captured frames, no motion vectors, vendor-agnostic)
- https://github.com/PancakeTAS/lsfg-vk (GPL-3.0, active 2026-06, requires owning Lossless Scaling / `Lossless.dll`)
- lsfg-vk source: `lsfg-vk-cli/src/tools/debug.cpp` and `main.cpp` (file-in image-folder debug mode, no file-out)
- https://github.com/xXJSONDeruloXx/decky-lsfg-vk (Decky/SteamOS plugin, active 2026-07)

---

## 5. AI Subtitling (roadmap)

Goal: Upflow as a complete anime-watching suite — transcribe (JP), time, translate (JP→ES/EN), and mux subtitles during the existing FFmpeg encode. Constraints unchanged: local, Windows, no CUDA; CPU acceptable, Vulkan/DirectML bonus.

### 5.1 Speech-to-text candidates

| Candidate | Quality (JP) | Speed on CPU (no CUDA) | AMD/no-CUDA GPU path | Standalone CLI | License | Maturity | Integration effort |
|---|---|---|---|---|---|---|---|
| **whisper.cpp (ggml-org)** | Full Whisper family incl. large-v3 / large-v3-turbo; quantized Q5/Q8 with minor quality loss | large-v3 ≈0.5–1x realtime on a modern 8-core; medium ≈2x; small ≈4–6x (estimates; quantization helps). **With Vulkan on AMD: ~8x realtime with large model** (Phoronix, whisper.cpp 1.8.3, RX 9070 XT, real 45-min episode; ~12x speedup vs CPU on AMD iGPU) | **Yes — first-class Vulkan backend** (cross-vendor, RX 6xxx/7xxx OK). Official Windows release zips are CPU/BLAS/CUDA only; Vulkan needs a source build (MSVC + Vulkan SDK, documented) or community prebuilt (`jerryshell/whisper.cpp-windows-vulkan-bin` v1.0.0 2025-08; `DomoticX/whisper.cpp-windows-vulkan`) | **Yes** — `whisper-cli.exe`, single binary + ggml model file | MIT | Very active — v1.9.1 released 2026-06-19 | **Low** — same vendored-binary/subprocess pattern as Real-ESRGAN |
| **faster-whisper (SYSTRAN, CTranslate2)** | Same Whisper weights, same accuracy | int8 CPU ≈4x faster than openai/whisper (official claim); large-v3 ≈1–2x realtime on 8-core, medium ≈3–4x | **CPU-only on AMD Windows** — CTranslate2 GPU = CUDA only, no Vulkan/DirectML | Via **Purfview/whisper-standalone-win** (Faster-Whisper-XXL standalone `.exe`, no Python, 3.1k stars, last release 2025-11); emits SRT directly, built-in VAD | MIT (faster-whisper); Purfview builds free, check per-release terms | Active | Low (Purfview exe) / Medium (Python lib) |
| **whisperX** | Whisper + wav2vec2 forced alignment: word-level timestamps <100ms + speaker diarization | CPU-capable but CUDA-oriented; slower than faster-whisper on CPU (extra alignment pass) | CPU only on AMD Windows | No — Python package; diarization needs gated pyannote weights (free, HF account) | BSD-2-Clause (one source cites BSD-4; verify at vendoring time) | Active (v3.8.6, 2025-05) | Medium-High; only if per-word karaoke timing or speaker labels are wanted |
| **distil-whisper** | English-only distillations — **not useful for JP source audio** | ~6x faster than large | CPU | Runs under whisper.cpp/faster-whisper | MIT | Active | N/A for JP |
| **kotoba-whisper / anime-whisper (litagin)** | **JP-specialized**: anime-whisper (fine-tuned on anime speech, based on kotoba-whisper-v2.0) reports **CER 13.0% vs 16.5% for stock large-v3** on an anime test set | kotoba-whisper is distil-type — faster than large-v3 | HF PyTorch weights; convertible to CTranslate2 and ggml — one-time conversion effort | Not directly; after conversion inherits whisper.cpp/faster-whisper CLI | HF model cards (kotoba-whisper Apache-2.0 family; verify anime-whisper card) | Community, active 2025 | Medium (conversion), then Low |

**Demucs/vocal-separation pre-pass:** evidence mixed but positive for noisy tracks — music-source-separation + Whisper research shows separation reduces WER when dialogue is buried under score/SFX, and anime Blu-ray dialogue-extraction workflows report large WER drops with separation + large-v3. Demucs (MIT, `adefossez/demucs`, Topic 2) fits as an optional "noisy audio" toggle before STT — not default (minutes-scale on CPU, slowest stage in the chain).

### 5.2 Subtitle file generation

- **whisper.cpp** emits subtitles natively: `-osrt` (SRT), `-ovtt` (VTT), plus JSON/LRC/karaoke (`-owts`) — verified in the official CLI README. Zero post-processing for the basic flow.
- **Purfview Faster-Whisper-XXL** writes SRT directly with built-in VAD, which curbs Whisper's hallucinated timestamps during OP/ED music — a real problem on anime episodes.
- ASS styling (fonts/positioning) is a mechanical SRT→ASS conversion at mux time; FFmpeg handles either.
- Muxing in the existing encode: MKV `-c:s srt`/`-c:s ass` (soft subs, preferred), MP4 `-c:s mov_text`, or hard-burn via `-vf subtitles=file.ass`.

### 5.3 Translation JP→ES/EN (local)

| Option | Path | Anime-dialogue reality check | License | Effort |
|---|---|---|---|---|
| Whisper built-in `translate` (`-tr`) | JP audio → **EN text only** (X→EN is the only direction Whisper supports) | Decent EN; loses honorifics/nuance; free one-pass | MIT | Zero (a flag) |
| **MADLAD-400 3B-MT** via llama.cpp/CTranslate2 | Direct JP→ES (419 languages) | Benchmarks favor MADLAD over NLLB on Japanese (EN→JA: 24.9 spBLEU vs NLLB-3.3B 16.0); still literal-leaning on casual speech | **Apache 2.0** (redistribution-safe) | Medium — GGUF quant on llama.cpp CPU, subprocess-friendly |
| NLLB-200 (3.3B/1.3B) | Direct JP→ES | Good on high-resource pairs, weaker than MADLAD on JP per available benchmarks | **CC-BY-NC 4.0 — non-commercial**; avoid bundling | Medium |
| Local LLM (Qwen/Gemma class via llama.cpp) | JP→ES with context prompt | Best nuance/register handling (honorifics, tone) but slowest; needs prompt engineering + hallucination guards | Model-dependent | High |
| Cloud API (DeepL/GPT), opt-in toggle | JP→ES | Highest quality; breaks local-only default — optional only | Per-service | Low |
| Chain: `-tr` (JP→EN) → EN→ES MT | Two hops | Compounding loss; noticeably worse than direct JP→ES — fallback only | — | Low |

Recommended: whisper.cpp `-tr` for EN subs (free), MADLAD-400 3B GGUF on llama.cpp CPU for ES subs (Apache 2.0 — avoids NLLB's non-commercial trap), local-LLM path as a later "quality" tier.

### 5.4 Existing integrated tools to learn from / wrap

| Tool | What it solves | License | Notes for Upflow |
|---|---|---|---|
| **subsai** (absadiki/subsai) | Web-UI + CLI + Python pkg wrapping whisper.cpp/faster-whisper/whisperX/stable-ts behind one API | GPL-3.0 | Best multi-backend architecture reference; GPL blocks embedding code — wrap via subprocess or copy the *design* only |
| **Buzz** (chidiwilliams/buzz) | Desktop GUI transcription (Whisper/whisper.cpp/faster-whisper) | MIT | Code reusable; GUI-oriented |
| **pyvideotrans** (jianchang512) | Full video translate+dub pipeline (ASR→MT→TTS) — closest to the "complete suite" vision | GPL-3.0 | Feature-checklist gold mine (their docs recommend faster-whisper large-v3); GPL — study, don't embed |
| **Subtitle Edit** | Editor with Whisper integration (CPP, Purfview XXL, whisperX engines); has an open issue requesting Vulkan whisper.cpp builds for AMD | MIT (per current GitHub license metadata) | Validates the exact engine choices above; its Purfview integration is the proven Windows-no-CUDA route |

No open-source tool does upscale+interpolate+subtitle in one pipeline — the subtitle stage is Upflow's differentiator, and it composes cleanly from vendored binaries.

### 5.5 Integration sketch for Upflow

1. **Vendor `whisper-cli.exe`** (whisper.cpp) + a ggml model (`large-v3-turbo-q5_0` default, `medium-q5_0` fast tier) under `vendor/whisper/`, mirroring `vendor/realesrgan/`. Start with the official CPU x64 release zip (`whisper-bin-x64.zip`, works everywhere); add a Vulkan build (self-compiled once, or community bin) as the "AMD GPU accel" upgrade — same CLI, ~8x realtime on RX 7800 XT-class for the large model.
2. **New engine class** (`app/services/engines/whisper_cpp.py`) following `RealEsrganNcnnEngine`: `available()` checks binary+model; async `run()` builds `[exe, -m, model, -f, wav, -l, ja, -osrt, -of, outbase]` (+`-tr` for EN) through `run_guarded_process`.
3. **Pipeline position:** extract audio once with FFmpeg (`-vn -ar 16000 -ac 1` WAV) → run STT **in parallel with** upscale/interpolation (STT is CPU-bound, video is GPU-bound — no contention) → optional MADLAD JP→ES pass over the SRT text → mux SRT/ASS at the final encode (`-c:s ass` for MKV).
4. **Effort:** STT+SRT+mux (JP/EN subs) ≈ Low-Medium — 1–2 days reusing existing engine/subprocess/vendor patterns. JP→ES local translation adds Medium (llama.cpp + MADLAD GGUF vendoring + SRT rewrite step). whisperX word timing/diarization: defer.

**Top pick verified no-CUDA Windows path (primary sources):** whisper.cpp is MIT, v1.9.1 (2026-06-19), and its GitHub release assets include `whisper-bin-x64.zip` (pure CPU, no CUDA dependency) — verified directly against the release-asset list via the GitHub API. The Vulkan backend for AMD is documented in the repo and independently corroborated by Phoronix AMD benchmarks and Subtitle Edit's AMD-Vulkan integration request (#10200).

### 5.6 Subtitling sources
- https://github.com/ggml-org/whisper.cpp (MIT, backends, release assets incl. whisper-bin-x64.zip)
- https://github.com/ggml-org/whisper.cpp/blob/master/examples/cli/README.md (`-osrt`, `-ovtt`, `-tr` flags)
- https://www.phoronix.com/news/Whisper-cpp-1.8.3-12x-Perf (Vulkan on AMD: ~8x realtime large model)
- https://github.com/jerryshell/whisper.cpp-windows-vulkan-bin and https://github.com/DomoticX/whisper.cpp-windows-vulkan (community Vulkan Windows builds)
- https://github.com/SubtitleEdit/subtitleedit/issues/10200 (Vulkan whisper.cpp for AMD — corroborates the AMD path)
- https://github.com/SYSTRAN/faster-whisper (CT2, int8 CPU, 4x claim, MIT)
- https://github.com/Purfview/whisper-standalone-win (standalone Windows exes, last release 2025-11)
- https://github.com/m-bain/whisperX (word timestamps + diarization, v3.8.6 2025-05)
- https://model.aibase.com/models/details/1915701890467389442 and https://dataloop.ai/library/model/litagin_anime-whisper/ (anime-whisper: CER 13.0% vs 16.5% large-v3)
- https://huggingface.co/datasets/joujiboi/japanese-anime-speech (JP anime speech dataset context)
- https://arxiv.org/pdf/2506.15514 (music source separation improves Whisper transcription)
- https://picovoice.ai/blog/open-source-translation/ (NLLB CC-BY-NC vs MADLAD Apache 2.0)
- https://insiderllm.com/guides/best-local-llms-translation/ (MADLAD vs NLLB JP benchmarks, GGUF/llama.cpp CPU path)
- https://github.com/absadiki/subsai (GPL-3.0), https://github.com/chidiwilliams/buzz (MIT), https://github.com/jianchang512/pyvideotrans (GPL-3.0), https://github.com/SubtitleEdit/subtitleedit (MIT)

---

## 6. Sources

**Frame interpolation / frame generation:**
- https://github.com/nihui/rife-ncnn-vulkan
- https://github.com/nihui/rife-ncnn-vulkan/releases
- https://github.com/TNTwise/rife-ncnn-vulkan
- https://github.com/TNTwise/REAL-Video-Enhancer (archived 2026-07-13, historical reference for model list)
- https://github.com/nihui/ifrnet-ncnn-vulkan
- https://github.com/Justin62628/gmfss-ncnn-vulkan
- https://github.com/Justin62628/gmfss-to-ncnn
- https://github.com/hzwer/Practical-RIFE
- https://doc.svfi.group/en/pages/about-svfi/
- https://github.com/google-research/frame-interpolation
- https://film-net.github.io/
- https://github.com/GSeanCDAT/GIMM-VFI
- https://arxiv.org/abs/2407.08680 (GIMM-VFI)
- https://arxiv.org/pdf/2407.02315 (VFIMamba)
- https://github.com/CMLab-Korea/Awesome-Video-Frame-Interpolation (AceVFI survey)
- https://arxiv.org/abs/2506.01061 (AceVFI)
- https://gpuopen.com/amd-fsr-framegeneration/
- https://gpuopen.com/manuals/fidelityfx_sdk/techniques/super-resolution-interpolation/
- https://github.com/NVIDIA-RTX/Streamline/blob/main/docs/ProgrammingGuideDLSS_G.md
- https://www.pcgamer.com/amd-fluid-motion-frames-finally-goes-official-driver-based-frame-generation-for-any-dx1112-game/
- https://www.xda-developers.com/amd-fluid-motion-frames/

**Audio enhancement:**
- https://github.com/Rikorose/DeepFilterNet
- https://crates.io/crates/deep_filter/0.1.6
- https://arxiv.org/abs/2305.08227 (DeepFilterNet)
- https://arxiv.org/abs/2205.05474 (DeepFilterNet2)
- https://github.com/resemble-ai/resemble-enhance
- https://github.com/resemble-ai/resemble-enhance/issues/16
- https://github.com/resemble-ai/resemble-enhance/issues/17
- https://github.com/daswer123/resemble-enhance-windows
- https://github.com/haoheliu/voicefixer
- https://github.com/facebookresearch/demucs (archived)
- https://github.com/adefossez/demucs (active)
- https://github.com/JusperLee/Apollo
- https://arxiv.org/abs/2409.08514 (Apollo)
- https://github.com/Saganaki22/ComfyUI-AudioSR
- https://arxiv.org/abs/2309.07314 (AudioSR)
- https://github.com/jakeoneijk/FlashSR_Inference
- https://github.com/ysharma3501/FlashSR
- https://arxiv.org/abs/2501.10807 (FlashSR)
- https://ffmpeg.org/ffmpeg-filters.html (arnndn)

**Quality/speed knobs:**
- https://github.com/nihui/rife-ncnn-vulkan (CLI options)
- https://github.com/xinntao/Real-ESRGAN-ncnn-vulkan (CLI options, currently vendored at v0.2.0 in this repo)

**Real-time gaming scaling:**
- https://github.com/Blinue/Magpie
- https://github.com/Blinue/Magpie/wiki/Comparison-of-capture-methods
- https://losslessscaling.com/
- https://windowsforum.com/threads/lossless-scaling-2026-guide-lsfg-frame-gen-upscaling-without-gpu-lock-in.435159/
- https://www.amd.com/en/resources/support-articles/release-notes/RN-RAD-WIN-23-30-AFMF-TECH-PREVIEW.html

**AMD/DirectML platform verification:**
- https://onnxruntime.ai/docs/execution-providers/DirectML-ExecutionProvider.html
- https://gpuopen.com/learn/onnx-directlml-execution-provider-guide-part1/
- https://gpuopen.com/learn/onnx-directlml-execution-provider-guide-part2/
