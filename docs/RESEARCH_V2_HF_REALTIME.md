# Upflow v2 Research — HF Model Runtime (any GPU/NPU) + Realtime Upscale/Frame-Gen Module

Date: 2026-07-14 · Scope: Windows-first, AMD RX 7800 XT primary target, must also run on NVIDIA/Intel.

## Executive Summary

1. **DirectML is legacy.** Microsoft's own README says DirectML is "in maintenance mode" and points to **Windows ML** as the successor for Win11 24H2+. Build against Windows ML's EP catalog, not raw DirectML, going forward.
2. **No ONNX Runtime Vulkan EP exists** (open GH issues since 2021, unimplemented). Vulkan-based cross-vendor inference means **ncnn** or **IREE**, not ONNX Runtime.
3. **OpenVINO GPU/NPU acceleration is Intel-only** — confirmed on official docs. It has zero path to an AMD RX 7800 XT; CPU-only fallback on AMD.
4. **torch-directml is unmaintained-in-practice** (DirectML repo itself says "issues and samples...will not be updated"); do not build v2's "any .pth model" pipeline on it.
5. **Spandrel (MIT/permissive) + ONNX export, not torch-directml**, is the practical path to "paste an HF/.pth model and run it" on AMD.
6. **A desktop RX 7800 XT machine has no NPU** unless the CPU is a Ryzen AI 300/400-series or Intel Core Ultra chip with an integrated NPU — confirmed: NPU silicon ships only in APU/mobile-derived CPU dies, never on the discrete GPU itself.
7. **Windows ML now exposes a real Python device-picker API** (`ort.get_ep_devices()` + `add_provider_for_devices()`, GA in current ORT/Windows ML) — this directly answers the "CPU/GPU0/GPU1/NPU picker" requirement.
8. **Magpie already runs custom ONNX SR models today** on an experimental branch (`onnx-preview2`), with real user-reported AMD numbers: RX 9070 XT ~70-80 fps at 1080p→4K via DirectML. This makes Magpie the strongest reuse candidate for Topic B, not a from-scratch build.
9. **No open-source, Windows-native, driver-independent realtime frame-generation network exists.** `lsfg-vk` (MIT) reimplements Lossless Scaling's proprietary optical-flow model but is **Linux-only** (Vulkan+DXVK hook). AMD AFMF is driver-level and not controllable by a third-party app.
10. AMD's own **FidelityFX Frame Interpolation SDK is open source (MIT)** but requires **engine-level motion-vector integration** — unusable for a generic "any window" tool like Upflow/Magpie.
11. **MVP for Topic B is real and cheap**: fork/extend Magpie's existing ONNX-effect branch for AMD-optimized SR shaders (Anime4K/RTMoSR/compact-ESRGAN), skip frame generation entirely for v1.
12. Full comparison tables, exact package names/licenses, and a concrete recommended stack for both topics are below.

Report path: `C:\Users\santi\.openclaw\workspace\image-upscaler-amd\.superpowers\sdd\research-v2-hf-realtime.md`

---

## Topic A — Running HF Models on Any GPU/NPU (No CUDA)

### A.1 Runtime comparison

| Runtime | Vulkan/DX backend | Device enumeration/selection from Python | Maturity (2026) | Python package | License |
|---|---|---|---|---|---|
| **ONNX Runtime + DirectML EP** | DirectX 12 (DXGI adapters) | `device_id` = DXGI adapter index (`IDXGIFactory::EnumAdapters` order); **no built-in name→id mapping API** — must cross-reference via WMI/`pywin32`/DXGI yourself. Newer ORT adds `ort.get_ep_devices()` returning `OrtHardwareDeviceType` (GPU/NPU/CPU) + vendor id, usable with `add_provider_for_devices()`. | Mature, but Microsoft has frozen new features ("maintenance mode") | `onnxruntime-directml` | MIT (ORT) / MIT (DirectML) |
| **ONNX Runtime + Windows ML EP catalog** (2026 successor) | Wraps DirectML (legacy, always present) + dynamically-installed vendor EPs: `MIGraphXExecutionProvider` (AMD GPU, ROCm-based), `VitisAIExecutionProvider` (AMD Ryzen AI NPU), `OpenVINOExecutionProvider` (Intel CPU/GPU/NPU), `NvTensorRtRtxExecutionProvider` (NVIDIA RTX 30xx+), `QNNExecutionProvider` (Qualcomm Hexagon NPU) | Same `GetEpDevices()`/`ort.get_ep_devices()` API, filter by `HardwareDevice.Type` (NPU/GPU/CPU) and `EpName`, or use `SetEpSelectionPolicy(MAX_PERFORMANCE / MAX_EFFICIENCY / PREFER_NPU)` for automatic picking | New (Win11 24H2+ only), actively developed by MS through 2026 (weekly EP point releases visible in docs) | `onnxruntime-windowsml` (pip) or `wasdk-Microsoft.Windows.AI.MachineLearning` | MIT (ORT core); individual EPs carry vendor licenses (AMD Ryzen AI license, Intel OBL, NVIDIA SLA, Qualcomm QNN license) — **must be reviewed per EP before shipping** |
| **ONNX Runtime + Vulkan EP** | — | — | **Does not exist.** Open feature request since 2021 (`microsoft/onnxruntime#7433`, `#10603`, `#21917`), never implemented. ORT's only Vulkan-adjacent path is the WebGPU EP (browser-targeted). | n/a | n/a |
| **ncnn (Vulkan compute)** | Vulkan 1.x | `ncnn::get_gpu_count()`, `net.set_vulkan_device(index)`; Python via `ncnn-vulkan` package, `ncnn.get_gpu_device(ncnn.get_default_gpu_index())`. Defaults to discrete GPU > integrated > other. | Very mature for this exact use case — it is what Upscayl/rife-ncnn-vulkan/waifu2x-ncnn-vulkan already ship in production on AMD | `ncnn` / `ncnn-vulkan` | BSD-3-Clause |
| **IREE + Vulkan HAL** | Vulkan 1.3 (requires timelineSemaphore, scalarBlockLayout, synchronization2) | `iree.runtime.system_setup.query_available_drivers()` lists drivers (`vulkan`, `cuda`, `local-task`...); GPU arch is targeted at compile time via `--iree-vulkan-target=gfx1100` (RX 7900 XTX-class) style codes, no adapter-index picker at runtime beyond driver selection | Compiler-driven (AOT compile to `.vmfb`), more engineering overhead than ncnn/ORT; not a drop-in "load any .onnx" story | `iree-runtime`, `iree-compiler` | Apache-2.0 |
| **OpenVINO** | Intel oneAPI/Level-Zero (GPU), proprietary NPU driver | `ov.Core().available_devices` lists `CPU`/`GPU`/`NPU`; clean Python API | Very mature **for Intel only** | `openvino` | Apache-2.0 |
| **DirectML (standalone, no ORT)** | DirectX 12 | Same DXGI adapter-index model as above, raw `IDMLDevice` | Maintenance mode per official README | `torch-directml` (PyTorch backend) / C++ headers | MIT |
| **llama.cpp/ggml-style vision backends** | Vulkan (via `ggml-vulkan`) | No standardized SR/upscale model support in ggml today | Immature for image SR specifically — ggml's vision support is CLIP/LLaVA-encoder-oriented, not SR architectures | n/a | MIT |

**Key verified facts:**
- OpenVINO GPU/NPU acceleration is Intel-hardware-exclusive; AMD falls back to CPU-only. (docs.openvino.ai, official supported-devices page)
- DirectML README, verbatim: *"⚠️ DirectML is in maintenance mode ⚠️"* and *"If your PC runs Windows 11, version 24H2 (build 26100) or later, consider using Windows ML."* (github.com/microsoft/DirectML)
- Windows ML's `GetEpDevices()`/`ort.get_ep_devices()` + `AppendExecutionProvider_V2`/`add_provider_for_devices()` is a **real, current, cross-language (C#/C++/Python) API** for enumerating and filtering by `OrtHardwareDeviceType.{NPU,GPU,CPU}` — this is the concrete answer to "list adapters and pick GPU vs NPU vs CPU programmatically." (learn.microsoft.com/windows/ai/new-windows-ml/select-execution-providers)
- AMD's Windows ML GPU path is `MIGraphXExecutionProvider` (ROCm-based), separate from the AMD NPU path `VitisAIExecutionProvider` (Ryzen AI only). Both are currently AMD-specific closed vendor packages distributed through the Windows ML EP catalog, each with its own license (Ryzen AI Licensing terms) that must be reviewed before bundling.

### A.2 NPU reality check (2026)

| NPU | Where it ships | Usable via | Confirmed |
|---|---|---|---|
| AMD XDNA/XDNA2 | Ryzen AI 300-series mobile, and **new in 2026**: Ryzen AI 400-series **desktop APUs** (AM5, Zen5 + RDNA3.5 iGPU + XDNA2 NPU, 50-60 TOPS) | VitisAI EP (Windows ML), Ryzen AI Software / ROCm/Vitis AI stack | Yes, but Ryzen AI 400 desktop parts are **OEM-only at launch, no boxed retail SKU** |
| Intel NPU | Core Ultra (Meteor Lake+) laptops and desktops | OpenVINO NPU device, OpenVINO EP in Windows ML | Yes |
| Qualcomm Hexagon | Snapdragon X Elite/Plus (ARM64 Windows) | QNN EP (Windows ML / ONNX Runtime), needs QDQ-quantized ONNX; quantization tooling is x64-only, must cross-quantize | Yes |

**Can a desktop RX 7800 XT box have an NPU at all?** Confirmed **no**, unless the motherboard's CPU is itself an NPU-equipped part (Ryzen AI 300/400 or Intel Core Ultra) — NPU silicon is fused onto the CPU die, never onto a discrete GPU. An AM5 board with, e.g., a Ryzen 7 7800X3D + RX 7800 XT has **zero NPU**. Only if a user swaps in a Ryzen AI 400-series desktop APU (once retail SKUs exist) alongside their RX 7800 XT would both an NPU and the dGPU coexist on one machine — and even then AMD's own material frames Ryzen AI desktop APUs as pairing *with* discrete GPUs being the traditional desktop model, i.e. the combination is plausible but currently rare/OEM-locked.

**Implication for Upflow**: NPU support is a "nice to have for a subset of users," never the primary path for the RX 7800 XT target. Design the device picker so NPU is one optional entry, GPU (DirectML/MIGraphX/ncnn-Vulkan) is the default, CPU is the fallback.

### A.3 Model formats and "paste an HF repo id" pipeline

**What ships where:**
- **Real-ESRGAN**: official `.pth` (BSD-3-Clause, xinntao). Community ONNX ports exist on HF (`bukuroo/RealESRGAN-ONNX`, `qualcomm/Real-ESRGAN-x4plus`, `qualcomm/Real-ESRGAN-General-x4v3`, `facefusion/models`) — but these are **third-party conversions**, not official releases; quality/opset parity isn't guaranteed.
- **SwinIR / HAT / DAT**: research-repo `.pth` checkpoints; ONNX export is possible via `torch.onnx.export` (community has done it for SwinIR — `rocca/swin-ir-onnx` on HF) but there is **no official/first-party ONNX release** for any of these three as of this research.
- **4x-UltraSharp and the broader community ESRGAN "OpenModelDB" zoo**: almost universally `.pth`, SRVGGNetCompact/ESRGAN-family architectures, designed to be loaded by **Spandrel**, not shipped as ONNX.
- **Spandrel** (`chaiNNer-org/spandrel`, MIT for the core package, `spandrel_extra_arches` MIT-licensed wrapper around architectures with non-permissive per-arch licenses): auto-detects architecture + hyperparameters from the checkpoint and returns a normal `torch.nn.Module`. Loads `.pth`, TorchScript `.pt`, some `.ckpt`, and `.safetensors`. **It does not load `.onnx`** — Spandrel is a PyTorch-checkpoint loader, not an inference runtime.

**Practical pipeline for "paste an HF repo id / upload a checkpoint":**

| Option | How it works | Verdict for Upflow |
|---|---|---|
| (a) Spandrel + torch-directml | Spandrel loads `.pth`→`nn.Module`, `.to(torch_directml.device())` for inference | **Not recommended.** torch-directml is effectively frozen (DirectML repo explicitly says its issues/samples "will not be updated"); betting v2's flagship feature on it is a maintenance liability. |
| (b) Spandrel + on-the-fly ONNX export + onnxruntime-directml (or Windows ML EP) | Load with Spandrel (validates architecture, gets clean `nn.Module` + metadata), `torch.onnx.export(...)` once per model to a cache dir, then run inference through `onnxruntime` with whatever EP the user's device picker selected | **Recommended.** Reuses Spandrel's huge "which of these 50 obscure GAN architectures is this .pth" solved problem, but hands the actual inference to the far more portable/maintained ONNX Runtime + DirectML/Windows ML/ncnn stack. Export happens once (cached), not per-inference. |
| (c) onnxruntime-directml direct, .onnx-only | If the user already has/paste an `.onnx` model (increasingly common — Qualcomm AI Hub, HF ONNX community ports), skip Spandrel entirely | **Also recommended as the fast path** — this is strictly the cheapest integration and should be the "first thing that works." |

Net: **(b) + (c) together**, Spandrel-as-architecture-detector feeding a one-time ONNX export, is the correct v2 architecture. Do not adopt torch-directml as a standing runtime dependency.

**chaiNNer / Upscayl architecture study:**
- **Upscayl** = Electron GUI + `upscayl-ncnn` backend (C++/CMake, `ncnn` + Vulkan compute). Model-agnostic in practice by shipping several pre-converted `.param`/`.bin` ncnn models (Real-ESRGAN, Remacri, UltraMix, Ultrasharp) — i.e., it sidesteps the "arbitrary .pth" problem by pre-converting a curated model set, not by solving generic runtime loading. **Reusable as a reference for the ncnn-Vulkan backend integration pattern**, not for the "any HF model" story.
- **chaiNNer** = the actual generic loader; it *is* Spandrel's origin project (Spandrel was extracted from chaiNNer specifically so other tools could reuse chaiNNer's "load 50+ obscure PyTorch SR architectures" logic without pulling in chaiNNer's whole node-graph UI). This confirms Spandrel is the right dependency to pull in directly rather than re-deriving chaiNNer's model-detection logic.

### A.4 Recommendation — concrete stack

**Runtime/loader stack:**
1. `onnxruntime` (CPU always available) + `onnxruntime-directml` (broadest GPU coverage today, AMD/NVIDIA/Intel/Qualcomm, MIT) as the baseline shipped dependency.
2. Add `onnxruntime-windowsml` (or the `Microsoft.Windows.AI.MachineLearning` EP catalog) as an **optional, dynamically-probed** path on Win11 24H2+ for users who want the newer AMD MIGraphX / NVIDIA TensorRT-RTX / Intel OpenVINO / Qualcomm QNN EPs — gate this behind an OS-version check, don't hard-require it.
3. `spandrel` + `spandrel_extra_arches` (MIT) as the checkpoint-format/architecture detector for user-supplied `.pth`/`.safetensors` models; export to ONNX once via `torch.onnx.export` (CPU-only PyTorch is enough for export — no torch-directml needed) and cache the `.onnx` alongside the source checkpoint.
4. `ncnn`/`ncnn-vulkan` kept as a **secondary backend**, not replaced — it's what already ships in Upflow v1 and remains the best-tested Vulkan path for AMD; keep it for pre-packaged models (Real-ESRGAN, Anime4K-style nets) and use the ONNX/DirectML path specifically for the new "bring your own HF/.pth model" feature.
5. Device picker UI backed by `ort.get_ep_devices()`: group by `OrtHardwareDeviceType` (GPU/NPU/CPU), label with vendor + device name, default to the first discrete GPU, expose NPU only if present.

**Integration effort estimate (subprocess-or-Python FastAPI app):**
- Straight-Python integration (no subprocess) is viable: `onnxruntime` and `spandrel` are both pure pip installs with no separate service process needed. Effort: **small** (days, not weeks) for the loader + export pipeline; most of the work is UI (model picker, progress reporting for the one-time export) and cache management, not the runtime plumbing itself.
- Risk area: PyTorch itself (needed for Spandrel + `torch.onnx.export`) is a heavy dependency (~2-4 GB with CUDA wheels) — **use the CPU-only PyTorch wheel** (`pip install torch --index-url .../cpu`) since export doesn't need GPU acceleration, keeping the install footprint reasonable.

**VRAM/tiling considerations:**
- Both ncnn and onnxruntime-directml SR models need **tiling** for large images/video frames — an untiled 4x ESRGAN-family pass on a 4K frame can exceed 8-12 GB VRAM depending on architecture (attention-based nets like SwinIR/HAT/DAT are much heavier per-pixel than compact CNNs). Upflow v1's existing ncnn tiling logic should be generalized and reused for the ONNX path (same tile-and-stitch approach, different backend call underneath).
- Windows ML's `MAX_EFFICIENCY` policy will silently pick an NPU if present — NPUs typically have far less usable memory/bandwidth than a dGPU for large tiles, so if adding automatic device-policy selection, cap tile size more aggressively when the resolved device type is NPU.

---

## Topic B — Realtime Upscaling + Frame Generation Module (Requirements Study)

### B.1 Architecture requirements

**Capture:**

| API | Latency | Cross-GPU | Notes |
|---|---|---|---|
| DXGI Desktop Duplication | Lowest, full control | **No** — capturer must run on the same GPU as the display output | Best choice when Upflow's helper runs on the same GPU driving the display (the common case for a single-dGPU AMD box) |
| Windows.Graphics.Capture (WGC) | Slightly higher overhead (usually negligible) but needs HDR/HAGS tuned right for parity | **Yes** — OS abstracts the GPU, works natively with D3D11/D3D12 | Better default for laptops/hybrid-graphics systems; more "modern Windows" blessed API |

Magpie itself hedges by supporting **both** (`GraphicsCapture`, `DesktopDuplication`, plus `GDI` and `DwmSharedSurface` for edge cases) — that multi-backend fallback list is itself a strong argument for reusing Magpie's capture layer rather than re-deriving it.

**Processing:** D3D11 compute shaders (what Magpie uses — 91.6% HLSL codebase) is the proven approach; a 1080p→4K pass at 60-120 fps needs each shader/inference pass to fit in a ~8-16 ms (60fps) or ~4-8ms (120fps) budget, which is why compact/shader-based upscalers (Anime4K, RTMoSR) matter more here than heavyweight transformer SR nets (SwinIR/HAT/DAT) — those are not realtime-viable at all at this frame budget.

**Present:** Magpie uses a borderless overlay window approach (not a swapchain hook like SpecialK). This avoids the fragility/anti-cheat risk of hooking into a game's own DirectX swapchain, at the cost of not being usable for exclusive-fullscreen-only titles without a windowed/borderless mode.

### B.2 Magpie — is it a reusable base?

**Yes, directly, and more so than expected.** Verified facts:
- Open source, HLSL-effect pipeline (`MagpieFX` compute shaders), built-in Anime4K/FSR/CRT-shader effects, WinUI GUI, multi-capture-backend (GraphicsCapture/DesktopDuplication/GDI/DwmSharedSurface).
- **Custom ONNX model inference is already being built into Magpie** on the `onnx-preview2` branch/discussion (`Blinue/Magpie#1121`, active as of the last few months). Confirmed backends: **TensorRT** (NVIDIA, primary/fastest), **DirectML** (broad GPU coverage), and **experimental AMD MIGraphX** support (community-reported as currently behind TensorRT in speed).
- **Real user-reported AMD numbers on this branch**: RX 9070 XT achieving ~70-80 fps at 1080p→4K running `rewaifu_anime_rtmosr_gan_x2_fp16_op18.onnx` via DirectML; RTX 4080 ~57 fps; Intel Arc A750 "full frame rate." These are community/GitHub-discussion-reported numbers (not an official benchmark suite), but they are current, real, and directly on-topic for the RX 7800 XT class of hardware.
- Caveat: this ONNX effect support is **preview/experimental, not in a stable release**, and the overlay/duplicate-frame-detection integration is explicitly noted as incomplete in the discussion.

**Verdict:** forking/contributing to Magpie (or vendoring its capture+present+HLSL-effect engine) is a materially smaller lift than building a capture/present pipeline from scratch, and it gets Upflow ONNX-SR-in-realtime almost for free once the preview branch stabilizes.

### B.3 Realtime-capable model landscape (2025-2026)

| Model/technique | Published | License | 1080p perf (midrange GPU) | Runtime needed |
|---|---|---|---|---|
| **Anime4K** | github.com/bloc97/Anime4K | MIT | Sub-ms per pass (pure shader, no ML weights) — realtime by construction | HLSL/GLSL, runs directly in Magpie/mpv/any shader host |
| **RTMoSR** (arch by `umzi2`, in `traiNNer-redux`) | github.com/the-database/traiNNer-redux | Follows traiNNer-redux/Spandrel-family permissive licensing | Community ONNX build (`rewaifu_anime_rtmosr_gan_x2_fp16_op18.onnx`) hits ~57-80 fps at **1080p→4K** (i.e., 4x) across RTX 4080 / RX 9070 XT / Arc A750 | ONNX Runtime (TensorRT/DirectML/MIGraphX) |
| **SPAN** (Swift Parameter-free Attention Network) | github.com/hongyuanyu/SPAN (won CVPR2024 NTIRE ESR challenge) | Included permissively in Spandrel's core (permissive-license bucket) | ~21 fps at 256x256 tiles in neosr benchmarking (not a direct 1080p number — treat as "faster than transformer SR, slower than Compact/RTMoSR") | PyTorch, ONNX-exportable |
| **Compact ESRGAN / SRVGGNetCompact** (`realesr-general-x4v3`) | github.com/xinntao/Real-ESRGAN | BSD-3-Clause | ~26 fps at 256x256 tiles in same neosr benchmark; this is the architecture Upscayl/rife-class tools already lean on for speed | ncnn-Vulkan or ONNX |
| **RTSR** (AV1-compressed-content real-time SR) | arXiv 2411.13362 / NTIRE-lineage academic work | Research-paper release, not a maintained OSS repo with packaged weights | Academic-only publication; no evidence of a ready-to-run consumer package | n/a |
| **xBRZ** | Long-standing OSS pixel-art scaler | GPL-ish (check per-port) | Trivial cost, not ML | Any shader host |
| **RIFE (frame interpolation, not SR)** | github.com/hzwer/ECCV2022-RIFE, ncnn port `nihui/rife-ncnn-vulkan` | MIT-family | **Not realtime on modest hardware**: only ~4 fps at 1080p measured on a mobile Adreno GPU (4 worker threads); desktop dGPU numbers are informally "good enough for offline 2x/4x video interpolation," not a 60-120fps live pipeline. TensorRT port can be up to 2x faster than ncnn/Vulkan on NVIDIA, still not confirmed at 60fps+ live 1080p. | ncnn-Vulkan or TensorRT |
| **LSFG** (Lossless Scaling's frame-gen model) | Proprietary, Steam app only | Commercial, closed weights | Claimed realtime by the vendor (runs on integrated GPUs per their marketing) | Proprietary runtime |
| **lsfg-vk** | github.com/PancakeTAS/lsfg-vk | MIT | Reimplements LSFG's pipeline speed characteristics (inherits LSFG's model, hooked at the Vulkan layer) | Vulkan + DXVK (**Linux-only**, confirmed — no native Windows build) |
| **AMD AFMF (Fluid Motion Frames)** | AMD driver feature (Adrenalin), not a downloadable model | Proprietary, driver-embedded | Realtime by design (it's the shipped optimization target) | GPU driver only — **no third-party app control API**; it is a global driver toggle applied transparently to any DX11/12/OpenGL/Vulkan swapchain, cannot be invoked/orchestrated from Upflow's own process |
| **AMD FidelityFX Frame Interpolation (FSR3, part of FidelityFX SDK)** | github.com/GPUOpen-LibrariesAndSDKs/FidelityFX-SDK | MIT | Realtime by design | D3D12 native (Vulkan "in development" per the SDK docs) — **requires the host application to integrate the SDK and supply its own motion vectors**; not usable for generic desktop-window capture-and-interpolate the way Upflow/Magpie would need |

**Confirmed:** GMFSS-class optical-flow VFI is not realtime — not even evaluated above because it's architecturally in the same "offline, quality-first" tier as SwinIR/HAT/DAT, several times more expensive per-frame than RIFE.

### B.4 Integration verdict for Upflow

| Option | What it requires | Effort | Assessment |
|---|---|---|---|
| (a) Native C++/Rust capture+present+inference helper process, controlled by FastAPI-as-UI | DXGI/WGC capture, D3D11/12 compute pipeline, ONNX Runtime or ncnn embedding, IPC to the Python backend | **Large** (multiple weeks minimum for a correct, low-latency implementation) — this is essentially "build Magpie from scratch" | Only justified if Magpie's license/architecture proves unworkable to extend; not the first move |
| (b) Fork/extend Magpie with custom model effects | Track Magpie's `onnx-preview2` work, contribute AMD-path fixes upstream or maintain a light fork adding Upflow-specific model management (tie into the same Spandrel/ONNX pipeline built for Topic A) | **Small-to-medium** — capture/present/HLSL-effect engine is already solved and MIT-licensed(-equivalent; verify Magpie's exact license file before shipping) | **Recommended starting point.** Reuses a proven, actively-developed, Windows-native, multi-vendor-GPU engine; ONNX SR support is already landing in it. |
| (c) Python-side with D3D shared textures | Python would need a D3D11/12 interop layer (e.g., via `pywin32`/`ctypes` DXGI bindings + shared-handle textures) to avoid GPU→CPU→GPU round-trips | **Not viable as a primary path** — no mature, maintained Python library does zero-copy DXGI-capture→GPU-tensor→present today; anything built this way would be a bespoke ctypes project with the same engineering cost as option (a) but in a worse-suited language for the hot path | Rule out for the realtime module; Python stays as the control-plane/UI layer only, same as it already is for the ncnn subprocess model in v1 |

**What to build FIRST (MVP):** a **shader-only upscaling overlay**, no frame generation — Anime4K and/or a compact-SR ONNX model (RTMoSR/Compact-ESRGAN) running through Magpie's existing effect pipeline (or a light fork of it), driven from Upflow's FastAPI UI purely as a "launch/configure/stop this helper process" control surface. This sidesteps all of the frame-generation feasibility problems entirely and ships something real: realtime SR overlay for games/video with a genuinely competitive AMD-path (DirectML numbers already demonstrated in the wild on RX 9070 XT-class hardware, directly relevant to RX 7800 XT).

**What is NOT feasible, stated plainly:**
- A cross-platform, open-source, Windows-native **realtime frame-generation** network competitive with LSFG does not exist today. Building one from scratch is a research project, not a v1 engineering task.
- Hooking AMD's driver-level AFMF from a third-party app is not possible — there is no public API for it; it's a system-wide toggle, not an integratable component.
- FidelityFX Frame Interpolation cannot be retrofitted onto arbitrary captured windows/games the way Upflow would need (it needs the source engine's own motion vectors, which Upflow never has for a captured desktop window).
- RIFE-class VFI is not fast enough today for a live 1080p 60fps+ pipeline on realistic consumer hardware based on the available benchmark data; it remains an "offline batch" tool for Upflow's existing video-upscale feature, not a realtime-overlay candidate.
- A pure-Python realtime capture/present/inference hot path is not viable; any realtime module necessarily needs a native helper process, whether built from scratch or (preferably) inherited from Magpie.

---

## Sources

**Topic A — Runtimes, NPUs, model formats**
- [ONNX Runtime DirectML Execution Provider docs](https://onnxruntime.ai/docs/execution-providers/DirectML-ExecutionProvider.html)
- [microsoft/onnxruntime#16644 — device_id GPU mapping ambiguity](https://github.com/microsoft/onnxruntime/issues/16644)
- [microsoft/onnxruntime#7433, #10603, #21917 — Vulkan EP feature requests, unimplemented](https://github.com/microsoft/onnxruntime/issues/21917)
- [OpenVINO Supported Devices (Intel-only GPU/NPU)](https://docs.openvino.ai/2024/about-openvino/compatibility-and-support/supported-devices.html)
- [IREE Vulkan GPU deployment guide](https://iree.dev/guides/deployment-configurations/gpu-vulkan/)
- [microsoft/DirectML README — "in maintenance mode," Windows ML successor statement](https://github.com/microsoft/DirectML)
- [torch-directml on PyPI](https://pypi.org/project/torch-directml/)
- [AMD Ryzen AI 400 Series desktop APU announcement — TechPowerUp](https://www.techpowerup.com/346918/amd-launches-ryzen-ai-400-series-processors-for-mobile-and-desktop)
- [AMD XDNA — Wikipedia](https://en.wikipedia.org/wiki/AMD_XDNA)
- [wccftech — Ryzen AI 400/PRO 400 AM5 desktop confirmation, OEM-only](https://wccftech.com/amd-confirms-ryzen-ai-400-ryzen-ai-pro-400-apus-for-am5-desktops/)
- [Spandrel repository (chaiNNer-org)](https://github.com/chaiNNer-org/spandrel)
- [Spandrel LICENSE](https://github.com/chaiNNer-org/spandrel/blob/main/LICENSE)
- [spandrel-extra-arches on PyPI](https://pypi.org/project/spandrel-extra-arches/)
- [chaiNNer repository](https://github.com/chaiNNer-org/chaiNNer)
- [Upscayl repository](https://github.com/upscayl-ai/upscayl) / [upscayl-ncnn backend](https://github.com/upscayl/upscayl-ncnn)
- [Real-ESRGAN repository + LICENSE (BSD-3-Clause)](https://github.com/xinntao/Real-ESRGAN/blob/master/LICENSE)
- [qualcomm/Real-ESRGAN-General-x4v3 on Hugging Face](https://huggingface.co/qualcomm/Real-ESRGAN-General-x4v3)
- [bukuroo/RealESRGAN-ONNX on Hugging Face](https://huggingface.co/bukuroo/RealESRGAN-ONNX)
- [rocca/swin-ir-onnx on Hugging Face](https://huggingface.co/rocca/swin-ir-onnx)
- [What is Windows ML? — Microsoft Learn (primary, fetched in full)](https://learn.microsoft.com/en-us/windows/ai/new-windows-ml/overview)
- [Select execution providers using Windows ML — Microsoft Learn (primary, fetched in full, Python code samples)](https://learn.microsoft.com/en-us/windows/ai/new-windows-ml/select-execution-providers)
- [Windows ML execution providers (EP catalog: MIGraphX/NvTensorRtRtx/OpenVINO/QNN/VitisAI) — Microsoft Learn (primary, fetched in full)](https://learn.microsoft.com/en-us/windows/ai/new-windows-ml/supported-execution-providers)
- [onnxruntime-windowsml on PyPI](https://pypi.org/project/onnxruntime-windowsml/)
- [QNN Execution Provider docs](https://onnxruntime.ai/docs/execution-providers/QNN-ExecutionProvider.html)
- [ncnn Vulkan notes](https://ncnn.readthedocs.io/en/latest/how-to-use-and-FAQ/vulkan-notes.html)
- [Tencent/ncnn discussion #2705 — device selection API](https://github.com/Tencent/ncnn/discussions/2705)

**Topic B — Realtime capture/present/models**
- [Blinue/Magpie repository](https://github.com/Blinue/Magpie)
- [Blinue/Magpie discussion #1121 — onnx-preview2, AMD/NVIDIA/Intel benchmark reports](https://github.com/Blinue/Magpie/discussions/1121)
- [Magpie built-in effects wiki](https://github.com/Blinue/Magpie/wiki/Built-in-effects)
- [bloc97/Anime4K repository (MIT)](https://github.com/bloc97/Anime4K)
- [the-database/traiNNer-redux (RTMoSR architecture home)](https://github.com/the-database/traiNNer-redux)
- [hongyuanyu/SPAN repository](https://github.com/hongyuanyu/SPAN)
- [nihui/rife-ncnn-vulkan repository](https://github.com/nihui/rife-ncnn-vulkan)
- [hzwer/ECCV2022-RIFE repository](https://github.com/hzwer/ECCV2022-RIFE)
- [PancakeTAS/lsfg-vk repository (MIT, Linux-only)](https://github.com/PancakeTAS/lsfg-vk)
- [TechPowerUp — Lossless Scaling Frame Gen on Linux via lsfg-vk](https://www.techpowerup.com/338745/lossless-scalings-frame-generation-lands-on-linux-works-on-steam-deck)
- [AMD Fluid Motion Frames product page](https://www.amd.com/en/products/software/adrenalin/afmf.html)
- [PC Gamer — AFMF driver-level, any DX11/12 game](https://www.pcgamer.com/amd-fluid-motion-frames-finally-goes-official-driver-based-frame-generation-for-any-dx1112-game/)
- [GPUOpen-LibrariesAndSDKs/FidelityFX-SDK repository (MIT)](https://github.com/GPUOpen-LibrariesAndSDKs/FidelityFX-SDK)
- [FidelityFX frame-interpolation-swap-chain docs](https://github.com/azagramac/FidelityFX-SDK-FSR3/blob/master/docs/techniques/frame-interpolation-swap-chain.md)
- [OBS Forums — Windows Graphics Capture vs DXGI Desktop Duplication](https://obsproject.com/forum/threads/windows-graphics-capture-vs-dxgi-desktop-duplication.149320/)
- [Lossless Scaling docs — DXGI vs WGC explainer](https://sageinfinity.github.io/docs/FAQ/dxgiwgc)
- [RTSR: Real-Time Super-Resolution for AV1 Compressed Content (arXiv 2411.13362)](https://arxiv.org/abs/2411.13362)
