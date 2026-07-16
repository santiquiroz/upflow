# Research: AI Audio Restoration / De-Compression (Bandwidth Extension + Codec De-Artifacting) — July 2026

Scope: Upflow — local Windows/AMD (RX 7800 XT, no CUDA) video/audio pipeline. Existing audio capability is `AudioEnhancer` (`app/services/engines/audio_enhance.py`), which wraps two **denoise-only** modes (DeepFilterNet Rust CLI, ffmpeg `arnndn`/RNNoise) via `run_guarded_process` subprocess. This research is scoped to a **distinct, new** capability: reconstructing audio that went through heavy social-media/messenger compression (WhatsApp/Telegram voice notes and video audio — low-bitrate Opus/AAC, band-limited, codec-artifacted) — bandwidth extension + de-artifacting + general audio super-resolution, for music/SFX/voice alike, not just speech clarity.

---

## 1. Executive Summary

1. **No model available today simultaneously satisfies all three hard constraints** — (a) general-audio codec-artifact-removal + bandwidth extension, (b) a *verified* AMD/no-CUDA local path, (c) a license permissive enough to bundle. Every serious candidate fails at least one axis.
2. **Apollo** (JusperLee, ICASSP 2025) is the closest architectural/training match to the actual use case — it is trained directly on MP3-compressed-music pairs (not just clean-audio-then-lowpassed pairs like the SR models), and it is a **feed-forward** band-split + Roformer + TCN network, not a diffusion model — the easiest of the serious candidates to export to ONNX (single forward pass, no multi-step sampling loop). Blocked by **CC BY-SA 4.0** (copyleft/share-alike) and by the fact that nobody has published an ONNX/DirectML port yet — you would build it yourself (Medium-High effort).
3. **AudioSR** (Haohe Liu) is MIT-licensed and pip-installable (`pip install audiosr`), general audio (music/speech/SFX), but is a **latent-diffusion** model (multi-step sampling + vocoder) — architecturally the same class as Stable Diffusion, so it inherits SD's DirectML conversion pain (control-flow-heavy sampling loop) and nobody has shipped an onnxruntime-directml port. Its conversion IS proven possible, though: **Intel independently ported it to OpenVINO IR** for the Audacity AI plugins (`Intel/versatile_audio_super_resolution_openvino` on HF) — but OpenVINO's GPU plugin is **Intel-only, confirmed no AMD GPU support**, so on the RX 7800 XT that port would only run on OpenVINO's CPU plugin, not accelerated.
4. **A2SB** (NVIDIA, 2025) does music bandwidth-extension + inpainting at 44.1kHz and looks technically strong, but is released under the **NVIDIA Source Code License – Non-Commercial** for both code and weights — disqualifying for bundling regardless of runtime, ruled out on license alone.
5. A GitHub project also named **"FlashSR"** (`ysharma3501/FlashSR`, distinct from the KAIST arxiv:2501.10807 paper of the same name) has a genuine, working **ONNX export** (converted by Xenova, ~500KB, Apache-2.0, `pip install`-able, demonstrated `onnxruntime` CPU code in the README, claims 200-400x realtime and "real-time on a single CPU core"). This is the *only* candidate in this research with a ready-to-try ONNX artifact under a permissive license — but it is derived from **HierSpeech++** (a speech-synthesis vocoder architecture), does fixed 16kHz→48kHz upsampling only (assumes a clean low band, does not target in-band codec artifacts), and its suitability for music/general audio is **unverified** and architecturally suspect (speech lineage). No one has reported testing it with the DirectML execution provider.
6. **VoiceFixer** (MIT, general *speech* restoration — noise/reverb/clip/low-res in one model) is CPU-friendly and license-clean, but is explicitly speech-only per its own README ("General Speech Restoration") — doesn't meet the "general audio, not just speech" requirement, though it's a reasonable bonus for voice-note-heavy WhatsApp clips.
7. **Resemble Enhance** remains ruled out: CUDA hardcoded as the preferred device, open GitHub issues (#16, #17) reporting broken/slow CPU fallback, no DirectML build — unchanged from the prior research pass.
8. A hobbyist project literally named for this exact use case — **`kroll-software/AudioDelossifier`** (TensorFlow/Keras, MIT) plus its OpenVINO port **`MarcoRavich/OV-AudioDelossifier`** (MIT) — exists, is general-audio and permissively licensed, but the original author states outright "it is unclear whether this always succeeds," has 12 commits and no releases. Low maturity, unverified quality; the OpenVINO port also inherits the Intel-only-GPU limitation. Worth a cheap spike, not a recommendation.
9. **ONNX Runtime's DirectML execution provider does support STFT/DFT operators** (merged via microsoft/onnxruntime PR #14856), which lowers — but doesn't remove — the general risk that spectral-domain audio models fail to run on DirectML; the constraint is power-of-two window sizes and native fp16 hardware, both satisfied by RX 7800 XT (RDNA3, native fp16). This is a genuinely new and encouraging finding since the DirectML EP historically lacked complex/FFT op coverage.
10. **FFmpeg alone has no true bandwidth-extension/de-artifacting filter** (no SBR-equivalent found) — cosmetic-only options like `aexciter`/high-shelf boosting can make audio *sound* brighter but do not reconstruct lost information; honest verdict: not a substitute for a model, only worth a one-line mention as a "does nothing real" baseline.

---

## 2. Comparison Table

| Candidate | Purpose / audio scope | Architecture | Quality for lossy-codec music+voice | AMD/no-CUDA path | Standalone CLI / pip | License | Maturity | Realistic speed | Integration effort |
|---|---|---|---|---|---|---|---|---|---|
| **Apollo** (JusperLee) | MP3/lossy-codec **music restoration** (trained directly on compressed-audio pairs — closest match to the use case) | Feed-forward: frequency band-split + Roformer + TCN band-sequence modeling + band reconstruction | Outperforms SR-GAN baselines on MUSDB18-HQ/MoisesDB across bitrates/genres per paper; explicitly targets mid/high-frequency codec degradation, not just missing highs | **No confirmed ONNX/DirectML port found anywhere.** Architecturally the most tractable to convert (no diffusion loop) — DIY effort | No pip/CLI — raw `train.py`/`inference.py`; community Colab wrapper (Jarredou) popular; referenced as "Apollo Restore" inside the UVR desktop GUI ecosystem but **not** in the `audio-separator` CLI package (verified: that package ships only MDX-Net/VR-Arch/Demucs/MDXC-Roformer, no Apollo) | **CC BY-SA 4.0** (copyleft/share-alike) — friction for bundling in a closed pipeline | Active-ish (391 stars, last meaningful commit 2025-03, ICASSP 2025 paper) | No CPU/DirectML numbers published; RTF benchmarked GPU/CPU in the paper but not disclosed in README excerpts reviewed | **Medium-High** — you must build the PyTorch→ONNX export + validate DirectML EP yourself; nobody has done this publicly |
| **AudioSR** (Haohe Liu) | General audio (music/speech/SFX) any→48kHz super-resolution | Latent diffusion U-Net + HiFi-GAN-style vocoder (same architecture *class* as Stable Diffusion) | High quality per paper/demo; general-purpose, not codec-artifact-specialized (trained mainly on bandwidth-reduction, less on codec quantization noise) | No official ONNX/DirectML port. **Existence proof of convertibility**: Intel ported it to OpenVINO IR for Audacity plugins (`Intel/versatile_audio_super_resolution_openvino`, Basic/General + Speech variants) — but OpenVINO GPU plugin is **Intel-only** (confirmed: "AMD discrete graphics cards are not supported by OpenVINO's GPU plugin"), so on RX 7800 XT this specific port is CPU-only | `pip install audiosr==0.0.7`, has a CLI script; no packaged binary | **MIT** | Active community usage (ComfyUI nodes, HF Spaces, cloud "AudioSR Online" services) | Diffusion multi-step sampling — no CPU RTF published; expect well below real-time on CPU given the sampling-step count; GPU-oriented in practice | **High** — diffusion control-flow makes ONNX export + DirectML validation a multi-week, SD-DirectML-style effort; nobody has published this port |
| **"FlashSR" (academic, KAIST, arxiv:2501.10807)** | General audio one-step distilled diffusion SR, 16-32kHz→48kHz | Single-step distilled diffusion (distillation + adversarial + DMD losses) + dedicated SR vocoder | Paper claims comparable/superior quality to AudioSR at ~22x its speed (0.36s per 5.12s audio on A6000) | No public code repo identified with confidence in this research; no CPU/DirectML numbers | Unclear — no confirmed pip/CLI | Unclear (paper only found; check per any future code release) | Research paper (IEEE/arxiv, Jan 2025), no confirmed public code | GPU-only numbers (A6000) | **High** / not actionable without code |
| **"FlashSR" (`ysharma3501/FlashSR` GitHub)** — **different project, same name, do not confuse with the paper above** | Fixed 16kHz→48kHz upsampling, **derived from HierSpeech++** (a speech-synthesis vocoder) | Tiny distilled model, non-diffusion (single fast forward pass) | **Unverified for music/general audio** — HierSpeech++ lineage means it may be speech-tuned; does not address in-band codec artifacts, only extends missing highs above 16kHz assuming a clean low band | **Real ONNX export exists** (converted by Xenova, ~500KB) — nobody has confirmed running it via `onnxruntime-directml`/AMD specifically, but the artifact is small and ready to spike-test today | `pip install`, working `onnxruntime` CPU code sample in README, plus a streaming-mode class (`StreamingFASRONNX`) | **Apache-2.0** | Small/new (215 stars, 21 commits, 2 open feature-request issues, no quality-complaint issues found) | Claims 200-400x realtime on GPU; README also claims real-time on a single modern CPU core | **Low** to *try* (artifact + code ready), **unknown** whether output quality is acceptable for music/general audio |
| **A2SB** (NVIDIA/diffusion-audio-restoration) | Music bandwidth extension + inpainting, 44.1kHz, hour-long inputs | Schrödinger Bridge diffusion, no vocoder needed (direct waveform) | SOTA claims on OOD music test sets per paper | PyTorch/CUDA-oriented; no CPU/ONNX info found; no hardware validation details disclosed | No pip/CLI; raw training/inference scripts + HF checkpoints | **NVIDIA Source Code License – Non-Commercial** (code AND model) | Active (2025 paper, GitHub + HF present) | Unknown | **Not viable — license disqualifies regardless of runtime** |
| **VoiceFixer** | General **speech** restoration (noise, reverb, clipping, low-res 2-44.1kHz) in one model | ResUNet analysis stage + neural vocoder synthesis stage | Good all-rounder for speech; **not** designed for music/general audio | CPU supported, no hard CUDA requirement documented | `pip`/library, no single binary | **MIT** | Active, pushed 2025 | Not benchmarked here; moderate CPU cost expected | **Medium** — but out of scope for "not just speech" requirement; good as a WhatsApp-voice-note bonus, not the primary mode |
| **Resemble Enhance** | Denoise + bandwidth extension, speech-oriented, 44.1kHz | Diffusion-adjacent enhancement pipeline | Good for speech restoration | CUDA hardcoded as preferred device; open issues #16/#17 report broken/slow CPU fallback; **no DirectML build** | No, Python (Gradio/library) | MIT | Active-ish | CPU fallback reported buggy/slow | **High** (unchanged from prior research — do not pursue) |
| **`kroll-software/AudioDelossifier`** + **`MarcoRavich/OV-AudioDelossifier`** | General audio, explicitly targets "delossifying" MP3/compressed audio | TensorFlow/Keras CNN-family model (original); OpenVINO IR port (community) | **Author-acknowledged uncertain**: "It is unclear whether this always succeeds"; MSE-based evaluation admittedly weak at higher bitrates | OpenVINO port again hits the Intel-only-GPU wall → CPU-only on AMD; original TF model has no ONNX/DirectML mention at all | Python scripts only (`audio_predict.py` / `Deloss.py`), no CLI packaging | **MIT** (both) | **Low** — 12 commits/no releases (original); OV port has 40 commits, single June-2025 release | Unknown, untested | **Low to spike, but quality is explicitly a gamble** — a genuine long-shot worth a quick trial given the permissive license, not a serious recommendation |
| **FFmpeg-only baseline (no ML)** | Cosmetic brightness only — high-shelf EQ, exciters/harmonic generators (`aexciter`, `equalizer`) | DSP, no learned model | Does **not** reconstruct anything — adds synthetic harmonics/brightness, can sound "less muffled" but is not restoration | Trivially available (already vendored ffmpeg) | Yes, already in the binary | ffmpeg's own (LGPL/GPL) | Mature | Real-time trivial | **Lowest effort, but honestly not the feature being asked for** — mention only as a non-answer fallback |

**Verified AMD/no-CUDA claims (2+ sources) for the space overall:**
- OpenVINO GPU-plugin Intel-exclusivity was cross-checked against OpenVINO's own official docs ("GPU plugin... for inference on Intel GPUs") and an independent Intel Community forum thread flagging AMD discrete GPU incompatibility — consistent across both.
- ONNX Runtime DirectML EP's STFT/DFT support was confirmed directly in the merged `microsoft/onnxruntime` pull request (#14856) implementing the DML STFT kernel with its power-of-two-window and native-fp16 constraints — a primary source (the PR itself), with the constraint description matching DirectML's own DFT-kernel documentation referenced in the same thread.
- No candidate model (Apollo/AudioSR/A2SB/either "FlashSR"/Resemble Enhance/AudioDelossifier) has a **verified, demonstrated** onnxruntime-directml run reported anywhere found in this research — the "AMD path" for every serious restoration model in this space is either unconfirmed, DIY, or explicitly Intel/NVIDIA-only today.

---

## 3. Key Question, Answered Directly

**Is there ANY model that (a) does general-audio codec-artifact-removal + bandwidth extension AND (b) has a verified AMD/no-CUDA local path AND (c) a permissive-enough license to bundle in a closed local app?**

**No — not today, not as a finished, verified artifact.** The honest breakdown:

- Closest on **(a) capability match**: Apollo — but fails (b) (DIY ONNX/DirectML, unverified) and strains (c) (CC BY-SA 4.0).
- Closest on **(c) license + readiness**: `ysharma3501/FlashSR`'s ONNX export — Apache-2.0, ready today — but its capability for (a) is unverified/likely-poor for music given its speech-synthesis lineage, and it doesn't do codec de-artifacting (BWE only, assumes clean low-band).
- Best-documented **conversion feasibility precedent**: AudioSR, via Intel's independent OpenVINO IR port — proves the model graph is exportable outside PyTorch/CUDA, but that specific port has no AMD GPU acceleration (Intel-only), and nobody has taken the further step to onnxruntime-directml.
- A2SB is the strongest research result but is license-disqualified outright regardless of runtime.

So: the realistic paths are "CPU-only and slow" (AudioSR/Apollo via raw PyTorch, no GPU accel), "license-encumbered" (Apollo CC BY-SA, A2SB non-commercial), or "conversion effort required with no verified precedent for this exact model" (Apollo/AudioSR → ONNX → DirectML, all DIY).

---

## 4. Recommendation

### Do not ship a v1 "audio restoration" mode today as a finished, verified pipeline.

Nothing clears the bar of "verified AMD path + adequate license + confirmed general-audio quality" simultaneously. Shipping something half-verified under an "Audio Restoration" label risks user-visible quality regressions (a bandwidth-extension model can make already-degraded WhatsApp audio sound *worse* — ringing/pre-echo artifacts — if misapplied) with no verified fallback.

### Least-bad concrete path, phased:

**Phase 0 — near-zero-cost spike (a few hours):** Try `ysharma3501/FlashSR`'s ONNX model (`YatharthS/FlashSR`, Apache-2.0, ~500KB) directly through `onnxruntime-directml` — the exact runtime already vendored in Upflow's image pipeline. Since the model is tiny and the code sample is already onnxruntime-based, this is a same-day experiment: load `model.onnx`, run the `DmlExecutionProvider`, listen to output on a few real WhatsApp/Telegram-compressed music and voice clips. Expected outcome: works technically (DirectML EP + STFT support should handle it), but quality on music is the open question given its HierSpeech++/speech lineage — treat this purely as a go/no-go spike, not a commitment.

**Phase 1 — the real investment, if Phase 0 or user feedback says "we need this for real" (Medium-High effort, 1-2 weeks):** Port **Apollo** (JusperLee) to ONNX. It is the only candidate actually trained on the target problem (lossy-codec music restoration) and, being feed-forward (band-split + Roformer + TCN, no diffusion sampling loop), is the most tractable of the serious models to export — similar effort class to exporting a Demucs/BS-Roformer separation model, which the existing `audio-separator` ecosystem does routinely for architecturally similar band-split networks. Steps:
1. Export the PyTorch checkpoint to ONNX (watch for the model's STFT window size — must be power-of-two for the DirectML STFT kernel; if not, decompose STFT into matmul-based DFT before export, a known workaround pattern).
2. Validate under `onnxruntime-directml` on the RX 7800 XT; fall back to CPU execution provider if any op is unsupported.
3. Add a new `services/engines/apollo_restore.py` mirroring the `AudioEnhancer` pattern — `available()` / async `run()` — but ONNX-session-based instead of subprocess-based (matches the "ONNX Runtime DirectML path" precedent already used for image models in this repo, rather than the vendored-binary/subprocess pattern used for DeepFilterNet).
4. **Before shipping**: resolve the CC BY-SA 4.0 question explicitly — at minimum, ship clear attribution/license disclosure for the bundled model weights in Upflow's credits; if Upflow is ever distributed more broadly than "zip for a friend," get this reviewed rather than silently bundling a share-alike-licensed model inside closed application code.

**Phase 2 — fallback if Phase 1's DirectML port doesn't pan out:** CPU-only Apollo via raw PyTorch inference (accept it as a slow, no-GPU-accel "Best but slow" tier, the same way Demucs/AudioSR are already deferred in the prior research doc) — still gated on the same CC BY-SA 4.0 review.

**Do not pursue:** A2SB (non-commercial license kills it outright, independent of quality), Resemble Enhance (no AMD path, buggy CPU, unchanged from prior finding), raw AudioSR-to-DirectML (diffusion control-flow makes this a much larger effort than Apollo for a similar or worse general-audio-restoration payoff — Apollo's training objective is the better match anyway).

**Keep clearly separate from "denoise" branding:** whatever ships here should be a distinctly-labeled "Restore / Upscale Audio" mode, separate from the existing DeepFilterNet/RNNoise "Denoise" mode — they solve different problems (noise removal vs. reconstructing missing/damaged signal) and users should be able to pick one, the other, or both in sequence (denoise first, then restore, is the sane pipeline order if both are ever combined).

### Integration sketch (if Phase 1 is greenlit)

```
app/services/engines/
  audio_enhance.py        # existing: DeepFilterNet / RNNoise, subprocess pattern
  audio_restore.py         # NEW: Apollo-ONNX, onnxruntime-directml pattern (mirrors the
                           # existing Spandrel→ONNX image pipeline's runtime plumbing,
                           # not the vendored-binary subprocess plumbing)
```
- `AudioRestorer.available()`: checks the ONNX model file exists + `onnxruntime` importable with `DmlExecutionProvider` (or falls back to `CPUExecutionProvider`).
- `AudioRestorer.run(input_wav, output_wav)`: loads a persistent `ort.InferenceSession`, chunks the input (feed-forward models typically need fixed/segment-length windows — check Apollo's default segment size), runs inference per chunk, overlap-adds, writes output.
- Exposed as a new `restore` mode alongside the existing `AUDIO_ENHANCE_MODES`, or as a distinct `AUDIO_RESTORE_MODES` config list — keep it separate from `DEEPFILTER_MODE`/RNNoise since the failure modes and quality trade-offs are different enough to confuse users if merged into one dropdown.

---

## 5. Honest Verdict on AMD/No-CUDA Feasibility

**Nothing in this space has a demonstrated, working, AMD-verified local pipeline today.** Every serious audio-restoration/SR model found in this research (AudioSR, Apollo, A2SB, both "FlashSR"s, Resemble Enhance, VoiceFixer) has been built and validated CUDA-first; the one credible non-CUDA existence proof (Intel's OpenVINO port of AudioSR) explicitly does not extend to AMD GPUs by design (OpenVINO's GPU plugin is Intel-only), leaving CPU as the only confirmed fallback for that specific artifact. The most encouraging general-purpose finding is that ONNX Runtime's DirectML execution provider has grown real STFT/DFT operator support, which removes what used to be the single biggest structural blocker to porting spectral-domain audio models to DirectML — but "the blocker is smaller than it used to be" is not the same as "a working port exists." Anyone pursuing this will be the first to publish a working Apollo-or-AudioSR-on-DirectML pipeline as far as this research could determine; budget accordingly (Medium-High effort, with real risk the STFT window-size/fp16 constraints or some other op in the graph forces a CPU fallback for part of the model).

---

## 6. Sources

**Apollo:**
- https://github.com/JusperLee/Apollo
- https://huggingface.co/JusperLee/Apollo
- https://arxiv.org/abs/2409.08514 / https://arxiv.org/html/2409.08514v1 (paper: architecture, feed-forward band-split+Roformer+TCN, CC BY-SA 4.0)
- https://colab.research.google.com/github/jarredou/Apollo-Colab-Inference/blob/main/Apollo_Audio_Restoration_Colab.ipynb (community wrapper, adoption signal)
- https://github.com/nomadkaraoke/python-audio-separator (verified: MDX-Net/VR-Arch/Demucs/MDXC only, Apollo NOT included in this CLI package)

**AudioSR:**
- https://github.com/haoheliu/versatile_audio_super_resolution (MIT, `pip install audiosr`)
- https://arxiv.org/abs/2309.07314 (paper)
- https://huggingface.co/Intel/versatile_audio_super_resolution_openvino (Intel's OpenVINO IR port — Basic/General + Speech variants)
- https://github.com/intel/openvino-plugins-ai-audacity (GPL-3.0 plugin wrapper; "Super Resolution" effect = C++/OpenVINO IR port of AudioSR)
- https://docs.openvino.ai/2024/openvino-workflow/running-inference/inference-devices-and-modes/gpu-device.html and https://community.intel.com/t5/Graphics/OpenVINO-install-in-Debian-Linux-when-Discrete-AMD-GPU-is-also/m-p/1662384 (OpenVINO GPU plugin = Intel-only, no AMD, cross-checked 2 sources)

**A2SB:**
- https://github.com/NVIDIA/diffusion-audio-restoration (NVIDIA Source Code License – Non Commercial)
- https://huggingface.co/nvidia/audio_to_audio_schrodinger_bridge
- https://arxiv.org/abs/2501.11311

**"FlashSR" (two distinct projects):**
- https://arxiv.org/abs/2501.10807 (KAIST academic paper — one-step diffusion distillation, ~22x AudioSR speed on A6000, no confirmed public code found)
- https://github.com/ysharma3501/FlashSR (different project, Apache-2.0, ONNX export by Xenova, ~500KB, HierSpeech++-derived, README onnxruntime code sample, streaming class)
- https://huggingface.co/YatharthS/FlashSR

**VoiceFixer:**
- https://github.com/haoheliu/voicefixer (MIT, "General Speech Restoration")

**Resemble Enhance (unchanged from prior research, re-verified not stale):**
- https://github.com/resemble-ai/resemble-enhance
- https://github.com/resemble-ai/resemble-enhance/issues/16, /issues/17

**AudioDelossifier:**
- https://github.com/kroll-software/AudioDelossifier (MIT, TensorFlow/Keras, author-acknowledged uncertain quality)
- https://github.com/MarcoRavich/OV-AudioDelossifier (MIT, OpenVINO port)
- https://github.com/intel/openvino-plugins-ai-audacity/discussions/356 and /issues/354 (community delossifier feature request thread that surfaced this project)

**DirectML / ONNX Runtime platform verification:**
- https://github.com/microsoft/onnxruntime/pull/14856 (STFT operator implementation for DirectML EP, power-of-two window + native-fp16 constraints)
- https://onnxruntime.ai/docs/execution-providers/DirectML-ExecutionProvider.html
- https://github.com/microsoft/DirectML (README notes DirectML is "in maintenance mode" — general platform caveat, not specific to audio)

**FFmpeg baseline (no true reconstruction filter found):**
- https://ffmpeg.org/ffmpeg-filters.html (searched for SBR/harmonic-extension equivalents; none found — `aexciter`/EQ-based brightening only, not reconstruction)

**Prior internal research (context, re-verified where overlapping):**
- `.superpowers/sdd/research-framegen-audio.md` (this repo's earlier pass — Topic 2 audio table; this new research supersedes it for the restoration/SR sub-topic with deeper verification)
