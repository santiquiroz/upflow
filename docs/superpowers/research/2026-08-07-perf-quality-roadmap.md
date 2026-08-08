# Roadmap de optimización: tiempos y calidad (investigación 2026-08-07)

Fuentes: workflow de 6 agentes (base del repo + 4 investigadores web + juez, 611k tokens),
pulso comunitario last30days (92 items: Reddit/YouTube/TikTok/HN/GitHub), suplementos web.
Restricciones respetadas: sin CUDA, sin torch en runtime, DirectML baseline multi-GPU,
vetos medidos del repo intactos.

## Resumen ejecutivo

El lane de video ya agotó las palancas grandes (uint8, whole-frame, fp16 7.26x, pipeline
threaded); ahí quedan mejoras de 10-20%. La palanca grande REAL está en generación, donde
hoy no hay NINGÚN tuning:

1. **Scheduler swap + presets few-step** — el scheduler es componente diffusers CPU-side,
   intercambiable en pipelines ORT sin tocar el grafo. Con guidance 0/1 cada paso cuesta
   la mitad (sin batch CFG). 25 steps+CFG → 4-8 steps = **~3-6x solo con Python**.
2. **Speed-LoRAs fusionadas antes del export** (LCM-LoRA, Hyper-SD) — el UNet fusionado
   conserva la firma ONNX estándar. Cero clases nuevas.
3. **Fusión offline de grafo (Olive / optimize_pipeline)** — el único trabajo de runtime
   100% compatible con DML que falta. Decenas de % esperables sobre la base fp16.

Además: AMD+Stability publicaron modelos ONNX oficiales `_amdgpu` (SDXL 1.0/Turbo,
SD3.5 Large/Turbo) con 3.1-3.8x medido vs PyTorch — integrables directo en nuestro stack
(candidatos a builtins u opción de instalación destacada).

## Quick wins (días)

- Gate A/B de `ENABLE_STREAM_PIPELINE` (definido en el diseño, nunca corrido). Medición pura.
- Anillo de K buffers CPU preasignados para readback en onnx_video_upscaler (~11% medido,
  54.3→48.4 ms/frame). NUNCA K=1 (corrompe, medido). K > frames en vuelo.
- Garantía fp16 en builtins + alerta visible cuando un job cae a fp32 (evita regresión 7.26x silenciosa).
- Speed-presets de scheduler en generación (`Scheduler.from_config` bajo `_pipeline_run_locks`;
  LCMScheduler, Euler trailing; exponer scheduler/steps/guidance + metadata por modelo).
- Paso opcional "fuse speed-LoRA" en el conversor (torch offline → export estándar).
  Empezar por LCM-LoRA: única familia con receta oficial de inpainting.
- **Piso de resolución en inpaint**: reescalar el crop a resolución nativa del modelo
  (1024 SDXL / 512 SD1.5) antes de samplear. El techo max_side existe; falta el piso.
  Parche de 300px difundido a 300px = anatomía blanda.
- Pack de prompting de inpaint: prompt acotado al PARCHE (no la escena), negative anatómico
  default en modo persona, máscara generosa hasta la articulación, batch de 4 variaciones
  del crop (barato) con seeds distintas.
- Cache de pipelines a 2-3 entradas gateado por VRAM (reusar branch gpu-capacity-admission).
- SDXL-Turbo: anclar por metadata (1-4 steps, guidance 0, 512px), excluir del editor,
  ofrecer como "preview instantáneo" de txt2img.

## Medium term (semanas)

- **Fusión offline de grafo** sobre los ONNX fp16 (optimize_pipeline / Olive
  OrtTransformersOptimization: MHA/GroupNorm/SkipLayerNorm que DML acelera con
  metacomandos) + `ORT_ENABLE_BASIC` + `AddFreeDimensionOverrideByName`. LA palanca de
  runtime restante.
- **Checkpoint inpaint 9ch**: detectar `in_channels==9` en conversión y rutear a
  `ORTStableDiffusion(XL)InpaintPipeline` (YA existe en optimum). Palanca #1 de morfología
  + desbloquea strength parcial real.
- **Soft inpainting / differential diffusion** en el loop del sampler (denoise proporcional
  al gris de la máscara, re-inyección del latente original entre llamadas al UNet; puro
  numpy). Elimina el seam duro. La MISMA pieza sirve a object transfer fase 2.
- **Hyper-SD 8-step CFG-preserved** como preset rápido del EDITOR (única familia few-step
  que conserva guidance y negative prompt; Lightning/Turbo/DMD2 corren a guidance 0 =
  negative muerto). TCDScheduler. Precedente DirectML: TensorStack/Hyper-SD-onnx. ~3x en inpaint.
- `ort.run_async` en video upscaler (esconder readback de 10.9 ms; techo ~19.5%).
- Bucketing de resoluciones de parche (múltiplos de 128) — evita re-JIT de sesión por shape.
- Retrofit IOBinding en OnnxUpscaler (fp32 NCHW, ~177 MB/frame readback) patrón apollo_restore.
- WebGPU EP nativo como lane experimental (v0.2.1 trae FlashAttention; DML nunca lo tendrá).
- Windows ML EP catalog (fase 2, implementable ya: build 26100+): TensorRT-RTX/OpenVINO/QNN/
  VitisAI auto-distribuidos. Sin efecto en RX 7800 XT hoy (MIGraphX "not supported for GenAI").
- PCT-Net a ONNX (MPL-2.0, CPU-suficiente) — armonización dedicada, fase 3 de object transfer.
- Plugin OpenVINO: NPU solo para SD1.5 (térmica en laptops Intel), SDXL queda en GPU.

## Morfología corporal en inpaint (orden recomendado)

1. **YA**: piso de resolución + pack de prompting (costo ~0, ataca 2 causas frecuentes).
2. **Inversión real**: checkpoint 9ch. Causa raíz: con base 4ch a strength 1.0 el UNet
   NUNCA VE la imagen enmascarada — solo el blending latente la reintroduce; de ahí los
   cuerpos que no calzan con la pose. El 9ch la recibe como input. Trabajo = detección en
   conversión + ruteo (optimum ya lo trae).
3. Soft inpainting (seam invisible en piel/hombros).
4. Hyper-SD 8-step para velocidad sin perder adherencia.
5. Solo si sigue fallando: **ControlNet openpose dentro del inpaint** (research bet:
   re-export del UNet con inputs residuales + loop custom; empezar SD1.5 — precedente
   Amblyopius/Stable-Diffusion-ONNX-FP16 en DirectML; DWPose ONNX puro extrae pose DEL CROP).
   +25-40%/paso, +2.5GB VRAM. Comunidad 2026: dos ControlNets a peso 0.35-0.55 > uno a 1.0.

BrushNet/PowerPaint/Fooocus patch: descartados (sin exports ONNX; diffusers migró a
ControlNet Union).

## Transferencia de objetos entre imágenes (diseño por fases)

- **F0 (días, CPU puro)**: MobileSAM (ya existe) → recorte → paste con feather por distance
  transform (3-8px) ± seamlessClone Poisson; igualación de color Monge-Kantorovich/Reinhard
  en LAB (matriz 3x3 closed-form, ~30 líneas numpy). OJO: reimplementar del paper —
  el paquete `color-matcher` es GPL-3.0.
- **F1 (días-1 sem)**: pase de armonización con `run_masked_edit` TAL CUAL: inpaint sobre la
  zona pegada, strength 0.25-0.45, máscara = objeto dilatado 10-20% + blur fuerte, prompt =
  escena DESTINO. El CONTEXT_RATIO=0.75 nuevo juega directo a favor.
- **F2 (sem)**: differential diffusion — strength por píxel (centro bajo = identidad, borde
  alto = fusión). Misma pieza que el soft inpainting.
- **F3 (sem)**: PCT-Net ONNX para mismatch de iluminación local.
- **F4 (bets)**: IP-Adapter horneado en el UNet (identidad a strength alto), IC-Light fbc
  export in-house (re-iluminación direccional), lane FLUX Kontext DirectML (16GB+).
- Descartados: AnyDoor/TF-ICON/Insert-Anything (torch-only/CUDA 12B), Harmonizer (CC BY-NC).
- F0-F3 cubren ~80% de los casos sin ningún modelo nuevo hasta F3.

## Bug conocido (pre-existente, confirmado y reproducido en review 2026-08-07)

`inpaint_mask._expand_to_square`: una marca más ANCHA que la dimensión menor de la
imagen (ej. 1500px de ancho en una foto 1920x1080) produce un crop cuadrado clampado
que NO cubre el bbox marcado → solo se edita la franja central, con costura dura y
sin aviso. Fix: generalizar el crop a rectangular (el bucketing de 128 ya acepta
width/height distintos); `resolve_model_side` debe pasar a (work_w, work_h).
Va con el trabajo de geometría de la ola 2.

## Research bets

- ControlNet inpaint (arriba). DeepCache vía export dual deep/shallow (2.6x claim; alternar
  sesiones SECUENCIAL no viola el veto 887A0005). IP-Adapter/IC-Light (object transfer F4).
- Interop D3D12 → encoder por hardware (elimina readback completo; SDK por vendor).
- Throughput real de EPs nativos + EP context caching TensorRT-RTX (falta hardware local).

## Medición 2026-08-07 (noche): fusión de grafo directa sobre fp16 = DESCARTADA

Spike en RX 7800 XT, UNet epicrealism SDXL fp16 (export optimum local), DML, 1024px batch-2:
- Baseline: **2224.8 ms/inferencia UNet** (mediana, 12 runs).
- `optimize_model(model_type="unet")` sobre el fp16 directo: GroupNorm fusion **crashea**
  (asume pesos fp32: "Expected 1280 bytes, got 640"); sin GroupNorm, fusiona 199
  SkipLayerNorm + 70 BiasSplitGelu + 51 NhwcConv pero **MultiHeadAttention: 0** (la
  fusión grande no matchea el export fp16) → **0.96x, levemente PEOR** (2309.8 ms).
- Conclusión: el 6x famoso de Olive era mayormente el fp16 que ya tenemos. La única
  variante viva es el camino completo fp32 → fusionar (GroupNorm+MHA) → convertir fp16,
  que exige re-export desde los pesos originales (~30+ min por modelo) — pasa a research
  bet con ese scope, no a quick win. Spike: scratchpad spike_olive_fusion_bench.py.

## Descartes firmes (no volver sin evidencia nueva)

IOBinding+DML en optimum (issues ORT #11666/#21239 abiertos, C API D3D12 requerida);
ReBAR (0.2% techo, medido); dos sesiones DML concurrentes (887A0005 medido); tiling DML
(1.26 vs 11.49 fps); ToMe (incompatible grafo estático); NPU Snapdragon QNN y Ryzen AI
para checkpoints arbitrarios (solo modelos pre-tuneados del vendor); DMD2 default de
inpaint (guidance 0); onnxruntime-gpu (pisa a -directml); strength<1 con 4ch (no-op medido).

## Pulso comunitario (last30days, ventana 2026-07/08)

- Modelos `_amdgpu` de Stability: 3.1-3.8x medido (Tom's Hardware) — integración directa.
- La fricción AMD del último mes es SETUP, no velocidad: los tutoriales de instalación
  dominan; apps ONNX empaquetadas (Amuse; Upflow) son la respuesta al hueco.
- Build 2026: AMD empuja Windows ML + onnxruntime-genai + DirectML más estable —
  la fase 2 del repo apunta donde va la plataforma.
- Few-step: Lightning 1024px 2-8 steps; en inpaint SOLO vía LoRA (TurboFill, arXiv 2504.00996).
- Raw: ~/Documents/Last30Days/stable-diffusion-amd-gpu-raw-v3.md
