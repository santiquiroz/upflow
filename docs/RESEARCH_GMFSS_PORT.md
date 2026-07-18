# GMFSS → ONNX (any-GPU) — informe de viabilidad (2026-07-18)

**Veredicto: VIABLE-WITH-CAVEATS.** Todos los componentes neuronales son convnets/transformers exportables a ONNX y corren en DirectML. El único bloqueador es **softmax splatting (softsplat)**: mapea exacto a `ScatterND(reduction='add')` (opset 16+), pero el EP DirectML de onnxruntime lo **rechaza explícitamente** (verificado en `DmlOperatorScatter.cpp`: "DML does not support reduction") → en AMD debe vivir en el driver Python (CPU primero, kernel GPU después) — el mismo patrón grafos+driver de AudioSR. Nadie ha publicado un ONNX de GMFSS: sería el primero.

## Variante recomendada
**GMFSS_Fortuna (98mxr), pesos "PG" (pg104), MIT.** Sigue siendo el rey de calidad para interpolación de anime (SVFI model-spec). Bonus clave: el repo trae `model/softsplat_torch.py` — softsplat PyTorch puro (sin cupy) = referencia dorada + spec de port + camino a Scatter ONNX. HolyWu/vs-gmfss_fortuna (MIT) tiene la composición de inferencia más limpia y los pesos commiteados (~75MB fp32 total).

**Descartados:** GIMM-VFI (licencia S-Lab no comercial), VFIMamba (selective-scan CUDA — peor que softsplat para portar), SGM-VFI (sin licencia), sniklaus/softmax-splatting (TRAMPA LEGAL: academic-only; derivar SOLO del softsplat_torch.py MIT de Fortuna). **DRBA (MIT)**: add-on para timing de anime (doses/treses) sobre rife/gmfss — buen v2.

## Descomposición
| Componente | Qué es | ONNX | DML | Riesgo |
|---|---|---|---|---|
| FeatureNet (3.3MB) / MetricNet (0.5MB) / FusionNet-GridNet (31MB) / IFNet-RIFE | convnets | ✅ trivial | ✅ | Bajo |
| GMFlow (19MB) | CNN + Swin-attention + matmul-correlation + unfold + grid_sample | ✅ shapes fijos (pad /64; precedente ptlflow) | ✅ (GridSample 4D opset16+ con shader D3D12 propio) | Medio |
| **softsplat** (×~7 por frame) | scatter-add bilineal + norm softmax | ✅ expresable (clamp+zero-weight, sin masks) | ❌ **ScatterND-add → cae a CPU EP** | **Alto** |

Ports previos: TensorRT (HolyWu) y NCNN (SVFI) TAMBIÉN dejan softsplat fuera del grafo (eager CUDA) — todos terminan en grafos-alrededor-de-un-splat-python. ComfyUI usa Taichi como escape no-NVIDIA.

## Performance (enhancr, 1080p 2x)
GMFSS ≈ 4.3fps en A4000, 16.4fps en 4090 (PyTorch CUDA) = **10-15× más lento que RIFE por diseño**. Estimado 7800 XT vía DML: **~2-5 fps con splat GPU (Fase 4) = viable offline** (episodio 24min ≈ 2-5h); ~0.2-0.6 fps con splat CPU (solo parity/dev).

## Plan por fases
0. Referencia dorada SIN CUDA: Fortuna + softsplat_torch en CPU local, dump de tensores por componente (lujo que AudioSR no tuvo).
1. Exportar nets fáciles (Metric/Feature/Fusion/IFNet) + parity por componente.
2. GMFlow a shapes fijos scale=0.5 (riesgo medio; fallback = partir en sub-grafos con glue numpy).
3. Softsplat en el driver (torch-CPU index_add_ o np.bincount por bloques) + parity E2E pixel-exacta.
4. **El riesgo real, aislado**: splat rápido — (a) kernel OpenCL scatter-add vía pyopencl (~50 líneas, corre en AMD/Intel/NVIDIA) ← recomendado; (b) torch-directml entre sesiones; (c) grafo único con ScatterND-add = rápido solo en CUDA EP (regalo para NVIDIA). NO aproximar con backward-warp (reintroduce el ghosting que softsplat existe para evitar).
5. Integración Upflow (fp16+IOBinding, raw-pipe, cancel) + DRBA opcional.

**Esfuerzo:** fases 0-3 ≈ port AudioSR (1-2 semanas focalizadas); fase 4a días, acotada, testeable contra los tensores de fase 3. **Criterio de kill:** si fase 4 no pasa ~1fps@1080p en la 7800 XT, queda como modo "max quality (slow)" y anime default sigue en RIFE(+DRBA).
