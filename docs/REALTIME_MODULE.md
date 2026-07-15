# Módulo de tiempo real (Fase 7 — requisitos y arquitectura)

> Estado: **documento de diseño, sin implementar.** Base técnica: `docs/RESEARCH_V2_HF_REALTIME.md` (Topic B, verificado en fuentes primarias 2026-07-14). Este doc define qué construir, en qué orden, y qué NO es viable — para que la UI de Upflow v2 lo anticipe (módulo "Tiempo real" con tarjeta "próximamente") sin comprometer decisiones prematuras.

## Objetivo

Overlay de reescalado en tiempo real estilo Lossless Scaling para juegos/video: capturar una ventana → reescalar con un modelo → presentar en un overlay a pantalla completa, con baja latencia (presupuesto ~8-16 ms por frame a 60 fps). Upflow (FastAPI/Python) actúa como **plano de control/UI**: lanza, configura y detiene un proceso helper nativo. La captura/inferencia/presentación viven en el helper, NO en Python.

## Decisión de arquitectura: extender Magpie, no construir desde cero

**Recomendado: fork ligero / vendorizado de [Magpie](https://github.com/Blinue/Magpie) (GPL-3.0).**

Magpie ya resuelve el problema difícil: motor de captura multi-backend (Windows.Graphics.Capture, DXGI Desktop Duplication, GDI, DwmSharedSurface), pipeline de efectos HLSL (MagpieFX), presentación por ventana overlay borderless (sin hook al swapchain del juego → sin riesgo anti-cheat). Además, **la inferencia de modelos ONNX ya está aterrizando en Magpie** en la rama experimental `onnx-preview2` (issue Blinue/Magpie#1121), con backends DirectML (cobertura amplia AMD/NVIDIA/Intel) y MIGraphX experimental para AMD.

Números reales reportados por la comunidad en esa rama: RX 9070 XT ~70-80 fps a 1080p→4K con `rewaifu_anime_rtmosr_gan_x2_fp16_op18.onnx` vía DirectML. Directamente relevante para la RX 7800 XT del target.

**Restricción de licencia (binding):** Magpie es **GPL-3.0**. Upflow no puede linkear ni derivar código GPL en su propio binario sin volverse GPL. Modelo de integración obligatorio: Magpie (o el fork) corre como **proceso separado**, invocado por Upflow vía línea de comandos/IPC — sin linking, sin embeber su código. La UI de Upflow configura y lanza; el binario de Magpie es un artefacto distribuible aparte (igual que ffmpeg/realesrgan vendored, que traen sus propias licencias). Verificar el archivo de licencia exacto de Magpie antes de distribuir.

### Alternativas descartadas

| Opción | Por qué no |
|---|---|
| Helper nativo C++/Rust desde cero (captura+present+inferencia) | Es "reconstruir Magpie". Semanas de trabajo para una implementación correcta de baja latencia. Solo justificado si la licencia/arquitectura de Magpie resulta inviable de extender. |
| Hot-path en Python con texturas D3D compartidas | No viable. No existe librería Python mantenida que haga captura DXGI→tensor-GPU→present sin round-trips GPU→CPU→GPU. Sería un proyecto ctypes a medida con el mismo costo que el helper nativo, en un lenguaje peor para el hot-path. Python se queda como plano de control, igual que hoy con los subprocess ncnn. |

## MVP (Fase 7.1): overlay de solo-reescalado, SIN frame generation

Lo primero que se construye y que sí es real y barato:

- **Overlay de super-resolución en tiempo real** usando el pipeline de efectos de Magpie: shaders **Anime4K** (MIT, sub-ms por pase, realtime por construcción) y/o modelos ONNX compactos (**RTMoSR**, **Compact-ESRGAN/SRVGGNetCompact** `realesr-general-x4v3`) vía DirectML.
- Driven desde la UI de Upflow como superficie "lanzar/configurar/detener este proceso helper".
- **Reutiliza el pipeline ONNX/Spandrel de SP1**: los mismos modelos instalados desde Hugging Face (formato .onnx, o .pth convertido) sirven para el overlay realtime, no un catálogo aparte.
- Esquiva por completo los problemas de viabilidad del frame generation.

Modelos realtime-viables (del research, verificados):

| Modelo | Licencia | Perf 1080p→4K (GPU media) | Runtime |
|---|---|---|---|
| Anime4K | MIT | sub-ms/pase (shader puro) | HLSL, corre en Magpie directo |
| RTMoSR (build ONNX comunitario) | permisiva (familia traiNNer-redux) | ~57-80 fps (RTX 4080 / RX 9070 XT / Arc A750) | ONNX Runtime DirectML/MIGraphX |
| Compact-ESRGAN (`realesr-general-x4v3`) | BSD-3 | ~26 fps @ tiles 256² (base de Upscayl) | ncnn-Vulkan u ONNX |

NO realtime-viables (tier offline, presupuesto de frame excedido varias veces): SwinIR/HAT/DAT (transformer SR), GMFSS, RIFE (~4 fps @ 1080p en hardware modesto — se queda como herramienta offline de la feature de video existente).

## Lo que NO es viable (declarado sin rodeos)

- **Frame generation open-source competitivo con LSFG en Windows: no existe hoy.** Construirlo desde cero es un proyecto de investigación, no una tarea de ingeniería v1.
- **AFMF de AMD no se puede orquestar desde una app tercera.** Es un toggle a nivel driver aplicado transparentemente a cualquier swapchain DX11/12/OpenGL/Vulkan — sin API pública de control.
- **FidelityFX Frame Interpolation (FSR3, MIT) requiere motion vectors provistos por el motor del juego** — imposibles de obtener para una ventana capturada genérica.
- **lsfg-vk** (reimplementación MIT de LSFG) es **solo Linux** (Vulkan+DXVK), sin build nativo Windows.
- Conclusión frame-gen: re-evaluar en 6-12 meses si lsfg-vk gana camino Windows, si FidelityFX FI madura sin requerir motion vectors, o si aparece un modelo VFI realtime open-source. Hasta entonces: recomendar a los usuarios Lossless Scaling (Steam, ~$7) o Magpie para su uso de gaming; Upflow queda como el pipeline offline de máxima calidad.

## Presupuesto de latencia y hardware

- 60 fps → ~16 ms/frame; 120 fps → ~8 ms/frame. Cada pase de shader/inferencia debe caber en ese presupuesto → solo upscalers compactos/shader, nunca transformers.
- Captura: DXGI Desktop Duplication (menor latencia, misma GPU que el display — caso común dGPU único AMD) vs Windows.Graphics.Capture (abstrae GPU, mejor para híbridos/laptops). Magpie soporta ambos con fallback — razón extra para reusarlo.
- Hardware mínimo objetivo: clase RX 7800 XT / Arc A750 / RTX 4060+ con Vulkan+DirectML. NPU no aplica (el silicio NPU va fusionado al CPU, nunca a la dGPU; un box con dGPU discreta no tiene NPU salvo APU Ryzen AI 300/400).

## Fases sugeridas

- **7.1 MVP**: fork/vendor Magpie, overlay SR solo-shader/ONNX, control desde UI Upflow (lanzar/config/stop), reuso de modelos SP1. Verificar licencia GPL y modelo de proceso-separado.
- **7.2**: gestión de perfiles por juego/ventana, hotkeys, selección de modelo/dispositivo desde la UI de Upflow.
- **7.3 (condicional)**: frame generation — SOLO si el landscape open-source Windows cambia (re-evaluación programada).
