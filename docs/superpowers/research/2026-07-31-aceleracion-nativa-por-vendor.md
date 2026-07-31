# Aceleración nativa por fabricante (NVIDIA / AMD / Intel) manteniendo DirectML como baseline

Investigación 2026-07-31 (autónoma, pedida por Santiago). Fuentes: run de `/last30days`
(crudos en `~/Documents/Last30Days/aceleracion-ia-local-por-gpu-vulkan-ml-directml-raw-v3.md`,
con apéndice de fuentes web) + docs de vendors. Pregunta original: ¿podemos darle a cada
fabricante su tecnología propia de una vez, en vez de convertir, y qué pasó con el "gran
salto" de Vulkan?

## Hallazgo central

**La industria ya construyó exactamente lo pedido, y NO requiere reconvertir modelos.**
El formato ONNX es el contrato estable; lo que cambia por fabricante es el
*execution provider* (EP) que lo ejecuta. Nuestros modelos actuales (los mismos
archivos que hoy corren por DirectML) pueden correr por el EP nativo de cada
vendor cambiando la configuración de sesión, no el modelo.

- **Microsoft** puso DirectML en *sustained engineering* (mantenimiento: sigue
  soportado, no evoluciona) y movió el desarrollo a **Windows ML**: un ONNX
  Runtime del sistema (Windows 11 24H2+) con un **catálogo de EPs por hardware**
  que Windows descarga y actualiza solo — `NvTensorRtRtx` (NVIDIA),
  `OpenVINO` (Intel), `MIGraphX`/`VitisAI` (AMD), `QNN` (Qualcomm) — con CPU y
  DirectML legacy incluidos como fallback.
- **NVIDIA**: TensorRT-RTX EP mide **+50% de throughput vs DirectML** en RTX y
  hasta **20x menos tiempo de carga** con EP context caching. Se actualiza por
  Windows Update (KB5089168). El EP embebido en el repo de onnxruntime quedó
  deprecado a favor del **plugin ABI standalone** (nuevo modelo de EPs como
  plugins dinámicos).
- **Intel**: OpenVINO EP acelera ONNX en CPU/GPU/NPU Intel con "una línea en
  session options, sin reconvertir el modelo" (copy literal de Intel). OpenVINO
  2026.x sumó compilación AOT para NPU.
- **AMD (nuestro caso)**: el hueco sigue siendo ONNX-en-Windows: **no existe
  provider ROCm/MIGraphX de onnxruntime instalable en Windows hoy** (ROCm/ROCm
  issue #6294 lo pide). El MIGraphX EP anunciado llega **vía Windows ML** para
  Ryzen AI (APUs primero; discretas RDNA por confirmar). El camino AMD que sí
  maduró en Windows es PyTorch: ComfyUI + ROCm 7.1.1 one-click. Para un stack
  ONNX como el nuestro, DirectML/WinML sigue siendo EL camino AMD hoy.

## El "gran salto" de Vulkan

Es real y tiene nombre: **cooperative matrix (KHR + NV_cooperative_matrix2)**.
El backend Vulkan de llama.cpp ya es competitivo con CUDA en NVIDIA (a veces lo
supera) y a veces le gana a ROCm en AMD (Phoronix; charla dedicada en FOSDEM
2026; port AMD reportando 4x). Caveats de la comunidad: muy sensible al driver.

Dónde nos toca:

1. **Ya lo tenemos**: Real-ESRGAN/RIFE NCNN corren por Vulkan — ese carril ya es
   nativo multi-vendor y es la razón por la que el upscaler funciona igual en
   NVIDIA/AMD/Intel sin conversión.
2. Para difusión, el equivalente emergente es `stable-diffusion.cpp` con backend
   Vulkan — experimental, no listo para reemplazar nuestro pipeline ONNX, pero
   es el candidato de rescate para GPUs sin EP nativo si algún día hace falta.
3. Señal de ecosistema: Collabora está portando RADV (driver Vulkan open-source
   de AMD) a Win32.

## Qué significa para Upflow

Hoy: un solo runtime (onnxruntime-directml) para todo. Plan aditivo — DirectML
queda como baseline universal (decisión explícita de Santiago), lo nativo se
suma por dispositivo detectado. El formato de modelos NO cambia: los packs y
conversiones actuales sirven para todos los vendors (beneficio directo pedido:
"facilita descargar modelos" = un solo artefacto por modelo, siempre).

### Fase 1 — Selector de EP por dispositivo (bajo riesgo, gana NVIDIA e Intel ya)

- Generalizar la creación de sesiones ONNX (upscaler ONNX, generación, Apollo,
  AudioSR, GMFSS) detrás de un `ep_registry`: dado un device y el hardware
  detectado, elegir EP nativo si está disponible y sano; sino DirectML; sino CPU.
  Mismo patrón que `backend_registry` de SP11 y `restorer_registry` de SP9.
- NVIDIA: probar el **plugin ABI EP de TensorRT-RTX** (standalone, se registra
  sobre el runtime existente) — evita el conflicto clásico de paquetes
  onnxruntime-directml vs onnxruntime-gpu. Si el plugin no convive con nuestro
  onnxruntime-directml pineado, gate por spike antes de comprometer.
- Intel: `onnxruntime-openvino` / OpenVINO EP con el mismo gate.
- Primer arranque por EP nativo compila/cachea (TensorRT engines, OpenVINO AOT):
  UX de "preparando aceleración para tu GPU" reutilizando la visibilidad de
  conversiones de v0.22.0.
- Validación: no tenemos hardware NVIDIA/Intel local — smoke con el amigo
  (RX 7900 XT no aplica; buscar tester NVIDIA) o gate funcional + fallback
  automático a DirectML ante cualquier fallo de sesión (patrón stream-pipeline:
  el job nunca falla por el camino nuevo).

### Fase 2 — Windows ML como runtime del sistema (el destino)

- Adapter WinML detrás del mismo selector: en Windows 11 24H2+, dejar que el
  catálogo del sistema resuelva el EP (incluido el futuro MIGraphX para AMD) y
  las actualizaciones lleguen por Windows Update en vez de nuestros wheels.
- Gate: versión mínima de Windows del usuario + paridad de operadores con
  nuestros grafos (uint8 IO, io16) — spike antes de plan.

### Fase 3 — opcional/experimental

- Lane `stable-diffusion.cpp` Vulkan para difusión en hardware sin EP (mismo rol
  que GMFSS: opt-in, "experimental").

### Riesgos honestos

- TensorRT-RTX/OpenVINO recompilan por GPU/driver: primer uso lento por modelo
  (mitigado con caches EP context + copy de expectativa).
- Convivencia de paquetes onnxruntime: el motivo por el que el spike de Fase 1
  va ANTES de cualquier promesa. El plugin ABI existe justo para esto, pero hay
  que verificarlo con nuestro pin de onnxruntime-directml 1.24.x.
- MIGraphX-en-WinML para discretas RDNA3 no confirmado: para nuestro propio
  hardware el plan no cambia nada a corto plazo (DirectML sigue siendo el camino
  AMD-ONNX); la ganancia inmediata es para usuarios NVIDIA/Intel.

## Recomendación

Ejecutar Fase 1 como spike gateado (¿convive el plugin EP con nuestro runtime?
¿cuánto gana un SDXL real en una RTX?) antes de comprometer el feature. Si el
spike confirma los números de vendor (+50% NVIDIA), el selector de EP es la
mejor relación esfuerzo/beneficio del backlog de rendimiento — y no toca ni un
modelo ya distribuido.
