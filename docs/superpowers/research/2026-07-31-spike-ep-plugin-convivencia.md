# Spike Fase 1a: convivencia de plugins EP nativos con onnxruntime-directml 1.24.4

Ejecutado 2026-07-31 en el rig AMD (RX 7800 XT + iGPU, Windows 11 26200, Python 3.12).
Gate del plan de [aceleración nativa por vendor](2026-07-31-aceleracion-nativa-por-vendor.md).

## Veredicto: PASA

Los plugins EP standalone conviven con nuestro pin `onnxruntime-directml==1.24.4`
en el mismo proceso. Verificado empíricamente, no solo por docs:

| Prueba | Resultado |
|---|---|
| pip: `onnxruntime-directml` + plugin NVIDIA (`--no-deps`) + DLLs OpenVINO + `openvino` + `tensorrt-rtx` SDK | Sin colisión de módulos — namespaces distintos |
| API plugin ABI en el wheel directml 1.24.4 | Presente y funcional: `register_execution_provider_library`, `get_ep_devices`, `SessionOptions.add_provider_for_devices`, `set_provider_selection_policy` |
| Registro plugin OpenVINO (DLL del NuGet Intel 1.6.1) sobre directml | **REGISTRA OK**; enumera 0 devices en hardware AMD (gate de vendor limpio, sin error) |
| Registro plugin TensorRT-RTX (wheel pip 0.3.0) sobre directml | Falla limpio en AMD: `tensorrt_rtx_1_5.dll` importa estáticamente `nvcuda.dll` (driver NVIDIA). Excepción catcheable, proceso sano |
| **Inferencia end-to-end vía plugin** (webgpu 0.2.1, compilado contra ORT ~1.27) | **FUNCIONA sobre host 1.24.4**: registró, enumeró las GPUs AMD, corrió Conv correcto. Prueba el camino completo Y la forward-compat del ABI (append-only) |
| DML después de registros fallidos/exitosos | Intacto. Sesiones DML idénticas antes/después. El camino nuevo no puede romper el existente |

## Artefactos exactos (jul-2026)

- **NVIDIA**: `pip install --no-deps onnxruntime-ep-nv-tensorrt-rtx-cu12` (0.3.0, ~103 MB,
  autocontenido: plugin + TRT-RTX 1.5 + cudart). Sin `--no-deps` pisa onnxruntime-directml
  (declara dep `onnxruntime>=1.24`). Alternativa sin pip: zips en
  github.com/NVIDIA/TensorRT-RTX-EP-ABI/releases. Requiere RTX 30xx+ (EP), driver 576+.
  El EP embebido en el árbol de onnxruntime está deprecado oficialmente.
- **Intel**: NuGet `Intel.ML.OnnxRuntime.EP.OpenVINO` 1.6.1 (MIT, ~116 MB, OpenVINO runtime
  embebido). No hay wheel pip aún — el .nupkg es un zip; DLL en
  `runtimes/win-x64/native/onnxruntime_providers_openvino_plugin.dll` (+ 18 DLLs de soporte).
- **Windows ML** (Fase 2): mismos plugins vía catálogo del sistema (24H2+), KB5089168.

## Reglas para ep_registry (aprendidas empíricamente)

1. **Gate por hardware ANTES de registrar**: TRT-RTX ni carga sin driver NVIDIA
   (nvcuda.dll). Registrar solo si `get_ep_devices()`/detección propia ve GPU del vendor.
   El fallo igual es limpio — el gate es para no ensuciar logs.
2. **Preload de deps**: ORT no resuelve las dependencias del plugin desde el dir del
   plugin (probado: `add_dll_directory` solo NO alcanza; el diagnóstico de deps de ORT
   además culpa al DLL equivocado). Antes de registrar: `os.add_dll_directory(dir)` +
   `ctypes.WinDLL` de cada DLL bundled.
3. **`get_available_providers()` NO lista plugins** (by design, issue #27832).
   Usar `get_ep_devices()` filtrando por `ep_name`.
4. **Sesión nativa**: `SessionOptions.add_provider_for_devices(devices, opts)`.
   Fallback DML→CPU ante CUALQUIER excepción del camino nuevo (patrón stream-pipeline).
5. **Nunca registrar como plugin un EP ya embebido en el build** (issue #29372,
   double-free). En nuestro wheel solo DML/CPU están embebidos — sin riesgo con
   TRT-RTX/OpenVINO, pero regla fija para el registry.
6. **Sin zero-copy DML↔plugin** en 1.24.x (issue #26821 abierto): cada sesión vive
   entera en un EP; no mezclar EPs dentro de un mismo job.

## Riesgos vigentes

- `onnxruntime-directml` está congelado en 1.24.4 (vanilla va por 1.28; sin release DML
  desde mar-2026). El ABI append-only nos protege (webgpu 1.27 corrió sobre 1.24.4),
  pero el destino de largo plazo es Windows ML (Fase 2 del plan).
- Números TRT-RTX reales pendientes de tester NVIDIA (ver abajo). NVIDIA promete +50%
  vs DML; el spike solo prueba que el camino existe y no rompe nada.
- Enumeración OpenVINO en CPU AMD: 0 devices — el plugin Intel exige hardware Intel
  incluso para su lane CPU. La ganancia Intel es solo para usuarios Intel (esperado).

## Medición local (RX 7800 XT, modelo sintético conv x2, 512→1024, uint8 IO)

| EP | ms/frame | vs DML |
|---|---|---|
| CPU | 28.7 | 0.08x |
| **DirectML** | **2.4** | **1.00x** |
| WebGPU plugin | 9.5 | 0.25x |

WebGPU no compite con DML en AMD para este workload — descartado como lane AMD
por ahora (modelo sintético chico; si algún día se revisa, medir con modelo real).
DirectML confirma su rol de baseline universal.

## Smoke para testers (NVIDIA / RX 7900 XT)

`tools/ep-spike/` — autocontenido (genera el modelo en memoria, no descarga nada):
`README.md` con pasos de venv + `smoke_ep_plugin.py` que imprime JSON copy-pasteable.
Cubre: baseline CPU/DML, registro TRT-RTX + inferencia + speedup vs DML, lane WebGPU
opcional, y verificación de que DML sigue vivo tras registrar plugins.

Lo que buscamos del tester NVIDIA: `trt_rtx_inferencia OK` con su `X.XXx vs DML`
(y el tiempo de compilación de primer uso, que en producción se mitiga con cache
EP context + copy de expectativa estilo v0.22.0).
