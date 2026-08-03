# Aceleradores nativos por marca — hallazgos verificados (2026-08-01)

Qué se puede shippear de verdad para NVIDIA e Intel, y qué había que arreglar antes
de que sirviera. Lo que no se pudo comprobar en esta máquina está marcado como tal.

## Alcance: para AMD no hay un tercero

DirectML y Vulkan **ya son** el camino nativo de AMD en Windows. No existe un
execution provider de plugin de AMD para onnxruntime en Windows que agregue algo
sobre DirectML. Los aceleradores que se agregan son dos: NVIDIA e Intel.

## Los artefactos

| | Paquete | Baja | En disco | Licencia |
|---|---|---|---|---|
| NVIDIA | `onnxruntime-ep-nv-tensorrt-rtx-cu13==0.3.0` (pip) | 93 MB | 246 MB | Apache-2.0 el código, binarios propietarios NVIDIA bajo el SLA de TensorRT-for-RTX |
| Intel | NuGet `Intel.ML.OnnxRuntime.EP.OpenVINO` 1.6.1 | 116 MB | 60,5 MB recortado | MIT (redistribuye OpenVINO/oneTBB Apache-2.0) |

Ninguno viaja dentro del instalador: se bajan en la máquina del usuario, igual que el
resto de los packs opcionales. En el caso de NVIDIA eso además evita tener que
propagar los términos del SLA en nuestro EULA.

### Trampas de nombre, las dos reales

- **NVIDIA**: `onnxruntime-ep-nv-tensorrt-rtx` es un **meta-paquete vacío** (2 KB) que
  solo declara `-cu13` como dependencia. Instalarlo con `--no-deps` deja una
  instalación sin binarios que no expone `get_library_path()`. Hay que pedir el
  `-cu13` directo. *Lo pisé: el script falló ruidosamente y por eso se detectó.*
  Existe además `onnxruntime-trt-rtx`, que **no** es un plugin sino un build completo
  de onnxruntime y pisaría `onnxruntime-directml`.
- **Intel**: `Intel.ML.OnnxRuntime.OpenVino` (sin `.EP.`) es el paquete equivocado —
  modelo viejo de shared provider, exige ORT ≥ 1.25 y falla con Error 1114 en 1.24.4.
  El correcto lleva `.EP.` y aparece más abajo al buscar en NuGet.

## Lo que hubo que arreglar antes

**Los plugins no llegaban a la mitad de la app.** `generation_onnx` y `transcribe_onnx`
armaban providers DirectML a mano en vez de pasar por el `ep_registry`. Con un plugin
instalado, escalado, editor, audio y GMFSS aceleraban, pero generar imágenes y
transcribir seguían en DirectML sin ninguna señal.

No era un cambio de una línea: `create_session` devuelve una sesión ya construida y
optimum/transformers la construyen ellos. Y un EP de plugin **no se puede pedir por
nombre**, porque optimum valida el provider contra `get_available_providers()`, que no
ve plugins. La forma que sí funciona es `providers=[]` con el EP puesto en el
`SessionOptions`. Eso es `ep_registry.loader_kwargs()`.

Verificado contra onnxruntime real en esta máquina, no deducido:

- `providers=[]` + `SessionOptions` con `add_provider_for_devices` → sesión funcionando.
- **Reusar el mismo `SessionOptions` en varias sesiones funciona**, que es lo que hace
  un pipeline de difusión (unet, vae, encoders).
- **Llamar `add_provider_for_devices` dos veces sobre el mismo objeto falla** con
  "Provider has already been registered". Por eso se arma uno nuevo por llamada.

**Bug de multi-GPU, latente y activado al shippear.** `_native_plugin_for` ignoraba el
dispositivo: devolvía el primer plugin con devices para cualquier `dml:N`. En una
máquina con integrada Intel y placa NVIDIA, un trabajo fijado a `dml:0` corría en la
NVIDIA **en silencio**. Ahora se resuelve el vendor del adaptador N y solo se usa el
plugin de esa marca. Los EP del catálogo de Windows ML no declaran placa, así que solo
se usan cuando hay una sola GPU.

**`EP_PLUGINS_DIR` venía vacío por defecto**, o sea que los DLLs de Intel se instalaban
donde nadie los miraba. Ahora apunta a `vendor/ep-plugins`, que es donde los deja el
script.

**El pin de onnxruntime era `>=1.19`** y la API de plugins no existe antes de 1.24.

## Cobertura real, sin marketing

- **NVIDIA**: gate duro en Compute Capability 8.0, con el mensaje literal dentro del
  DLL. Cubre RTX 30xx, 40xx y 50xx. **Las RTX 20xx y las GTX 16xx quedan afuera.**
  Necesita driver NVIDIA: el DLL importa `nvml.dll`, que no viene en el wheel.
- **Intel**: la GPU exige driver Intel (el plugin importa `OpenCL.dll`, que tampoco
  viene). Sin driver el device no aparece y se cae a DirectML, limpio.

Del paquete de Intel se descarta el plugin de CPU (45 MB — onnxruntime ya trae su
propio provider de CPU), el de NPU (7 MB, inútil sin su compilador de 76 MB) y
`tbbmalloc`/`tbbmalloc_proxy`, que enganchan el allocator del proceso sin que el
plugin los necesite.

## Verificado en esta máquina (AMD)

Con el plugin de Intel físicamente presente en `vendor/ep-plugins/openvino/`:

```
adaptadores -> ['0x1002', '0x1002', '0x1002']
create_session dml:0 -> [2. 4. 6. 8.] | ['DmlExecutionProvider', 'CPUExecutionProvider']
estado reportado -> baseline | DirectML
```

El gate por marca evita siquiera intentar cargar el DLL de Intel, y el estado reportado
dice la verdad en vez de anunciar un acelerador que no se está usando.

## NO verificado

**Que los aceleradores realmente aceleren.** Esta máquina es AMD: no hay GPU NVIDIA ni
silicio Intel. Se comprobó que los artefactos existen, que instalan, que conviven con
`onnxruntime-directml` sin pisarlo, que el registro y el fallback funcionan y que el
plugin de Intel se registra sin romper DirectML. Lo que falta es una máquina con RTX
30xx+ o con gráfica Intel para confirmar que `get_ep_devices()` publica el device y que
la inferencia corre por ahí. **No hay ningún número de rendimiento medido.**
