# Smoke: aceleración nativa por GPU (plugins EP de ONNX Runtime)

Prueba si tu GPU acelera con su tecnología nativa (TensorRT-RTX en NVIDIA,
WebGPU en cualquiera) comparado con DirectML. Autocontenido: no descarga
modelos, genera uno chico en memoria. Tarda ~2 minutos.

## Requisitos

- Windows 11, Python 3.11 o 3.12 (`python --version`)
- NVIDIA: driver reciente (Game Ready/Studio 576+; GPU RTX 30xx o más nueva)
- AMD/Intel: solo el driver normal de la GPU

## Pasos (PowerShell)

```powershell
cd carpeta\donde\descargaste\esto
python -m venv ep-smoke-venv
.\ep-smoke-venv\Scripts\Activate.ps1
pip install onnxruntime-directml==1.24.4 numpy onnx

# GPU NVIDIA (RTX 30xx+): el plugin TensorRT-RTX (~100 MB)
# OJO el --no-deps: sin eso pip pisa onnxruntime-directml
pip install --no-deps onnxruntime-ep-nv-tensorrt-rtx-cu12

# Cualquier GPU (opcional, segundo lane a comparar):
pip install --no-deps onnxruntime-ep-webgpu

python smoke_ep_plugin.py
```

## Qué mandar de vuelta

Todo lo que imprime desde `=== RESULTADO ===` hasta el final (es un JSON).
Si algo explota antes, mandá la pantalla completa.

## Qué esperar

- `dml`: tu baseline actual (lo que usa Upflow hoy).
- `trt_rtx_inferencia`: si aparece `OK` con `X.XXx vs DML`, ese es el número
  que buscamos (NVIDIA promete ~1.5x). La primera creación de sesión compila
  el motor y puede tardar — eso es normal y se cachea.
- `webgpu_inferencia`: comparación del lane WebGPU en tu GPU.
- `SKIP`/`EMPTY` no son errores: significan "ese plugin no aplica a esta GPU".
- `dml_post_plugins OK`: confirma que registrar plugins no rompe lo existente.
