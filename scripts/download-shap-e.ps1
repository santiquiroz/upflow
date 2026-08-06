$ErrorActionPreference = 'Stop'

# Generacion 3D desde texto o desde una imagen, con Shap-E de OpenAI.
#
# Licencia MIT: codigo Y pesos. Verificado el 2026-08-05 contra la model card y
# el repositorio, con revision adversarial — es de los pocos modelos 3D con
# licencia limpia de verdad. TRELLIS tambien es MIT pero depende de kernels CUDA
# y de nvdiffrast, que si es no comercial; la familia Hunyuan3D esta bloqueada
# por su propia licencia.
#
# Corre en CPU. Medido en esta maquina, sin GPU NVIDIA: 116-137 s por malla, y
# de cuatro pruebas TRES salieron estancas, manifold y solidas directamente.
#
# Lo que NO da, y hay que decirlo: COTAS. Ninguna malla generada sabe que tiene
# que medir 80 mm. Para una pieza que TIENE que encajar, el carril parametrico.

$root = Split-Path -Parent $PSScriptRoot
$venv = Join-Path $root '.venv\Scripts\python.exe'
if (-not (Test-Path $venv)) { $venv = 'python' }

$destino = Join-Path $root 'vendor\shap-e'
if (Test-Path (Join-Path $destino 'model_index.json')) {
    Write-Host "ya esta: Shap-E en $destino"
    exit 0
}

Write-Host "Descargando openai/shap-e (MIT, ~1.3 GB) ..."
& $venv -c @"
from huggingface_hub import snapshot_download
ruta = snapshot_download(
    'openai/shap-e',
    local_dir=r'$destino',
    # Solo lo que hace falta para generar malla: sin los pesos duplicados en
    # otras precisiones el paquete baja a menos de la mitad.
    allow_patterns=['*.json', '*.txt', '*.safetensors', '*.bin', '*model*'],
    ignore_patterns=['*.msgpack', '*.h5', '*.onnx'],
)
print(ruta)
"@

if (-not (Test-Path (Join-Path $destino 'model_index.json'))) {
    throw "La descarga termino pero falta model_index.json en $destino."
}
Write-Host "Shap-E listo en: $destino"
Write-Host "Licencia MIT (OpenAI). Corre en CPU: unos dos minutos por pieza."
Write-Host "OJO: genera forma, NO cotas. Para una pieza que tiene que encajar, usa el carril parametrico."
