$ErrorActionPreference = 'Stop'

# Generacion 3D desde una FOTO, con Shap-E img2img de OpenAI.
#
# Es OTRO repo que el de texto (openai/shap-e-img2img, MIT igual que aquel):
# comparte el renderer pero cambia el prior y trae un image_encoder CLIP en vez
# de tokenizer + text_encoder. Verificado el 2026-08-08 cargando la tuberia
# real en CPU con estos archivos y nada mas.
#
# Corre en CPU: 16 pasos con guidance 3.0 dan ~100-136 s por malla, y las mallas
# salen estancas y manifold directamente.
#
# Lo que NO da, y hay que decirlo: es una INTERPRETACION del objeto, no una
# replica. Una imagen sin pistas 3D (un dibujo plano) colapsa a una extrusion
# plana de la silueta. Y como todo Shap-E, tampoco da COTAS.

$root = Split-Path -Parent $PSScriptRoot
$venv = Join-Path $root '.venv\Scripts\python.exe'
if (-not (Test-Path $venv)) { $venv = 'python' }

$destino = Join-Path $root 'vendor\shap-e-img2img'
if (Test-Path (Join-Path $destino 'model_index.json')) {
    Write-Host "ya esta: Shap-E img2img en $destino"
    exit 0
}

Write-Host "Descargando openai/shap-e-img2img (MIT, ~2,1 GB) ..."

# El here-string es LITERAL (@'...'@) y la ruta viaja por variable de entorno.
# Con el interpolante (@"..."@) PowerShell procesa el backtick como escape: un
# comentario que decia `renderer/` se convertia en un retorno de carro seguido de
# "enderer/", partia la linea y Python moria con SyntaxError sin bajar nada.
# Pasado el 2026-08-05 en la instalacion real de shap-e. Por eso: nada de
# escapes, y el codigo va a un archivo en vez de a un argumento.
$env:UPFLOW_SHAPE_IMG_DEST = $destino
$scriptPy = Join-Path $env:TEMP 'upflow-download-shap-e-img2img.py'

@'
import os

from huggingface_hub import snapshot_download

# Solo lo que la tuberia usa de verdad, medido leyendo el repo archivo por
# archivo el 2026-08-08:
#   - prior e image_encoder tienen fp16 en safetensors: se baja esa variante.
#   - shap_e_renderer NO tiene fp16: solo existe el .bin, asi que se baja ese.
#     (diffusers cae al .bin con un warning; cargado y verificado en CPU.)
#   - la carpeta legacy renderer/ se excluye entera: la clave "renderer" del
#     model_index.json es legado que la tuberia ignora.
#   - image_processor/ es solo el preprocessor_config.json del CLIP.
# Total: 2,14 GB. Bajar el repo sin filtrar duplica eso largamente.
ruta = snapshot_download(
    'openai/shap-e-img2img',
    local_dir=os.environ['UPFLOW_SHAPE_IMG_DEST'],
    allow_patterns=[
        'model_index.json',
        '*/config.json',
        'image_processor/*',
        'scheduler/*',
        'prior/*.fp16.safetensors',
        'image_encoder/*.fp16.safetensors',
        'shap_e_renderer/*.bin',
    ],
)
print(ruta)
'@ | Set-Content -Path $scriptPy -Encoding UTF8

& $venv $scriptPy
Remove-Item -Force $scriptPy -ErrorAction SilentlyContinue

if (-not (Test-Path (Join-Path $destino 'model_index.json'))) {
    throw "La descarga termino pero falta model_index.json en $destino."
}
Write-Host "Shap-E img2img listo en: $destino"
Write-Host "Licencia MIT (OpenAI). Corre en CPU: unos dos minutos por pieza."
Write-Host "OJO: genera una interpretacion del objeto, no una replica, y sin cotas."
Write-Host "Mejor con fotos reales del objeto; un dibujo plano sale como extrusion plana."
