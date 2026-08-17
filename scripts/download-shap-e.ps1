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

Write-Host "Descargando openai/shap-e (MIT, ~1,7 GB) ..."

# El here-string es LITERAL (@'...'@) y la ruta viaja por variable de entorno.
# Con el interpolante (@"..."@) PowerShell procesa el backtick como escape: un
# comentario que decia `renderer/` se convertia en un retorno de carro seguido de
# "enderer/", partia la linea y Python moria con SyntaxError sin bajar nada.
# Pasado el 2026-08-05 en la instalacion real. Por eso: nada de escapes, y el
# codigo va a un archivo en vez de a un argumento.
$env:UPFLOW_SHAPE_DEST = $destino
$scriptPy = Join-Path $env:TEMP 'upflow-download-shap-e.py'

@'
import os

from huggingface_hub import snapshot_download

# Solo lo que hace falta para generar malla: sin los pesos duplicados en otras
# precisiones el paquete baja a menos de la mitad.
# El conjunto MINIMO, medido leyendo el repo archivo por archivo:
#   - prior y text_encoder tienen fp16 en safetensors: se baja esa.
#   - shap_e_renderer, que es el que la tuberia usa de verdad, NO tiene fp16:
#     solo existe el .bin, asi que se baja ese.
#   - la carpeta renderer se excluye entera: la tuberia no la usa y son 1,3 GB.
# Bajar todo sin filtrar da 4,6 GB; esto da menos de la mitad.
ruta = snapshot_download(
    'openai/shap-e',
    local_dir=os.environ['UPFLOW_SHAPE_DEST'],
    allow_patterns=[
        'model_index.json',
        '*/config.json',
        'scheduler/*',
        'tokenizer/*',
        'prior/*.fp16.safetensors',
        'text_encoder/*.fp16.safetensors',
        'shap_e_renderer/*.bin',
    ],
)
print(ruta)
'@ | Set-Content -Path $scriptPy -Encoding UTF8

& $venv $scriptPy
$downloadExitCode = $LASTEXITCODE
Remove-Item -Force $scriptPy -ErrorAction SilentlyContinue

if ($downloadExitCode -ne 0) {
    throw "La descarga de Shap-E fallo (codigo $downloadExitCode)."
}
if (-not (Test-Path (Join-Path $destino 'model_index.json'))) {
    throw "La descarga termino pero falta model_index.json en $destino."
}
Write-Host "Shap-E listo en: $destino"
Write-Host "Licencia MIT (OpenAI). Corre en CPU: unos dos minutos por pieza."
Write-Host "OJO: genera forma, NO cotas. Para una pieza que tiene que encajar, usa el carril parametrico."
