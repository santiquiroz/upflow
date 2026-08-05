param(
    # Par de idiomas, como los publica OPUS-MT: en-es, es-en, en-pt, etc.
    [string]$Pair = 'en-es'
)

$ErrorActionPreference = 'Stop'

# Traduccion local. Un modelo POR PAR de idiomas, que es como los publica
# OPUS-MT. Los multilingues buenos (NLLB) son CC-BY-NC, o sea NO comerciales.
#
# Verificado el 2026-08-05: 93% de parecido en la ida y vuelta ingles -> espanol
# -> ingles, a 0,3 s por frase.

$root = Split-Path -Parent $PSScriptRoot
$venv = Join-Path $root '.venv\Scripts\python.exe'
if (-not (Test-Path $venv)) { $venv = 'python' }

if ($Pair -notmatch '^[a-z]{2,3}-[a-z]{2,3}$') {
    throw "El par tiene que verse como 'en-es'. Recibido: '$Pair'"
}

$destino = Join-Path $root "vendor\translation\opus-mt-$Pair"
if (Test-Path (Join-Path $destino 'onnx')) {
    Write-Host "ya esta: opus-mt-$Pair en $destino"
    exit 0
}

Write-Host "Descargando onnx-community/opus-mt-$Pair ..."
& $venv -c @"
from huggingface_hub import snapshot_download
import sys
repo, destino = sys.argv[1], sys.argv[2]
# Solo los pesos sin cuantizar: las variantes cuantizadas de este repo no se
# midieron, y en este proyecto una variante sin medir ya devolvio audio NaN.
snapshot_download(repo, local_dir=destino, allow_patterns=[
    '*.json', '*.txt', '*.model', '*.spm',
    'onnx/encoder_model.onnx', 'onnx/decoder_model.onnx', 'onnx/decoder_with_past_model.onnx',
])
print('listo:', destino)
"@ "onnx-community/opus-mt-$Pair" $destino

if (-not (Test-Path (Join-Path $destino 'onnx'))) {
    throw "La descarga termino pero falta la carpeta onnx en $destino."
}
$total = (Get-ChildItem -Path $destino -File -Recurse | Measure-Object -Property Length -Sum).Sum
Write-Host ("Total: {0:N1} MB" -f ($total / 1MB))
Write-Host 'Los modelos OPUS-MT en onnx-community son CC-BY-4.0: uso comercial permitido, con atribucion.'
