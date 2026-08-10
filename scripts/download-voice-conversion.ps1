$ErrorActionPreference = 'Stop'

# Conversion de voz: decir lo mismo con la voz de otra grabacion.
#
# Son TRES piezas y el pack no sirve sin las tres:
#   vendor\speecht5-vc        SpeechT5 VC (MIT)
#   vendor\speecht5-hifigan   el vocoder (MIT)
#   vendor\xvector\tdnn.onnx  el encoder de x-vector (Apache-2.0)
#
# El x-vector se baja de un port propio (santiquiroz/port-xvector-onnx): es el
# TDNN de speechbrain/spkrec-xvect-voxceleb exportado a ONNX. Antes habia que
# generarlo a mano en un venv con torch+speechbrain (~2 GB) y este script solo
# avisaba con un Write-Warning y terminaba en 0, asi que la app marcaba el pack
# instalado y la tarjeta decia "disponible" con la conversion rota.
#
# Los dos exports publicos de terceros NO sirven (probados 2026-08-05): uno no
# inicializa la sesion y el otro devuelve embeddings fuera del espacio de
# SpeechT5, con lo que la conversion sale en NaN.

[Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12

$root = Split-Path -Parent $PSScriptRoot
$venv = Join-Path $root '.venv\Scripts\python.exe'
if (-not (Test-Path $venv)) { $venv = 'python' }

$destino = Join-Path $root 'vendor'
$vcDir = Join-Path $destino 'speecht5-vc'
$vocoderDir = Join-Path $destino 'speecht5-hifigan'
$xvectorDir = Join-Path $destino 'xvector'
$xvector = Join-Path $xvectorDir 'tdnn.onnx'

$xvectorUrl = 'https://github.com/santiquiroz/port-xvector-onnx/releases/download/models-v1.0/tdnn.onnx'
$xvectorSha = 'dfb4daeb9b0a9aa33b0993e35f22841b34718fef1cfdcded2c7ff75867ddc7f8'
$xvectorBytes = 16897902

function Get-Sha256([string]$path) {
    # .NET directo y no Get-FileHash: ese cmdlet vive en un modulo, y un server
    # lanzado con PSModulePath contaminado (pasado real 2026-08-09) deja al
    # powershell 5.1 hijo sin poder resolverlo.
    $sha256 = [System.Security.Cryptography.SHA256]::Create()
    try {
        $stream = [System.IO.File]::OpenRead($path)
        try {
            $hash = $sha256.ComputeHash($stream)
        } finally {
            $stream.Dispose()
        }
    } finally {
        $sha256.Dispose()
    }
    return ([System.BitConverter]::ToString($hash) -replace '-', '').ToLowerInvariant()
}

if (-not ((Test-Path (Join-Path $vcDir 'config.json')) -and (Test-Path (Join-Path $vocoderDir 'config.json')))) {
    Write-Host 'Descargando microsoft/speecht5_vc y su vocoder (MIT)...'
    & $venv -c @"
from huggingface_hub import snapshot_download
import sys
raiz = sys.argv[1]
for repo, carpeta in (('microsoft/speecht5_vc','speecht5-vc'), ('microsoft/speecht5_hifigan','speecht5-hifigan')):
    destino = f'{raiz}/{carpeta}'
    snapshot_download(repo, local_dir=destino)
    print(f'  {repo} -> {destino}')
"@ $destino
    if ($LASTEXITCODE -ne 0) {
        throw "La descarga de SpeechT5 fallo (codigo $LASTEXITCODE)."
    }
} else {
    Write-Host "Ya presentes: SpeechT5 VC y su vocoder en $destino"
}

if (Test-Path $xvector) {
    Write-Host "Ya presente: encoder de x-vector en $xvector"
} else {
    Write-Host 'Descargando el encoder de x-vector (Apache-2.0)...'
    New-Item -ItemType Directory -Force -Path $xvectorDir | Out-Null
    $parcial = "$xvector.part"
    if (Test-Path $parcial) { Remove-Item -Force $parcial }
    Invoke-WebRequest -Uri $xvectorUrl -OutFile $parcial -UseBasicParsing

    # Se verifica ANTES de moverlo al nombre final: un archivo corrupto en su
    # sitio pasaria el chequeo de existencia y la conversion saldria en NaN.
    $tamano = (Get-Item $parcial).Length
    if ($tamano -ne $xvectorBytes) {
        Remove-Item -Force $parcial
        throw "El x-vector bajo con $tamano bytes y se esperaban $xvectorBytes."
    }
    $hash = Get-Sha256 $parcial
    if ($hash -ne $xvectorSha) {
        Remove-Item -Force $parcial
        throw "El sha256 del x-vector no coincide. Esperado $xvectorSha, obtenido $hash."
    }
    Move-Item -Force $parcial $xvector
}

# Verificacion final: el pack se da por instalado SOLO si estan las tres piezas.
# Terminar en 0 con una faltando es lo que hacia que la app marcara el pack
# instalado y la tarjeta prometiera una conversion que el motor rechazaba.
$requeridos = @{
    'SpeechT5 VC'         = Join-Path $vcDir 'config.json'
    'vocoder HiFi-GAN'    = Join-Path $vocoderDir 'config.json'
    'encoder de x-vector' = $xvector
}
$faltantes = @($requeridos.Keys | Where-Object { -not (Test-Path $requeridos[$_]) } | Sort-Object)
if ($faltantes.Count -gt 0) {
    throw "El paquete de conversion de voz quedo incompleto; falta: $($faltantes -join ', '). Volve a correr la descarga."
}

Write-Host "Conversion de voz lista en: $destino"
Write-Host 'SpeechT5 VC y su vocoder son MIT; el encoder de x-vector es Apache-2.0'
Write-Host '(speechbrain/spkrec-xvect-voxceleb, exportado en santiquiroz/port-xvector-onnx).'
