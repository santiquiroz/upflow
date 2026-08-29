$ErrorActionPreference = 'Stop'

# Transcripcion por stem a MIDI/MusicXML/tab (F3a): Basic Pitch ONNX
# (spotify/basic-pitch, Apache-2.0), un solo archivo de ~230 KB. Hosteado
# directo en el repo de origen (raw.githubusercontent.com), no en un mirror.

[Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12

$root = Split-Path -Parent $PSScriptRoot
$vendorDir = Join-Path $root 'vendor\music-transcription'
$modelPath = Join-Path $vendorDir 'nmp.onnx'

$modelUrl = 'https://raw.githubusercontent.com/spotify/basic-pitch/main/basic_pitch/saved_models/icassp_2022/nmp.onnx'
$modelSha256 = '2c3c1d144bfa61ad236e92e169c13535c880469a12a047d4e73451f2c059a0ec'
$modelBytes = 230444

function Get-Sha256([string]$path) {
    # .NET directo y no Get-FileHash: ese cmdlet vive en un modulo, y un server
    # lanzado con PSModulePath contaminado deja al powershell 5.1 hijo sin
    # poder resolverlo. .NET siempre esta.
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

if (Test-Path $modelPath) {
    $tamano = (Get-Item $modelPath).Length
    if ($tamano -eq $modelBytes) {
        $hash = Get-Sha256 $modelPath
        if ($hash -eq $modelSha256) {
            Write-Host 'Modelo de transcripcion (Basic Pitch) ya presente en:' $modelPath
            Write-Host 'Borra el archivo para forzar una re-descarga.'
            return
        }
    }
    Write-Host 'El archivo existente no pasa la verificacion; se vuelve a bajar.'
    Remove-Item -Force $modelPath
}

New-Item -ItemType Directory -Force -Path $vendorDir | Out-Null

$parcial = "$modelPath.part"
if (Test-Path $parcial) { Remove-Item -Force $parcial }

Write-Host 'Descargando el modelo de transcripcion Basic Pitch (~230 KB)...'
Invoke-WebRequest -Uri $modelUrl -OutFile $parcial -UseBasicParsing

# Se verifica ANTES de moverlo al nombre final: un archivo corrupto en su
# sitio pasaria el chequeo de existencia y la transcripcion saldria en NaN.
$tamano = (Get-Item $parcial).Length
if ($tamano -ne $modelBytes) {
    Remove-Item -Force $parcial
    throw "El modelo de transcripcion bajo con $tamano bytes y se esperaban $modelBytes."
}
$hash = Get-Sha256 $parcial
if ($hash -ne $modelSha256) {
    Remove-Item -Force $parcial
    throw "El sha256 del modelo de transcripcion no coincide. Esperado $modelSha256, obtenido $hash."
}
Move-Item -Force $parcial $modelPath

Write-Host 'Modelo de transcripcion listo en:' $modelPath
Write-Host 'Basic Pitch (spotify/basic-pitch) es Apache-2.0.'
Write-Host 'Con esto se habilita transcribir stems a MIDI/MusicXML/tab en el modulo Audio.'
