$ErrorActionPreference = 'Stop'

# Older Windows PowerShell 5.1 defaults to TLS 1.0, which GitHub rejects.
[Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12

$root = Split-Path -Parent $PSScriptRoot
$vendorDir = Join-Path $root 'vendor\apollo'
$modelPath = Join-Path $vendorDir 'apollo.onnx'

# Apollo (JusperLee/Apollo, CC BY-SA 4.0) exported to ONNX for the EXPERIMENTAL
# audio-restoration mode. Runs on any ONNX Runtime EP (DirectML / CPU / CUDA).
# The .onnx is a derived artifact of the CC BY-SA weights; attribution:
# https://github.com/JusperLee/Apollo (arXiv:2409.08514). Hosted as a release asset.
$releaseTag = 'v0.2.0'
$downloadUrl = "https://github.com/santiquiroz/upflow/releases/download/$releaseTag/apollo.onnx"

if (Test-Path $modelPath) {
    Write-Host 'Apollo restore model already present at:' $modelPath
    Write-Host 'Delete vendor\apollo\apollo.onnx to force a re-download.'
    return
}

New-Item -ItemType Directory -Force -Path $vendorDir | Out-Null

Write-Host 'Downloading Apollo audio-restore ONNX model (~74 MB)...'
Invoke-WebRequest -Uri $downloadUrl -OutFile $modelPath -UseBasicParsing

if (-not (Test-Path $modelPath)) {
    throw 'La descarga del modelo Apollo fallo (apollo.onnx no quedo en vendor\apollo).'
}

Write-Host 'Apollo restore model listo en:' $modelPath
Write-Host 'Activa el modo con ENABLE_AUDIO_RESTORE=true en .env.'
