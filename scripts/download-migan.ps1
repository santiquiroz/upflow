$ErrorActionPreference = 'Stop'
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

# MI-GAN (Picsart) para el borrado rapido del Editor: rellena el hueco continuando
# el entorno, sin prompts ni pasos de difusion. Revision pinneada; tamano y sha256
# verificados 2026-07-31.
$revision = '1538c135034b8cfe7a8472f34d09c8a5a45b17a7'
$fileName = 'migan_pipeline_v2.onnx'
$expectedBytes = 28079181

$root = Split-Path -Parent $PSScriptRoot
$vendorDir = Join-Path $root 'vendor\migan'
$dest = Join-Path $vendorDir $fileName

if ((Test-Path $dest) -and ((Get-Item $dest).Length -eq $expectedBytes)) {
    Write-Host "MI-GAN ya esta en $dest"
    return
}

New-Item -ItemType Directory -Force $vendorDir | Out-Null
$url = "https://huggingface.co/andraniksargsyan/migan/resolve/$revision/$fileName"
Write-Host "Descargando $url"
Invoke-WebRequest -Uri $url -OutFile $dest -UseBasicParsing

if ((Get-Item $dest).Length -ne $expectedBytes) {
    throw "$fileName : tamano inesperado (descarga corrupta)"
}

Write-Host "MI-GAN listo en $dest"
