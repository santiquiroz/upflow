$ErrorActionPreference = 'Stop'
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

# stable-diffusion.cpp (leejet) build Vulkan para Windows x64 — lane experimental
# de difusion (Fase 3 del plan de aceleracion por vendor, 2026-07-31). Tag
# pinneado: los autobuilds master-* quedan publicados, no se podan como BtbN.
$releaseTag = 'master-805-e31a86c'
$assetName = 'sd-master-e31a86c-bin-win-vulkan-x64.zip'

# Checkpoint default para que el lane funcione SIN que el usuario consiga un
# modelo por su cuenta (los .safetensors originales no sobreviven a la
# conversion ONNX de la app). DreamShaper 8 (SD1.5 fp16, ~2GB), URL pinneada
# por commit y verificada 2026-07-31 (HTTP 200, 2,132,625,894 bytes).
$modelUrl = 'https://huggingface.co/digiplay/DreamShaper_8/resolve/a5883e31f50b133f37342aadb482e1405cb4b008/dreamshaper_8.safetensors'

$root = Split-Path -Parent $PSScriptRoot
$vendorDir = Join-Path $root 'vendor\sdcpp'
$tempDir = Join-Path $root 'runtime\temp'
# El ejecutable del release se llama sd-cli.exe (sd.exe no existe; tambien
# viene un sd-server.exe que no usamos).
$binaryPath = Join-Path $vendorDir 'sd-cli.exe'
$modelsDir = Join-Path $vendorDir 'models'
$modelPath = Join-Path $modelsDir 'dreamshaper_8.safetensors'

if ((Test-Path $binaryPath) -and (Test-Path $modelPath)) {
    Write-Host "sd.cpp ya esta en $binaryPath (modelo incluido)"
    return
}

New-Item -ItemType Directory -Force $vendorDir | Out-Null
New-Item -ItemType Directory -Force $modelsDir | Out-Null
New-Item -ItemType Directory -Force $tempDir | Out-Null

if (-not (Test-Path $binaryPath)) {
    $zipPath = Join-Path $tempDir $assetName
    $url = "https://github.com/leejet/stable-diffusion.cpp/releases/download/$releaseTag/$assetName"
    Write-Host "Descargando $url"
    Invoke-WebRequest -Uri $url -OutFile $zipPath -UseBasicParsing

    Expand-Archive -Path $zipPath -DestinationPath $vendorDir -Force
    Remove-Item $zipPath -Force

    # Algunos zips traen subcarpeta: aplanar el exe al nivel del vendorDir.
    if (-not (Test-Path $binaryPath)) {
        $found = Get-ChildItem $vendorDir -Recurse -Filter 'sd-cli.exe' | Select-Object -First 1
        if ($null -eq $found) { throw "sd-cli.exe no aparecio tras extraer $assetName" }
        Get-ChildItem $found.Directory.FullName | Move-Item -Destination $vendorDir -Force
    }
}

if (-not (Test-Path $modelPath)) {
    $modelTemp = Join-Path $tempDir 'sdcpp-model.partial'
    Write-Host "Descargando checkpoint default (DreamShaper 8, ~2GB)..."
    Invoke-WebRequest -Uri $modelUrl -OutFile $modelTemp -UseBasicParsing
    Move-Item $modelTemp $modelPath -Force
}

Write-Host "sd.cpp listo en $binaryPath (modelo: $modelPath)"
