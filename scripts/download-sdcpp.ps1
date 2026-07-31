$ErrorActionPreference = 'Stop'
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

# stable-diffusion.cpp (leejet) build Vulkan para Windows x64 — lane experimental
# de difusion (Fase 3 del plan de aceleracion por vendor, 2026-07-31). Tag
# pinneado: los autobuilds master-* quedan publicados, no se podan como BtbN.
$releaseTag = 'master-805-e31a86c'
$assetName = 'sd-master-e31a86c-bin-win-vulkan-x64.zip'

$root = Split-Path -Parent $PSScriptRoot
$vendorDir = Join-Path $root 'vendor\sdcpp'
$tempDir = Join-Path $root 'runtime\temp'
$binaryPath = Join-Path $vendorDir 'sd.exe'

if (Test-Path $binaryPath) {
    Write-Host "sd.cpp ya esta en $binaryPath"
    return
}

New-Item -ItemType Directory -Force $vendorDir | Out-Null
New-Item -ItemType Directory -Force $tempDir | Out-Null

$zipPath = Join-Path $tempDir $assetName
$url = "https://github.com/leejet/stable-diffusion.cpp/releases/download/$releaseTag/$assetName"
Write-Host "Descargando $url"
Invoke-WebRequest -Uri $url -OutFile $zipPath -UseBasicParsing

Expand-Archive -Path $zipPath -DestinationPath $vendorDir -Force
Remove-Item $zipPath -Force

# Algunos zips traen subcarpeta: aplanar el exe al nivel del vendorDir.
if (-not (Test-Path $binaryPath)) {
    $found = Get-ChildItem $vendorDir -Recurse -Filter 'sd.exe' | Select-Object -First 1
    if ($null -eq $found) { throw "sd.exe no aparecio tras extraer $assetName" }
    Get-ChildItem $found.Directory.FullName | Move-Item -Destination $vendorDir -Force
}

Write-Host "sd.cpp listo en $binaryPath"
