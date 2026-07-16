$ErrorActionPreference = 'Stop'

# Older Windows PowerShell 5.1 defaults to TLS 1.0, which GitHub rejects.
[Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12

$root = Split-Path -Parent $PSScriptRoot
$vendorDir = Join-Path $root 'vendor\ffmpeg-dist'
$tempDir = Join-Path $root 'runtime\temp'
$zipPath = Join-Path $tempDir 'ffmpeg-win64.zip'
# Usamos el tag 'latest' (master nightly) a proposito: BtbN ya no publica
# builds de rama estable (n7.x) y sus autobuilds fechados se podan a los
# ~meses, asi que un pin fechado haria 404 para quien descargue mas tarde.
# 'latest' es el unico URL durable. CONSECUENCIA (binding): el pipeline SOLO
# puede usar flags de ffmpeg NO deprecados — master elimina flags viejos sin
# aviso. Ej.: '-vsync' fue removido (2026-07), usar '-fps_mode' en su lugar.
$downloadUrl = 'https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip'

New-Item -ItemType Directory -Force -Path $vendorDir | Out-Null
New-Item -ItemType Directory -Force -Path $tempDir | Out-Null

Invoke-WebRequest -Uri $downloadUrl -OutFile $zipPath
Expand-Archive -Path $zipPath -DestinationPath $vendorDir -Force

$ffmpegRoot = Get-ChildItem $vendorDir -Directory | Select-Object -First 1
if (-not $ffmpegRoot) {
    throw 'FFmpeg archive extracted but no directory was found.'
}

$targetDir = Join-Path $root 'vendor\ffmpeg'
if (Test-Path $targetDir) {
    Remove-Item -Recurse -Force $targetDir
}
New-Item -ItemType Directory -Force -Path $targetDir | Out-Null
Copy-Item -Recurse -Force (Join-Path $ffmpegRoot.FullName 'bin') (Join-Path $targetDir 'bin')
Copy-Item -Recurse -Force (Join-Path $ffmpegRoot.FullName 'doc') (Join-Path $targetDir 'doc')
Copy-Item -Recurse -Force (Join-Path $ffmpegRoot.FullName 'presets') (Join-Path $targetDir 'presets')
Copy-Item -Force (Join-Path $ffmpegRoot.FullName 'LICENSE.txt') (Join-Path $targetDir 'LICENSE.txt')

Write-Host 'FFmpeg downloaded to:' $targetDir
Write-Host 'ffmpeg:' (Join-Path $targetDir 'bin\ffmpeg.exe')
Write-Host 'ffprobe:' (Join-Path $targetDir 'bin\ffprobe.exe')
