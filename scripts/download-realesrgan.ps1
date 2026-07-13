$ErrorActionPreference = 'Stop'

$root = Split-Path -Parent $PSScriptRoot
$vendorDir = Join-Path $root 'vendor\realesrgan'
$tempDir = Join-Path $root 'runtime\temp'
$zipPath = Join-Path $tempDir 'realesrgan-ncnn-vulkan.zip'
$downloadUrl = 'https://github.com/xinntao/Real-ESRGAN-ncnn-vulkan/releases/download/v0.2.0/realesrgan-ncnn-vulkan-v0.2.0-windows.zip'

New-Item -ItemType Directory -Force -Path $vendorDir | Out-Null
New-Item -ItemType Directory -Force -Path $tempDir | Out-Null

Invoke-WebRequest -Uri $downloadUrl -OutFile $zipPath
Expand-Archive -Path $zipPath -DestinationPath $vendorDir -Force

Write-Host 'Real-ESRGAN NCNN downloaded to:' $vendorDir
Write-Host 'If you want extra custom ESRGAN models (UltraSharp, Remacri, etc.), place them in vendor\realesrgan\models.'
