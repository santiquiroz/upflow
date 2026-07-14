$ErrorActionPreference = 'Stop'

$root = Split-Path -Parent $PSScriptRoot
$vendorDir = Join-Path $root 'vendor\realesrgan'
$binaryPath = Join-Path $vendorDir 'realesrgan-ncnn-vulkan.exe'
$tempDir = Join-Path $root 'runtime\temp'
$zipPath = Join-Path $tempDir 'realesrgan-ncnn-vulkan.zip'
$extractDir = Join-Path $tempDir 'realesrgan-ncnn-vulkan-extract'
$downloadUrl = 'https://github.com/xinntao/Real-ESRGAN-ncnn-vulkan/releases/download/v0.2.0/realesrgan-ncnn-vulkan-v0.2.0-windows.zip'

New-Item -ItemType Directory -Force -Path $vendorDir | Out-Null
New-Item -ItemType Directory -Force -Path $tempDir | Out-Null

try {
    if (Test-Path $extractDir) {
        Remove-Item -Recurse -Force $extractDir
    }
    New-Item -ItemType Directory -Force -Path $extractDir | Out-Null

    Invoke-WebRequest -Uri $downloadUrl -OutFile $zipPath
    Expand-Archive -Path $zipPath -DestinationPath $extractDir -Force

    # The release zip nests everything under a
    # realesrgan-ncnn-vulkan-v0.2.0-windows\ folder, so locate the exe
    # recursively and flatten its folder into vendorDir (app/config.py
    # expects the flat path vendor\realesrgan\realesrgan-ncnn-vulkan.exe).
    $exeSource = Get-ChildItem $extractDir -Filter 'realesrgan-ncnn-vulkan.exe' -Recurse -File | Select-Object -First 1
    if (-not $exeSource) {
        throw 'realesrgan-ncnn-vulkan.exe not found inside the downloaded archive.'
    }
    $contentRoot = $exeSource.Directory

    Copy-Item -Force $exeSource.FullName $binaryPath

    Get-ChildItem $contentRoot.FullName -File -Filter '*.dll' | ForEach-Object {
        Copy-Item -Force $_.FullName (Join-Path $vendorDir $_.Name)
    }

    $licenseSource = Join-Path $contentRoot.FullName 'LICENSE'
    if (Test-Path $licenseSource) {
        Copy-Item -Force $licenseSource (Join-Path $vendorDir 'LICENSE')
    }
}
finally {
    if (Test-Path $extractDir) {
        Remove-Item -Recurse -Force $extractDir
    }
    if (Test-Path $zipPath) {
        Remove-Item -Force $zipPath
    }
}

Write-Host 'Real-ESRGAN NCNN downloaded to:' $vendorDir
Write-Host 'Binary:' $binaryPath
Write-Host 'If you want extra custom ESRGAN models (UltraSharp, Remacri, etc.), place them in vendor\realesrgan\models.'
