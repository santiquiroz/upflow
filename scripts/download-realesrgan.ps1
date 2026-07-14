$ErrorActionPreference = 'Stop'

$root = Split-Path -Parent $PSScriptRoot
$vendorDir = Join-Path $root 'vendor\realesrgan'
$binaryPath = Join-Path $vendorDir 'realesrgan-ncnn-vulkan.exe'
$modelsDir = Join-Path $vendorDir 'models'
$tempDir = Join-Path $root 'runtime\temp'
$zipPath = Join-Path $tempDir 'realesrgan-ncnn-vulkan.zip'
$extractDir = Join-Path $tempDir 'realesrgan-ncnn-vulkan-extract'

# Pinned bundle from the main Real-ESRGAN repo (same 2022-04-24 build as the
# Real-ESRGAN-ncnn-vulkan v0.2.0 release, but this asset also ships the
# models\ folder the standalone v0.2.0 zip lacks entirely).
$downloadUrl = 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesrgan-ncnn-vulkan-20220424-windows.zip'

# Checking a model file (not just models\) guards against a previous
# partially-completed install.
if ((Test-Path $binaryPath) -and (Test-Path (Join-Path $modelsDir 'realesrgan-x4plus.param'))) {
    Write-Host 'Real-ESRGAN NCNN already present at:' $vendorDir
    Write-Host 'Delete vendor\realesrgan to force a re-download.'
    return
}

New-Item -ItemType Directory -Force -Path $vendorDir | Out-Null
New-Item -ItemType Directory -Force -Path $modelsDir | Out-Null
New-Item -ItemType Directory -Force -Path $tempDir | Out-Null

try {
    if (Test-Path $extractDir) {
        Remove-Item -Recurse -Force $extractDir
    }
    New-Item -ItemType Directory -Force -Path $extractDir | Out-Null

    Invoke-WebRequest -Uri $downloadUrl -OutFile $zipPath
    Expand-Archive -Path $zipPath -DestinationPath $extractDir -Force

    # Release zips vary in nesting (the v0.2.0 zip nests under a versioned
    # folder; the main-repo bundle extracts flat), so locate the exe
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

    # Merge per-file (not folder replace) so user-added custom models
    # (UltraSharp, Remacri, etc.) already in vendor\realesrgan\models survive.
    $modelsSource = Join-Path $contentRoot.FullName 'models'
    if (-not (Test-Path $modelsSource)) {
        throw 'models folder not found inside the downloaded archive.'
    }
    Get-ChildItem $modelsSource -File | ForEach-Object {
        Copy-Item -Force $_.FullName (Join-Path $modelsDir $_.Name)
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
Write-Host 'Models:' $modelsDir
Write-Host 'If you want extra custom ESRGAN models (UltraSharp, Remacri, etc.), place them in vendor\realesrgan\models.'
