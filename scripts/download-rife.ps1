$ErrorActionPreference = 'Stop'

# Older Windows PowerShell 5.1 defaults to TLS 1.0, which GitHub rejects.
[Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12

$root = Split-Path -Parent $PSScriptRoot
$vendorDir = Join-Path $root 'vendor\rife'
$modelsDir = Join-Path $vendorDir 'models'
$binaryPath = Join-Path $vendorDir 'rife-ncnn-vulkan.exe'
$tempDir = Join-Path $root 'runtime\temp'
$zipPath = Join-Path $tempDir 'rife-ncnn-vulkan.zip'
$extractDir = Join-Path $tempDir 'rife-ncnn-vulkan-extract'

# TNTwise/rife-ncnn-vulkan (MIT, active fork of nihui/rife-ncnn-vulkan with
# newer v4.x models). Pinned tag for reproducibility instead of "latest";
# asset name verified against the release's actual asset list.
# https://github.com/TNTwise/rife-ncnn-vulkan/releases/tag/20250112
$releaseTag = '20250112'
$downloadUrl = "https://github.com/TNTwise/rife-ncnn-vulkan/releases/download/$releaseTag/windows.zip"

if ((Test-Path $binaryPath) -and (Test-Path $modelsDir)) {
    Write-Host 'RIFE NCNN Vulkan already present at:' $vendorDir
    Write-Host 'Delete vendor\rife to force a re-download.'
    return
}

New-Item -ItemType Directory -Force -Path $vendorDir | Out-Null
New-Item -ItemType Directory -Force -Path $modelsDir | Out-Null
New-Item -ItemType Directory -Force -Path $tempDir | Out-Null

if (Test-Path $extractDir) {
    Remove-Item -Recurse -Force $extractDir
}
New-Item -ItemType Directory -Force -Path $extractDir | Out-Null

Write-Host 'Downloading RIFE NCNN Vulkan (TNTwise fork, release' $releaseTag ')...'
Invoke-WebRequest -Uri $downloadUrl -OutFile $zipPath

Write-Host 'Extracting archive...'
Expand-Archive -Path $zipPath -DestinationPath $extractDir -Force

# The release zip nests everything under a single CI-artifact-style folder
# (e.g. rife-ncnn-vulkan-refs/heads/master-windows/) instead of a flat
# layout, so find it rather than hardcoding its name.
$releaseRoot = Get-ChildItem $extractDir -Directory | Select-Object -First 1
if (-not $releaseRoot) {
    throw 'RIFE archive extracted but no directory was found.'
}

$exeSource = Get-ChildItem $releaseRoot.FullName -Filter 'rife-ncnn-vulkan.exe' -File | Select-Object -First 1
if (-not $exeSource) {
    throw 'rife-ncnn-vulkan.exe not found inside the downloaded archive.'
}
Copy-Item -Force $exeSource.FullName $binaryPath

Get-ChildItem $releaseRoot.FullName -File -Filter '*.dll' | ForEach-Object {
    Copy-Item -Force $_.FullName (Join-Path $vendorDir $_.Name)
}

Write-Host 'Copying models (this bundles every RIFE version shipped in the archive)...'
Get-ChildItem $releaseRoot.FullName -Directory | ForEach-Object {
    $destination = Join-Path $modelsDir $_.Name
    if (Test-Path $destination) {
        Remove-Item -Recurse -Force $destination
    }
    Copy-Item -Recurse -Force $_.FullName $destination
}

Remove-Item -Recurse -Force $extractDir
Remove-Item -Force $zipPath

Write-Host 'RIFE NCNN Vulkan downloaded to:' $vendorDir
Write-Host 'Binary:' $binaryPath
Write-Host 'Models:' $modelsDir
Write-Host 'Default model (recommended, general-purpose): rife-v4.6 -- set RIFE_MODEL to switch (e.g. rife-anime, rife-v4.25-lite, rife-v4.26).'
