$ErrorActionPreference = 'Stop'

# Older Windows PowerShell 5.1 defaults to TLS 1.0, which GitHub rejects.
[Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12

$root = Split-Path -Parent $PSScriptRoot
$vendorDir = Join-Path $root 'vendor\rife'
$modelsDir = Join-Path $vendorDir 'models'
$binaryPath = Join-Path $vendorDir 'rife-ncnn-vulkan.exe'
$defaultModelDir = Join-Path $modelsDir 'rife-v4.6'
$tempDir = Join-Path $root 'runtime\temp'
$zipPath = Join-Path $tempDir 'rife-ncnn-vulkan.zip'
$extractDir = Join-Path $tempDir 'rife-ncnn-vulkan-extract'

# TNTwise/rife-ncnn-vulkan (MIT, active fork of nihui/rife-ncnn-vulkan with
# newer v4.x models). Pinned tag for reproducibility instead of "latest";
# asset name verified against the release's actual asset list.
# https://github.com/TNTwise/rife-ncnn-vulkan/releases/tag/20250112
$releaseTag = '20250112'
$downloadUrl = "https://github.com/TNTwise/rife-ncnn-vulkan/releases/download/$releaseTag/windows.zip"

# Checking the default model folder (not just models\) guards against a
# previous partially-completed install.
if ((Test-Path $binaryPath) -and (Test-Path $defaultModelDir)) {
    Write-Host 'RIFE NCNN Vulkan already present at:' $vendorDir
    Write-Host 'Delete vendor\rife to force a re-download.'
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

    Write-Host 'Downloading RIFE NCNN Vulkan (TNTwise fork, release' $releaseTag ')...'
    Invoke-WebRequest -Uri $downloadUrl -OutFile $zipPath

    Write-Host 'Extracting archive...'
    Expand-Archive -Path $zipPath -DestinationPath $extractDir -Force

    # The release zip nests everything under a CI-artifact-style path
    # (rife-ncnn-vulkan-refs/heads/master-windows/), so locate the exe
    # recursively and treat its folder as the content root.
    $exeSource = Get-ChildItem $extractDir -Filter 'rife-ncnn-vulkan.exe' -Recurse -File | Select-Object -First 1
    if (-not $exeSource) {
        throw 'rife-ncnn-vulkan.exe not found inside the downloaded archive.'
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

    Write-Host 'Copying models (this bundles every RIFE version shipped in the archive)...'
    Get-ChildItem $contentRoot.FullName -Directory | ForEach-Object {
        $destination = Join-Path $modelsDir $_.Name
        if (Test-Path $destination) {
            Remove-Item -Recurse -Force $destination
        }
        Copy-Item -Recurse -Force $_.FullName $destination
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

Write-Host 'RIFE NCNN Vulkan downloaded to:' $vendorDir
Write-Host 'Binary:' $binaryPath
Write-Host 'Models:' $modelsDir
Write-Host 'Default model (recommended, general-purpose): rife-v4.6 -- set RIFE_MODEL to switch (e.g. rife-anime, rife-v4.25-lite, rife-v4.26).'
