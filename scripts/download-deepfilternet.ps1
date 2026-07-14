$ErrorActionPreference = 'Stop'

# Older Windows PowerShell 5.1 defaults to TLS 1.0, which GitHub rejects.
[Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12

$root = Split-Path -Parent $PSScriptRoot
$vendorDir = Join-Path $root 'vendor\deepfilternet'
$modelsDir = Join-Path $vendorDir 'models'
$binaryPath = Join-Path $vendorDir 'deep-filter.exe'
$rnnoiseModelPath = Join-Path $modelsDir 'sh.rnnn'
$tempDir = Join-Path $root 'runtime\temp'
$tempBinaryPath = Join-Path $tempDir 'deep-filter.exe.download'
$tempRnnoiseModelPath = Join-Path $tempDir 'sh.rnnn.download'

# Rikorose/DeepFilterNet (dual MIT/Apache-2.0). Pinned tag for reproducibility
# instead of "latest"; asset name verified against the release's actual asset
# list via `gh release view v0.5.6 --repo Rikorose/DeepFilterNet --json assets`.
# https://github.com/Rikorose/DeepFilterNet/releases/tag/v0.5.6
$releaseTag = 'v0.5.6'
$deepFilterVersion = '0.5.6'
$deepFilterUrl = "https://github.com/Rikorose/DeepFilterNet/releases/download/$releaseTag/deep-filter-$deepFilterVersion-x86_64-pc-windows-msvc.exe"

# The Windows release binary is built with the `default-model` Cargo feature
# (verified against .github/workflows/publish.yml's build-deepfilter-rs job:
# FEATURES: bin,tract,use-jemalloc,wav-utils,transforms,default-model), so it
# embeds the default DeepFilterNet3 model -- no separate model download for
# the "deepfilter" mode.

# ffmpeg's arnndn filter (used for the "rnnoise" mode) requires a `.rnnn`
# model file and does not ship one -- the vendored ffmpeg build (BtbN
# FFmpeg-Builds) has none either. GregorR/rnnoise-models's "somnolent-hogwash"
# suite (speech in a recording environment, e.g. fans/AC/computers -- a good
# general match for media audio) is the standard source used by ffmpeg
# arnndn guides. Its README states: "With the exception of the tools/
# directory and this file, none of this work is creative and thus none of it
# is subject to copyright" -- i.e. the .rnnn model files themselves are
# public domain.
# https://github.com/GregorR/rnnoise-models/tree/master/somnolent-hogwash-2018-09-01
$rnnoiseModelUrl = 'https://raw.githubusercontent.com/GregorR/rnnoise-models/master/somnolent-hogwash-2018-09-01/sh.rnnn'

# Checking both files (not just the binary) guards against a previous
# partially-completed install.
if ((Test-Path $binaryPath) -and (Test-Path $rnnoiseModelPath)) {
    Write-Host 'DeepFilterNet already present at:' $vendorDir
    Write-Host 'Delete vendor\deepfilternet to force a re-download.'
    return
}

New-Item -ItemType Directory -Force -Path $vendorDir | Out-Null
New-Item -ItemType Directory -Force -Path $modelsDir | Out-Null
New-Item -ItemType Directory -Force -Path $tempDir | Out-Null

try {
    Write-Host 'Downloading deep-filter CLI (Rikorose/DeepFilterNet, release' $releaseTag ')...'
    Invoke-WebRequest -Uri $deepFilterUrl -OutFile $tempBinaryPath
    Copy-Item -Force $tempBinaryPath $binaryPath

    Write-Host 'Downloading rnnoise model for ffmpeg arnndn (GregorR/rnnoise-models, somnolent-hogwash)...'
    Invoke-WebRequest -Uri $rnnoiseModelUrl -OutFile $tempRnnoiseModelPath
    Copy-Item -Force $tempRnnoiseModelPath $rnnoiseModelPath

    $noticePath = Join-Path $vendorDir 'NOTICE.txt'
    @"
deep-filter.exe
  Source:  $deepFilterUrl
  License: MIT/Apache-2.0 (Rikorose/DeepFilterNet)

models\sh.rnnn
  Source:  $rnnoiseModelUrl
  License: Public domain per GregorR/rnnoise-models README ("none of this
           work is creative and thus none of it is subject to copyright"),
           excluding the tools/ directory and README itself (not vendored).
"@ | Set-Content -Path $noticePath -Encoding utf8
}
finally {
    if (Test-Path $tempBinaryPath) {
        Remove-Item -Force $tempBinaryPath
    }
    if (Test-Path $tempRnnoiseModelPath) {
        Remove-Item -Force $tempRnnoiseModelPath
    }
}

Write-Host 'DeepFilterNet downloaded to:' $vendorDir
Write-Host 'Binary:' $binaryPath
Write-Host 'Rnnoise model:' $rnnoiseModelPath
Write-Host 'The deep-filter binary embeds the default DeepFilterNet3 model; no ENGINE_MODELS_DIR-style folder is needed for it.'
