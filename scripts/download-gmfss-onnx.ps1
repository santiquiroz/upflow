$ErrorActionPreference = 'Stop'

# Older Windows PowerShell 5.1 defaults to TLS 1.0, which GitHub rejects.
[Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12

$root = Split-Path -Parent $PSScriptRoot
$vendorDir = Join-Path $root 'vendor\gmfss'

# GMFSS_Fortuna (98mxr/GMFSS_Fortuna + HolyWu/vs-gmfss_fortuna, MIT) ported to
# ONNX for the EXPERIMENTAL max-quality frame-interpolation engine (GmfssEngine,
# second engine next to RIFE, 10x or more slower but much higher quality). Own port:
# https://github.com/santiquiroz/port-gmfss-onnx
# ~55MB total (4 fp32 graphs + fp16 fusionnet variant). Runs on any ORT EP.
$releaseTag = 'models-v1.0'
$baseUrl = "https://github.com/santiquiroz/port-gmfss-onnx/releases/download/$releaseTag"

# Required (GmfssAssets.is_complete gates on these -- see manifest.json's own
# required_files list, mirrored here).
$requiredAssets = @(
    'manifest.json',
    'featurenet.onnx',
    'gmflow.onnx',
    'metricnet.onnx',
    'metricnet.onnx.data',
    'fusionnet.onnx'
)

# Optional: fp16 fusionnet variant, only used on a DML/GPU device
# (gmfss_engine.py._should_use_fp16_fusionnet) -- CPU always keeps fp32. Real
# measured config (port project Task 3.2, RX 7800 XT): DirectML + fp16
# fusionnet + OpenCL splat = 0.72-0.73fps @1080p 2x, vs ~0.2fps fp32/CPU-splat.
# Placed flat (matches the release asset name 1:1) -- GmfssEngine looks for
# <model_dir>/fusionnet_fp16.onnx, NOT manifest.json's optional_files path
# (fp16/fusionnet.onnx); that manifest field documents the port repo's own
# on-disk layout, not this vendored consumer's.
$optionalAssets = @(
    'fusionnet_fp16.onnx'
)

New-Item -ItemType Directory -Force -Path $vendorDir | Out-Null

function Get-Asset([string]$asset, [bool]$required) {
    $target = Join-Path $vendorDir $asset
    if (Test-Path $target) {
        Write-Host "Ya presente: $asset"
        return
    }
    Write-Host "Descargando $asset ..."
    $temp = "$target.part"
    try {
        Invoke-WebRequest -Uri "$baseUrl/$asset" -OutFile $temp -UseBasicParsing
        Move-Item -Force $temp $target
    } catch {
        if ($required) {
            throw
        }
        Write-Host "Opcional no disponible, se omite: $asset"
        Remove-Item -Force -ErrorAction SilentlyContinue $temp
    }
}

foreach ($asset in $requiredAssets) {
    Get-Asset $asset $true
}
foreach ($asset in $optionalAssets) {
    Get-Asset $asset $false
}

$missing = $requiredAssets | Where-Object { -not (Test-Path (Join-Path $vendorDir $_)) }
if ($missing) {
    throw "La descarga de GMFSS quedo incompleta; faltan: $($missing -join ', ')"
}

Write-Host 'Modelos GMFSS listos en:' $vendorDir
if (Test-Path (Join-Path $vendorDir 'fusionnet_fp16.onnx')) {
    Write-Host 'Variante fp16 de fusionnet instalada (acelera en GPU DirectML/OpenCL).'
} else {
    Write-Host 'Variante fp16 de fusionnet no instalada -- GMFSS correra en fp32 (mas lento en GPU).'
}
Write-Host 'Activa el motor con ENABLE_GMFSS=true en .env.'
