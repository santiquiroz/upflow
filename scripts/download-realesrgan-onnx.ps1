$ErrorActionPreference = 'Stop'

# Downloads the official xinntao/Real-ESRGAN PyTorch weights and exports them to
# the uint8-in/out ONNX graphs the optimized ONNX video backend uses (SP11).
# The .onnx files land in vendor\realesrgan-onnx\ (gitignored, vendored).

# Older Windows PowerShell 5.1 defaults to TLS 1.0, which GitHub rejects.
[Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12

$root = Split-Path -Parent $PSScriptRoot
$vendorDir = Join-Path $root 'vendor\realesrgan-onnx'
$weightsDir = Join-Path $vendorDir 'weights'
$exportScript = Join-Path $PSScriptRoot 'export-realesrgan-onnx.py'

# Official PyTorch weights (each from the release that first shipped it).
$weights = @(
    @{ Name = 'realesr-animevideov3.pth';        Url = 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth' },
    @{ Name = 'RealESRGAN_x4plus.pth';           Url = 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth' },
    @{ Name = 'RealESRGAN_x4plus_anime_6B.pth';  Url = 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth' }
)

# The .onnx files this script produces (must match app/services/backend_registry.py).
$expectedOnnx = @(
    'realesr-animevideov3-x2-uint8.onnx',
    'realesr-animevideov3-x3-uint8.onnx',
    'realesr-animevideov3-x4-uint8.onnx',
    'realesrgan-x4plus-uint8.onnx',
    'realesrgan-x4plus-anime-uint8.onnx'
)

$missingOnnx = $expectedOnnx | Where-Object { -not (Test-Path (Join-Path $vendorDir $_)) }
if ($missingOnnx.Count -eq 0) {
    Write-Host 'Real-ESRGAN ONNX exports already present at:' $vendorDir
    Write-Host 'Delete vendor\realesrgan-onnx to force a re-export.'
    return
}

New-Item -ItemType Directory -Force -Path $weightsDir | Out-Null

foreach ($w in $weights) {
    $dest = Join-Path $weightsDir $w.Name
    if (Test-Path $dest) {
        Write-Host 'Weights already present:' $w.Name
        continue
    }
    Write-Host 'Downloading' $w.Name '...'
    Invoke-WebRequest -Uri $w.Url -OutFile $dest
}

# Prefer the project venv's python (has torch/onnx); fall back to python on PATH.
$venvPython = Join-Path $root '.venv\Scripts\python.exe'
$python = if (Test-Path $venvPython) { $venvPython } else { 'python' }

Write-Host 'Exporting builtin models to uint8 ONNX (this needs torch + onnx) ...'
& $python $exportScript --weights-dir $weightsDir --out-dir $vendorDir
if ($LASTEXITCODE -ne 0) {
    throw "ONNX export failed (exit $LASTEXITCODE). Ensure torch + onnx are installed in the venv."
}

$stillMissing = $expectedOnnx | Where-Object { -not (Test-Path (Join-Path $vendorDir $_)) }
if ($stillMissing.Count -gt 0) {
    throw "Export finished but these ONNX files are missing: $($stillMissing -join ', ')"
}

Write-Host 'Real-ESRGAN ONNX exports written to:' $vendorDir
Write-Host 'The optimized ONNX video backend will now pick them up automatically (UPSCALE_BACKEND=auto).'
