$ErrorActionPreference = 'Stop'

# Older Windows PowerShell 5.1 defaults to TLS 1.0, which GitHub rejects.
[Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12

$root = Split-Path -Parent $PSScriptRoot
$vendorDir = Join-Path $root 'vendor\audiosr'

# AudioSR (haoheliu/versatile_audio_super_resolution, MIT) ported to ONNX for
# the EXPERIMENTAL diffusion audio super-resolution mode. First known
# ONNX/DirectML port: https://github.com/santiquiroz/port-audiosr-onnx
# ~2.6 GB total (UNet 258M + VAE + vocoder). Runs on any ORT EP.
$releaseTag = 'models-v1.0'
$baseUrl = "https://github.com/santiquiroz/port-audiosr-onnx/releases/download/$releaseTag"

$assets = @(
    'manifest.json',
    'alphas_cumprod.npy',
    'mel_basis.npy',
    'vae_decoder.onnx',
    'vae_feature_extract.onnx',
    'vocoder.onnx',
    'ddpm.onnx',
    'ddpm.onnx.data'
)

New-Item -ItemType Directory -Force -Path $vendorDir | Out-Null

foreach ($asset in $assets) {
    $target = Join-Path $vendorDir $asset
    if (Test-Path $target) {
        Write-Host "Ya presente: $asset"
        continue
    }
    Write-Host "Descargando $asset ..."
    $temp = "$target.part"
    Invoke-WebRequest -Uri "$baseUrl/$asset" -OutFile $temp -UseBasicParsing
    Move-Item -Force $temp $target
}

$missing = $assets | Where-Object { -not (Test-Path (Join-Path $vendorDir $_)) }
if ($missing) {
    throw "La descarga de AudioSR quedo incompleta; faltan: $($missing -join ', ')"
}

Write-Host 'Modelos AudioSR listos en:' $vendorDir
Write-Host 'Activa el modo con ENABLE_AUDIOSR=true en .env.'
