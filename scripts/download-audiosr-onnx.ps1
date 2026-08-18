param(
    # fp16 es la mitad de descarga y de disco, 9% mas rapido, y su salida esta a
    # 59.4 dB SI-SDR de la fp32 (inaudible). NO sirve para CPU: el EP de CPU tiene
    # muchos menos kernels fp16 y los que hay suelen ser mas lentos.
    [ValidateSet('fp32', 'fp16')]
    [string]$Precision = 'fp32'
)

$ErrorActionPreference = 'Stop'

# Older Windows PowerShell 5.1 defaults to TLS 1.0, which GitHub rejects.
[Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12

$root = Split-Path -Parent $PSScriptRoot
$vendorDir = Join-Path $root 'vendor\audiosr'

# AudioSR (haoheliu/versatile_audio_super_resolution, MIT) ported to ONNX for
# the EXPERIMENTAL diffusion audio super-resolution mode. First known
# ONNX/DirectML port: https://github.com/santiquiroz/port-audiosr-onnx
# ~2.6 GB total (UNet 258M + VAE + vocoder). Runs on any ORT EP.
# fp16 son los MISMOS grafos con los mismos nombres de archivo, asi que el motor
# no cambia; lo unico distinto es de que release salen y que los tres grafos
# chicos tambien traen datos externos.
if ($Precision -eq 'fp16') {
    $releaseTag = 'models-fp16-v1.0'
} else {
    $releaseTag = 'models-v1.0'
}
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
if ($Precision -eq 'fp16') {
    $assets += @(
        'vae_decoder.onnx.data',
        'vae_feature_extract.onnx.data',
        'vocoder.onnx.data'
    )
}

# Cambiar de precision reemplaza los grafos: mezclar fp16 y fp32 en la misma
# carpeta deja archivos de los dos y el manifest describiendo solo unos.
$manifestPath = Join-Path $vendorDir 'manifest.json'
if (Test-Path $manifestPath) {
    $instalado = 'fp32'
    $previo = Get-Content $manifestPath -Raw | ConvertFrom-Json
    if ($previo.PSObject.Properties.Name -contains 'precision') {
        $instalado = $previo.precision
    }
    if ($instalado -ne $Precision) {
        Write-Host "Habia $instalado instalado y se pidio $Precision; se borran los grafos viejos."
        Remove-Item -Path (Join-Path $vendorDir '*.onnx') -Force -ErrorAction SilentlyContinue
        Remove-Item -Path (Join-Path $vendorDir '*.onnx.data') -Force -ErrorAction SilentlyContinue
        Remove-Item -Path $manifestPath -Force -ErrorAction SilentlyContinue
    }
}

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

Write-Host "Modelos AudioSR ($Precision) listos en: $vendorDir"
Write-Host 'Activa el modo con ENABLE_AUDIOSR=true en .env.'
