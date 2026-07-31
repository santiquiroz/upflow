$ErrorActionPreference = 'Stop'
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

# MobileSAM ONNX (Acly/MobileSAM, MIT) para el "tocar un objeto" del Editor.
# Revision pinneada; URLs y tamanos verificados 2026-07-31 (encoder 28,157,093
# bytes, decoder 16,501,323 bytes).
$revision = '0d3b403339b4674a82493d5e97964dd78089ddc8'
$files = @(
    @{ Name = 'mobile_sam_image_encoder.onnx'; Bytes = 28157093 },
    @{ Name = 'sam_mask_decoder_single.onnx'; Bytes = 16501323 }
)

$root = Split-Path -Parent $PSScriptRoot
$vendorDir = Join-Path $root 'vendor\mobilesam'
New-Item -ItemType Directory -Force $vendorDir | Out-Null

foreach ($file in $files) {
    $dest = Join-Path $vendorDir $file.Name
    if ((Test-Path $dest) -and ((Get-Item $dest).Length -eq $file.Bytes)) {
        Write-Host "$($file.Name) ya esta"
        continue
    }
    $url = "https://huggingface.co/Acly/MobileSAM/resolve/$revision/$($file.Name)"
    Write-Host "Descargando $url"
    Invoke-WebRequest -Uri $url -OutFile $dest -UseBasicParsing
    if ((Get-Item $dest).Length -ne $file.Bytes) {
        throw "$($file.Name): tamano inesperado (descarga corrupta)"
    }
}

Write-Host "MobileSAM listo en $vendorDir"
