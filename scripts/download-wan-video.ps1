$ErrorActionPreference = 'Stop'

# Older Windows PowerShell 5.1 defaults to TLS 1.0, which some hosts reject.
[Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12

$root = Split-Path -Parent $PSScriptRoot
$vendorDir = Join-Path $root 'vendor\sdcpp\models\video'

New-Item -ItemType Directory -Force -Path $vendorDir | Out-Null
# Lista de assets: URL -> destino -> tamaño esperado (bytes)
$assets = @(
    @{ url = 'https://huggingface.co/QuantStack/Wan2.2-TI2V-5B-GGUF/resolve/main/Wan2.2-TI2V-5B-Q8_0.gguf'; name = 'Wan2.2-TI2V-5B-Q8_0.gguf'; expected = 5400179040 },
    @{ url = 'https://huggingface.co/QuantStack/Wan2.2-TI2V-5B-GGUF/resolve/main/VAE/Wan2.2_VAE.safetensors'; name = 'Wan2.2_VAE.safetensors'; expected = 1409400960 },
    @{ url = 'https://huggingface.co/city96/umt5-xxl-encoder-gguf/resolve/main/umt5-xxl-encoder-Q5_K_M.gguf'; name = 'umt5-xxl-encoder-Q5_K_M.gguf'; expected = 4145878880 },
    @{ url = 'https://huggingface.co/hum-ma/Wan2.2-TI2V-5B-Turbo-GGUF/resolve/main/Wan2_2-TI2V-5B-Turbo-Q8_0.gguf'; name = 'Wan2_2-TI2V-5B-Turbo-Q8_0.gguf'; expected = 5404990176 },
    # Decodificador chico: sin el, el VAE de Wan no entra en 16 GB de VRAM.
    @{ url = 'https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/taew2_2.safetensors'; name = 'taew2_2.safetensors'; expected = 22848080 }
)
function Get-Asset($asset) {
    $url = $asset.url
    $name = $asset.name
    $expected = [int64]$asset.expected
    $tolerance = 0
    if ($asset.ContainsKey('tolerance_percent')) { $tolerance = [double]$asset.tolerance_percent }

    $target = Join-Path $vendorDir $name

    if (Test-Path $target) {
        try {
            $existingSize = (Get-Item $target).Length
        } catch {
            # si no se puede leer size, forzar re-descarga
            $existingSize = -1
        }
        if ($tolerance -gt 0) {
            $min = [int64]([math]::Floor($expected * (1 - $tolerance/100.0)))
            $max = [int64]([math]::Ceiling($expected * (1 + $tolerance/100.0)))
            if (($existingSize -ge $min) -and ($existingSize -le $max)) {
                Write-Host "ya está: $name"
                return
            }
        } else {
            if ($existingSize -eq $expected) {
                Write-Host "ya está: $name"
                return
            }
        }
        Write-Host "Archivo existente con tamaño inesperado. Se re-descargará: $name"
        Remove-Item -Force -ErrorAction SilentlyContinue $target
    }

    Write-Host "Descargando $name ..."
    $temp = "$target.part"
    if (Test-Path $temp) { Remove-Item -Force -ErrorAction SilentlyContinue $temp }

    Invoke-WebRequest -Uri $url -OutFile $temp -UseBasicParsing

    # Verificar tamaño
    try {
        $got = (Get-Item $temp).Length
    } catch {
        Remove-Item -Force -ErrorAction SilentlyContinue $temp
        throw "Fallo al leer el archivo temporal descargado: $temp"
    }

    if ($tolerance -gt 0) {
        $min = [int64]([math]::Floor($expected * (1 - $tolerance/100.0)))
        $max = [int64]([math]::Ceiling($expected * (1 + $tolerance/100.0)))
        if (($got -lt $min) -or ($got -gt $max)) {
            Remove-Item -Force -ErrorAction SilentlyContinue $temp
            throw "El archivo $name tiene tamaño inesperado: $got bytes (esperado aprox entre $min y $max). Se borró $temp"
        }
    } else {
        if ($got -ne $expected) {
            Remove-Item -Force -ErrorAction SilentlyContinue $temp
            throw "El archivo $name tiene tamaño inesperado: $got bytes (esperado $expected). Se borró $temp"
        }
    }

    Move-Item -Force $temp $target
}

foreach ($a in $assets) {
    Get-Asset $a
}

# Verificacion final: asegurar que todos estan presentes (dentro de tolerancias)
$missing = @()
foreach ($a in $assets) {
    $name = $a.name
    $expected = [int64]$a.expected
    $tolerance = 0
    if ($a.ContainsKey('tolerance_percent')) { $tolerance = [double]$a.tolerance_percent }
    $target = Join-Path $vendorDir $name
    if (-not (Test-Path $target)) { $missing += $name; continue }
    $size = (Get-Item $target).Length
    if ($tolerance -gt 0) {
        $min = [int64]([math]::Floor($expected * (1 - $tolerance/100.0)))
        $max = [int64]([math]::Ceiling($expected * (1 + $tolerance/100.0)))
        if (($size -lt $min) -or ($size -gt $max)) { $missing += $name }
    } else {
        if ($size -ne $expected) { $missing += $name }
    }
}

if ($missing.Count -gt 0) {
    throw "La descarga quedo incompleta o con tamaños incorrectos; faltan/invalidos: $($missing -join ', ')"
}

# Informar carpeta y total en GB
$all = Get-ChildItem -Path $vendorDir -File -Recurse | Measure-Object -Property Length -Sum
$sum = [int64]$all.Sum
$gb = 1024 * 1024 * 1024
$sumGb = [math]::Round($sum / $gb, 2)
Write-Host 'Modelos Wan-Video listos en:' $vendorDir
Write-Host "Total: $sumGb GB"
