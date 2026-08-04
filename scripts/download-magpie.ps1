$ErrorActionPreference = 'Stop'

# Overlay de reescalado en tiempo real (Fase 7.1).
#
# Magpie es GPL-3.0. NO se redistribuye dentro del instalador: se baja aca, en
# la maquina del usuario, desde su release oficial. Asi no hay redistribucion
# nuestra que justificar, y ademas corre como proceso APARTE — linkearlo
# volveria GPL a Upflow.
#
# NO hace falta ningun driver: Magpie presenta en una ventana overlay sin
# enganchar el swapchain del juego (por eso no tiene riesgo de anti-cheat).
#
# Se baja tambien el LICENSE, que el zip del release NO incluye.

[Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12

$root = Split-Path -Parent $PSScriptRoot
$destDir = Join-Path $root 'vendor\magpie'

$version = 'v0.12.1'
$url = "https://github.com/Blinue/Magpie/releases/download/$version/Magpie-$version-x64.zip"
$expected = 10832614
$licenseUrl = 'https://raw.githubusercontent.com/Blinue/Magpie/main/LICENSE'

$exePath = Join-Path $destDir 'Magpie.exe'
if (Test-Path $exePath) {
    Write-Host "ya esta: Magpie $version en $destDir"
    exit 0
}

$zipPath = Join-Path $env:TEMP "magpie-$version.zip"
$partPath = "$zipPath.part"

try {
    Write-Host "Descargando Magpie $version (~10 MB)..."
    if (Test-Path $partPath) { Remove-Item -Force -ErrorAction SilentlyContinue $partPath }
    Invoke-WebRequest -Uri $url -OutFile $partPath -UseBasicParsing

    $got = (Get-Item $partPath).Length
    if ($got -ne $expected) {
        Remove-Item -Force -ErrorAction SilentlyContinue $partPath
        throw "Magpie-$version-x64.zip tiene tamano inesperado: $got bytes (esperado $expected)."
    }
    Move-Item -Force $partPath $zipPath

    New-Item -ItemType Directory -Force -Path $destDir | Out-Null
    Expand-Archive -LiteralPath $zipPath -DestinationPath $destDir -Force

    if (-not (Test-Path $exePath)) {
        throw "El zip se extrajo pero falta Magpie.exe en $destDir."
    }

    # El zip del release no trae la licencia y es GPL: se baja aparte.
    try {
        Invoke-WebRequest -Uri $licenseUrl -OutFile (Join-Path $destDir 'LICENSE') -UseBasicParsing
    } catch {
        Write-Warning "No se pudo bajar el LICENSE de Magpie: $_"
    }

    $total = (Get-ChildItem -Path $destDir -File -Recurse | Measure-Object -Property Length -Sum).Sum
    Write-Host "Magpie listo en: $destDir"
    Write-Host ("Total: {0:N1} MB" -f ($total / 1MB))
    Write-Host 'Magpie es GPL-3.0 y corre como proceso separado. No necesita drivers.'
} finally {
    try { if (Test-Path $zipPath) { Remove-Item -Force -ErrorAction SilentlyContinue $zipPath } } catch {}
    try { if (Test-Path $partPath) { Remove-Item -Force -ErrorAction SilentlyContinue $partPath } } catch {}
}
