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
$licensePath = Join-Path $destDir 'LICENSE'
$licensePresente = (Test-Path $licensePath -PathType Leaf) -and ((Get-Item $licensePath).Length -gt 0)
if ((Test-Path $exePath -PathType Leaf) -and $licensePresente) {
    Write-Host "ya esta: Magpie $version en $destDir"
    exit 0
}

$zipPath = Join-Path $env:TEMP "magpie-$version.zip"
$partPath = "$zipPath.part"
$licensePartPath = "$licensePath.part"

try {
    if (-not (Test-Path $exePath -PathType Leaf)) {
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

        if (-not (Test-Path $exePath -PathType Leaf)) {
            throw "El zip se extrajo pero falta Magpie.exe en $destDir."
        }
    } else {
        Write-Host "Magpie.exe ya esta presente; falta completar su licencia GPL-3.0."
    }

    # El zip del release no trae la licencia y es GPL: se baja aparte.
    if (-not $licensePresente) {
        if (Test-Path $licensePartPath) { Remove-Item -Force -ErrorAction SilentlyContinue $licensePartPath }
        try {
            Invoke-WebRequest -Uri $licenseUrl -OutFile $licensePartPath -UseBasicParsing
        } catch {
            throw "No se pudo bajar el LICENSE de Magpie. Un componente GPL-3.0 no puede darse por instalado sin su licencia: $_"
        }
        if (-not (Test-Path $licensePartPath -PathType Leaf) -or (Get-Item $licensePartPath).Length -eq 0) {
            throw 'La descarga del LICENSE de Magpie no produjo un archivo valido; el componente GPL-3.0 queda incompleto.'
        }
        Move-Item -Force $licensePartPath $licensePath
    }

    if (-not (Test-Path $licensePath -PathType Leaf) -or (Get-Item $licensePath).Length -eq 0) {
        throw 'Magpie.exe esta presente pero falta su LICENSE GPL-3.0; la instalacion no puede darse por completa.'
    }

    $total = (Get-ChildItem -Path $destDir -File -Recurse | Measure-Object -Property Length -Sum).Sum
    Write-Host "Magpie listo en: $destDir"
    Write-Host ("Total: {0:N1} MB" -f ($total / 1MB))
    Write-Host 'Magpie es GPL-3.0 y corre como proceso separado. No necesita drivers.'
} finally {
    try { if (Test-Path $zipPath) { Remove-Item -Force -ErrorAction SilentlyContinue $zipPath } } catch {}
    try { if (Test-Path $partPath) { Remove-Item -Force -ErrorAction SilentlyContinue $partPath } } catch {}
    try { if (Test-Path $licensePartPath) { Remove-Item -Force -ErrorAction SilentlyContinue $licensePartPath } } catch {}
}
