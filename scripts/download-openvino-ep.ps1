$ErrorActionPreference = 'Stop'

# Forzar TLS 1.2 en PowerShell 5.1
[Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12

$root = Split-Path -Parent $PSScriptRoot
$destDir = Join-Path $root 'vendor\ep-plugins\openvino'
New-Item -ItemType Directory -Force -Path $destDir | Out-Null

# Paquete NuGet (es un zip):
$url = 'https://api.nuget.org/v3-flatcontainer/intel.ml.onnxruntime.ep.openvino/1.6.1/intel.ml.onnxruntime.ep.openvino.1.6.1.nupkg'
$name = 'intel.ml.onnxruntime.ep.openvino.1.6.1.nupkg'
$expected = 121782203

$downloadPath = Join-Path $env:TEMP $name
$partPath = "$downloadPath.part"
$zipPath = [IO.Path]::ChangeExtension($downloadPath, '.zip')
$extractDir = Join-Path $env:TEMP ([IO.Path]::GetFileNameWithoutExtension($zipPath) + '_' + [guid]::NewGuid().ToString())

# Nombre del DLL principal que esperamos encontrar en la carpeta destino
$requiredDll = 'onnxruntime_providers_openvino_plugin.dll'
$requiredDllPath = Join-Path $destDir $requiredDll

if (Test-Path $requiredDllPath) {
    Write-Host "ya está: $requiredDll"
    return
}

try {
    Write-Host "Descargando paquete OpenVINO EP ..."

    if (Test-Path $partPath) { Remove-Item -Force -ErrorAction SilentlyContinue $partPath }
    if (Test-Path $downloadPath) { Remove-Item -Force -ErrorAction SilentlyContinue $downloadPath }
    if (Test-Path $zipPath) { Remove-Item -Force -ErrorAction SilentlyContinue $zipPath }

    Invoke-WebRequest -Uri $url -OutFile $partPath -UseBasicParsing

    try {
        $got = (Get-Item $partPath).Length
    } catch {
        Remove-Item -Force -ErrorAction SilentlyContinue $partPath
        throw "Fallo al leer el archivo temporal descargado: $partPath"
    }

    if ($got -ne $expected) {
        Remove-Item -Force -ErrorAction SilentlyContinue $partPath
        throw "El archivo $name tiene tamaño inesperado: $got bytes (esperado $expected). Se borró $partPath"
    }

    Move-Item -Force $partPath $downloadPath

    # Rename to .zip para que Expand-Archive funcione
    Move-Item -Force $downloadPath $zipPath

    New-Item -ItemType Directory -Force -Path $extractDir | Out-Null
    Expand-Archive -LiteralPath $zipPath -DestinationPath $extractDir -Force

    # Buscar la carpeta runtimes/win-x64/native dentro del paquete
    $nativeDirs = Get-ChildItem -Path $extractDir -Directory -Recurse -ErrorAction SilentlyContinue | Where-Object { $_.FullName -like '*runtimes*win-x64*native*' }
    if ($nativeDirs.Count -eq 0) {
        # también intentar ruta literal por si acaso
        $possible = Join-Path $extractDir 'runtimes\win-x64\native'
        if (-not (Test-Path $possible)) {
            throw "No se encontró la carpeta runtimes\win-x64\native en el paquete extraído"
        } else {
            $nativeDirs = @(Get-Item $possible)
        }
    }

    # Excluir archivos listados y cualquiera con _debug en el nombre
    
    # Se recorta a lo que de verdad usamos: la GPU Intel (integrada o Arc).
    #  - cpu_plugin (45 MB): onnxruntime ya trae su propio provider de CPU.
    #  - npu_plugin (7 MB) sin su compilador (76 MB, excluido por peso) no sirve.
    #  - tbbmalloc/tbbmalloc_proxy enganchan el allocator del proceso y el
    #    plugin no los necesita: riesgo gratis.
    $excludes = @(
        'openvino_intel_npu_compiler.dll',
        'openvino_intel_cpu_plugin.dll',
        'openvino_intel_npu_plugin.dll',
        'openvino_intel_npu_compiler_loader.dll',
        'cache.json',
        'tbbmalloc.dll',
        'tbbmalloc_proxy.dll'
    )

    foreach ($nd in $nativeDirs) {
        $files = Get-ChildItem -Path $nd.FullName -File -ErrorAction SilentlyContinue
        foreach ($f in $files) {
            $basename = $f.Name
            if ($excludes -contains $basename) { continue }
            if ($basename -match '_debug') { continue }

            $target = Join-Path $destDir $basename
            Copy-Item -Force -Path $f.FullName -Destination $target
        }
    }

    # Copiar third-party-programs.txt si existe en cualquier subcarpeta
    $thirdParty = Get-ChildItem -Path $extractDir -Filter 'third-party-programs.txt' -File -Recurse -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($thirdParty) {
        Copy-Item -Force -Path $thirdParty.FullName -Destination (Join-Path $destDir $thirdParty.Name)
    }

    # Verificar DLL principal
    if (-not (Test-Path $requiredDllPath)) {
        throw "Falta el archivo requerido en ${destDir}: $requiredDll. La copia falló o el paquete no contiene el DLL esperado."
    }

    # Informar carpeta y total en GB
    $all = Get-ChildItem -Path $destDir -File -Recurse | Measure-Object -Property Length -Sum
    $sum = [int64]$all.Sum
    $sumMb = [math]::Round($sum / 1MB, 1)
    Write-Host 'OpenVINO EP listo en:' $destDir
    Write-Host "Total: $sumMb MB"
} finally {
    # Limpieza: borrar carpeta temporal y paquete descargado si existen
    try { if (Test-Path $extractDir) { Remove-Item -Recurse -Force -ErrorAction SilentlyContinue $extractDir } } catch {}
    try { if (Test-Path $zipPath) { Remove-Item -Force -ErrorAction SilentlyContinue $zipPath } } catch {}
    try { if (Test-Path $downloadPath) { Remove-Item -Force -ErrorAction SilentlyContinue $downloadPath } } catch {}
    try { if (Test-Path $partPath) { Remove-Item -Force -ErrorAction SilentlyContinue $partPath } } catch {}
}
