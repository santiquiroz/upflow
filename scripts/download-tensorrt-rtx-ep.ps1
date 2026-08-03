$ErrorActionPreference = 'Stop'

# Acelerador nativo de NVIDIA (TensorRT-RTX) como plugin EP de onnxruntime.
#
# Se instala con pip EN LA MAQUINA DEL USUARIO y NO viaja dentro del instalador:
# los DLLs de TensorRT-RTX son propietarios de NVIDIA (LicenseRef-NvidiaProprietary
# dentro del wheel). Bajandolo aca no hay redistribucion nuestra que justificar.
#
# --no-deps a proposito: el wheel no declara dependencias de runtime, y sin el
# flag pip podria intentar resolver `onnxruntime` vanilla y pisar
# onnxruntime-directml, que es el baseline de toda la app.
#
# Requiere driver NVIDIA presente: el DLL importa nvml.dll, que lo aporta el
# driver y NO viene en el wheel. Sin driver el registro falla con Error 126 y el
# ep_registry cae a DirectML solo.
#
# Cobertura real: Compute Capability >= 8.0, o sea RTX 30xx/40xx/50xx. Las
# RTX 20xx y GTX 16xx quedan afuera por un gate duro del propio EP.

$root = Split-Path -Parent $PSScriptRoot
$pythonExe = Join-Path $root 'python\python.exe'
if (-not (Test-Path $pythonExe)) {
    $pythonExe = (Get-Command python -ErrorAction SilentlyContinue).Source
}
if (-not $pythonExe) {
    throw 'No se encontro un interprete de Python para instalar el acelerador de NVIDIA.'
}

# Con $ErrorActionPreference='Stop', el stderr de un ejecutable nativo se vuelve
# error terminante. El probe FALLA a proposito cuando el plugin no esta, asi que
# se corre con la preferencia relajada y se juzga por el exit code.
function Get-NvidiaEpLibraryPath {
    $previous = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        $out = & $pythonExe -c "import onnxruntime_ep_nv_tensorrt_rtx as m; print(m.get_library_path())" 2>$null
        if ($LASTEXITCODE -eq 0 -and $out) { return ($out | Select-Object -Last 1).Trim() }
        return $null
    } finally {
        $ErrorActionPreference = $previous
    }
}

$probe = Get-NvidiaEpLibraryPath
if ($probe) {
    Write-Host "ya esta: acelerador NVIDIA presente en $probe"
    exit 0
}

# Se instala el paquete REAL (-cu13) y no el meta-paquete: el meta no trae
# binarios, solo declara al -cu13 como dependencia, y --no-deps bloquearia
# justamente eso dejando una instalacion vacia que no expone get_library_path().
# cu13 es el carril al que apunta el meta-paquete. El wheel de Windows pesa
# ~93 MB y ocupa ~246 MB en disco (trae TensorRT-RTX y el runtime de CUDA).
Write-Host 'Descargando el acelerador nativo de NVIDIA (~93 MB, ~246 MB en disco)...'
& $pythonExe -m pip install --quiet --no-deps --no-warn-script-location `
    'onnxruntime-ep-nv-tensorrt-rtx-cu13==0.3.0'
if ($LASTEXITCODE -ne 0) {
    throw 'No se pudo instalar el acelerador de NVIDIA. Revisa tu conexion a internet.'
}

$libPath = Get-NvidiaEpLibraryPath
if (-not $libPath) {
    throw 'El acelerador de NVIDIA se instalo pero no expone get_library_path(). Instalacion incompleta.'
}
if (-not (Test-Path $libPath)) {
    throw "El acelerador de NVIDIA declara $libPath pero ese archivo no existe."
}

Write-Host "Acelerador NVIDIA listo en: $libPath"
Write-Host 'Nota: requiere driver NVIDIA instalado y una GPU RTX 30xx o mas nueva.'
