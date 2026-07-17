param(
    [switch]$Zip,
    [switch]$Installer
)

$ErrorActionPreference = 'Stop'

# Default (no switches) matches the historical behavior of this script:
# just the portable zip. Passing -Installer alone builds only the
# installer; both switches together build both artifacts.
if (-not $Zip -and -not $Installer) {
    $Zip = $true
}

$root = Split-Path -Parent $PSScriptRoot
$pyprojectPath = Join-Path $root 'pyproject.toml'

$versionMatch = Select-String -Path $pyprojectPath -Pattern '^version\s*=\s*"([^"]+)"' | Select-Object -First 1
if (-not $versionMatch) {
    throw 'No se pudo leer la version desde pyproject.toml.'
}
$version = $versionMatch.Matches[0].Groups[1].Value

$distDir = Join-Path $root 'dist'
$frontendDir = Join-Path $root 'frontend'
$frontendDistDir = Join-Path $frontendDir 'dist'
$frontendIndexPath = Join-Path $frontendDistDir 'index.html'

$installerDir = Join-Path $root 'installer'
$installerBuildDir = Join-Path $installerDir 'build'
$installerAppDir = Join-Path $installerBuildDir 'app'
$installerPythonDir = Join-Path $installerBuildDir 'python'
$installerCacheDir = Join-Path $installerBuildDir '_cache'
$isscPath = Join-Path $installerDir 'upflow.iss'

$pythonEmbedVersion = '3.12.10'
$pythonEmbedUrl = "https://www.python.org/ftp/python/$pythonEmbedVersion/python-$pythonEmbedVersion-embed-amd64.zip"
$getPipUrl = 'https://bootstrap.pypa.io/get-pip.py'

function Write-Step {
    param([string]$Message)
    Write-Host ''
    Write-Host "==> $Message" -ForegroundColor Cyan
}

function Build-Frontend {
    # The React SPA is the only UI FastAPI serves (app/main.py mounts
    # frontend/dist unconditionally) — both release artifacts are broken
    # without a fresh build, so this always rebuilds instead of trusting a
    # stale dist/.
    Write-Step 'Compilando la SPA de React (frontend/)...'
    Push-Location $frontendDir
    try {
        npm ci
        if ($LASTEXITCODE -ne 0) {
            throw 'npm ci fallo en frontend/.'
        }
        npm run build
        if ($LASTEXITCODE -ne 0) {
            throw 'npm run build fallo en frontend/.'
        }
    }
    finally {
        Pop-Location
    }

    if (-not (Test-Path $frontendIndexPath)) {
        throw "El build de frontend no genero frontend/dist/index.html."
    }
}

function Copy-AppAllowlist {
    param([string]$Destination)

    if (Test-Path $Destination) {
        Remove-Item -Recurse -Force $Destination
    }
    New-Item -ItemType Directory -Force -Path $Destination | Out-Null

    # Allowlist (not a denylist of vendor/runtime/.venv/.git/tests/docs/.superpowers):
    # only end-user-facing files travel in a release artifact. Binaries in
    # vendor/ are downloaded on first launch by upflow-launcher.ps1.
    $includeDirs = @('app', 'scripts')
    $includeFiles = @('pyproject.toml', 'README.md', 'LICENSE', '.env.example', 'Upflow.bat')

    foreach ($dir in $includeDirs) {
        $source = Join-Path $root $dir
        if (-not (Test-Path $source)) {
            throw "Carpeta esperada no encontrada: $dir"
        }
        Copy-Item -Recurse -Force $source (Join-Path $Destination $dir)
    }

    foreach ($file in $includeFiles) {
        $source = Join-Path $root $file
        if (-not (Test-Path $source)) {
            throw "Archivo esperado no encontrado: $file"
        }
        Copy-Item -Force $source (Join-Path $Destination $file)
    }

    # Only the built frontend/dist/ output travels — src/, node_modules/ and
    # the rest of frontend/ are build-time only and stay out.
    $destinationFrontendDistDir = Join-Path $Destination 'frontend\dist'
    New-Item -ItemType Directory -Force -Path $destinationFrontendDistDir | Out-Null
    Copy-Item -Recurse -Force (Join-Path $frontendDistDir '*') $destinationFrontendDistDir

    # Bytecode cache never belongs in a distributed artifact.
    Get-ChildItem $Destination -Recurse -Directory -Filter '__pycache__' | ForEach-Object {
        Remove-Item -Recurse -Force $_.FullName
    }
    Get-ChildItem $Destination -Recurse -File -Filter '*.pyc' | ForEach-Object {
        Remove-Item -Force $_.FullName
    }
}

function New-PortableZip {
    $stagingDir = Join-Path $distDir "upflow-v$version"
    $zipPath = Join-Path $distDir "upflow-v$version.zip"

    New-Item -ItemType Directory -Force -Path $distDir | Out-Null
    if (Test-Path $zipPath) {
        Remove-Item -Force $zipPath
    }

    Copy-AppAllowlist -Destination $stagingDir

    # Both Compress-Archive and ZipFile.CreateFromDirectory leave literal
    # backslashes in entry names on Windows/.NET Framework (Windows
    # PowerShell 5.1), which violates the ZIP spec (entries must use '/')
    # and breaks extraction on tools that follow it strictly. Building
    # entries manually guarantees forward-slash separators regardless of
    # PowerShell/.NET version.
    Add-Type -AssemblyName System.IO.Compression
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $zipStream = [System.IO.File]::Open($zipPath, [System.IO.FileMode]::Create)
    try {
        $archive = New-Object System.IO.Compression.ZipArchive($zipStream, [System.IO.Compression.ZipArchiveMode]::Create)
        try {
            Get-ChildItem -Path $stagingDir -Recurse -File | ForEach-Object {
                $relativePath = $_.FullName.Substring($stagingDir.Length + 1) -replace '\\', '/'
                [System.IO.Compression.ZipFileExtensions]::CreateEntryFromFile($archive, $_.FullName, $relativePath, [System.IO.Compression.CompressionLevel]::Optimal) | Out-Null
            }
        }
        finally {
            $archive.Dispose()
        }
    }
    finally {
        $zipStream.Dispose()
    }

    Remove-Item -Recurse -Force $stagingDir

    Write-Host 'Release (zip portable) generado:' $zipPath
}

function Get-IsccPath {
    $fromPath = Get-Command ISCC -ErrorAction SilentlyContinue
    if ($fromPath) {
        return $fromPath.Source
    }

    $candidates = @(
        (Join-Path ${env:ProgramFiles} 'Inno Setup 6\ISCC.exe'),
        (Join-Path ${env:ProgramFiles(x86)} 'Inno Setup 6\ISCC.exe'),
        (Join-Path $env:LOCALAPPDATA 'Programs\Inno Setup 6\ISCC.exe')
    )
    foreach ($candidate in $candidates) {
        if ($candidate -and (Test-Path $candidate)) {
            return $candidate
        }
    }
    return $null
}

function Assert-IsccAvailable {
    $foundIsccPath = Get-IsccPath
    if ($foundIsccPath) {
        Write-Host "ISCC encontrado: $foundIsccPath"
        return $foundIsccPath
    }

    $message = 'No se encontro ISCC.exe (compilador de Inno Setup), necesario para -Installer.'
    $winget = Get-Command winget -ErrorAction SilentlyContinue
    if ($winget) {
        $message += "`n`nInstalalo con:`n  winget install JRSoftware.InnoSetup"
    } else {
        $message += "`n`nDescargalo desde https://jrsoftware.org/isdl.php"
    }
    throw $message
}

function Enable-EmbeddedPythonSitePackages {
    param([string]$PythonDir)

    $pthFile = Get-ChildItem -Path $PythonDir -Filter 'python3*._pth' | Select-Object -First 1
    if (-not $pthFile) {
        throw "No se encontro el archivo ._pth dentro de $PythonDir (layout de Python embeddable inesperado)."
    }

    # El embeddable trae "import site" comentado y sin Lib\site-packages en
    # el path: hay que habilitar ambos para que pip pueda instalar y
    # encontrar paquetes (si no, "pip install -e ." en el launcher falla
    # silenciosamente en el primer arranque).
    $lines = Get-Content $pthFile.FullName
    $lines = $lines -replace '^#\s*import site', 'import site'
    if ($lines -notcontains 'Lib\site-packages') {
        $lines += 'Lib\site-packages'
    }
    # Windows PowerShell 5.1's "-Encoding utf8" always writes a BOM. Python's
    # ._pth parser runs before the interpreter can initialize codecs, so a
    # BOM corrupts the first path entry (e.g. "﻿python312.zip") and the
    # embedded interpreter fails to boot with "No module named 'encodings'".
    # ._pth content is pure ASCII (Windows paths + "import site"), so ascii
    # is both correct and guaranteed BOM-free on every PowerShell version.
    Set-Content -Path $pthFile.FullName -Value $lines -Encoding ascii
}

function Initialize-EmbeddedPython {
    New-Item -ItemType Directory -Force -Path $installerCacheDir | Out-Null

    if (Test-Path $installerPythonDir) {
        Remove-Item -Recurse -Force $installerPythonDir
    }
    New-Item -ItemType Directory -Force -Path $installerPythonDir | Out-Null

    $embedZipPath = Join-Path $installerCacheDir "python-$pythonEmbedVersion-embed-amd64.zip"
    $getPipPath = Join-Path $installerCacheDir 'get-pip.py'

    Write-Step "Descargando Python $pythonEmbedVersion embeddable..."
    Invoke-WebRequest -Uri $pythonEmbedUrl -OutFile $embedZipPath -UseBasicParsing
    Expand-Archive -Path $embedZipPath -DestinationPath $installerPythonDir -Force

    Enable-EmbeddedPythonSitePackages -PythonDir $installerPythonDir

    Write-Step 'Descargando get-pip.py...'
    Invoke-WebRequest -Uri $getPipUrl -OutFile $getPipPath -UseBasicParsing

    Write-Step 'Instalando pip en el Python embebido...'
    $embedPythonExe = Join-Path $installerPythonDir 'python.exe'
    & $embedPythonExe $getPipPath --no-warn-script-location --quiet
    if ($LASTEXITCODE -ne 0) {
        throw 'No se pudo instalar pip en el Python embebido (get-pip.py fallo).'
    }

    & $embedPythonExe -m pip --version | Out-Null
    if ($LASTEXITCODE -ne 0) {
        throw 'pip no quedo funcional dentro del Python embebido.'
    }

    # pip's normal build-isolation flow spawns a subprocess with the build
    # requirements (setuptools/wheel from [build-system] requires) injected
    # via PYTHONPATH into a throwaway env — but the embeddable's ._pth file
    # makes every subprocess of this interpreter ignore PYTHONPATH entirely
    # (that is the whole point of ._pth-driven isolation), so that injection
    # silently no-ops and "pip install -e ." fails later with
    # "BackendUnavailable: Cannot import 'setuptools.build_meta'". Installing
    # setuptools/wheel directly into the embedded interpreter's own
    # site-packages and having the launcher pass --no-build-isolation (see
    # upflow-launcher.ps1) sidesteps the broken injection entirely.
    Write-Step 'Instalando setuptools y wheel en el Python embebido (necesarios para pip install -e . --no-build-isolation)...'
    & $embedPythonExe -m pip install --quiet --no-warn-script-location setuptools wheel
    if ($LASTEXITCODE -ne 0) {
        throw 'No se pudo instalar setuptools/wheel en el Python embebido.'
    }

    Write-Host 'Python embebido listo con pip, setuptools y wheel funcionales.'
}

function Invoke-InnoSetupCompile {
    param([string]$IsccPath)

    Write-Step 'Compilando el instalador con Inno Setup (ISCC)...'
    & $IsccPath "/DMyAppVersion=$version" $isscPath
    if ($LASTEXITCODE -ne 0) {
        throw 'ISCC fallo al compilar installer\upflow.iss.'
    }

    $setupExePath = Join-Path $distDir "upflow-setup-v$version.exe"
    if (-not (Test-Path $setupExePath)) {
        throw "ISCC termino sin errores pero no se genero $setupExePath."
    }
    Write-Host 'Instalador generado:' $setupExePath
}

function Add-ApolloModelToInstaller {
    # El modelo Apollo (~74MB) se BUNDLEA en el instalador (a diferencia del resto
    # de binarios vendored, que se bajan de repos publicos upstream). Motivo: se
    # hostea en el release PRIVADO, asi que descargarlo falla para quien no tiene
    # acceso al repo; incluirlo en el setup.exe da el restore con un solo .exe.
    # El [Files] del .iss copia build\app\* recursivo, asi que basta con stagearlo aca.
    $apolloSrc = Join-Path $root 'vendor\apollo\apollo.onnx'
    if (-not (Test-Path $apolloSrc)) {
        Write-Host 'AVISO: vendor\apollo\apollo.onnx no encontrado; el instalador NO incluira el restore de audio.'
        return
    }
    $apolloDst = Join-Path $installerAppDir 'vendor\apollo'
    New-Item -ItemType Directory -Force -Path $apolloDst | Out-Null
    Copy-Item -Force $apolloSrc (Join-Path $apolloDst 'apollo.onnx')
    Write-Host 'Modelo Apollo (restauracion de audio) incluido en el instalador.'
}

function Add-RealesrganOnnxModelsToInstaller {
    # Los exports ONNX uint8 (backend rapido SP11) se BUNDLEAN en el instalador,
    # igual que Apollo. Aunque los .pth de origen son publicos, el export local
    # exige torch+onnx (~2GB) y varios minutos; incluir los .onnx ya horneados
    # (animevideov3 x2/x3/x4 ~2.5MB c/u, x4plus ~67MB, anime ~18MB) le da al
    # usuario el runtime ONNX (2x en video) sin instalar torch ni exportar nada.
    # Solo se copian los .onnx: la subcarpeta weights\ (.pth) no viaja.
    $onnxSrcDir = Join-Path $root 'vendor\realesrgan-onnx'
    $onnxFiles = @()
    if (Test-Path $onnxSrcDir) {
        $onnxFiles = Get-ChildItem -Path $onnxSrcDir -Filter '*.onnx' -File
    }
    if ($onnxFiles.Count -eq 0) {
        Write-Host 'AVISO: no hay exports en vendor\realesrgan-onnx; el instalador usara solo NCNN (correr scripts\download-realesrgan-onnx.ps1 para el backend ONNX).'
        return
    }
    $onnxDst = Join-Path $installerAppDir 'vendor\realesrgan-onnx'
    New-Item -ItemType Directory -Force -Path $onnxDst | Out-Null
    foreach ($file in $onnxFiles) {
        Copy-Item -Force $file.FullName (Join-Path $onnxDst $file.Name)
    }
    Write-Host "Modelos ONNX ($($onnxFiles.Count)) incluidos en el instalador (backend rapido SP11)."
}

function New-Installer {
    param([string]$IsccPath)

    Initialize-EmbeddedPython
    Copy-AppAllowlist -Destination $installerAppDir
    Add-ApolloModelToInstaller
    Add-RealesrganOnnxModelsToInstaller
    New-Item -ItemType Directory -Force -Path $distDir | Out-Null
    Invoke-InnoSetupCompile -IsccPath $IsccPath
}

# ISCC se valida primero (antes del build de frontend, que es lento) para
# fallar rapido si Inno Setup no esta instalado.
$isccPathForInstaller = $null
if ($Installer) {
    $isccPathForInstaller = Assert-IsccAvailable
}

Build-Frontend

if ($Zip) {
    New-PortableZip
}

if ($Installer) {
    New-Installer -IsccPath $isccPathForInstaller
}

Write-Host ''
Write-Host 'Version:' $version
