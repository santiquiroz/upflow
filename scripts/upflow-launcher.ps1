# 'Continue', NO 'Stop': el launcher maneja errores con checks manuales de
# $LASTEXITCODE + throw explicitos en cada llamada nativa. Con 'Stop',
# Windows PowerShell convierte CUALQUIER escritura a stderr de un comando
# nativo en un error terminante — incluso con 2>$null — asi que el probe
# esperado-a-fallar `python -c "import uvicorn"` (falla a proposito en una
# instalacion fresca, antes de instalar deps) crasheaba el arranque con un
# "Traceback (most recent call last):" en vez de devolver $LASTEXITCODE=1.
$ErrorActionPreference = 'Continue'

# Resolved against the script's own location, so the launcher works no
# matter what directory it was double-clicked or invoked from.
$root = Split-Path -Parent $PSScriptRoot
$venvPath = Join-Path $root '.venv'
$venvPython = Join-Path $venvPath 'Scripts\python.exe'

# The installer bundles a Python 3.12 embeddable + pip at {app}\python (see
# installer/upflow.iss and package-release.ps1 -Installer). When present it
# takes priority and no system Python / venv is needed at all; the portable
# zip has no python\ folder, so it falls back to the venv-based flow below.
$bundledPythonPath = Join-Path $root 'python\python.exe'
$usingBundledPython = Test-Path $bundledPythonPath
$pythonExe = if ($usingBundledPython) { $bundledPythonPath } else { $venvPython }
$installedSentinel = if ($usingBundledPython) {
    Join-Path $root 'python\.upflow-installed'
} else {
    Join-Path $venvPath '.upflow-installed'
}

$envPath = Join-Path $root '.env'
$envExamplePath = Join-Path $root '.env.example'
$minPythonMajor = 3
$minPythonMinor = 11

function Write-Step {
    param([string]$Message)
    Write-Host ''
    Write-Host "==> $Message" -ForegroundColor Cyan
}

function Get-SystemPythonVersion {
    $pythonCommand = Get-Command python -ErrorAction SilentlyContinue
    if (-not $pythonCommand) {
        return $null
    }
    $versionOutput = python -c "import sys; print(f'{sys.version_info[0]}.{sys.version_info[1]}')" 2>$null
    if ($LASTEXITCODE -ne 0 -or -not $versionOutput) {
        return $null
    }
    return $versionOutput.Trim()
}

function Assert-SystemPythonOk {
    $version = Get-SystemPythonVersion
    $winget = Get-Command winget -ErrorAction SilentlyContinue

    if (-not $version) {
        $message = "No se encontro Python en el PATH del sistema.`n`n" +
            "Instalalo desde https://www.python.org/downloads/ (marca 'Add python.exe to PATH' durante la instalacion)"
        if ($winget) {
            $message += "`n`nO, si preferis winget, abri una consola y corre:`n  winget install Python.Python.3.12"
        }
        $message += "`n`nLuego volve a ejecutar Upflow.bat."
        throw $message
    }

    $parts = $version.Split('.')
    $major = [int]$parts[0]
    $minor = [int]$parts[1]
    $isOldVersion = ($major -lt $minPythonMajor) -or ($major -eq $minPythonMajor -and $minor -lt $minPythonMinor)
    if ($isOldVersion) {
        $message = "Se encontro Python $version, pero Upflow necesita Python $minPythonMajor.$minPythonMinor o superior.`n`n" +
            "Instala una version mas nueva desde https://www.python.org/downloads/"
        if ($winget) {
            $message += "`n`nO con winget:`n  winget install Python.Python.3.12"
        }
        throw $message
    }

    Write-Host "Python $version detectado en el PATH."
}

function Test-UpflowAlreadyInstalled {
    if (Test-Path $installedSentinel) {
        return $true
    }
    # Probe una dep de terceros (uvicorn), NO el paquete local `app`: con cwd en
    # la raiz de instalacion (que contiene app/), un interprete estandar mete
    # cwd en sys.path, asi que `import app` tiene exito hasta en un .venv fresco
    # SIN deps -> el check daria "ya instalado", se saltaria pip install, y
    # Start-Upflow crashearia al correr uvicorn (no instalado). uvicorn solo
    # existe despues de `pip install -e .`, asi que es el proxy correcto de
    # "deps instaladas" en ambas ramas (embebida y venv).
    & $pythonExe -c "import uvicorn" 2>$null
    return ($LASTEXITCODE -eq 0)
}

function Install-PythonEnvironment {
    if ($usingBundledPython) {
        Install-BundledPythonDependencies
    } else {
        Install-VenvPythonDependencies
    }
}

function Install-BundledPythonDependencies {
    if (Test-UpflowAlreadyInstalled) {
        Write-Host 'Upflow ya esta instalado en el Python embebido, se omite este paso.'
        New-Item -ItemType File -Force -Path $installedSentinel | Out-Null
        return
    }

    Write-Step 'Instalando Upflow con el Python embebido (primera vez, puede tardar un minuto)...'
    & $pythonExe -m pip install --upgrade pip --quiet
    if ($LASTEXITCODE -ne 0) {
        throw 'No se pudo actualizar pip en el Python embebido.'
    }
    # --no-build-isolation: el instalador ya deja setuptools/wheel instalados
    # en el Python embebido (ver package-release.ps1 Initialize-EmbeddedPython).
    # pip build-isolation normal falla aca porque inyecta las build
    # dependencies via PYTHONPATH en un subproceso, y el ._pth del embebido
    # ignora PYTHONPATH por diseno (falla con "BackendUnavailable: Cannot
    # import 'setuptools.build_meta'" si se omite este flag).
    & $pythonExe -m pip install --no-build-isolation --quiet -e $root
    if ($LASTEXITCODE -ne 0) {
        throw 'No se pudo instalar Upflow (pip install -e .). Revisa tu conexion a internet.'
    }
    New-Item -ItemType File -Force -Path $installedSentinel | Out-Null
}

function Install-VenvPythonDependencies {
    if (-not (Test-Path $venvPython)) {
        Write-Step 'Creando entorno virtual (.venv)...'
        python -m venv $venvPath
        if ($LASTEXITCODE -ne 0) {
            throw 'No se pudo crear el entorno virtual (.venv). Revisa la instalacion de Python.'
        }
    }

    if (Test-UpflowAlreadyInstalled) {
        Write-Host 'Upflow ya esta instalado en el entorno virtual, se omite este paso.'
        New-Item -ItemType File -Force -Path $installedSentinel | Out-Null
        return
    }

    Write-Step 'Instalando Upflow (primera vez, puede tardar un minuto)...'
    & $venvPython -m pip install --upgrade pip --quiet
    if ($LASTEXITCODE -ne 0) {
        throw 'No se pudo actualizar pip en el entorno virtual.'
    }
    & $venvPython -m pip install --quiet -e $root
    if ($LASTEXITCODE -ne 0) {
        throw 'No se pudo instalar Upflow (pip install -e .). Revisa tu conexion a internet.'
    }
    New-Item -ItemType File -Force -Path $installedSentinel | Out-Null
}

function Test-RealesrganPresent {
    $binary = Join-Path $root 'vendor\realesrgan\realesrgan-ncnn-vulkan.exe'
    $sampleModel = Join-Path $root 'vendor\realesrgan\models\realesrgan-x4plus.param'
    return (Test-Path $binary) -and (Test-Path $sampleModel)
}

function Test-FfmpegPresent {
    return Test-Path (Join-Path $root 'vendor\ffmpeg\bin\ffmpeg.exe')
}

function Test-RifePresent {
    $binary = Join-Path $root 'vendor\rife\rife-ncnn-vulkan.exe'
    $defaultModel = Join-Path $root 'vendor\rife\models\rife-v4.25'
    return (Test-Path $binary) -and (Test-Path $defaultModel)
}

function Test-DeepfilternetPresent {
    $binary = Join-Path $root 'vendor\deepfilternet\deep-filter.exe'
    $rnnoiseModel = Join-Path $root 'vendor\deepfilternet\models\sh.rnnn'
    return (Test-Path $binary) -and (Test-Path $rnnoiseModel)
}

function Invoke-DownloadScript {
    param(
        [string]$ScriptName,
        [string]$Label
    )
    $scriptPath = Join-Path $root "scripts\$ScriptName"
    Write-Step "Descargando $Label (puede tardar varios minutos segun tu conexion)..."
    & powershell -NoProfile -ExecutionPolicy Bypass -File $scriptPath
    if ($LASTEXITCODE -ne 0) {
        throw "No se pudo descargar $Label. Revisa tu conexion a internet y volve a intentar."
    }
}

function Install-MissingBinaries {
    if (Test-RealesrganPresent) {
        Write-Host 'Real-ESRGAN NCNN Vulkan ya esta descargado.'
    } else {
        Invoke-DownloadScript -ScriptName 'download-realesrgan.ps1' -Label 'Real-ESRGAN NCNN Vulkan (motor de upscaling)'
    }

    if (Test-FfmpegPresent) {
        Write-Host 'FFmpeg ya esta descargado.'
    } else {
        Invoke-DownloadScript -ScriptName 'download-ffmpeg.ps1' -Label 'FFmpeg'
    }

    if (Test-RifePresent) {
        Write-Host 'RIFE NCNN Vulkan ya esta descargado.'
    } else {
        Invoke-DownloadScript -ScriptName 'download-rife.ps1' -Label 'RIFE NCNN Vulkan (FPS boost)'
    }

    if (Test-DeepfilternetPresent) {
        Write-Host 'DeepFilterNet ya esta descargado.'
    } else {
        Invoke-DownloadScript -ScriptName 'download-deepfilternet.ps1' -Label 'DeepFilterNet (mejora de audio con IA)'
    }

    # download-apollo.ps1 se auto-saltea si el modelo ya esta presente.
    Invoke-DownloadScript -ScriptName 'download-apollo.ps1' -Label 'Apollo (restauracion de audio por compresion, experimental)'
}

function New-EnvFileWithFeaturesEnabled {
    if (Test-Path $envPath) {
        Write-Host 'Archivo .env ya existe, no se modifica.'
        return
    }

    Write-Step 'Generando .env con el FPS boost y la mejora de audio activados...'
    $lines = Get-Content $envExamplePath
    $lines = $lines -replace '^ENABLE_INTERPOLATION=.*', 'ENABLE_INTERPOLATION=True'
    $lines = $lines -replace '^ENABLE_AUDIO_ENHANCE=.*', 'ENABLE_AUDIO_ENHANCE=True'
    $lines = $lines -replace '^ENABLE_AUDIO_RESTORE=.*', 'ENABLE_AUDIO_RESTORE=True'
    Set-Content -Path $envPath -Value $lines -Encoding utf8
}

function Get-EnvValue {
    param(
        [string]$Key,
        [string]$Default
    )
    if (-not (Test-Path $envPath)) {
        return $Default
    }
    $line = Get-Content $envPath | Where-Object { $_ -match "^\s*$Key\s*=" } | Select-Object -First 1
    if (-not $line) {
        return $Default
    }
    # .env.example values carry trailing "  # explicacion" comments; strip
    # those before trimming or callers get a broken value (e.g. a port
    # string with a comment glued to it).
    $rawValue = ($line -split '=', 2)[1]
    $valueWithoutComment = ($rawValue -split '#', 2)[0]
    return $valueWithoutComment.Trim()
}

function Start-Upflow {
    $appHost = Get-EnvValue -Key 'APP_HOST' -Default '127.0.0.1'
    $appPort = Get-EnvValue -Key 'APP_PORT' -Default '8090'
    $browserHost = if ($appHost -eq '0.0.0.0') { '127.0.0.1' } else { $appHost }
    $url = "http://${browserHost}:${appPort}"
    $healthUrl = "$url/api/v1/health"

    $browserJob = Start-Job -ScriptBlock {
        param($Url, $HealthUrl)
        $deadline = (Get-Date).AddSeconds(30)
        while ((Get-Date) -lt $deadline) {
            try {
                $response = Invoke-WebRequest -Uri $HealthUrl -UseBasicParsing -TimeoutSec 2
                if ($response.StatusCode -eq 200) {
                    Start-Process $Url
                    return
                }
            } catch {
                Start-Sleep -Milliseconds 500
            }
        }
    } -ArgumentList $url, $healthUrl

    Write-Step "Iniciando Upflow en $url ..."
    Write-Host 'La ventana va a mostrar los logs del servidor. Cerra esta ventana o presiona Ctrl+C para detenerlo.'

    try {
        & $pythonExe -m uvicorn app.main:app --host $appHost --port $appPort
    } finally {
        Stop-Job $browserJob -ErrorAction SilentlyContinue | Out-Null
        Remove-Job $browserJob -Force -ErrorAction SilentlyContinue | Out-Null
        Write-Host ''
        Write-Host 'Upflow se detuvo.'
    }
}

function Main {
    Write-Host '=== Upflow ===' -ForegroundColor Green

    # uvicorn importa `app` y pydantic-settings lee `.env` desde el CWD del
    # proceso: si el .bat se lanza desde otra carpeta (Start-Process, acceso
    # directo sin "Iniciar en"), el server serviria OTRO codigo/config.
    Set-Location $root

    Write-Step 'Verificando Python...'
    if ($usingBundledPython) {
        Write-Host "Python embebido detectado en $bundledPythonPath, no hace falta Python del sistema."
    } else {
        Assert-SystemPythonOk
    }

    Install-PythonEnvironment
    Install-MissingBinaries
    New-EnvFileWithFeaturesEnabled
    Start-Upflow
}

try {
    Main
} catch {
    Write-Host ''
    Write-Host 'Ocurrio un error y Upflow no pudo iniciar:' -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Red
    Write-Host ''
    Read-Host 'Presiona Enter para cerrar esta ventana'
    exit 1
}
