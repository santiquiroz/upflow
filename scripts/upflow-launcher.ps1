param(
    # Instala todo sin preguntar. Lo usa el instalador silencioso y CI.
    [switch]$InstallAll,
    # Instala solo lo imprescindible: el motor de upscaling y ffmpeg.
    [switch]$SkipOptional
)

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

# El Python embebido trae `import site` habilitado (lo necesita pip), y eso arrastra
# el site-packages de USUARIO del Python del sistema (%APPDATA%\Python\PythonXY\
# site-packages), que se comparte por version de Python. En una maquina con Python
# instalado y paquetes propios, Upflow terminaba cargando la mitad de sus
# dependencias desde ahi: reportado en vivo como "se abre y se cierra sola", con un
# ModuleNotFoundError de pydantic_settings mientras uvicorn se resolvia desde
# AppData\Roaming. Peor: el chequeo de "ya esta instalado" (import uvicorn) daba
# verdadero por ese uvicorn ajeno y se salteaba la instalacion entera.
$env:PYTHONNOUSERSITE = '1'
$env:PYTHONPATH = ''
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

# La version del pyproject que viaja con el codigo es la del codigo que se va a
# correr. El sentinel guarda la version con la que se hizo el ultimo
# `pip install -e .`: si difieren, hubo una actualizacion y hay que reinstalar
# para que la metadata del paquete no quede congelada en la version vieja.
$declaredVersion = $null
$pyprojectPath = Join-Path $root 'pyproject.toml'
if (Test-Path $pyprojectPath) {
    $versionLine = Select-String -Path $pyprojectPath -Pattern '^version\s*=\s*"([^"]+)"' | Select-Object -First 1
    if ($versionLine) {
        $declaredVersion = $versionLine.Matches[0].Groups[1].Value
    }
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

function Test-RuntimeDepsPresent {
    # Tres dependencias de terceros que el arranque necesita si o si. Se prueban
    # juntas y NO solo uvicorn: una instalacion a medias (o contaminada por el
    # Python del sistema) tenia uvicorn y le faltaba pydantic_settings, y el
    # servidor moria recien al importar la config.
    & $pythonExe -c "import uvicorn, pydantic_settings, fastapi" 2>$null
    return ($LASTEXITCODE -eq 0)
}

function Test-UpflowAlreadyInstalled {
    if (Test-Path $installedSentinel) {
        # Un sentinel de una version distinta significa que el instalador
        # actualizo el codigo por debajo: reinstalar para regenerar la metadata
        # del paquete. Sin esto, `pip install -e .` corria UNA sola vez en la
        # vida de la instalacion y la version reportada quedaba clavada para
        # siempre (visto en instalaciones reales: codigo 0.14.0, metadata
        # 0.10.0, y el banner de "hay version nueva" que nunca se iba).
        $stamped = (Get-Content -Path $installedSentinel -Raw -ErrorAction SilentlyContinue)
        if ($stamped) { $stamped = $stamped.Trim() }
        if ($declaredVersion -and $stamped -ne $declaredVersion) {
            Write-Host "Actualizacion detectada ($stamped -> $declaredVersion): se reinstala el paquete."
            return $false
        }
        # El sentinel dice "instalado", pero eso no garantiza que el entorno SIGA
        # sano: una instalacion que se completo tomando dependencias del Python
        # del sistema deja el sentinel escrito y el entorno roto. Se comprueba.
        if (-not (Test-RuntimeDepsPresent)) {
            Write-Host 'El entorno de Python quedo incompleto: se reinstalan las dependencias.' -ForegroundColor Yellow
            return $false
        }
        return $true
    }
    # Probe una dep de terceros (uvicorn), NO el paquete local `app`: con cwd en
    # la raiz de instalacion (que contiene app/), un interprete estandar mete
    # cwd en sys.path, asi que `import app` tiene exito hasta en un .venv fresco
    # SIN deps -> el check daria "ya instalado", se saltaria pip install, y
    # Start-Upflow crashearia al correr uvicorn (no instalado). uvicorn solo
    # existe despues de `pip install -e .`, asi que es el proxy correcto de
    # "deps instaladas" en ambas ramas (embebida y venv).
    return (Test-RuntimeDepsPresent)
}

function Install-PythonEnvironment {
    if ($usingBundledPython) {
        Install-BundledPythonDependencies
    } else {
        Install-VenvPythonDependencies
    }
}

function Get-WheelVersion {
    param([string]$FileName)
    # nombre-version-pytag-abi-plataforma.whl
    $parts = $FileName -split '-'
    if ($parts.Count -lt 2) { return $null }
    try { return [version]$parts[1] } catch { return $null }
}

function Select-NewestWheelPerPackage {
    param([System.IO.FileInfo[]]$Wheels)
    # Una actualizacion deja las wheels viejas al lado de la nueva (el instalador
    # copia, no reemplaza el directorio). Instalarlas todas en fila hacia que pip
    # dejara instalada una version vieja a mitad de camino y escupiera un
    # "requires fetchflow>=X but you have Y" que asustaba sin ser un problema real.
    # Se instala solo la mas nueva de cada paquete.
    if ($null -eq $Wheels -or $Wheels.Count -eq 0) { return @() }
    return @(
        $Wheels | Group-Object { ($_.Name -split '-')[0] } | ForEach-Object {
            $_.Group | Sort-Object @{ Expression = { Get-WheelVersion $_.Name } }, Name | Select-Object -Last 1
        }
    )
}

function Install-VendoredWheels {
    # Wheels que viajan EN el instalador, instaladas ANTES de `pip install -e .` para
    # que pip las vea ya satisfechas y no salga a buscarlas.
    #
    # Hoy es fetchflow (github.com/santiquiroz/fetchflow), el motor del descargador.
    # Va asi y no como dependencia `git+` porque esa forma exigiria el binario git en
    # esta maquina, y sin el fallaria la instalacion ENTERA de Upflow -- no solo el
    # apartado de descargas. Tampoco desde PyPI todavia: no esta publicado ahi.
    #
    # Ausente no es fatal a proposito: una instalacion desde el repo (sin instalador)
    # no tiene vendor\wheels, y ahi se espera que fetchflow ya este en el entorno.
    $wheelDir = Join-Path $root 'vendor\wheels'
    if (-not (Test-Path $wheelDir)) { return }
    $wheels = @(Select-NewestWheelPerPackage (Get-ChildItem -Path $wheelDir -Filter '*.whl' -File))
    if ($wheels.Count -eq 0) { return }

    Write-Step "Instalando $($wheels.Count) componente(s) incluido(s)..."
    foreach ($wheel in $wheels) {
        # SIN --no-deps: la wheel trae solo fetchflow, y sus dependencias (yt-dlp,
        # curl-cffi) tienen que bajarse igual. Con --no-deps el descargador quedaria
        # instalado y sin motor, fallando recien al primer uso.
        #
        # --no-warn-script-location: los .exe de las dependencias (yt-dlp) van a
        # Scripts\ del Python embebido, que no esta en PATH y no tiene por que
        # estarlo -- se usan como libreria. El aviso solo asustaba.
        & $pythonExe -m pip install --quiet --force-reinstall --no-warn-script-location $wheel.FullName
        if ($LASTEXITCODE -ne 0) {
            throw "No se pudo instalar $($wheel.Name)."
        }
    }
}

function Install-BundledPythonDependencies {
    if (Test-UpflowAlreadyInstalled) {
        Write-Host 'Upflow ya esta instalado en el Python embebido, se omite este paso.'
        Write-InstalledSentinel
        return
    }

    Write-Step 'Instalando Upflow con el Python embebido (primera vez, puede tardar un minuto)...'
    & $pythonExe -m pip install --upgrade pip --quiet
    if ($LASTEXITCODE -ne 0) {
        throw 'No se pudo actualizar pip en el Python embebido.'
    }
    Install-VendoredWheels

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
    Remove-StaleEggInfo
    Write-InstalledSentinel
}

function Remove-StaleEggInfo {
    # setuptools viejo dejaba un upflow.egg-info en la RAIZ del proyecto. La app
    # arranca con `python -m uvicorn` desde esa raiz, asi que el cwd entra en
    # sys.path y ese egg-info le gana al dist-info bueno de site-packages: una
    # instalacion real reportaba 0.10.0 (egg-info de la primera instalacion)
    # teniendo 0.13.0 en site-packages y codigo 0.14.0.
    $eggInfo = Join-Path $root 'upflow.egg-info'
    if (Test-Path $eggInfo) {
        Remove-Item -Recurse -Force $eggInfo -ErrorAction SilentlyContinue
    }
}

function Write-InstalledSentinel {
    if ($declaredVersion) {
        Set-Content -Path $installedSentinel -Value $declaredVersion -NoNewline -Encoding ascii
    } else {
        New-Item -ItemType File -Force -Path $installedSentinel | Out-Null
    }
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
        Write-InstalledSentinel
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
    Remove-StaleEggInfo
    Write-InstalledSentinel
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

# Los paquetes que el usuario PUEDE saltear, descritos por lo que habilitan y no
# por el nombre del binario: en el primer arranque nadie sabe que es "RIFE".
#
# Saltear es seguro y reversible: la pantalla de tareas de la app muestra cada
# capacidad a la que le falta su paquete, con un boton que corre ESTE MISMO script.
# Por eso preguntar no deja a nadie encerrado.
$script:OptionalPacks = @(
    @{
        Key     = 'rife'
        Feature = 'Generar fotogramas: duplicar o triplicar los FPS de un video'
        Size    = '~45 MB'
    },
    @{
        Key     = 'deepfilternet'
        Feature = 'Quitar ruido de audio con un modelo de IA (mas fuerte que el filtro rapido)'
        Size    = '~20 MB'
    },
    @{
        Key     = 'apollo'
        Feature = 'Restaurar los agudos que perdio un MP3 o un AAC'
        Size    = '~90 MB'
    },
    @{
        Key     = 'mobilesam'
        Feature = 'Editor: seleccionar objetos con un toque para quitarlos o reemplazarlos'
        Size    = '~45 MB'
    }
)

# Lo que el usuario tildo en el asistente del instalador. Una clave por linea.
# El instalador lo escribe SIEMPRE, incluso vacio, en CurStepChanged(ssPostInstall).
$script:OptionalPacksFile = Join-Path $root 'optional-packs.txt'

# Los logs del servidor van tambien a un archivo: si el arranque falla, la ventana
# se cierra con el error adentro y sin esto no queda rastro para diagnosticar.
$script:ServerLogPath = Join-Path $root 'runtime\logs\startup.log'

function Select-OptionalPacks {
    # NO se pregunta por consola: la eleccion se hace en el asistente grafico del
    # instalador. Aca solo se lee lo que ya se decidio.
    if ($SkipOptional) {
        Write-Host 'Se saltean los paquetes opcionales (-SkipOptional).'
        Write-Host 'Se pueden instalar despues desde la pantalla de tareas de la app.'
        return @()
    }
    if ($InstallAll) { return $script:OptionalPacks }

    # Archivo AUSENTE y archivo VACIO son distintos: ausente significa que no hubo
    # instalador (zip portable, o una instalacion previa a esta version) y ahi se
    # instala todo, que es el comportamiento historico. Vacio significa que el
    # usuario destildo las tres.
    if (-not (Test-Path $script:OptionalPacksFile)) {
        return $script:OptionalPacks
    }

    $chosenKeys = @(
        Get-Content -LiteralPath $script:OptionalPacksFile -ErrorAction SilentlyContinue |
            ForEach-Object { $_.Trim() } |
            Where-Object { $_ -ne '' }
    )
    Write-Host ('Funciones extra elegidas en el instalador: {0}' -f
        $(if ($chosenKeys.Count -gt 0) { $chosenKeys -join ', ' } else { 'ninguna' }))
    return @($script:OptionalPacks | Where-Object { $chosenKeys -contains $_.Key })
}

function Install-RequiredBinaries {
    # Estos dos NO son opcionales: Real-ESRGAN es el motor de upscaling y ffmpeg lo
    # usa todo, incluida la cadena de mejora de voz. Sin ellos la app no hace nada.
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
}

function Install-MissingBinaries {
    Install-RequiredBinaries

    $selectedKeys = @((Select-OptionalPacks) | ForEach-Object { $_.Key })

    if ($selectedKeys -contains 'rife') {
        if (Test-RifePresent) {
            Write-Host 'RIFE NCNN Vulkan ya esta descargado.'
        } else {
            Invoke-DownloadScript -ScriptName 'download-rife.ps1' -Label 'RIFE NCNN Vulkan (FPS boost)'
        }
    }

    if ($selectedKeys -contains 'deepfilternet') {
        if (Test-DeepfilternetPresent) {
            Write-Host 'DeepFilterNet ya esta descargado.'
        } else {
            Invoke-DownloadScript -ScriptName 'download-deepfilternet.ps1' -Label 'DeepFilterNet (mejora de audio con IA)'
        }
    }

    if ($selectedKeys -contains 'apollo') {
        # download-apollo.ps1 se auto-saltea si el modelo ya esta presente.
        Invoke-DownloadScript -ScriptName 'download-apollo.ps1' -Label 'Apollo (restauracion de audio por compresion, experimental)'
    }

    if ($selectedKeys -contains 'mobilesam') {
        # download-mobilesam.ps1 se auto-saltea si los dos ONNX ya estan presentes.
        Invoke-DownloadScript -ScriptName 'download-mobilesam.ps1' -Label 'MobileSAM (seleccion por toque del Editor)'
    }

    $skipped = @($script:OptionalPacks | Where-Object { $selectedKeys -notcontains $_.Key })
    if ($skipped.Count -gt 0) {
        Write-Host ''
        Write-Host 'Sin instalar (se agregan despues desde la pantalla de tareas):'
        foreach ($pack in $skipped) {
            Write-Host ('  - {0}' -f $pack.Feature)
        }
    }
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

function Test-PortInUse {
    param([int]$Port)
    # Un Upflow ya abierto (o cualquier cosa en el puerto) hacia que uvicorn
    # muriera al instante y la ventana se cerrara con el error adentro: el
    # usuario solo veia "se cierra solo".
    try {
        $listening = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction Stop
        return ($null -ne $listening)
    } catch {
        return $false
    }
}

function Wait-BeforeClosing {
    param([string]$Message)
    Write-Host ''
    Write-Host $Message -ForegroundColor Red
    Write-Host ''
    Write-Host "El detalle quedo guardado en: $script:ServerLogPath"
    if ($env:UPFLOW_NO_PAUSE) { return }
    Write-Host 'Presiona Enter para cerrar esta ventana...' -ForegroundColor Yellow
    try { [void](Read-Host) } catch { Start-Sleep -Seconds 30 }
}

function Start-Upflow {
    $appHost = Get-EnvValue -Key 'APP_HOST' -Default '127.0.0.1'
    $appPort = Get-EnvValue -Key 'APP_PORT' -Default '8090'
    $browserHost = if ($appHost -eq '0.0.0.0') { '127.0.0.1' } else { $appHost }
    $url = "http://${browserHost}:${appPort}"
    $healthUrl = "$url/api/v1/health"

    if (Test-PortInUse -Port ([int]$appPort)) {
        Write-Host ''
        Write-Host "Upflow ya parece estar abierto: algo esta usando el puerto $appPort." -ForegroundColor Yellow
        Write-Host "Se abre la ventana que ya esta funcionando en vez de iniciar otra."
        Start-Process $url
        return
    }

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

    $exitCode = 0
    New-Item -ItemType Directory -Force (Split-Path -Parent $script:ServerLogPath) | Out-Null
    try {
        # Tee: los logs siguen a la vista Y quedan en un archivo. Sin esto, si el
        # servidor moria al arrancar la ventana se cerraba con el error adentro y
        # no habia forma de saber que paso.
        & $pythonExe -m uvicorn app.main:app --host $appHost --port $appPort 2>&1 |
            Tee-Object -FilePath $script:ServerLogPath
        $exitCode = $LASTEXITCODE
    } finally {
        Stop-Job $browserJob -ErrorAction SilentlyContinue | Out-Null
        Remove-Job $browserJob -Force -ErrorAction SilentlyContinue | Out-Null
        Write-Host ''
        Write-Host 'Upflow se detuvo.'
    }

    if ($exitCode -ne 0) {
        Wait-BeforeClosing -Message "Upflow no pudo iniciarse (codigo $exitCode). El motivo esta unas lineas mas arriba."
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
