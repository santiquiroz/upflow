$ErrorActionPreference = 'Stop'

# Older Windows PowerShell 5.1 defaults to TLS 1.0, which GitHub rejects.
[Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12

# Descarga de YouTube: motor JS + acuñador de PO Token.
#
# YouTube exige DOS cosas que la app no traia:
#   1. Resolver un desafio en JavaScript, que necesita un motor JS.
#   2. Un "PO Token" (proof-of-origin). Sin el, la extraccion anda pero bajar el
#      medio devuelve 403.
# Faltando cualquiera de las dos, no hay descarga de YouTube. Los demas sitios
# siguen funcionando sin nada de esto.
#
# Se baja Deno y NO el acuñador ya compilado porque Deno cubre los dos roles: es
# el motor JS de yt-dlp y ademas ejecuta el acuñador. Un binario de ~40 MB en vez
# de ~150 MB entre los dos.

$root = Split-Path -Parent $PSScriptRoot
$denoDir = Join-Path $root 'vendor\deno'
$cecaDir = Join-Path $root 'vendor\ceca'
$denoPath = Join-Path $denoDir 'deno.exe'
$tempDir = Join-Path $root 'runtime\temp'
$tempZip = Join-Path $tempDir 'deno.zip.download'

# Tags fijos y no "latest": una descarga reproducible es la diferencia entre un
# fallo que se puede reproducir y uno que depende del dia.
$denoVersion = 'v2.9.5'
$denoUrl = "https://github.com/denoland/deno/releases/download/$denoVersion/deno-x86_64-pc-windows-msvc.zip"

# github.com/santiquiroz/ceca (MIT). Se baja el CODIGO, que corre Deno.
$cecaRef = 'master'
$cecaFiles = @('src/main.ts', 'src/botguard.ts', 'deno.json')

New-Item -ItemType Directory -Force $denoDir | Out-Null
New-Item -ItemType Directory -Force $cecaDir | Out-Null
New-Item -ItemType Directory -Force $tempDir | Out-Null

if (Test-Path $denoPath) {
    Write-Host "Deno ya esta en $denoPath"
} else {
    Write-Host "==> Descargando Deno $denoVersion (motor JS)..."
    Invoke-WebRequest -Uri $denoUrl -OutFile $tempZip -UseBasicParsing
    Expand-Archive -Path $tempZip -DestinationPath $denoDir -Force
    Remove-Item $tempZip -Force
    if (-not (Test-Path $denoPath)) {
        throw "El zip de Deno no contenia deno.exe"
    }
}

Write-Host "==> Descargando ceca (acuñador de PO Token)..."
foreach ($file in $cecaFiles) {
    $url = "https://raw.githubusercontent.com/santiquiroz/ceca/$cecaRef/$file"
    # Aplanado: el acuñador se invoca por main.ts y no necesita el arbol.
    $destination = Join-Path $cecaDir (Split-Path -Leaf $file)
    Invoke-WebRequest -Uri $url -OutFile $destination -UseBasicParsing
}

# El import de main.ts es "./botguard.ts", que al aplanar sigue resolviendo
# porque los dos quedan en la misma carpeta. No hay nada que reescribir.

Write-Host "==> Precargando las dependencias del acuñador..."
# Ahora y no en la primera descarga: bajar npm a mitad de un trabajo lo hace
# parecer colgado, y aca el usuario ya esta esperando una instalacion.
& $denoPath cache (Join-Path $cecaDir 'main.ts')

Write-Host ""
Write-Host "Listo. Deno en $denoPath"
Write-Host "Acuñador en $cecaDir"
Write-Host "Agrega a tu .env si cambiaste las rutas por defecto:"
Write-Host "  DENO_BINARY=vendor/deno/deno.exe"
Write-Host "  CECA_ENTRYPOINT=vendor/ceca/main.ts"
