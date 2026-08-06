$ErrorActionPreference = 'Stop'

# OpenSCAD es GPL-2.0. Por eso se baja aparte y corre como PROCESO SEPARADO,
# nunca enlazado contra la app — el mismo trato que ffmpeg. Bajarlo acá no
# cambia la licencia de nada: sigue siendo un ejecutable que se invoca.

[Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12

$root = Split-Path -Parent $PSScriptRoot
$vendorDir = Join-Path $root 'vendor\openscad'
$tempDir = Join-Path $root 'runtime\temp'
$zipPath = Join-Path $tempDir 'openscad-portable.zip'

# La build portable, no el instalador: no toca el sistema ni pide permisos de
# administrador, y queda dentro del proyecto igual que los demás motores.
$downloadUrl = 'https://files.openscad.org/OpenSCAD-2021.01-x86-64.zip'

New-Item -ItemType Directory -Force -Path $tempDir | Out-Null

Write-Host 'Bajando OpenSCAD (portable, ~35 MB)...'
Invoke-WebRequest -Uri $downloadUrl -OutFile $zipPath

$extractDir = Join-Path $tempDir 'openscad-extract'
if (Test-Path $extractDir) {
    Remove-Item -Recurse -Force $extractDir
}
Expand-Archive -Path $zipPath -DestinationPath $extractDir -Force

$binario = Get-ChildItem $extractDir -Recurse -Filter 'openscad.exe' | Select-Object -First 1
if (-not $binario) {
    throw 'El ZIP se extrajo pero no apareció openscad.exe adentro.'
}

if (Test-Path $vendorDir) {
    Remove-Item -Recurse -Force $vendorDir
}
Copy-Item -Recurse -Force $binario.Directory.FullName $vendorDir
Remove-Item -Recurse -Force $extractDir
Remove-Item -Force $zipPath

$destino = Join-Path $vendorDir 'openscad.exe'
if (-not (Test-Path $destino)) {
    throw "Se copió el directorio pero no quedó openscad.exe en $destino"
}

Write-Host 'OpenSCAD listo en:' $destino
Write-Host ''
Write-Host 'Falta la otra mitad del carril: un servidor de modelo local que hable'
Write-Host 'el protocolo de OpenAI (Ollama, LM Studio, llama.cpp server).'
Write-Host 'Poné su dirección en los ajustes, por ejemplo:'
Write-Host '  CAD_LLM_BASE_URL=http://localhost:11434/v1'
