$ErrorActionPreference = 'Stop'

# Older Windows PowerShell 5.1 defaults to TLS 1.0, which GitHub rejects.
[Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12

# Conversion de timbre con OpenVoice V2, por ONNX.
#
# Reemplaza al camino SpeechT5 para "cambiar de voz". Se puede bajar y
# redistribuir porque los pesos de OpenVoice son MIT: los clonadores que salen
# primero en cualquier busqueda (F5-TTS, E2-TTS, XTTS) los publican con CC-BY-NC,
# o sea no comercial, y eso no se puede meter en una app que otra gente descarga.
#
# Los grafos salen de un port propio, con la paridad medida contra el modelo
# original: https://github.com/santiquiroz/port-openvoice-onnx

$root = Split-Path -Parent $PSScriptRoot
$destino = Join-Path $root 'vendor\openvoice'
$tempDir = Join-Path $root 'runtime\temp'

# Tag fijo y no "latest": una descarga reproducible es la diferencia entre un
# fallo que se puede reproducir y uno que depende del dia.
$tag = 'models-v1.0'
$repo = 'santiquiroz/port-openvoice-onnx'

$archivos = @(
    @{ nombre = 'openvoice_converter.onnx'; mb = 122 },
    @{ nombre = 'openvoice_speaker.onnx';   mb = 3   }
)

New-Item -ItemType Directory -Force $destino | Out-Null
New-Item -ItemType Directory -Force $tempDir | Out-Null

foreach ($archivo in $archivos) {
    $final = Join-Path $destino $archivo.nombre
    if (Test-Path $final) {
        Write-Host "$($archivo.nombre) ya esta"
        continue
    }

    $url = "https://github.com/$repo/releases/download/$tag/$($archivo.nombre)"
    # A un temporal y recien despues al destino: una descarga cortada a mitad
    # dejaria un .onnx truncado con el nombre bueno, y la app lo daria por
    # instalado hasta que fallara al cargarlo.
    $temporal = Join-Path $tempDir "$($archivo.nombre).download"
    Write-Host "==> Descargando $($archivo.nombre) (~$($archivo.mb) MB)..."
    Invoke-WebRequest -Uri $url -OutFile $temporal -UseBasicParsing

    $mbReales = (Get-Item $temporal).Length / 1MB
    if ($mbReales -lt ($archivo.mb * 0.5)) {
        Remove-Item $temporal -Force
        throw "$($archivo.nombre) bajo con $([math]::Round($mbReales)) MB y se esperaban ~$($archivo.mb). Reintenta."
    }
    Move-Item -Force $temporal $final
}

Write-Host ""
Write-Host "Listo. Conversion de voz con OpenVoice en $destino"
