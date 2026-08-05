$ErrorActionPreference = 'Stop'

# Generacion de voz desde texto (Kokoro).
#
# Kokoro es Apache 2.0 y corre por ONNX, igual que el resto de los modelos de la
# app. Se eligio sobre MMS-TTS, que anda igual de bien pero es CC-BY-NC-4.0 (NO
# comercial) — ver docs/superpowers/specs/2026-08-04-tts-spike-findings.md.
#
# IMPORTANTE, medido el 2026-08-04 con onnxruntime 1.24.4: de las seis variantes
# ONNX que publica el repo, DOS fallan y ninguna avisa bien.
#   - model_q8f16.onnx  -> segfault al crear la sesion, se lleva el proceso.
#   - model_fp16.onnx   -> carga, corre, devuelve audio de la duracion correcta
#                          y ese audio es TODO NaN.
# Por eso se baja model.onnx (fp32) y nada mas.

[Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12

$root = Split-Path -Parent $PSScriptRoot
$destDir = Join-Path $root 'vendor\kokoro'
$repo = 'onnx-community/Kokoro-82M-v1.0-ONNX'
$base = "https://huggingface.co/$repo/resolve/main"

# Voces incluidas por defecto: dos en ingles (una de cada genero) para no bajar
# las 50 y pico que trae el repo.
# Una voz por idioma al que se puede doblar: Kokoro nombra cada voz con la
# inicial del idioma, y doblar al espanol con voz inglesa suena a extranjero
# leyendo.
$voices = @('af_heart', 'am_michael', 'ef_dora', 'em_alex', 'ff_siwis', 'if_sara', 'pf_dora')

$modelPath = Join-Path $destDir 'model.onnx'
New-Item -ItemType Directory -Force -Path (Join-Path $destDir 'voices') | Out-Null

# Se revisa el modelo Y las voces, no solo el modelo: una instalacion vieja tiene
# el modelo pero le faltan las voces de los idiomas que se agregaron despues, y
# salir temprano la dejaria doblando espanol con voz inglesa para siempre.
$faltantes = @($voices | Where-Object { -not (Test-Path (Join-Path $destDir "voices\$_.bin")) })
if ((Test-Path $modelPath) -and $faltantes.Count -eq 0) {
    Write-Host "ya esta: Kokoro en $destDir"
    exit 0
}

function Get-File($relative, $destination) {
    $url = "$base/$relative"
    $part = "$destination.part"
    if (Test-Path $part) { Remove-Item -Force -ErrorAction SilentlyContinue $part }
    Write-Host "Descargando $relative ..."
    Invoke-WebRequest -Uri $url -OutFile $part -UseBasicParsing
    Move-Item -Force $part $destination
}

try {
    if (-not (Test-Path $modelPath)) {
        Get-File 'onnx/model.onnx' $modelPath
        Get-File 'tokenizer.json' (Join-Path $destDir 'tokenizer.json')
    }
    foreach ($voice in $faltantes) {
        Get-File "voices/$voice.bin" (Join-Path $destDir "voices\$voice.bin")
    }

    if (-not (Test-Path $modelPath)) {
        throw "La descarga termino pero falta model.onnx en $destDir."
    }
    $total = (Get-ChildItem -Path $destDir -File -Recurse | Measure-Object -Property Length -Sum).Sum
    Write-Host "Kokoro listo en: $destDir"
    Write-Host ("Total: {0:N1} MB" -f ($total / 1MB))
    Write-Host 'Kokoro es Apache 2.0. El fonemizador (espeak-ng) viene con las dependencias de Python.'
} catch {
    Write-Error "No se pudo bajar Kokoro: $_"
    exit 1
}
