param(
    # El idioma del banco. La alineación no es igual de fácil en todos, así que
    # se mide por idioma: de los que no se probaron no se sabe nada.
    [ValidateSet('en', 'es')]
    [string]$Language = 'en'
)

$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Speech

# Banco con VERDAD DE REFERENCIA para medir la alineación palabra por palabra.
#
# Cada palabra se sintetiza SOLA, así se conoce su duración exacta; después se
# pegan con silencios fijos (eso lo hace el script de Python). Los límites de
# cada palabra terminan siendo un cálculo y no una estimación, que es lo que
# permite medir el error en milisegundos en vez de mirar si "se ve razonable".
#
# Se usa la voz que ya trae Windows: no hace falta descargar ni instalar nada.

$config = @{
    en = @{
        voz      = 'Microsoft Zira Desktop'
        palabras = @('hello', 'world', 'this', 'is', 'a', 'test', 'of', 'word', 'level', 'timing')
    }
    es = @{
        voz      = 'Microsoft Sabina Desktop'
        palabras = @('hola', 'mundo', 'esto', 'es', 'una', 'prueba', 'de', 'tiempos', 'por', 'palabra')
    }
}

$voz = $config[$Language].voz
$palabras = $config[$Language].palabras

$instaladas = (New-Object System.Speech.Synthesis.SpeechSynthesizer).GetInstalledVoices() |
    ForEach-Object { $_.VoiceInfo.Name }
if ($instaladas -notcontains $voz) {
    throw ("No esta instalada la voz '$voz', que es la que este banco usa para " +
           "'$Language'. Instaladas: " + ($instaladas -join ', '))
}

$raiz = Split-Path -Parent $PSScriptRoot
$carpeta = Join-Path $raiz "runtime\bench\bench_words_$Language"
New-Item -ItemType Directory -Force $carpeta | Out-Null

# 16 kHz mono: es exactamente lo que come Whisper, así que no hay remuestreo en
# el medio que corra los límites que se estan tratando de medir.
$formato = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(16000, [System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen, [System.Speech.AudioFormat.AudioChannel]::Mono)

foreach ($p in $palabras) {
    $s = New-Object System.Speech.Synthesis.SpeechSynthesizer
    $s.SelectVoice($voz)
    $s.Rate = 0
    $destino = Join-Path $carpeta "$p.wav"
    $s.SetOutputToWaveFile($destino, $formato)
    $s.Speak($p)
    $s.Dispose()
}

Write-Host "banco '$Language' listo en $carpeta ($($palabras.Count) palabras, voz $voz)"
