$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Speech

# Banco con VERDAD DE REFERENCIA: cada palabra se sintetiza sola, así se conoce
# su duración exacta, y después se pegan con silencios fijos. Los límites de cada
# palabra no se estiman: se calculan.
$palabras = @('hello','world','this','is','a','test','of','word','level','timing')
$carpeta = Join-Path $PSScriptRoot 'bench_words'
New-Item -ItemType Directory -Force $carpeta | Out-Null

$formato = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(16000, [System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen, [System.Speech.AudioFormat.AudioChannel]::Mono)

foreach ($p in $palabras) {
    $s = New-Object System.Speech.Synthesis.SpeechSynthesizer
    $s.SelectVoice('Microsoft Zira Desktop')
    $s.Rate = 0
    $destino = Join-Path $carpeta "$p.wav"
    $s.SetOutputToWaveFile($destino, $formato)
    $s.Speak($p)
    $s.Dispose()
}
Write-Host ("generadas: " + $palabras.Count)
$palabras -join ','
