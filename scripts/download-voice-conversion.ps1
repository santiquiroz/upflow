$ErrorActionPreference = 'Stop'

# Conversion de voz: decir lo mismo con la voz de otra grabacion.
#
# Baja SpeechT5 VC y su vocoder (los dos MIT) desde Hugging Face.
#
# ADEMAS hace falta el encoder de x-vector, que NO se baja: se genera una sola
# vez con scripts/export_xvector_onnx.py, en un venv aparte. Los dos exports de
# terceros que hay publicados se probaron el 2026-08-05 y ninguno sirve: uno no
# inicializa la sesion y el otro devuelve embeddings fuera de espacio, con lo
# que la conversion sale en NaN.

$root = Split-Path -Parent $PSScriptRoot
$venv = Join-Path $root '.venv\Scripts\python.exe'
if (-not (Test-Path $venv)) { $venv = 'python' }

$destino = Join-Path $root 'vendor'
$xvector = Join-Path $destino 'xvector\tdnn.onnx'

Write-Host 'Descargando microsoft/speecht5_vc y su vocoder (MIT)...'
& $venv -c @"
from huggingface_hub import snapshot_download
import sys
raiz = sys.argv[1]
for repo, carpeta in (('microsoft/speecht5_vc','speecht5-vc'), ('microsoft/speecht5_hifigan','speecht5-hifigan')):
    destino = f'{raiz}/{carpeta}'
    snapshot_download(repo, local_dir=destino)
    print(f'  {repo} -> {destino}')
"@ $destino

if (-not (Test-Path $xvector)) {
    Write-Warning "Falta el encoder de x-vector en $xvector."
    Write-Warning 'Sin el no se puede sacar la voz de una muestra. Generalo con:'
    Write-Warning '  python -m venv .venv-xvect'
    Write-Warning '  .venv-xvect\Scripts\pip install torch==2.5.1 torchaudio==2.5.1 speechbrain==1.0.2 onnx onnxscript onnxruntime requests soundfile numpy huggingface_hub==0.25.2'
    Write-Warning '  .venv-xvect\Scripts\python scripts\export_xvector_onnx.py'
} else {
    Write-Host "Encoder de x-vector ya presente: $xvector"
}

Write-Host 'Listo. SpeechT5 VC y su vocoder son MIT.'
