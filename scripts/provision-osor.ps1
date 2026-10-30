param(
    # Carpeta completa del paquete, la que lee Settings.osor_model_path. La app la
    # pasa siempre; sin ella, la corrida manual usa runtime\models\eraser-osor.
    [string]$PackDir = ''
)

$ErrorActionPreference = 'Stop'

# Pack OSOR (borrado con IA). Baja la base SDXL-inpainting y los pesos OSOR en
# revisiones fijas, verifica tamano y sha256, y arma el paquete con
# build_osor_pack.py, que calcula los embeddings del prompt fijo solo.
# Licencia CreativeML Open RAIL++-M: la app exige aceptarla antes de llegar aca.

$root = Split-Path -Parent $PSScriptRoot
if (-not $PackDir) { $PackDir = Join-Path $root 'runtime\models\eraser-osor' }
$venv = Join-Path $root '.venv\Scripts\python.exe'
if (-not (Test-Path $venv)) { $venv = 'python' }

$destino = $PackDir
$parcial = "$destino.partial"
$trabajo = "$destino.work"
$base = Join-Path $trabajo 'base'
$pesos = Join-Path $trabajo 'weights\osor-sdxlinpainting\weights\sdxlinpainting_phase2.pth'

# Los mismos archivos que Settings.osor_available() revisa en app/config.py.
$requeridos = @(
    'unet\config.json',
    'unet\diffusion_pytorch_model.fp16.safetensors',
    'vae\config.json',
    'vae\diffusion_pytorch_model.fp16.safetensors',
    'scheduler\scheduler_config.json',
    'prompt_embeds.pt'
)

function Test-OsorInstalado {
    $faltantes = @($requeridos | Where-Object { -not (Test-Path -LiteralPath (Join-Path $destino $_) -PathType Leaf) })
    return $faltantes.Count -eq 0
}

function Test-Peft {
    # El exit code basta: un import que falla escribe en stderr y con Stop eso cortaria el script.
    $previo = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        & $venv -c 'import peft' 2>$null
        return ($LASTEXITCODE -eq 0)
    } finally {
        $ErrorActionPreference = $previo
    }
}

if (Test-OsorInstalado) {
    Write-Host "ya esta: OSOR en $destino"
    exit 0
}

if (-not (Test-Peft)) {
    throw ("Falta la dependencia 'peft' (>=0.14) que necesita el paquete OSOR. " +
        "Instalala en el entorno de Upflow con: '" + $venv + "' -m pip install 'peft>=0.14'")
}

# build_osor_pack.py se niega a arrancar si existe el .partial de una corrida que murio a mitad.
if (Test-Path -LiteralPath $parcial) { Remove-Item -Recurse -Force -LiteralPath $parcial }
New-Item -ItemType Directory -Force -Path $trabajo | Out-Null

Write-Host 'Bajando la base SDXL-inpainting y los pesos OSOR (revisiones fijas, unos 9 GB)...'
& $venv (Join-Path $root 'scripts\fetch_osor_sources.py') '--dest' $trabajo
if ($LASTEXITCODE -ne 0) {
    throw "La descarga de OSOR fallo (codigo $LASTEXITCODE). Vuelve a correrla: retoma lo que quedo en $trabajo."
}

Write-Host 'Armando el paquete OSOR (embeddings del prompt, LoRA fusionada y verificacion de paridad)...'
& $venv (Join-Path $root 'scripts\build_osor_pack.py') '--base' $base '--weights' $pesos '--out' $destino
if ($LASTEXITCODE -ne 0) {
    throw "El armado del paquete OSOR fallo (codigo $LASTEXITCODE). La carpeta de trabajo queda en $trabajo para revisar."
}

if (-not (Test-OsorInstalado)) {
    throw "El paquete OSOR quedo incompleto en $destino."
}

Remove-Item -Recurse -Force -LiteralPath $trabajo
Write-Host "OSOR listo en: $destino"
