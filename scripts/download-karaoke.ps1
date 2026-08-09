param(
    # Cual modelo del catalogo bajar. Tiene que coincidir con los ids de
    # app/services/engines/mdx_models.py.
    [ValidateSet('inst_hq_3', 'voc_ft')]
    [string]$Model = 'inst_hq_3'
)

$ErrorActionPreference = 'Stop'

# Older Windows PowerShell 5.1 defaults to TLS 1.0, which GitHub rejects.
[Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12

$root = Split-Path -Parent $PSScriptRoot
$vendorDir = Join-Path $root 'vendor\karaoke'

# Modelos MDX-Net del propio equipo de Ultimate Vocal Remover (Anjok07 & aufr33,
# MIT). El hash es el "hash UVR": MD5 de los ultimos 10000 KiB del archivo, que
# es como UVR identifica sus modelos en model_data_new.json. El hash UVR solo
# cubre la cola (~15% del archivo), asi que ADEMAS se pinea el SHA-256 del
# archivo completo: el mirror TRvlvr/model_repo es de terceros, no del equipo
# UVR. Verificados contra los archivos reales el 2026-08-09.
$modelos = @{
    'inst_hq_3' = @{
        File   = 'UVR-MDX-NET-Inst_HQ_3.onnx'
        Url    = 'https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/UVR-MDX-NET-Inst_HQ_3.onnx'
        Hash   = '55657dd70583b0fedfba5f67df11d711'
        Sha256 = '317554b07fe1ea5279a77f2b1520a41ea4b93432560c4ffd08792c30fddf9adc'
        Label  = 'MDX-Net Inst HQ 3 (saca la instrumental)'
    }
    'voc_ft' = @{
        File   = 'UVR-MDX-NET-Voc_FT.onnx'
        Url    = 'https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/UVR-MDX-NET-Voc_FT.onnx'
        Hash   = '77d07b2667ddf05b9e3175941b4454a0'
        Sha256 = '534b2070fcc7df514b13ef660dc8cbb328679c2374d04354a5c42bb14ecce111'
        Label  = 'MDX-Net Voc FT (saca la voz)'
    }
}

function Get-UvrHash([string]$path) {
    # MD5 de los ultimos 10000 KiB (o del archivo entero si es mas chico).
    $tailBytes = 10000 * 1024
    $stream = [System.IO.File]::OpenRead($path)
    try {
        if ($stream.Length -gt $tailBytes) {
            $null = $stream.Seek(-$tailBytes, [System.IO.SeekOrigin]::End)
        }
        $md5 = [System.Security.Cryptography.MD5]::Create()
        try {
            $hash = $md5.ComputeHash($stream)
        } finally {
            $md5.Dispose()
        }
    } finally {
        $stream.Dispose()
    }
    return ([System.BitConverter]::ToString($hash) -replace '-', '').ToLowerInvariant()
}

function Get-Sha256([string]$path) {
    # .NET directo y no Get-FileHash: ese cmdlet vive en un modulo, y un server
    # lanzado con PSModulePath contaminado (pasado real 2026-08-09: relanzado
    # desde pwsh 7) deja al powershell 5.1 hijo sin poder resolverlo. .NET
    # siempre esta.
    $sha256 = [System.Security.Cryptography.SHA256]::Create()
    try {
        $stream = [System.IO.File]::OpenRead($path)
        try {
            $hash = $sha256.ComputeHash($stream)
        } finally {
            $stream.Dispose()
        }
    } finally {
        $sha256.Dispose()
    }
    return ([System.BitConverter]::ToString($hash) -replace '-', '').ToLowerInvariant()
}

function Get-ModelIntegrityError([string]$path, $info) {
    # $null si pasa ambas verificaciones; si no, un texto que dice CUAL fallo.
    $uvr = Get-UvrHash $path
    if ($uvr -ne $info.Hash) {
        return "hash UVR $uvr, se esperaba $($info.Hash)"
    }
    $sha = Get-Sha256 $path
    if ($sha -ne $info.Sha256) {
        return "SHA-256 $sha, se esperaba $($info.Sha256)"
    }
    return $null
}

function Write-KaraokeCredits([string]$directory) {
    $credits = @(
        'Models: UVR-MDX-NET Inst HQ 3 / UVR-MDX-NET Voc FT'
        'Ultimate Vocal Remover (Anjok07 & aufr33), MIT'
        'github.com/Anjok07/ultimatevocalremovergui'
    ) -join [Environment]::NewLine
    Set-Content -Path (Join-Path $directory 'CREDITS.txt') -Value $credits -Encoding UTF8
}

$info = $modelos[$Model]
$destino = Join-Path $vendorDir $info.File

if (Test-Path $destino) {
    $fallo = Get-ModelIntegrityError $destino $info
    if ($null -eq $fallo) {
        Write-KaraokeCredits $vendorDir
        Write-Host "Ya esta: $($info.Label) en $destino"
        Write-Host "Borra el archivo para forzar una re-descarga."
        return
    }
    Write-Host "El archivo existente no pasa la verificacion ($fallo); se vuelve a bajar."
    Remove-Item -Force $destino
}

New-Item -ItemType Directory -Force -Path $vendorDir | Out-Null

$temporal = "$destino.download"
Write-Host "Descargando $($info.Label) (~67 MB)..."
Invoke-WebRequest -Uri $info.Url -OutFile $temporal -UseBasicParsing

$fallo = Get-ModelIntegrityError $temporal $info
if ($null -ne $fallo) {
    Remove-Item -Force $temporal -ErrorAction SilentlyContinue
    throw "La descarga no paso la verificacion: $fallo."
}
Move-Item -Force $temporal $destino

Write-KaraokeCredits $vendorDir

Write-Host "Modelo de separacion listo en: $destino"
Write-Host 'Licencia MIT (Ultimate Vocal Remover, Anjok07 & aufr33).'
Write-Host 'El modo Karaoke del apartado Audio queda habilitado con este modelo.'
